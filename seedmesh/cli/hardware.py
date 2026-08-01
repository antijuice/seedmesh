"""Turn "what hardware do I have?" into "host N blocks of this model".

A volunteer should not have to compute a block count. Petals' server takes `--num_blocks`
and leaves the arithmetic to the operator; getting it wrong means either wasting donated
capacity or being killed by the OOM reaper mid-request, which reads to the swarm as churn.

Everything here is deliberately torch-free arithmetic over a model config, so `seedmesh
probe` works before any of the heavy backend is installed -- which is exactly when someone
is deciding whether to bother.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

# Bytes per parameter, measured rather than assumed -- see spike/quantization/README.md.
# int8 matched Petals' own constant exactly; NF4 came out slightly under its 0.531 estimate.
BYTES_PER_PARAM = {
    "none": 2.000,   # fp16/bf16
    "int8": 1.000,
    "nf4": 0.516,
}

# Reserved for activations, the attention cache and allocator fragmentation. A fraction
# alone under-reserves small cards and over-reserves large ones, so take the larger of the
# two. Heuristic, not measured -- the honest way to tune it is watching a real server.
MIN_RESERVE_BYTES = 1 * 2**30
RESERVE_FRACTION = 0.15


@dataclass(frozen=True, slots=True)
class GpuInfo:
    name: str
    total_bytes: int
    free_bytes: int
    compute_capability: Optional[str] = None

    @property
    def total_gib(self) -> float:
        return self.total_bytes / 2**30


@dataclass(frozen=True, slots=True)
class BlockPlan:
    model: str
    quant: str
    n_layers: int
    params_per_block: int
    bytes_per_block: int
    usable_bytes: int
    recommended_blocks: int

    @property
    def covers_whole_model(self) -> bool:
        return self.recommended_blocks >= self.n_layers

    @property
    def fraction_of_model(self) -> float:
        return min(1.0, self.recommended_blocks / self.n_layers) if self.n_layers else 0.0


def detect_gpus() -> list[GpuInfo]:
    """Query nvidia-smi. Returns an empty list when there is no NVIDIA GPU."""
    binary = shutil.which("nvidia-smi")
    if not binary:
        return []
    try:
        result = subprocess.run(
            [
                binary,
                "--query-gpu=name,memory.total,memory.free,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            # nvidia-smi reports MiB with nounits.
            total = int(float(parts[1])) * 2**20
            free = int(float(parts[2])) * 2**20
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                name=parts[0],
                total_bytes=total,
                free_bytes=free,
                compute_capability=parts[3] if len(parts) > 3 else None,
            )
        )
    return gpus


def params_per_block(config: dict) -> int:
    """Parameter count of one transformer block, from a HuggingFace config dict.

    Computed analytically rather than by instantiating the model, so this needs neither
    torch nor a weight download -- a volunteer can size a 70B model before committing to
    tens of gigabytes of traffic.

    Exact for Llama-family architectures (checked against a real block: a 128-hidden /
    256-intermediate / 8-head / 2-kv-head config gives 139,520 both ways).
    """
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", heads))
    head_dim = int(config.get("head_dim") or hidden // heads)

    q = hidden * heads * head_dim
    k = hidden * kv_heads * head_dim
    v = k
    o = heads * head_dim * hidden
    mlp = 3 * hidden * intermediate
    norms = 2 * hidden
    return q + k + v + o + mlp + norms


def plan_blocks(
    config: dict,
    gpu: GpuInfo,
    *,
    quant: str = "nf4",
    model_name: str = "",
    reserve_bytes: Optional[int] = None,
) -> BlockPlan:
    """How many blocks of this model fit on this GPU."""
    if quant not in BYTES_PER_PARAM:
        raise ValueError(f"unknown quantization {quant!r}; expected one of {sorted(BYTES_PER_PARAM)}")

    per_block = params_per_block(config)
    bytes_per_block = int(per_block * BYTES_PER_PARAM[quant])

    if reserve_bytes is None:
        reserve_bytes = max(MIN_RESERVE_BYTES, int(gpu.total_bytes * RESERVE_FRACTION))
    usable = max(0, gpu.free_bytes - reserve_bytes)

    n_layers = int(config.get("num_hidden_layers", 0))
    fits = usable // bytes_per_block if bytes_per_block else 0
    recommended = int(min(fits, n_layers)) if n_layers else int(fits)

    return BlockPlan(
        model=model_name or config.get("_name_or_path", "?"),
        quant=quant,
        n_layers=n_layers,
        params_per_block=per_block,
        bytes_per_block=bytes_per_block,
        usable_bytes=usable,
        recommended_blocks=recommended,
    )


# Petals ships block wrappers for exactly these architectures -- petals/models/ has one
# subpackage each, and AutoDistributedConfig raises "Petals does not support model type X"
# for anything else. Verified against the registry the ported checkout actually populates,
# not against upstream docs.
#
# This list is here rather than imported from petals on purpose: `probe` must run on a
# machine with no backend installed at all, which is most of its value.
SUPPORTED_MODEL_TYPES = ("bloom", "falcon", "llama", "mixtral")


class UnsupportedModelError(Exception):
    """The model's architecture has no Petals block implementation."""


def check_model_supported(config: dict, model_name: str = "") -> str:
    """Return the model_type, or explain why this model can never be served.

    Worth checking early and separately from sizing: `probe` will happily compute a block
    plan for any config.json with the right numeric fields, and reporting "you can host 30
    of 36 blocks" for an architecture Petals cannot load is worse than saying nothing.
    """
    model_type = str(config.get("model_type", "")).lower()
    if model_type in SUPPORTED_MODEL_TYPES:
        return model_type

    label = model_name or config.get("_name_or_path", "this model")
    described = f"model type {model_type!r}" if model_type else "an unrecognised model type"
    raise UnsupportedModelError(
        f"{label} is {described}, which Petals has no block implementation for.\n"
        f"  Supported: {', '.join(SUPPORTED_MODEL_TYPES)}.\n"
        f"  Note this is an *architecture* limit, not a licence one -- Qwen, Gemma, Phi and\n"
        f"  Mistral-dense all fail here regardless of how open their weights are.\n"
        f"  Llama-architecture models are the safe default; see docs/QUICKSTART.md."
    )


class ConfigFetchError(Exception):
    """Raised with an explanation a volunteer can act on."""


CONFIG_FETCH_ATTEMPTS = 3


def fetch_config(model_name: str, *, attempts: int = CONFIG_FETCH_ATTEMPTS) -> dict:
    """Fetch a model's config.json without downloading weights.

    Retries on connection-level failures. Measured: fetching ten configs in a row, three
    ungated repos reset the connection on the first try and succeeded on the second. A
    one-shot fetch reports those as unreachable, which a volunteer reads as "this model is
    unavailable" when it is simply rate limiting.

    HTTP failures are translated rather than surfaced raw -- a gated repo 401s, which as a
    bare HTTPError sends someone debugging the wrong thing -- but are *not* retried, since
    401/404 will not change on a second attempt.
    """
    import time
    import urllib.error
    import urllib.request

    url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
    request = urllib.request.Request(url, headers={"User-Agent": "seedmesh-probe"})

    last_reason = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise ConfigFetchError(
                    f"{model_name} is gated -- it needs a Hugging Face account and accepted "
                    f"licence terms.\n  Seedmesh prefers unambiguously permissive models; see "
                    f"seedmesh/models/registry.yaml."
                ) from exc
            if exc.code == 404:
                raise ConfigFetchError(f"no model named {model_name} on Hugging Face") from exc
            raise ConfigFetchError(f"HTTP {exc.code} fetching {model_name}") from exc
        except urllib.error.URLError as exc:
            last_reason = exc.reason
            if attempt < attempts:
                time.sleep(attempt)  # 1s, then 2s

    raise ConfigFetchError(
        f"could not reach huggingface.co after {attempts} attempts ({last_reason}).\n"
        f"  Hugging Face resets anonymous connections under load, so this is often "
        f"transient -- try again before assuming {model_name} is unavailable."
    )


def describe_plan(plan: BlockPlan, gpu: GpuInfo) -> list[str]:
    """Human-readable sizing report."""
    lines = [
        f"  GPU               {gpu.name} ({gpu.total_gib:.1f} GiB total, "
        f"{gpu.free_bytes / 2**30:.1f} GiB free)",
        f"  model             {plan.model} ({plan.n_layers} blocks)",
        f"  quantization      {plan.quant} ({BYTES_PER_PARAM[plan.quant]} bytes/param)",
        f"  per block         {plan.params_per_block / 1e6:.1f}M params, "
        f"{plan.bytes_per_block / 2**20:.0f} MiB",
        f"  usable VRAM       {plan.usable_bytes / 2**30:.1f} GiB (after reserve)",
    ]
    if plan.recommended_blocks <= 0:
        lines.append("  recommendation    cannot host even one block -- try --quant nf4, "
                     "a smaller model, or free VRAM")
    elif plan.covers_whole_model:
        lines.append(f"  recommendation    all {plan.n_layers} blocks (the whole model fits)")
    else:
        lines.append(
            f"  recommendation    {plan.recommended_blocks} blocks "
            f"({plan.fraction_of_model:.0%} of the model)"
        )
    return lines
