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


class ConfigFetchError(Exception):
    """Raised with an explanation a volunteer can act on."""


def fetch_config(model_name: str) -> dict:
    """Fetch a model's config.json without downloading weights.

    Failure modes are translated rather than surfaced raw: a gated repository closes the
    connection or 401s, which as a bare URLError reads like a broken network and sends
    someone debugging the wrong thing.
    """
    import urllib.error
    import urllib.request

    url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
    request = urllib.request.Request(url, headers={"User-Agent": "seedmesh-probe"})
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
        raise ConfigFetchError(
            f"could not reach huggingface.co ({exc.reason}).\n"
            f"  If {model_name} is a gated repo, that is the likely cause rather than your "
            f"network -- gated repos often drop the connection for anonymous requests."
        ) from exc


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
