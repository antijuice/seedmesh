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

    # Split out so `describe_plan` can show both. A volunteer who sees only a total cannot
    # tell that quantizing harder will not buy them what they expect -- nf4 shrinks the
    # weights and leaves the cache untouched.
    weight_bytes_per_block: int = 0
    cache_bytes_per_block: int = 0

    @property
    def covers_whole_model(self) -> bool:
        return self.recommended_blocks >= self.n_layers

    @property
    def fraction_of_model(self) -> float:
        return min(1.0, self.recommended_blocks / self.n_layers) if self.n_layers else 0.0


# The exact wheel that matches the CPU build most people end up with. Verified available
# on 2026-08-02: the cu126 index carries torch 2.13.0, the same version as the default CPU
# wheel, so this is a swap rather than an upgrade that drags dependencies with it.
CUDA_TORCH_FIX = (
    "pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 "
    "torch==2.13.0+cu126"
)


def torch_cuda_ready() -> tuple[bool, str]:
    """Whether *torch* can use a GPU -- which is not the same question as whether one exists.

    `detect_gpus` asks nvidia-smi; Petals asks `torch.cuda`. Those disagree in a way that
    fails silently and expensively: `pip install petals` pulls the CPU-only torch wheel by
    default, so probe reports "27 blocks on your RTX 3050 at nf4", serve accepts it, and the
    whole model runs on the CPU with nobody told. The server works, slowly, and looks
    correct. Found on this machine, which had been serving on CPU while reporting a GPU.
    """
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"
    if torch.cuda.is_available():
        return True, f"torch {torch.__version__}"
    if "+cpu" in torch.__version__:
        return False, (
            f"torch {torch.__version__} is the CPU-only build -- it has no CUDA compiled in, "
            "whatever nvidia-smi reports"
        )
    return False, f"torch {torch.__version__} has CUDA support but cannot see a device"


def warn_if_gpu_unusable(gpus: list) -> list[str]:
    """Lines to print when a GPU exists but torch cannot reach it. Empty when all is well."""
    if not gpus:
        return []  # no GPU at all is a different, already-handled situation
    ready, detail = torch_cuda_ready()
    if ready:
        return []
    return [
        "",
        f"  WARNING: nvidia-smi sees {gpus[0].name}, but torch cannot use it.",
        f"    {detail}",
        "    Serving will fall back to the CPU. It will work and it will be slow, and",
        "    nothing else would have told you. To fix:",
        f"      {CUDA_TORCH_FIX}",
        "",
    ]


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

    Mixture-of-experts models are handled explicitly, because ignoring the experts is not a
    small error. A Mixtral block holds ``num_local_experts`` copies of the MLP -- 8x -- and
    a DeepSeek-V3-class block holds 256. Sizing one as if it were dense understates it by
    two orders of magnitude, and ``probe`` would confidently tell a volunteer to host blocks
    that cannot fit.
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
    norms = 2 * hidden

    # Experts, if any. `num_local_experts` is Mixtral's spelling, `n_routed_experts`
    # DeepSeek-V3's. Each routed expert is a full gate/up/down MLP; DeepSeek additionally
    # keeps `n_shared_experts` always-on ones at `moe_intermediate_size`.
    n_routed = int(config.get("num_local_experts") or config.get("n_routed_experts") or 0)
    if n_routed:
        moe_intermediate = int(config.get("moe_intermediate_size") or intermediate)
        experts = n_routed * 3 * hidden * moe_intermediate
        shared = int(config.get("n_shared_experts") or 0) * 3 * hidden * moe_intermediate
        router = hidden * n_routed
        mlp = experts + shared + router
    else:
        mlp = 3 * hidden * intermediate

    return q + k + v + o + mlp + norms


def is_moe_layer(config: dict, layer_index: int) -> bool:
    """Is this specific layer a mixture-of-experts layer?

    Blocks are not uniform in this family. DeepSeek-V3 sets ``first_k_dense_replace = 3``:
    the first three layers are ordinary dense MLPs and the remaining 58 are MoE, so a block
    plan that assumes one size is wrong for every layer it is applied to.
    """
    if not (config.get("num_local_experts") or config.get("n_routed_experts")):
        return False
    first_dense = int(config.get("first_k_dense_replace") or 0)
    if layer_index < first_dense:
        return False
    # `moe_layer_freq` of N means every Nth layer past the dense prefix is MoE (1 = all).
    freq = int(config.get("moe_layer_freq") or 1)
    return freq <= 1 or (layer_index - first_dense) % freq == 0


def params_per_layer(config: dict, layer_index: int) -> int:
    """Parameter count of one *specific* layer, respecting dense/MoE mixing."""
    if is_moe_layer(config, layer_index):
        return params_per_block(config)
    dense = dict(config)
    for key in ("num_local_experts", "n_routed_experts", "n_shared_experts"):
        dense.pop(key, None)
    return params_per_block(dense)


def attn_cache_bytes_per_block(
    config: dict, *, dtype_bytes: int = 2, tokens: Optional[int] = None
) -> int:
    """VRAM Petals reserves for one block's attention cache.

    This is charged PER BLOCK, which is why leaving it out of the plan gets worse the more
    blocks you take -- exactly backwards from a safety margin. Measured on Llama-3.1-8B:
    64 MiB per block, so the recommendation of 25 blocks came to 2.61 GiB of weights plus
    1.56 GiB of cache = 4.17 GiB on a 4.0 GiB card. It would have OOM'd shortly after
    starting, and the volunteer would have read it as their hardware being inadequate.

    Mirrors `Server.__init__` (petals/server/server.py): the token budget quadruples for
    grouped/multi-query attention, and MLA needs the width attention actually stores rather
    than `2 * hidden`. Kept in step with `tools/port_petals.py` patch 11.
    """
    hidden = int(config.get("hidden_size", 0))
    heads = int(config.get("num_attention_heads", 0) or 0)
    kv_heads = int(config.get("num_key_value_heads", heads) or heads or 1)
    groups = max(1, heads // kv_heads) if heads else 1

    # Petals: 16384 tokens when grouped/multi-query, else 4096. This is the single biggest
    # lever a volunteer has. It bounds how much concurrent session length a server can hold,
    # NOT the model's quality or context window -- so on a small card, lowering it is how you
    # trade concurrency for blocks, and on an 8B model it is the difference between two 4 GiB
    # laptops covering the model and falling two blocks short.
    if tokens is None:
        tokens = 16384 if groups > 1 else 4096

    qk_nope = config.get("qk_nope_head_dim")
    qk_rope = config.get("qk_rope_head_dim")
    if qk_nope is not None and qk_rope is not None:
        v_dim = config.get("v_head_dim", qk_nope)
        width = heads * (int(qk_nope) + int(qk_rope) + int(v_dim))
    else:
        width = (2 * hidden) // groups

    return width * tokens * dtype_bytes


def plan_blocks(
    config: dict,
    gpu: GpuInfo,
    *,
    quant: str = "nf4",
    model_name: str = "",
    reserve_bytes: Optional[int] = None,
    attn_cache_tokens: Optional[int] = None,
) -> BlockPlan:
    """How many blocks of this model fit on this GPU."""
    if quant not in BYTES_PER_PARAM:
        raise ValueError(f"unknown quantization {quant!r}; expected one of {sorted(BYTES_PER_PARAM)}")

    # Size against the LARGEST layer, not an average. Blocks are not uniform in MoE models
    # (DeepSeek-V3: 3 dense then 58 MoE, a 19x difference), and a volunteer does not choose
    # which block range they are assigned -- so a plan that fits only the small ones is a
    # plan that OOMs. For dense models every layer is identical and this is a no-op.
    n_layers = int(config.get("num_hidden_layers", 0))
    if n_layers:
        per_block = max(params_per_layer(config, i) for i in range(n_layers))
    else:
        per_block = params_per_block(config)
    weight_bytes = int(per_block * BYTES_PER_PARAM[quant])

    # Weights are not the whole per-block cost. Quantization shrinks the weights and does
    # nothing to the attention cache, so the cache's share GROWS as quantization improves:
    # on Llama-3.1-8B at nf4 it is 64 MiB against 107 MiB of weights, well over a third of
    # the real cost of a block.
    cache_bytes = attn_cache_bytes_per_block(config, tokens=attn_cache_tokens)
    bytes_per_block = weight_bytes + cache_bytes

    if reserve_bytes is None:
        reserve_bytes = max(MIN_RESERVE_BYTES, int(gpu.total_bytes * RESERVE_FRACTION))
    usable = max(0, gpu.free_bytes - reserve_bytes)

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
        weight_bytes_per_block=weight_bytes,
        cache_bytes_per_block=cache_bytes,
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
        f"{plan.bytes_per_block / 2**20:.0f} MiB "
        f"({plan.weight_bytes_per_block / 2**20:.0f} weights + "
        f"{plan.cache_bytes_per_block / 2**20:.0f} attention cache)",
        f"  usable VRAM       {plan.usable_bytes / 2**30:.1f} GiB (after reserve)",
    ]
    if plan.cache_bytes_per_block > plan.weight_bytes_per_block // 2 and plan.n_layers:
        with_smaller = plan.usable_bytes // max(
            1, plan.weight_bytes_per_block + plan.cache_bytes_per_block // 4
        )
        if with_smaller > plan.recommended_blocks:
            lines.append(
                f"  cache is large    --attn-cache-tokens 4096 would fit "
                f"{min(int(with_smaller), plan.n_layers)} blocks instead of "
                f"{plan.recommended_blocks}"
            )
            lines.append(
                "                    (bounds concurrent sessions, not context length)"
            )
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
