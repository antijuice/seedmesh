"""How far is DeepSeek-V3 / Kimi-K2 from something Petals could host?

Petals hosts *blocks* by importing a decoder-layer class directly and wrapping it. That only
works if `transformers` implements the architecture natively -- a model that needs
`trust_remote_code` ships its layer class in the repo, which Petals has no path to load and
which would also mean every volunteer executes code from a model repo.

So the first question is not "how much work is the wrapper" but "does transformers know this
architecture at all". Checks, in order:

  1. is deepseek_v3 in the installed transformers' model registry?
  2. can the decoder layer class be imported?
  3. what is structurally different from Llama (which the port already handles)?
"""

import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

print(f"transformers {transformers.__version__}\n")

print("=== 1. architecture registry ===")
for name in ("llama", "mixtral", "deepseek_v3", "kimi_k2", "qwen3_moe"):
    known = name in CONFIG_MAPPING_NAMES
    print(f"  {name:<14} {'registered' if known else 'NOT registered'}")

print("\n=== 2. can the decoder layer be imported? ===")
targets = [
    ("llama", "transformers.models.llama.modeling_llama", "LlamaDecoderLayer"),
    ("mixtral", "transformers.models.mixtral.modeling_mixtral", "MixtralDecoderLayer"),
    ("deepseek_v3", "transformers.models.deepseek_v3.modeling_deepseek_v3",
     "DeepseekV3DecoderLayer"),
]
for label, module, cls in targets:
    try:
        mod = __import__(module, fromlist=[cls])
        getattr(mod, cls)
        print(f"  {label:<14} OK  ({cls})")
    except Exception as exc:
        print(f"  {label:<14} {type(exc).__name__}: {str(exc)[:80]}")

print("\n=== 3. structural differences from Llama ===")
try:
    from transformers import AutoConfig

    cfg = AutoConfig.for_model("deepseek_v3")
    interesting = [
        "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
        "n_routed_experts", "n_shared_experts", "first_k_dense_replace", "moe_layer_freq",
        "topk_method", "scoring_func",
    ]
    for key in interesting:
        if hasattr(cfg, key):
            print(f"  {key:<22} {getattr(cfg, key)}")
except Exception as exc:
    print(f"  could not introspect: {type(exc).__name__}: {exc}")

print("""
=== what each finding implies ===
  registered + importable -> a petals/models/deepseek_v3/ subpackage, following the llama
                             template that already works. Bounded work.
  not registered          -> needs trust_remote_code; Petals cannot host it without a much
                             deeper change, and volunteers would be running model-repo code.
  MLA fields present      -> the KV cache is a compressed latent, not full K/V. Petals sizes
                             its attention cache assuming standard KV, so `attn_cache_tokens`
                             accounting is wrong for this family until adjusted.
  first_k_dense_replace   -> the first N layers are dense, the rest MoE. Blocks are NOT
                             uniform in size, so per-block VRAM planning must be per-layer.
""")
