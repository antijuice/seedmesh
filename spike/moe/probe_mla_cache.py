"""Does DeepSeek-V3's multi-head latent attention use the same cache shape as Llama?

This decides how much of the Llama port transfers. The Llama and Mixtral blocks translate
between Petals' flattened bloom-style cache and `(batch, heads, seq, head_dim)`, which is
what standard attention stores. MLA does not store K and V -- it stores a *compressed
latent* (`kv_lora_rank=512`) plus a small RoPE part (`qk_rope_head_dim=64`), which is the
entire point of the design and why its KV cache is so much smaller.

If MLA still calls `Cache.update(key_states, value_states, ...)` with standard-shaped
tensors, the existing translation works untouched. If it stores something else, the
translation is wrong and Petals' `attn_cache_tokens` accounting is wrong with it.

Answering this from the source rather than by guessing, because a wrong cache translation
would not crash -- it would silently corrupt attention, which in a swarm that judges servers
by output agreement is the worst failure mode available.
"""

import inspect
import re

from transformers.models.deepseek_v3 import modeling_deepseek_v3 as ds
from transformers.models.llama import modeling_llama as llama

print("=== what each attention passes to Cache.update ===")
for label, module, cls in (
    ("Llama", llama, "LlamaAttention"),
    ("DeepSeek-V3", ds, "DeepseekV3Attention"),
):
    src = inspect.getsource(getattr(module, cls).forward)
    calls = re.findall(r"\.update\((.*?)\)", src, re.S)
    print(f"\n{label}:")
    for call in calls:
        print(f"  update({' '.join(call.split())})")

print("\n=== what DeepSeek-V3 attention actually computes ===")
src = inspect.getsource(ds.DeepseekV3Attention.forward)
for line in src.splitlines():
    stripped = line.strip()
    if any(k in stripped for k in ("k_pe", "kv_a", "kv_b", "compressed", "q_a", "q_b",
                                   "key_states", "value_states", "cache_kwargs")):
        print(f"  {stripped[:110]}")

print("\n=== shapes that would matter for a cache translation ===")
for name in ("kv_lora_rank", "q_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim",
             "v_head_dim"):
    print(f"  {name}")

print("""
=== how to read this ===
  update(key_states, value_states, ...) with standard shapes
      -> the Llama/Mixtral cache translation transfers unchanged.
  update(compressed_kv, k_pe, ...) or anything latent-shaped
      -> the translation must be rewritten, and Petals' attn_cache_tokens sizing (which
         assumes num_kv_heads * head_dim per token) over-reserves by a large factor.
""")
