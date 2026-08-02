"""What shapes does an MLA cache actually hold?

I claimed in docs/findings-moe.md that the Llama cache translation "transfers to DeepSeek-V3
unchanged". Re-reading the source, that looks wrong: `update()` is called with

    key_states   width qk_nope_head_dim + qk_rope_head_dim   (128 + 64 = 192)
    value_states width v_head_dim                            (128)

and the F.pad that widens value to match happens *after* update, i.e. on the way out, not on
the way in. If so, K and V are stored at different widths -- and the Llama translation does
`value_states.view(*key_states.shape)`, which assumes they are identical.

Measure it rather than reading it. Two wrong readings of this area already.
"""

import torch
from transformers import DeepseekV3Config
from transformers.cache_utils import DynamicCache
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3DecoderLayer

config = DeepseekV3Config(
    hidden_size=128,
    intermediate_size=256,
    moe_intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_routed_experts=4,
    n_shared_experts=1,
    num_experts_per_tok=2,
    first_k_dense_replace=1,
    q_lora_rank=64,
    kv_lora_rank=32,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    max_position_embeddings=64,
    vocab_size=128,
)
config._attn_implementation = "eager"
config._experts_implementation = "grouped_mm"

print("config geometry")
print(f"  heads={config.num_attention_heads}  qk_nope={config.qk_nope_head_dim} "
      f"qk_rope={config.qk_rope_head_dim}  v_head_dim={config.v_head_dim}")
print(f"  => qk_head_dim = {config.qk_nope_head_dim + config.qk_rope_head_dim}")

layer = DeepseekV3DecoderLayer(config, layer_idx=0)
layer.eval()

from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding

rotary = DeepseekV3RotaryEmbedding(config=config)
hidden = torch.randn(1, 6, config.hidden_size)
position_ids = torch.arange(6).unsqueeze(0)
position_embeddings = rotary(hidden, position_ids)

cache = DynamicCache(config=config)
with torch.inference_mode():
    out = layer(
        hidden,
        position_embeddings=position_embeddings,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=cache,
    )

tensor = out[0] if isinstance(out, tuple) else out
print(f"\noutput {tuple(tensor.shape)}  finite={bool(torch.isfinite(tensor).all())}")

print("\n=== what the cache holds ===")
try:
    keys, values = cache[0]
except Exception:
    keys, values = cache.layers[0].keys, cache.layers[0].values
print(f"  key   {tuple(keys.shape)}")
print(f"  value {tuple(values.shape)}")

same = keys.shape == values.shape
print(f"\n  K and V same shape: {same}")
print("""
  same  -> the Llama translation (value.view(*key.shape)) transfers unchanged.
  differ-> it does NOT. The translation must carry separate widths for K and V, and my
           earlier "transfers unchanged" claim in findings-moe.md is wrong.
""")
