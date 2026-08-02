"""How much attention cache does a DeepSeek-V3 block actually need per token?

CORRECTION to an earlier claim in docs/findings-moe.md. I wrote that Petals would
"over-reserve, badly, since MLA's whole point is a far smaller cache". That reasoning was
about the MLA *design*, not about the implementation actually in use. `probe_mla_cache.py`
shows transformers decompresses the latent back to full K/V before `Cache.update`, so the
stored cache is standard-shaped -- and, because DeepSeek-V3 has 128 heads with a 192-wide
qk head, it is very large rather than very small.

So the risk is the opposite of what I said: **under**-reserving, not over-reserving. Which
matters more, because under-reserving means a server accepts sessions it cannot serve and
OOMs mid-request.

Compares what Petals computes against what the implementation actually stores.
"""

import json

from huggingface_hub import hf_hub_download


def cfg(repo):
    return json.loads(open(hf_hub_download(repo, "config.json"), encoding="utf-8").read())


def petals_bytes_per_token(c, dtype_bytes=2):
    """What Petals actually reserves, from petals/server/server.py:209::

        cache_values_per_block = 2 * self.block_config.hidden_size * attn_cache_tokens

    ...followed immediately by line 210, which is easy to miss and matters::

        cache_values_per_block //= self.block_config.num_key_value_groups

    With that division the formula is correct for grouped-query attention: hidden_size /
    groups == kv_heads * head_dim. It is wrong only where that identity fails, which is MLA
    -- there `num_key_value_groups` is 1 (heads == kv_heads) so no division happens, while
    the real cache is much wider than hidden_size because qk_head_dim = nope + rope.
    """
    heads = c["num_attention_heads"]
    kv_heads = c.get("num_key_value_heads", heads)
    groups = max(1, heads // kv_heads)
    return (2 * c["hidden_size"] // groups) * dtype_bytes


def standard_bytes_per_token(c, dtype_bytes=2):
    """What a standard GQA/MHA attention actually stores."""
    heads = c["num_attention_heads"]
    kv_heads = c.get("num_key_value_heads", heads)
    head_dim = c.get("head_dim") or c["hidden_size"] // heads
    return 2 * kv_heads * head_dim * dtype_bytes


def actual_bytes_per_token(c, dtype_bytes=2):
    """What transformers' DeepseekV3Attention actually stores after `update`.

    key_states is (b, heads, seq, qk_head_dim) where qk_head_dim = nope + rope.
    value_states is v_head_dim wide, then F.pad'd up to qk_head_dim.
    """
    heads = c["num_attention_heads"]
    qk_head_dim = c["qk_nope_head_dim"] + c["qk_rope_head_dim"]
    return 2 * heads * qk_head_dim * dtype_bytes


print(f"{'model':<26} {'attn':<8} {'petals':>13} {'needs':>13} {'ratio':>7}")
print("-" * 82)

for repo, label in (
    ("NousResearch/Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    ("deepseek-ai/DeepSeek-V3", "DeepSeek-V3"),
    ("moonshotai/Kimi-K2-Instruct", "Kimi K2"),
):
    try:
        c = cfg(repo)
    except Exception as exc:
        print(f"{label:<26} unavailable ({type(exc).__name__})")
        continue

    reserved = petals_bytes_per_token(c)
    if "qk_nope_head_dim" in c:
        actual = actual_bytes_per_token(c)
        kind = "MLA"
    else:
        actual = standard_bytes_per_token(c)
        kind = "GQA/MHA"
    ratio = actual / reserved
    if ratio > 1.05:
        flag = "  <-- UNDER-reserves, OOMs mid-request"
    elif ratio < 0.95:
        flag = "  <-- over-reserves, wastes VRAM"
    else:
        flag = "  <-- correct"
    print(f"{label:<26} {kind:<8} {reserved:>11,} B {actual:>11,} B {ratio:>6.2f}x{flag}")

print("""
Per token PER LAYER, bf16. Multiply by sequence length and by blocks hosted.

A server that reserves too little accepts sessions it cannot serve and OOMs partway through
a request -- which in this swarm looks like an unreliable peer rather than a misconfigured
one, and costs it reputation it did not deserve to lose.
""")
