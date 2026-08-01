"""Inventory the transformers API surface that Petals' Llama block depends on.

Petals pins ``transformers==4.43.1``. This script reports, for whatever version is actually
installed, which of the symbols and attributes that ``petals/models/llama/block.py`` reaches
for still exist. It is the cheapest possible measurement of the porting gap: no model
weights, no downloads, no GPU.

Run it against several transformers versions to see where each dependency broke.
"""

from __future__ import annotations

import inspect
import sys


def check_import(module_path: str, name: str) -> tuple[bool, str]:
    try:
        module = __import__(module_path, fromlist=[name])
    except Exception as exc:
        return False, f"module import failed: {type(exc).__name__}: {exc}"
    if not hasattr(module, name):
        return False, "symbol not found"
    return True, "present"


def signature_of(module_path: str, cls_name: str, method: str) -> str:
    try:
        module = __import__(module_path, fromlist=[cls_name])
        cls = getattr(module, cls_name)
        return str(inspect.signature(getattr(cls, method)))
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def main() -> int:
    import torch
    import transformers

    print(f"python       {sys.version.split()[0]}")
    print(f"torch        {torch.__version__}")
    print(f"transformers {transformers.__version__}")
    print()

    # Every symbol block.py imports at module scope.
    imports = [
        ("transformers.modeling_attn_mask_utils", "_prepare_4d_causal_attention_mask"),
        ("transformers.models.llama.modeling_llama", "LlamaAttention"),
        ("transformers.models.llama.modeling_llama", "LlamaConfig"),
        ("transformers.models.llama.modeling_llama", "LlamaDecoderLayer"),
        ("transformers.models.llama.modeling_llama", "LlamaMLP"),
        ("transformers.models.llama.modeling_llama", "LlamaRMSNorm"),
        ("transformers.models.llama.modeling_llama", "repeat_kv"),
        ("transformers.models.llama.modeling_llama", "rotate_half"),
        ("transformers.models.llama.modeling_llama", "apply_rotary_pos_emb"),
        ("transformers.models.llama.modeling_llama", "LlamaRotaryEmbedding"),
    ]

    print("=== module-scope imports in petals/models/llama/block.py ===")
    broken = 0
    for module_path, name in imports:
        ok, detail = check_import(module_path, name)
        if not ok:
            broken += 1
        print(f"  {'OK  ' if ok else 'GONE'}  {module_path}.{name:38s} {'' if ok else detail}")

    print()
    print("=== forward() signatures Petals overrides ===")
    for cls_name in ("LlamaAttention", "LlamaDecoderLayer"):
        print(f"  {cls_name}.forward")
        print(f"    {signature_of('transformers.models.llama.modeling_llama', cls_name, 'forward')}")
    print("  LlamaAttention.__init__")
    print(f"    {signature_of('transformers.models.llama.modeling_llama', 'LlamaAttention', '__init__')}")

    print()
    print("=== instance attributes Petals reads off LlamaAttention ===")
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig

        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
        )
        attention = LlamaAttention(config=config, layer_idx=0)
        for attribute in (
            "num_heads",
            "hidden_size",
            "num_key_value_heads",
            "num_key_value_groups",
            "head_dim",
            "rotary_emb",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ):
            present = hasattr(attention, attribute)
            print(f"  {'OK  ' if present else 'GONE'}  self.{attribute}")
    except Exception as exc:
        print(f"  could not instantiate LlamaAttention: {type(exc).__name__}: {exc}")

    print()
    print("=== config fields Petals branches on ===")
    try:
        from transformers.models.llama.modeling_llama import LlamaConfig

        config = LlamaConfig()
        for field in ("pretraining_tp", "rms_norm_eps", "hidden_size"):
            print(f"  {'OK  ' if hasattr(config, field) else 'GONE'}  config.{field}")
    except Exception as exc:
        print(f"  config probe failed: {exc}")

    print()
    print("=== cache API ===")
    for module_path, name in [
        ("transformers", "Cache"),
        ("transformers", "DynamicCache"),
        ("transformers.cache_utils", "DynamicCache"),
    ]:
        ok, detail = check_import(module_path, name)
        print(f"  {'OK  ' if ok else 'GONE'}  {module_path}.{name:24s} {'' if ok else detail}")

    print()
    print("=== mask utilities (where _prepare_4d_causal_attention_mask went) ===")
    for module_path, name in [
        ("transformers.masking_utils", "create_causal_mask"),
        ("transformers.modeling_attn_mask_utils", "AttentionMaskConverter"),
    ]:
        ok, detail = check_import(module_path, name)
        print(f"  {'OK  ' if ok else 'GONE'}  {module_path}.{name:32s} {'' if ok else detail}")

    print()
    print(f"RESULT: {broken}/{len(imports)} module-scope imports are gone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
