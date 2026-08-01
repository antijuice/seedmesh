"""Does the ported block compute the same thing as stock transformers?

"It imports and runs" is not a port. A block that silently computes slightly wrong
attention would produce plausible text, pass every smoke test, and quietly poison every
inference routed through it -- and in Seedmesh specifically, it would be indistinguishable
from an honest server to the verification layer, because redundant execution checks
*agreement*, not correctness. If every server runs the same subtly wrong port, they all
agree.

So the port is checked against the reference implementation numerically, including the
incremental-decoding path where the cache-layout translation lives. That path is where the
real bugs are: a cache reshape that transposes two dimensions produces correct output for
a single forward pass and wrong output from the second token onward.

Run with the spike venv:
    .venv-spike/Scripts/python.exe spike/transformers_port/test_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)

sys.path.insert(0, str(Path(__file__).parent))
from ported_block import WrappedLlamaBlock, build_causal_mask  # noqa: E402

TOLERANCE = 1e-5


def make_config() -> LlamaConfig:
    """Small but structurally realistic: GQA on (4 query heads, 2 kv heads)."""
    return LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
    )


def reference_forward(
    layer: LlamaDecoderLayer,
    rotary: LlamaRotaryEmbedding,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """Drive the stock layer the way ``LlamaModel`` drives it: full sequence, no cache."""
    batch_size, seq_length, _ = hidden.shape
    position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
    position_embeddings = rotary(hidden, position_ids)
    mask = build_causal_mask(batch_size, seq_length, seq_length, hidden.dtype, hidden.device)

    out = layer(
        hidden,
        attention_mask=mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=position_embeddings,
    )
    return out[0] if isinstance(out, tuple) else out


def test_single_pass_matches_reference() -> bool:
    torch.manual_seed(0)
    config = make_config()

    reference = LlamaDecoderLayer(config, layer_idx=0).eval()
    rotary = LlamaRotaryEmbedding(config=config)

    ported = WrappedLlamaBlock(config, layer_idx=0).eval()
    missing, unexpected = ported.load_state_dict(reference.state_dict(), strict=False)

    hidden = torch.randn(2, 6, config.hidden_size)

    with torch.no_grad():
        expected = reference_forward(reference, rotary, hidden)
        actual = ported(hidden)[0]

    delta = (expected - actual).abs().max().item()
    ok = delta < TOLERANCE

    print(f"  state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing or unexpected:
        print(f"    missing={missing} unexpected={unexpected}")
    print(f"  max abs difference vs reference: {delta:.3e}")
    return ok and not unexpected


def test_incremental_decoding_matches_full_sequence() -> bool:
    """The test that actually catches cache-layout bugs.

    Feed 4 tokens, then 1 more using the returned cache, and compare that final token's
    output against feeding all 5 at once. A wrong cache reshape passes the single-pass test
    above and fails here.
    """
    torch.manual_seed(1)
    config = make_config()

    reference = LlamaDecoderLayer(config, layer_idx=0).eval()
    rotary = LlamaRotaryEmbedding(config=config)
    ported = WrappedLlamaBlock(config, layer_idx=0).eval()
    ported.load_state_dict(reference.state_dict(), strict=False)

    hidden = torch.randn(2, 5, config.hidden_size)

    with torch.no_grad():
        full = reference_forward(reference, rotary, hidden)

        first = ported(hidden[:, :4, :], use_cache=True)
        cache = first[1]
        second = ported(hidden[:, 4:, :], layer_past=cache, use_cache=True)

    delta_prefix = (full[:, :4, :] - first[0]).abs().max().item()
    delta_final = (full[:, 4:, :] - second[0]).abs().max().item()

    print(f"  prefix (4 tokens)  max abs difference: {delta_prefix:.3e}")
    print(f"  final token (cached) max abs difference: {delta_final:.3e}")
    return delta_prefix < TOLERANCE and delta_final < TOLERANCE


def test_cache_roundtrip_preserves_values() -> bool:
    """The Petals<->transformers cache reshape must be exactly invertible."""
    torch.manual_seed(2)
    batch_size, heads, seq, dim = 2, 2, 7, 16

    key = torch.randn(batch_size, heads, seq, dim)
    value = torch.randn(batch_size, heads, seq, dim)

    petals_form = WrappedLlamaBlock._cache_to_petals(key, value, batch_size, heads, dim)
    back_key, back_value = WrappedLlamaBlock._cache_from_petals(
        petals_form, batch_size, heads, dim
    )

    delta = max(
        (key - back_key).abs().max().item(), (value - back_value).abs().max().item()
    )
    print(f"  roundtrip max abs difference: {delta:.3e}")
    return delta == 0.0


def test_causal_mask_structure() -> bool:
    """Check the mask tensor directly.

    Testing this end-to-end is misleading: causal masking already blocks later positions,
    so an end-to-end padding test passes even when the padding logic is entirely broken.
    The mask itself is what needs asserting.
    """
    minimum = torch.finfo(torch.float32).min
    ok = True

    # Prefill: strictly lower-triangular attention.
    mask = build_causal_mask(1, 4, 4, torch.float32, torch.device("cpu"))
    expected_blocked = [(i, j) for i in range(4) for j in range(4) if j > i]
    for i, j in expected_blocked:
        if mask[0, 0, i, j] != minimum:
            print(f"  prefill: query {i} should not see key {j}")
            ok = False
    for i in range(4):
        for j in range(i + 1):
            if mask[0, 0, i, j] != 0:
                print(f"  prefill: query {i} should see key {j}")
                ok = False

    # Incremental decode: one new query, absolute position 4, sees all five keys.
    mask = build_causal_mask(1, 1, 5, torch.float32, torch.device("cpu"))
    if (mask[0, 0, 0, :] != 0).any():
        print("  decode: the new token should attend to the whole cached prefix")
        ok = False

    # Padding must block padded keys even where causality would allow them.
    padding = torch.tensor([[True, True, False, True]])
    mask = build_causal_mask(1, 4, 4, torch.float32, torch.device("cpu"), padding_mask=padding)
    if mask[0, 0, 3, 2] != minimum:
        print("  padding: query 3 should not see padded key 2")
        ok = False
    if mask[0, 0, 3, 1] != 0:
        print("  padding: query 3 should still see real key 1")
        ok = False

    print(f"  mask shape {tuple(mask.shape)}, structure {'correct' if ok else 'WRONG'}")
    return ok


def main() -> int:
    import transformers

    print(f"torch {torch.__version__} / transformers {transformers.__version__}\n")

    tests = [
        ("single forward pass matches stock LlamaDecoderLayer", test_single_pass_matches_reference),
        ("incremental decoding matches full sequence", test_incremental_decoding_matches_full_sequence),
        ("cache format roundtrip is lossless", test_cache_roundtrip_preserves_values),
        ("causal mask structure is correct", test_causal_mask_structure),
    ]

    failures = 0
    for name, test in tests:
        print(f"* {name}")
        try:
            passed = test()
        except Exception as exc:
            import traceback

            print(f"  RAISED {type(exc).__name__}: {exc}")
            traceback.print_exc()
            passed = False
        print(f"  -> {'PASS' if passed else 'FAIL'}\n")
        failures += not passed

    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
