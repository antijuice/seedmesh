"""Block-count sizing.

Getting this wrong has two failure modes and they are not symmetric: under-recommending
wastes donated capacity, while over-recommending gets the server OOM-killed mid-request --
which the swarm reads as churn and the volunteer reads as "this software broke my machine".
So the arithmetic is pinned by tests, including the case where nothing fits.
"""

from __future__ import annotations

import pytest

from seedmesh.cli.hardware import (
    BYTES_PER_PARAM,
    GpuInfo,
    describe_plan,
    params_per_block,
    plan_blocks,
)

# The config used in tools/verify_petals_port.py, whose real block measured 139,520 params.
TINY = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
}

# Llama-3-8B shaped.
BIG = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
}


def gpu(total_gib: float, free_gib: float | None = None) -> GpuInfo:
    total = int(total_gib * 2**30)
    free = int((free_gib if free_gib is not None else total_gib) * 2**30)
    return GpuInfo("Test GPU", total, free, "8.6")


def test_params_per_block_matches_a_real_measured_block():
    """Ground truth: this exact config built a real block of 139,520 parameters."""
    assert params_per_block(TINY) == 139_520


def test_params_per_block_handles_grouped_query_attention():
    """With kv_heads < heads, k and v are smaller than q -- the GQA saving."""
    mha = dict(BIG, num_key_value_heads=32)
    assert params_per_block(BIG) < params_per_block(mha)


def test_params_per_block_honours_explicit_head_dim():
    """Some configs set head_dim independently of hidden_size // heads."""
    explicit = dict(TINY, head_dim=32)
    assert params_per_block(explicit) > params_per_block(TINY)


def test_quantization_lets_a_small_card_host_more():
    """The reason NF4 matters: it is what makes ordinary hardware useful."""
    card = gpu(4)
    counts = {
        quant: plan_blocks(BIG, card, quant=quant).recommended_blocks
        for quant in ("none", "int8", "nf4")
    }
    assert counts["nf4"] > counts["int8"] > counts["none"]


def test_recommendation_never_exceeds_the_model():
    """A big GPU hosts the whole model, not more than exists."""
    plan = plan_blocks(BIG, gpu(80), quant="nf4")
    assert plan.recommended_blocks == BIG["num_hidden_layers"]
    assert plan.covers_whole_model


def test_tiny_gpu_recommends_zero_rather_than_a_negative_or_optimistic_count():
    """The OOM case must be reported, not rounded up to 'one block, probably fine'."""
    plan = plan_blocks(BIG, gpu(1), quant="none")
    assert plan.recommended_blocks == 0
    assert not plan.covers_whole_model
    assert any("cannot host even one block" in line for line in describe_plan(plan, gpu(1)))


def test_reserve_is_subtracted_from_free_not_total():
    """A GPU with a display or another process attached has less to give."""
    busy = plan_blocks(BIG, gpu(24, free_gib=6), quant="nf4").recommended_blocks
    idle = plan_blocks(BIG, gpu(24, free_gib=24), quant="nf4").recommended_blocks
    assert busy < idle


def test_explicit_reserve_overrides_the_heuristic():
    """Unquantized on purpose: at NF4 a 24 GiB card fits all 32 blocks either way, so the
    model-size cap would hide the reserve's effect entirely."""
    generous = plan_blocks(BIG, gpu(24), quant="none", reserve_bytes=0).recommended_blocks
    cautious = plan_blocks(BIG, gpu(24), quant="none", reserve_bytes=12 * 2**30).recommended_blocks
    assert generous > cautious
    assert cautious > 0


def test_unknown_quantization_is_rejected():
    with pytest.raises(ValueError):
        plan_blocks(BIG, gpu(24), quant="int3")


def test_measured_bytes_per_param_are_the_ones_the_spike_produced():
    """Regression: these came from measurement, not from Petals' constants."""
    assert BYTES_PER_PARAM["int8"] == 1.0
    assert BYTES_PER_PARAM["nf4"] == pytest.approx(0.516, abs=1e-3)


def test_4gb_card_sizing_matches_the_quantization_spike():
    """The spike measured ~28 NF4 blocks of an 8B-class model on a 4GB card."""
    plan = plan_blocks(BIG, gpu(4), quant="nf4")
    # The spike reserved a flat 1 GiB; this reserves max(1 GiB, 15%), so expect >= 20.
    assert 20 <= plan.recommended_blocks <= 32
