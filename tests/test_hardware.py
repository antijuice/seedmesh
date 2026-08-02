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
    SUPPORTED_MODEL_TYPES,
    GpuInfo,
    UnsupportedModelError,
    check_model_supported,
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


# ---- supported architectures ------------------------------------------------
#
# A volunteer was told to serve Qwen3-8B. `probe` sized it happily and reported a block
# plan; the server then downloaded config.json and died with "Petals does not support model
# type qwen3" at the bottom of a traceback. Sizing a model that can never be loaded is a
# confidently wrong answer, which is worse than no answer.


@pytest.mark.parametrize("model_type", SUPPORTED_MODEL_TYPES)
def test_every_architecture_petals_ships_is_accepted(model_type):
    assert check_model_supported({**BIG, "model_type": model_type}) == model_type


def test_the_list_matches_what_the_ported_petals_registers():
    """Measured from petals.utils.auto_config._CLASS_MAPPING on the patched checkout.
    If Petals gains an architecture, this is the assertion that should fail first."""
    assert set(SUPPORTED_MODEL_TYPES) == {"bloom", "falcon", "llama", "mixtral"}


def test_qwen3_is_rejected_by_name():
    with pytest.raises(UnsupportedModelError) as exc:
        check_model_supported({**BIG, "model_type": "qwen3"}, "Qwen/Qwen3-8B")
    message = str(exc.value)
    assert "Qwen/Qwen3-8B" in message and "qwen3" in message
    assert "llama" in message, "must name a working alternative, not just refuse"


def test_rejection_does_not_blame_licensing():
    """The failure is architectural. Someone reading 'unsupported' next to an open-weights
    model will otherwise go hunting for a licence or an HF token."""
    with pytest.raises(UnsupportedModelError) as exc:
        check_model_supported({**BIG, "model_type": "gemma2"}, "google/gemma-2-9b")
    assert "architecture" in str(exc.value).lower()


def test_model_type_is_matched_case_insensitively():
    assert check_model_supported({**BIG, "model_type": "LlaMA"}) == "llama"


def test_a_config_with_no_model_type_is_rejected_not_assumed():
    with pytest.raises(UnsupportedModelError):
        check_model_supported(dict(BIG), "mystery/model")


# ---- mixture-of-experts sizing ----------------------------------------------
#
# Exposed by the MoE spike. `params_per_block` counted one MLP, but a Mixtral block holds
# `num_local_experts` copies of it (8) and a DeepSeek-V3 block holds 256 routed experts plus
# shared ones. Sizing an MoE block as if it were dense understates it by 6-20x, and `probe`
# would confidently tell a volunteer to host blocks that cannot fit.


def test_experts_are_counted_not_ignored():
    from seedmesh.cli.hardware import params_per_block

    moe = {"hidden_size": 4096, "intermediate_size": 14336, "num_attention_heads": 32,
           "num_key_value_heads": 8, "num_local_experts": 8}
    dense = {k: v for k, v in moe.items() if k != "num_local_experts"}
    assert params_per_block(moe) > 5 * params_per_block(dense)


def test_deepseek_layer_sum_reproduces_the_published_total():
    """The strongest available check that the analytic formula is right: 58 MoE layers plus
    3 dense ones should add up to DeepSeek-V3's published 671B parameters."""
    from seedmesh.cli.hardware import params_per_layer

    cfg = {"hidden_size": 7168, "intermediate_size": 18432, "num_attention_heads": 128,
           "num_key_value_heads": 128, "n_routed_experts": 256, "n_shared_experts": 1,
           "moe_intermediate_size": 2048, "first_k_dense_replace": 3,
           "moe_layer_freq": 1, "num_hidden_layers": 61}
    total = sum(params_per_layer(cfg, i) for i in range(61))
    assert 0.95 * 671e9 < total < 1.05 * 671e9, f"{total/1e9:.0f}B vs published 671B"


def test_dense_prefix_layers_are_not_sized_as_moe():
    from seedmesh.cli.hardware import is_moe_layer, params_per_layer

    cfg = {"hidden_size": 7168, "intermediate_size": 18432, "num_attention_heads": 128,
           "num_key_value_heads": 128, "n_routed_experts": 256, "n_shared_experts": 1,
           "moe_intermediate_size": 2048, "first_k_dense_replace": 3, "num_hidden_layers": 61}
    assert not is_moe_layer(cfg, 0) and is_moe_layer(cfg, 3)
    assert params_per_layer(cfg, 10) > 10 * params_per_layer(cfg, 0)


def test_plan_sizes_against_the_largest_layer_not_the_smallest():
    """A volunteer does not choose their block range, so a plan that fits only the dense
    prefix is a plan that OOMs on assignment."""
    from seedmesh.cli.hardware import GpuInfo, plan_blocks

    cfg = {"hidden_size": 7168, "intermediate_size": 18432, "num_attention_heads": 128,
           "num_key_value_heads": 128, "n_routed_experts": 256, "n_shared_experts": 1,
           "moe_intermediate_size": 2048, "first_k_dense_replace": 3, "num_hidden_layers": 61}
    gpu = GpuInfo(name="test", total_bytes=24 * 2**30, free_bytes=24 * 2**30,
                  compute_capability="8.6")
    plan = plan_blocks(cfg, gpu, quant="nf4")
    # The MoE layer is ~11.5B params; at NF4 that is ~5.9 GB, so 24 GB cannot hold many.
    assert plan.params_per_block > 10e9, "sized against a dense layer by mistake"
    assert plan.recommended_blocks <= 4


def test_dense_models_are_unaffected():
    from seedmesh.cli.hardware import params_per_block, params_per_layer

    llama = {"hidden_size": 128, "intermediate_size": 256, "num_attention_heads": 8,
             "num_key_value_heads": 2, "num_hidden_layers": 4}
    assert params_per_block(llama) == 139_520
    assert params_per_layer(llama, 0) == 139_520
