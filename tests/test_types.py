

# ---- MoE expert kernel -------------------------------------------------------
#
# transformers 5.x dispatches mixture-of-experts through `_experts_implementation`, exactly
# as it dispatches attention through `_attn_implementation`. A hosted block has no model
# wrapper to set it. Measured on a tiny Mixtral config, CPU/fp32: unset and `batched_mm`
# both return NaN; only `grouped_mm` returns finite values. So the kernel is load-bearing
# for correctness, not just for numerics -- and once servers can differ, it has to be
# published or honest disagreement is read as fraud.


def test_dense_profiles_keep_their_key_unchanged():
    """Existing calibrated tolerances are keyed by this string; it must not shift."""
    from seedmesh.core.types import ComputeProfile

    assert ComputeProfile(quant="nf4", dtype="bf16", attn="eager").key == "nf4/bf16/eager"
    assert ComputeProfile().experts == "none"


def test_expert_kernel_enters_the_key_when_set():
    from seedmesh.core.types import ComputeProfile

    moe = ComputeProfile(quant="nf4", dtype="bf16", attn="eager", experts="grouped_mm")
    assert moe.key == "nf4/bf16/eager/grouped_mm"


def test_two_expert_kernels_are_different_profiles():
    """The whole point: they must not be compared against one tolerance."""
    from seedmesh.core.types import ComputeProfile

    a = ComputeProfile(experts="grouped_mm")
    b = ComputeProfile(experts="batched_mm")
    assert a.key != b.key
