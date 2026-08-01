"""Sketching, tolerance comparison, calibration and sampling.

The first test in this file is the one that motivated the whole module: the spec's
hash-based comparison would have failed it, and with it every honest heterogeneous pair in
the swarm.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

from seedmesh.core.types import BlockRange, ServerInfo, Verdict
from seedmesh.reputation.diversity import ClusterIndex
from seedmesh.reputation.scorer import ReputationScorer, ScorerConfig
from seedmesh.verification.calibrate import (
    MIN_CALIBRATION_SAMPLES,
    CalibrationCollector,
    calibrate,
    empirical_false_positive_rate,
)
from seedmesh.verification.compare import Tolerance, compare_sketches, detect_passthrough
from seedmesh.verification.sampler import SamplerConfig, VerificationSampler
from seedmesh.verification.sketch import SketchSpec, derive_seed, sketch_hidden_state

RANGE = BlockRange("m", 0, 8)


def _hidden(rng, tokens=8, dim=256):
    return rng.standard_normal((tokens, dim))


def _fp_noise(rng, array, magnitude):
    """Stand-in for the low-bit disagreement between different GPUs and dtypes."""
    scale = np.linalg.norm(array) / np.sqrt(array.size)
    return array + rng.standard_normal(array.shape) * magnitude * scale


# ---- the headline correction --------------------------------------------------


def test_exact_hashing_would_reject_honest_pairs():
    """Demonstrates the defect this module exists to fix.

    Two honest servers computing the same thing on different hardware differ in the low
    bits. Under the spec's hash comparison every such pair reads as a mismatch, so the
    check would evict exactly the heterogeneous volunteers the network is built from.
    """
    rng = np.random.default_rng(0)
    truth = _hidden(rng)
    a = _fp_noise(rng, truth, 3e-3)
    b = _fp_noise(rng, truth, 4e-3)

    assert hashlib.sha256(a.tobytes()).digest() != hashlib.sha256(b.tobytes()).digest()

    seed = derive_seed(b"\x01" * 16, RANGE.key(), 0)
    comparison = compare_sketches(
        sketch_hidden_state(a, seed=seed), sketch_hidden_state(b, seed=seed), Tolerance()
    )
    assert comparison.verdict is Verdict.MATCH


def test_fabricated_output_is_caught():
    rng = np.random.default_rng(1)
    truth = _hidden(rng)
    honest = _fp_noise(rng, truth, 3e-3)
    garbage = rng.standard_normal(truth.shape)

    seed = derive_seed(b"\x02" * 16, RANGE.key(), 0)
    comparison = compare_sketches(
        sketch_hidden_state(honest, seed=seed),
        sketch_hidden_state(garbage, seed=seed),
        Tolerance(),
    )
    assert comparison.verdict is Verdict.MISMATCH


def test_subtle_corruption_is_separated_from_honest_noise():
    rng = np.random.default_rng(2)
    truth = _hidden(rng)
    seed = derive_seed(b"\x03" * 16, RANGE.key(), 0)

    honest = sketch_hidden_state(_fp_noise(rng, truth, 3e-3), seed=seed)
    corrupt = sketch_hidden_state(_fp_noise(rng, truth, 0.25), seed=seed)

    assert compare_sketches(honest, sketch_hidden_state(_fp_noise(rng, truth, 3e-3), seed=seed)).verdict is Verdict.MATCH
    assert compare_sketches(honest, corrupt).verdict is not Verdict.MATCH


# ---- sketch mechanics ----------------------------------------------------------


def test_sketch_is_deterministic_for_a_given_seed():
    rng = np.random.default_rng(3)
    hidden = _hidden(rng)
    left = sketch_hidden_state(hidden, seed=99)
    right = sketch_hidden_state(hidden, seed=99)
    assert np.allclose(left.values, right.values)


def test_different_seeds_give_different_projections():
    rng = np.random.default_rng(4)
    hidden = _hidden(rng)
    assert not np.allclose(
        sketch_hidden_state(hidden, seed=1).values, sketch_hidden_state(hidden, seed=2).values
    )


def test_sketch_is_far_smaller_than_the_tensor():
    rng = np.random.default_rng(5)
    hidden = _hidden(rng, tokens=64, dim=4096)
    sketch = sketch_hidden_state(hidden, seed=7)
    assert sketch.components == 64
    assert sketch.components < hidden.size / 1000


def test_sketch_size_is_independent_of_sequence_length():
    """Comparison must not require both servers to have seen identical batch shapes."""
    rng = np.random.default_rng(6)
    short = sketch_hidden_state(_hidden(rng, tokens=4), seed=11)
    long = sketch_hidden_state(_hidden(rng, tokens=64), seed=11)
    assert short.components == long.components


def test_seed_derivation_binds_to_range_and_step():
    """Stops a server replaying a sketch computed for a different pipeline position."""
    nonce = b"\x09" * 16
    assert derive_seed(nonce, "m:0-8", 0) != derive_seed(nonce, "m:8-16", 0)
    assert derive_seed(nonce, "m:0-8", 0) != derive_seed(nonce, "m:0-8", 1)


def test_short_nonces_are_rejected():
    with pytest.raises(ValueError):
        derive_seed(b"tooshort", "m:0-8", 0)


def test_non_finite_hidden_states_are_rejected():
    bad = np.array([[1.0, float("nan")]])
    with pytest.raises(ValueError):
        sketch_hidden_state(bad, seed=1)


def test_tiny_sketches_are_rejected():
    with pytest.raises(ValueError):
        SketchSpec(components=4)


# ---- comparison semantics ------------------------------------------------------


def test_ambiguous_band_yields_inconclusive_not_mismatch():
    rng = np.random.default_rng(7)
    truth = _hidden(rng)
    seed = derive_seed(b"\x0a" * 16, RANGE.key(), 0)
    a = sketch_hidden_state(truth, seed=seed)
    b = sketch_hidden_state(_fp_noise(rng, truth, 0.04), seed=seed)

    tolerance = Tolerance(match_distance=0.001, mismatch_distance=0.9, cosine_distance=1e-6)
    assert compare_sketches(a, b, tolerance).verdict is Verdict.INCONCLUSIVE


def test_tolerance_requires_a_nonempty_inconclusive_band():
    with pytest.raises(ValueError):
        Tolerance(match_distance=0.5, mismatch_distance=0.1)


def test_mismatched_dimensions_are_inconclusive_not_accusatory():
    """A model/catalog mismatch is a routing error, not evidence of cheating."""
    rng = np.random.default_rng(8)
    a = sketch_hidden_state(_hidden(rng, dim=128), seed=1)
    b = sketch_hidden_state(_hidden(rng, dim=256), seed=1)
    comparison = compare_sketches(a, b)
    assert comparison.verdict is Verdict.INCONCLUSIVE
    assert not comparison.comparable, "structural failures must not accumulate as evidence"


def test_ambiguous_band_results_are_marked_comparable():
    """Unlike structural failures, these are weak evidence and must accumulate."""
    rng = np.random.default_rng(18)
    truth = _hidden(rng)
    seed = derive_seed(b"\x0f" * 16, RANGE.key(), 0)
    a = sketch_hidden_state(truth, seed=seed)
    b = sketch_hidden_state(_fp_noise(rng, truth, 0.04), seed=seed)

    tolerance = Tolerance(match_distance=0.001, mismatch_distance=0.9, cosine_distance=1e-6)
    comparison = compare_sketches(a, b, tolerance)
    assert comparison.verdict is Verdict.INCONCLUSIVE
    assert comparison.comparable


def test_structural_inconclusives_are_not_charged_to_either_peer():
    from seedmesh.core.types import Verdict as V

    sampler = _sampler()
    subject = _server("smsubject", asn=1, first_seen=0.0)
    verifier = _server("smverifier", asn=2, first_seen=50_000.0)
    task = sampler.plan(subject, RANGE, [verifier], force=True)

    for _ in range(20):
        sampler.record_outcome(task, V.INCONCLUSIVE, comparable=False)

    assert sampler.scorer.breakdown("smsubject").integrity == pytest.approx(1.0)
    assert sampler.scorer.breakdown("smverifier").integrity == pytest.approx(1.0)


def test_passthrough_detection_catches_an_echoing_server():
    rng = np.random.default_rng(9)
    hidden = _hidden(rng)
    seed = derive_seed(b"\x0b" * 16, RANGE.key(), 0)

    echoed = sketch_hidden_state(_fp_noise(rng, hidden, 1e-3), seed=seed)
    assert detect_passthrough(sketch_hidden_state(hidden, seed=seed), echoed)


def test_passthrough_detection_clears_a_working_server():
    rng = np.random.default_rng(10)
    hidden = _hidden(rng)
    seed = derive_seed(b"\x0c" * 16, RANGE.key(), 0)
    transformed = hidden + np.tanh(hidden @ rng.standard_normal((256, 256)) / 16)

    assert not detect_passthrough(
        sketch_hidden_state(hidden, seed=seed), sketch_hidden_state(transformed, seed=seed)
    )


# ---- calibration ---------------------------------------------------------------


def test_calibration_refuses_a_sample_too_small_to_mean_anything():
    collector = CalibrationCollector()
    for _ in range(MIN_CALIBRATION_SAMPLES - 1):
        collector.add(0.01, 0.0001, 0.001)
    with pytest.raises(ValueError):
        calibrate(collector.sample())


def test_calibration_hits_its_target_false_positive_rate():
    rng = np.random.default_rng(11)
    distances = np.abs(rng.normal(0.01, 0.003, size=4000))

    collector = CalibrationCollector()
    for value in distances:
        collector.add(float(value), float(value) / 100, float(value) / 10)

    tolerance = calibrate(collector.sample(), target_false_positive_rate=0.01)
    observed = empirical_false_positive_rate(list(distances), tolerance)
    assert observed == pytest.approx(0.01, abs=0.005)


def test_calibration_leaves_an_inconclusive_band():
    collector = CalibrationCollector()
    for _ in range(MIN_CALIBRATION_SAMPLES * 2):
        collector.add(0.01, 0.0001, 0.001)
    tolerance = calibrate(collector.sample())
    assert tolerance.mismatch_distance > tolerance.match_distance


# ---- sampling and verifier selection -------------------------------------------


def _server(peer_id, asn, first_seen=0.0, start=0, end=8):
    return ServerInfo(
        peer_id=peer_id,
        block_range=BlockRange("m", start, end),
        first_seen=first_seen,
        address=f"198.51.{asn % 250}.{abs(hash(peer_id)) % 250 + 1}",
        asn=asn,
    )


def _sampler(**overrides):
    scorer = ReputationScorer(ScorerConfig())
    return VerificationSampler(
        scorer, ClusterIndex(), SamplerConfig(**overrides), rng=random.Random(0)
    )


def test_unproven_peers_are_verified_more_often_than_proven_ones():
    from seedmesh.core.clock import ManualClock
    from seedmesh.core.types import Observation, Outcome

    clock = ManualClock()
    scorer = ReputationScorer(ScorerConfig(), clock=clock)
    sampler = VerificationSampler(scorer, ClusterIndex(), SamplerConfig(), rng=random.Random(0))

    unproven = sampler.verification_rate("smnew")
    for _ in range(400):
        scorer.record(Observation("smobs", "smproven", RANGE, Outcome.OK, 30.0, clock.now()))

    assert unproven > sampler.verification_rate("smproven")
    assert sampler.verification_rate("smproven") == pytest.approx(
        sampler.config.base_rate, abs=0.03
    )


def test_a_verifier_in_the_same_cluster_is_refused():
    """The spec's own open question: 'any two servers hosting this block range' is
    trivially defeated when one operator controls both."""
    sampler = _sampler()
    subject = _server("smsubject", asn=64999, first_seen=0.0)
    sibling = _server("smsibling", asn=64999, first_seen=0.0)

    assert sampler.select_verifier(subject, RANGE, [sibling]) is None


def test_a_verifier_that_appeared_simultaneously_is_refused():
    sampler = _sampler(min_first_seen_gap_s=900.0)
    subject = _server("smsubject", asn=1, first_seen=1000.0)
    twin = _server("smtwin", asn=2, first_seen=1010.0)

    assert sampler.select_verifier(subject, RANGE, [twin]) is None


def test_an_independent_verifier_is_accepted():
    sampler = _sampler()
    subject = _server("smsubject", asn=1, first_seen=0.0)
    independent = _server("smindependent", asn=2, first_seen=50_000.0)

    assert sampler.select_verifier(subject, RANGE, [independent]).peer_id == "smindependent"


def test_peers_with_unknown_location_are_never_paired():
    """Address-less peers share one bucket, so they can never be shown to be independent."""
    sampler = _sampler()
    subject = ServerInfo("smsubject", RANGE, 0.0)
    other = ServerInfo("smother", RANGE, 50_000.0)
    assert sampler.select_verifier(subject, RANGE, [other]) is None


def test_a_verifier_that_cannot_cover_the_range_is_refused():
    sampler = _sampler()
    subject = _server("smsubject", asn=1, first_seen=0.0)
    partial = _server("smpartial", asn=2, first_seen=50_000.0, start=0, end=4)
    assert sampler.select_verifier(subject, BlockRange("m", 0, 8), [partial]) is None


def test_pairings_rotate_rather_than_fixating():
    sampler = _sampler()
    subject = _server("smsubject", asn=1, first_seen=0.0)
    candidates = [_server(f"smv{i}", asn=100 + i, first_seen=50_000.0) for i in range(4)]

    chosen = set()
    for _ in range(12):
        verifier = sampler.select_verifier(subject, RANGE, candidates)
        chosen.add(verifier.peer_id)
        sampler.history.record_pairing(subject.peer_id, verifier.peer_id)

    assert len(chosen) == 4


# ---- compute profiles and per-pair tolerance ----------------------------------
#
# Regression tests for a defect found by the quantization spike. Measured honest-pair
# distances (spike/quantization/): fp16-vs-int8 ~0.005, but fp16-vs-NF4 ~0.055. Against a
# single global threshold fitted to the former, an honest NF4 server lands in the ambiguous
# band on *every* check and is quarantined after six.


def _profiled(peer_id, asn, quant, first_seen=0.0):
    from seedmesh.core.types import ComputeProfile

    return ServerInfo(
        peer_id=peer_id,
        block_range=BlockRange("m", 0, 8),
        first_seen=first_seen,
        address=f"198.51.{asn % 250}.{abs(hash(peer_id)) % 250 + 1}",
        asn=asn,
        profile=ComputeProfile(quant=quant),
    )


def _table_without_nf4_cross():
    from seedmesh.verification.calibrate import ToleranceTable

    table = ToleranceTable()
    tight = Tolerance(match_distance=0.0065, mismatch_distance=0.039)
    table.set("none/fp16/eager", "none/fp16/eager", tight)
    table.set("int8/fp16/eager", "int8/fp16/eager", tight)
    table.set("none/fp16/eager", "int8/fp16/eager", tight)
    return table


def test_tolerance_lookup_is_order_independent():
    """Comparing A to B must give the same verdict as B to A."""
    table = _table_without_nf4_cross()
    assert table.get("none/fp16/eager", "int8/fp16/eager") is table.get(
        "int8/fp16/eager", "none/fp16/eager"
    )


def test_uncalibrated_profile_pair_is_not_verifiable():
    """A check judged against a made-up threshold is worse than no check."""
    table = _table_without_nf4_cross()
    assert table.get("none/fp16/eager", "nf4/fp16/eager") is None
    assert not table.can_compare("none/fp16/eager", "nf4/fp16/eager")


def test_sampler_refuses_an_uncalibrated_cross_profile_verifier():
    """Regression: without this, honest NF4 volunteers get quarantined after ~6 checks."""
    scorer = ReputationScorer(ScorerConfig())
    sampler = VerificationSampler(
        scorer,
        ClusterIndex(),
        SamplerConfig(),
        rng=random.Random(0),
        tolerances=_table_without_nf4_cross(),
    )
    nf4_subject = _profiled("smnf4", asn=1, quant="nf4")
    fp16_verifier = _profiled("smfp16", asn=2, quant="none", first_seen=50_000.0)

    assert sampler.select_verifier(nf4_subject, RANGE, [fp16_verifier]) is None


def test_sampler_accepts_a_calibrated_cross_profile_verifier():
    table = _table_without_nf4_cross()
    table.set(
        "none/fp16/eager",
        "nf4/fp16/eager",
        Tolerance(match_distance=0.0694, mismatch_distance=0.4162),
    )

    scorer = ReputationScorer(ScorerConfig())
    sampler = VerificationSampler(
        scorer, ClusterIndex(), SamplerConfig(), rng=random.Random(0), tolerances=table
    )
    nf4_subject = _profiled("smnf4", asn=1, quant="nf4")
    fp16_verifier = _profiled("smfp16", asn=2, quant="none", first_seen=50_000.0)

    chosen = sampler.select_verifier(nf4_subject, RANGE, [fp16_verifier])
    assert chosen is not None and chosen.peer_id == "smfp16"


def test_sampler_prefers_a_same_profile_verifier():
    """Same-profile pairs discriminate ~10x more sharply, so they should be used first."""
    table = _table_without_nf4_cross()
    table.set(
        "nf4/fp16/eager", "nf4/fp16/eager", Tolerance(match_distance=0.0001, mismatch_distance=0.0006)
    )
    table.set(
        "none/fp16/eager", "nf4/fp16/eager",
        Tolerance(match_distance=0.0694, mismatch_distance=0.4162),
    )

    scorer = ReputationScorer(ScorerConfig())
    sampler = VerificationSampler(
        scorer, ClusterIndex(), SamplerConfig(), rng=random.Random(0), tolerances=table
    )
    subject = _profiled("smnf4a", asn=1, quant="nf4")
    candidates = [
        _profiled("smfp16", asn=2, quant="none", first_seen=50_000.0),
        _profiled("smnf4b", asn=3, quant="nf4", first_seen=60_000.0),
    ]

    assert sampler.select_verifier(subject, RANGE, candidates).peer_id == "smnf4b"


def test_plan_carries_the_pair_specific_tolerance():
    from seedmesh.verification.calibrate import ToleranceTable

    table = ToleranceTable()
    wide = Tolerance(match_distance=0.0694, mismatch_distance=0.4162)
    table.set("none/fp16/eager", "nf4/fp16/eager", wide)

    scorer = ReputationScorer(ScorerConfig())
    sampler = VerificationSampler(
        scorer, ClusterIndex(), SamplerConfig(), rng=random.Random(0), tolerances=table
    )
    task = sampler.plan(
        _profiled("smnf4", asn=1, quant="nf4"),
        RANGE,
        [_profiled("smfp16", asn=2, quant="none", first_seen=50_000.0)],
        force=True,
    )
    assert task is not None
    assert task.tolerance == wide


def test_measured_nf4_distance_matches_under_the_pair_tolerance_but_not_the_global_one():
    """The measurement itself, encoded: 0.055 is a MATCH for the NF4 pair, ambiguous globally."""
    from seedmesh.verification.calibrate import ToleranceTable

    measured_nf4_distance = 0.0554
    global_tolerance = Tolerance(match_distance=0.01707, mismatch_distance=0.10241)
    assert global_tolerance.match_distance < measured_nf4_distance < global_tolerance.mismatch_distance

    table = ToleranceTable()
    table.set(
        "none/fp16/eager", "nf4/fp16/eager",
        Tolerance(match_distance=0.0694, mismatch_distance=0.4162),
    )
    pair_tolerance = table.get("none/fp16/eager", "nf4/fp16/eager")
    assert measured_nf4_distance <= pair_tolerance.match_distance


def test_tolerance_table_roundtrips_through_json(tmp_path):
    from seedmesh.verification.calibrate import ToleranceTable

    table = _table_without_nf4_cross()
    path = tmp_path / "tolerances.json"
    table.save(path, notes="spike/quantization, RTX 3050")
    restored = ToleranceTable.load(path)

    assert len(restored) == len(table)
    assert restored.get("none/fp16/eager", "int8/fp16/eager").match_distance == pytest.approx(0.0065)
    assert restored.get("none/fp16/eager", "none/fp16/eager") is not None


def test_plan_attributes_clusters_to_both_parties():
    sampler = _sampler()
    subject = _server("smsubject", asn=1, first_seen=0.0)
    verifier = _server("smverifier", asn=2, first_seen=50_000.0)

    task = sampler.plan(subject, RANGE, [verifier], force=True)
    assert task is not None
    assert task.subject_cluster == "asn:1"
    assert task.verifier_cluster == "asn:2"
    assert len(task.nonce) == 16
