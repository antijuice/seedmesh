"""Deciding whether two servers did the same work, under floating-point noise.

Comparison returns one of three verdicts, and the third one is the important one:

``MATCH``         the two sketches agree within calibrated tolerance.
``MISMATCH``      they disagree by more than any plausible numerical noise.
``INCONCLUSIVE``  the distance landed in the band between those thresholds.

Most naive designs are binary. A binary decision on a noisy measurement means every
borderline case is charged as fraud, and on a volunteer network the cost of that error is
asymmetric: wrongly evicting an honest volunteer loses capacity and goodwill permanently,
while missing one cheat costs one request that redundant sampling will catch again later.
So the ambiguous band records nothing, and :meth:`ReputationScorer.record_verification`
treats ``INCONCLUSIVE`` as no evidence rather than weak guilt.

The thresholds are not guessed. :mod:`seedmesh.verification.calibrate` derives them from
observed honest-pair distances against a target false-positive rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from seedmesh.core.types import Verdict
from seedmesh.verification.sketch import Sketch

EPS = 1e-12


@dataclass(frozen=True, slots=True)
class Tolerance:
    """Calibrated decision boundaries for sketch comparison.

    Defaults are deliberately loose. They are placeholders sized for mixed fp16/bf16
    hardware and must be replaced by :func:`seedmesh.verification.calibrate.calibrate`
    output measured on the real swarm before the thresholds mean anything.
    """

    match_distance: float = 0.02
    mismatch_distance: float = 0.15
    cosine_distance: float = 0.01
    norm_deviation: float = 0.05

    def __post_init__(self) -> None:
        if self.mismatch_distance <= self.match_distance:
            raise ValueError(
                "mismatch_distance must exceed match_distance; the gap is the "
                "inconclusive band that keeps honest nodes from being charged as cheats"
            )


@dataclass(frozen=True, slots=True)
class Comparison:
    """Full result of comparing two sketches, including the evidence."""

    verdict: Verdict
    relative_distance: float
    cosine_distance: float
    norm_deviation: float
    reason: str
    comparable: bool = True
    """Whether the two sketches could be meaningfully compared at all.

    Distinguishes the two very different things an ``INCONCLUSIVE`` verdict can mean. A
    distance in the ambiguous band is *weak evidence about the peer* and accumulates. A
    structural failure -- mismatched dimensions, mismatched component counts -- is evidence
    about the routing or the model catalog and must never be held against anyone."""

    def as_detail(self) -> dict:
        return {
            "relative_distance": self.relative_distance,
            "cosine_distance": self.cosine_distance,
            "norm_deviation": self.norm_deviation,
            "reason": self.reason,
        }


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance normalized by magnitude, so the metric is scale-free."""
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), EPS)
    return float(np.linalg.norm(a - b) / scale)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity. Catches direction changes that magnitude alone would hide."""
    denominator = max(float(np.linalg.norm(a)) * float(np.linalg.norm(b)), EPS)
    similarity = float(np.dot(a, b) / denominator)
    return float(1.0 - max(-1.0, min(1.0, similarity)))


def compare_sketches(
    left: Sketch,
    right: Sketch,
    tolerance: Optional[Tolerance] = None,
) -> Comparison:
    """Compare two sketches of what should be the same computation."""
    tolerance = tolerance or Tolerance()

    if left.components != right.components:
        return Comparison(
            Verdict.INCONCLUSIVE, float("inf"), float("inf"), float("inf"),
            "sketch component counts differ; cannot compare", comparable=False,
        )
    if left.dim != right.dim:
        # Different hidden dimension means the peers are not running the same model at all.
        # That is a routing or catalog error, not proof of cheating.
        return Comparison(
            Verdict.INCONCLUSIVE, float("inf"), float("inf"), float("inf"),
            f"hidden dimensions differ ({left.dim} vs {right.dim})", comparable=False,
        )

    distance = relative_l2(left.values, right.values)
    cosine = cosine_distance(left.values, right.values)
    norm_scale = max(left.norm, right.norm, EPS)
    norm_dev = abs(left.norm - right.norm) / norm_scale

    # MATCH is conservative: every metric must agree. Cosine and norm add discrimination
    # that relative distance alone can miss at the margins.
    if (
        distance <= tolerance.match_distance
        and cosine <= tolerance.cosine_distance
        and norm_dev <= tolerance.norm_deviation
    ):
        return Comparison(Verdict.MATCH, distance, cosine, norm_dev, "within tolerance")

    # MISMATCH rests on relative distance alone. Earlier revisions also convicted on
    # multiples of the cosine and norm thresholds, which let a sample sitting inside the
    # inconclusive band be charged as fraud anyway -- the band stopped meaning what its
    # name says. Nothing is lost: relative L2 is near zero only when two vectors agree in
    # both direction and magnitude, so it already subsumes what those triggers detected.
    if distance >= tolerance.mismatch_distance:
        return Comparison(
            Verdict.MISMATCH, distance, cosine, norm_dev,
            f"relative distance {distance:.4f} exceeds mismatch bound "
            f"{tolerance.mismatch_distance:.4f}",
        )

    return Comparison(
        Verdict.INCONCLUSIVE, distance, cosine, norm_dev,
        f"distance {distance:.4f} falls in the ambiguous band "
        f"[{tolerance.match_distance}, {tolerance.mismatch_distance})",
    )


def detect_passthrough(
    input_sketch: Sketch,
    output_sketch: Sketch,
    *,
    min_change: float = 0.05,
) -> bool:
    """Whether a server appears to have returned its input essentially unchanged.

    This catches the cheapest possible cheat, and one that redundant sampling alone would
    miss: a "lazy" server that skips the forward pass and echoes the activations it
    received. It costs no GPU time and, without this check, looks like a fast successful
    request. Real transformer blocks always move the hidden state appreciably.

    Applied only where it is valid: over a block range large enough that a near-identity
    mapping is genuinely implausible. Callers must not run it on a single-block range,
    where residual connections can legitimately leave the state nearly unchanged.
    """
    if input_sketch.components != output_sketch.components:
        return False
    return relative_l2(input_sketch.values, output_sketch.values) < min_change
