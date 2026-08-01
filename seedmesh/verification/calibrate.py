"""Deriving comparison thresholds from measurement rather than intuition.

A tolerance threshold is a claim about how far apart two *honest* servers can land. Nobody
can guess that number: it depends on the dtype mix, the hardware mix, the model and the
sequence length. Guessing high lets cheats through; guessing low evicts honest volunteers.

So it is measured. Collect relative distances from pairs known to be honest -- during
bring-up, servers you run yourself -- and set the match threshold at a high quantile of
that distribution. Choosing the 99.9th percentile is choosing a false-positive rate: about
one honest pair in a thousand will fall outside, and since a MATCH failure only demotes to
INCONCLUSIVE rather than MISMATCH, that has no punitive effect at all.

The gap between the match and mismatch thresholds is the inconclusive band. It should be
wide when the honest distribution has a heavy tail, which is exactly what
:meth:`HonestDistanceSample.tail_ratio` reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from seedmesh.verification.compare import Tolerance


@dataclass(frozen=True, slots=True)
class HonestDistanceSample:
    """Observed distances between pairs of servers believed to be honest."""

    relative: tuple[float, ...]
    cosine: tuple[float, ...]
    norm: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.relative)

    def tail_ratio(self) -> float:
        """p99.9 / p50 of the relative distances. High values mean a heavy tail."""
        if len(self.relative) < 2:
            return 1.0
        values = np.asarray(self.relative, dtype=np.float64)
        median = float(np.quantile(values, 0.5))
        extreme = float(np.quantile(values, 0.999))
        return extreme / median if median > 0 else float("inf")


class CalibrationCollector:
    """Accumulates honest-pair distances during bring-up."""

    def __init__(self) -> None:
        self._relative: list[float] = []
        self._cosine: list[float] = []
        self._norm: list[float] = []

    def add(self, relative: float, cosine: float, norm_deviation: float) -> None:
        self._relative.append(float(relative))
        self._cosine.append(float(cosine))
        self._norm.append(float(norm_deviation))

    def extend(self, rows: Iterable[tuple[float, float, float]]) -> None:
        for relative, cosine, norm_deviation in rows:
            self.add(relative, cosine, norm_deviation)

    def sample(self) -> HonestDistanceSample:
        return HonestDistanceSample(
            relative=tuple(self._relative),
            cosine=tuple(self._cosine),
            norm=tuple(self._norm),
        )

    def __len__(self) -> int:
        return len(self._relative)


MIN_CALIBRATION_SAMPLES = 30


def calibrate(
    sample: HonestDistanceSample,
    *,
    target_false_positive_rate: float = 0.001,
    mismatch_multiplier: float = 6.0,
    floor: float = 1e-4,
) -> Tolerance:
    """Derive a :class:`Tolerance` from measured honest-pair distances.

    ``mismatch_multiplier`` sets how far beyond the honest tail a distance must fall before
    it is charged as fraud. The default of 6x is intentionally conservative: the asymmetry
    described in :mod:`seedmesh.verification.compare` means a false accusation costs more
    than a missed detection, which sampling will get another chance at.
    """
    if len(sample) < MIN_CALIBRATION_SAMPLES:
        raise ValueError(
            f"need at least {MIN_CALIBRATION_SAMPLES} honest pairs to calibrate, "
            f"got {len(sample)}; thresholds from a smaller sample are guesses with extra steps"
        )
    if not 0.0 < target_false_positive_rate < 0.5:
        raise ValueError("target_false_positive_rate must be in (0, 0.5)")
    if mismatch_multiplier <= 1.0:
        raise ValueError("mismatch_multiplier must exceed 1.0 to leave an inconclusive band")

    quantile = 1.0 - target_false_positive_rate
    relative = max(float(np.quantile(np.asarray(sample.relative), quantile)), floor)
    cosine = max(float(np.quantile(np.asarray(sample.cosine), quantile)), floor)
    norm = max(float(np.quantile(np.asarray(sample.norm), quantile)), floor)

    return Tolerance(
        match_distance=relative,
        mismatch_distance=relative * mismatch_multiplier,
        cosine_distance=cosine,
        norm_deviation=norm,
    )


def save_tolerance(tolerance: Tolerance, path: Path, *, notes: Optional[str] = None) -> None:
    """Persist a calibration with provenance.

    Provenance is not decoration: a threshold measured on one hardware mix is not valid on
    another, and a bare number in a config file loses the context needed to know that.
    """
    payload = {
        "match_distance": tolerance.match_distance,
        "mismatch_distance": tolerance.mismatch_distance,
        "cosine_distance": tolerance.cosine_distance,
        "norm_deviation": tolerance.norm_deviation,
        "notes": notes or "",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_tolerance(path: Path) -> Tolerance:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Tolerance(
        match_distance=float(data["match_distance"]),
        mismatch_distance=float(data["mismatch_distance"]),
        cosine_distance=float(data["cosine_distance"]),
        norm_deviation=float(data["norm_deviation"]),
    )


def empirical_false_positive_rate(
    distances: Sequence[float], tolerance: Tolerance
) -> float:
    """Fraction of honest distances that would fail the MATCH test. For validating a fit."""
    if not distances:
        return 0.0
    values = np.asarray(distances, dtype=np.float64)
    return float(np.mean(values > tolerance.match_distance))


class ToleranceTable:
    """Per-compute-profile-pair tolerances.

    A single global tolerance is wrong, and measurably so. Honest servers holding identical
    weights disagree by ~0.005 across fp16/bf16/int8 but by **~0.055 when one is NF4**
    (``spike/quantization/``). Judged against one threshold fitted to the former, every NF4
    server lands in the ambiguous band on every check and is quarantined after six.

    So tolerance is looked up by the *pair* of profiles being compared, and the lookup is
    unordered -- comparing A against B must give the same verdict as B against A, or the
    result would depend on which server the client happened to call the subject.

    **An uncalibrated pair returns ``None``, meaning "do not verify this pair."** That is
    deliberate and matches how the sampler already handles the absence of an independent
    verifier: a check run against the wrong tolerance is worse than no check, because it
    manufactures evidence -- either false guilt or false confidence -- that the reputation
    layer will then act on.
    """

    def __init__(self, tolerances: Optional[dict[frozenset[str], Tolerance]] = None) -> None:
        self._tolerances: dict[frozenset[str], Tolerance] = dict(tolerances or {})

    @staticmethod
    def _key(left: str, right: str) -> frozenset[str]:
        # A frozenset of one element when the profiles match, which is correct: comparing a
        # profile against itself is a single calibration case.
        return frozenset((left, right))

    def set(self, left: str, right: str, tolerance: Tolerance) -> None:
        self._tolerances[self._key(left, right)] = tolerance

    def get(self, left: str, right: str) -> Optional[Tolerance]:
        """Tolerance for this profile pair, or ``None`` if it has never been calibrated."""
        return self._tolerances.get(self._key(left, right))

    def can_compare(self, left: str, right: str) -> bool:
        return self._key(left, right) in self._tolerances

    def pairs(self) -> list[tuple[str, ...]]:
        return [tuple(sorted(key)) for key in self._tolerances]

    def __len__(self) -> int:
        return len(self._tolerances)

    def __contains__(self, key: object) -> bool:
        return key in self._tolerances

    # ---- persistence -----------------------------------------------------------

    def to_dict(self) -> dict:
        out: dict[str, dict] = {}
        for key, tolerance in self._tolerances.items():
            parts = sorted(key)
            name = "|".join(parts if len(parts) == 2 else parts * 2)
            out[name] = {
                "match_distance": tolerance.match_distance,
                "mismatch_distance": tolerance.mismatch_distance,
                "cosine_distance": tolerance.cosine_distance,
                "norm_deviation": tolerance.norm_deviation,
            }
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "ToleranceTable":
        table = cls()
        for name, values in data.items():
            left, _, right = name.partition("|")
            table.set(
                left,
                right or left,
                Tolerance(
                    match_distance=float(values["match_distance"]),
                    mismatch_distance=float(values["mismatch_distance"]),
                    cosine_distance=float(values["cosine_distance"]),
                    norm_deviation=float(values["norm_deviation"]),
                ),
            )
        return table

    def save(self, path: Path, *, notes: Optional[str] = None) -> None:
        payload = {"version": 1, "notes": notes or "", "tolerances": self.to_dict()}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ToleranceTable":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data["tolerances"])
