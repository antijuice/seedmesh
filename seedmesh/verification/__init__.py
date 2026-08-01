"""Verification: cheap, probabilistic detection of servers returning bad work.

A pragmatic stand-in for proof-of-computation, as the spec intends -- but built around two
corrections to the spec's sketch of it:

* hidden states are compared by *tolerance*, never by hash, because honest servers on
  different GPUs never produce bit-identical floats (:mod:`~seedmesh.verification.sketch`);
* thresholds are *calibrated* from measured honest-pair distances rather than guessed
  (:mod:`~seedmesh.verification.calibrate`).
"""

from seedmesh.verification.calibrate import (
    CalibrationCollector,
    HonestDistanceSample,
    ToleranceTable,
    calibrate,
    empirical_false_positive_rate,
    load_tolerance,
    save_tolerance,
)
from seedmesh.verification.compare import (
    Comparison,
    Tolerance,
    compare_sketches,
    cosine_distance,
    detect_passthrough,
    relative_l2,
)
from seedmesh.verification.sampler import (
    PairHistory,
    SamplerConfig,
    VerificationSampler,
    VerificationTask,
)
from seedmesh.verification.sketch import (
    Sketch,
    SketchSpec,
    derive_seed,
    sketch_hidden_state,
)

__all__ = [
    "CalibrationCollector",
    "Comparison",
    "HonestDistanceSample",
    "PairHistory",
    "SamplerConfig",
    "Sketch",
    "SketchSpec",
    "Tolerance",
    "ToleranceTable",
    "VerificationSampler",
    "VerificationTask",
    "calibrate",
    "compare_sketches",
    "cosine_distance",
    "derive_seed",
    "detect_passthrough",
    "empirical_false_positive_rate",
    "load_tolerance",
    "relative_l2",
    "save_tolerance",
    "sketch_hidden_state",
]
