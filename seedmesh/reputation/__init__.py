"""Reputation: measure peers first-hand, share it signed, aggregate it defensively.

The layering is deliberate and the boundaries carry the trust assumptions:

``scorer``      first-hand measurement only -- everything here was witnessed directly.
``records``     signed, replay-resistant transport for sharing that belief.
``aggregate``   merging strangers' claims under explicit anti-sybil rules.
``diversity``   network-locality clustering, the scarce resource the rules price in.
``routing_bias`` turning scores into an actual pipeline, exactly and deterministically.
"""

from seedmesh.reputation.aggregate import (
    AggregateScore,
    AggregationConfig,
    ReputationAggregator,
    blend,
    weighted_median,
)
from seedmesh.reputation.diversity import (
    ClusterIndex,
    NullAsnResolver,
    StaticAsnResolver,
    UNKNOWN_CLUSTER,
)
from seedmesh.reputation.records import (
    EpochStore,
    ObservationBatch,
    PeerReport,
    RecordError,
    build_batch,
    verify_batch,
)
from seedmesh.reputation.routing_bias import Hop, Route, Router, RoutingConfig
from seedmesh.reputation.scorer import ReputationScorer, ScoreBreakdown, ScorerConfig

__all__ = [
    "AggregateScore",
    "AggregationConfig",
    "ClusterIndex",
    "EpochStore",
    "Hop",
    "NullAsnResolver",
    "ObservationBatch",
    "PeerReport",
    "RecordError",
    "ReputationAggregator",
    "ReputationScorer",
    "Route",
    "Router",
    "RoutingConfig",
    "ScoreBreakdown",
    "ScorerConfig",
    "StaticAsnResolver",
    "UNKNOWN_CLUSTER",
    "blend",
    "build_batch",
    "verify_batch",
    "weighted_median",
]
