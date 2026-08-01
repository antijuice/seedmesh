# Contributing

## Setup

```bash
pip install -e ".[dev]"
pytest
seedmesh simulate
```

Python ≥3.10. No GPU needed, no torch, works on Windows — the trust layer deliberately
depends on neither.

## What's most useful right now

1. **Break the trust layer.** Add a scenario to `seedmesh/sim/` that defeats a defence. A
   scenario that fails is worth more than a feature — every defence currently in the code
   exists because simulation exposed something broken.
2. **Backend adapters.** `seedmesh/backends/base.py` is three methods. A llama.cpp RPC
   adapter would reach hardware Petals never could (Apple Silicon, AMD, plain CPU), which
   matters more for a volunteer network than PyTorch fidelity does.
3. **Calibration on real hardware.** Every threshold is currently fitted to *simulated*
   floating-point noise. Honest-pair distances measured on real mixed GPUs would replace
   guesses with data.

## Testing conventions

**Test the property, not the implementation.** Name what would break in the real world:

```python
def test_a_sybil_fleet_in_one_cluster_cannot_convict_an_honest_peer():
def test_refused_is_neither_rewarded_nor_punished():
```

**Time is injected.** Use `ManualClock`; never `time.time()`. Reputation is a function of
elapsed time, so a test that cannot control time can only assert trivia.

**Randomness is seeded.** Pass explicit `random.Random(seed)` and `np.random.default_rng(seed)`.
Simulation results must be reproducible or they are anecdotes.

**Regressions get named.** When you fix a defect, the test says which one:

```python
def test_a_single_dispute_does_not_quarantine_either_party():
    """Regression: charging both sides of a mismatch at full weight evicted the honest
    peer that caught the cheat, which makes verifying actively dangerous."""
```

## Changing trust parameters

Defaults in `ScorerConfig`, `AggregationConfig`, `RoutingConfig` and `SamplerConfig` decide
who gets excluded from the network. Treat them as governance (see
[GOVERNANCE.md](GOVERNANCE.md)), which means:

* **Sweep, don't guess.** Run the change across several seeds and all three presets, and put
  the table in the docstring. Existing examples: `RoutingConfig.load_weight` and
  `ScorerConfig.ambiguous_rate_threshold` both carry the measurements that chose them.
* **Report the trade.** `load_weight=4.0` buys full detection and costs 39% latency. Both
  halves belong in the docstring; a defence with no stated cost is a defence nobody can
  evaluate.
* **Check false positives every time.** Detection rate alone is not a result. Wrongly
  excluding an honest volunteer is the expensive error — capacity and goodwill do not come
  back.

## Comments

Explain **why**, especially where the obvious implementation is wrong. The code is full of
places where the natural approach fails subtly:

```python
# Keying by cluster rather than by peer id is not a detail -- it is what makes the
# corroboration rule survive a sybil fleet.
```

Don't narrate what the code plainly does.

## Pull requests

* One concern per PR.
* `pytest` green, and say whether `seedmesh simulate` changed.
* If you found a defect, say what it was and how you found it. That is the interesting part.

## Reporting vulnerabilities

Don't open a public issue for something exploitable. There is no swarm to attack yet, so
this is currently theoretical — but the habit should start before it isn't.
