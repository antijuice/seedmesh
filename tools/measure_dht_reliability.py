"""How often does a DHT store actually fail, and does retrying in a tight loop help?

Reputation gossip rests on this number, and the number I recorded does not survive its own
consequences. Measured 2026-08-01: 8/12 stores succeeded, so ~1 failure in 3. `publish()`
retries 4 times, which at an independent 33% failure rate should lose a whole round about
1.2% of the time. A later live run lost **2 rounds out of 3**. Two independent 1.2% events
in three trials is not a plausible story, so at least one of these is wrong:

  H1  the failure rate is far worse than 33%
  H2  failures are CORRELATED in time -- if the DHT is in a bad state, all four retries fail
      together, and a tight retry loop buys almost nothing
  H3  repeat stores to the SAME key behave differently from stores to fresh keys. The
      original measurement used a fresh key per attempt (`warm0`..`warm11`); gossip
      republishes to one key forever. This difference was never tested.
  H4  payload size matters -- the original probe stored `{"i": i}`, and a signed observation
      batch is far larger.

Each condition below isolates one of those. The last section asks a different question that
the return value cannot answer on its own: when `store` returns False, did the value reach
the network anyway? A local `get` is no use -- hivemind serves a key this node stored from
its own cache -- so propagation is checked from a SECOND, independent client.

Run:  python tools/measure_dht_reliability.py [--trials 20]
"""

from __future__ import annotations

import argparse
import time

from hivemind.dht import DHT

from seedmesh.cli.swarm import resolve_peers

PREFIX = "seedmesh/reliability-probe"


def big_value(n_reports: int = 24) -> dict:
    """Roughly the shape and size of a signed ObservationBatch."""
    return {
        "observer": "1" * 52,
        "epoch": 1,
        "issued_at": 0.0,
        "expires_at": 0.0,
        "reports": [
            {"subject": f"{i:052d}", "ok": 10, "failed": 0, "mismatches": 0,
             "p50_ms": 12.5, "p95_ms": 40.0}
            for i in range(n_reports)
        ],
        "key": "A" * 44,
        "sig": "B" * 88,
    }


def run_condition(dht, name: str, trials: int, *, same_key: bool, gap: float, value) -> dict:
    key_base = f"{PREFIX}/{name}"
    results = []
    for i in range(trials):
        key = key_base if same_key else f"{key_base}/{i}"
        ok = False
        try:
            ok = bool(dht.store(key, value, expiration_time=time.time() + 600))
        except Exception:
            ok = False
        results.append(ok)
        if gap:
            time.sleep(gap)

    successes = sum(results)
    # The number that actually matters for `publish(attempts=4)`: how often do four
    # consecutive attempts ALL fail? Independence predicts p^4; correlation does not.
    quads = [results[i:i + 4] for i in range(0, len(results) - 3)]
    all_fail = sum(1 for q in quads if not any(q))
    rate = 1 - successes / len(results)
    return {
        "name": name,
        "trials": len(results),
        "failure_rate": rate,
        "predicted_quad_failure": rate ** 4,
        "observed_quad_failure": all_fail / len(quads) if quads else 0.0,
        "pattern": "".join("." if r else "X" for r in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    peers, note = resolve_peers(None, None)
    print(note or f"{len(peers)} bootstrap peers")
    dht = DHT(initial_peers=peers, client_mode=True, start=True)
    time.sleep(5)  # let the routing table populate; asking immediately understates reach

    small = {"probe": 1}
    large = big_value()
    print(f"payload sizes: small ~{len(str(small))}B, large ~{len(str(large))}B\n")

    conditions = [
        ("fresh-key-spaced", dict(same_key=False, gap=2.0, value=small)),
        ("same-key-spaced", dict(same_key=True, gap=2.0, value=small)),
        ("same-key-burst", dict(same_key=True, gap=0.0, value=small)),
        ("fresh-key-burst", dict(same_key=False, gap=0.0, value=small)),
        ("same-key-large", dict(same_key=True, gap=2.0, value=large)),
    ]

    print(f"{'condition':<20} {'fail rate':>10} {'4x pred':>9} {'4x obs':>8}  pattern")
    print("-" * 78)
    rows = []
    for name, kwargs in conditions:
        row = run_condition(dht, name, args.trials, **kwargs)
        rows.append(row)
        print(f"{row['name']:<20} {row['failure_rate']:>9.0%} "
              f"{row['predicted_quad_failure']:>9.1%} {row['observed_quad_failure']:>8.1%}  "
              f"{row['pattern']}")

    # ---- does a False return mean the value is really gone? ------------------
    print("\n=== when store returns False, did the value propagate anyway? ===")
    print("(checked from a SECOND client -- a local get is served from this node's own")
    print(" cache and would say yes either way)")

    marker = f"{PREFIX}/propagation"
    verdicts = []
    for i in range(6):
        key = f"{marker}/{i}"
        payload = {"i": i}
        returned = False
        try:
            returned = bool(dht.store(key, payload, expiration_time=time.time() + 600))
        except Exception:
            returned = False
        verdicts.append((key, payload, returned))
        time.sleep(1)

    reader = DHT(initial_peers=peers, client_mode=True, start=True)
    time.sleep(5)
    print(f"\n{'key':<12} {'store said':>11} {'2nd client sees':>16}")
    disagreements = 0
    for key, payload, returned in verdicts:
        try:
            found = reader.get(key, latest=True)
        except Exception:
            found = None
        visible = found is not None and getattr(found, "value", None) == payload
        if returned != visible:
            disagreements += 1
        print(f"{key.rsplit('/', 1)[-1]:<12} {str(returned):>11} {str(visible):>16}")
    reader.shutdown()

    print("\n=== reading this ===")
    print("  4x obs >> 4x pred      -> failures are CORRELATED; a tight retry loop is")
    print("                            near-useless and retries need spacing")
    print("  same-key >> fresh-key  -> republishing to one key is the problem, not the DHT")
    print("  large >> small         -> payload size; cap reports per batch")
    if disagreements:
        print(f"  {disagreements}/6 store returns disagreed with what a second client could see")
        print("     -> the boolean is not a reliable signal on its own")
    else:
        print("  store's return value agreed with propagation in every sample")
    dht.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
