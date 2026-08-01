"""Run a real hivemind DHT and find out whether Seedmesh's records can live in it.

Two spikes so far each found that a design assumption did not survive contact with the
real component. This one asks the equivalent question for the DHT:

**Can a signed observation batch actually be published?** The reputation design assumes
records up to 512 reports, one batch per observer, TTL-expiring, with newest-wins. Every
one of those assumptions is a claim about the DHT's storage model, and none of them has
been checked against an actual DHT.

Also checked, because it decides how much of the trust layer is load-bearing: **can one
peer overwrite another peer's record?** If yes, DHT-layer storage offers no protection and
Seedmesh's Ed25519 attribution is doing all the work (which is what it was designed for).
If no, that is defence in depth worth knowing about.

Run inside WSL:
    ~/hmspike/bin/python3 spike/hivemind_dht/test_dht.py
"""

from __future__ import annotations

import time

import hivemind
from hivemind.utils import get_dht_time

from seedmesh.core.clock import SystemClock
from seedmesh.core.identity import Identity
from seedmesh.reputation.records import PeerReport, build_batch, verify_batch

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    print(f"  -> {'PASS' if passed else 'FAIL'}  {detail}")


def make_batch(identity: Identity, n_reports: int, epoch: int = 1) -> dict:
    reports = [
        PeerReport(
            subject=f"smsubject{i:06d}",
            ok=float(i % 40),
            failed=float(i % 7),
            mismatches=0.0,
            p50_ms=40.0 + i,
            p95_ms=120.0 + i,
        )
        for i in range(n_reports)
    ]
    return build_batch(identity, reports, epoch=epoch, clock=SystemClock(), ttl_s=900.0)


def main() -> int:
    print(f"hivemind {hivemind.__version__}\n")

    print("starting 3-node DHT (1 bootstrap + 2 peers)...")
    bootstrap = hivemind.DHT(start=True)
    peers = bootstrap.get_visible_maddrs()
    alice = hivemind.DHT(initial_peers=peers, start=True)
    bob = hivemind.DHT(initial_peers=peers, start=True)
    time.sleep(2)
    print(f"  bootstrap peer id: {bootstrap.peer_id}")
    print(f"  alice: {alice.peer_id}")
    print(f"  bob:   {bob.peer_id}\n")

    try:
        # -- 1. basic store/get across nodes --------------------------------------
        print("1. store on alice, read on bob")
        alice.store("seedmesh/smoke", "hello", get_dht_time() + 60)
        got = bob.get("seedmesh/smoke", latest=True)
        record("basic store/get", got is not None and got.value == "hello",
               f"value={got.value if got else None}")

        # -- 2. subkey semantics ---------------------------------------------------
        print("\n2. two observers write subkeys under one key (the reputation pattern)")
        key = "seedmesh/reputation/smtarget"
        alice.store(key, subkey="observer-alice", value={"reliability": 0.2},
                    expiration_time=get_dht_time() + 60)
        bob.store(key, subkey="observer-bob", value={"reliability": 0.9},
                  expiration_time=get_dht_time() + 60)
        time.sleep(1)
        got = bootstrap.get(key, latest=True)
        subkeys = set(got.value.keys()) if got else set()
        record("independent subkeys coexist",
               subkeys == {"observer-alice", "observer-bob"},
               f"subkeys={sorted(subkeys)}")

        # -- 3. can one peer overwrite another's subkey? ---------------------------
        print("\n3. can bob overwrite alice's subkey? (DHT-layer impersonation)")
        bob.store(key, subkey="observer-alice", value={"reliability": 1.0},
                  expiration_time=get_dht_time() + 120)
        time.sleep(1)
        got = bootstrap.get(key, latest=True)
        alice_value = got.value.get("observer-alice").value if got else None
        overwritten = alice_value == {"reliability": 1.0}
        record("bob CAN overwrite alice's subkey" if overwritten
               else "DHT rejected the overwrite",
               True,
               f"alice's slot now = {alice_value} "
               f"({'no DHT-layer protection' if overwritten else 'DHT protects slots'})")

        # -- 4. TTL expiry ---------------------------------------------------------
        print("\n4. TTL expiry (churn handling comes free only if this works)")
        alice.store("seedmesh/ttl", "transient", get_dht_time() + 3)
        immediately = bob.get("seedmesh/ttl", latest=True)
        time.sleep(6)
        after = bob.get("seedmesh/ttl", latest=True)
        record("value expires after its TTL",
               immediately is not None and after is None,
               f"before={immediately.value if immediately else None}, after={after}")

        # -- 5. signed Seedmesh batch roundtrip ------------------------------------
        print("\n5. publish a real signed ObservationBatch and verify it after retrieval")
        identity = Identity.generate()
        batch = make_batch(identity, 10)
        alice.store("seedmesh/batch/small", subkey=identity.peer_id, value=batch,
                    expiration_time=get_dht_time() + 60)
        time.sleep(1)
        got = bob.get("seedmesh/batch/small", latest=True)
        retrieved = got.value[identity.peer_id].value if got else None
        try:
            verified = verify_batch(retrieved, clock=SystemClock())
            ok = verified.observer == identity.peer_id and len(verified.reports) == 10
            detail = f"observer matches, {len(verified.reports)} reports, signature valid"
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        record("signed batch survives DHT roundtrip", ok, detail)

        # -- 6. how big a batch actually fits? -------------------------------------
        print("\n6. maximum publishable batch size (design assumes up to 512 reports)")
        import json

        print(f"  {'reports':>8} {'bytes':>9}  result")
        largest_ok = 0
        for n_reports in (1, 10, 50, 100, 256, 512):
            candidate = make_batch(identity, n_reports, epoch=n_reports + 1)
            size = len(json.dumps(candidate).encode())
            store_key = f"seedmesh/batch/size{n_reports}"
            try:
                alice.store(store_key, subkey=identity.peer_id, value=candidate,
                            expiration_time=get_dht_time() + 60)
                time.sleep(0.6)
                fetched = bob.get(store_key, latest=True)
                got_back = fetched.value[identity.peer_id].value if fetched else None
                roundtripped = got_back is not None and len(got_back["reports"]) == n_reports
                if roundtripped:
                    largest_ok = n_reports
                print(f"  {n_reports:>8} {size:>9}  {'ok' if roundtripped else 'FAILED to roundtrip'}")
            except Exception as exc:
                print(f"  {n_reports:>8} {size:>9}  raised {type(exc).__name__}: {exc}")
        record("512-report batches are publishable", largest_ok >= 512,
               f"largest batch that roundtripped: {largest_ok} reports")

    finally:
        print("\nshutting down DHT nodes...")
        for node in (alice, bob, bootstrap):
            try:
                node.shutdown()
            except Exception:
                pass

    print("\n" + "=" * 62)
    failures = sum(1 for _, passed, _ in RESULTS if not passed)
    for name, passed, detail in RESULTS:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"{len(RESULTS) - failures}/{len(RESULTS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
