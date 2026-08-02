"""Carrying signed reputation records over a hivemind DHT.

`gossip.py` deliberately takes its transport as two callables so the whole publish/verify/
aggregate cycle is testable without hivemind. This is the one place that binds those
callables to a real DHT, and it lives here rather than on `PetalsBackend` because the client
that most needs to gossip -- `seedmesh chat` -- talks to Petals directly and never builds a
backend.

**The DHT authenticates nothing.** Measured in `spike/hivemind_dht/`: on a default hivemind
DHT any peer can overwrite any key. So the slot a record sits in carries no authority at all;
the Ed25519 signature inside it is the entire basis for trusting a record, and `verify_batch`
runs on everything fetched. Storing to a well-known key is therefore safe in the only sense
that matters -- an attacker who overwrites it can delete evidence, not forge it.

Both functions swallow their exceptions and report absence instead. A DHT round-trip failing
mid-chat should cost the swarm one round of gossip, not the user's session.
"""

from __future__ import annotations

from typing import Any, Optional


def dht_publish(
    dht: Any, key: str, value: dict, expiration_time: float, subkey: Optional[str] = None
) -> bool:
    try:
        if subkey is not None:
            return bool(dht.store(key, subkey=subkey, value=value, expiration_time=expiration_time))
        return bool(dht.store(key, value, expiration_time=expiration_time))
    except Exception:
        return False


def dht_fetch(dht: Any, keys: list[str]) -> dict[str, Optional[dict]]:
    results: dict[str, Optional[dict]] = {}
    for key in keys:
        try:
            found = dht.get(key, latest=False)
        except Exception:
            found = None
        if found is None:
            results[key] = None
            continue
        value = found.value
        # A subkeyed record comes back as {subkey: ValueWithExpiration}; unwrap the inner
        # values so callers see plain dicts either way.
        if isinstance(value, dict):
            value = {
                subkey: (inner.value if hasattr(inner, "value") else inner)
                for subkey, inner in value.items()
            }
        results[key] = value
    return results


def transport_for(dht: Any) -> tuple:
    """The ``(publish_fn, fetch_fn)`` pair `ReputationGossip` expects, bound to ``dht``."""

    def publish(key: str, value: dict, expiration_time: float, subkey: Optional[str] = None) -> bool:
        return dht_publish(dht, key, value, expiration_time, subkey)

    def fetch(keys: list[str]) -> dict[str, Optional[dict]]:
        return dht_fetch(dht, keys)

    return publish, fetch
