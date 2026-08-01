"""Would hivemind's own record validator have stopped the overwrite?

``test_dht.py`` showed that on a default DHT, any peer can overwrite any other peer's
subkey. hivemind ships ``RSASignatureValidator`` for exactly this. The question is whether
enabling it makes Seedmesh's Ed25519 record signing redundant, or whether the two protect
different things.

The answer decides a real design point, so it is tested rather than reasoned about.

Run inside WSL:
    ~/hmspike/bin/python3 spike/hivemind_dht/test_validator.py
"""

from __future__ import annotations

import time

import hivemind
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.utils import get_dht_time
from hivemind.utils.crypto import RSAPrivateKey

RESULTS: list[tuple[str, str]] = []


def main() -> int:
    print(f"hivemind {hivemind.__version__}\n")

    # Each validator MUST get its own key. The default is RSAPrivateKey.process_wide(),
    # which is correct in production -- every peer is a separate process -- but in a
    # single-process test it hands all three nodes the same identity, so "Bob" would
    # cryptographically *be* Alice and every overwrite would look legitimate.
    alice_validator = RSASignatureValidator(RSAPrivateKey())
    bob_validator = RSASignatureValidator(RSAPrivateKey())
    assert alice_validator.local_public_key != bob_validator.local_public_key

    print("starting 3-node DHT with RSASignatureValidator enabled...")
    bootstrap = hivemind.DHT(start=True, record_validators=[RSASignatureValidator(RSAPrivateKey())])
    peers = bootstrap.get_visible_maddrs()
    alice = hivemind.DHT(initial_peers=peers, start=True, record_validators=[alice_validator])
    bob = hivemind.DHT(initial_peers=peers, start=True, record_validators=[bob_validator])
    time.sleep(2)

    # hivemind binds ownership by requiring the subkey to carry the publisher's public key.
    alice_subkey = b"observer-alice" + alice_validator.local_public_key
    print(f"  alice's protected subkey ends with her public key ({len(alice_validator.local_public_key)} bytes)\n")

    try:
        key = "seedmesh/reputation/smtarget"

        print("1. alice writes her own protected subkey")
        stored = alice.store(key, subkey=alice_subkey, value={"reliability": 0.2},
                             expiration_time=get_dht_time() + 120)
        time.sleep(1)
        got = bootstrap.get(key, latest=True)
        present = got is not None and alice_subkey in got.value
        print(f"   store returned {stored}, readable={present}")
        RESULTS.append(("alice can write her own subkey", "PASS" if stored and present else "FAIL"))

        print("\n2. bob tries to overwrite alice's protected subkey")
        hijacked = bob.store(key, subkey=alice_subkey, value={"reliability": 1.0},
                             expiration_time=get_dht_time() + 240)
        time.sleep(1)
        got = bootstrap.get(key, latest=True)
        current = got.value[alice_subkey].value if got and alice_subkey in got.value else None
        blocked = current == {"reliability": 0.2}
        print(f"   store returned {hijacked}, alice's slot now = {current}")
        RESULTS.append((
            "bob is blocked from overwriting alice",
            "PASS (validator blocked it)" if blocked else "FAIL (overwrite succeeded)",
        ))

        print("\n3. does an unprotected subkey (no public key suffix) still get hijacked?")
        plain_key = "seedmesh/reputation/plain"
        alice.store(plain_key, subkey="observer-alice", value={"reliability": 0.2},
                    expiration_time=get_dht_time() + 120)
        time.sleep(0.5)
        bob.store(plain_key, subkey="observer-alice", value={"reliability": 1.0},
                  expiration_time=get_dht_time() + 240)
        time.sleep(1)
        got = bootstrap.get(plain_key, latest=True)
        plain_current = got.value["observer-alice"].value if got and "observer-alice" in got.value else None
        print(f"   alice's unprotected slot now = {plain_current}")
        RESULTS.append((
            "unprotected subkeys remain hijackable",
            "CONFIRMED" if plain_current == {"reliability": 1.0} else "unexpectedly protected",
        ))

    finally:
        print("\nshutting down...")
        for node in (alice, bob, bootstrap):
            try:
                node.shutdown()
            except Exception:
                pass

    print("\n" + "=" * 62)
    for name, verdict in RESULTS:
        print(f"  {verdict:<28} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
