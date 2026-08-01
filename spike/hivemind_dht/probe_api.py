"""Inventory the hivemind API surface Petals depends on, against whatever is installed.

Petals pins hivemind at a git SHA from roughly the 1.1.10 era (Aug 2023); the current
release is 1.1.12 (Jan 2026). Unlike bitsandbytes, this is not a narrow surface: Petals
reaches into ``hivemind.moe`` (repurposing the mixture-of-experts machinery for block
hosting), ``hivemind.p2p.p2p_daemon_bindings.control`` internals, and mutates a
module-level global in ``hivemind.compression.base``.

That mix is the interesting part. Public API imports are cheap to keep working; the
``moe.*`` and ``p2p_daemon_bindings.*`` reaches are the ones that historically move.
"""

from __future__ import annotations

import importlib
import inspect
import sys

# Every hivemind symbol Petals imports, grouped by how exposed it is.
PUBLIC = [
    ("hivemind", "DHT"),
    ("hivemind", "PeerID"),
    ("hivemind", "MSGPackSerializer"),
    ("hivemind", "TensorDescriptor"),
    ("hivemind", "BatchTensorDescriptor"),
    ("hivemind", "serialize_torch_tensor"),
    ("hivemind", "deserialize_torch_tensor"),
    ("hivemind", "nested_flatten"),
    ("hivemind", "nested_pack"),
    ("hivemind", "nested_compare"),
    ("hivemind", "get_logger"),
    ("hivemind", "anext"),
    ("hivemind.dht", "DHT"),
    ("hivemind.dht", "DHTNode"),
    ("hivemind.dht", "DHTValue"),
    ("hivemind.p2p", "P2P"),
    ("hivemind.p2p", "PeerID"),
    ("hivemind.p2p", "StubBase"),
    ("hivemind.proto", "runtime_pb2"),
    ("hivemind.utils", "DHTExpiration"),
    ("hivemind.utils", "MPFuture"),
    ("hivemind.utils", "get_dht_time"),
    ("hivemind.utils", "get_logger"),
    ("hivemind.utils", "limits"),
    ("hivemind.utils.logging", "use_hivemind_log_handler"),
    ("hivemind.utils.networking", "log_visible_maddrs"),
    ("hivemind.utils.nested", "nested_flatten"),
    ("hivemind.utils.streaming", "split_for_streaming"),
    ("hivemind.utils.tensor_descr", "BatchTensorDescriptor"),
    ("hivemind.utils.asyncio", "aiter_with_timeout"),
    ("hivemind.utils.asyncio", "iter_as_aiter"),
    ("hivemind.compression.serialization", "serialize_torch_tensor"),
    ("hivemind.compression.serialization", "deserialize_torch_tensor"),
    ("hivemind.compression.serialization", "deserialize_tensor_stream"),
]

DEEPER = [
    # MoE machinery, repurposed by Petals for block hosting rather than experts.
    ("hivemind.moe.expert_uid", "ExpertUID"),
    ("hivemind.moe.server.module_backend", "ModuleBackend"),
    ("hivemind.moe.client.remote_expert_worker", "RemoteExpertWorker"),
    # p2p daemon binding internals.
    ("hivemind.p2p.p2p_daemon_bindings.control", "DEFAULT_MAX_MSG_SIZE"),
    ("hivemind.p2p.p2p_daemon_bindings.control", "MAX_UNARY_PAYLOAD_SIZE"),
    # A module-level global Petals assigns to (petals/__init__.py line 31).
    ("hivemind.compression.base", "USE_LEGACY_BFLOAT16"),
]

# Things Seedmesh would need to publish signed reputation records.
SEEDMESH_NEEDS = [
    ("hivemind.dht.validation", "RecordValidatorBase"),
    ("hivemind.dht.validation", "CompoundValidator"),
    ("hivemind.dht.crypto", "RSASignatureValidator"),
    ("hivemind.dht.schema", "SchemaValidator"),
    ("hivemind.dht.routing", "DHTID"),
]


def check(module_path: str, name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        return False, f"import failed: {type(exc).__name__}: {exc}"
    if not hasattr(module, name):
        return False, "symbol missing"
    return True, ""


def report(title: str, entries: list[tuple[str, str]]) -> int:
    print(f"=== {title} ===")
    broken = 0
    for module_path, name in entries:
        ok, detail = check(module_path, name)
        if not ok:
            broken += 1
        label = f"{module_path}.{name}"
        print(f"  {'OK  ' if ok else 'GONE'}  {label:58s} {detail}")
    print(f"  -> {len(entries) - broken}/{len(entries)} present\n")
    return broken


def main() -> int:
    import hivemind

    print(f"python   {sys.version.split()[0]}")
    print(f"hivemind {hivemind.__version__}  (Petals pins a ~1.1.10-era git SHA)\n")

    broken = 0
    broken += report("public API", PUBLIC)
    broken += report("deeper reaches (moe / p2p internals / module globals)", DEEPER)
    broken += report("what Seedmesh would need for signed DHT records", SEEDMESH_NEEDS)

    print("=== signatures of the DHT calls Petals actually makes ===")
    try:
        from hivemind.dht import DHT, DHTNode

        for owner, method in ((DHT, "store"), (DHT, "get"), (DHT, "run_coroutine"),
                              (DHTNode, "store_many"), (DHTNode, "get_many")):
            try:
                sig = inspect.signature(getattr(owner, method))
                print(f"  {owner.__name__}.{method}{sig}")
            except Exception as exc:
                print(f"  {owner.__name__}.{method}: <{exc}>")
    except Exception as exc:
        print(f"  could not inspect: {exc}")

    print()
    print(f"RESULT: {broken} broken symbol(s) across all groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
