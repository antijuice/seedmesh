"""Verify a ported Petals checkout actually works, not just that it imports.

`port_petals.py` applies the patches; this checks they produced something functional. The
distinction matters: an earlier run of the codemod reported four clean "OK" lines and then
failed on the very next import, because a partial rewrite counted as success.

Checks, in increasing order of how much they prove:

1. `import petals` on current dependencies at all.
2. Petals' own model config resolves the ported block class.
3. A block constructs through Petals' `config.block_class`, not by direct instantiation.
4. Forward pass runs and returns the shape Petals' server expects.
5. Incremental decoding through Petals' `layer_past` convention -- the path that exercises
   the cache-layout translation, where a transposed reshape passes a single forward and
   fails from the second token on.
6. `convert_block` runs end-to-end on a single device -- the tensor-parallel skip.
7. Every server, client and CLI entry point imports.

Usage:
    python tools/verify_petals_port.py
"""

from __future__ import annotations

import torch

RESULTS: list[tuple[str, bool, str]] = []

# The server, client and CLI surface a working backend adapter needs.
ENTRY_POINTS = [
    ("petals", "AutoDistributedModelForCausalLM"),
    ("petals.server.server", "Server"),
    ("petals.server.handler", "TransformerConnectionHandler"),
    ("petals.server.backend", "TransformerBackend"),
    ("petals.server.block_functions", "run_rpc_forward"),
    ("petals.server.from_pretrained", "load_pretrained_block"),
    ("petals.client.routing.sequence_manager", "RemoteSequenceManager"),
    ("petals.client.inference_session", "InferenceSession"),
    ("petals.client.remote_sequential", "RemoteSequential"),
    ("petals.utils.dht", "declare_active_modules"),
    ("petals.utils.convert_block", "convert_block"),
    ("petals.cli.run_server", "main"),
]


def check(name: str, passed: bool, detail: str = "") -> bool:
    RESULTS.append((name, passed, detail))
    print(f"  {'PASS' if passed else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return passed


def small_config():
    from petals.models.llama.config import DistributedLlamaConfig

    return DistributedLlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        vocab_size=1000,
        max_position_embeddings=512,
        _attn_implementation="eager",
    )


def main() -> int:
    print("verifying ported Petals checkout\n")

    # 1 -----------------------------------------------------------------------
    try:
        import petals
        import transformers

        import hivemind

        check("import petals", True,
              f"petals {petals.__version__}, transformers {transformers.__version__}, "
              f"hivemind {hivemind.__version__}, torch {torch.__version__}")
    except Exception as exc:
        check("import petals", False, f"{type(exc).__name__}: {exc}")
        return 1

    # 2 -----------------------------------------------------------------------
    try:
        config = small_config()
        block_class = config.block_class
        check("config resolves the ported block class", True, block_class.__name__)
    except Exception as exc:
        check("config resolves the ported block class", False, f"{type(exc).__name__}: {exc}")
        return 1

    # 3 -----------------------------------------------------------------------
    try:
        block = block_class(config).eval()
        params = sum(p.numel() for p in block.parameters())
        check("block constructs via Petals' block_class", True, f"{params:,} params")
    except Exception as exc:
        check("block constructs via Petals' block_class", False, f"{type(exc).__name__}: {exc}")
        return 1

    # 4 -----------------------------------------------------------------------
    hidden = torch.randn(2, 6, config.hidden_size)
    try:
        with torch.no_grad():
            output = block(hidden)
        tensor = output[0] if isinstance(output, tuple) else output
        ok = tuple(tensor.shape) == (2, 6, config.hidden_size)
        check("forward pass returns the expected shape", ok, str(tuple(tensor.shape)))
    except Exception as exc:
        check("forward pass returns the expected shape", False, f"{type(exc).__name__}: {exc}")

    # 5 -----------------------------------------------------------------------
    try:
        with torch.no_grad():
            full = block(hidden)[0]
            prefix = block(hidden[:, :4, :], use_cache=True)
            cached = block(hidden[:, 4:, :], layer_past=prefix[1], use_cache=True)
        delta = (full[:, 4:, :] - cached[0]).abs().max().item()
        check("incremental decode matches full sequence", delta < 1e-4, f"max diff {delta:.3e}")
    except Exception as exc:
        check("incremental decode matches full sequence", False, f"{type(exc).__name__}: {exc}")

    # 6 -----------------------------------------------------------------------
    try:
        from petals.utils.convert_block import QuantType, convert_block

        fresh = block_class(config).eval()
        converted = convert_block(
            fresh,
            0,
            config,
            tensor_parallel_devices=(torch.device("cpu"),),
            output_device=torch.device("cpu"),
            quant_type=QuantType.NONE,
            freeze=True,
        )
        with torch.no_grad():
            result = converted(hidden)
        tensor = result[0] if isinstance(result, tuple) else result
        ok = tuple(tensor.shape) == (2, 6, config.hidden_size)
        # The TensorParallel wrapper is retained (an earlier revision removed it, which
        # broke TransformerBackend's six dependencies on it); what was fixed instead is the
        # removed `num_heads` read inside it.
        check("convert_block works on a single device", ok,
              f"wrapped as {type(converted).__name__}")
    except Exception as exc:
        import traceback

        traceback.print_exc()
        check("convert_block works on a single device", False, f"{type(exc).__name__}: {exc}")

    # 7 -----------------------------------------------------------------------
    import importlib

    broken = []
    for module_path, symbol in ENTRY_POINTS:
        try:
            getattr(importlib.import_module(module_path), symbol)
        except Exception as exc:
            broken.append(f"{module_path}.{symbol}: {type(exc).__name__}: {exc}")
    check(f"all {len(ENTRY_POINTS)} server/client/CLI entry points import",
          not broken, f"{len(ENTRY_POINTS) - len(broken)}/{len(ENTRY_POINTS)}")
    for failure in broken:
        print(f"          {failure}")

    print()
    failures = sum(1 for _, passed, _ in RESULTS if not passed)
    print(f"{len(RESULTS) - failures}/{len(RESULTS)} checks passed")
    if not failures:
        print("\nNOTE: this proves the port loads and computes correctly at block level.")
        print("It does NOT prove a running swarm -- no DHT server, no real weights, no GPU,")
        print("and quantized convert_block paths need CUDA. Two-node inference is next.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
