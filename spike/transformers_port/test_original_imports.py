"""Load Petals' *unmodified* Llama block against the installed transformers, and report
exactly what breaks.

This executes the real upstream file straight from a Petals checkout rather than a copy, so
the result cannot drift from what Petals actually ships. The only thing stubbed is
``petals.utils.cuda_graphs``, which is Petals-internal and would otherwise mask the
transformers failures behind an unrelated ImportError.

Usage:
    python test_original_imports.py <path-to-petals-checkout>
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def install_petals_stubs() -> None:
    """Stub the Petals-internal import so only transformers failures surface."""
    petals = types.ModuleType("petals")
    utils = types.ModuleType("petals.utils")
    cuda_graphs = types.ModuleType("petals.utils.cuda_graphs")

    def make_inference_graphed_callable(func, sample_args, **kwargs):  # noqa: ANN001
        return func

    cuda_graphs.make_inference_graphed_callable = make_inference_graphed_callable
    utils.cuda_graphs = cuda_graphs
    petals.utils = utils

    sys.modules["petals"] = petals
    sys.modules["petals.utils"] = utils
    sys.modules["petals.utils.cuda_graphs"] = cuda_graphs


def try_load(block_path: Path) -> list[str]:
    """Attempt to exec the original module; return a list of failures encountered."""
    failures: list[str] = []
    source = block_path.read_text(encoding="utf-8")
    namespace: dict = {"__name__": "petals_original_block", "__file__": str(block_path)}

    try:
        exec(compile(source, str(block_path), "exec"), namespace)
    except Exception as exc:
        failures.append(f"module import: {type(exc).__name__}: {exc}")
        return failures

    # Import succeeded -- now check the classes can actually be constructed, which is where
    # attribute-level drift (self.num_heads, self.rotary_emb) shows up.
    try:
        from transformers.models.llama.modeling_llama import LlamaConfig

        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=256,
        )
        block = namespace["WrappedLlamaBlock"](config)
    except Exception as exc:
        failures.append(f"construct WrappedLlamaBlock: {type(exc).__name__}: {exc}")
        return failures

    try:
        import torch

        block.eval()
        with torch.no_grad():
            block(torch.randn(1, 4, config.hidden_size))
    except Exception as exc:
        failures.append(f"forward pass: {type(exc).__name__}: {exc}")

    return failures


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    checkout = Path(sys.argv[1])
    block_path = checkout / "src" / "petals" / "models" / "llama" / "block.py"
    if not block_path.exists():
        print(f"not found: {block_path}")
        return 2

    import torch
    import transformers

    print(f"torch {torch.__version__} / transformers {transformers.__version__}")
    print(f"loading unmodified: {block_path}\n")

    install_petals_stubs()
    failures = try_load(block_path)

    if not failures:
        print("RESULT: upstream block loads, constructs and runs unmodified.")
        return 0

    print("RESULT: upstream block does NOT work. Failures, in the order hit:\n")
    for index, failure in enumerate(failures, start=1):
        print(f"  {index}. {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
