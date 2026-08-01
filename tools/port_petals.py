"""Port a Petals checkout onto current transformers / hivemind / bitsandbytes.

Petals has been unmaintained since 2024-09-07. Four measurement spikes established that
reviving it is bounded and mostly mechanical (`spike/*/README.md`), and this applies those
findings to a real checkout.

**A codemod, not a fork.** Vendoring 67 files would make "what did we actually change to
upstream?" unanswerable, and would silently absorb upstream's bugs as ours. A patch script
keeps the delta explicit, reviewable, and re-appliable to a newer checkout.

Three changes, each traced to the spike that justified it:

1. **hivemind import paths** (`spike/hivemind_dht/`). hivemind 1.1.12 narrowed its top-level
   namespace; every symbol still exists at its canonical submodule path. 7 statement forms
   across ~22 files. Purely mechanical.

2. **Llama block wrapper** (`spike/transformers_port/`). Petals forked transformers'
   attention body to splice in two CUDA-graph optimizations; that fork is what broke. The
   replacement deletes it and delegates to the stock `LlamaDecoderLayer` — 138 lines
   replacing 221, verified numerically exact including the cached-decode path.

3. **Attention head counts.** `convert_block` and `TransformerBackend` both read
   `LlamaAttention.num_heads`, removed in transformers 4.48+. Derived from config instead,
   keeping upstream's assertions so they validate the derivation.

4. **`get_file_from_repo`** removed from `transformers.utils` in 5.x, shimmed for every
   importer.

Usage:
    python tools/port_petals.py --petals-root ~/petals --check
    python tools/port_petals.py --petals-root ~/petals
"""

from __future__ import annotations

import argparse
import importlib
import re
from pathlib import Path

# Canonical home of each symbol Petals imports from the hivemind root namespace.
# Verified against the installed hivemind before anything is written -- a stale entry here
# would rewrite working imports into broken ones, which is worse than not running.
HIVEMIND_MOVES: dict[str, str] = {
    "PeerID": "hivemind.p2p",
    "P2P": "hivemind.p2p",
    "DHT": "hivemind.dht",
    "DHTNode": "hivemind.dht",
    "DHTValue": "hivemind.dht",
    "MSGPackSerializer": "hivemind.utils",
    "get_logger": "hivemind.utils",
    "use_hivemind_log_handler": "hivemind.utils.logging",
    "TensorDescriptor": "hivemind.utils",
    "BatchTensorDescriptor": "hivemind.utils.tensor_descr",
    "nested_flatten": "hivemind.utils",
    "nested_pack": "hivemind.utils",
    "nested_compare": "hivemind.utils.nested",
    "serialize_torch_tensor": "hivemind.compression",
    "deserialize_torch_tensor": "hivemind.compression",
    "deserialize_tensor_stream": "hivemind.compression",
    "P2PContext": "hivemind.p2p",
    "anext": "hivemind.utils.asyncio",
    "get_dht_time": "hivemind.utils",
    "DHTExpiration": "hivemind.utils",
    "MPFuture": "hivemind.utils",
    "MAX_DHT_TIME_DISCREPANCY_SECONDS": "hivemind.utils",
}

# Root-namespace *attribute* access: `import hivemind` then `hivemind.PeerID`.
#
# A second breakage form, and one an import-statement rewrite misses entirely -- which is
# how the first run of this tool produced a clean report and then failed on the very next
# import. Rewriting to the canonical submodule needs no new import lines, because
# `hivemind.p2p`, `hivemind.utils` and `hivemind.moe` all still resolve as attributes of the
# root package. That keeps this a pure textual substitution and avoids monkey-patching
# symbols back onto a third-party namespace.
HIVEMIND_ATTR_MOVES: dict[str, str] = {
    "PeerID": "hivemind.p2p",
    "get_dht_time": "hivemind.utils",
    "TimedStorage": "hivemind.utils",
    "Runtime": "hivemind.moe.server.runtime",
}

# `from hivemind import ...`, including the parenthesised multi-line form.
HIVEMIND_IMPORT_RE = re.compile(
    r"^from hivemind import (?:\(\s*(?P<paren>[^)]*?)\s*\)|(?P<plain>[^(\n]+))$",
    re.MULTILINE,
)

PORTED_BLOCK = Path(__file__).resolve().parents[1] / "spike" / "transformers_port" / "ported_block.py"

HF_COMPAT_MODULE = '''"""Shims for transformers APIs Petals used that current versions removed.

Written by tools/port_petals.py. See spike/transformers_port/README.md.
"""

from __future__ import annotations

from typing import Optional

from transformers.utils import cached_file


def get_file_from_repo(path_or_repo_id, filename: str, **kwargs) -> Optional[str]:
    """Restore `transformers.utils.get_file_from_repo`, removed in transformers 5.x.

    `cached_file` is the replacement, but it differs in two ways that matter to every
    call site in Petals' weight loader:

    * it raises for a missing entry, where `get_file_from_repo` returned ``None`` --
      and Petals uses the ``None`` return as a "does this file exist?" probe;
    * the auth keyword was renamed from ``use_auth_token`` to ``token``.

    Exceptions are narrowed to OSError/ValueError deliberately. The original propagated
    genuine failures and only returned ``None`` for a missing file, so catching everything
    here would turn a network or auth outage into a silent "file not found".
    """
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    try:
        return cached_file(path_or_repo_id, filename, **kwargs)
    except (OSError, ValueError):
        return None
'''

HF_IMPORT_ORIGINAL = "from transformers.utils import get_file_from_repo\n"
HF_IMPORT_PATCHED = "from petals.utils.hf_compat import get_file_from_repo\n"

# `_supports_cache_class` was a transformers flag meaning "this model understands the Cache
# API". Every model does in 5.x, so the attribute was removed -- and reading a missing
# attribute on an nn.Module raises rather than returning None. Defaulting to True preserves
# the intent, since the ported block does support Cache.
# transformers 5.x `Cache.__init__` requires exactly one of `layers` or
# `layer_class_to_replicate`. RemotePastKeyValues is a stub that only counts seen tokens
# (it overrides __getitem__ and get_seq_length), so an empty layer list is the honest
# representation of "this cache holds nothing locally".
CACHE_INIT_ORIGINAL = """    def __init__(self) -> None:
        super().__init__()
        self._seen_tokens = 0"""
CACHE_INIT_PATCHED = """    def __init__(self) -> None:
        # PORT: transformers 5.x requires exactly one of `layers` /
        # `layer_class_to_replicate`. This cache stores nothing locally -- the real KV
        # cache lives on the servers -- so an empty layer list is accurate.
        super().__init__(layers=[])
        self._seen_tokens = 0"""

# Publish the attention kernel in the server record.
#
# Petals already announces `torch_dtype` and `quant_type` -- two thirds of what determines a
# server's numerical noise floor. The third is the attention implementation, and without it
# a client cannot know whether two servers are comparable. Seedmesh calibrates verification
# tolerances per pair of compute profiles (spike/quantization/), so an unpublished kernel
# means comparing against a threshold fitted for a different one.
#
# Safe to add: ServerInfo serializes as (state, throughput, extra_dict) and `from_tuple`
# ignores unknown keys, so an old client skips the field and a new client reads None from an
# old server.
ATTN_FIELD_ORIGINAL = """    torch_dtype: Optional[str] = None
    quant_type: Optional[str] = None"""
ATTN_FIELD_PATCHED = """    torch_dtype: Optional[str] = None
    quant_type: Optional[str] = None
    # PORT: the attention kernel completes the compute profile alongside dtype and
    # quant_type. Clients need it to know whether two servers are numerically comparable.
    attn_impl: Optional[str] = None"""

ATTN_ANNOUNCE_ORIGINAL = """            torch_dtype=str(torch_dtype).replace("torch.", ""),
            quant_type=quant_type.name.lower(),"""
ATTN_ANNOUNCE_PATCHED = """            torch_dtype=str(torch_dtype).replace("torch.", ""),
            quant_type=quant_type.name.lower(),
            # PORT: publish the attention kernel actually in use, so clients can select the
            # right verification tolerance. Falls back to "eager", which is what a hosted
            # block defaults to when no model wrapper sets the dispatch.
            attn_impl=str(getattr(self.block_config, "_attn_implementation", None) or "eager"),"""

GENERATION_ORIGINAL = '            if self._supports_cache_class and "past_key_values" not in kwargs:'
GENERATION_PATCHED = (
    "            # PORT: _supports_cache_class was removed in transformers 5.x; every model\n"
    "            # supports the Cache API now, so absence means True.\n"
    '            if getattr(self, "_supports_cache_class", True) and "past_key_values" not in kwargs:'
)

VERSION_GUARD_ORIGINAL = '''    assert (
        version.parse("4.43.1") <= version.parse(transformers.__version__) < version.parse("4.44.0")
    ), "Please install a proper transformers version: pip install transformers>=4.43.1,<4.44.0"
'''

VERSION_GUARD_PATCHED = '''    # PORT: widened, not deleted. The original pinned transformers to a single patch
    # release; the block wrapper was ported to the 4.48+ attention interface and verified
    # numerically exact against stock LlamaDecoderLayer (spike/transformers_port/). The
    # guard still exists to catch a genuinely incompatible version -- anything below the
    # attention refactor will not work with the ported block.
    assert version.parse(transformers.__version__) >= version.parse("4.48.0"), (
        "This port requires transformers>=4.48 (the attention-interface refactor). "
        "Install with: pip install 'transformers>=4.48'"
    )
'''

# Head counting, in the two places that read the removed `LlamaAttention.num_heads`.
#
# The quantization spike concluded tensor_parallel was "deletable for single-GPU". Running
# a real server proved that wrong: `TransformerBackend` depends on the TensorParallel
# wrapper in six places (module_shards, devices, output_device_index, PerDeviceTensors, and
# two asserts), so removing it cascades. And tensor_parallel==1.0.23 itself imports and
# works on current torch -- only this attribute read was broken.
#
# So the wrapper stays and the head count is derived from config instead. Both call sites
# keep their original assertions, which then validate the derivation.
NUM_HEADS_ORIGINAL_CONVERT = """                total_heads += submodule.num_heads"""
NUM_HEADS_PATCHED_CONVERT = """                # PORT: LlamaAttention.num_heads was removed in transformers 4.48+.
                total_heads += getattr(
                    submodule,
                    "num_heads",
                    model_config.num_attention_heads // len(tp_block.module_shards),
                )"""

NUM_HEADS_ORIGINAL_BACKEND = """                    self.shard_num_heads.append(submodule.num_heads)"""
NUM_HEADS_PATCHED_BACKEND = """                    # PORT: LlamaAttention.num_heads was removed in transformers 4.48+.
                    self.shard_num_heads.append(
                        getattr(
                            submodule,
                            "num_heads",
                            config.num_attention_heads // len(self.module.module_shards),
                        )
                    )"""


def verify_mapping() -> list[str]:
    """Confirm every mapped symbol really resolves in the installed hivemind."""
    problems = []
    combined = {**HIVEMIND_MOVES, **HIVEMIND_ATTR_MOVES}
    for symbol, module_path in sorted(combined.items()):
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:
            problems.append(f"{module_path} does not import ({type(exc).__name__}: {exc})")
            continue
        if not hasattr(module, symbol):
            problems.append(f"{module_path}.{symbol} not found")
    return problems


def apply_attribute_fix(root: Path, write: bool) -> tuple[int, int]:
    """Rewrite `hivemind.X` to `<canonical module>.X` for symbols that left the root."""
    files_changed = rewrites = 0
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        updated = source
        for symbol, module_path in HIVEMIND_ATTR_MOVES.items():
            # \b...(?!\w) so hivemind.PeerID does not also match hivemind.PeerIDSomething,
            # and the negative lookbehind leaves already-rewritten text alone (idempotence).
            pattern = re.compile(rf"(?<![\w.])hivemind\.{symbol}(?!\w)")
            updated, count = pattern.subn(f"{module_path}.{symbol}", updated)
            rewrites += count
        if updated != source:
            files_changed += 1
            if write:
                path.write_text(updated, encoding="utf-8")
    return files_changed, rewrites


def rewrite_hivemind_imports(source: str) -> tuple[str, int, set[str]]:
    """Rewrite root-namespace hivemind imports to canonical submodule paths.

    Returns the new source, how many statements changed, and any symbols with no mapping.
    A statement containing an unmapped symbol is left **entirely** alone -- emitting a
    half-rewritten import would hide the gap behind a partial success.
    """
    replacements = 0
    unmapped: set[str] = set()

    def replace(match: re.Match) -> str:
        nonlocal replacements
        body = match.group("paren") or match.group("plain")
        raw_names = [n.strip() for n in body.replace("\n", " ").split(",") if n.strip()]

        by_module: dict[str, list[str]] = {}
        missing = [n.split(" as ")[0].strip() for n in raw_names
                   if n.split(" as ")[0].strip() not in HIVEMIND_MOVES]
        if missing:
            unmapped.update(missing)
            return match.group(0)

        for entry in raw_names:
            target = HIVEMIND_MOVES[entry.split(" as ")[0].strip()]
            by_module.setdefault(target, []).append(entry)

        replacements += 1
        return "\n".join(
            f"from {module} import {', '.join(sorted(entries))}"
            for module, entries in sorted(by_module.items())
        )

    return HIVEMIND_IMPORT_RE.sub(replace, source), replacements, unmapped


def apply_import_fix(root: Path, write: bool) -> tuple[int, int, set[str]]:
    files_changed = statements = 0
    unmapped: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "from hivemind import" not in source:
            continue
        updated, count, missing = rewrite_hivemind_imports(source)
        unmapped |= missing
        if count and updated != source:
            files_changed += 1
            statements += count
            if write:
                path.write_text(updated, encoding="utf-8")
    return files_changed, statements, unmapped


def apply_block_port(root: Path, write: bool) -> str:
    target = root / "src" / "petals" / "models" / "llama" / "block.py"
    if not target.exists():
        return f"SKIP  {target} not found"
    if not PORTED_BLOCK.exists():
        return f"SKIP  ported block missing at {PORTED_BLOCK}"

    ported = PORTED_BLOCK.read_text(encoding="utf-8")
    if target.read_text(encoding="utf-8") == ported:
        return "OK    llama block already ported"
    if write:
        target.write_text(ported, encoding="utf-8")
    return "OK    llama block replaced with the delegating port"


def apply_hf_compat(root: Path, write: bool) -> str:
    """Install the transformers compat shim and repoint every importer of it.

    Scans the whole tree rather than a known file list. An earlier version patched only
    `server/from_pretrained.py`, which passed every import check and then failed at
    *runtime* -- `utils/peft.py` imports the same symbol, and is reached only once a server
    is actually loading blocks.
    """
    shim = root / "src" / "petals" / "utils" / "hf_compat.py"
    importers = [
        path
        for path in sorted((root / "src").rglob("*.py"))
        if HF_IMPORT_ORIGINAL in path.read_text(encoding="utf-8")
    ]

    if not importers:
        return (
            "OK    transformers compat shim already installed"
            if shim.exists()
            else "WARN  no get_file_from_repo imports found; nothing to shim"
        )

    if write:
        shim.write_text(HF_COMPAT_MODULE, encoding="utf-8")
        for path in importers:
            source = path.read_text(encoding="utf-8")
            path.write_text(source.replace(HF_IMPORT_ORIGINAL, HF_IMPORT_PATCHED), encoding="utf-8")

    names = ", ".join(p.relative_to(root / "src").as_posix() for p in importers)
    return f"OK    get_file_from_repo shimmed in {len(importers)} file(s): {names}"


def apply_attn_publication(root: Path, write: bool) -> str:
    """Add `attn_impl` to ServerInfo and populate it when the server announces itself."""
    sites = [
        (root / "src" / "petals" / "data_structures.py", ATTN_FIELD_ORIGINAL, ATTN_FIELD_PATCHED),
        (root / "src" / "petals" / "server" / "server.py",
         ATTN_ANNOUNCE_ORIGINAL, ATTN_ANNOUNCE_PATCHED),
    ]
    done = patched = 0
    for path, original, replacement in sites:
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        if "attn_impl" in source:
            done += 1
            continue
        if original not in source:
            return f"WARN  {path.name} does not match expected upstream text; patch by hand"
        if write:
            path.write_text(source.replace(original, replacement), encoding="utf-8")
        patched += 1

    if patched:
        return f"OK    attention kernel published in the server record ({patched} file(s))"
    if done:
        return "OK    attention kernel publication already applied"
    return "WARN  no attn publication sites found"


def apply_generation_fix(root: Path, write: bool) -> str:
    target = root / "src" / "petals" / "client" / "remote_generation.py"
    if not target.exists():
        return f"SKIP  {target} not found"
    source = target.read_text(encoding="utf-8")
    if "_supports_cache_class was removed" in source and "layers=[]" in source:
        return "OK    generation cache fixes already applied"

    updated, changes = source, []
    if GENERATION_ORIGINAL in updated:
        updated = updated.replace(GENERATION_ORIGINAL, GENERATION_PATCHED)
        changes.append("_supports_cache_class")
    if CACHE_INIT_ORIGINAL in updated:
        updated = updated.replace(CACHE_INIT_ORIGINAL, CACHE_INIT_PATCHED)
        changes.append("RemotePastKeyValues.__init__")
    if not changes:
        return "WARN  remote_generation does not match expected upstream text; patch by hand"
    if write:
        target.write_text(updated, encoding="utf-8")
    return f"OK    generation patched for transformers 5.x ({', '.join(changes)})"


def apply_version_guard(root: Path, write: bool) -> str:
    target = root / "src" / "petals" / "__init__.py"
    if not target.exists():
        return f"SKIP  {target} not found"
    source = target.read_text(encoding="utf-8")
    if "the attention-interface refactor" in source:
        return "OK    transformers version guard already widened"
    if VERSION_GUARD_ORIGINAL not in source:
        return "WARN  version guard does not match expected upstream text; patch by hand"
    if write:
        target.write_text(source.replace(VERSION_GUARD_ORIGINAL, VERSION_GUARD_PATCHED), encoding="utf-8")
    return "OK    transformers version guard widened to >=4.48"


def apply_num_heads_fix(root: Path, write: bool) -> str:
    """Derive attention head counts from config instead of the removed `num_heads`."""
    sites = [
        (root / "src" / "petals" / "utils" / "convert_block.py",
         NUM_HEADS_ORIGINAL_CONVERT, NUM_HEADS_PATCHED_CONVERT),
        (root / "src" / "petals" / "server" / "backend.py",
         NUM_HEADS_ORIGINAL_BACKEND, NUM_HEADS_PATCHED_BACKEND),
    ]
    done = patched = 0
    for path, original, replacement in sites:
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        if "num_heads was removed in transformers" in source:
            done += 1
            continue
        if original not in source:
            return f"WARN  {path.name} does not match expected upstream text; patch by hand"
        if write:
            path.write_text(source.replace(original, replacement), encoding="utf-8")
        patched += 1

    if patched:
        return f"OK    num_heads derived from config in {patched} file(s)"
    if done:
        return "OK    num_heads fix already applied"
    return "WARN  no num_heads sites found"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--petals-root", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="report without writing")
    args = parser.parse_args()

    root = args.petals_root.expanduser()
    if not (root / "src" / "petals").is_dir():
        print(f"not a Petals checkout: {root}")
        return 2

    write = not args.check
    print(f"{'checking' if args.check else 'porting'} {root}\n")

    try:
        importlib.import_module("hivemind")
    except ImportError:
        # Distinguish "hivemind changed" from "hivemind is missing" -- they look identical
        # in the mapping report but have completely different fixes, and the wall of
        # "No module named 'hivemind'" lines that results is actively misleading.
        print("hivemind is not installed, so the symbol mapping cannot be verified.")
        print("The port rewrites imports to canonical hivemind paths and refuses to do so")
        print("blind. Install dependencies first:")
        print("    pip install 'hivemind==1.1.12' 'transformers>=4.48' torch")
        print("  or let `seedmesh setup` do it in the right order.")
        return 2

    problems = verify_mapping()
    if problems:
        print("hivemind mapping is stale -- refusing to rewrite imports:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"OK    hivemind mapping verified ({len(HIVEMIND_MOVES)} symbols resolve)")

    files_changed, statements, unmapped = apply_import_fix(root, write)
    print(f"OK    {statements} import statement(s) across {files_changed} file(s)")
    attr_files, attr_rewrites = apply_attribute_fix(root, write)
    print(f"OK    {attr_rewrites} attribute access(es) across {attr_files} file(s)")
    print(apply_block_port(root, write))
    print(apply_num_heads_fix(root, write))
    print(apply_hf_compat(root, write))
    print(apply_attn_publication(root, write))
    print(apply_generation_fix(root, write))
    print(apply_version_guard(root, write))

    if unmapped:
        # Loud, and a non-zero exit. Silence here once produced a "successful" port whose
        # very next import failed -- a partial rewrite that reports success is worse than
        # one that reports nothing.
        print(f"\nFAIL  {len(unmapped)} symbol(s) have no mapping; their import statements")
        print("      were left untouched and will still fail:")
        for symbol in sorted(unmapped):
            print(f"        {symbol}")
        print("      Add them to HIVEMIND_MOVES and re-run.")
        return 1

    if args.check:
        print("\n(--check: nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
