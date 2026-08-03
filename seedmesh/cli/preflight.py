"""Checks that run before a client commits to a long, silent download.

Two problems, reported by real use while bringing up the 8B model.

**The silence.** `AutoDistributedModelForCausalLM.from_pretrained` fetches the embedding
matrix and lm_head -- ~2 GiB for an 8B model -- before it can do anything. Seedmesh printed
nothing before that call, so `seedmesh gateway --model 8b` emitted two lines and then sat
there. Indistinguishable from a hang, and the natural response is to kill it and retry, which
starts the download again.

**The wasted download, which is the worse one.** That fetch happened before anyone asked
whether the swarm can serve the model at all. Downloading 2 GiB and waiting out a five-minute
routing warm-up, only to discover that 9 of 32 blocks have no host, is the wrong order of
operations: the DHT can answer that in seconds, and the answer does not depend on having the
weights. So coverage is checked first.

The check is deliberately advisory rather than authoritative. A DHT read can fail for reasons
that say nothing about the swarm, and refusing to start because a lookup timed out would be a
worse failure than the one being prevented. Anything unexpected means "proceed".
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple


def client_download_bytes(model: str) -> Optional[int]:
    """Roughly what a *client* must fetch: embeddings + lm_head, at bf16.

    Not the whole model -- a client never holds blocks. For Llama-3.1-8B this is 1.96 GiB
    against 15 GiB of weights, which is worth saying out loud because "8B model" primes
    people to expect the larger number and abandon the download.
    """
    try:
        from seedmesh.cli.hardware import fetch_config

        config = fetch_config(model)
        vocab = int(config["vocab_size"])
        hidden = int(config["hidden_size"])
    except Exception:
        return None
    # Two matrices of vocab x hidden, two bytes per parameter.
    return 2 * vocab * hidden * 2


def describe_download(model: str) -> List[str]:
    """What to print immediately before the long silent call."""
    size = client_download_bytes(model)
    if size is None:
        return ["loading the model (first run downloads the tokenizer and embeddings)..."]
    if size < 256 * 2**20:
        return [f"loading the model (~{size / 2**20:.0f} MiB on first run, then cached)..."]
    return [
        f"loading the model -- first run downloads ~{size / 2**30:.1f} GiB of embeddings",
        "  (this is the client's share; block weights live on the servers, not here)",
        "  cached afterwards, so this is a one-time cost.",
    ]


def coverage_problem(dht: Any, model: str, prefix: str = "seedmesh") -> Optional[List[str]]:
    """Lines explaining why this model cannot be served, or None if it can.

    Returns None on any failure to determine the answer -- see the module docstring.
    """
    try:
        from seedmesh.cli.monitor import collect, render_text

        report = collect(dht, model, prefix=prefix)
    except Exception:
        return None

    if getattr(report, "covered", False):
        return None

    # Reuse the monitor's wording rather than inventing a second vocabulary for the same
    # situation. Someone who has run `seedmesh monitor` should recognise this immediately.
    try:
        lines = [line for line in render_text(report) if line.strip()]
    except Exception:
        return None
    return lines or None


def check_before_loading(
    dht: Any, model: str, *, prefix: str = "seedmesh", skip: bool = False
) -> Tuple[bool, List[str]]:
    """(ok_to_proceed, lines_to_print).

    `skip` exists because the check reads the DHT's *current* view, and someone who knows a
    server is seconds from finishing should be able to start anyway rather than be told no.
    """
    if skip:
        return True, []
    problem = coverage_problem(dht, model, prefix=prefix)
    if problem is None:
        return True, []
    return False, problem + [
        "",
        "Not downloading the model, because it could not be used once downloaded.",
        "Re-run when coverage is complete, or pass --skip-coverage-check to proceed anyway.",
    ]
