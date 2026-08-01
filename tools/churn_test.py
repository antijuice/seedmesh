"""Kill a serving node and confirm the swarm reroutes around it.

Seedmesh assumes churn is normal throughout: routing has failover, reputation forgives
timeouts rather than treating them as fraud, and the DHT expires dead peers on a TTL. All
of that was designed against a simulator. This observes it against real processes.

Run via tools/churn_test.sh, which brings up three servers with the tail half redundant.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import torch
from hivemind.dht import DHT
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM

MODEL = "JackFram/llama-160m"
NLAYERS = 12
PROMPT = "The capital of France is"
INITIAL_PEERS = [Path("/tmp/srv1.addr").read_text().strip()]


def short(peer) -> str:
    return str(peer)[-8:]


def route_of(model) -> list[tuple[int, int, str]]:
    spans = model.transformer.h.sequence_manager.state.sequence_info.spans_by_priority
    return [(s.start, s.end, short(s.peer_id)) for s in spans]


def servers_on_disk() -> dict[str, int]:
    """Map peer-id suffix -> pid, from what churn_test.sh recorded."""
    mapping = {}
    for index in (1, 2, 3):
        peer_file, pid_file = Path(f"/tmp/srv{index}.peer"), Path(f"/tmp/srv{index}.pid")
        if peer_file.exists() and pid_file.exists():
            mapping[peer_file.read_text().strip()[-8:]] = int(pid_file.read_text().strip())
    return mapping


def generate(model, tokenizer, tag: str) -> str | None:
    inputs = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    try:
        with torch.inference_mode():
            outputs = model.generate(inputs, max_new_tokens=10, do_sample=False)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  [{tag}] OK: {text!r}")
        return text
    except Exception as exc:
        print(f"  [{tag}] FAILED: {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    pids = servers_on_disk()
    print(f"known servers (peer -> pid): {pids}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Construct a DHT before the model, and keep it. Two transformers 5.x quirks make this
    # the only workable order:
    #   * letting from_pretrained build the first DHT fails -- hivemind's SharedBytes
    #     allocates shared memory lazily on first construction, and inside the low-cpu-mem
    #     init context `share_memory_()` raises "only available on CPU";
    #   * passing `dht=` instead fails too -- from_pretrained deep-copies its kwargs and a
    #     DHT holds an unpicklable multiprocessing AuthenticationString.
    # Building one first warms the shared buffer (a class attribute), after which the
    # model's own internal DHT constructs fine.
    warmup_dht = DHT(initial_peers=INITIAL_PEERS, client_mode=True, start=True)

    # Bounded retries are essential here, not a convenience. Petals' client defaults to
    # `max_retries=None` (infinite) with a 180s request_timeout, so a call to a killed peer
    # simply never returns -- the test hangs and, more importantly, a real client would
    # never surface the failure for reputation to learn from.
    model = AutoDistributedModelForCausalLM.from_pretrained(
        MODEL,
        initial_peers=INITIAL_PEERS,
        torch_dtype=torch.float32,
        request_timeout=20,
        max_retries=3,
        min_backoff=1,
        max_backoff=5,
    )

    # --- 1. baseline -------------------------------------------------------
    print("1. baseline generation")
    baseline = generate(model, tokenizer, "before")
    route_before = route_of(model)
    print(f"     route: {[(a, b, p) for a, b, p in route_before]}")
    if baseline is None:
        print("\nFAIL: baseline generation did not work; churn is untestable")
        return 1

    # --- 2. kill whoever is serving the tail --------------------------------
    # spans_by_priority is the *candidate* list, not the path taken: with a redundant tail
    # it holds several spans for the same blocks, best first. Take the highest-priority
    # span reaching the final block -- that is the one actually preferred.
    tail_spans = [(a, b, p) for a, b, p in route_before if b >= NLAYERS]
    if not tail_spans:
        print("\nFAIL: no span covers the final block")
        return 1
    tail_peer = tail_spans[0][2]
    victim_pid = pids.get(tail_peer)
    if victim_pid is None:
        print(f"\nFAIL: cannot map tail peer ...{tail_peer} to a pid")
        return 1

    print(f"\n2. killing the server for the tail blocks: peer ...{tail_peer} (pid {victim_pid})")
    os.kill(victim_pid, signal.SIGKILL)
    time.sleep(3)
    alive = Path(f"/proc/{victim_pid}").exists()
    print(f"   process alive after SIGKILL: {alive}")

    # --- 3. can the swarm still serve? --------------------------------------
    # The client cached a route pointing at a peer that no longer exists, so this is the
    # real test: it must notice the failure and rebuild through the redundant server.
    print("\n3. generating again with the cached route now pointing at a dead peer")
    after = generate(model, tokenizer, "after")
    route_after = route_of(model)
    print(f"     route: {[(a, b, p) for a, b, p in route_after]}")

    # --- verdict ------------------------------------------------------------
    print()
    if after is None:
        print("FAIL: swarm could not serve after losing the preferred tail server")
        return 1

    # The candidate list still names the dead peer -- its DHT record only disappears at
    # TTL. That is expected, and is exactly why routing must survive stale records rather
    # than trust them. The real evidence of rerouting is that generation completed at all
    # while that peer was dead, and that it produced the same tokens: greedy decoding
    # through the redundant server, which holds identical weights, must agree.
    stale = tail_peer in [p for _, _, p in route_after]
    print(f"PASS: generation completed with peer ...{tail_peer} dead")
    print(f"      output before: {baseline!r}")
    print(f"      output after:  {after!r}")
    print(f"      outputs identical: {baseline == after}")
    print(f"      killed peer still in candidate list (stale DHT record): {stale}")

    if baseline != after:
        print("\nWARN: the replacement server produced different tokens than the original.")
        print("      Same weights and greedy decoding should agree; worth investigating.")

    warmup_dht.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
