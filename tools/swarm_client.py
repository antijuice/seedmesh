"""Client for the private two-server swarm: discover blocks, then generate tokens.

Proves what import checks could not — that `RemoteSequential` and the sequence manager
build a real pipeline spanning two hosts and stream activations through it.

Launched by tools/swarm_demo.sh, which brings the servers up first.
"""

from __future__ import annotations

from pathlib import Path

import torch
from hivemind.dht import DHT
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM
from petals.utils.dht import get_remote_module_infos

MODEL = "JackFram/llama-160m"
NLAYERS = 12
PROMPT = "The capital of France is"
INITIAL_PEERS = [Path("/tmp/srv1.addr").read_text().strip()]


def short(peer) -> str:
    return str(peer)[-8:]


def main() -> int:
    print(f"initial peers: {INITIAL_PEERS}\n")

    # --- 1. what does the DHT say is available? -----------------------------
    dht = DHT(initial_peers=INITIAL_PEERS, client_mode=True, start=True)
    uids = [f"llama-160m-hf.{i}" for i in range(NLAYERS)]
    infos = get_remote_module_infos(dht, uids, latest=True)

    coverage = {
        index: [short(p) for p in (info.servers or {})] if info else []
        for index, info in enumerate(infos)
    }
    dht.shutdown()

    print("block -> hosting peer")
    for index, peers in coverage.items():
        print(f"  block {index:2d}: {peers}")

    distinct = sorted({p for peers in coverage.values() for p in peers})
    covered = sum(1 for peers in coverage.values() if peers)
    print(f"\n{covered}/{NLAYERS} blocks announced across {len(distinct)} server(s): {distinct}")

    if covered < NLAYERS:
        print("\nFAIL: the model is not fully covered; generation would be impossible")
        return 1
    if len(distinct) < 2:
        print("\nFAIL: only one server -- this would not test multi-host routing")
        return 1

    # --- 2. actually generate ------------------------------------------------
    print("\nloading distributed model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        MODEL, initial_peers=INITIAL_PEERS, torch_dtype=torch.float32
    )

    inputs = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    print(f"prompt: {PROMPT!r}")

    with torch.inference_mode():
        outputs = model.generate(inputs, max_new_tokens=12, do_sample=False)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGENERATED: {text!r}")
    print(f"tokens in {inputs.shape[1]} -> out {outputs.shape[1]}")

    # --- 3. which servers did the pipeline actually use? ---------------------
    try:
        manager = model.transformer.h.sequence_manager
        spans = manager.state.sequence_info.spans_by_priority
        print("\nroute actually used:")
        for span in spans:
            print(f"  blocks {span.start:2d}-{span.end - 1:2d} via peer ...{short(span.peer_id)}")
        hops = len({short(s.peer_id) for s in spans})
        print(f"  {len(spans)} span(s) across {hops} distinct host(s)")
    except Exception as exc:
        print(f"\n(could not introspect route: {type(exc).__name__}: {exc})")

    print("\nSWARM INFERENCE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
