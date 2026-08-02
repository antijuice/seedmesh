"""What would it actually take to serve a trillion-parameter MoE on volunteer hardware?

Three numbers decide this, and they pull in opposite directions:

  bytes on the wire   scales with HIDDEN SIZE, not parameter count. Activations flow between
                      block ranges; expert routing happens *inside* a block. So an MoE costs
                      the same per token as a dense model of the same hidden size -- its
                      parameters are nearly free on the network.
  storage per host    scales with TOTAL parameters. This is what decides how many volunteers
                      are needed.
  compute per token   scales with ACTIVE parameters, which for an MoE is a small fraction.

Reads real configs from the Hub (a few KB each, no weights). Everything else is arithmetic
from the measured constants in this project:
  * 128 KiB  -- go-libp2p circuit-relay budget, the cap on a relayed volunteer
  * hidden x dtype bytes per token per block-range hop, verified against llama-160m
    (hidden 768, fp32 = 3072 B/token, predicted the stall period to within 10%)
"""

import json

from huggingface_hub import hf_hub_download

RELAY_BUDGET_KIB = 128
CANDIDATES = [
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", "Mixtral 8x7B — the only MoE Petals registers"),
    ("deepseek-ai/DeepSeek-V3", "DeepSeek-V3 — the architecture large open MoEs now follow"),
    ("moonshotai/Kimi-K2-Instruct", "Kimi K2 — trillion-scale MoE"),
]


def load_config(repo):
    path = hf_hub_download(repo, "config.json")
    return json.loads(open(path, encoding="utf-8").read())


def describe(repo, note):
    try:
        cfg = load_config(repo)
    except Exception as exc:
        print(f"\n### {repo}\n  unavailable: {type(exc).__name__}: {str(exc)[:120]}")
        return None

    hidden = cfg.get("hidden_size")
    layers = cfg.get("num_hidden_layers")
    experts = cfg.get("num_local_experts") or cfg.get("n_routed_experts")
    active = cfg.get("num_experts_per_tok")
    arch = (cfg.get("architectures") or ["?"])[0]
    model_type = cfg.get("model_type")

    print(f"\n### {repo}")
    print(f"  {note}")
    print(f"  model_type={model_type}  arch={arch}")
    print(f"  hidden={hidden}  layers={layers}  experts={experts}  active/token={active}")

    if not hidden:
        print("  (no hidden_size — cannot size)")
        return None

    # Wire cost per block-range hop. bf16 is what a server would actually use.
    for dtype, width in (("bf16", 2), ("fp32", 4)):
        per_token = hidden * width
        # A modest request: 32-token prompt + 128 generated.
        per_request_kb = per_token * (32 + 128) / 1024
        fits = per_request_kb < RELAY_BUDGET_KIB
        print(f"  [{dtype}] {per_token:>6,} B/token  |  160-token request = "
              f"{per_request_kb:>8,.0f} KB  |  fits in a {RELAY_BUDGET_KIB} KiB relay: "
              f"{'yes' if fits else 'NO'}")
    return cfg


print("=" * 78)
print("WIRE COST — scales with hidden size, NOT parameter count")
print("=" * 78)
configs = {repo: describe(repo, note) for repo, note in CANDIDATES}

print("\n" + "=" * 78)
print("STORAGE — what a swarm needs, and how many volunteers that is")
print("=" * 78)
print(f"\n{'model':<38} {'params':>8} {'@4bit':>9} {'per layer':>10} {'8GB hosts':>10}")
print("-" * 78)

# Total parameter counts are not in config.json; these are the published figures.
KNOWN_PARAMS_B = {
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 46.7,
    "deepseek-ai/DeepSeek-V3": 671.0,
    "moonshotai/Kimi-K2-Instruct": 1000.0,
}
for repo, cfg in configs.items():
    if not cfg:
        continue
    params_b = KNOWN_PARAMS_B.get(repo)
    layers = cfg.get("num_hidden_layers")
    if not params_b or not layers:
        continue
    gb_4bit = params_b * 1e9 * 0.5156 / 2**30  # NF4 incl. scales, as measured in this project
    per_layer_gb = gb_4bit / layers
    hosts_8gb = -(-gb_4bit // 8)  # ceil
    print(f"{repo.split('/')[-1]:<38} {params_b:>7.0f}B {gb_4bit:>8.0f}G {per_layer_gb:>9.1f}G "
          f"{hosts_8gb:>10.0f}")

print("\nNotes")
print("  * 'per layer' is what one volunteer must hold to host a single block. Where that")
print("    exceeds a consumer GPU, blocks cannot be split further without changing Petals.")
print("  * '8GB hosts' assumes perfect packing and no redundancy. Real swarms want each")
print("    block range covered 2-3x, so multiply accordingly.")
print("  * Wire cost is independent of all of this — an MoE is cheap on the network and")
print("    expensive on disk, which is the opposite of what the relay budget punishes.")
