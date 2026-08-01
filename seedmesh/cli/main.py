"""``seedmesh`` command line.

Four commands, aimed at the two things a volunteer actually does -- find out what they can
contribute, and contribute it:

    seedmesh setup     install and patch the Petals backend
    seedmesh probe     what can this machine host?
    seedmesh serve     host blocks for a swarm
    seedmesh chat      talk to a swarm

`simulate` remains for developing the trust layer against adversarial scenarios without any
backend at all.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from seedmesh import __version__

DEFAULT_MODEL = "JackFram/llama-160m"


# ---- simulate ---------------------------------------------------------------


def _cmd_simulate(args: argparse.Namespace) -> int:
    from seedmesh.sim import calibrate_world, run_gossip, run_swarm
    from seedmesh.sim.presets import healthy_swarm, mixed_threat_swarm, sybil_swarm
    from seedmesh.sim.report import format_gossip, format_run, format_tolerance

    presets = {
        "healthy": (healthy_swarm, "healthy swarm (honest, heterogeneous hardware)"),
        "threats": (mixed_threat_swarm, "mixed threats (lazy / byzantine / subtle / flaky)"),
        "sybil": (sybil_swarm, "sybil fleet flooding one block range"),
    }
    chosen = list(presets) if args.scenario == "all" else [args.scenario]

    for name in chosen:
        factory, title = presets[name]
        tolerance = calibrate_world(
            healthy_swarm(seed=args.seed).build(), samples=args.calibration_samples
        )
        if name == chosen[0]:
            print(format_tolerance(tolerance))
            print()
        world = factory(seed=args.seed).build()
        metrics = run_swarm(world, requests=args.requests, tolerance=tolerance, seed=args.seed)
        print(format_run(metrics, world, title))
        print()

    if args.scenario in ("all", "sybil"):
        print(format_gossip(run_gossip()))
        print()
    return 0


# ---- probe ------------------------------------------------------------------


def _cmd_probe(args: argparse.Namespace) -> int:
    from seedmesh.cli.hardware import describe_plan, detect_gpus, fetch_config, plan_blocks

    print(f"seedmesh {__version__}  |  python {sys.version.split()[0]} on {sys.platform}\n")

    if sys.platform == "win32":
        print("note: the Petals backend needs Linux or WSL2. The trust layer runs natively.\n")

    gpus = detect_gpus()
    if not gpus:
        print("No NVIDIA GPU detected.")
        print("  You can still run a client, and still donate CPU-only in a private swarm,")
        print("  but block hosting on the public swarm wants a CUDA GPU.")
        return 0

    for index, gpu in enumerate(gpus):
        print(f"GPU {index}: {gpu.name}  {gpu.total_gib:.1f} GiB "
              f"({gpu.free_bytes / 2**30:.1f} GiB free, compute {gpu.compute_capability})")

    try:
        config = fetch_config(args.model)
    except Exception as exc:
        print(f"\ncannot size {args.model}:\n  {exc}")
        return 1

    print()
    for gpu in gpus:
        for quant in (args.quant,) if args.quant else ("nf4", "int8", "none"):
            plan = plan_blocks(config, gpu, quant=quant, model_name=args.model)
            print(f"[{quant}]")
            for line in describe_plan(plan, gpu):
                print(line)
            print()
    print("Host with:")
    best = plan_blocks(config, gpus[0], quant=args.quant or "nf4", model_name=args.model)
    print(f"  seedmesh serve --model {args.model} --quant {best.quant} "
          f"--num-blocks {max(1, best.recommended_blocks)} --initial-peers <addr>")
    return 0


# ---- serve ------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    """Wrap Petals' server with auto-sizing and friendlier defaults."""
    from seedmesh.cli.hardware import detect_gpus, fetch_config, plan_blocks

    num_blocks = args.num_blocks
    if num_blocks is None:
        gpus = detect_gpus()
        if not gpus:
            print("No GPU detected and --num-blocks not given; refusing to guess.")
            print("Pass --num-blocks explicitly, or run `seedmesh probe` first.")
            return 2
        try:
            config = fetch_config(args.model)
        except Exception as exc:
            print(f"could not size automatically ({exc}); pass --num-blocks")
            return 2
        plan = plan_blocks(config, gpus[0], quant=args.quant, model_name=args.model)
        num_blocks = max(1, plan.recommended_blocks)
        print(f"auto-sized to {num_blocks} block(s) on {gpus[0].name} at {args.quant}")
        if plan.recommended_blocks <= 0:
            print("  (that GPU cannot really host a block; starting one anyway may OOM)")

    command = [
        sys.executable, "-m", "petals.cli.run_server", args.model,
        "--num_blocks", str(num_blocks),
        "--quant_type", args.quant,
    ]
    if args.initial_peers:
        command += ["--initial_peers", *args.initial_peers]
    else:
        command += ["--new_swarm"]
        print("no --initial-peers given: starting a NEW private swarm")
    if args.device:
        command += ["--device", args.device]
    if args.public_name:
        command += ["--public_name", args.public_name]
    if args.host_maddrs:
        command += ["--host_maddrs", *args.host_maddrs]
    command += args.passthrough

    print(f"$ {' '.join(command)}\n")
    import subprocess

    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 0
    except FileNotFoundError:
        print("Could not launch the Petals server. Run `seedmesh setup` first.")
        return 2


# ---- chat -------------------------------------------------------------------


def _cmd_chat(args: argparse.Namespace) -> int:
    """Minimal interactive client against a swarm."""
    try:
        import torch
        from hivemind.dht import DHT
        from transformers import AutoTokenizer

        from petals import AutoDistributedModelForCausalLM
    except ImportError as exc:
        print(f"backend not installed ({exc}). Run `seedmesh setup` first.")
        return 2

    peers = list(args.initial_peers or [])
    if not peers:
        print("--initial-peers is required (ask whoever runs the swarm for the address).")
        return 2

    print(f"connecting to {peers[0]}")
    # Build the DHT before the model: hivemind allocates shared memory on first DHT
    # construction, and doing that inside transformers' init context fails. See
    # docs/petals-port.md.
    dht = DHT(initial_peers=peers, client_mode=True, start=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model, initial_peers=peers, torch_dtype=torch.float32,
        request_timeout=args.timeout, max_retries=3,
    )
    print("connected. Type a prompt, or Ctrl-C to quit.\n")

    try:
        while True:
            try:
                prompt = input("> ").strip()
            except EOFError:
                break
            if not prompt:
                continue
            inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
            with torch.inference_mode():
                outputs = model.generate(inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            print(tokenizer.decode(outputs[0], skip_special_tokens=True) + "\n")
    except KeyboardInterrupt:
        print()
    finally:
        dht.shutdown()
    return 0


# ---- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seedmesh", description=__doc__)
    parser.add_argument("--version", action="version", version=f"seedmesh {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="install and patch the Petals backend")
    setup.add_argument("--petals-dir", default=None)
    setup.add_argument("--skip-install", action="store_true")
    setup.add_argument("--force", action="store_true", help="continue despite platform warnings")

    probe = subparsers.add_parser("probe", help="what can this machine host?")
    probe.add_argument("--model", default=DEFAULT_MODEL)
    probe.add_argument("--quant", choices=["none", "int8", "nf4"], default=None)

    serve = subparsers.add_parser("serve", help="host blocks for a swarm")
    serve.add_argument("--model", default=DEFAULT_MODEL)
    serve.add_argument("--num-blocks", type=int, default=None, help="default: auto-size")
    serve.add_argument("--quant", choices=["none", "int8", "nf4"], default="nf4")
    serve.add_argument("--initial-peers", nargs="+", default=None)
    serve.add_argument("--device", default=None)
    serve.add_argument("--public-name", default=None, help="shown on the leaderboard")
    serve.add_argument("--host-maddrs", nargs="+", default=None)
    serve.add_argument(
        "passthrough",
        nargs="*",
        metavar="-- PETALS_ARGS",
        help=(
            "anything after a bare '--' is forwarded verbatim to petals.cli.run_server, "
            "e.g. `seedmesh serve --num-blocks 4 -- --attn_cache_tokens 8192`. Petals has "
            "far more flags than this wrapper mirrors; unknown flags are NOT accepted "
            "without the separator, so typos still fail loudly."
        ),
    )

    chat = subparsers.add_parser("chat", help="talk to a swarm")
    chat.add_argument("--model", default=DEFAULT_MODEL)
    chat.add_argument("--initial-peers", nargs="+", default=None)
    chat.add_argument("--max-new-tokens", type=int, default=32)
    chat.add_argument("--timeout", type=float, default=60.0)

    simulate = subparsers.add_parser(
        "simulate", help="run trust-layer scenarios against the swarm simulator"
    )
    simulate.add_argument("--scenario", choices=["healthy", "threats", "sybil", "all"], default="all")
    simulate.add_argument("--requests", type=int, default=400)
    simulate.add_argument("--seed", type=int, default=7)
    simulate.add_argument("--calibration-samples", type=int, default=150)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "setup":
        from seedmesh.cli.setup import DEFAULT_DIR, cmd_setup

        if args.petals_dir is None:
            args.petals_dir = str(DEFAULT_DIR)
        return cmd_setup(args)

    return {
        "probe": _cmd_probe,
        "serve": _cmd_serve,
        "chat": _cmd_chat,
        "simulate": _cmd_simulate,
    }[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
