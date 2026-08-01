"""``seedmesh`` command line.

Currently exposes the simulator and the hardware probe. The client and server commands
arrive with the backend adapter; until then this CLI deliberately offers nothing that
implies a working swarm exists, because none does -- see ``docs/findings-upstream-audit.md``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from seedmesh import __version__


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

        # Calibrate on a clean copy of the swarm: thresholds must be fitted to honest
        # behaviour, and fitting them on a population containing cheats would widen the
        # tolerance until the cheats fit inside it.
        tolerance = calibrate_world(healthy_swarm(seed=args.seed).build(), samples=args.calibration_samples)
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


def _cmd_probe(args: argparse.Namespace) -> int:
    """Report what this machine could contribute, without pretending to know more."""
    import shutil

    print(f"seedmesh {__version__}")
    print(f"python  {sys.version.split()[0]} on {sys.platform}")

    if sys.platform == "win32":
        print(
            "\nnote: the Petals/hivemind backend does not run natively on Windows.\n"
            "      Use WSL2 for the server role. The trust layer in this package runs\n"
            "      natively and needs no GPU."
        )

    try:
        import subprocess

        smi = shutil.which("nvidia-smi")
        if smi:
            output = subprocess.run(
                [smi, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if output.returncode == 0 and output.stdout.strip():
                print("\ndetected NVIDIA GPUs:")
                for line in output.stdout.strip().splitlines():
                    print(f"  {line.strip()}")
                print(
                    "\n  Block-count recommendations require the backend adapter to know the\n"
                    "  model's per-block memory footprint; that lands with the adapter."
                )
            else:
                print("\nno NVIDIA GPU reported by nvidia-smi")
        else:
            print("\nnvidia-smi not found; no CUDA GPU detected")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"\nGPU probe failed: {exc}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seedmesh", description=__doc__)
    parser.add_argument("--version", action="version", version=f"seedmesh {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser(
        "simulate", help="run trust-layer scenarios against the swarm simulator"
    )
    simulate.add_argument(
        "--scenario", choices=["healthy", "threats", "sybil", "all"], default="all"
    )
    simulate.add_argument("--requests", type=int, default=400)
    simulate.add_argument("--seed", type=int, default=7)
    simulate.add_argument("--calibration-samples", type=int, default=150)
    simulate.set_defaults(func=_cmd_simulate)

    probe = subparsers.add_parser("probe", help="report local hardware and platform notes")
    probe.set_defaults(func=_cmd_probe)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
