"""``seedmesh`` command line.

Six commands, aimed at the two things a volunteer actually does -- find out what they can
contribute, and contribute it:

    seedmesh setup      install and patch the Petals backend
    seedmesh probe      what can this machine host?
    seedmesh serve      host blocks for a swarm
    seedmesh bootstrap  run the rendezvous peer a swarm needs
    seedmesh doctor     can this machine host, and if not why
    seedmesh chat       talk to a swarm

`simulate` remains for developing the trust layer against adversarial scenarios without any
backend at all.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

from seedmesh import __version__

DEFAULT_MODEL = "JackFram/llama-160m"

# Measured on a real relayed swarm (2026-08-01, 20 + 30 request runs against a NAT'd
# laptop server) rather than inherited from Petals' 180s default:
#
#   healthy latency   median 0.88s, max 3.93s, nothing above 5s
#   stall rate        ~1 request in 3
#   stalls            NEVER self-recover -- they sit until whatever timeout is set
#   fresh attempt     succeeded 14/14 immediately after a stall
#
# Because a stalled request never completes, a long timeout rescues nothing; it only delays
# the retry that actually works. Petals' 180s turns a one-second recovery into a
# three-minute hang. 10s is ~2.5x the slowest healthy request seen here -- short, because
# retrying with a fresh session (see generate_with_retry) is the real recovery mechanism.
# Raise both with --timeout / --attempts for a large model on a slow link.
CHAT_REQUEST_TIMEOUT = 10
CHAT_ATTEMPTS = 4

# A new client must wait for the server's peer-routing record to propagate before it can
# dial anyone. Measured ~180-195s repeatedly on a real swarm, and it applies per client, not
# per server. 5 minutes leaves headroom.
CHAT_WARMUP_TIMEOUT = 300.0


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
    from seedmesh.cli.hardware import (
        UnsupportedModelError,
        check_model_supported,
        describe_plan,
        detect_gpus,
        fetch_config,
        plan_blocks,
    )

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

    # Before sizing, not after: a block plan for an architecture Petals cannot load is
    # a confidently wrong answer.
    try:
        check_model_supported(config, args.model)
    except UnsupportedModelError as exc:
        print(f"\n{exc}")
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


def _refuse_on_windows(what: str) -> bool:
    """The backend is Linux-only, and failing later is much less clear than failing here.

    Without this the command shells out to a petals that was never installed (setup refuses
    on Windows) and the volunteer gets a ModuleNotFoundError instead of the actual reason.
    """
    if sys.platform != "win32":
        return False
    print(f"{what} needs the Petals backend, which does not run natively on Windows.\n")
    print("Use WSL2:")
    print("    wsl --install -d Ubuntu          # once, then reopen a terminal")
    print("    wsl                              # drop into Ubuntu")
    print("  then clone the repo and run `seedmesh setup` *inside* Ubuntu.\n")
    print("`seedmesh probe` and `seedmesh simulate` do run natively here.")
    return True


def _backend_missing(what: str) -> bool:
    """Check the backend exists in THIS interpreter before shelling out to it.

    `serve` and `bootstrap` re-invoke `sys.executable`, so a `seedmesh` on PATH from a
    different environment than the one `seedmesh setup` installed into produces
    "No module named 'petals'" from a subprocess -- which reads like a broken install rather
    than the wrong Python. A conda base env shadowing a venv is the usual cause.

    find_spec rather than import: no need to pay for loading torch just to answer this.
    """
    import importlib.util

    try:
        if importlib.util.find_spec("petals") is not None:
            return False
    except (ImportError, ValueError):
        pass

    print(f"{what} needs the Petals backend, which is not installed in this Python:")
    print(f"    {sys.executable}\n")
    print("`seedmesh` is probably being run from a different environment than the one")
    print("`seedmesh setup` installed into. Check which copies exist:")
    print("    which -a seedmesh\n")
    print("Then either run the one whose environment has the backend, e.g.")
    print("    ~/.venv/bin/seedmesh ...")
    print("or install the backend into this environment:")
    print("    seedmesh setup")
    return True


def _cmd_serve(args: argparse.Namespace) -> int:
    """Wrap Petals' server with auto-sizing and friendlier defaults."""
    if _refuse_on_windows("Hosting blocks"):
        return 2
    if _backend_missing("Hosting blocks"):
        return 2

    from seedmesh.cli.hardware import (
        UnsupportedModelError,
        check_model_supported,
        detect_gpus,
        fetch_config,
        plan_blocks,
    )

    # Architecture check before launching anything. Petals downloads config.json and only
    # then raises "Petals does not support model type qwen3" from the bottom of a traceback.
    # One HTTP request turns that into a sentence. A *failed* fetch is deliberately not
    # fatal -- the server fetches too and will report the real problem itself.
    config = None
    try:
        config = fetch_config(args.model)
    except Exception as exc:
        print(f"note: could not pre-check {args.model} ({exc}); letting the server try\n")
    else:
        try:
            check_model_supported(config, args.model)
        except UnsupportedModelError as exc:
            print(exc)
            return 2

    num_blocks = args.num_blocks
    if num_blocks == 0:
        # Petals accepts --num_blocks 0, measures throughput for ~a minute, and only then
        # dies in ModuleAnnouncerThread on `module_uids[0]` with an empty list. Looking
        # healthy for a minute first is what makes this worth catching here.
        print("A bootstrap peer is not a zero-block server -- Petals crashes on an empty")
        print("block list once it finishes measuring throughput. Use:\n")
        print("  seedmesh bootstrap --port 31337 --announce-ip <your public IP>\n")
        print("It needs no --model: a DHT node relays discovery for any swarm.")
        return 2

    if num_blocks is None:
        gpus = detect_gpus()
        if not gpus:
            print("No GPU detected and --num-blocks not given; refusing to guess.")
            print("Pass --num-blocks explicitly, or run `seedmesh probe` first.")
            return 2
        if config is None:
            print("could not fetch the model config, so auto-sizing is impossible; "
                  "pass --num-blocks")
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
    from seedmesh.cli.swarm import resolve_peers

    peers, note = resolve_peers(args.initial_peers, args.swarm_file)
    if note:
        print(note)
    if peers:
        command += ["--initial_peers", *peers]
    else:
        command += ["--new_swarm"]
        print("no bootstrap peers found: starting a NEW private swarm")
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


# ---- bootstrap --------------------------------------------------------------


def _announce_maddr(ip: str, port: int) -> str:
    """Validate an announce IP before it reaches libp2p.

    Placeholder IPs are a real failure mode -- the guide says YOUR_PUBLIC_IP and that string
    gets pasted verbatim. libp2p's own complaint about it is not obviously about the IP.
    """
    import ipaddress

    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        raise ValueError(
            f"{ip!r} is not an IP address. Pass the VPS's actual public IPv4, e.g. "
            f"--announce-ip 203.0.113.10 (find it with `curl -4 ifconfig.me`)."
        ) from None
    if not address.is_global:
        raise ValueError(
            f"{ip} is not a public address, so other peers cannot dial it.\n"
            f"  A VPS usually sees only its private address locally -- take the public one "
            f"from your provider's console, or `curl -4 ifconfig.me`."
        )
    family = "ip6" if address.version == 6 else "ip4"
    return f"/{family}/{ip}/tcp/{port}"


def _cmd_bootstrap(args: argparse.Namespace) -> int:
    """Run a DHT-only rendezvous peer.

    NOT `serve --num-blocks 0`. A Petals server with zero blocks builds a ModuleAnnouncer
    from an empty uid list and dies on `module_uids[0]` -- but only after ~1 minute of
    throughput measurement, so it looks like it started fine first. A bootstrap peer is a
    DHT node, which is a different program (petals.cli.run_dht).
    """
    if _backend_missing("Running a bootstrap peer"):
        return 2

    import subprocess
    from pathlib import Path

    host_maddrs = args.host_maddrs or [f"/ip4/0.0.0.0/tcp/{args.port}"]

    # NOTE: raising go-libp2p's 128 KiB circuit-relay budget was tried here and does not
    # work -- p2pd accepts -relayDataLimit/-relayTimeLimit but never applies them to the
    # relay service, so the service keeps DefaultResources(). Measured three times against a
    # real relay: byte-for-byte identical stall patterns with and without the flags. Its own
    # help already advertises a 4 GiB default while the observed budget is 128 KiB, which is
    # the tell. Don't re-attempt it; see docs/NAT-AND-RELAYS.md.
    command = [sys.executable, "-m", "petals.cli.run_dht", "--host_maddrs", *host_maddrs]

    if args.announce_ip:
        try:
            command += ["--announce_maddrs", _announce_maddr(args.announce_ip, args.port)]
        except ValueError as exc:
            print(exc)
            return 2
    else:
        print("no --announce-ip given: the peer will advertise whatever address it sees "
              "locally,\n  which on most VPSs is a private one nobody can dial.\n")

    from seedmesh.cli.swarm import resolve_peers

    peers, note = resolve_peers(args.initial_peers, args.swarm_file)
    if note:
        print(note, flush=True)
    if peers:
        # Without this a new bootstrap starts its OWN isolated DHT and everything looks
        # healthy while nobody can see anyone.
        command += ["--initial_peers", *peers]

    # A stable peer id matters more here than anywhere else: the address is the thing every
    # volunteer has pasted into their own command line.
    identity = Path(args.identity_path).expanduser()
    identity.parent.mkdir(parents=True, exist_ok=True)
    command += ["--identity_path", str(identity)]
    if not identity.exists():
        print(f"creating a new peer identity at {identity}")
        print("  Keep this file. Losing it changes the bootstrap address and everyone "
              "has to re-paste it.\n")

    command += args.passthrough

    print(f"$ {' '.join(command)}\n", flush=True)

    # Unbuffered on both sides. Reading the child through a pipe makes *both* stdouts
    # block-buffered when they are not a terminal -- under systemd that means journalctl
    # shows nothing for minutes and the address never appears when you need it.
    import os

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
    except FileNotFoundError:
        print("Could not launch the DHT node. Run `seedmesh setup` first.", flush=True)
        return 2

    # Echo the daemon's output, and the first time a peer id appears, print the commands a
    # friend should actually run. hivemind prints its own `--initial_peers` line, and that
    # underscore spelling does not match this CLI -- copying what is on screen produced
    # "unrecognized arguments: --initial_peers". Printing the real thing beats aliasing it.
    announced = False
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if announced:
                continue
            match = re.search(r"/p2p/([A-Za-z0-9]+)", line)
            if match:
                announced = True
                _print_join_instructions(args, match.group(1))
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        return 0


def _print_join_instructions(args: argparse.Namespace, peer_id: str) -> None:
    if args.announce_ip:
        address = f"{_announce_maddr(args.announce_ip, args.port)}/p2p/{peer_id}"
    else:
        address = f"/ip4/<this machine's public IP>/tcp/{args.port}/p2p/{peer_id}"

    # One line per command, deliberately: a backslash-continued command breaks the moment
    # someone's terminal or chat client reflows it, and these get pasted through both.
    rule = "=" * 78
    banner = f"""
{rule}
BOOTSTRAP READY -- send your friends the commands below, verbatim.
On Windows they must run these inside WSL2; the backend is Linux-only.

Bootstrap address:
  {address}

To donate compute (host blocks):
  seedmesh serve --model {DEFAULT_MODEL} --initial-peers {address} --public-name "their-name"

To use the swarm (send prompts):
  seedmesh chat --model {DEFAULT_MODEL} --initial-peers {address}

Everyone must use the SAME --model string -- it sets the DHT prefix, so a mismatch
puts people in separate swarms with no error message.
{rule}
"""
    print(banner, flush=True)


# ---- chat -------------------------------------------------------------------


def _quiet_hivemind_finalizers() -> None:
    """Suppress the three tracebacks hivemind prints on the way out.

    `P2P.__del__` -> `cancel_task_if_running` -> `asyncio.get_event_loop()` raises under
    uvloop once the loop is gone, and CPython reports that as an "Exception ignored in"
    traceback. It happens *after* an explicit `dht.shutdown()`, so nothing is leaked -- but
    it is the last thing on screen after a working session and it reads like a crash.

    Scoped to hivemind.p2p so a real unraisable error elsewhere still surfaces.
    """
    import sys

    previous = sys.unraisablehook

    def hook(unraisable):
        owner = getattr(unraisable.object, "__module__", "") or ""
        if owner.startswith("hivemind.p2p"):
            return
        previous(unraisable)

    sys.unraisablehook = hook


def is_routing_failure(exc: BaseException) -> bool:
    """`routing: not found` -- the DHT knows the server's blocks but not yet its address."""
    return "routing: not found" in str(exc)


def wait_until_routable(probe, timeout=CHAT_WARMUP_TIMEOUT, interval=15.0, notify=None):
    """Block until the swarm can actually be reached, or give up.

    A server's *block* records reach the DHT long before its libp2p *peer routing* record
    does, so a client that connects during that window sees full block coverage and cannot
    dial anyone. Measured repeatedly on a real swarm: ~180-195s, and it applies per client,
    not per server -- a second client joining an already-serving swarm waits again.

    Without this the volunteer's first prompt fails instantly and the swarm looks broken at
    exactly the moment they are deciding whether this thing works.

    Returns True once routable, False on timeout. Any error that is not a routing failure
    propagates -- a real fault should not be mistaken for warm-up.
    """
    import time

    deadline = time.monotonic() + timeout
    attempt = 0
    while True:
        try:
            probe()
            return True
        except Exception as exc:
            if not is_routing_failure(exc):
                raise
            if time.monotonic() + interval >= deadline:
                return False
            attempt += 1
            if notify is not None:
                notify(attempt, max(0.0, deadline - time.monotonic()))
            time.sleep(interval)


def generate_with_retry(generate, attempts=CHAT_ATTEMPTS, on_retry=None):
    """Run `generate`, retrying on a stalled request with a brand-new inference session.

    Measured on a relayed swarm: ~1 request in 3 stalls forever, but a *fresh* attempt
    succeeded 14/14 immediately afterwards, in ~1.2s. So the failure is per-session, not
    per-peer -- retrying is nearly free and nearly always works.

    This deliberately sits above Petals rather than inside it. Petals' own retry reuses the
    inference session and rebuilds only the failed span, which is exactly the path that
    raises AssertionError('0 and N') -- it restarts a server session at position 0 while the
    client is mid-sequence. A new session has no position to disagree about, so retrying at
    this level sidesteps that bug instead of triggering it. `max_retries=1` on the client
    keeps Petals from taking the broken path first.

    Independent attempts at a ~1/3 stall rate: 4 attempts leaves roughly a 1.2% chance of
    giving up, against 33% with none.
    """
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return generate()
        except Exception as exc:  # noqa: BLE001 -- reported below if every attempt fails
            last_error = exc
            if attempt < attempts and on_retry is not None:
                on_retry(attempt, exc)
    raise last_error


def explain_inference_failure(exc: BaseException) -> str:
    """Turn a routing failure into something a volunteer can act on.

    Measured 2026-08-01 on a real two-host swarm: a laptop server behind NAT announced its
    blocks and showed ONLINE in the DHT, but clients got
    `P2PDaemonError('routing: not found')` for several minutes afterwards. Block records and
    libp2p peer-routing records propagate independently, so there is a window where the
    swarm advertises a server nobody can dial yet. The same query succeeded ~5 minutes
    later with no changes. Retrying is genuinely the fix, so say so instead of unwinding a
    60-frame traceback that looks like a crash.
    """
    detail = str(exc)
    if "routing: not found" in detail or "no route" in detail.lower():
        return (
            "\nFound a server in the DHT, but could not open a connection to it yet.\n"
            "  This is usually normal and usually temporary. A server's block records\n"
            "  appear in the DHT before its address does, so there is a window -- a few\n"
            "  minutes after it starts -- where it looks available but cannot be dialled.\n"
            "  Wait a minute and send the prompt again.\n"
            "  If it persists past ~10 minutes, the server is behind NAT and no relay path\n"
            "  formed; see docs/NAT-AND-RELAYS.md.\n"
        )
    if "no servers" in detail.lower() or "not enough" in detail.lower():
        return (
            "\nNo server covers part of this model.\n"
            "  Every block must be hosted by someone online. Check `seedmesh serve` is\n"
            "  running somewhere, on the SAME --model string.\n"
        )
    return f"\ninference failed: {type(exc).__name__}: {detail}\n"


def _cmd_chat(args: argparse.Namespace) -> int:
    """Minimal interactive client against a swarm."""
    if _refuse_on_windows("Talking to a swarm"):
        return 2

    if _backend_missing("Talking to a swarm"):
        return 2

    try:
        import torch
        from hivemind.dht import DHT
        from transformers import AutoTokenizer

        from petals import AutoDistributedModelForCausalLM
    except ImportError as exc:
        print(f"backend present but not importable ({exc}). Re-run `seedmesh setup`.")
        return 2

    from seedmesh.cli.swarm import resolve_peers

    peers, note = resolve_peers(args.initial_peers, args.swarm_file)
    if note:
        print(note)
    if not peers:
        print("no bootstrap peers. Either pass --initial-peers, or point --swarm-file at a")
        print("swarm definition. Ask whoever runs the swarm for its addresses.")
        return 2

    _quiet_hivemind_finalizers()

    print(f"connecting to {peers[0]}")
    # Build the DHT before the model: hivemind allocates shared memory on first DHT
    # construction, and doing that inside transformers' init context fails. See
    # docs/petals-port.md.
    dht = DHT(initial_peers=peers, client_mode=True, start=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # max_retries is bounded on purpose -- Petals defaults to retrying forever, which in a
    # small swarm means hanging instead of reporting. request_timeout is NOT tightened:
    # see CHAT_REQUEST_TIMEOUT.
    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model, initial_peers=peers, torch_dtype=torch.float32,
        request_timeout=args.timeout, max_retries=1,
    )
    # Wait out the warm-up window here rather than letting it eat the first prompt.
    probe_ids = tokenizer("hello", return_tensors="pt")["input_ids"]

    def probe():
        with torch.inference_mode():
            model.generate(probe_ids, max_new_tokens=1, do_sample=False)

    def waiting(attempt, remaining):
        if attempt == 1:
            print("\nA server is announced but not dialable yet -- normal for a few minutes")
            print("after one starts, because its blocks reach the DHT before its address")
            print(f"does. Waiting up to {CHAT_WARMUP_TIMEOUT:.0f}s...", flush=True)
        else:
            print(f"  still waiting ({remaining:.0f}s left)", flush=True)

    try:
        if not wait_until_routable(probe, notify=waiting):
            print(explain_inference_failure(RuntimeError("routing: not found")))
            dht.shutdown()
            return 1
    except Exception as exc:
        print(explain_inference_failure(exc))
        dht.shutdown()
        return 1

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
            def run():
                with torch.inference_mode():
                    return model.generate(
                        inputs, max_new_tokens=args.max_new_tokens, do_sample=False
                    )

            def note_retry(attempt, exc):
                print(f"  (attempt {attempt} stalled, retrying)", flush=True)

            try:
                outputs = generate_with_retry(run, args.attempts, note_retry)
            except Exception as exc:
                print(explain_inference_failure(exc))
                continue
            print(tokenizer.decode(outputs[0], skip_special_tokens=True) + "\n")
    except KeyboardInterrupt:
        print()
    finally:
        dht.shutdown()
    return 0


# ---- parser -----------------------------------------------------------------
#
# Every multi-word flag also accepts its underscore spelling. Petals and hivemind print
# their own flag names in log output -- run_dht literally says "use --initial_peers ..." --
# so copying what is on screen otherwise hits "unrecognized arguments".


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seedmesh", description=__doc__)
    parser.add_argument("--version", action="version", version=f"seedmesh {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="install and patch the Petals backend")
    setup.add_argument("--petals-dir", "--petals_dir", default=None)
    setup.add_argument("--skip-install", "--skip_install", action="store_true")
    setup.add_argument("--force", action="store_true", help="continue despite platform warnings")
    setup.add_argument(
        "--cpu-torch", "--cpu_torch",
        action="store_true",
        help="install CPU-only torch even if a GPU is present (default: auto-detect)",
    )

    probe = subparsers.add_parser("probe", help="what can this machine host?")
    probe.add_argument("--model", default=DEFAULT_MODEL)
    probe.add_argument("--quant", choices=["none", "int8", "nf4"], default=None)

    serve = subparsers.add_parser("serve", help="host blocks for a swarm")
    serve.add_argument("--model", default=DEFAULT_MODEL)
    serve.add_argument("--num-blocks", "--num_blocks", type=int, default=None,
                       help="default: auto-size")
    serve.add_argument("--quant", choices=["none", "int8", "nf4"], default="nf4")
    serve.add_argument("--initial-peers", "--initial_peers", nargs="+", default=None)
    serve.add_argument("--swarm-file", "--swarm_file", default=None,
                      help="JSON swarm definition (default: the packaged one)")
    serve.add_argument("--device", default=None)
    serve.add_argument("--public-name", "--public_name", default=None,
                       help="shown on the leaderboard")
    serve.add_argument("--host-maddrs", "--host_maddrs", nargs="+", default=None)
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

    bootstrap = subparsers.add_parser(
        "bootstrap",
        help="run the always-on rendezvous peer a swarm needs (no GPU, no model)",
    )
    bootstrap.add_argument("--port", type=int, default=31337)
    bootstrap.add_argument(
        "--announce-ip", "--announce_ip",
        default=None,
        help="the machine's PUBLIC IPv4 -- what it tells other peers to dial. A VPS "
             "usually only sees its private address, so without this nobody can reach it.",
    )
    bootstrap.add_argument(
        "--initial-peers", "--initial_peers", nargs="+", default=None,
        help="join an existing DHT (for running a second bootstrap); omit to start a new one",
    )
    bootstrap.add_argument("--swarm-file", "--swarm_file", default=None,
                           help="JSON swarm definition to join (default: the packaged one)")
    bootstrap.add_argument(
        "--identity-path", "--identity_path",
        default=str(Path.home() / ".seedmesh" / "bootstrap.key"),
        help="keeps the peer id stable across restarts; losing it changes the address",
    )
    bootstrap.add_argument("--host-maddrs", "--host_maddrs", nargs="+", default=None,
                           help="advanced; overrides --port")
    bootstrap.add_argument(
        "passthrough", nargs="*", metavar="-- DHT_ARGS",
        help="anything after a bare '--' goes to petals.cli.run_dht verbatim",
    )

    doctor = subparsers.add_parser(
        "doctor", help="can this machine host? and if not, exactly why"
    )
    doctor.add_argument("--initial-peers", "--initial_peers", nargs="+", default=None)
    doctor.add_argument("--swarm-file", "--swarm_file", default=None)
    doctor.add_argument("--port", type=int, default=31338,
                        help="port to listen on while testing; use the one you would serve on")
    doctor.add_argument("--timeout", type=float, default=120.0,
                        help="how long to wait for a public address to be observed")

    chat = subparsers.add_parser("chat", help="talk to a swarm")
    chat.add_argument("--model", default=DEFAULT_MODEL)
    chat.add_argument("--initial-peers", "--initial_peers", nargs="+", default=None)
    chat.add_argument("--swarm-file", "--swarm_file", default=None,
                      help="JSON swarm definition (default: the packaged one)")
    chat.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=32)
    chat.add_argument("--timeout", type=float, default=CHAT_REQUEST_TIMEOUT,
                      help="per-attempt timeout in seconds")
    chat.add_argument("--attempts", type=int, default=CHAT_ATTEMPTS,
                      help="retries with a fresh session when a request stalls")

    simulate = subparsers.add_parser(
        "simulate", help="run trust-layer scenarios against the swarm simulator"
    )
    simulate.add_argument(
        "--scenario", choices=["healthy", "threats", "sybil", "all"], default="all"
    )
    simulate.add_argument("--requests", type=int, default=400)
    simulate.add_argument("--seed", type=int, default=7)
    simulate.add_argument("--calibration-samples", "--calibration_samples",
                          type=int, default=150)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "setup":
        from seedmesh.cli.setup import DEFAULT_DIR, cmd_setup

        if args.petals_dir is None:
            args.petals_dir = str(DEFAULT_DIR)
        return cmd_setup(args)

    if args.command == "doctor":
        from seedmesh.cli.doctor import cmd_doctor

        return cmd_doctor(args)

    return {
        "probe": _cmd_probe,
        "serve": _cmd_serve,
        "bootstrap": _cmd_bootstrap,
        "chat": _cmd_chat,
        "simulate": _cmd_simulate,
    }[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
