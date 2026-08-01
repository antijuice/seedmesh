"""CLI surface.

These are cheap and worth having because the CLI is the *only* thing a volunteer touches.
A parser regression here is invisible to every other test in the suite and immediately fatal
to onboarding.
"""

from __future__ import annotations

import pytest

from seedmesh.cli.main import build_parser


@pytest.fixture
def parser():
    return build_parser()


def test_all_commands_are_registered(parser):
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions, "no subcommand group"
    assert set(actions[0].choices) == {
        "setup", "probe", "serve", "bootstrap", "chat", "simulate",
    }


def test_a_command_is_required(parser):
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_serve_defaults_to_auto_sizing(parser):
    args = parser.parse_args(["serve", "--model", "m"])
    assert args.num_blocks is None, "None means auto-size; a default number would guess"
    assert args.quant == "nf4", "NF4 is what makes ordinary hardware useful"


def test_serve_accepts_multiple_initial_peers(parser):
    args = parser.parse_args(["serve", "--initial-peers", "/ip4/1.2.3.4/tcp/1", "/ip4/5.6.7.8/tcp/2"])
    assert len(args.initial_peers) == 2


def test_serve_forwards_args_after_a_double_dash(parser):
    """An escape hatch matters: Petals has dozens of flags this wrapper does not mirror."""
    args = parser.parse_args(
        ["serve", "--num-blocks", "4", "--", "--attn_cache_tokens", "8192"]
    )
    assert args.passthrough == ["--attn_cache_tokens", "8192"]
    assert args.num_blocks == 4


def test_unknown_flags_without_the_separator_still_fail(parser):
    """The separator is the whole point: a typo must not be silently forwarded."""
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--num-blcoks", "4"])


def test_probe_defaults_to_showing_every_quantization(parser):
    """With no --quant, probe should compare options rather than pick one silently."""
    assert parser.parse_args(["probe"]).quant is None


def test_quant_choices_are_constrained(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--quant", "int3"])


def test_chat_requires_peers_at_runtime_not_parse_time(parser):
    """Parsing must succeed so the command can print a helpful message instead of a usage dump."""
    args = parser.parse_args(["chat"])
    assert args.initial_peers is None


def test_setup_flags(parser):
    args = parser.parse_args(["setup", "--skip-install", "--force"])
    assert args.skip_install and args.force
    assert args.cpu_torch is False, "auto-detect by default; the flag only forces CPU wheels"


# ---- setup step ordering ----------------------------------------------------
#
# This is a regression guard, not a unit test. `setup` once ran the codemod before
# installing anything, and the codemod verifies its symbol mapping against the *installed*
# hivemind -- so a volunteer's first command failed with a wall of "No module named
# 'hivemind'". The bug is invisible on any developer machine that already has the deps,
# which is exactly why it shipped.


class _RecordingRun:
    """Stand-in for setup.run that records commands instead of executing them."""

    def __init__(self):
        self.commands: list[list[str]] = []

    def __call__(self, command, *, check=True):
        import subprocess

        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="OK  fake\n", stderr="")

    def index_of(self, needle: str) -> int:
        for position, command in enumerate(self.commands):
            if any(needle in part for part in command):
                return position
        raise AssertionError(f"no command containing {needle!r} in {self.commands}")


@pytest.fixture
def recorded_setup(tmp_path, monkeypatch):
    from seedmesh.cli import setup as setup_module

    recorder = _RecordingRun()
    monkeypatch.setattr(setup_module, "run", recorder)
    monkeypatch.setattr(setup_module, "has_nvidia_gpu", lambda: False)
    # The real gate refuses to run on Windows, where most of this project is developed.
    # Ordering is platform-independent, so the test should be too.
    monkeypatch.setattr(setup_module, "check_platform", lambda: [])
    # Pretend the checkout is already there so the test needs no network.
    checkout = tmp_path / "petals"
    (checkout / ".git").mkdir(parents=True)
    return recorder, checkout


def _setup_args(parser, checkout, *extra):
    args = parser.parse_args(["setup", "--petals-dir", str(checkout), *extra])
    return args


def test_setup_installs_dependencies_before_running_the_codemod(parser, recorded_setup):
    from seedmesh.cli.setup import cmd_setup

    recorder, checkout = recorded_setup
    assert cmd_setup(_setup_args(parser, checkout)) == 0

    assert recorder.index_of("hivemind==1.1.12") < recorder.index_of("port_petals.py"), (
        "the codemod imports hivemind to verify its mapping, so it cannot run first"
    )


def test_setup_without_a_gpu_uses_cpu_torch_wheels(parser, recorded_setup):
    """A bootstrap peer hosts no blocks; CUDA torch is ~2.5 GiB it will never use, and
    enough to OOM the 1 GB VPS the guide recommends."""
    from seedmesh.cli.setup import CPU_TORCH_INDEX, cmd_setup

    recorder, checkout = recorded_setup
    assert cmd_setup(_setup_args(parser, checkout)) == 0

    torch_command = recorder.commands[recorder.index_of("torch")]
    assert CPU_TORCH_INDEX in torch_command


def test_setup_uses_cuda_torch_when_a_gpu_is_present(parser, recorded_setup, monkeypatch):
    from seedmesh.cli import setup as setup_module

    recorder, checkout = recorded_setup
    monkeypatch.setattr(setup_module, "has_nvidia_gpu", lambda: True)
    assert setup_module.cmd_setup(_setup_args(parser, checkout)) == 0

    torch_command = recorder.commands[recorder.index_of("torch")]
    assert setup_module.CPU_TORCH_INDEX not in torch_command


def test_setup_cpu_torch_flag_overrides_a_present_gpu(parser, recorded_setup, monkeypatch):
    from seedmesh.cli import setup as setup_module

    recorder, checkout = recorded_setup
    monkeypatch.setattr(setup_module, "has_nvidia_gpu", lambda: True)
    assert setup_module.cmd_setup(_setup_args(parser, checkout, "--cpu-torch")) == 0

    torch_command = recorder.commands[recorder.index_of("torch")]
    assert setup_module.CPU_TORCH_INDEX in torch_command


def test_simulate_still_works_without_a_backend(parser):
    args = parser.parse_args(["simulate", "--scenario", "sybil", "--requests", "10"])
    assert args.scenario == "sybil" and args.requests == 10


# ---- bootstrap --------------------------------------------------------------
#
# `serve --num-blocks 0` was documented as the way to run a bootstrap peer. It is not: with
# no blocks, Petals builds a ModuleAnnouncerThread from an empty uid list and dies on
# `module_uids[0]`. Critically it does that *after* ~1 minute of throughput measurement, so
# it announces an address and looks healthy first -- which is how it survived a 30s test.


def test_bootstrap_is_a_registered_command(parser):
    args = parser.parse_args(["bootstrap"])
    assert args.port == 31337
    assert args.announce_ip is None
    assert args.initial_peers is None


def test_bootstrap_needs_no_model(parser):
    """A DHT node relays discovery for any swarm; requiring --model would imply otherwise."""
    args = parser.parse_args(["bootstrap", "--announce-ip", "203.0.113.10"])
    assert not hasattr(args, "model")


def test_serve_with_zero_blocks_refuses_and_points_at_bootstrap(parser, capsys, monkeypatch):
    from seedmesh.cli import main as main_module

    # The platform and backend guards would otherwise short-circuit this on the dev machine
    # (Windows, no petals). The zero-block refusal is independent of both, so the test
    # should be too.
    monkeypatch.setattr(main_module, "_refuse_on_windows", lambda what: False)
    monkeypatch.setattr(main_module, "_backend_missing", lambda what: False)

    args = parser.parse_args(["serve", "--model", "JackFram/llama-160m", "--num-blocks", "0"])
    assert main_module._cmd_serve(args) == 2
    out = capsys.readouterr().out
    assert "seedmesh bootstrap" in out, "refusing without naming the fix is not useful"


def test_serve_on_windows_names_wsl_rather_than_failing_obscurely(parser, capsys, monkeypatch):
    """Without this, `serve` shells out to a petals that setup refused to install on
    Windows, and the volunteer sees ModuleNotFoundError instead of the reason."""
    from seedmesh.cli import main as main_module

    monkeypatch.setattr(main_module.sys, "platform", "win32")
    args = parser.parse_args(["serve", "--model", "JackFram/llama-160m", "--num-blocks", "4"])
    assert main_module._cmd_serve(args) == 2
    out = capsys.readouterr().out
    assert "WSL2" in out and "wsl --install" in out


def test_underscore_flag_spellings_are_accepted(parser):
    """hivemind's own output says `--initial_peers`, and that is what gets copied. The
    hyphenated form is canonical; the underscore form must not be an error wall."""
    args = parser.parse_args([
        "serve", "--model", "m", "--initial_peers", "/ip4/1.2.3.4/tcp/1/p2p/Qm",
        "--public_name", "hewitt", "--num_blocks", "4",
    ])
    assert args.initial_peers == ["/ip4/1.2.3.4/tcp/1/p2p/Qm"]
    assert args.public_name == "hewitt" and args.num_blocks == 4

    chat = parser.parse_args(["chat", "--initial_peers", "/ip4/1.2.3.4/tcp/1/p2p/Qm"])
    assert chat.initial_peers == ["/ip4/1.2.3.4/tcp/1/p2p/Qm"]


def test_bootstrap_prints_pasteable_commands(capsys):
    """The bootstrap operator should not have to hand-translate hivemind's output."""
    from seedmesh.cli.main import _print_join_instructions, build_parser

    args = build_parser().parse_args(["bootstrap", "--announce-ip", "159.89.52.179"])
    _print_join_instructions(args, "QmTG981oPjsPFX5WWNegdXmkiNUgWqjUvK9spieg4hSi1h")
    out = capsys.readouterr().out

    address = "/ip4/159.89.52.179/tcp/31337/p2p/QmTG981oPjsPFX5WWNegdXmkiNUgWqjUvK9spieg4hSi1h"
    assert address in out
    assert "seedmesh serve" in out and "seedmesh chat" in out
    assert "--initial-peers" in out, "must print the spelling this CLI actually accepts"
    assert "--initial_peers" not in out


# ---- announce address validation --------------------------------------------


def test_placeholder_ip_is_rejected():
    """The guide literally says YOUR_PUBLIC_IP, and that string gets pasted verbatim."""
    from seedmesh.cli.main import _announce_maddr

    with pytest.raises(ValueError, match="not an IP address"):
        _announce_maddr("YOUR_PUBLIC_IP", 31337)


def test_private_ip_is_rejected_with_a_reason():
    """A VPS usually only sees its private address, so announcing it strands the swarm."""
    from seedmesh.cli.main import _announce_maddr

    for private in ("10.0.0.5", "192.168.1.20", "172.16.0.1", "127.0.0.1"):
        with pytest.raises(ValueError, match="not a public address"):
            _announce_maddr(private, 31337)


def test_public_ipv4_builds_a_multiaddr():
    from seedmesh.cli.main import _announce_maddr

    assert _announce_maddr("159.89.52.179", 31337) == "/ip4/159.89.52.179/tcp/31337"


def test_ipv6_uses_the_ip6_protocol_token():
    from seedmesh.cli.main import _announce_maddr

    assert _announce_maddr("2606:4700:4700::1111", 31337) == "/ip6/2606:4700:4700::1111/tcp/31337"


# ---- shutdown noise ---------------------------------------------------------


def test_hivemind_finalizer_noise_is_suppressed_but_other_errors_are_not():
    """After a working chat, hivemind's P2P.__del__ printed three tracebacks -- the last
    thing a volunteer sees, and it reads like a crash. Suppress those, and only those."""
    import sys
    import types

    from seedmesh.cli.main import _quiet_hivemind_finalizers

    original = sys.unraisablehook
    seen = []
    try:
        sys.unraisablehook = lambda u: seen.append(u)
        _quiet_hivemind_finalizers()

        noisy = types.SimpleNamespace()
        noisy.__module__ = "hivemind.p2p.p2p_daemon"
        sys.unraisablehook(types.SimpleNamespace(object=noisy, exc_type=RuntimeError))
        assert seen == [], "hivemind p2p shutdown noise should be swallowed"

        real = types.SimpleNamespace()
        real.__module__ = "seedmesh.reputation.scorer"
        sys.unraisablehook(types.SimpleNamespace(object=real, exc_type=RuntimeError))
        assert len(seen) == 1, "a genuine unraisable error must still surface"
    finally:
        sys.unraisablehook = original


# ---- routing failures -------------------------------------------------------
#
# Measured on a real two-host swarm: a NAT'd laptop server showed ONLINE in the DHT with
# all 12 blocks, but clients got P2PDaemonError('routing: not found') for several minutes.
# The identical query then succeeded with no changes. Block records and libp2p peer-routing
# records propagate independently, so "advertised but not yet dialable" is a normal state.


def test_routing_failure_says_to_retry_rather_than_looking_like_a_crash():
    from seedmesh.cli.main import explain_inference_failure

    message = explain_inference_failure(RuntimeError("routing: not found"))
    assert "temporary" in message.lower()
    assert "again" in message.lower(), "must tell the user the actual remedy"
    assert "NAT-AND-RELAYS" in message, "and where to go if it is not temporary"


def test_missing_coverage_is_distinguished_from_a_routing_failure():
    """Different cause, different fix: nobody is serving vs. cannot reach who is."""
    from seedmesh.cli.main import explain_inference_failure

    message = explain_inference_failure(RuntimeError("No servers found for block 7"))
    assert "seedmesh serve" in message
    assert "--model" in message, "a model-string mismatch is the usual reason"


def test_an_unrecognised_error_is_reported_verbatim_not_swallowed():
    from seedmesh.cli.main import explain_inference_failure

    message = explain_inference_failure(ValueError("something else entirely"))
    assert "ValueError" in message and "something else entirely" in message


def test_chat_timeout_is_short_enough_to_retry(parser):
    """Measured, not inherited. On a relayed swarm ~1 request in 3 stalls, a stall never
    self-recovers, and a fresh attempt succeeded 14/14 immediately afterwards. Healthy
    requests finished in under 4s. So the timeout only decides how long a doomed request
    blocks: Petals' 180s default makes a one-second retry cost three minutes."""
    from seedmesh.cli.main import CHAT_REQUEST_TIMEOUT

    assert CHAT_REQUEST_TIMEOUT <= 60, "a long timeout cannot rescue a stall, only prolong it"
    assert CHAT_REQUEST_TIMEOUT >= 5, "must still clear the 3.93s slowest healthy request"
    assert parser.parse_args(["chat"]).timeout == CHAT_REQUEST_TIMEOUT


# ---- retry with a fresh session ---------------------------------------------
#
# Measured: ~1 request in 3 stalls on a relayed swarm and never recovers, but a fresh
# attempt succeeded 14/14 immediately. Retrying is the recovery mechanism, which is why the
# timeout is short rather than generous.


def test_retry_returns_the_first_success():
    from seedmesh.cli.main import generate_with_retry

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("stalled")
        return "answer"

    assert generate_with_retry(flaky, attempts=4) == "answer"
    assert len(calls) == 3, "must stop as soon as one attempt works"


def test_retry_gives_up_and_reraises_the_last_error():
    from seedmesh.cli.main import generate_with_retry

    def always_stalls():
        raise TimeoutError("stalled")

    with pytest.raises(TimeoutError):
        generate_with_retry(always_stalls, attempts=3)


def test_retry_reports_each_stall_so_a_slow_answer_is_not_silent():
    from seedmesh.cli.main import generate_with_retry

    seen = []
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise TimeoutError("stalled")
        return "ok"

    generate_with_retry(flaky, attempts=3, on_retry=lambda n, exc: seen.append(n))
    assert seen == [1], "the user should see why a response took longer than usual"


def test_a_single_attempt_still_works():
    from seedmesh.cli.main import generate_with_retry

    assert generate_with_retry(lambda: "x", attempts=1) == "x"


def test_chat_disables_the_petals_internal_retry_path(parser):
    """Petals' own retry reuses the session and rebuilds only the failed span -- the path
    that raises AssertionError('0 and N'). A fresh session has no position to disagree
    about, so recovery belongs above Petals, not inside it."""
    import inspect

    from seedmesh.cli import main as main_module

    source = inspect.getsource(main_module._cmd_chat)
    assert "max_retries=1" in source, "internal retry must stay off; it triggers the bug"
    assert "generate_with_retry" in source


# ---- wrong-environment detection --------------------------------------------
#
# `serve` and `bootstrap` re-invoke sys.executable. A `seedmesh` on PATH from a different
# environment than the one `seedmesh setup` installed into (a conda base shadowing a venv)
# produced a bare "No module named 'petals'" from a subprocess, which reads as a broken
# install rather than the wrong Python.


def test_missing_backend_names_the_interpreter_and_the_fix(capsys, monkeypatch):
    import importlib.util

    from seedmesh.cli import main as main_module

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert main_module._backend_missing("Hosting blocks") is True

    out = capsys.readouterr().out
    assert "which -a seedmesh" in out, "must show how to find the shadowing copy"
    assert "seedmesh setup" in out
    assert "Hosting blocks" in out


def test_backend_present_is_not_reported_as_missing(capsys, monkeypatch):
    import importlib.util

    from seedmesh.cli import main as main_module

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert main_module._backend_missing("Hosting blocks") is False
    assert capsys.readouterr().out == ""


# ---- warm-up window ----------------------------------------------------------
#
# A server's block records reach the DHT well before its libp2p peer-routing record, so a
# client that connects during that window sees full coverage and can dial nobody. Measured
# ~180-195s on a real swarm, repeatedly, and it applies PER CLIENT -- a second client
# joining an already-serving swarm waits again. Without handling, the volunteer's first
# prompt fails instantly and the swarm looks broken.


def test_routing_failure_is_recognised():
    from seedmesh.cli.main import is_routing_failure

    assert is_routing_failure(RuntimeError("P2PDaemonError('routing: not found')"))
    assert not is_routing_failure(TimeoutError())


def test_wait_returns_as_soon_as_the_swarm_is_routable():
    from seedmesh.cli.main import wait_until_routable

    calls = []

    def probe():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("routing: not found")

    assert wait_until_routable(probe, timeout=100, interval=0) is True
    assert len(calls) == 3


def test_wait_gives_up_rather_than_blocking_forever():
    from seedmesh.cli.main import wait_until_routable

    def never():
        raise RuntimeError("routing: not found")

    assert wait_until_routable(never, timeout=0.05, interval=0) is False


def test_a_real_fault_is_not_mistaken_for_warm_up():
    """Waiting out a genuine error would hide it for five minutes."""
    from seedmesh.cli.main import wait_until_routable

    def broken():
        raise ValueError("something actually wrong")

    with pytest.raises(ValueError):
        wait_until_routable(broken, timeout=100, interval=0)


def test_the_user_is_told_why_it_is_waiting():
    from seedmesh.cli.main import wait_until_routable

    seen = []

    def probe():
        if len(seen) < 1:
            raise RuntimeError("routing: not found")

    wait_until_routable(probe, timeout=100, interval=0,
                        notify=lambda n, remaining: seen.append(n))
    assert seen, "silently hanging for three minutes is indistinguishable from a freeze"


# ---- swarm definition --------------------------------------------------------
#
# `seedmesh serve` with no arguments has to work, or this is a tutorial rather than a
# command. The peer list also has a hard floor: go-libp2p accepts an observed public address
# only after FOUR distinct peers report it (identify/obsaddr.go, ActivationThresh = 4).
# Below four, a NAT'd host never learns its own public address, advertises nothing dialable,
# and every connection degrades to a relay that is cut off at 128 KiB. Measured, not guessed.


def test_packaged_swarm_has_at_least_four_peers():
    from seedmesh.cli.swarm import MIN_BOOTSTRAP_PEERS, load_swarm

    swarm = load_swarm()
    assert swarm is not None, "the packaged swarm.json should exist"
    assert len(swarm.bootstrap_peers) >= MIN_BOOTSTRAP_PEERS
    assert all(p.startswith("/ip4/") and "/p2p/" in p for p in swarm.bootstrap_peers)


def test_explicit_peers_win_over_the_swarm_file():
    from seedmesh.cli.swarm import resolve_peers

    peers, note = resolve_peers(["/ip4/1.2.3.4/tcp/1/p2p/Qm"], None)
    assert peers == ["/ip4/1.2.3.4/tcp/1/p2p/Qm"]
    assert note is None, "the explicit path should be quiet"


def test_swarm_file_supplies_peers_when_none_given():
    from seedmesh.cli.swarm import resolve_peers

    peers, note = resolve_peers(None, None)
    assert len(peers) >= 4
    assert "seedmesh-alpha" in note


def test_too_few_peers_warns_loudly(tmp_path):
    """Silently degrading to a 128 KiB relay is the worst possible failure here."""
    import json

    from seedmesh.cli.swarm import resolve_peers

    thin = tmp_path / "thin.json"
    thin.write_text(json.dumps({
        "name": "thin", "model": "m",
        "bootstrap_peers": ["/ip4/1.2.3.4/tcp/1/p2p/Qm"],
    }))
    peers, note = resolve_peers(None, str(thin))
    assert len(peers) == 1
    assert "WARNING" in note and "128 KiB" in note


def test_a_missing_swarm_file_is_reported_not_ignored(tmp_path):
    from seedmesh.cli.swarm import resolve_peers

    peers, note = resolve_peers(None, str(tmp_path / "nope.json"))
    assert peers == []
    assert "no swarm file" in note


def test_serve_and_chat_accept_a_swarm_file(parser):
    for sub in ("serve", "chat"):
        args = parser.parse_args([sub, "--swarm-file", "/tmp/x.json"])
        assert args.swarm_file == "/tmp/x.json"
