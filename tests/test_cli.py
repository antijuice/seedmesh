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


def test_serve_with_zero_blocks_refuses_and_points_at_bootstrap(parser, capsys):
    from seedmesh.cli.main import _cmd_serve

    args = parser.parse_args(["serve", "--model", "JackFram/llama-160m", "--num-blocks", "0"])
    assert _cmd_serve(args) == 2
    out = capsys.readouterr().out
    assert "seedmesh bootstrap" in out, "refusing without naming the fix is not useful"


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
