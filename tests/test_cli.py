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
    assert set(actions[0].choices) == {"setup", "probe", "serve", "chat", "simulate"}


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


def test_simulate_still_works_without_a_backend(parser):
    args = parser.parse_args(["simulate", "--scenario", "sybil", "--requests", "10"])
    assert args.scenario == "sybil" and args.requests == 10
