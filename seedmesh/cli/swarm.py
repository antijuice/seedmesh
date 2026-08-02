"""Where the swarm's bootstrap peers come from.

A volunteer should be able to type `seedmesh serve` and have it work. Making them paste four
70-character multiaddrs is the difference between "run one command" and "follow a tutorial",
and it is also error-prone in a way that fails confusingly: connect to fewer than four peers
and the host silently never learns its own public address, so it advertises nothing dialable
and every connection degrades to a relay that dies at 128 KiB.

Resolution order, first hit wins:
  1. --initial-peers on the command line
  2. --swarm-file / SEEDMESH_SWARM pointing at a JSON file
  3. the swarm.json shipped inside the package
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Below this, a NAT'd host cannot learn its own public address: go-libp2p's
# identify/obsaddr.go accepts an observed address only once `len(seenBy) >= ActivationThresh`
# and ActivationThresh is 4. Measured -- with one bootstrap the address is never learned and
# every request stalls at the relay's 128 KiB budget; with four it works directly.
MIN_BOOTSTRAP_PEERS = 4

ENV_VAR = "SEEDMESH_SWARM"
PACKAGED = Path(__file__).resolve().parent.parent / "swarm.json"


@dataclass(frozen=True)
class ModelChoice:
    """One model a swarm knows about, plus the settings it needs to run well.

    Settings live here because forgetting one is not a mild inconvenience. On a 4 GiB card
    the 8B model needs `attn_cache_tokens=4096`; at Petals' default of 16384 the same card
    fits 15 blocks instead of 21, so two volunteers cover 30 of 32 blocks and the model does
    not work at all. That is a bad thing to carry in someone's shell history.
    """

    id: str
    attn_cache_tokens: Optional[int] = None
    note: str = ""


@dataclass(frozen=True)
class Swarm:
    name: str
    model: str
    bootstrap_peers: List[str] = field(default_factory=list)
    source: str = "packaged"
    models: Dict[str, ModelChoice] = field(default_factory=dict)
    """Short aliases -> models. Everyone in a swarm must agree on the model string, so the
    alternatives belong in the file everyone shares rather than in each person's command."""

    @property
    def is_underprovisioned(self) -> bool:
        return 0 < len(self.bootstrap_peers) < MIN_BOOTSTRAP_PEERS

    def choose(self, name: Optional[str]) -> Optional[ModelChoice]:
        """Resolve an alias, or None if this is not one.

        Aliases are looked up before being treated as a model id. A Hugging Face repo id
        effectively always contains a '/', and an alias never should, so the two cannot
        collide in practice -- but the lookup is exact-match against names the swarm itself
        declared, so it cannot capture a model the swarm never mentioned.
        """
        if not name:
            return None
        return self.models.get(name)


def load_swarm(path: Optional[str] = None) -> Optional[Swarm]:
    """Read the swarm definition, or None if there isn't one."""
    candidate = path or os.environ.get(ENV_VAR)
    chosen = Path(candidate).expanduser() if candidate else PACKAGED
    if not chosen.is_file():
        if candidate:  # an explicit request that cannot be honoured should not fail silently
            raise FileNotFoundError(f"no swarm file at {chosen}")
        return None

    data = json.loads(chosen.read_text(encoding="utf-8"))
    peers = [p for p in data.get("bootstrap_peers", []) if p and not p.startswith("_")]

    # A value may be a bare model id or an object carrying the settings that model needs.
    models = {}
    for alias, value in (data.get("models") or {}).items():
        if alias.startswith("_"):
            continue  # comment key
        if isinstance(value, str):
            models[alias] = ModelChoice(id=value)
        elif isinstance(value, dict) and value.get("id"):
            models[alias] = ModelChoice(
                id=value["id"],
                attn_cache_tokens=value.get("attn_cache_tokens"),
                note=value.get("note", ""),
            )

    return Swarm(
        name=data.get("name", "unnamed"),
        model=data.get("model", ""),
        bootstrap_peers=peers,
        source=str(chosen),
        models=models,
    )


DEFAULT_MODEL = "JackFram/llama-160m"


def resolve_model(explicit: Optional[str], swarm_file: Optional[str] = None) -> str:
    """Which model to use, in the same precedence order as the peers.

    The swarm file carried a `model` field from the start and nothing read it: every command
    defaulted to a constant instead. Handing someone a swarm definition for a different model
    therefore did nothing at all, and the failure is silent in the worst way -- the model
    string sets the DHT prefix, so a peer on the wrong one joins a *different swarm*, sees
    nobody, and gets no error.

    Everyone in a swarm must agree on this, which is exactly why it belongs in the file you
    hand people rather than in each person's command line.
    """
    choice, _ = resolve_model_choice(explicit, swarm_file)
    return choice.id


def resolve_model_choice(
    explicit: Optional[str], swarm_file: Optional[str] = None
) -> tuple[ModelChoice, Optional[Swarm]]:
    """The model *and* the settings the swarm says it needs. Returns (choice, swarm)."""
    try:
        swarm = load_swarm(swarm_file)
    except FileNotFoundError:
        swarm = None

    if swarm is not None:
        alias = swarm.choose(explicit)
        if alias is not None:
            return alias, swarm
        if not explicit and swarm.model:
            # The default model may itself be one of the declared aliases' targets, in which
            # case it should come with the same settings rather than silently without them.
            for candidate in swarm.models.values():
                if candidate.id == swarm.model:
                    return candidate, swarm
            return ModelChoice(id=swarm.model), swarm

    return ModelChoice(id=explicit or DEFAULT_MODEL), swarm


def resolve_peers(explicit: Optional[List[str]], swarm_file: Optional[str] = None):
    """Return (peers, note) for the CLI to use and print.

    `note` is None when nothing needs saying -- the common path should be quiet.
    """
    if explicit:
        return list(explicit), None

    try:
        swarm = load_swarm(swarm_file)
    except FileNotFoundError as exc:
        return [], str(exc)
    if swarm is None or not swarm.bootstrap_peers:
        return [], None

    note = f"using swarm '{swarm.name}' ({len(swarm.bootstrap_peers)} bootstrap peers)"
    if swarm.is_underprovisioned:
        note += (
            f"\n  WARNING: only {len(swarm.bootstrap_peers)} peers listed; "
            f"{MIN_BOOTSTRAP_PEERS} are needed before a machine behind NAT can learn its own\n"
            f"  public address. Below that, connections fall back to a relay that is cut off\n"
            f"  after 128 KiB. See docs/NAT-AND-RELAYS.md."
        )
    return list(swarm.bootstrap_peers), note
