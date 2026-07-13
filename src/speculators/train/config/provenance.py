"""Typed provenance: which layer supplied each config value.

Provenance is computed in the *same precedence walk* that resolves values (see
:func:`compute`), so "which layer won" and "which value won" cannot drift. It
replaces the old side-channel ``provided`` set with a typed value that both the
draft-init conflict check (:func:`Provenance.provided`) and the audit sidecar
(:func:`Provenance.as_sidecar`) read.

"Explicitly provided" means *won by any non-default layer* (flag, ``--set``, or
YAML) -- a flag passed at its default value still counts, preserving the previous
draft-init behaviour.
"""

from dataclasses import dataclass
from typing import Literal

from .schema import CONFIG_DESTS

# The layers that can supply a value, highest precedence first. ``default`` is
# the implicit lowest layer (a coded field default that no source overrode).
Layer = Literal["flag", "set", "yaml", "default"]

# Highest-precedence-first order used for both the resolution walk and the
# contributor trail. Mirrors ``flag > --set > YAML > default``.
_ORDER: tuple[Layer, ...] = ("flag", "set", "yaml")


@dataclass(frozen=True)
class Provenance:
    """Per-dest record of the winning layer and the full contributor trail.

    :param winner: dest -> the highest-precedence layer that supplied it.
    :param trail: dest -> every layer that supplied a value, in precedence order
        (empty means the coded default won). The trail is what the audit sidecar
        renders so "why is ``lr`` 3e-4?" is answerable mechanically.
    """

    winner: dict[str, Layer]
    trail: dict[str, tuple[Layer, ...]]

    @classmethod
    def all_default(cls) -> "Provenance":
        """Provenance for a config built with no explicit sources: every dest
        resolved from its coded default."""
        return cls(
            winner=dict.fromkeys(CONFIG_DESTS, "default"),
            trail=dict.fromkeys(CONFIG_DESTS, ()),
        )

    def provided(self) -> set[str]:
        """The dests won by a non-default layer -- the "explicitly provided" set
        the draft-init conflict check consumes."""
        return {dest for dest, layer in self.winner.items() if layer != "default"}

    def as_sidecar(self) -> dict[str, dict[str, object]]:
        """Render to the ``run.provenance.yaml`` shape: ``{dest: {winner, trail}}``,
        sorted by dest for a stable, diffable audit artifact."""
        return {
            dest: {
                "winner": self.winner[dest],
                "trail": list(self.trail[dest]) or ["default"],
            }
            for dest in sorted(self.winner)
        }


def compute(*, flag: set[str], overrides: set[str], yaml: set[str]) -> Provenance:
    """Walk the config surface once, recording the winning layer + trail per dest.

    The winner is the first (highest-precedence) layer that supplied the dest;
    the trail is every layer that supplied it, in precedence order. Kept in lock
    step with the value-resolution source order in :mod:`.sources`.
    """
    present: dict[Layer, set[str]] = {"flag": flag, "set": overrides, "yaml": yaml}
    winner: dict[str, Layer] = {}
    trail: dict[str, tuple[Layer, ...]] = {}
    for dest in CONFIG_DESTS:
        contributors = tuple(layer for layer in _ORDER if dest in present[layer])
        trail[dest] = contributors
        winner[dest] = contributors[0] if contributors else "default"
    return Provenance(winner=winner, trail=trail)
