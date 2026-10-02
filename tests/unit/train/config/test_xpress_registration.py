from speculators.model import SpeculatorModel
from speculators.models.xpress.core import XPressDraftModel
from speculators.train.config.resolution import _ALGORITHM_GROUP_USERS
from speculators.train.config.schema import TrainConfig


def test_speculator_type_xpress_resolves_to_the_model():
    """``--speculator-type xpress`` is resolved through the global registry.

    Without the ``@register`` decorator the flag parses fine and then fails at
    model construction, so pin the lookup rather than the decorator.
    """
    registry = SpeculatorModel.registry

    assert registry is not None
    assert registry["xpress"] is XPressDraftModel


def test_xpress_group_is_exposed_with_the_released_defaults():
    cfg = TrainConfig()

    assert cfg.xpress.xpress_rank == 256
    assert cfg.xpress.xpress_mlp_ratio == 2
    assert cfg.xpress.num_jacobi_passes == 6


def test_xpress_reads_the_dflash_group_and_owns_its_own():
    """XPress is-a DFlash, so a dflash-group flag must not warn under xpress.

    The group table is what decides whether a flag is silently ignored; leaving
    xpress out of the dflash entry would make every backbone knob a no-op warning
    on an xpress run.
    """
    assert "xpress" in _ALGORITHM_GROUP_USERS["dflash"]
    assert _ALGORITHM_GROUP_USERS["xpress"] == frozenset({"xpress"})
    assert "xpress" not in _ALGORITHM_GROUP_USERS["dspark"]


def test_xpress_flags_survive_a_flatten_roundtrip():
    cfg = TrainConfig()
    cfg.xpress.xpress_rank = 128

    flat = cfg.flatten()

    assert flat["xpress_rank"] == 128
    assert "num_jacobi_passes" in flat
