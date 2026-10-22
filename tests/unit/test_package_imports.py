"""Tests for the public package import surface."""

import speculators
from speculators import models


def test_model_definitions_are_importable_from_package_root():
    for name in models.__all__:
        if name.endswith(("DraftModel", "SpeculatorConfig")):
            assert name in speculators.__all__
            assert getattr(speculators, name) is getattr(models, name)
