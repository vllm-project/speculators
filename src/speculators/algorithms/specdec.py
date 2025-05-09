from pathlib import Path
from typing import Optional, Union

from torch.nn import Module

from speculators.algorithms.drafters import IndependentDrafter
from speculators.algorithms.proposals import (
    GreedyTokenProposal,
)
from speculators.base import SpeculatorConfig, SpeculatorModel, TokenProposal
from speculators.utils import load_model


class SpecDecSpeculator(SpeculatorModel):
    @classmethod
    def from_config(
        cls,
        config: Union[str, Path, SpeculatorConfig],
        verifier: Optional[Union[str, Path, Module]] = None,
    ) -> "SpecDecSpeculator":
        """
        Create an SpecDecSpeculator instance from the provided config.

        :param config: The configuration for the SpecDecSpeculator.
        :param verifier: The verifier model to be used.
        :return: The instance of the SpecDecSpeculator.
        """
        # extract expected args from the config
        return cls(...)

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize an SpecDec speculator instance with the provided arguments and
        hyperparameters. Specifically, it implements the following paper:
        https://arxiv.org/abs/2404.19124v1

        :param kwargs: Additional arguments for the SpecDec speculator.
            Need to define exact arguments for the implementation.
        """
        drafter = IndependentDrafter(...)
        proposals: dict[str, TokenProposal] = {
            "greedy": GreedyTokenProposal(...),
        }
        verifier = load_model(...)
        self.default_proposal_method = "greedy"
        self._config = SpeculatorConfig(
            speculators_algorithm="specdec",
            draft_model=drafter.config,
            proposal_methods={key: val.config for key, val in proposals.items()},
            default_proposal_method=self.default_proposal_method,
            verifier=...,
        )
        super().__init__(
            drafter=drafter,
            verifier=verifier,
            proposals=proposals,
        )

    @property
    def config(self) -> SpeculatorConfig:
        return self._config
