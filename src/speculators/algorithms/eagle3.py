from pathlib import Path
from typing import Optional, Union

from torch.nn import Module

from speculators.algorithms.drafters import TransformerDrafter
from speculators.algorithms.proposals import (
    GreedyTokenProposal,
    SamplingTokenProposal,
    TreeTokenProposal,
)
from speculators.base import SpeculatorConfig, SpeculatorModel, TokenProposal
from speculators.utils import load_model


class Eagle3Speculator(SpeculatorModel):
    @classmethod
    def from_config(
        cls,
        config: Union[str, Path, SpeculatorConfig],
        verifier: Optional[Union[str, Path, Module]] = None,
    ) -> "Eagle3Speculator":
        """
        Create an Eagle3Speculator instance from the provided config.

        :param config: The configuration for the Eagle3Speculator.
        :param verifier: The verifier model to be used.
        :return: The instance of the Eagle3Speculator.
        """
        # extract expected args from the config
        return cls(...)

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize an Eagle 3 speculator instance with the provided arguments and
        hyperparameters. Specifically, it implements the following paper:
        https://arxiv.org/abs/2503.01840

        :param kwargs: Additional arguments for the Eagle 3 speculator.
            Need to define exact arguments for the implementation.
        """
        drafter = TransformerDrafter(...)
        proposals: dict[str, TokenProposal] = {
            "greedy": GreedyTokenProposal(...),
            "sampling": SamplingTokenProposal(...),
            "tree": TreeTokenProposal(...),
        }
        verifier = load_model(...)
        self.default_proposal_method = "tree"
        self._config = SpeculatorConfig(
            speculators_algorithm="eagle3",
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
