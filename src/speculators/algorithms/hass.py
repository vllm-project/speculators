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


class HASSSpeculator(SpeculatorModel):
    @classmethod
    def from_config(
        cls,
        config: Union[str, Path, SpeculatorConfig],  # noqa: ARG003
        verifier: Optional[Union[str, Path, Module]] = None,  # noqa: ARG003
    ) -> "HASSSpeculator":
        """
        Create a HASSSpeculator instance from the provided config.

        :param config: The configuration for the HASSSpeculator.
        :param verifier: The verifier model to be used.
        :return: The instance of the HASSSpeculator.
        """
        # extract expected args from the config
        return cls(...)  # type: ignore[arg-type,call-arg]

    def __init__(
        self,
        **kwargs,  # noqa: ARG002
    ):
        """
        Initialize a HASS speculator instance with the provided arguments and
        hyperparameters. Specifically, it implements the following paper:
        https://arxiv.org/abs/2408.15766

        :param kwargs: Additional arguments for the HASS speculator.
            Need to define exact arguments for the implementation.
        """
        drafter = TransformerDrafter(...)  # type: ignore[arg-type,call-arg]
        proposals: dict[str, TokenProposal] = {
            "greedy": GreedyTokenProposal(...),  # type: ignore[arg-type,call-arg]
            "sampling": SamplingTokenProposal(...),  # type: ignore[arg-type,call-arg]
            "tree": TreeTokenProposal(...),  # type: ignore[arg-type,call-arg]
        }
        verifier = load_model(...)  # type: ignore[arg-type,call-arg]
        self.default_proposal_method = "tree"
        self._config = SpeculatorConfig(
            speculators_algorithm="hass",
            draft_model=drafter.config,
            proposal_methods={key: val.config for key, val in proposals.items()},
            default_proposal_method=self.default_proposal_method,
            verifier=...,  # type: ignore[arg-type]
        )
        super().__init__(
            drafter=drafter,
            verifier=verifier,
            proposals=proposals,
        )

    @property
    def config(self) -> SpeculatorConfig:
        return self._config
