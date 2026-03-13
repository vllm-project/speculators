from typing import ClassVar

from pydantic import Field

from speculators.utils import PydanticClassRegistryMixin

__all__ = ["TokenProposalConfig"]


class TokenProposalConfig(PydanticClassRegistryMixin):
    """
    The base config for a token proposal method which defines how tokens are generated
    by the speculator, how they are passed to the verifier, and how they are scored
    for acceptance or rejection. All implementations of token proposal methods
    must inherit from this class, set the proposal_type to a unique value, and
    add any additional parameters needed to instantiate and implement the method.

    It uses pydantic to validate the parameters, provide default values, and
    enable automatic serialization and deserialization of the correct class
    types based on the proposal_type field.
    """

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type["TokenProposalConfig"]:
        if cls.__name__ == "TokenProposalConfig":
            return cls

        return TokenProposalConfig

    auto_package: ClassVar[str] = "speculators.proposals"
    registry_auto_discovery: ClassVar[bool] = True
    schema_discriminator: ClassVar[str] = "proposal_type"

    proposal_type: str = Field(
        description=(
            "The type of token proposal the config is for. "
            "Must be a supported proposal type from the Speculators repo."
        ),
    )
