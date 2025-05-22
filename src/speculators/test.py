from speculators import SpeculatorModelConfig, SpeculatorsConfig, VerifierConfig
from speculators.models.independent import IndependentSpeculatorConfig
from transformers import PretrainedConfig


def test_speculator_model_config():
    config = SpeculatorModelConfig(
        speculators_model_type="test_model",
        speculators_config=SpeculatorsConfig(
            algorithm="test_algorithm",
            proposal_methods=[],
            default_proposal_method="test_proposal",
            verifier=VerifierConfig(
                name_or_path="test_verifier",
                architectures=[],
                hidden_size=768,
                intermediate_size=3072,
                vocab_size=30522,
                max_position_embeddings=512,
                bos_token_id=101,
                eos_token_id=102,
            ),
        ),
        name_or_path="name_or_path",
        model_type="test_model_type",
    )
    print(config.to_dict())
    print(config.to_diff_dict())
    print("SpeculatorModelConfig test passed successfully.")


def test_independent_speculator_config():
    pretrained_config = PretrainedConfig(
        model_type="independent",
        hidden_size=768,
        intermediate_size=3072,
        vocab_size=30522,
        max_position_embeddings=512,
        bos_token_id=101,
        eos_token_id=102,
    )

    independent_config = IndependentSpeculatorConfig.from_pretrained_config(
        pretrained_config,
        speculators_config=SpeculatorsConfig(
            algorithm="independent_algorithm",
            proposal_methods=[],
            default_proposal_method="default_method",
            verifier=VerifierConfig(
                name_or_path="independent_verifier",
                architectures=["arch1", "arch2"],
                hidden_size=768,
                intermediate_size=3072,
                vocab_size=30522,
                max_position_embeddings=512,
                bos_token_id=101,
                eos_token_id=102,
            ),
        ),
    )
    print(independent_config.to_dict())
    print(independent_config.to_diff_dict())
    print("IndependentSpeculatorConfig test passed successfully.")


if __name__ == "__main__":
    test_independent_speculator_config()
