"""
Unit tests for the config module in the Speculators library.
"""

import json
import tempfile
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from transformers import PretrainedConfig

from speculators import (
    SpeculatorModelConfig,
    SpeculatorsConfig,
    TokenProposalConfig,
    VerifierConfig,
    reload_and_populate_configs,
)


class TokenProposalConfigTest(TokenProposalConfig):
    proposal_type: Literal["test_proposal"] = "test_proposal"
    test_field: int = 123


class SpeculatorModelConfigTest(SpeculatorModelConfig):
    speculators_model_type: Literal["test_model"] = "test_model"
    test_field: int = 456


class TestTokenProposalConfig:
    @pytest.fixture(
        params=[
            {"proposal_type": "test_proposal"},
            {"proposal_type": "test_proposal", "test_field": 789},
        ],
        ids=["basic_config", "custom_field"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for TokenProposalConfig."""
        constructor_args = request.param
        instance = TokenProposalConfigTest(**constructor_args)
        return instance, constructor_args

    def setup_method(self):
        self._original_registry = (
            TokenProposalConfig.registry.copy()  # type: ignore[misc]
            if TokenProposalConfig.registry  # type: ignore[misc]
            else {}
        )
        TokenProposalConfig.register_decorator(TokenProposalConfigTest, "test_proposal")

    def teardown_method(self):
        TokenProposalConfig.registry = self._original_registry  # type: ignore[misc]
        TokenProposalConfig.reload_schema()

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test TokenProposalConfig inheritance and class variables."""
        from speculators.utils import PydanticClassRegistryMixin

        assert issubclass(TokenProposalConfig, PydanticClassRegistryMixin)
        assert hasattr(TokenProposalConfig, "auto_package")
        assert TokenProposalConfig.auto_package == "speculators.proposals"
        assert hasattr(TokenProposalConfig, "registry_auto_discovery")
        assert TokenProposalConfig.registry_auto_discovery is True
        assert hasattr(TokenProposalConfig, "schema_discriminator")
        assert TokenProposalConfig.schema_discriminator == "proposal_type"
        assert hasattr(TokenProposalConfig, "__pydantic_schema_base_type__")

    @pytest.mark.smoke
    def test_initialization(
        self, valid_instances: tuple[TokenProposalConfigTest, dict]
    ):
        """Test TokenProposalConfig initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, TokenProposalConfig)
        assert instance.proposal_type == constructor_args["proposal_type"]
        if "test_field" in constructor_args:
            assert instance.test_field == constructor_args["test_field"]
        else:
            assert instance.test_field == 123  # default value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("proposal_type", None),
            ("proposal_type", 123),
            ("proposal_type", []),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test TokenProposalConfig with invalid field values."""
        data = {field: value}
        with pytest.raises(ValidationError):
            TokenProposalConfig(**data)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test TokenProposalConfig initialization without required field."""
        with pytest.raises(ValidationError) as exc_info:
            TokenProposalConfig()  # type: ignore[call-arg]

        assert "proposal_type" in str(exc_info.value)

    @pytest.mark.smoke
    def test_subclass_initialization(self):
        """Test direct subclass initialization."""
        config = TokenProposalConfigTest()
        assert config.proposal_type == "test_proposal"
        assert config.test_field == 123

    @pytest.mark.smoke
    def test_auto_registry(self):
        """Test TokenProposalConfig registry population."""
        classes = TokenProposalConfig.registered_classes()
        class_names = [cls.__name__ for cls in classes]
        assert len(class_names) > 0
        assert "DynamicTreeTokenProposalConfig" in class_names
        assert "GreedyTokenProposalConfig" in class_names
        assert "SamplingTokenProposalConfig" in class_names
        assert "StaticTreeTokenProposalConfig" in class_names

    @pytest.mark.smoke
    def test_pydantic_schema_base_type(self):
        """Test __pydantic_schema_base_type__ method."""
        base_type = TokenProposalConfig.__pydantic_schema_base_type__()
        assert base_type is TokenProposalConfig

        subclass_type = TokenProposalConfigTest.__pydantic_schema_base_type__()
        assert subclass_type is TokenProposalConfig

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test TokenProposalConfig serialization and deserialization."""
        instance, constructor_args = valid_instances

        config_dict = instance.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["proposal_type"] == constructor_args["proposal_type"]
        if "test_field" in constructor_args:
            assert config_dict["test_field"] == constructor_args["test_field"]

        recreated_config: TokenProposalConfigTest = (
            TokenProposalConfig.model_validate(config_dict)  # type: ignore[assignment]
        )
        assert recreated_config.proposal_type == instance.proposal_type
        assert recreated_config.test_field == instance.test_field


class TestVerifierConfig:
    @pytest.fixture(
        params=[
            {"name_or_path": "test/verifier", "architectures": ["TestModel"]},
            {
                "name_or_path": "another/model",
                "architectures": ["AnotherModel", "SecondModel"],
            },
            {"name_or_path": None, "architectures": []},
        ],
        ids=["basic_config", "multi_architectures", "minimal_config"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for VerifierConfig."""
        constructor_args = request.param
        instance = VerifierConfig(**constructor_args)
        return instance, constructor_args

    @pytest.fixture
    def mock_pretrained_config(self):
        config = MagicMock(spec=PretrainedConfig)
        config.name_or_path = "test/verifier"
        config.architectures = ["TestModel"]
        config.to_dict.return_value = {
            "architectures": ["TestModel"],
            "hidden_size": 768,
            "intermediate_size": 3072,
            "vocab_size": 50000,
            "max_position_embeddings": 512,
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
        return config

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test VerifierConfig inheritance and methods."""
        from pydantic import BaseModel

        assert issubclass(VerifierConfig, BaseModel)
        assert hasattr(VerifierConfig, "from_pretrained")
        assert callable(VerifierConfig.from_pretrained)

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test VerifierConfig initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, VerifierConfig)
        assert instance.name_or_path == constructor_args["name_or_path"]
        assert instance.architectures == constructor_args["architectures"]

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("name_or_path", 123),
            ("architectures", "not_a_list"),
            ("architectures", [123, 456]),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test VerifierConfig with invalid field values."""
        data = {"name_or_path": "test/verifier", "architectures": ["TestModel"]}
        data[field] = value
        with pytest.raises(ValidationError):
            VerifierConfig(**data)  # type: ignore[arg-type]

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test VerifierConfig initialization without required fields."""
        with pytest.raises(ValidationError) as exc_info:
            VerifierConfig()  # type: ignore[call-arg]

        error_str = str(exc_info.value)
        assert "name_or_path" in error_str
        assert "architectures" in error_str

    @pytest.mark.smoke
    def test_from_pretrained_with_config(self, mock_pretrained_config):
        """Test VerifierConfig.from_pretrained with PretrainedConfig."""
        config = VerifierConfig.from_pretrained(mock_pretrained_config)

        assert config.name_or_path == "test/verifier"
        assert config.architectures == ["TestModel"]

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("config_input", "expected_name"),
        [
            ({"name_or_path": "dict/path"}, "dict/path"),
            ({"_name_or_path": "private/path"}, "private/path"),
        ],
    )
    def test_from_pretrained_edge_cases(self, config_input, expected_name):
        """Test VerifierConfig.from_pretrained with edge cases."""
        config = VerifierConfig.from_pretrained(config_input)
        assert config.name_or_path == expected_name
        assert config.architectures == []

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test VerifierConfig serialization and deserialization."""
        instance, constructor_args = valid_instances

        config_dict = instance.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["name_or_path"] == constructor_args["name_or_path"]
        assert config_dict["architectures"] == constructor_args["architectures"]

        recreated_config = VerifierConfig.model_validate(config_dict)
        assert recreated_config.name_or_path == instance.name_or_path
        assert recreated_config.architectures == instance.architectures


class TestSpeculatorsConfig:
    @pytest.fixture
    def sample_token_proposal_config(self):
        return TokenProposalConfigTest()

    @pytest.fixture
    def sample_verifier_config(self):
        return VerifierConfig(
            name_or_path="test/verifier",
            architectures=["TestModel"],
        )

    @pytest.fixture(
        params=[
            {
                "algorithm": "test_algorithm",
                "default_proposal_method": "test_proposal",
            },
            {
                "algorithm": "another_algorithm",
                "default_proposal_method": "test_proposal",
            },
        ],
        ids=["basic_config", "alternative_algorithm"],
    )
    def valid_instances(
        self, request, sample_token_proposal_config, sample_verifier_config
    ):
        """Fixture providing test data for SpeculatorsConfig."""
        constructor_args = request.param
        constructor_args.update(
            {
                "proposal_methods": [sample_token_proposal_config],
                "verifier": sample_verifier_config,
            }
        )
        instance = SpeculatorsConfig(**constructor_args)
        return instance, constructor_args

    def setup_method(self):
        self._original_registry = (
            TokenProposalConfig.registry.copy()  # type: ignore[misc]
            if TokenProposalConfig.registry  # type: ignore[misc]
            else {}
        )
        TokenProposalConfig.register_decorator(TokenProposalConfigTest, "test_proposal")
        SpeculatorsConfig.reload_schema()

    def teardown_method(self):
        TokenProposalConfig.registry = self._original_registry  # type: ignore[misc]
        TokenProposalConfig.reload_schema()

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test SpeculatorsConfig inheritance and methods."""
        from speculators.utils import ReloadableBaseModel

        assert issubclass(SpeculatorsConfig, ReloadableBaseModel)
        assert hasattr(SpeculatorsConfig, "reload_schema")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test SpeculatorsConfig initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, SpeculatorsConfig)
        assert instance.algorithm == constructor_args["algorithm"]
        assert len(instance.proposal_methods) == 1
        assert instance.proposal_methods[0].proposal_type == "test_proposal"
        assert (
            instance.default_proposal_method
            == constructor_args["default_proposal_method"]
        )
        assert instance.verifier.name_or_path == "test/verifier"

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("algorithm", None),
            ("algorithm", 123),
            ("proposal_methods", "not_a_list"),
            ("default_proposal_method", None),
            ("default_proposal_method", 456),
            ("verifier", "not_a_config"),
        ],
    )
    def test_invalid_initialization_values(
        self, field, value, sample_token_proposal_config, sample_verifier_config
    ):
        """Test SpeculatorsConfig with invalid field values."""
        data = {
            "algorithm": "test_algorithm",
            "proposal_methods": [sample_token_proposal_config],
            "default_proposal_method": "test_proposal",
            "verifier": sample_verifier_config,
        }
        data[field] = value
        with pytest.raises(ValidationError):
            SpeculatorsConfig(**data)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test SpeculatorsConfig initialization without required fields."""
        with pytest.raises(ValidationError) as exc_info:
            SpeculatorsConfig()  # type: ignore[call-arg]

        error_str = str(exc_info.value)
        assert "algorithm" in error_str
        assert "proposal_methods" in error_str
        assert "default_proposal_method" in error_str
        assert "verifier" in error_str

    @pytest.mark.smoke
    def test_empty_proposal_methods_valid(self, sample_verifier_config):
        """Test that empty proposal_methods list is valid."""
        config = SpeculatorsConfig(
            algorithm="test_algorithm",
            proposal_methods=[],  # Empty list should be valid
            default_proposal_method="test_proposal",
            verifier=sample_verifier_config,
        )
        assert config.algorithm == "test_algorithm"
        assert len(config.proposal_methods) == 0
        assert config.default_proposal_method == "test_proposal"

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test SpeculatorsConfig serialization and deserialization."""
        instance, constructor_args = valid_instances

        config_dict = instance.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["algorithm"] == constructor_args["algorithm"]
        assert len(config_dict["proposal_methods"]) == 1
        assert config_dict["proposal_methods"][0]["proposal_type"] == "test_proposal"
        assert (
            config_dict["default_proposal_method"]
            == constructor_args["default_proposal_method"]
        )

        recreated_config = SpeculatorsConfig.model_validate(config_dict)
        assert recreated_config.algorithm == instance.algorithm
        assert (
            recreated_config.proposal_methods[0].proposal_type
            == instance.proposal_methods[0].proposal_type
        )
        assert (
            recreated_config.default_proposal_method == instance.default_proposal_method
        )
        assert recreated_config.verifier.name_or_path == instance.verifier.name_or_path


class TestSpeculatorModelConfig:
    @pytest.fixture
    def sample_verifier_config(self):
        return VerifierConfig(
            name_or_path="test/verifier",
            architectures=["TestModel"],
        )

    @pytest.fixture
    def sample_speculators_config(self, sample_verifier_config):
        return SpeculatorsConfig(
            algorithm="test_algorithm",
            proposal_methods=[TokenProposalConfigTest()],
            default_proposal_method="test_proposal",
            verifier=sample_verifier_config,
        )

    @pytest.fixture(
        params=[
            {"speculators_model_type": "test_model"},
            {"speculators_model_type": "test_model", "test_field": 999},
        ],
        ids=["basic_config", "custom_field"],
    )
    def valid_instances(self, request, sample_speculators_config):
        """Fixture providing test data for SpeculatorModelConfig."""
        constructor_args = request.param
        constructor_args["speculators_config"] = sample_speculators_config
        instance = SpeculatorModelConfigTest(**constructor_args)
        return instance, constructor_args

    def setup_method(self):
        self._original_token_proposal_registry = (
            TokenProposalConfig.registry.copy()  # type: ignore[misc]
            if TokenProposalConfig.registry  # type: ignore[misc]
            else {}
        )
        TokenProposalConfig.register_decorator(TokenProposalConfigTest, "test_proposal")
        TokenProposalConfig.reload_schema()
        SpeculatorsConfig.reload_schema()
        self._original_model_registry = (
            SpeculatorModelConfig.registry.copy()  # type: ignore[misc]
            if SpeculatorModelConfig.registry  # type: ignore[misc]
            else {}
        )
        SpeculatorModelConfig.register_decorator(
            SpeculatorModelConfigTest, "test_model"
        )
        SpeculatorModelConfig.reload_schema()
        SpeculatorModelConfigTest.reload_schema()

    def teardown_method(self):
        TokenProposalConfig.registry = self._original_token_proposal_registry  # type: ignore[misc]
        TokenProposalConfig.reload_schema()
        SpeculatorModelConfig.registry = self._original_model_registry  # type: ignore[misc]
        SpeculatorModelConfig.reload_schema()
        SpeculatorsConfig.reload_schema()
        SpeculatorModelConfigTest.reload_schema()

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test SpeculatorModelConfig inheritance and class variables."""
        from transformers import PretrainedConfig

        from speculators.utils import PydanticClassRegistryMixin

        assert issubclass(SpeculatorModelConfig, PydanticClassRegistryMixin)
        assert issubclass(SpeculatorModelConfig, PretrainedConfig)
        assert hasattr(SpeculatorModelConfig, "model_type")
        assert SpeculatorModelConfig.model_type == "speculator_model"
        assert hasattr(SpeculatorModelConfig, "auto_package")
        assert SpeculatorModelConfig.auto_package == "speculators.models"
        assert hasattr(SpeculatorModelConfig, "registry_auto_discovery")
        assert SpeculatorModelConfig.registry_auto_discovery is True
        assert hasattr(SpeculatorModelConfig, "schema_discriminator")
        assert SpeculatorModelConfig.schema_discriminator == "speculators_model_type"
        assert hasattr(SpeculatorModelConfig, "from_pretrained")
        assert hasattr(SpeculatorModelConfig, "from_dict")
        assert hasattr(SpeculatorModelConfig, "to_dict")
        assert hasattr(SpeculatorModelConfig, "to_diff_dict")
        assert hasattr(SpeculatorModelConfig, "__pydantic_schema_base_type__")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test SpeculatorModelConfig initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, SpeculatorModelConfig)
        assert (
            instance.speculators_model_type
            == constructor_args["speculators_model_type"]
        )
        assert instance.speculators_config.algorithm == "test_algorithm"
        assert instance.speculators_version is not None
        if "test_field" in constructor_args:
            assert instance.test_field == constructor_args["test_field"]

        # Check that PretrainedConfig attributes are accessible
        assert hasattr(instance, "to_dict")
        assert hasattr(instance, "to_diff_dict")
        assert hasattr(instance, "to_json_string")
        assert hasattr(instance, "to_json_file")
        assert hasattr(instance, "save_pretrained")

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("speculators_model_type", None),
            ("speculators_model_type", 123),
            ("speculators_config", "not_a_config"),
            ("speculators_config", None),
        ],
    )
    def test_invalid_initialization_values(
        self, field, value, sample_speculators_config
    ):
        """Test SpeculatorModelConfig with invalid field values."""
        data = {
            "speculators_model_type": "test_model",
            "speculators_config": sample_speculators_config,
        }
        data[field] = value
        with pytest.raises((ValidationError, ValueError)):
            SpeculatorModelConfig(**data)

    @pytest.mark.smoke
    def test_auto_registry(self):
        """Test SpeculatorModelConfig registry population."""
        classes = SpeculatorModelConfig.registered_classes()
        class_names = [cls.__name__ for cls in classes]
        assert len(class_names) > 0
        assert "EagleSpeculatorConfig" in class_names
        assert "IndependentSpeculatorConfig" in class_names
        assert "MLPSpeculatorConfig" in class_names

    @pytest.mark.smoke
    def test_pydantic_schema_base_type(self):
        """Test __pydantic_schema_base_type__ method."""
        base_type = SpeculatorModelConfig.__pydantic_schema_base_type__()
        assert base_type is SpeculatorModelConfig

        subclass_type = SpeculatorModelConfigTest.__pydantic_schema_base_type__()
        assert subclass_type is SpeculatorModelConfig

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test SpeculatorModelConfig serialization and deserialization."""
        instance, constructor_args = valid_instances

        config_dict = instance.model_dump()
        assert isinstance(config_dict, dict)
        assert (
            config_dict["speculators_model_type"]
            == constructor_args["speculators_model_type"]
        )
        assert config_dict["speculators_config"]["algorithm"] == "test_algorithm"
        if "test_field" in constructor_args:
            assert config_dict["test_field"] == constructor_args["test_field"]

        recreated_config = SpeculatorModelConfig.model_validate(config_dict)
        assert (
            recreated_config.speculators_model_type == instance.speculators_model_type
        )
        assert (
            recreated_config.speculators_config.algorithm
            == instance.speculators_config.algorithm
        )
        if hasattr(instance, "test_field"):
            assert recreated_config.test_field == instance.test_field

    @pytest.mark.smoke
    def test_dict_marshaling(self, sample_speculators_config):
        config: SpeculatorModelConfigTest = SpeculatorModelConfigTest(
            speculators_model_type="test_model",
            speculators_config=sample_speculators_config,
            test_field=678,
        )

        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["speculators_model_type"] == "test_model"
        assert config_dict["speculators_config"]["algorithm"] == "test_algorithm"
        assert config_dict["test_field"] == 678

        config_diff_dict = config.to_diff_dict()
        assert isinstance(config_diff_dict, dict)
        assert config_diff_dict["speculators_model_type"] == "test_model"
        assert config_diff_dict["speculators_config"]["algorithm"] == "test_algorithm"
        assert config_diff_dict["test_field"] == 678

        reload_config = SpeculatorModelConfig.from_dict(config_dict)
        assert reload_config.speculators_model_type == "test_model"
        assert reload_config.speculators_config.algorithm == "test_algorithm"
        assert reload_config.test_field == 678

        reload_diff_config = SpeculatorModelConfig.from_dict(config_diff_dict)
        assert reload_diff_config.speculators_model_type == "test_model"
        assert reload_diff_config.speculators_config.algorithm == "test_algorithm"
        assert reload_diff_config.test_field == 678

    @pytest.mark.sanity
    def test_from_dict_invalid(self, sample_speculators_config):
        with pytest.raises(ValueError) as exc_info:
            SpeculatorModelConfig.from_dict({})

        assert (
            "The config dictionary must contain the 'speculators_model_type' field"
            in str(exc_info.value)
        )

        with pytest.raises(ValueError) as exc_info:
            SpeculatorModelConfig.from_dict(
                {
                    "speculators_config": sample_speculators_config.model_dump(),
                    "test_field": 678,
                }
            )

        assert (
            "The config dictionary must contain the 'speculators_model_type' field "
            in str(exc_info.value)
        )

    @pytest.mark.smoke
    def test_from_pretrained_local_marshalling(
        self,
        sample_speculators_config,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "config.json"
            config: SpeculatorModelConfigTest = SpeculatorModelConfigTest(
                speculators_model_type="test_model",
                speculators_config=sample_speculators_config,
                test_field=678,
            )
            config.save_pretrained(tmp_path)
            assert tmp_path.exists()

            reloaded_config = SpeculatorModelConfig.from_pretrained(tmp_path)
            assert reloaded_config.speculators_model_type == "test_model"
            assert reloaded_config.speculators_config.algorithm == "test_algorithm"
            assert reloaded_config.test_field == 678

    @pytest.mark.smoke
    def test_from_pretrained_hf_hub(self, sample_speculators_config):
        config_data = {
            "speculators_model_type": "test_model",
            "speculators_config": sample_speculators_config.model_dump(),
            "test_field": 678,
        }

        with patch(
            "speculators.config.load_model_checkpoint_config_dict"
        ) as mock_load_config:
            mock_load_config.return_value = config_data
            config = SpeculatorModelConfig.from_pretrained("test/fake-model-hub-name")

            mock_load_config.assert_called_once_with(
                "test/fake-model-hub-name",
                cache_dir=None,
                force_download=False,
                local_files_only=False,
                token=None,
                revision=None,
            )

            # Verify the config was loaded correctly
            assert config.speculators_model_type == "test_model"
            assert config.speculators_config.algorithm == "test_algorithm"
            assert config.test_field == 678

    @pytest.mark.smoke
    def test_from_pretrained_conversion(
        self,
        sample_speculators_config,
    ):
        # conversion not implemented yet, ensure it raises NotImplementedError
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "config.json"
            config_data = {
                "speculators_config": sample_speculators_config.model_dump(),
                "test_field": 678,
            }
            with tmp_path.open("w") as f:
                json.dump(config_data, f)

            with pytest.raises(NotImplementedError) as exc_info:
                SpeculatorModelConfig.from_pretrained(
                    tmp_path, convert_to_speculator=True
                )

        assert "Loading a non-speculator model config is not supported" in str(
            exc_info.value
        )


class TestReloadAndPopulateConfigs:
    """Test suite for reload_and_populate_configs function."""

    def setup_method(self):
        self._original_token_proposal_registry = (
            TokenProposalConfig.registry.copy()  # type: ignore[misc]
            if TokenProposalConfig.registry  # type: ignore[misc]
            else {}
        )
        self._original_model_registry = (
            SpeculatorModelConfig.registry.copy()  # type: ignore[misc]
            if SpeculatorModelConfig.registry  # type: ignore[misc]
            else {}
        )

    def teardown_method(self):
        TokenProposalConfig.registry = self._original_token_proposal_registry  # type: ignore[misc]
        SpeculatorModelConfig.registry = self._original_model_registry  # type: ignore[misc]
        TokenProposalConfig.reload_schema()
        SpeculatorsConfig.reload_schema()
        SpeculatorModelConfig.reload_schema()

    @pytest.mark.smoke
    def test_invocation(self):
        """Test reload_and_populate_configs function execution."""
        # Store original registry sizes
        original_token_classes = len(TokenProposalConfig.registered_classes())
        original_model_classes = len(SpeculatorModelConfig.registered_classes())

        # Call the function
        reload_and_populate_configs()

        # Verify that registries are still populated (function ensures population)
        token_classes = TokenProposalConfig.registered_classes()
        model_classes = SpeculatorModelConfig.registered_classes()

        assert len(token_classes) >= original_token_classes
        assert len(model_classes) >= original_model_classes
        assert len(token_classes) > 0
        assert len(model_classes) > 0

        # Verify expected classes are present
        token_class_names = [cls.__name__ for cls in token_classes]
        model_class_names = [cls.__name__ for cls in model_classes]

        assert "DynamicTreeTokenProposalConfig" in token_class_names
        assert "GreedyTokenProposalConfig" in token_class_names
        assert "EagleSpeculatorConfig" in model_class_names
        assert "IndependentSpeculatorConfig" in model_class_names

    @pytest.mark.sanity
    def test_invalid_invocation(self):
        """Test reload_and_populate_configs with invalid arguments."""
        # Function takes no arguments, so this should raise TypeError
        with pytest.raises(TypeError):
            reload_and_populate_configs("invalid_arg")  # type: ignore[call-arg]
