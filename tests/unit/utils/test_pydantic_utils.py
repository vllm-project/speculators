"""
Unit tests for the pydantic_utils module.
"""

from __future__ import annotations

from typing import ClassVar, TypeVar
from unittest import mock

import pytest
from pydantic import BaseModel, Field, ValidationError

from speculators.utils import PydanticClassRegistryMixin, ReloadableBaseModel
from speculators.utils.pydantic_utils import BaseModelT, RegisterClassT


@pytest.mark.smoke
def test_base_model_t():
    """Test that BaseModelT is configured correctly as a TypeVar."""
    assert isinstance(BaseModelT, type(TypeVar("test")))
    assert BaseModelT.__name__ == "BaseModelT"
    assert BaseModelT.__bound__ is BaseModel
    assert BaseModelT.__constraints__ == ()


@pytest.mark.smoke
def test_register_class_t():
    """Test that RegisterClassT is configured correctly as a TypeVar."""
    assert isinstance(RegisterClassT, type(TypeVar("test")))
    assert RegisterClassT.__name__ == "RegisterClassT"
    assert RegisterClassT.__bound__ is not None
    assert RegisterClassT.__constraints__ == ()


class TestReloadableBaseModel:
    """Test suite for ReloadableBaseModel."""

    @pytest.fixture(
        params=[
            {"name": "test_value"},
            {"name": "hello_world"},
            {"name": "another_test"},
        ],
        ids=["basic_string", "multi_word", "underscore"],
    )
    def valid_instances(self, request) -> tuple[ReloadableBaseModel, dict[str, str]]:
        """Fixture providing test data for ReloadableBaseModel."""

        class TestModel(ReloadableBaseModel):
            name: str

        constructor_args = request.param
        instance = TestModel(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test ReloadableBaseModel inheritance and class variables."""
        assert issubclass(ReloadableBaseModel, BaseModel)
        assert hasattr(ReloadableBaseModel, "model_config")
        assert hasattr(ReloadableBaseModel, "reload_schema")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test ReloadableBaseModel initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, ReloadableBaseModel)
        assert instance.name == constructor_args["name"]  # type: ignore[attr-defined]

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("name", None),
            ("name", 123),
            ("name", []),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test ReloadableBaseModel with invalid field values."""

        class TestModel(ReloadableBaseModel):
            name: str

        data = {field: value}
        with pytest.raises(ValidationError):
            TestModel(**data)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test ReloadableBaseModel initialization without required field."""

        class TestModel(ReloadableBaseModel):
            name: str

        with pytest.raises(ValidationError):
            TestModel()  # type: ignore[call-arg]

    @pytest.mark.smoke
    def test_reload_schema(self):
        """Test ReloadableBaseModel.reload_schema method."""

        class TestModel(ReloadableBaseModel):
            name: str

        # Mock the model_rebuild method to simulate schema reload
        with mock.patch.object(TestModel, "model_rebuild") as mock_rebuild:
            TestModel.reload_schema()
            mock_rebuild.assert_called_once_with(force=True)

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test ReloadableBaseModel serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["name"] == constructor_args["name"]

        recreated = instance.__class__.model_validate(data_dict)
        assert isinstance(recreated, instance.__class__)
        assert recreated.name == constructor_args["name"]


class TestPydanticClassRegistryMixin:
    """Test suite for PydanticClassRegistryMixin."""

    @pytest.fixture(
        params=[
            {"test_type": "test_sub", "value": "test_value"},
            {"test_type": "test_sub", "value": "hello_world"},
        ],
        ids=["basic_value", "multi_word"],
    )
    def valid_instances(
        self, request
    ) -> tuple[PydanticClassRegistryMixin, dict, type, type]:
        """Fixture providing test data for PydanticClassRegistryMixin."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register("test_sub")
        class TestSubModel(TestBaseModel):
            test_type: str = "test_sub"
            value: str

        TestBaseModel.reload_schema()

        constructor_args = request.param
        instance = TestSubModel(value=constructor_args["value"])
        return instance, constructor_args, TestBaseModel, TestSubModel

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test PydanticClassRegistryMixin inheritance and class variables."""
        assert issubclass(PydanticClassRegistryMixin, ReloadableBaseModel)
        assert hasattr(PydanticClassRegistryMixin, "schema_discriminator")
        assert PydanticClassRegistryMixin.schema_discriminator == "model_type"
        assert hasattr(PydanticClassRegistryMixin, "register_decorator")
        assert hasattr(PydanticClassRegistryMixin, "__get_pydantic_core_schema__")
        assert hasattr(PydanticClassRegistryMixin, "__pydantic_generate_base_schema__")
        assert hasattr(PydanticClassRegistryMixin, "auto_populate_registry")
        assert hasattr(PydanticClassRegistryMixin, "registered_classes")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test PydanticClassRegistryMixin initialization."""
        instance, constructor_args, base_class, sub_class = valid_instances
        assert isinstance(instance, sub_class)
        assert isinstance(instance, base_class)
        assert instance.test_type == constructor_args["test_type"]
        assert instance.value == constructor_args["value"]

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("test_type", None),
            ("test_type", 123),
            ("value", None),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test PydanticClassRegistryMixin with invalid field values."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register("test_sub")
        class TestSubModel(TestBaseModel):
            test_type: str = "test_sub"
            value: str

        data = {field: value}
        if field == "test_type":
            data["value"] = "test"
        else:
            data["test_type"] = "test_sub"

        with pytest.raises(ValidationError):
            TestSubModel(**data)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test PydanticClassRegistryMixin initialization without required field."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register("test_sub")
        class TestSubModel(TestBaseModel):
            test_type: str = "test_sub"
            value: str

        with pytest.raises(ValidationError):
            TestSubModel()  # type: ignore[call-arg]

    @pytest.mark.smoke
    def test_register_decorator(self):
        """Test PydanticClassRegistryMixin.register_decorator method."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register()
        class TestSubModel(TestBaseModel):
            test_type: str = "TestSubModel"
            value: str

        assert TestBaseModel.registry is not None  # type: ignore[misc]
        assert "TestSubModel" in TestBaseModel.registry  # type: ignore[misc]
        assert TestBaseModel.registry["TestSubModel"] is TestSubModel  # type: ignore[misc]

    @pytest.mark.sanity
    def test_register_decorator_with_name(self):
        """Test PydanticClassRegistryMixin.register_decorator with custom name."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register("custom_name")
        class TestSubModel(TestBaseModel):
            test_type: str = "custom_name"
            value: str

        assert TestBaseModel.registry is not None  # type: ignore[misc]
        assert "custom_name" in TestBaseModel.registry  # type: ignore[misc]
        assert TestBaseModel.registry["custom_name"] is TestSubModel  # type: ignore[misc]

    @pytest.mark.sanity
    def test_register_decorator_invalid_type(self):
        """Test PydanticClassRegistryMixin.register_decorator with invalid type."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        class RegularClass:
            pass

        with pytest.raises(TypeError) as exc_info:
            TestBaseModel.register_decorator(RegularClass)  # type: ignore[type-var]

        assert "not a subclass of Pydantic BaseModel" in str(exc_info.value)

    @pytest.mark.smoke
    def test_auto_populate_registry(self):
        """Test PydanticClassRegistryMixin.auto_populate_registry method."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str
            registry_auto_discovery: ClassVar[bool] = True

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        with (
            mock.patch.object(TestBaseModel, "reload_schema") as mock_reload,
            mock.patch(
                "speculators.utils.registry.RegistryMixin.auto_populate_registry",
                return_value=True,
            ),
        ):
            result = TestBaseModel.auto_populate_registry()
            assert result is True
            mock_reload.assert_called_once()

    @pytest.mark.smoke
    def test_registered_classes(self):
        """Test PydanticClassRegistryMixin.registered_classes method."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str
            registry_auto_discovery: ClassVar[bool] = False

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        @TestBaseModel.register("test_sub_a")
        class TestSubModelA(TestBaseModel):
            test_type: str = "test_sub_a"
            value_a: str

        @TestBaseModel.register("test_sub_b")
        class TestSubModelB(TestBaseModel):
            test_type: str = "test_sub_b"
            value_b: int

        # Test normal case with registered classes
        registered = TestBaseModel.registered_classes()
        assert isinstance(registered, tuple)
        assert len(registered) == 2
        assert TestSubModelA in registered
        assert TestSubModelB in registered

    @pytest.mark.sanity
    def test_registered_classes_with_auto_discovery(self):
        """Test PydanticClassRegistryMixin.registered_classes with auto discovery."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str
            registry_auto_discovery: ClassVar[bool] = True

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        with mock.patch.object(
            TestBaseModel, "auto_populate_registry"
        ) as mock_auto_populate:
            # Mock the registry to simulate registered classes
            TestBaseModel.registry = {"test_class": type("TestClass", (), {})}  # type: ignore[misc]
            mock_auto_populate.return_value = False

            registered = TestBaseModel.registered_classes()
            mock_auto_populate.assert_called_once()
            assert isinstance(registered, tuple)
            assert len(registered) == 1

    @pytest.mark.sanity
    def test_registered_classes_no_registry(self):
        """Test PydanticClassRegistryMixin.registered_classes with no registry."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

        # Ensure registry is None
        TestBaseModel.registry = None  # type: ignore[misc]

        with pytest.raises(ValueError) as exc_info:
            TestBaseModel.registered_classes()

        assert "must be called after registering classes" in str(exc_info.value)

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test PydanticClassRegistryMixin serialization and deserialization."""
        instance, constructor_args, base_class, sub_class = valid_instances

        # Test serialization with model_dump
        dump_data = instance.model_dump()
        assert isinstance(dump_data, dict)
        assert dump_data["test_type"] == constructor_args["test_type"]
        assert dump_data["value"] == constructor_args["value"]

        # Test deserialization via subclass
        recreated = sub_class.model_validate(dump_data)
        assert isinstance(recreated, sub_class)
        assert recreated.test_type == constructor_args["test_type"]
        assert recreated.value == constructor_args["value"]

        # Test polymorphic deserialization via base class
        recreated_base = base_class.model_validate(dump_data)  # type: ignore[assignment]
        assert isinstance(recreated_base, sub_class)
        assert recreated_base.test_type == constructor_args["test_type"]
        assert recreated_base.value == constructor_args["value"]

    @pytest.mark.regression
    def test_polymorphic_container_marshalling(self):
        """Test PydanticClassRegistryMixin in container models."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "test_type"
            test_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls
                return TestBaseModel

            @classmethod
            def __pydantic_generate_base_schema__(cls, handler):
                return handler(cls)

        @TestBaseModel.register("sub_a")
        class TestSubModelA(TestBaseModel):
            test_type: str = "sub_a"
            value_a: str

        @TestBaseModel.register("sub_b")
        class TestSubModelB(TestBaseModel):
            test_type: str = "sub_b"
            value_b: int

        class ContainerModel(BaseModel):
            name: str
            model: TestBaseModel
            models: list[TestBaseModel]

        sub_a = TestSubModelA(value_a="test")
        sub_b = TestSubModelB(value_b=123)

        container = ContainerModel(name="container", model=sub_a, models=[sub_a, sub_b])

        # Verify container construction
        assert isinstance(container.model, TestSubModelA)
        assert container.model.test_type == "sub_a"
        assert container.model.value_a == "test"
        assert len(container.models) == 2
        assert isinstance(container.models[0], TestSubModelA)
        assert isinstance(container.models[1], TestSubModelB)

        # Test serialization
        dump_data = container.model_dump()
        assert isinstance(dump_data, dict)
        assert dump_data["name"] == "container"
        assert dump_data["model"]["test_type"] == "sub_a"
        assert dump_data["model"]["value_a"] == "test"
        assert len(dump_data["models"]) == 2
        assert dump_data["models"][0]["test_type"] == "sub_a"
        assert dump_data["models"][1]["test_type"] == "sub_b"

        # Test deserialization
        recreated = ContainerModel.model_validate(dump_data)
        assert isinstance(recreated, ContainerModel)
        assert recreated.name == "container"
        assert isinstance(recreated.model, TestSubModelA)
        assert len(recreated.models) == 2
        assert isinstance(recreated.models[0], TestSubModelA)
        assert isinstance(recreated.models[1], TestSubModelB)

    @pytest.mark.smoke
    def test_register_preserves_pydantic_metadata(self):  # noqa: C901
        """Test that registered Pydantic classes retain docs, types, and methods."""

        class TestBaseModel(PydanticClassRegistryMixin):
            schema_discriminator: ClassVar[str] = "model_type"
            model_type: str

            @classmethod
            def __pydantic_schema_base_type__(cls) -> type[TestBaseModel]:
                if cls.__name__ == "TestBaseModel":
                    return cls

                return TestBaseModel

        @TestBaseModel.register("documented_model")
        class DocumentedModel(TestBaseModel):
            """This is a documented Pydantic model with methods and type hints."""

            model_type: str = "documented_model"
            value: int = Field(description="An integer value for the model")

            def get_value(self) -> int:
                """Get the stored value.

                :return: The stored integer value
                """
                return self.value

            def set_value(self, new_value: int) -> None:
                """Set a new value.

                :param new_value: The new integer value to set
                """
                self.value = new_value

            @classmethod
            def from_string(cls, value_str: str) -> DocumentedModel:
                """Create instance from string.

                :param value_str: String representation of value
                :return: New DocumentedModel instance
                """
                return cls(value=int(value_str))

            @staticmethod
            def validate_value(value: int) -> bool:
                """Validate that a value is positive.

                :param value: Value to validate
                :return: True if positive, False otherwise
                """
                return value > 0

            def model_post_init(self, __context) -> None:
                """Post-initialization processing.

                :param __context: Validation context
                """
                if self.value < 0:
                    raise ValueError("Value must be non-negative")

        # Check that the class was registered
        assert TestBaseModel.is_registered("documented_model")
        registered_class = TestBaseModel.get_registered_object("documented_model")
        assert registered_class is DocumentedModel

        # Check that the class retains its documentation
        assert registered_class.__doc__ is not None
        assert "documented Pydantic model with methods" in registered_class.__doc__

        # Check that methods retain their documentation
        assert registered_class.get_value.__doc__ is not None
        assert "Get the stored value" in registered_class.get_value.__doc__
        assert registered_class.set_value.__doc__ is not None
        assert "Set a new value" in registered_class.set_value.__doc__
        assert registered_class.from_string.__doc__ is not None
        assert "Create instance from string" in registered_class.from_string.__doc__
        assert registered_class.validate_value.__doc__ is not None
        assert (
            "Validate that a value is positive"
            in registered_class.validate_value.__doc__
        )
        assert registered_class.model_post_init.__doc__ is not None
        assert (
            "Post-initialization processing" in registered_class.model_post_init.__doc__
        )

        # Check that methods are callable and work correctly
        instance = DocumentedModel(value=42)
        assert isinstance(instance, DocumentedModel)
        assert instance.get_value() == 42
        instance.set_value(100)
        assert instance.get_value() == 100
        assert instance.model_type == "documented_model"

        # Check class methods work
        instance2 = DocumentedModel.from_string("123")
        assert instance2.get_value() == 123
        assert instance2.model_type == "documented_model"

        # Check static methods work
        assert DocumentedModel.validate_value(10) is True
        assert DocumentedModel.validate_value(-5) is False

        # Check that Pydantic functionality is preserved
        data_dict = instance.model_dump()
        assert data_dict["value"] == 100
        assert data_dict["model_type"] == "documented_model"

        recreated = DocumentedModel.model_validate(data_dict)
        assert isinstance(recreated, DocumentedModel)
        assert recreated.value == 100
        assert recreated.model_type == "documented_model"

        # Test field validation
        with pytest.raises(ValidationError):
            DocumentedModel(value="not_an_int")  # type: ignore[arg-type]

        # Test post_init validation
        with pytest.raises(ValueError, match="Value must be non-negative"):
            DocumentedModel(value=-10)

        # Check that Pydantic field metadata is preserved
        value_field = DocumentedModel.model_fields["value"]
        assert value_field.description == "An integer value for the model"

        # Check that type annotations are preserved (if accessible)
        import inspect

        if hasattr(inspect, "get_annotations"):
            # Python 3.10+
            try:
                annotations = inspect.get_annotations(DocumentedModel.get_value)
                return_ann = annotations.get("return")
                assert return_ann is int or return_ann == "int"
            except (AttributeError, NameError):
                # Fallback for older Python or missing annotations
                pass

        # Check that the class name is preserved
        assert DocumentedModel.__name__ == "DocumentedModel"
        assert DocumentedModel.__qualname__.endswith("DocumentedModel")

        # Verify that the class is still properly integrated with the registry system
        all_registered = TestBaseModel.registered_classes()
        assert DocumentedModel in all_registered

        # Test that the registered class is the same as the original
        assert registered_class is DocumentedModel
