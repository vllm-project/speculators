"""
General pydantic utilities for Speculators.

This module provides integration between Pydantic and the Speculators library,
enabling things like polymorphic serialization and deserialization of Pydantic
models using a discriminator field and registry.

Classes:
    PydanticClassRegistryMixin: A mixin that combines Pydantic models with the
        ClassRegistryMixin to support polymorphic model instantiation based on
        a discriminator field
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, Union

from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from speculators.utils.registry import ClassRegistryMixin

__all__ = ["PydanticClassRegistryMixin", "ReloadableBaseModel"]


class ReloadableBaseModel(BaseModel):
    @classmethod
    def reload_schema(cls):
        """
        Reloads the schema for the class, ensuring that the registry is populated
        and that the schema is up-to-date.

        This method is useful when the registry has been modified or when the
        class needs to be re-validated with the latest schema.
        """
        cls.model_rebuild(force=True)


class PydanticClassRegistryMixin(ReloadableBaseModel, ABC, ClassRegistryMixin):
    """
    A mixin class that integrates Pydantic models with the ClassRegistryMixin to enable
    polymorphic serialization and deserialization based on a discriminator field.

    This mixin allows Pydantic models to be registered in a registry and dynamically
    instantiated based on a discriminator field in the input data.
    It overrides Pydantic's validation system to correctly instantiate the appropriate
    subclass based on the discriminator value and the name of the registered classes.

    The mixin is particularly useful for implementing base registry classes that need to
    support multiple implementations, such as different token proposal methods or
    speculative decoding algorithms.

    Usage Example:
    ```python
    from typing import ClassVar
    from pydantic import BaseModel, Field
    from speculators.utils import PydanticClassRegistryMixin

    class BaseConfig(PydanticClassRegistryMixin):
        @classmethod
        def __pydantic_schema_base_type__(cls) -> type["BaseConfig"]:
            if cls.__name__ == "BaseConfig":
                return cls
            return BaseConfig

        schema_discriminator: ClassVar[str] = "config_type"
        config_type: str = Field(description="The type of configuration")

    @BaseConfig.register("config_a")
    class ConfigA(BaseConfig):
        config_type: str = "config_a"
        value_a: str = Field(description="A value specific to ConfigA")

    @BaseConfig.register("config_b")
    class ConfigB(BaseConfig):
        config_type: str = "config_b"
        value_b: int = Field(description="A value specific to ConfigB")

    BaseConfig.reload_schema()  # Ensures the schema is up-to-date with registry

    # Dynamic instantiation based on config_type
    config_data = {"config_type": "config_a", "value_a": "test"}
    config = BaseConfig.model_validate(config_data)  # Returns ConfigA instance
    print(config)
    dump_data = config.model_dump()  # Dumps the data to a dictionary
    print(dump_data)  # Output: {'config_type': 'config_a', 'value_a': 'test'}
    ```

    :cvar schema_discriminator: The field name used as the discriminator in the JSON
        schema. Default is "model_type".
    :cvar registry: A dictionary mapping discriminator values to pydantic model classes.
    """

    schema_discriminator: ClassVar[str] = "model_type"
    registry: ClassVar[Optional[dict[str, BaseModel]]] = None  # type: ignore[assignment]

    @classmethod
    def register_decorator(
        cls, clazz: type[BaseModel], name: Optional[Union[str, list[str]]] = None
    ) -> type[BaseModel]:
        """
        Registers a Pydantic model class with the registry.

        This method extends the ClassRegistryMixin.register_decorator method by adding
        a type check to ensure only Pydantic BaseModel subclasses can be registered.

        :param clazz: The Pydantic model class to register
        :param name: Optional name to register the class under. If None, the class name
            is used as the registry key.
        :return: The registered class.
        :raises TypeError: If clazz is not a subclass of Pydantic BaseModel
        """
        if not issubclass(clazz, BaseModel):
            raise TypeError(
                f"Cannot register {clazz.__name__} as it is not a subclass of "
                "Pydantic BaseModel"
            )

        return super().register_decorator(clazz, name=name)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Customizes the Pydantic schema for polymorphic model validation.

        This method is part of Pydantic's validation system and is called during
        schema generation. It checks if the source_type matches the base type of the
        polymorphic model. If it does, it generates a tagged union schema that allows
        for dynamic instantiation of the appropriate subclass based on the discriminator
        field.

        :param source_type: The type for which the schema is being generated
        :param handler: Handler for generating core schema
        :return: A CoreSchema object with the custom validator if appropriate
        """
        if source_type == cls.__pydantic_schema_base_type__():
            if not cls.registry:
                return cls.__pydantic_generate_base_schema__(handler)

            choices = {
                name: handler(model_class) for name, model_class in cls.registry.items()
            }

            return core_schema.tagged_union_schema(
                choices=choices,
                discriminator=cls.schema_discriminator,
            )

        return handler(cls)

    @classmethod
    @abstractmethod
    def __pydantic_schema_base_type__(cls) -> type[Any]:
        """
        Abstract method that must be implemented by subclasses to define the base type.

        This method should return the base class type that serves as the root of the
        polymorphic hierarchy. The returned type is used to determine when to apply
        the custom validation logic for polymorphic instantiation.

        Example implementation:
        ```python
        @classmethod
        def __pydantic_schema_base_type__(cls) -> type["MyBaseClass"]:
            return MyBaseClass
        ```

        :return: The base class type for polymorphic validation
        """
        ...

    @classmethod
    def __pydantic_generate_base_schema__(
        cls, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Generates the base schema for the polymorphic model.

        This method is used by the Pydantic validation system to create the core
        schema for the base class. By default, it returns an any_schema which accepts
        any valid input, relying on the validator function to perform the actual
        validation and model instantiation.

        Subclasses can override this method to provide a more specific base schema
        if needed.

        :param handler: Handler for generating core schema
        :return: A CoreSchema object representing the base schema
        """
        return core_schema.any_schema()

    @classmethod
    def auto_populate_registry(cls) -> bool:
        """
        Ensures that all registered classes in the registry are properly initialized.

        This method is called automatically by Pydantic when the model is instantiated
        or validated. It ensures that all classes in the registry are loaded and ready
        for use.

        This is particularly useful for ensuring that all subclasses are registered
        before any validation occurs.

        :return: True if the registry was populated, False if it was already populated
        :raises ValueError: If called when registry_auto_discovery is False
        """
        populated = super().auto_populate_registry()
        cls.reload_schema()

        return populated
