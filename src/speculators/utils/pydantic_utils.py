"""
Pydantic utilities for polymorphic model serialization and registry integration.

Provides integration between Pydantic and the registry system, enabling
polymorphic serialization and deserialization of Pydantic models using
a discriminator field and dynamic class registry. Includes base model classes
with standardized configurations and generic status breakdown models for
structured result organization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar, get_args, get_origin

from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from speculators.utils.registry import RegistryMixin

__all__ = [
    "BaseModelT",
    "PydanticClassRegistryMixin",
    "RegisterClassT",
    "ReloadableBaseModel",
]


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)
RegisterClassT = TypeVar("RegisterClassT", bound=type[BaseModel])


class ReloadableBaseModel(BaseModel):
    """
    Base Pydantic model with schema reloading capabilities.

    Provides dynamic schema rebuilding functionality for models that need to
    update their validation schemas at runtime, particularly useful when
    working with registry-based polymorphic models where new types are
    registered after initial class definition.
    """

    @classmethod
    def reload_schema(cls, dependencies: bool = True):
        """
        Reload and rebuild the Pydantic model validation schema.

        Forces reconstruction of the model schema and optionally rebuilds
        schemas for all dependent models in the reloadable dependency chains.
        Essential when new types are registered that affect polymorphic validation.

        :param dependencies: Whether to reload dependent model schemas as well
        """
        cls.model_rebuild(force=True)

        if dependencies:
            for chain in cls.reloadable_dependency_chains():
                for clazz in chain:
                    clazz.model_rebuild(force=True)

    @classmethod
    def reloadable_dependency_chains(
        cls, target: type[ReloadableBaseModel] | None = None
    ) -> list[list[type[ReloadableBaseModel]]]:
        """
        Find all dependency chains leading to the target model class.

        Uses depth-first search to identify dependency paths between reloadable
        models, ensuring proper schema reload ordering to maintain validation
        consistency across the polymorphic model hierarchy.

        :param target: Target model class to find chains for. Uses cls if None
        :return: List of dependency chains ending at the target class
        """
        if target is None:
            target = cls

        # Build a map of all reloadable classes to their dependencies
        dependencies: dict[
            type[ReloadableBaseModel], list[type[ReloadableBaseModel]]
        ] = {}

        for reloadable in cls.reloadable_descendants(ReloadableBaseModel):
            deps = []
            for field_deps in reloadable.reloadable_fields().values():
                deps.extend(field_deps)
            dependencies[reloadable] = deps

        # Find all dependency chains ending at target using DFS
        chains = []

        def _find_chains(
            current: type[ReloadableBaseModel], path: list[type[ReloadableBaseModel]]
        ):
            if current == target:
                chains.append(path)
                return

            for dependent in dependencies.get(current, []):
                if dependent not in path:  # Avoid cycles
                    _find_chains(dependent, [current] + path)

        for cls_type, deps in dependencies.items():
            if deps and cls_type != target:
                _find_chains(cls_type, [])

        return chains

    @classmethod
    def reloadable_fields(
        cls,
    ) -> dict[str, list[type[ReloadableBaseModel]]]:
        """
        Identify model fields containing reloadable model types.

        Recursively analyzes field type annotations to find all ReloadableBaseModel
        subclasses used within the model schema, enabling dependency tracking for
        proper schema reload ordering.

        :return: Mapping of field names to lists of reloadable model types
        """

        def _recursive_resolve_reloadable_types(type_: type | None) -> list[type]:
            if type_ is None:
                return []

            if (origin := get_origin(type_)) is None:
                return [type_] if issubclass(type_, ReloadableBaseModel) else []

            resolved = []
            if issubclass(origin, ReloadableBaseModel):
                resolved.append(origin)

            for arg in get_args(type_):
                resolved.extend(_recursive_resolve_reloadable_types(arg))

            return resolved

        fields = {}

        for name, info in cls.model_fields.items():
            if reloadable_types := _recursive_resolve_reloadable_types(info.annotation):
                fields[name] = reloadable_types

        return fields

    @classmethod
    def reloadable_descendants(
        cls, target: type[ReloadableBaseModel] | None = None
    ) -> set[type[ReloadableBaseModel]]:
        """
        Find all ReloadableBaseModel descendants of the target class.

        Traverses the inheritance hierarchy to collect all subclasses that inherit
        from ReloadableBaseModel, enabling comprehensive dependency analysis for
        schema reloading operations.

        :param target: Base class to find descendants for. Uses cls if None
        :return: Set of all descendant ReloadableBaseModel classes
        """
        if target is None:
            target = cls

        descendants: set[type[ReloadableBaseModel]] = set()
        stack: list[type[ReloadableBaseModel]] = [target]

        while stack:
            current = stack.pop()
            for subclass in current.__subclasses__():
                if (
                    issubclass(subclass, ReloadableBaseModel)
                    and subclass is not cls
                    and subclass not in descendants
                ):
                    descendants.add(subclass)
                    stack.append(subclass)

        return descendants


class PydanticClassRegistryMixin(
    ReloadableBaseModel, RegistryMixin[type[BaseModelT]], ABC, Generic[BaseModelT]
):
    """
    Polymorphic Pydantic model enabling registry-based dynamic type instantiation.

    Integrates Pydantic validation with the registry system for polymorphic
    serialization and deserialization using a discriminator field. Automatically
    instantiates the correct subclass during validation based on registry mappings.

    Example:
    ::
        from speculators.utils import PydanticClassRegistryMixin

        class BaseConfig(PydanticClassRegistryMixin["BaseConfig"]):
            schema_discriminator: ClassVar[str] = "config_type"
            config_type: str = Field(description="Configuration type identifier")

            @classmethod
            def __pydantic_schema_base_name__(cls) -> str:
                return "BaseConfig"

        @BaseConfig.register("database")
        class DatabaseConfig(BaseConfig):
            config_type: str = "database"
            connection_string: str = Field(description="Database connection string")

        # Dynamic instantiation based on discriminator
        config = BaseConfig.model_validate({
            "config_type": "database",
            "connection_string": "postgresql://localhost:5432/db"
        })

    :cvar schema_discriminator: Field name for polymorphic type discrimination
    """

    schema_discriminator: ClassVar[str] = "model_type"

    @classmethod
    def register_decorator(  # type: ignore[override]
        cls, clazz: RegisterClassT, name: str | list[str] | None = None
    ) -> RegisterClassT:
        """
        Register a Pydantic model class with type validation and schema reload.

        Validates that the class is a proper Pydantic BaseModel subclass before
        registering it in the class registry. Automatically triggers schema
        reload to incorporate the new type into polymorphic validation.

        :param clazz: Pydantic model class to register in the polymorphic hierarchy
        :param name: Registry identifier for the class. Uses class name if None
        :return: The registered class unchanged for decorator chaining
        :raises TypeError: If clazz is not a Pydantic BaseModel subclass
        """
        if not issubclass(clazz, BaseModel):
            raise TypeError(
                f"Cannot register {clazz.__name__} as it is not a subclass of "
                "Pydantic BaseModel"
            )

        super().register_decorator(clazz, name=name)
        cls.reload_schema()

        return clazz

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Generate polymorphic validation schema for dynamic type instantiation.

        Creates a tagged union schema that enables Pydantic to automatically
        instantiate the correct subclass based on the discriminator field value.
        Falls back to base schema generation when no registry is available.

        :param source_type: Type being processed for schema generation
        :param handler: Pydantic core schema generation handler
        :return: Tagged union schema for polymorphic validation or base schema
        """
        if (
            source_type is None
            or not isinstance(source_type, type)
            or source_type.__name__ != cls.__pydantic_schema_base_name__()
        ):
            return handler(cls)

        if not cls.registry:
            return cls.__pydantic_generate_base_schema__(handler)

        choices = {
            name: handler(model_class) for name, model_class in cls.registry.items()
        }

        return core_schema.tagged_union_schema(
            choices=choices,
            discriminator=cls.schema_discriminator,
        )

    @classmethod
    @abstractmethod
    def __pydantic_schema_base_name__(cls) -> str:
        """
        Define the name of the base type for polymorphic validation hierarchy.

        Must be implemented by subclasses to specify which type serves as the
        root of the polymorphic hierarchy for schema generation and validation.

        :return: Base class name for the polymorphic model hierarchy
        """
        ...

    @classmethod
    def __pydantic_generate_base_schema__(
        cls, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Generate fallback schema for polymorphic models without registry.

        Provides a base schema that accepts any valid input when no registry
        is available for polymorphic validation. Used as fallback during
        schema generation when the registry has not been populated.

        :param handler: Pydantic core schema generation handler
        :return: Base CoreSchema that accepts any valid input
        """
        return core_schema.any_schema()

    @classmethod
    def auto_populate_registry(cls) -> bool:
        """
        Initialize registry with auto-discovery and reload validation schema.

        Triggers automatic population of the class registry through the parent
        RegistryMixin functionality and ensures the Pydantic validation schema
        is updated to include all discovered types for polymorphic validation.

        :return: True if registry was populated, False if already populated
        :raises ValueError: If called when registry_auto_discovery is disabled
        """
        populated = super().auto_populate_registry()
        cls.reload_schema()

        return populated

    @classmethod
    def registered_classes(cls) -> tuple[type[BaseModelT], ...]:
        """
        Get all registered pydantic classes from the registry.

        Automatically triggers auto-discovery if registry_auto_discovery is enabled
        to ensure all available implementations are included.

        :return: Tuple of all registered classes including auto-discovered ones
        :raises ValueError: If called before any objects have been registered
        """
        if cls.registry_auto_discovery:
            cls.auto_populate_registry()

        if cls.registry is None:
            raise ValueError(
                "ClassRegistryMixin.registered_classes() must be called after "
                "registering classes with ClassRegistryMixin.register()."
            )

        return tuple(cls.registry.values())
