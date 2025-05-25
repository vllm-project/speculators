"""
Registry system for classes in the Speculators library.

This module provides a flexible class registration and discovery system used
throughout the Speculators library. It enables automatic registration of classes
and discovery of implementations through class decorators and module imports.

The registry system is used to track different implementations of token proposal
methods, speculative decoding algorithms, and speculator models, allowing for
dynamic discovery and instantiation based on configuration parameters.

Classes:
    ClassRegistryMixin: Base mixin for creating class registries with decorators
    AutoClassRegistryMixin: Extended mixin that combines registry with auto-importing
"""

from typing import Any, Callable, ClassVar, Optional

from speculators.utils.auto_importer import AutoImporterMixin

__all__ = ["AutoClassRegistryMixin", "ClassRegistryMixin"]


class ClassRegistryMixin:
    """
    A mixin class that provides a registration system for tracking class
    implementations.

    This mixin allows classes to maintain a registry of subclasses that can be
    dynamically discovered and instantiated. Classes that inherit from this mixin
    can use the @register decorator to add themselves to the registry.

    The registry is class-specific, meaning each class that inherits from this mixin
    will have its own separate registry of implementations.

    Example:
    ```python
    class BaseAlgorithm(ClassRegistryMixin):
        pass

    @BaseAlgorithm.register()
    class ConcreteAlgorithm(BaseAlgorithm):
        pass

    @BaseAlgorithm.register("custom_name")
    class AnotherAlgorithm(BaseAlgorithm):
        pass

    # Get all registered algorithm implementations
    algorithms = BaseAlgorithm.registered_classes()
    ```

    :cvar registry: A dictionary mapping class names to classes that have been
        registered to the extending subclass through the @subclass.register() decorator
    """

    registry: ClassVar[Optional[dict[str, type[Any]]]] = None

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable[[type[Any]], type[Any]]:
        """
        An invoked class decorator that registers that class with the registry under
        either the provided name or the class name if no name is provided.

        Example:
        ```python
        @ClassRegistryMixin.register()
        class ExampleClass:
            ...

        @ClassRegistryMixin.register("custom_name")
        class AnotherExampleClass:
            ...
        ```

        :param name: Optional name to register the class under. If None, the class name
            is used as the registry key.
        :return: A decorator function that registers the decorated class.
        :raises ValueError: If name is provided but is not a string.
        """
        if name is not None and not isinstance(name, str):
            raise ValueError(
                "ClassRegistryMixin.register() name must be a string or None. "
                f"Got {name}."
            )

        return lambda subclass: cls.register_decorator(subclass, name=name)

    @classmethod
    def register_decorator(
        cls, clazz: type[Any], name: Optional[str] = None
    ) -> type[Any]:
        """
        A non-invoked class decorator that registers the class with the registry.
        If passed through a lambda, then name can be passed in as well.
        Otherwise, the only argument is the decorated class.

        Example:
        ```python
        @ClassRegistryMixin.register_decorator
        class ExampleClass:
            ...
        ```

        :param clazz: The class to register
        :param name: Optional name to register the class under. If None, the class name
            is used as the registry key.
        :return: The registered class.
        :raises TypeError: If the decorator is used incorrectly or if the class is not
            a type.
        :raises ValueError: If the class is already registered or if name is provided
            but is not a string.
        """

        if not isinstance(clazz, type):
            raise TypeError(
                "ClassRegistryMixin.register_decorator must be used as a class "
                "decorator and without invocation."
                f"Got improper clazz arg {clazz}."
            )

        if not name:
            name = clazz.__name__
        elif not isinstance(name, str):
            raise ValueError(
                "ClassRegistryMixin.register_decorator must be used as a class "
                "decorator and without invocation. "
                f"Got imporoper name arg {name}."
            )

        if cls.registry is None:
            cls.registry = {}

        if name in cls.registry:
            raise ValueError(
                f"ClassRegistryMixin.register_decorator cannot register a class "
                f"{clazz} with the name {name} because it is already registered."
            )

        cls.registry[name] = clazz

        return clazz

    @classmethod
    def registered_classes(cls) -> tuple[type[Any], ...]:
        """
        Returns a tuple of all classes that have been registered with this registry.

        :return: A tuple containing all registered class implementations.
        :raises ValueError: If called before any classes have been registered.
        """
        if cls.registry is None:
            raise ValueError(
                "ClassRegistryMixin.registered_classes() must be called after "
                "registering classes with ClassRegistryMixin.register()."
            )

        return tuple(cls.registry.values())


class AutoClassRegistryMixin(ClassRegistryMixin, AutoImporterMixin):
    """
    An extended registry mixin that combines class registration with auto-importing.

    This mixin inherits from both ClassRegistryMixin and AutoImporterMixin to provide
    automatic discovery and registration of classes from specified packages. Classes
    that use this mixin can define an auto_package class variable to specify which
    package(s) should be automatically imported to discover implementations.

    The mixin ensures that the registry is automatically populated with all compatible
    implementations when queried, without requiring explicit imports of each module.

    Example:
    ```python
    class TokenProposalConfig(AutoClassRegistryMixin):
        auto_package = "speculators.proposals"

    # This will automatically import all modules in the proposals package
    # and register any classes decorated with @TokenProposalConfig.register()
    proposals = TokenProposalConfig.registered_classes()
    ```

    :cvar registry_populated: A class variable that tracks whether the registry has
        been populated with classes from the specified package(s).
    """

    registry_populated: ClassVar[bool] = False

    @classmethod
    def auto_populate_registry(cls):
        """
        Ensures that all modules in the specified auto_package are imported.

        This method is called automatically by registered_classes to ensure that
        all available implementations are discovered and registered before returning
        the list of registered classes.
        """
        if not cls.registry_populated:
            cls.auto_import_package_modules()
        cls.registry_populated = True

    @classmethod
    def registered_classes(cls) -> tuple[type[Any], ...]:
        """
        Returns all registered classes after ensuring the registry is populated.

        This method overrides the parent class method to ensure that auto-importing
        occurs before returning the registered classes.

        :return: A tuple containing all registered class implementations, including
            those discovered through auto-importing.
        :raises ValueError: If no classes have been registered after auto-importing.
        """
        cls.auto_populate_registry()

        return super().registered_classes()
