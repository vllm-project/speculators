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
        and optional auto-discovery capabilities through registry_auto_discovery flag.
    AutoClassRegistryMixin: A backward-compatible version of ClassRegistryMixin with
        auto-discovery enabled by default
"""

from typing import Any, Callable, ClassVar, Optional, Union

from speculators.utils.auto_importer import AutoImporterMixin

__all__ = ["ClassRegistryMixin"]


class ClassRegistryMixin(AutoImporterMixin):
    """
    A mixin class that provides a registration system for tracking class
    implementations with optional auto-discovery capabilities.

    This mixin allows classes to maintain a registry of subclasses that can be
    dynamically discovered and instantiated. Classes that inherit from this mixin
    can use the @register decorator to add themselves to the registry.

    The registry is class-specific, meaning each class that inherits from this mixin
    will have its own separate registry of implementations.

    The mixin can also be configured to automatically discover and register classes
    from specified packages by setting registry_auto_discovery=True and defining
    an auto_package class variable to specify which package(s) should be automatically
    imported to discover implementations.

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

    Example with auto-discovery:
    ```python
    class TokenProposal(ClassRegistryMixin):
        registry_auto_discovery = True
        auto_package = "speculators.proposals"

    # This will automatically import all modules in the proposals package
    # and register any classes decorated with @TokenProposal.register()
    proposals = TokenProposal.registered_classes()
    ```

    :cvar registry: A dictionary mapping class names to classes that have been
        registered to the extending subclass through the @subclass.register() decorator
    :cvar registry_auto_discovery: A flag that enables automatic discovery and import of
        modules from the auto_package when set to True. Default is False.
    :cvar registry_populated: A flag that tracks whether the registry has been
        populated with classes from the specified package(s).
    """

    registry: ClassVar[Optional[dict[str, type[Any]]]] = None
    registry_auto_discovery: ClassVar[bool] = False
    registry_populated: ClassVar[bool] = False

    @classmethod
    def register(
        cls, name: Optional[Union[str, list[str]]] = None
    ) -> Callable[[type[Any]], type[Any]]:
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

        :param name: Optional name(s) to register the class under.
            If None, the class name is used as the registry key.
        :return: A decorator function that registers the decorated class.
        :raises ValueError: If name is provided but is not a string.
        """
        if name is not None and not isinstance(name, (str, list)):
            raise ValueError(
                "ClassRegistryMixin.register() name must be a string, list of strings, "
                f"or None. Got {name}."
            )

        return lambda subclass: cls.register_decorator(subclass, name=name)

    @classmethod
    def register_decorator(
        cls, clazz: type[Any], name: Optional[Union[str, list[str]]] = None
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
        :param name: Optional name(s) to register the class under.
            If None, the class name is used as the registry key.
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
        elif not isinstance(name, (str, list)):
            raise ValueError(
                "ClassRegistryMixin.register_decorator name must be a string or "
                f"an iterable of strings. Got {name}."
            )

        if cls.registry is None:
            cls.registry = {}

        names = [name] if isinstance(name, str) else list(name)

        for register_name in names:
            if not isinstance(register_name, str):
                raise ValueError(
                    "ClassRegistryMixin.register_decorator name must be a string or "
                    f"a list of strings. Got {register_name}."
                )

            if register_name in cls.registry:
                raise ValueError(
                    f"ClassRegistryMixin.register_decorator cannot register a class "
                    f"{clazz} with the name {register_name} because it is already "
                    "registered."
                )

            cls.registry[register_name.lower()] = clazz

        return clazz

    @classmethod
    def auto_populate_registry(cls) -> bool:
        """
        Ensures that all modules in the specified auto_package are imported.

        This method is called automatically by registered_classes when
        registry_auto_discovery==True to ensure that all available implementations are
        discovered and registered before returning the list of registered classes.

        To enable auto-discovery:
        1. Set registry_auto_discovery = True on the class
        2. Define an auto_package class variable with the package path to import

        :return: True if the registry was populated, False if it was already populated.
        :raises ValueError: If called when registry_auto_discovery is False
        """
        if not cls.registry_auto_discovery:
            raise ValueError(
                "ClassRegistryMixin.auto_populate_registry() cannot be called "
                "because registry_auto_discovery is set to False. "
                "Set registry_auto_discovery to True to enable auto-discovery."
            )

        if cls.registry_populated:
            return False

        cls.auto_import_package_modules()
        cls.registry_populated = True

        return True

    @classmethod
    def registered_classes(cls) -> tuple[type[Any], ...]:
        """
        Returns a tuple of all classes that have been registered with this registry.

        If registry_auto_discovery is True, this method will first call
        auto_populate_registry to ensure that all available implementations from
        the specified auto_package are discovered and registered before returning
        the list.

        :return: A tuple containing all registered class implementations, including
            those discovered through auto-importing when registry_auto_discovery==True.
        :raises ValueError: If called before any classes have been registered.
        """
        if cls.registry_auto_discovery:
            cls.auto_populate_registry()

        if cls.registry is None:
            raise ValueError(
                "ClassRegistryMixin.registered_classes() must be called after "
                "registering classes with ClassRegistryMixin.register()."
            )

        return tuple(cls.registry.values())
