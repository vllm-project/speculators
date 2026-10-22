"""Class registries populated by decorators when implementation modules import."""

from collections.abc import Callable
from typing import Any, ClassVar

__all__ = ["ClassRegistryMixin"]


class ClassRegistryMixin:
    """
    A mixin class that provides a registration system for tracking class
    implementations registered explicitly through decorators.

    This mixin allows classes to maintain a registry of subclasses that can be
    looked up and instantiated. Classes that inherit from this mixin
    can use the @register decorator to add themselves to the registry.

    The registry is class-specific, meaning each class that inherits from this mixin
    will have its own separate registry of implementations.

    Import implementation modules before looking up their registered classes.

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

    :cvar registry: Maps registration names to implementation classes.
    :cvar registry_auto_discovery: Legacy compatibility flag; does not trigger imports.
    :cvar registry_populated: Legacy compatibility state; unused by registration.
    """

    registry: ClassVar[dict[str, type[Any]] | None] = None
    registry_auto_discovery: ClassVar[bool] = False
    registry_populated: ClassVar[bool] = False

    @classmethod
    def register(cls, name: str | None = None) -> Callable[[type[Any]], type[Any]]:
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
    def register_decorator(cls, clazz: type[Any], name: str | None = None) -> type[Any]:
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

        # A subclass inherits its parent's registry attribute until it writes its
        # own. Allocate a new registry before registering so a subclass cannot
        # mutate its parent's registrations.
        if "registry" not in cls.__dict__ or cls.registry is None:
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

        Implementation modules must already have been imported.

        :return: A tuple containing all registered implementation classes.
        :raises ValueError: If called before any classes have been registered.
        """

        if cls.registry is None:
            raise ValueError(
                "ClassRegistryMixin.registered_classes() must be called after "
                "registering classes with ClassRegistryMixin.register()."
            )

        return tuple(cls.registry.values())
