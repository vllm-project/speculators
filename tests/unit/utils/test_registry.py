"""
Unit tests for the registry module in the Speculators library.
"""

import pytest

from speculators.utils.registry import ClassRegistryMixin

# ===== ClassRegistryMixin Tests =====


@pytest.mark.smoke
def test_class_registry_initialization():
    class TestRegistryClass(ClassRegistryMixin):
        pass

    assert TestRegistryClass.registry is None


@pytest.mark.smoke
def test_register_with_name():
    class TestRegistryClass(ClassRegistryMixin):
        pass

    @TestRegistryClass.register("custom_name")
    class TestClass:
        pass

    assert TestRegistryClass.registry is not None
    assert "custom_name" in TestRegistryClass.registry
    assert TestRegistryClass.registry["custom_name"] is TestClass


@pytest.mark.smoke
def test_register_without_name():
    class TestRegistryClass(ClassRegistryMixin):
        pass

    @TestRegistryClass.register()
    class TestClass:
        pass

    assert TestRegistryClass.registry is not None
    assert "TestClass" in TestRegistryClass.registry
    assert TestRegistryClass.registry["TestClass"] is TestClass


@pytest.mark.smoke
def test_register_decorator_direct():
    class TestRegistryClass(ClassRegistryMixin):
        pass

    @TestRegistryClass.register_decorator
    class TestClass:
        pass

    assert TestRegistryClass.registry is not None
    assert "TestClass" in TestRegistryClass.registry
    assert TestRegistryClass.registry["TestClass"] is TestClass


@pytest.mark.sanity
def test_register_invalid_name_type():
    class TestRegistryClass(ClassRegistryMixin):
        pass

    with pytest.raises(ValueError) as exc_info:
        TestRegistryClass.register(123)  # type: ignore[arg-type]

    assert "name must be a string or None" in str(exc_info.value)


@pytest.mark.sanity
def test_register_decorator_invalid_class():
    class TestRegistryClass(ClassRegistryMixin):
        pass

    with pytest.raises(TypeError) as exc_info:
        TestRegistryClass.register_decorator("not_a_class")  # type: ignore[arg-type]

    assert "must be used as a class decorator" in str(exc_info.value)


@pytest.mark.sanity
def test_register_decorator_invalid_name():
    class TestRegistryClass(ClassRegistryMixin):
        pass

    class TestClass:
        pass

    with pytest.raises(ValueError) as exc_info:
        TestRegistryClass.register_decorator(TestClass, name=123)  # type: ignore[arg-type]

    assert "must be used as a class decorator" in str(exc_info.value)


@pytest.mark.sanity
def test_register_duplicate_name():
    class TestRegistryClass(ClassRegistryMixin):
        pass

    @TestRegistryClass.register("test_name")
    class TestClass1:
        pass

    with pytest.raises(ValueError) as exc_info:

        @TestRegistryClass.register("test_name")
        class TestClass2:
            pass

    assert "already registered" in str(exc_info.value)


@pytest.mark.sanity
def test_registered_classes_empty():
    class TestRegistryClass(ClassRegistryMixin):
        pass

    with pytest.raises(ValueError) as exc_info:
        TestRegistryClass.registered_classes()

    assert "must be called after registering classes" in str(exc_info.value)


@pytest.mark.sanity
def test_registered_classes():
    class TestRegistryClass(ClassRegistryMixin):
        pass

    @TestRegistryClass.register()
    class TestClass1:
        pass

    @TestRegistryClass.register("custom_name")
    class TestClass2:
        pass

    registered = TestRegistryClass.registered_classes()
    assert isinstance(registered, tuple)
    assert len(registered) == 2
    assert TestClass1 in registered
    assert TestClass2 in registered


@pytest.mark.regression
def test_multiple_registries_isolation():
    class Registry1(ClassRegistryMixin):
        pass

    class Registry2(ClassRegistryMixin):
        pass

    @Registry1.register()
    class TestClass1:
        pass

    @Registry2.register()
    class TestClass2:
        pass

    assert Registry1.registry is not None
    assert Registry2.registry is not None
    assert Registry1.registry != Registry2.registry
    assert "TestClass1" in Registry1.registry
    assert "TestClass2" in Registry2.registry
    assert "TestClass1" not in Registry2.registry
    assert "TestClass2" not in Registry1.registry


# ===== Auto-Discovery Tests =====


@pytest.mark.smoke
def test_auto_discovery_registry_initialization():
    class TestAutoRegistry(ClassRegistryMixin):
        registry_auto_discovery = True
        auto_package = "test_package.modules"

    assert TestAutoRegistry.registry is None
    assert TestAutoRegistry.registry_populated is False
    assert TestAutoRegistry.auto_package == "test_package.modules"
    assert TestAutoRegistry.registry_auto_discovery is True
