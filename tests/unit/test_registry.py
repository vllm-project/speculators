"""
Unit tests for the registry module in the Speculators library.
"""

from unittest import mock

import pytest

from speculators.utils.registry import AutoClassRegistryMixin, ClassRegistryMixin

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


# ===== AutoClassRegistryMixin Tests =====


@pytest.mark.smoke
def test_auto_class_registry_initialization():
    class TestAutoRegistry(AutoClassRegistryMixin):
        auto_package = "test_package.modules"

    assert TestAutoRegistry.registry is None
    assert TestAutoRegistry.registry_populated is False
    assert TestAutoRegistry.auto_package == "test_package.modules"


@pytest.mark.smoke
def test_auto_populate_registry():
    class TestAutoRegistry(AutoClassRegistryMixin):
        auto_package = "test_package.modules"

    with mock.patch.object(
        TestAutoRegistry, "auto_import_package_modules"
    ) as mock_import:
        TestAutoRegistry.auto_populate_registry()
        mock_import.assert_called_once()
        assert TestAutoRegistry.registry_populated is True

        TestAutoRegistry.auto_populate_registry()
        mock_import.assert_called_once()


@pytest.mark.sanity
def test_auto_registered_classes():
    class TestAutoRegistry(AutoClassRegistryMixin):
        auto_package = "test_package.modules"

    with (
        mock.patch.object(TestAutoRegistry, "auto_populate_registry") as mock_populate,
        mock.patch.object(
            ClassRegistryMixin, "registered_classes", return_value=("class1", "class2")
        ) as mock_parent_registered,
    ):
        classes = TestAutoRegistry.registered_classes()
        mock_populate.assert_called_once()
        mock_parent_registered.assert_called_once()
        assert classes == ("class1", "class2")


@pytest.mark.regression
def test_auto_registry_integration():
    class TestAutoRegistry(AutoClassRegistryMixin):
        auto_package = "test_package.modules"

    with (
        mock.patch("pkgutil.walk_packages") as mock_walk,
        mock.patch("importlib.import_module") as mock_import,
    ):
        # Create a mock package with the necessary attributes
        mock_package = mock.MagicMock()
        mock_package.__path__ = ["test_package/modules"]
        mock_package.__name__ = "test_package.modules"

        def import_module(name: str):
            if name == "test_package.modules":
                return mock_package
            elif name == "test_package.modules.module1":
                module = mock.MagicMock()
                module.__name__ = "test_package.modules.module1"

                class Module1Class:
                    pass

                TestAutoRegistry.register_decorator(Module1Class, "Module1Class")
                return module
            else:
                raise ImportError(f"No module named {name}")

        def walk_packages(package_path, package_name):
            if package_name == "test_package.modules.":
                return [(None, "test_package.modules.module1", False)]
            else:
                raise ValueError(f"Unknown package: {package_name}")

        mock_walk.side_effect = walk_packages
        mock_import.side_effect = import_module

        classes = TestAutoRegistry.registered_classes()
        assert len(classes) == 1
        assert TestAutoRegistry.registry_populated is True
        assert TestAutoRegistry.registry is not None
        assert "Module1Class" in TestAutoRegistry.registry


@pytest.mark.regression
def test_auto_registry_with_multiple_packages():
    class TestMultiPackageRegistry(AutoClassRegistryMixin):
        auto_package = ("package1", "package2")

    with (
        mock.patch.object(
            TestMultiPackageRegistry, "auto_import_package_modules"
        ) as mock_import,
        mock.patch.object(
            ClassRegistryMixin, "registered_classes", return_value=("class1", "class2")
        ),
    ):
        TestMultiPackageRegistry.registered_classes()
        mock_import.assert_called_once()
        assert TestMultiPackageRegistry.registry_populated is True


@pytest.mark.regression
def test_auto_registry_no_package():
    class TestNoPackageRegistry(AutoClassRegistryMixin):
        pass

    with mock.patch.object(
        TestNoPackageRegistry,
        "auto_import_package_modules",
        side_effect=ValueError("auto_package must be set"),
    ) as mock_import:
        with pytest.raises(ValueError) as exc_info:
            TestNoPackageRegistry.auto_populate_registry()

        mock_import.assert_called_once()
        assert "auto_package must be set" in str(exc_info.value)
