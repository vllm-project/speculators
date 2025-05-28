"""
Unit tests for the auto_importer module in the Speculators library.
"""

from unittest import mock

import pytest

from speculators.utils.auto_importer import AutoImporterMixin

# ===== Basic Functionality Tests =====


@pytest.mark.smoke
def test_auto_importer_initialization():
    class TestAutoImporterClass(AutoImporterMixin):
        auto_package = "test_package.modules"

    assert AutoImporterMixin.auto_package is None
    assert AutoImporterMixin.auto_ignore_modules is None
    assert AutoImporterMixin.auto_imported_modules is None


@pytest.mark.smoke
def test_auto_importer_subclass_attributes():
    class TestAutoImporterClass(AutoImporterMixin):
        auto_package = "test_package.modules"

    assert TestAutoImporterClass.auto_package == "test_package.modules"
    assert TestAutoImporterClass.auto_ignore_modules is None
    assert TestAutoImporterClass.auto_imported_modules is None


@pytest.mark.smoke
def test_no_package_raises_error():
    class TestAutoImporterClass(AutoImporterMixin): ...

    with pytest.raises(ValueError) as exc_info:
        TestAutoImporterClass.auto_import_package_modules()

    assert "auto_package" in str(exc_info.value)
    assert "must be set" in str(exc_info.value)


# ===== Module Import Tests =====


@pytest.mark.smoke
def test_single_package_import():
    class TestAutoImporterClass(AutoImporterMixin):
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
                return module
            elif name == "test_package.modules.module2":
                module = mock.MagicMock()
                module.__name__ = "test_package.modules.module2"
                return module
            else:
                raise ImportError(f"No module named {name}")

        def walk_packages(package_path, package_name):
            if package_name == "test_package.modules.":
                return [
                    (None, "test_package.modules.module1", False),
                    (None, "test_package.modules.module2", False),
                ]
            else:
                raise ValueError(f"Unknown package: {package_name}")

        mock_walk.side_effect = walk_packages
        mock_import.side_effect = import_module
        TestAutoImporterClass.auto_import_package_modules()

        mock_import.assert_any_call("test_package.modules")
        assert TestAutoImporterClass.auto_imported_modules == [
            "test_package.modules.module1",
            "test_package.modules.module2",
        ]


@pytest.mark.sanity
def test_multiple_package_import():
    class TestAutoImporterClass(AutoImporterMixin):
        auto_package = ("test_package.modules1", "test_package.modules2")

    with (
        mock.patch("pkgutil.walk_packages") as mock_walk,
        mock.patch("importlib.import_module") as mock_import,
    ):
        # Create a mock package with the necessary attributes
        mock_package1 = mock.MagicMock()
        mock_package1.__path__ = ["test_package/modules1"]
        mock_package1.__name__ = "test_package.modules1"

        mock_package2 = mock.MagicMock()
        mock_package2.__path__ = ["test_package/modules2"]
        mock_package2.__name__ = "test_package.modules2"

        def import_module(name: str):
            if name == "test_package.modules1":
                return mock_package1
            elif name == "test_package.modules2":
                return mock_package2
            elif name == "test_package.modules1.moduleA":
                module = mock.MagicMock()
                module.__name__ = "test_package.modules1.moduleA"
                return module
            elif name == "test_package.modules2.moduleB":
                module = mock.MagicMock()
                module.__name__ = "test_package.modules2.moduleB"
                return module
            else:
                raise ImportError(f"No module named {name}")

        def walk_packages(package_path, package_name):
            if package_name == "test_package.modules1.":
                return [
                    (None, "test_package.modules1.moduleA", False),
                ]
            elif package_name == "test_package.modules2.":
                return [
                    (None, "test_package.modules2.moduleB", False),
                ]
            else:
                raise ValueError(f"Unknown package: {package_name}")

        mock_walk.side_effect = walk_packages
        mock_import.side_effect = import_module
        TestAutoImporterClass.auto_import_package_modules()

        assert TestAutoImporterClass.auto_imported_modules == [
            "test_package.modules1.moduleA",
            "test_package.modules2.moduleB",
        ]


@pytest.mark.sanity
def test_ignore_modules():
    class TestAutoImporterClass(AutoImporterMixin):
        auto_package = "test_package.modules"
        auto_ignore_modules = ("test_package.modules.module1",)

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
                return module
            elif name == "test_package.modules.module2":
                module = mock.MagicMock()
                module.__name__ = "test_package.modules.module2"
                return module
            else:
                raise ImportError(f"No module named {name}")

        def walk_packages(package_path, package_name):
            if package_name == "test_package.modules.":
                return [
                    (None, "test_package.modules.module1", False),
                    (None, "test_package.modules.module2", False),
                ]
            else:
                raise ValueError(f"Unknown package: {package_name}")

        mock_walk.side_effect = walk_packages
        mock_import.side_effect = import_module
        TestAutoImporterClass.auto_import_package_modules()

        assert TestAutoImporterClass.auto_imported_modules == [
            "test_package.modules.module2",
        ]
