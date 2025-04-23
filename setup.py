import os
import re
from datetime import datetime
from pathlib import Path

from packaging.version import Version
from setuptools import setup
from setuptools_git_versioning import count_since, get_branch, get_sha, get_tags

BASE_VERSION = Version("0.1.0")
TAG_VERSION_PATTERN = re.compile(r"^v(\d+\.\d+\.\d+)$")


def get_version_metadata() -> tuple[str, dict[str, str]]:
    """
    Generate the package version string and associated metadata based on git tags,
    environment variables, and the current build context.
    The function determines the version and metadata as follows:
    - Retrieves all git tags in the repository and filters those matching the version
      pattern.
    - Identifies the latest tag whose version is less than or equal to the base version.
    - Calculates the number of commits since the latest tag to determine the build
      iteration.
    - Reads the build type from the ``SPECULATORS_BUILD_TYPE`` environment variable
      (defaults to "dev").
    - Determines the build iteration from the ``SPECULATORS_BUILD_ITERATION``
      environment variable, or falls back to the number of commits since the last tag,
      or 0 if neither is available.

    The package version string is then constructed based on the build type:
    - If release then set to the base version if the current commit matches the base tag
      or the tag couldn't be determined, else set to the last tagged version with a
      ``.post{build_iteration}`` suffix.
    - If candidate then set to ``{base_version}.rc{build_iteration}``.
    - If nightly or alpha then set to ``{base_version}.a{build_iteration}``.
    - For dev (or anything else) set to ``{base_version}.dev{build_iteration}``.

    The metadata dictionary includes:
    - "version": The computed package version string.
    - "base_version": The base version string.
    - "commit": The current git commit SHA.
    - "branch": The current git branch name.
    - "build_type": The build type used for this build.
    - "build_iteration": The build iteration value.
    - "build_date": The current date in YYYY-MM-DD format.

    :returns: A tuple containing the package version string and a dictionary of metadata
    """

    version = BASE_VERSION
    metadata = {}
    tags = get_tags(root=Path(__file__).parent)
    tagged_versions = [
        (version, tag)
        for tag in tags
        if (match := TAG_VERSION_PATTERN.match(tag))
        and ((version := Version(match.group(1))) <= BASE_VERSION)
    ]
    last_ver, last_tag = max(
        tagged_versions, key=lambda tv: tv[0], default=(None, None)
    )
    commits_since_tag = (
        count_since(last_tag + "^{commit}", root=Path(__file__).parent)
        if last_tag
        else None
    )

    build_type = os.getenv("SPECULATORS_BUILD_TYPE", "dev").lower()
    build_iteration = os.getenv("SPECULATORS_BUILD_ITERATION") or commits_since_tag or 0

    if build_type == "release":
        package_version = (
            str(version)
            if not last_ver or version > last_ver
            else f"{last_ver}.post{build_iteration}"
        )
    elif build_type == "candidate":
        package_version = f"{version}.rc{build_iteration}"
    elif build_type in ["nightly", "alpha"]:
        package_version = f"{version}.a{build_iteration}"
    else:
        package_version = f"{version}.dev{build_iteration}"

    metadata = {
        "version": f'"{package_version}"',
        "base_version": f'"{BASE_VERSION}"',
        "commit": f'"{get_sha()}"',
        "branch": f'"{get_branch()}"',
        "build_type": f'"{build_type}"',
        "build_iteration": f'"{build_iteration}"',
        "build_date": f'"{datetime.now().strftime("%Y-%m-%d")}"',
    }

    return package_version, metadata


def write_module_version() -> str:
    """
    Utilizes the `get_version_metadata` function to generate the package version string
    and associated metadata, and writes them to the version.txt and version.py files
    within the src/speculators directory.

    :returns: The path to the version.txt file
    """

    version, metadata = get_version_metadata()
    module_path = Path(__file__).parent / "src" / "speculators"
    version_path = module_path / "version.txt"
    metadata_path = module_path / "version.py"

    with version_path.open("w") as file:
        file.write(version)

    with metadata_path.open("w") as file:
        file.writelines([f"{key} = {value}\n" for key, value in metadata.items()])

    return str(version_path)


setup(
    setuptools_git_versioning={
        "enabled": True,
        "version_file": write_module_version(),
    }
)
