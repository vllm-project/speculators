import os
import re
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from packaging.version import Version
from setuptools import setup
from setuptools_git_versioning import count_since, get_branch, get_sha, get_tags

REPO_ROOT = Path(__file__).parent
LAST_RELEASE_VERSION = Version("0.7.0")
TAG_VERSION_PATTERN = re.compile(r"^v(\d+\.\d+\.\d+)$")


def get_last_version_diff() -> tuple[Version, str | None, int | None]:
    """
    Get the last version, last tag, and the number of commits since the last tag.
    If no tags are found, return the last release version and None for the tag/commits.

    :returns: A tuple containing the last version, last tag, and number of commits since
        the last tag.
    """
    tagged_versions = [
        (Version(match.group(1)), tag)
        for tag in get_tags(root=REPO_ROOT)
        if (match := TAG_VERSION_PATTERN.match(tag))
    ]
    tagged_versions.sort(key=lambda tv: tv[0])
    last_version, last_tag = (
        tagged_versions[-1] if tagged_versions else (LAST_RELEASE_VERSION, None)
    )
    commits_since_last = (
        count_since(last_tag + "^{commit}", root=REPO_ROOT) if last_tag else None
    )

    return last_version, last_tag, commits_since_last


def get_next_version(
    build_type: str, build_iteration: str | int | None
) -> tuple[Version, str | None, int]:
    """
    Get the next version based on the build type and iteration.
    - build_type == release: take the last version and add a post if build iteration
    - build_type == nightly: increment to next minor, add 'a' with build iteration
    - build_type == alpha: increment to next minor, add 'a' with build iteration
    - build_type == dev: increment to next minor, add 'dev' with build iteration

    :param build_type: The type of build (release, candidate, nightly, alpha, dev).
    :param build_iteration: The build iteration number. If None, defaults to the number
        of commits since the last tag or 0 if no commits since the last tag.
    :returns: A tuple containing the next version, the last tag the version is based
        off of (if any), and the final build iteration used.
    """
    version, tag, commits_since_last = get_last_version_diff()

    if not build_iteration and build_iteration != 0:
        build_iteration = commits_since_last or 0
    elif isinstance(build_iteration, str):
        build_iteration = int(build_iteration)

    # in case tag is behind LAST_RELEASE_VERSION
    version = max(version, LAST_RELEASE_VERSION)

    if build_type == "release":
        if not tag:
            raise ValueError("RELEASE build requires a vX.Y.Z tag")
        if commits_since_last:
            raise ValueError(
                f"RELEASE build must be on tag {tag}; "
                f"HEAD is {commits_since_last} commit(s) ahead"
            )
        return version, tag, 0

    # not in release pathway, so need to increment minor to target next release version
    version = Version(f"{version.major}.{version.minor + 1}.0")

    if build_type in ["nightly", "alpha"]:
        # add 'a' since we are in nightly or alpha pathway
        version = Version(f"{version}.a{build_iteration}")
    else:
        # assume 'dev' if not in any of the above pathways
        version = Version(f"{version}.dev{build_iteration}")

    return version, tag, build_iteration


def read_existing_version(version_py: Path) -> tuple[Version, str | None, int]:
    if version_py.exists():
        text = version_py.read_text()

        match_version = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, re.M)
        match_tag = re.search(r'^git_last_tag\s*=\s*["\']([^"\']*)["\']', text, re.M)
        match_iteration = re.search(
            r'^build_iteration\s*=\s*["\']([^"\']+)["\']', text, re.M
        )

        version = Version(match_version.group(1)) if match_version else None
        tag = match_tag.group(1) if match_tag and match_tag.group(1) else None
        build_iteration = int(match_iteration.group(1)) if match_iteration else 0

    return version, tag, build_iteration


def building_from_sdist() -> bool:
    # sdist extracts as speculators-<version>/setup.py
    return REPO_ROOT.name.startswith("speculators-")


def write_version_files() -> tuple[Path, Path]:
    """
    Write the version information to version.txt and version.py files.
    version.txt contains the version string.
    version.py contains the version plus additional metadata.

    :returns: A tuple containing the paths to the version.txt and version.py files.
    """
    build_type = os.getenv("SPECULATORS_BUILD_TYPE", "dev").lower()
    module_path = REPO_ROOT / "src" / "speculators"
    version_txt_path = module_path / "version.txt"
    version_py_path = module_path / "version.py"

    if building_from_sdist() and version_py_path.exists():
        version, tag, build_iteration = read_existing_version(version_py_path)
    else:
        version, tag, build_iteration = get_next_version(
            build_type=build_type,
            build_iteration=os.getenv("SPECULATORS_BUILD_ITERATION"),
        )

    git_commit = get_sha(root=REPO_ROOT) if (not building_from_sdist()) else ""
    git_branch = get_branch(root=REPO_ROOT) if (not building_from_sdist()) else ""

    with version_txt_path.open("w") as file:
        file.write(str(version))

    with version_py_path.open("w") as file:
        file.writelines(
            [
                f'version = "{version}"\n',
                f'build_type = "{build_type}"\n',
                f'build_iteration = "{build_iteration}"\n',
                f'git_commit = "{git_commit}"\n',
                f'git_branch = "{git_branch}"\n',
                f'git_last_tag = "{tag or ""}"\n',
            ]
        )

    return version_txt_path, version_py_path


def get_hs_connectors_requirement() -> str:
    build_type = os.getenv("SPECULATORS_BUILD_TYPE", "dev").lower()
    version_py_path = REPO_ROOT / "src" / "speculators" / "version.py"

    if building_from_sdist() and version_py_path.exists():
        version, tag, build_iteration = read_existing_version(version_py_path)
    else:
        version, tag, build_iteration = get_next_version(
            build_type=build_type,
            build_iteration=os.getenv("SPECULATORS_BUILD_ITERATION"),
        )

    if build_type == "dev":
        # Source install: path dep for pip; uv workspace overrides anyway
        local = (REPO_ROOT / "hs_connectors").resolve()
        return f"hs-connectors @ file://{local.as_posix()}"
    elif build_type == "release":
        # Install release version: hs_connectors has the same version as speculators
        return f"hs-connectors=={version}"
    else:
        # Install nightly version
        return f"hs-connectors>{LAST_RELEASE_VERSION},<={version}"


def get_base_dependencies() -> list[str]:
    """
    Read the static base dependency list from pyproject.toml's
    [tool.speculators.dependencies].base table, so it stays hand-edited TOML
    rather than a Python literal duplicated here.
    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as file:
        data = tomllib.load(file)

    return data["tool"]["speculators"]["dependencies"]["base"]


setup(
    # set hs_connectors version to install
    install_requires=get_base_dependencies() + [get_hs_connectors_requirement()],
    setuptools_git_versioning={
        "enabled": True,
        "version_file": str(write_version_files()[0]),
    },
)
