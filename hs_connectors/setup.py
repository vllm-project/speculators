import os
import re
from pathlib import Path

from packaging.version import Version
from setuptools import setup
from setuptools_git_versioning import count_since, get_branch, get_sha, get_tags

REPO_ROOT = Path(__file__).parent.parent
INITIAL_RELEASE_VERSION = Version("0.7.0")
TAG_VERSION_PATTERN = re.compile(r"^hsc-v(\d+\.\d+\.\d+)$")


def get_last_version_diff() -> tuple[Version, str | None, int]:
    tagged_versions = [
        (Version(match.group(1)), tag)
        for tag in get_tags(root=REPO_ROOT)
        if (match := TAG_VERSION_PATTERN.match(tag))
    ]
    tagged_versions.sort(key=lambda tv: tv[0])
    last_version, last_tag = (
        tagged_versions[-1] if tagged_versions else (INITIAL_RELEASE_VERSION, None)
    )
    commits_since_last = (
        count_since(f"{last_tag}^{{commit}}", root=REPO_ROOT) if last_tag else 0
    )
    return last_version, last_tag, commits_since_last


def get_next_version(build_type: str) -> tuple[Version, str | None, int]:
    version, tag, commits_since_last = get_last_version_diff()

    if build_type == "release":
        if not tag:
            raise ValueError("RELEASE build requires an hsc-vX.Y.Z tag")
        if commits_since_last:
            raise ValueError(
                f"RELEASE build must be on tag {tag}; "
                f"HEAD is {commits_since_last} commit(s) ahead"
            )
        return version, tag, 0

    if build_type == "nightly":
        # nightly version will be patch+1 from last release tag:
        # e.g. tag hsc-v0.1.0 + 3 commits -> 0.1.1a3
        return (
            Version(
                f"{version.major}.{version.minor}.{version.micro + 1}"
                f".a{commits_since_last}"
            ),
            tag,
            commits_since_last,
        )

    raise ValueError(f"Unsupported HS_CONNECTORS_BUILD_TYPE={build_type!r}")


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
    # sdist extracts as hs_connectors-<version>/setup.py
    return Path(__file__).parent.name.startswith("hs_connectors-")


def write_version_files() -> tuple[Path, Path]:
    build_type = os.getenv("HS_CONNECTORS_BUILD_TYPE", "nightly").lower()
    module_path = Path(__file__).parent / "src" / "hs_connectors"
    version_txt_path = module_path / "version.txt"
    version_py_path = module_path / "version.py"

    if building_from_sdist() and version_py_path.exists():
        version, tag, build_iteration = read_existing_version(version_py_path)
    else:
        version, tag, build_iteration = get_next_version(build_type)

    git_commit = get_sha(root=REPO_ROOT) if (not building_from_sdist()) else ""
    git_branch = get_branch(root=REPO_ROOT) if (not building_from_sdist()) else ""

    with version_txt_path.open("w") as f:
        f.write(str(version))
    with version_py_path.open("w") as f:
        f.writelines(
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


setup(
    setuptools_git_versioning={
        "enabled": True,
        "version_file": str(write_version_files()[0]),
    }
)
