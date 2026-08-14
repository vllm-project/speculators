import os
import re
from pathlib import Path

from packaging.version import Version
from setuptools import setup
from setuptools_git_versioning import count_since, get_branch, get_sha, get_tags

REPO_ROOT = Path(__file__).parent.parent
LAST_RELEASE_VERSION = Version("0.0.1")
TAG_VERSION_PATTERN = re.compile(r"^hsc-v(\d+\.\d+\.\d+)$")


def get_last_version_diff() -> tuple[Version, str | None, int]:
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
        count_since(f"{last_tag}^{{commit}}", root=REPO_ROOT) if last_tag else 0
    )
    print("IN LAST")
    print(f"tagged_versions={tagged_versions}")
    print(f"last_version={last_version}")
    print(f"last_tag={last_tag}")
    print(f"commits_since_last={commits_since_last}")
    return last_version, last_tag, commits_since_last


def get_next_version(build_type: str) -> tuple[Version, str | None, int]:
    version, tag, commits_since_last = get_last_version_diff()

    print("HERE in NEXT")
    print(f"REPO_ROOT={REPO_ROOT}")
    print(f"version={version}")
    print(f"tag={tag}")
    print(f"commits_since_last={commits_since_last}")
    print(f"build_type={build_type}")

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
        # e.g. hs_connectors/0.1.0 + 3 commits -> 0.1.0a3; on tag -> 0.1.0a0
        return Version(f"{version}.a{commits_since_last}"), tag, commits_since_last

    raise ValueError(f"Unsupported HS_CONNECTORS_BUILD_TYPE={build_type!r}")


def read_existing_version(module_path: Path) -> Version | None:
    version_txt = module_path / "version.txt"
    if version_txt.exists():
        return Version(version_txt.read_text().strip())
    return None


def write_version_files() -> tuple[Path, Path]:
    build_type = os.getenv("HS_CONNECTORS_BUILD_TYPE", "nightly").lower()
    module_path = Path(__file__).parent / "src" / "hs_connectors"

    existing_version = read_existing_version(module_path)
    print(f"EXISTING VERSION={existing_version}")
    if existing_version is not None:
        version, tag, build_iteration = existing_version, None, 0
    else:
        version, tag, build_iteration = get_next_version(build_type)

    print("IN WRITE")
    print(f"build_type={build_type}")
    print(f"{version}")
    print(f"{tag}")
    print(f"build_iteration={build_iteration}")

    version_txt_path = module_path / "version.txt"
    version_py_path = module_path / "version.py"

    with version_txt_path.open("w") as f:
        f.write(str(version))
    with version_py_path.open("w") as f:
        f.writelines([
            f'version = "{version}"\n',
            f'build_type = "{build_type}"\n',
            f'build_iteration = "{build_iteration}"\n',
            f'git_commit = "{get_sha(root=REPO_ROOT)}"\n',
            f'git_branch = "{get_branch(root=REPO_ROOT)}"\n',
            f'git_last_tag = "{tag or ""}"\n',
        ])
    return version_txt_path, version_py_path


setup(
    setuptools_git_versioning={
        "enabled": True,
        "version_file": str(write_version_files()[0]),
    }
)
