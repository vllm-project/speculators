import hashlib
import importlib.metadata
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def get_git_sha(repo_root: str | Path) -> str:
    """Get the current git SHA of the repository."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_root),
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

def get_git_diff(repo_root: str | Path) -> str:
    """Get the uncommitted git diff of the repository."""
    try:
        return subprocess.run(
            ["git", "diff", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_root),
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""

def get_package_versions(packages: list[str]) -> list[str]:
    """Get version strings for the given packages."""
    pkg_versions: list[str] = []
    for pkg in packages:
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "not installed"
        pkg_versions.append(f"# {pkg}: {ver}")
    return pkg_versions

def hash_safetensors(model_dir: str | Path) -> str:
    """Compute SHA256 hashes of all .safetensors files in a directory."""
    path = Path(model_dir)
    if not path.exists() or not path.is_dir():
        return f"# {model_dir} is a remote reference or not a local directory."
    
    safetensors_files = sorted(path.rglob("*.safetensors"))
    if not safetensors_files:
        return f"# No .safetensors files found in {model_dir}"

    hashes = []
    for f in safetensors_files:
        sha256 = hashlib.sha256()
        with open(f, "rb") as file_obj:
            while chunk := file_obj.read(8192):
                sha256.update(chunk)
        hashes.append(f"{sha256.hexdigest()}  {f.relative_to(path)}")
    return "\n".join(hashes)

def write_atomic(save_path: str | Path, filename: str, content: str) -> None:
    """Write content to save_path/filename atomically."""
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=save_path, prefix=f".{filename}_", suffix=".tmp")
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        tmp_path.replace(path / filename)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

def get_package_root(package_name: str) -> Optional[Path]:
    """Return the source root directory of a package if it exists."""
    try:
        import importlib
        pkg = importlib.import_module(package_name)
        if hasattr(pkg, "__file__") and pkg.__file__:
            return Path(pkg.__file__).parent
    except ImportError:
        pass
    return None
