"""
Copy required files from outside of the docs directory into the docs directory
for the documentation build and site.
Uses mkdocs-gen-files to handle the file generation and compatibility with MkDocs.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import mkdocs_gen_files


@dataclass
class ProcessFile:
    root_path: Path
    docs_path: Path
    title: str
    weight: float


def find_project_root() -> Path:
    start_path = Path(__file__).absolute()
    current_path = start_path.parent

    while current_path:
        if (current_path / "mkdocs.yml").exists():
            return current_path
        current_path = current_path.parent

    raise FileNotFoundError(
        f"Could not find mkdocs.yml in the directory tree starting from {start_path}"
    )


GITHUB_BASE = "https://github.com/vllm-project/speculators/blob/main"


def remap_links(
    content: str,
    source_root_path: Path,
    files: list[ProcessFile],
):
    """Remap relative links to docs pages or GitHub absolute URLs.

    For each markdown link, resolve it relative to the source file's location
    in the repo. If the resolved path (or a README.md inside it) is a known
    ProcessFile, rewrite the link to point to the corresponding docs page.
    Otherwise, rewrite it as an absolute GitHub link.
    """
    source_dir = source_root_path.parent
    root_to_docs = {str(f.root_path): str(f.docs_path) for f in files}

    def resolve_link(link_target: str) -> str | None:
        # Split off any anchor
        anchor = ""
        if "#" in link_target:
            link_target, anchor = link_target.rsplit("#", 1)
            anchor = f"#{anchor}"

        if not link_target:
            return None

        # Resolve the path relative to the source file's directory,
        # or treat absolute paths (starting with /) as repo-root-relative
        starts_with_slash = False
        if link_target.startswith("/"):
            resolved = Path(link_target.lstrip("/"))
            starts_with_slash = True
        else:
            resolved = source_dir / link_target

        # Normalize (resolve .., etc) without requiring the path to exist
        resolved = Path(re.sub(r"(^|/)(\./)+", r"\1", str(resolved)))

        # Check if this path maps to a docs page
        resolved_str = str(resolved)
        if resolved_str in root_to_docs:
            resolved_target = root_to_docs[resolved_str] + anchor
            if starts_with_slash:
                resolved_target = "/" + resolved_target
            return resolved_target

        # Otherwise, link to GitHub if the path looks like a real file
        return f"{GITHUB_BASE}/{resolved_str}{anchor}"

    def replace_match(match: re.Match) -> str:
        link_text = match.group(1)
        link_target = match.group(2)

        # Skip absolute URLs and pure anchors
        if link_target.startswith(("http://", "https://", "#")):
            return match.group(0)

        new_target = resolve_link(link_target)
        if new_target is None:
            return match.group(0)
        return f"[{link_text}]({new_target})"

    return re.sub(r"\[([^\]]*)\]\(([^)]+)\)", replace_match, content)


def process_files(files: list[ProcessFile], project_root: Path):
    for file in files:
        source_path = project_root / file.root_path
        target_path = file.docs_path

        if not source_path.exists():
            raise FileNotFoundError(
                f"Source file {source_path} does not exist for copying into docs "
                f"directory at {target_path}"
            )

        frontmatter = f"---\ntitle: {file.title}\nweight: {file.weight}\n---\n\n"
        content = source_path.read_text(encoding="utf-8")
        content = remap_links(content, file.root_path, files)

        with mkdocs_gen_files.open(target_path, "w") as file_handle:
            file_handle.write(frontmatter)
            file_handle.write(content)

        mkdocs_gen_files.set_edit_path(target_path, source_path)


def migrate_developer_docs():
    project_root = find_project_root()
    files = [
        # Developer
        ProcessFile(
            root_path=Path("CODE_OF_CONDUCT.md"),
            docs_path=Path("developer/code-of-conduct.md"),
            title="Code of Conduct",
            weight=-10,
        ),
        ProcessFile(
            root_path=Path("CONTRIBUTING.md"),
            docs_path=Path("developer/contributing.md"),
            title="Contributing Guide",
            weight=-8,
        ),
        # Examples
        ProcessFile(
            root_path=Path("examples/data_generation_and_training/README.md"),
            docs_path=Path("examples/data_generation_and_training.md"),
            title="Train",
            weight=1,
        ),
        ProcessFile(
            root_path=Path("scripts/response_regeneration/README.md"),
            docs_path=Path("response_regeneration.md"),
            title="Response Regeneration",
            weight=2,
        ),
        ProcessFile(
            root_path=Path("examples/convert/README.md"),
            docs_path=Path("examples/convert.md"),
            title="Convert",
            weight=3,
        ),
        ProcessFile(
            root_path=Path("examples/evaluate/eval-guidellm/README.md"),
            docs_path=Path("examples/evaluate.md"),
            title="Evaluate",
            weight=4,
        ),
        ProcessFile(
            root_path=Path("scripts/README.md"),
            docs_path=Path("train.md"),
            title="Train",
            weight=-6,
        ),
    ]
    process_files(files, project_root)


migrate_developer_docs()
