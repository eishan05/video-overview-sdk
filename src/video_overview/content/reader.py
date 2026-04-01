"""Content reader for directory content ingestion."""

from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath

import pathspec

# Extensions that are always skipped (binary/compiled)
_SKIP_EXTENSIONS = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".egg-info",
        ".whl",
        ".tar",
        ".gz",
        ".zip",
        ".jar",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wav",
        ".flac",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".o",
        ".a",
        ".lib",
        ".class",
        ".sqlite",
        ".db",
    }
)

# Directories that are always skipped
_SKIP_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".egg-info",
        "dist",
        ".eggs",
        ".venv",
        "venv",
        "env",
    }
)

# Extension to language mapping
_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".R": "r",
    ".lua": "lua",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".xml": "xml",
    ".csv": "csv",
    ".dockerfile": "dockerfile",
    ".tf": "terraform",
    ".hcl": "hcl",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".vue": "vue",
    ".svelte": "svelte",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".clj": "clojure",
    ".dart": "dart",
    ".jl": "julia",
    ".m": "objective-c",
    ".pl": "perl",
    ".pm": "perl",
    ".makefile": "makefile",
}

# Max characters to keep from a single file before truncation
_FILE_TRUNCATION_LIMIT = 2000


def _detect_language(file_path: Path) -> str:
    """Detect programming language from file extension."""
    name_lower = file_path.name.lower()

    # Handle special filenames
    if name_lower in ("makefile", "gnumakefile"):
        return "makefile"
    if name_lower in ("dockerfile",):
        return "dockerfile"
    if name_lower in (".gitignore", ".dockerignore"):
        return "gitignore"

    suffix = file_path.suffix.lower()
    return _EXTENSION_TO_LANGUAGE.get(suffix, "text")


def _is_binary(content_bytes: bytes) -> bool:
    """Check if content appears to be binary by looking for null bytes."""
    return b"\x00" in content_bytes[:8192]


def _matches_any(rel_path: str, patterns: list[str]) -> bool:
    """Check if a POSIX relative path matches any of the given patterns.

    For patterns containing a ``/`` (path-based), the pattern is
    anchored to the source-dir root via ``pathspec`` gitignore
    matching.  For basename-only patterns (e.g. ``*.py``), uses
    ``fnmatch`` against the filename component.
    """
    basename = PurePosixPath(rel_path).name
    for pat in patterns:
        if "/" in pat:
            # Path-based pattern — use pathspec for root-anchored,
            # segment-aware matching (supports * and ** correctly).
            spec = pathspec.PathSpec.from_lines("gitignore", [pat])
            if spec.match_file(rel_path):
                return True
        else:
            # Basename-only pattern
            if fnmatch.fnmatch(basename, pat):
                return True
    return False


def _file_sort_key(rel_path: str) -> tuple[int, str]:
    """Return a sort key that puts README first, docs second, source last.

    Lower numbers sort first.
    """
    parts = rel_path.split("/")
    filename = parts[-1].lower()

    # README files first (priority 0)
    if filename.startswith("readme"):
        return (0, rel_path)

    # Documentation files (priority 1)
    if any(p.lower() in ("docs", "doc", "documentation") for p in parts[:-1]):
        return (1, rel_path)
    if filename.endswith((".md", ".rst", ".txt")) and not filename.startswith("readme"):
        return (2, rel_path)

    # Configuration files (priority 3)
    if filename.endswith((".toml", ".yaml", ".yml", ".json", ".cfg", ".ini")):
        return (3, rel_path)

    # Source files (priority 4)
    return (4, rel_path)


def _build_tree(source_dir: Path, entries: list[str]) -> str:
    """Build an indented tree string from a list of relative paths."""
    if not entries:
        return str(source_dir.name) + "/\n"

    lines: list[str] = [f"{source_dir.name}/"]

    # Collect unique directory paths and files
    all_items: set[str] = set()
    for entry in entries:
        parts = entry.split("/")
        # Add all parent directory prefixes
        for i in range(1, len(parts)):
            all_items.add("/".join(parts[:i]) + "/")
        all_items.add(entry)

    for item in sorted(all_items):
        depth = item.count("/")
        if item.endswith("/"):
            # Directory
            depth = item.rstrip("/").count("/")
            name = item.rstrip("/").split("/")[-1]
            lines.append("  " * (depth + 1) + name + "/")
        else:
            # File
            name = item.split("/")[-1]
            lines.append("  " * depth + name)

    return "\n".join(lines) + "\n"


class ContentReader:
    """Reads and bundles directory contents for analysis."""

    def read(
        self,
        source_dir: str | Path,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        max_chars: int = 100_000,
    ) -> dict:
        """Walk a directory tree and build a structured content bundle.

        Args:
            source_dir: Root directory to read.
            include: Glob patterns for files to include. If provided, only
                matching files are included.
            exclude: Glob patterns for files to exclude.
            max_chars: Maximum total characters to include across all files.

        Returns:
            A dict with keys ``directory_structure``, ``files``,
            ``total_files``, and ``total_chars``.
        """
        source_dir = Path(source_dir).resolve()

        if not source_dir.exists():
            raise FileNotFoundError(f"source_dir does not exist: {source_dir}")
        if not source_dir.is_dir():
            raise NotADirectoryError(f"source_dir is not a directory: {source_dir}")

        # Load .gitignore if present
        gitignore_spec = self._load_gitignore(source_dir)

        # Collect all candidate files
        candidates: list[Path] = []
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue

            # Skip files inside always-skipped directories
            rel_parts = path.relative_to(source_dir).parts
            if any(
                part in _SKIP_DIRS or part.endswith(".egg-info") for part in rel_parts
            ):
                continue

            # Skip files with known binary extensions
            if path.suffix.lower() in _SKIP_EXTENSIONS:
                continue

            rel_path_str = path.relative_to(source_dir).as_posix()

            # Apply .gitignore
            if gitignore_spec and gitignore_spec.match_file(rel_path_str):
                continue

            # Apply include filter — use path-aware matching so that
            # patterns like "src/*.py" respect directory segments.
            if include and not _matches_any(rel_path_str, include):
                continue

            # Apply exclude filter
            if exclude and _matches_any(rel_path_str, exclude):
                continue

            candidates.append(path)

        # Read file contents, skipping binary files
        file_entries: list[dict] = []
        for path in candidates:
            try:
                raw = path.read_bytes()
            except OSError:
                continue

            if _is_binary(raw):
                continue

            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue

            rel_path = path.relative_to(source_dir).as_posix()
            language = _detect_language(path)

            # Truncate large files
            if len(text) > _FILE_TRUNCATION_LIMIT:
                text = (
                    text[:_FILE_TRUNCATION_LIMIT]
                    + f"\n\n... [truncated — {len(text)} total chars]"
                )

            file_entries.append(
                {
                    "path": rel_path,
                    "content": text,
                    "language": language,
                }
            )

        # Sort files by relevance
        file_entries.sort(key=lambda f: _file_sort_key(f["path"]))

        # Build directory structure tree from ALL candidates (before
        # budget trimming) so the tree reflects the full repository.
        all_rel_paths = [e["path"] for e in file_entries]
        tree = _build_tree(source_dir, all_rel_paths)

        # Enforce max_chars budget
        _BUDGET_SUFFIX = "\n\n... [truncated to fit budget]"
        budget = max_chars
        kept: list[dict] = []
        for entry in file_entries:
            content_len = len(entry["content"])
            if content_len <= budget:
                kept.append(entry)
                budget -= content_len
            else:
                # Truncate this last file to fit within budget.
                if budget > 0:
                    suffix_len = len(_BUDGET_SUFFIX)
                    available = budget - suffix_len
                    if available > 0:
                        # Room for content + suffix
                        entry["content"] = entry["content"][:available] + _BUDGET_SUFFIX
                    else:
                        # Very small budget — include raw prefix only
                        entry["content"] = entry["content"][:budget]
                    kept.append(entry)
                break

        total_chars = sum(len(e["content"]) for e in kept)

        return {
            "directory_structure": tree,
            "files": kept,
            "total_files": len(kept),
            "total_chars": total_chars,
        }

    def _load_gitignore(self, source_dir: Path) -> pathspec.PathSpec | None:
        """Load and parse .gitignore from the source directory."""
        gitignore_path = source_dir / ".gitignore"
        if not gitignore_path.is_file():
            return None

        try:
            text = gitignore_path.read_text(encoding="utf-8")
            return pathspec.PathSpec.from_lines("gitignore", text.splitlines())
        except (OSError, UnicodeDecodeError):
            return None
