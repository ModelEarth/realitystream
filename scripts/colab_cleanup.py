"""Utilities for cleaning Colab run artifacts before a new RealityStream run.

This module implements one of the open TODO items from the team: deleting
existing files that were generated in previous Colab executions so that they do
not interfere with the next run.  The helper can work as a standalone CLI or
be imported from other notebooks/scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

try:  # Optional dependency: when unavailable the CLI can still run via --extra-path.
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised in the runtime environment.
    yaml = None  # type: ignore

_PATH_KEY_SUFFIXES = (
    "path",
    "paths",
    "dir",
    "dirs",
    "directory",
    "directories",
    "folder",
    "file",
    "files",
)
_REMOTE_PREFIXES = ("gs://", "s3://", "http://", "https://")
_GLOB_CHARS = set("*?[]")


@dataclass
class CleanupResult:
    """Summary of deleted/ignored paths for reporting purposes."""

    removed: List[Path]
    skipped: List[Path]


def load_parameters_file(path: Path) -> Dict[str, Any]:
    """Load a YAML (or JSON) parameters file."""

    if not path.exists():
        raise FileNotFoundError(f"Parameters file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return {}

    if yaml is not None:
        data = yaml.safe_load(raw_text)  # type: ignore[call-arg]
        return data or {}

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - depends on runtime env.
        raise RuntimeError(
            "PyYAML is required to parse parameters.yaml; install pyyaml or pass"
            " --extra-path entries"
        ) from exc


def collect_candidate_paths(parameters: Dict[str, Any], workspace: Path) -> Set[Path]:
    """Return filesystem paths that look relevant to cleanup.

    The collector scans for dictionary keys that look like file/directory fields.
    Only local filesystem paths that are inside ``workspace`` are returned; remote
    URIs (gs://, https://, etc.) and glob expressions are ignored.
    """

    workspace = workspace.resolve()
    discovered: Set[Path] = set()

    def _walk(node: Any, *, key: Optional[str] = None) -> Iterable[str]:
        if isinstance(node, dict):
            for child_key, child_value in node.items():
                yield from _walk(child_value, key=child_key)
        elif isinstance(node, list):
            for child in node:
                yield from _walk(child, key=key)
        elif isinstance(node, str):
            if _looks_like_path(node, key):
                yield node

    for raw_path in _walk(parameters):
        normalized = _normalize_local_path(raw_path, workspace)
        if normalized:
            discovered.add(normalized)

    return discovered


def cleanup_paths(
    paths: Iterable[Path],
    *,
    dry_run: bool = False,
    keep: Optional[Iterable[Path]] = None,
    clear_contents_only: bool = True,
) -> CleanupResult:
    """Delete files/directories gathered from parameters.yaml.

    Parameters
    ----------
    paths:
        Candidate filesystem paths to delete.  Entries that do not exist are
        silently ignored.
    dry_run:
        When ``True`` nothing is deleted; messages describing the intended
        actions are printed instead.
    keep:
        Optional iterable of paths that should never be removed.  Children of the
        preserved directories are also protected.
    clear_contents_only:
        When ``True`` directories are emptied but not removed; set to ``False``
        to remove the directory itself.
    """

    keep_set = {_resolve_path(p) for p in (keep or [])}
    removed: List[Path] = []
    skipped: List[Path] = []

    for path in sorted({_resolve_path(p) for p in paths}):
        if _is_preserved(path, keep_set):
            skipped.append(path)
            continue

        if not path.exists():
            continue

        if dry_run:
            print(f"[dry-run] Would remove {path}")
            removed.append(path)
            continue

        if path.is_dir():
            if clear_contents_only:
                _clear_directory_contents(path)
            else:
                shutil.rmtree(path)
        else:
            path.unlink()

        removed.append(path)

    return CleanupResult(removed=removed, skipped=skipped)


def _looks_like_path(value: str, key: Optional[str]) -> bool:
    if any(value.startswith(prefix) for prefix in _REMOTE_PREFIXES) or "://" in value:
        return False
    if any(char in value for char in _GLOB_CHARS):
        return False

    lower_key = (key or "").lower()
    if any(lower_key.endswith(suffix) for suffix in _PATH_KEY_SUFFIXES):
        return True
    if "/" in value or "\\" in value:
        return True
    return False


def _normalize_local_path(raw: str, workspace: Path) -> Optional[Path]:
    if not raw or raw.startswith(_REMOTE_PREFIXES) or "://" in raw:
        return None
    try:
        path = Path(raw).expanduser()
    except OSError:
        return None

    if not path.is_absolute():
        path = (workspace / path).resolve()
    else:
        path = path.resolve()

    workspace = workspace.resolve()
    try:
        path.relative_to(workspace)
    except ValueError:
        # Avoid deleting files outside the workspace tree.
        return None

    return path


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_preserved(path: Path, keep: Set[Path]) -> bool:
    return any(path == preserved or preserved in path.parents for preserved in keep)


def _clear_directory_contents(directory: Path) -> None:
    for child in list(directory.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove artifacts produced by previous Colab runs before starting a new one.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        help="Optional parameters.yaml file to scan for relevant paths.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root that hosts the RealityStream notebook.",
    )
    parser.add_argument(
        "--extra-path",
        dest="extra_paths",
        action="append",
        default=[],
        help="Additional files or directories to delete (relative to workspace).",
    )
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="Paths that should never be deleted (relative to workspace).",
    )
    parser.add_argument(
        "--remove-directories",
        action="store_true",
        help="Remove directories entirely instead of clearing their contents.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the operations instead of deleting files.",
    )
    return parser.parse_args(argv)


def _resolve_cli_paths(entries: Sequence[str], workspace: Path) -> List[Path]:
    resolved: List[Path] = []
    for entry in entries:
        normalized = _normalize_local_path(entry, workspace)
        if normalized:
            resolved.append(normalized)
    return resolved


def main(argv: Optional[Sequence[str]] = None) -> CleanupResult:
    args = _parse_arguments(argv)
    workspace = args.workspace.resolve()

    parameters: Dict[str, Any] = {}
    if args.params:
        parameters = load_parameters_file(args.params)

    paths: Set[Path] = collect_candidate_paths(parameters, workspace) if parameters else set()
    paths.update(_resolve_cli_paths(args.extra_paths, workspace))

    keep_paths = _resolve_cli_paths(args.keep, workspace)
    result = cleanup_paths(
        paths,
        dry_run=args.dry_run,
        keep=keep_paths,
        clear_contents_only=not args.remove_directories,
    )

    if args.dry_run:
        print("Cleanup dry-run complete.")
    else:
        print(f"Removed {len(result.removed)} paths, skipped {len(result.skipped)} protected entries.")

    return result


if __name__ == "__main__":  # pragma: no cover
    main()