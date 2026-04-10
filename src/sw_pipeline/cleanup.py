from __future__ import annotations

import shutil
from pathlib import Path

from .models import StorageLayout
from .storage import ensure_storage_write_allowed


def clean_workspace(project_root: Path) -> list[Path]:
    project_root = project_root.resolve()
    protected_root = (project_root / "storage" / "cache").resolve()
    removed: list[Path] = []

    candidates: list[Path] = []
    candidates.append(project_root / ".pytest_cache")
    candidates.extend(project_root.rglob("__pycache__"))
    candidates.extend(project_root.rglob("*.part"))

    for path in candidates:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved == protected_root or resolved.is_relative_to(protected_root):
            continue
        removed.append(_remove_path(resolved))

    for path in project_root.rglob("logs"):
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved == protected_root or resolved.is_relative_to(protected_root):
            continue
        if any(resolved.iterdir()):
            continue
        removed.append(_remove_path(resolved))

    return removed


def clean_run_outputs(storage: StorageLayout) -> Path:
    run_root = ensure_storage_write_allowed(storage.run_root, storage)
    allowed_root = storage.runs_root.resolve()
    if not (run_root == allowed_root or run_root.is_relative_to(allowed_root)):
        raise ValueError(f"Refusing to clean path outside storage/runs: {run_root}")
    if run_root.exists():
        shutil.rmtree(run_root, ignore_errors=True)
    return run_root


def archive_pre_refactor_path(source: Path, storage: StorageLayout) -> Path | None:
    resolved = ensure_storage_write_allowed(source, storage)
    if not resolved.exists():
        return None
    storage_root = storage.storage_root.resolve()
    archive_root = storage.archive_root.resolve()
    if not resolved.is_relative_to(storage_root) or resolved.is_relative_to(archive_root):
        raise ValueError(f"Refusing to archive path outside mutable storage roots: {resolved}")
    destination = storage.pre_refactor_archive_root / resolved.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Archive destination already exists: {destination}")
    return Path(shutil.move(str(resolved), str(destination)))


def _remove_path(path: Path) -> Path:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
    return path
