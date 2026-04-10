from __future__ import annotations

import shutil
from pathlib import Path

from .models import StorageLayout


def build_storage_layout(project_root: Path, storage_root: Path, event_id: str) -> StorageLayout:
    cache_root = storage_root / "cache"
    archive_root = storage_root / "archive"
    pre_refactor_archive_root = archive_root / "pre_refactor"
    runs_root = storage_root / "runs"
    run_root = runs_root / event_id
    manifests_dir = run_root / "manifests"
    processed_root = run_root / "processed"
    processed_gnss_dir = processed_root / "gnss"
    processed_gold_dir = processed_root / "gold"
    processed_omni_dir = processed_root / "omni"
    gnss_workspace_dir = processed_gnss_dir / "internal_workspace"
    grids_dir = run_root / "products" / "grids"
    figures_root = run_root / "figures"
    figures_gnss_dir = figures_root / "gnss"
    figures_gold_dir = figures_root / "gold"
    figures_omni_dir = figures_root / "omni"
    figures_overlays_dir = figures_root / "overlays"
    figures_panels_dir = figures_root / "panels"
    figures_station_series_dir = figures_root / "station_series"

    return StorageLayout(
        project_root=project_root,
        storage_root=storage_root,
        cache_root=cache_root,
        archive_root=archive_root,
        pre_refactor_archive_root=pre_refactor_archive_root,
        runs_root=runs_root,
        run_root=run_root,
        manifests_dir=manifests_dir,
        processed_root=processed_root,
        processed_gnss_dir=processed_gnss_dir,
        processed_gold_dir=processed_gold_dir,
        processed_omni_dir=processed_omni_dir,
        gnss_workspace_dir=gnss_workspace_dir,
        grids_dir=grids_dir,
        figures_root=figures_root,
        figures_gnss_dir=figures_gnss_dir,
        figures_gold_dir=figures_gold_dir,
        figures_omni_dir=figures_omni_dir,
        figures_overlays_dir=figures_overlays_dir,
        figures_panels_dir=figures_panels_dir,
        figures_station_series_dir=figures_station_series_dir,
    )


def ensure_storage_layout(storage: StorageLayout) -> None:
    for path in (
        storage.cache_root,
        storage.archive_root,
        storage.pre_refactor_archive_root,
        storage.runs_root,
        storage.manifests_dir,
        storage.processed_gnss_dir,
        storage.processed_gold_dir,
        storage.processed_omni_dir,
        storage.gnss_workspace_dir,
        storage.grids_dir,
        storage.figures_gnss_dir,
        storage.figures_gold_dir,
        storage.figures_omni_dir,
        storage.figures_overlays_dir,
        storage.figures_panels_dir,
        storage.figures_station_series_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_storage_write_allowed(path: Path, storage: StorageLayout) -> Path:
    resolved = path.resolve()
    protected_root = storage.cache_root.resolve()
    if resolved == protected_root or resolved.is_relative_to(protected_root):
        raise ValueError(f"Refusing to mutate protected cache path: {resolved}")
    return resolved


def reset_generated_directory(path: Path, storage: StorageLayout) -> None:
    resolved = ensure_storage_write_allowed(path, storage)
    run_root = storage.run_root.resolve()
    if not (resolved == run_root or resolved.is_relative_to(run_root)):
        raise ValueError(f"Refusing to reset path outside run root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved, ignore_errors=True)
    resolved.mkdir(parents=True, exist_ok=True)


def remove_generated_tree(path: Path, storage: StorageLayout) -> None:
    resolved = ensure_storage_write_allowed(path, storage)
    run_root = storage.run_root.resolve()
    if not (resolved == run_root or resolved.is_relative_to(run_root)):
        raise ValueError(f"Refusing to remove path outside run root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved, ignore_errors=True)
