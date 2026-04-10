from __future__ import annotations

from pathlib import Path

import pytest

from sw_pipeline.cleanup import clean_run_outputs
from sw_pipeline.models import StorageLayout


def test_clean_run_outputs_rejects_cache_root(tmp_path: Path):
    cache_root = tmp_path / "storage" / "cache"
    run_root = cache_root / "danger"
    layout = _build_storage_layout(tmp_path, cache_root, run_root)

    with pytest.raises(ValueError):
        clean_run_outputs(layout)


def _build_storage_layout(project_root: Path, cache_root: Path, run_root: Path) -> StorageLayout:
    storage_root = project_root / "storage"
    archive_root = storage_root / "archive"
    return StorageLayout(
        project_root=project_root,
        storage_root=storage_root,
        cache_root=cache_root,
        archive_root=archive_root,
        pre_refactor_archive_root=archive_root / "pre_refactor",
        runs_root=storage_root / "runs",
        run_root=run_root,
        manifests_dir=run_root / "manifests",
        processed_root=run_root / "processed",
        processed_gnss_dir=run_root / "processed" / "gnss",
        processed_gold_dir=run_root / "processed" / "gold",
        processed_omni_dir=run_root / "processed" / "omni",
        gnss_workspace_dir=run_root / "processed" / "gnss" / "internal_workspace",
        grids_dir=run_root / "products" / "grids",
        figures_root=run_root / "figures",
        figures_gnss_dir=run_root / "figures" / "gnss",
        figures_gold_dir=run_root / "figures" / "gold",
        figures_omni_dir=run_root / "figures" / "omni",
        figures_overlays_dir=run_root / "figures" / "overlays",
        figures_panels_dir=run_root / "figures" / "panels",
        figures_station_series_dir=run_root / "figures" / "station_series",
    )
