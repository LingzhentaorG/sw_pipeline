from __future__ import annotations

from pathlib import Path

import pytest

from sw_pipeline.app import run_event
from sw_pipeline.config import load_app_config

from tests.fixtures.helpers import (
    create_gold_tarball,
    create_isee_gnss_root,
    create_internal_workspace,
    create_local_omni_files,
    write_yaml,
)


def test_run_event_pipeline_with_local_sources(tmp_path: Path):
    pytest.importorskip("cartopy", reason="cartopy required for GOLD rendering")
    gnss_root = create_isee_gnss_root(tmp_path / "gnss")
    gold_tar = create_gold_tarball(tmp_path / "gold")
    bz_path, hourly_path, kp_path = create_local_omni_files(tmp_path / "omni", "fixture_evt")

    base_path = tmp_path / "base.yaml"
    event_path = tmp_path / "fixture_evt.yaml"
    write_yaml(
        base_path,
        {
            "paths": {"storage_root": str(tmp_path / "storage")},
            "bbox": {"lon_min": -150, "lon_max": 10, "lat_min": -80, "lat_max": 80},
            "runtime": {"gold_max_pair_minutes": 5},
            "plot_defaults": {
                "use_cartopy": True,
                "gnss_styles": {
                    "vtec": {"cmap": "viridis", "vmin": 0, "vmax": 80},
                    "roti": {"cmap": "viridis", "vmin": 0, "vmax": 1},
                },
            },
        },
    )
    write_yaml(
        event_path,
        {
            "event": {
                "id": "fixture_evt",
                "start": "2024-10-10T20:00:00Z",
                "end": "2024-10-10T20:59:59Z",
            },
            "sources": {
                "gnss_raw": {"enabled": False},
                "gnss_grid": {
                    "enabled": True,
                    "mode": "local",
                    "local_root": str(gnss_root),
                    "metrics": ["VTEC", "ROTI"],
                },
                "gold": {
                    "enabled": True,
                    "mode": "local",
                    "inputs": [str(gold_tar)],
                },
                "omni": {
                    "enabled": True,
                    "mode": "local",
                    "files": [str(bz_path), str(hourly_path), str(kp_path)],
                },
            },
            "products": {"gnss_grid": {"map_producers": ["isee"]}},
            "figures": {
                "gnss_maps": {"metrics": ["vtec", "roti"]},
                "overlays": [
                    {
                        "name": "roti_on_gold",
                        "threshold": 1.5,
                        "color": "red",
                        "producer": "isee",
                        "max_pair_delta_minutes": 15,
                    },
                ],
                "panels": [
                    {
                        "name": "fixture_roti_panel",
                        "layout": {"rows": 1, "cols": 1},
                        "shared_colorbar": "gnss_roti",
                        "slots": [
                            {
                                "kind": "gnss_roti",
                                "producer": "isee",
                                "timestamp": "2024-10-10T20:10:00Z",
                                "title": "2024-10-10 20:10 UTC",
                            }
                        ],
                    }
                ],
                "station_series": [],
            },
        },
    )

    spec = load_app_config("fixture_evt", base_path, event_path)
    run_event(spec, include_fetch=True)

    assert any((spec.storage.figures_gnss_dir / "isee" / "vtec").rglob("*.png"))
    assert any((spec.storage.figures_gnss_dir / "isee" / "roti").rglob("*.png"))
    assert any(spec.storage.figures_gold_dir.rglob("*.png"))
    assert any(spec.storage.figures_omni_dir.rglob("*.png"))
    assert any((spec.storage.figures_overlays_dir / "roti_on_gold" / "isee").rglob("*.png"))
    assert any(spec.storage.figures_panels_dir.rglob("*.png"))
    assert (spec.storage.manifests_dir / "panel_outputs.csv").exists()
    assert not (spec.storage.figures_overlays_dir / "vtec_on_gold").exists()


def test_run_event_pipeline_with_isee_maps_and_internal_station_series(tmp_path: Path):
    pytest.importorskip("cartopy", reason="cartopy required for GOLD rendering")
    internal_workspace = create_internal_workspace(tmp_path / "workspace", "hybrid_evt")
    gnss_root = create_isee_gnss_root(tmp_path / "gnss", timestamp="2024-10-10T23:10:00Z")
    gold_tar = create_gold_tarball(tmp_path / "gold", midpoint="2024-10-10T23:10:00Z")
    omni_root = tmp_path / "omni_local"
    create_local_omni_files(omni_root, "hybrid_evt")

    base_path = tmp_path / "base.yaml"
    event_path = tmp_path / "hybrid_evt.yaml"
    write_yaml(
        base_path,
        {
            "paths": {"storage_root": str(tmp_path / "storage")},
            "bbox": {"lon_min": -150, "lon_max": 10, "lat_min": -80, "lat_max": 80},
            "runtime": {"gold_max_pair_minutes": 5},
            "plot_defaults": {
                "use_cartopy": True,
                "gnss_styles": {
                    "vtec": {"cmap": "viridis", "vmin": 0, "vmax": 80},
                    "roti": {"cmap": "viridis", "vmin": 0, "vmax": 1},
                },
            },
        },
    )
    write_yaml(
        event_path,
        {
            "event": {
                "id": "hybrid_evt",
                "start": "2024-10-10T23:00:00Z",
                "end": "2024-10-11T00:59:59Z",
            },
            "sources": {
                "gnss_raw": {
                    "enabled": True,
                    "mode": "workspace_snapshot",
                    "workspace_root": str(internal_workspace),
                },
                "gnss_grid": {
                    "enabled": True,
                    "mode": "local",
                    "local_root": str(gnss_root),
                    "metrics": ["VTEC", "ROTI"],
                },
                "gold": {
                    "enabled": True,
                    "mode": "local",
                    "inputs": [str(gold_tar)],
                },
                "omni": {
                    "enabled": True,
                    "mode": "local",
                    "local_root": str(omni_root),
                },
            },
            "products": {"gnss_grid": {"map_producers": ["internal", "isee"]}},
            "figures": {
                "gnss_maps": {"metrics": ["vtec", "roti"]},
                "overlays": [
                    {
                        "name": "roti_on_gold",
                        "threshold": 1.5,
                        "color": "red",
                        "producer": "isee",
                        "max_pair_delta_minutes": 15,
                    },
                ],
                "panels": [],
                "station_series": [
                    {
                        "name": "boav_test",
                        "station_code": "BOAV",
                        "station_id": "41636M001",
                        "window": {
                            "start": "2024-10-10T23:00:00Z",
                            "end": "2024-10-11T04:00:00Z",
                        },
                        "satellites": ["G21", "G02"],
                    }
                ],
            },
        },
    )

    spec = load_app_config("hybrid_evt", base_path, event_path)
    run_event(spec, include_fetch=True)

    assert any((spec.storage.figures_gnss_dir / "internal" / "vtec").rglob("*.png"))
    assert any((spec.storage.figures_gnss_dir / "internal" / "roti").rglob("*.png"))
    assert any((spec.storage.figures_gnss_dir / "isee" / "vtec").rglob("*.png"))
    assert any((spec.storage.figures_gnss_dir / "isee" / "roti").rglob("*.png"))
    assert any((spec.storage.figures_overlays_dir / "roti_on_gold" / "isee").rglob("*.png"))
    assert any(spec.storage.figures_station_series_dir.rglob("*.png"))
    assert any(spec.storage.figures_omni_dir.rglob("*.png"))
