from __future__ import annotations

from pathlib import Path

import pandas as pd

from sw_pipeline.config import load_app_config
from sw_pipeline.pipelines.figures import plot_event_target
from sw_pipeline.renderers.station_series import _panel_title

from tests.fixtures.helpers import create_internal_workspace, write_yaml


def test_station_series_panel_title_includes_letter_and_context():
    assert _panel_title(0, 0, "BOAV", "G21", "VTEC") == "(a) BOAV G21 VTEC"
    assert _panel_title(0, 1, "BOAV", "G02", "VTEC") == "(b) BOAV G02 VTEC"
    assert _panel_title(1, 0, "BOAV", "G21", "ROTI") == "(c) BOAV G21 ROTI"
    assert _panel_title(1, 1, "BOAV", "G02", "ROTI") == "(d) BOAV G02 ROTI"


def test_station_series_render_from_internal_workspace(tmp_path: Path):
    workspace_root = create_internal_workspace(tmp_path / "workspace", "station_evt")
    base_path = tmp_path / "base.yaml"
    event_path = tmp_path / "station_evt.yaml"

    write_yaml(
        base_path,
        {
            "paths": {"storage_root": str(tmp_path / "storage")},
            "bbox": {"lon_min": -150, "lon_max": 10, "lat_min": -80, "lat_max": 80},
            "runtime": {"gold_max_pair_minutes": 5},
            "plot_defaults": {
                "use_cartopy": False,
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
                "id": "station_evt",
                "start": "2024-10-10T23:00:00Z",
                "end": "2024-10-11T04:00:00Z",
            },
            "sources": {
                "gnss_raw": {
                    "enabled": True,
                    "mode": "workspace_snapshot",
                    "workspace_root": str(workspace_root),
                },
                "gnss_grid": {"enabled": False},
                "gold": {"enabled": False},
                "omni": {"enabled": False},
            },
            "products": {"gnss_grid": {"map_producers": ["internal"]}},
            "figures": {
                "overlays": [],
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

    spec = load_app_config("station_evt", base_path, event_path)
    plot_event_target(spec, "station-series")
    assert (spec.storage.figures_station_series_dir / "boav_test.png").exists()


def test_station_series_falls_back_to_station_code4(tmp_path: Path):
    workspace_root = create_internal_workspace(tmp_path / "workspace", "station_code_evt")
    base_path = tmp_path / "base.yaml"
    event_path = tmp_path / "station_code_evt.yaml"

    vtec_path = workspace_root / "intermediate" / "vtec" / "station_code_evt.parquet"
    roti_path = workspace_root / "intermediate" / "roti" / "station_code_evt.parquet"
    pd.DataFrame(
        [
            {"time": "2024-10-10T23:00:00Z", "station_code4": "BOAV", "sv": "G21", "vtec": 12.0},
            {"time": "2024-10-10T23:30:00Z", "station_code4": "BOAV", "sv": "G02", "vtec": 13.0},
        ]
    ).to_parquet(vtec_path, index=False)
    pd.DataFrame(
        [
            {"time": "2024-10-10T23:00:00Z", "station_code4": "BOAV", "sv": "G21", "roti": 0.4},
            {"time": "2024-10-10T23:30:00Z", "station_code4": "BOAV", "sv": "G02", "roti": 0.6},
        ]
    ).to_parquet(roti_path, index=False)

    write_yaml(
        base_path,
        {
            "paths": {"storage_root": str(tmp_path / "storage")},
            "bbox": {"lon_min": -150, "lon_max": 10, "lat_min": -80, "lat_max": 80},
            "runtime": {"gold_max_pair_minutes": 5},
            "plot_defaults": {
                "use_cartopy": False,
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
                "id": "station_code_evt",
                "start": "2024-10-10T23:00:00Z",
                "end": "2024-10-11T04:00:00Z",
            },
            "sources": {
                "gnss_raw": {
                    "enabled": True,
                    "mode": "workspace_snapshot",
                    "workspace_root": str(workspace_root),
                },
                "gnss_grid": {"enabled": False},
                "gold": {"enabled": False},
                "omni": {"enabled": False},
            },
            "products": {"gnss_grid": {"map_producers": ["internal"]}},
            "figures": {
                "overlays": [],
                "station_series": [
                    {
                        "name": "boav_code_only",
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

    spec = load_app_config("station_code_evt", base_path, event_path)
    plot_event_target(spec, "station-series")
    assert (spec.storage.figures_station_series_dir / "boav_code_only.png").exists()
