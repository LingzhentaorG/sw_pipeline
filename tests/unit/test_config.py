from __future__ import annotations

from pathlib import Path

from sw_pipeline.config import load_app_config
from sw_pipeline.storage import ensure_storage_layout

from tests.fixtures.helpers import write_yaml


def test_load_app_config_from_repo_files():
    spec = load_app_config("storm_20241010_11")
    assert spec.event_id == "storm_20241010_11"
    assert spec.gnss_map_producers() == ("isee",)
    assert spec.sources["omni"]["mode"] == "local"
    assert str(spec.sources["omni"]["local_root"]).endswith("storage/cache/omni/storm_20241010_11")
    assert "providers" in spec.sources["gnss_raw"]
    assert "auxiliary" in spec.sources["gnss_raw"]
    assert spec.sources["gnss_raw"].get("pipeline_overrides", {}).get("processing", {}).get("max_station_days_per_event", 0) == 0
    assert spec.storage.run_root.name == "storm_20241010_11"
    assert spec.map_extent() == (-150.0, 10.0, -80.0, 80.0)
    assert spec.plot_defaults.gnss_styles["roti"].cmap == "viridis"
    overlay = spec.overlay_specs()[0]
    assert overlay.threshold == 1.0
    assert overlay.color == "red"
    assert overlay.producer == "isee"
    assert len(overlay.pairs) == 2
    assert str(overlay.pairs[0].gnss_time) == "2024-10-11 00:10:00+00:00"
    assert len(spec.omni_highlight_windows()) == 0
    assert len(spec.panel_specs()) == 4

    second = load_app_config("storm_20241231_20250101")
    assert second.gnss_map_producers() == ("isee",)
    assert second.sources["omni"]["mode"] == "local"
    assert str(second.sources["omni"]["local_root"]).endswith("storage/cache/omni/storm_20241231_20250101")
    assert second.sources["gnss_raw"].get("pipeline_overrides", {}).get("processing", {}).get("max_station_days_per_event", 0) == 0
    second_overlay = second.overlay_specs()[0]
    assert len(second_overlay.pairs) == 2
    assert str(second_overlay.pairs[1].gold_chb_time) == "2024-12-31 23:55:00+00:00"
    assert len(second.omni_highlight_windows()) == 0
    assert len(second.panel_specs()) == 4


def test_load_app_config_with_custom_event_file(tmp_path: Path):
    base_path = tmp_path / "base.yaml"
    event_path = tmp_path / "event.yaml"
    storage_root = tmp_path / "storage"
    write_yaml(
        base_path,
        {
            "paths": {"storage_root": str(storage_root)},
            "bbox": {"lon_min": -150, "lon_max": 10, "lat_min": -80, "lat_max": 80},
            "runtime": {"gold_max_pair_minutes": 5},
            "plot_defaults": {
                "gnss_styles": {
                    "vtec": {"cmap": "viridis", "vmin": 0, "vmax": 80},
                    "roti": {"cmap": "viridis", "vmin": 0, "vmax": 1},
                }
            },
        },
    )
    write_yaml(
        event_path,
        {
            "event": {
                "id": "custom_evt",
                "start": "2024-10-10T00:00:00Z",
                "end": "2024-10-10T01:00:00Z",
            },
            "sources": {
                "gnss_raw": {"enabled": False},
                "gnss_grid": {"enabled": True},
                "gold": {"enabled": False},
                "omni": {"enabled": False},
            },
            "products": {"gnss_grid": {"map_producers": ["internal", "isee"]}},
            "figures": {
                "overlays": [
                    {
                        "name": "roti_on_gold",
                        "threshold": 1.5,
                        "color": "red",
                        "producer": "isee",
                        "pairs": [
                            {
                                "gold_cha_time": "2024-10-10T00:10:00Z",
                                "gold_chb_time": "2024-10-10T00:12:00Z",
                                "gnss_time": "2024-10-10T00:15:00Z",
                            }
                        ],
                    }
                ],
                "station_series": [],
            },
        },
    )
    spec = load_app_config("custom_evt", base_path, event_path)
    assert spec.event_id == "custom_evt"
    assert spec.storage.storage_root == storage_root
    assert spec.sources["gnss_raw"]["providers"]["noaa"]["transport"] == "https"
    assert "broadcast" in spec.sources["gnss_raw"]["auxiliary"]
    assert not spec.storage.run_root.exists()
    custom_overlay = spec.overlay_specs()[0]
    assert custom_overlay.threshold == 1.5
    assert custom_overlay.color == "red"
    assert custom_overlay.producer == "isee"
    assert len(custom_overlay.pairs) == 1
    assert str(custom_overlay.pairs[0].gnss_time) == "2024-10-10 00:15:00+00:00"
    ensure_storage_layout(spec.storage)
    assert spec.storage.run_root.exists()


def test_load_app_config_rejects_missing_runtime(tmp_path: Path):
    base_path = tmp_path / "base.yaml"
    event_path = tmp_path / "event.yaml"
    write_yaml(base_path, {"paths": {"storage_root": str(tmp_path / "storage")}})
    write_yaml(
        event_path,
        {
            "event": {"id": "bad_evt", "start": "2024-10-10T00:00:00Z", "end": "2024-10-10T01:00:00Z"},
            "bbox": {"lon_min": -150, "lon_max": 10, "lat_min": -80, "lat_max": 80},
            "sources": {
                "gnss_raw": {"enabled": False},
                "gnss_grid": {"enabled": True},
                "gold": {"enabled": False},
                "omni": {"enabled": False},
            },
            "products": {"gnss_grid": {"map_producers": ["isee"]}},
            "figures": {"overlays": [], "station_series": []},
        },
    )
    try:
        load_app_config("bad_evt", base_path, event_path)
    except ValueError as exc:
        assert "runtime" in str(exc)
    else:  # pragma: no cover - safety
        raise AssertionError("Expected ValueError for missing runtime section")


def test_load_app_config_rejects_non_fixed_bbox(tmp_path: Path):
    base_path = tmp_path / "base.yaml"
    event_path = tmp_path / "event.yaml"
    write_yaml(
        base_path,
        {
            "paths": {"storage_root": str(tmp_path / "storage")},
            "runtime": {"gold_max_pair_minutes": 5},
        },
    )
    write_yaml(
        event_path,
        {
            "event": {"id": "bad_bbox", "start": "2024-10-10T00:00:00Z", "end": "2024-10-10T01:00:00Z"},
            "bbox": {"lon_min": -80, "lon_max": -40, "lat_min": -20, "lat_max": 20},
            "sources": {
                "gnss_raw": {"enabled": False},
                "gnss_grid": {"enabled": True},
                "gold": {"enabled": False},
                "omni": {"enabled": False},
            },
            "products": {"gnss_grid": {"map_producers": ["isee"]}},
            "figures": {"overlays": [], "station_series": []},
            "runtime": {},
        },
    )
    try:
        load_app_config("bad_bbox", base_path, event_path)
    except ValueError as exc:
        assert "fixed map extent" in str(exc)
    else:  # pragma: no cover - safety
        raise AssertionError("Expected ValueError for non-fixed bbox")


def test_load_app_config_rejects_unsupported_overlay(tmp_path: Path):
    base_path = tmp_path / "base.yaml"
    event_path = tmp_path / "event.yaml"
    write_yaml(
        base_path,
        {
            "paths": {"storage_root": str(tmp_path / "storage")},
            "bbox": {"lon_min": -150, "lon_max": 10, "lat_min": -80, "lat_max": 80},
            "runtime": {"gold_max_pair_minutes": 5},
        },
    )
    write_yaml(
        event_path,
        {
            "event": {"id": "bad_overlay", "start": "2024-10-10T00:00:00Z", "end": "2024-10-10T01:00:00Z"},
            "sources": {
                "gnss_raw": {"enabled": False},
                "gnss_grid": {"enabled": True},
                "gold": {"enabled": False},
                "omni": {"enabled": False},
            },
            "products": {"gnss_grid": {"map_producers": ["isee"]}},
            "figures": {"overlays": [{"name": "vtec_on_gold"}], "station_series": []},
            "runtime": {},
        },
    )
    try:
        load_app_config("bad_overlay", base_path, event_path)
    except ValueError as exc:
        assert "roti_on_gold" in str(exc)
    else:  # pragma: no cover - safety
        raise AssertionError("Expected ValueError for unsupported overlay")


def test_load_app_config_rejects_overlay_plot_extent_override(tmp_path: Path):
    base_path = tmp_path / "base.yaml"
    event_path = tmp_path / "event.yaml"
    write_yaml(
        base_path,
        {
            "paths": {"storage_root": str(tmp_path / "storage")},
            "bbox": {"lon_min": -150, "lon_max": 10, "lat_min": -80, "lat_max": 80},
            "runtime": {"gold_max_pair_minutes": 5},
        },
    )
    write_yaml(
        event_path,
        {
            "event": {"id": "bad_extent", "start": "2024-10-10T00:00:00Z", "end": "2024-10-10T01:00:00Z"},
            "sources": {
                "gnss_raw": {"enabled": False},
                "gnss_grid": {"enabled": True},
                "gold": {"enabled": False},
                "omni": {"enabled": False},
            },
            "products": {"gnss_grid": {"map_producers": ["isee"]}},
            "figures": {
                "overlays": [{"name": "roti_on_gold", "plot_extent": [-105, 15, -60, 60]}],
                "station_series": [],
            },
            "runtime": {},
        },
    )
    try:
        load_app_config("bad_extent", base_path, event_path)
    except ValueError as exc:
        assert "plot_extent" in str(exc)
    else:  # pragma: no cover - safety
        raise AssertionError("Expected ValueError for overlay plot_extent override")
