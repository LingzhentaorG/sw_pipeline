from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from sw_pipeline.config import load_app_config
from sw_pipeline.discovery import discover_omni_series
from sw_pipeline.providers import gnss_raw as gnss_raw_module
from sw_pipeline.providers.gnss_grid import fetch_gnss_grid_assets, process_gnss_grid_assets
from sw_pipeline.providers.gold import fetch_gold_assets, process_gold_assets
from sw_pipeline.providers.omni import process_omni_assets

from tests.fixtures.helpers import create_gold_tarball, create_isee_gnss_root, create_local_omni_files, write_yaml


def test_process_gnss_grid_assets_uses_cache_when_asset_manifest_missing(tmp_path):
    gnss_root = create_isee_gnss_root(tmp_path / "gnss")
    spec = _load_event_spec(
        tmp_path,
        "grid_cache_evt",
        {
            "event": {
                "id": "grid_cache_evt",
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
                "gold": {"enabled": False},
                "omni": {"enabled": False},
            },
            "products": {"gnss_grid": {"map_producers": ["isee"]}},
            "figures": {"overlays": [], "station_series": []},
        },
    )

    fetch_gnss_grid_assets(spec)
    (spec.storage.manifests_dir / "gnss_grid_assets.csv").unlink()

    products = process_gnss_grid_assets(spec)

    assert products
    assert all(product.producer == "isee" for product in products)
    assert any((spec.storage.grids_dir / "isee").rglob("*.nc"))


def test_process_gold_assets_uses_cache_when_scene_manifest_missing(tmp_path):
    gold_tar = create_gold_tarball(tmp_path / "gold")
    spec = _load_event_spec(
        tmp_path,
        "gold_cache_evt",
        {
            "event": {
                "id": "gold_cache_evt",
                "start": "2024-10-10T00:00:00Z",
                "end": "2024-10-10T23:59:59Z",
            },
            "sources": {
                "gnss_raw": {"enabled": False},
                "gnss_grid": {"enabled": False},
                "gold": {"enabled": True, "mode": "local", "inputs": [str(gold_tar)]},
                "omni": {"enabled": False},
            },
            "products": {"gnss_grid": {"map_producers": ["isee"]}},
            "figures": {"overlays": [], "station_series": []},
        },
    )

    fetch_gold_assets(spec)
    first_pass = process_gold_assets(spec)
    assert first_pass
    (spec.storage.manifests_dir / "gold_scenes.csv").unlink()

    rebuilt = process_gold_assets(spec)

    assert rebuilt
    assert rebuilt[0].cha_time is not None
    assert rebuilt[0].chb_time is not None


def test_process_omni_assets_uses_canonical_cache_root(tmp_path):
    spec = _load_event_spec(
        tmp_path,
        "omni_cache_evt",
        {
            "event": {
                "id": "omni_cache_evt",
                "start": "2024-10-10T00:00:00Z",
                "end": "2024-10-10T23:59:59Z",
            },
            "sources": {
                "gnss_raw": {"enabled": False},
                "gnss_grid": {"enabled": False},
                "gold": {"enabled": False},
                "omni": {"enabled": True, "mode": "local"},
            },
            "products": {"gnss_grid": {"map_producers": ["isee"]}},
            "figures": {"overlays": [], "station_series": []},
        },
    )
    canonical_root = spec.storage.cache_root / "omni" / spec.event_id
    create_local_omni_files(canonical_root, spec.event_id)

    discovered = discover_omni_series(spec)
    processed = process_omni_assets(spec)

    assert discovered.bz_csv_path.parent == canonical_root
    assert discovered.hourly_csv_path.parent == canonical_root
    assert processed.bz_csv_path.parent == spec.storage.processed_omni_dir
    assert processed.hourly_csv_path.parent == spec.storage.processed_omni_dir


def test_process_gnss_raw_assets_rebuilds_workspace_from_cache_without_fetch(tmp_path, monkeypatch):
    spec = _load_event_spec(
        tmp_path,
        "raw_cache_evt",
        {
            "event": {
                "id": "raw_cache_evt",
                "start": "2024-10-10T00:00:00Z",
                "end": "2024-10-10T23:59:59Z",
            },
            "sources": {
                "gnss_raw": {"enabled": True, "mode": "internal_pipeline"},
                "gnss_grid": {"enabled": False},
                "gold": {"enabled": False},
                "omni": {"enabled": False},
            },
            "products": {"gnss_grid": {"map_producers": ["internal"]}},
            "figures": {
                "overlays": [],
                "station_series": [
                    {
                        "name": "boav_cache_test",
                        "station_code": "BOAV",
                        "station_id": "41636M001",
                        "window": {
                            "start": "2024-10-10T00:00:00Z",
                            "end": "2024-10-10T01:00:00Z",
                        },
                        "satellites": ["G21", "G02"],
                    }
                ],
            },
        },
    )

    obs_path = spec.storage.cache_root / "gnss_raw" / "observations" / spec.event_id / "2024-10-10" / "noaa" / "BOAV0010.24d.gz"
    nav_path = spec.storage.cache_root / "gnss_raw" / "navigation" / spec.event_id / "2024-10-10" / "brdc2840.24n.gz"
    sp3_path = spec.storage.cache_root / "gnss_aux" / "sp3" / "2024" / "igs2024284.sp3"
    dcb_path = spec.storage.cache_root / "gnss_aux" / "dcb" / "2024" / "cod2024284.dcb"
    antex_path = spec.storage.cache_root / "gnss_aux" / "antex" / "igs20.atx"
    station_log_path = spec.storage.cache_root / "gnss_aux" / "station_logs" / "boav" / "boav.log"
    for path in (obs_path, nav_path, sp3_path, dcb_path, antex_path, station_log_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cached", encoding="utf-8")

    class FakeConfigModule:
        @staticmethod
        def load_pipeline_config(path):
            return SimpleNamespace(path=path)

    class FakePreprocessModule:
        @staticmethod
        def preprocess_records(pipeline_config):
            return None

    class FakeProcessingModule:
        @staticmethod
        def execute_processing_stage(pipeline_config):
            output_dir = spec.storage.gnss_workspace_dir / "products" / "netcdf"
            output_dir.mkdir(parents=True, exist_ok=True)
            xr.Dataset(
                {
                    "vtec": (("time", "lat", "lon"), np.full((1, 1, 1), 12.0, dtype=float)),
                    "roti": (("time", "lat", "lon"), np.full((1, 1, 1), 0.8, dtype=float)),
                },
                coords={
                    "time": pd.to_datetime(["2024-10-10T00:00:00Z"], utc=True).tz_convert(None),
                    "lat": np.array([0.0], dtype=float),
                    "lon": np.array([-70.0], dtype=float),
                },
            ).to_netcdf(output_dir / f"{spec.event_id}_20241010_0000.nc")

    monkeypatch.setattr(
        gnss_raw_module,
        "_load_internal_pipeline_modules",
        lambda include_processing=False: (
            (FakeConfigModule, object(), FakePreprocessModule, FakeProcessingModule)
            if include_processing
            else (FakeConfigModule, object())
        ),
    )

    products = gnss_raw_module.process_gnss_raw_assets(spec)

    observation_manifest = pd.read_csv(spec.storage.gnss_workspace_dir / "manifests" / "observation_manifest.csv")
    aux_manifest = pd.read_csv(spec.storage.gnss_workspace_dir / "manifests" / "aux_manifest.csv")
    assert products
    assert observation_manifest.iloc[0]["station_code4"] == "BOAV"
    assert observation_manifest.iloc[0]["station_id"] == "41636M001"
    assert aux_manifest["product_type"].astype(str).str.contains("sp3|dcb|antex|station_logs").any()
    assert any((spec.storage.grids_dir / "internal").rglob("*.nc"))


def _load_event_spec(tmp_path, event_id: str, event_payload: dict):
    base_path = tmp_path / f"{event_id}_base.yaml"
    event_path = tmp_path / f"{event_id}.yaml"
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
    write_yaml(event_path, event_payload)
    return load_app_config(event_id, base_path, event_path)
