from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sw_pipeline.config import load_app_config
from sw_pipeline.models import GnssDownloadAsset
from sw_pipeline.providers import gnss_grid as gnss_grid_module
from sw_pipeline.providers import gnss_raw as gnss_raw_module
from sw_pipeline.providers.gnss_grid import fetch_gnss_grid_assets
from sw_pipeline.storage import ensure_storage_layout

from tests.fixtures.helpers import write_yaml


def test_download_observation_record_uses_noaa_fallback(tmp_path: Path, monkeypatch):
    event_spec = _build_event_spec(tmp_path, event_id="fallback_evt")
    pipeline_config = SimpleNamespace(download={"max_retries": 2, "temp_suffix": ".part"})
    settings = SimpleNamespace(
        timeout_sec=30,
        params={
            "transport": "https",
            "base_obs_url": "https://primary.example/rinex",
            "fallback_base_obs_url": "https://fallback.example/rinex",
        },
    )
    record = SimpleNamespace(
        source="noaa",
        station_id="ABCD00USA",
        station_code4="ABCD",
        observation_date="2024-10-10",
        obs_url="https://primary.example/rinex/2024/284/abcd/abcd2840.24d.gz",
        sampling_sec=30,
    )

    calls: list[str] = []

    def fake_download(url, target, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            return SimpleNamespace(path=None, status="error", error="primary failed", attempts=1, protocol="https")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ok", encoding="utf-8")
        return SimpleNamespace(path=target, status="ok", error=None, attempts=1, protocol="https")

    monkeypatch.setattr(gnss_raw_module, "download_to_path", fake_download)

    asset = gnss_raw_module._download_observation_record(
        event_spec,
        pipeline_config,
        record,
        settings,
        event_spec.storage.cache_root / "gnss_raw" / "observations",
        http_session=object(),
        cddis_session=object(),
    )

    assert calls == [
        "https://primary.example/rinex/2024/284/abcd/abcd2840.24d.gz",
        "https://fallback.example/rinex/2024/284/abcd/abcd2840.24d.gz",
    ]
    assert asset.status == "ok"
    assert asset.provider == "noaa"
    assert asset.local_path is not None
    assert asset.local_path.exists()


def test_fetch_internal_pipeline_assets_uses_cddis_as_fallback(tmp_path: Path, monkeypatch):
    from sw_pipeline.internal.gnss_core.models import DownloadRecord

    event_spec = _build_event_spec(tmp_path, event_id="cddis_evt")

    pipeline_config = SimpleNamespace(
        events=[SimpleNamespace(event_id="cddis_evt")],
        bbox=event_spec.bbox,
        download={"max_retries": 1, "aux_retries": 1, "temp_suffix": ".part"},
        observation_sources={
            "noaa": SimpleNamespace(
                enabled=True,
                priority=10,
                timeout_sec=30,
                params={"base_obs_url": "https://primary.example/rinex"},
            ),
            "cddis": SimpleNamespace(
                enabled=True,
                priority=100,
                timeout_sec=30,
                params={"obs_url_template": "https://cddis.example/{station}{doy}0.{yy}d.Z"},
            ),
        },
        auxiliary_sources={
            "broadcast": SimpleNamespace(
                enabled=True,
                priority=10,
                timeout_sec=30,
                params={
                    "providers": [
                        {
                            "name": "cddis_nav",
                            "priority": 10,
                            "auth": "cddis",
                            "transport": "https",
                            "url_template": "https://cddis.example/nav/{yyyy}{ddd}.rnx.gz",
                        }
                    ]
                },
            )
        },
    )

    primary_record = DownloadRecord(
        event_id="cddis_evt",
        source="noaa",
        source_priority=10,
        observation_date="2024-10-10",
        station_id="ABCD00USA",
        station_code4="ABCD",
        sampling_sec=30,
        obs_url="https://primary.example/rinex/2024/284/abcd/abcd2840.24d.gz",
        nav_url="",
        lat=10.0,
        lon=-70.0,
        height_m=100.0,
    )

    class DummyAdapter:
        def __init__(self, records):
            self.settings = SimpleNamespace(name="dummy")
            self._records = records

        def discover(self, event):
            return list(self._records)

    def fake_make_adapters(sources, bbox, base_nav_url):
        if list(sources) == ["cddis"]:
            return [DummyAdapter([])]
        return [DummyAdapter([primary_record])]

    provider_calls: list[str] = []

    def fake_download_observation_record(event_spec, pipeline_config, record, settings, observations_root, **kwargs):
        provider_calls.append(record.source)
        if record.source == "noaa":
            return GnssDownloadAsset(
                event_id=event_spec.event_id,
                source_kind="gnss_observation",
                provider="noaa",
                protocol="https",
                station_id=record.station_id,
                station_code4=record.station_code4,
                observation_date=record.observation_date,
                url=record.obs_url,
                local_path=None,
                status="error",
                attempts=1,
                error="primary failed",
            )
        target = event_spec.storage.cache_root / "gnss_raw" / "observations" / event_spec.event_id / record.observation_date / "cddis" / "obs.Z"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ok", encoding="utf-8")
        return GnssDownloadAsset(
            event_id=event_spec.event_id,
            source_kind="gnss_observation",
            provider="cddis",
            protocol="https",
            station_id=record.station_id,
            station_code4=record.station_code4,
            observation_date=record.observation_date,
            url=record.obs_url,
            local_path=target,
            status="ok",
            attempts=1,
            auth_ref="cddis",
        )

    def fake_download_daily_aux_product(**kwargs):
        current_day = kwargs["current_day"]
        product_type = kwargs["product_type"]
        target = kwargs["cache_root"] / kwargs["event_spec"].event_id / current_day.isoformat() / f"{product_type}.dat"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ok", encoding="utf-8")
        source_kind = "gnss_navigation" if product_type == "broadcast" else "gnss_aux"
        return (
            GnssDownloadAsset(
                event_id=kwargs["event_spec"].event_id,
                source_kind=source_kind,
                provider="cddis_nav",
                protocol="https",
                station_id="",
                station_code4="",
                observation_date=current_day.isoformat(),
                url=f"https://cddis.example/{product_type}",
                local_path=target,
                status="ok",
                attempts=1,
                auth_ref="cddis",
                metadata={"product_type": product_type},
            ),
            [],
        )

    monkeypatch.setattr("sw_pipeline.internal.gnss_core.sources.make_adapters", fake_make_adapters)
    monkeypatch.setattr(gnss_raw_module, "_download_observation_record", fake_download_observation_record)
    monkeypatch.setattr(gnss_raw_module, "_download_daily_aux_product", fake_download_daily_aux_product)

    assets = gnss_raw_module._fetch_internal_pipeline_assets(event_spec, pipeline_config)

    assert provider_calls == ["noaa", "cddis"]
    assert assets
    assert assets[0].provider == "cddis"
    failures_manifest = event_spec.storage.manifests_dir / "gnss_download_failures.csv"
    assert failures_manifest.exists()


def test_fetch_gnss_grid_assets_remote_uses_standard_cache_layout(tmp_path: Path, monkeypatch):
    event_spec = _build_event_spec(
        tmp_path,
        event_id="isee_evt",
        event_overrides={
            "sources": {
                "gnss_raw": {"enabled": False},
                "gnss_grid": {"enabled": True, "mode": "remote", "metrics": ["VTEC"]},
                "gold": {"enabled": False},
                "omni": {"enabled": False},
            },
            "products": {"gnss_grid": {"map_producers": ["isee"]}},
        },
    )

    monkeypatch.setattr(gnss_grid_module, "fetch_text", lambda *args, **kwargs: '<a href="a.nc">a.nc</a><a href="b.nc">b.nc</a>')

    def fake_download(url, target, **kwargs):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("nc", encoding="utf-8")
        return SimpleNamespace(path=target, status="ok", error=None, attempts=1, protocol="https")

    monkeypatch.setattr(gnss_grid_module, "download_to_path", fake_download)

    assets = fetch_gnss_grid_assets(event_spec)

    assert len(assets) == 2
    assert all("VTEC" in str(asset.local_path) for asset in assets)
    assert all(asset.local_path.exists() for asset in assets)


def _build_event_spec(tmp_path: Path, *, event_id: str, event_overrides: dict | None = None):
    base_path = tmp_path / "base.yaml"
    event_path = tmp_path / f"{event_id}.yaml"
    write_yaml(
        base_path,
        {
            "paths": {"storage_root": str(tmp_path / "storage")},
            "bbox": {"lon_min": -150, "lon_max": 10, "lat_min": -80, "lat_max": 80},
            "runtime": {"gold_max_pair_minutes": 5},
            "plot_defaults": {
                "gnss_styles": {
                    "vtec": {"cmap": "viridis", "vmin": 0, "vmax": 80},
                    "roti": {"cmap": "magma", "vmin": 0, "vmax": 1},
                }
            },
        },
    )
    event_payload = {
        "event": {
            "id": event_id,
            "start": "2024-10-10T00:00:00Z",
            "end": "2024-10-10T01:00:00Z",
        },
        "sources": {
            "gnss_raw": {"enabled": True, "mode": "internal_pipeline"},
            "gnss_grid": {"enabled": True, "mode": "local"},
            "gold": {"enabled": False},
            "omni": {"enabled": False},
        },
        "products": {"gnss_grid": {"map_producers": ["internal", "isee"]}},
        "figures": {"overlays": [], "station_series": []},
        "runtime": {},
    }
    if event_overrides:
        _merge_dict(event_payload, event_overrides)
    write_yaml(event_path, event_payload)
    spec = load_app_config(event_id, base_path, event_path)
    ensure_storage_layout(spec.storage)
    return spec


def _merge_dict(base: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
