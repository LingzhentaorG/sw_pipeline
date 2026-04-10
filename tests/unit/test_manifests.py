from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sw_pipeline.models import (
    GnssDownloadAsset,
    GnssGridProduct,
    GnssStationCandidate,
    GoldScene,
    OmniSeries,
    SourceAsset,
)
from sw_pipeline.registry.manifests import (
    read_gnss_download_assets,
    read_gnss_grid_products,
    read_gnss_station_candidates,
    read_gold_scenes,
    read_omni_series,
    read_source_assets,
    write_gnss_download_assets,
    write_gnss_grid_products,
    write_gnss_station_candidates,
    write_gold_scenes,
    write_omni_series,
    write_source_assets,
)


def test_manifest_roundtrips(tmp_path: Path):
    source_assets = [
        SourceAsset(
            event_id="evt",
            source_kind="gold",
            provider="local",
            asset_id="asset-1",
            local_path=tmp_path / "scene.tar",
            status="ready",
            metadata={"kind": "test"},
        )
    ]
    write_source_assets(tmp_path / "assets.csv", source_assets)
    loaded_assets = read_source_assets(tmp_path / "assets.csv")
    assert loaded_assets[0].metadata == {"kind": "test"}

    products = [
        GnssGridProduct(
            event_id="evt",
            producer="internal",
            source_kind="gnss_grid",
            path=tmp_path / "grid.nc",
            metrics=("vtec", "roti"),
            time_start=datetime(2024, 10, 10, 0, 0, tzinfo=UTC),
            time_end=datetime(2024, 10, 10, 1, 0, tzinfo=UTC),
            metadata={"segment": "a"},
        )
    ]
    write_gnss_grid_products(tmp_path / "products.csv", products)
    loaded_products = read_gnss_grid_products(tmp_path / "products.csv")
    assert loaded_products[0].metrics == ("vtec", "roti")

    scenes = [
        GoldScene(
            event_id="evt",
            tar_path=tmp_path / "scene.tar",
            midpoint=datetime(2024, 10, 10, 0, 10, tzinfo=UTC),
            cha_member="CHA.nc",
            chb_member="CHB.nc",
            cha_time=datetime(2024, 10, 10, 0, 9, tzinfo=UTC),
            chb_time=datetime(2024, 10, 10, 0, 11, tzinfo=UTC),
            delta_minutes=2.5,
        )
    ]
    write_gold_scenes(tmp_path / "gold.csv", scenes)
    loaded_scenes = read_gold_scenes(tmp_path / "gold.csv")
    assert loaded_scenes[0].cha_member == "CHA.nc"
    assert loaded_scenes[0].cha_time == datetime(2024, 10, 10, 0, 9, tzinfo=UTC)
    assert loaded_scenes[0].chb_time == datetime(2024, 10, 10, 0, 11, tzinfo=UTC)

    series = OmniSeries(
        event_id="evt",
        start_utc=datetime(2024, 10, 10, 0, 0, tzinfo=UTC),
        end_utc=datetime(2024, 10, 10, 1, 0, tzinfo=UTC),
        bz_csv_path=tmp_path / "bz.csv",
        hourly_csv_path=tmp_path / "hourly.csv",
        kp_csv_path=tmp_path / "kp.csv",
    )
    write_omni_series(tmp_path / "omni.csv", series)
    loaded_series = read_omni_series(tmp_path / "omni.csv")
    assert loaded_series is not None
    assert loaded_series.kp_csv_path == Path(tmp_path / "kp.csv")

    station_candidates = [
        GnssStationCandidate(
            event_id="evt",
            provider="noaa",
            station_id="ABCD00USA",
            station_code4="ABCD",
            observation_date="2024-10-10",
            sampling_sec=30,
            lat=10.0,
            lon=-70.0,
            height_m=120.0,
            obs_url="https://example.test/abcd.obs.gz",
            nav_url="https://example.test/brdc.gz",
            metadata={"priority": 10},
        )
    ]
    write_gnss_station_candidates(tmp_path / "gnss_station_candidates.csv", station_candidates)
    loaded_candidates = read_gnss_station_candidates(tmp_path / "gnss_station_candidates.csv")
    assert loaded_candidates[0].station_code4 == "ABCD"
    assert loaded_candidates[0].metadata == {"priority": 10}

    gnss_download_assets = [
        GnssDownloadAsset(
            event_id="evt",
            source_kind="gnss_observation",
            provider="cddis",
            protocol="https",
            station_id="ABCD00USA",
            station_code4="ABCD",
            observation_date="2024-10-10",
            url="https://example.test/abcd.obs.gz",
            local_path=tmp_path / "abcd.obs.gz",
            status="ok",
            attempts=2,
            auth_ref="cddis",
            metadata={"sampling_sec": 30},
        )
    ]
    write_gnss_download_assets(tmp_path / "gnss_download_assets.csv", gnss_download_assets)
    loaded_download_assets = read_gnss_download_assets(tmp_path / "gnss_download_assets.csv")
    assert loaded_download_assets[0].auth_ref == "cddis"
    assert loaded_download_assets[0].metadata == {"sampling_sec": 30}
