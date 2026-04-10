from __future__ import annotations

from io import StringIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from ..discovery import discover_omni_assets, discover_omni_series
from ..models import EventSpec, OmniSeries, SourceAsset
from ..registry.manifests import read_source_assets, write_omni_series, write_source_assets
from ..utils import file_asset_id, stage_local_file


BZ_DATASET_ID = "OMNI_HRO2_1MIN"
INDEX_DATASET_ID = "OMNI2_H0_MRG1HR"
DATA_URL = "https://cdaweb.gsfc.nasa.gov/hapi/data"


def fetch_omni_assets(event_spec: EventSpec) -> list[SourceAsset]:
    if not event_spec.sources["omni"].get("enabled", False):
        return []

    manifest_path = event_spec.storage.manifests_dir / "omni_assets.csv"
    cached = read_source_assets(manifest_path)
    if cached:
        return cached

    source_cfg = event_spec.sources["omni"]
    mode = str(source_cfg.get("mode", "remote")).lower()
    cache_dir = event_spec.storage.cache_root / "omni" / event_spec.event_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    if mode == "local":
        local_root = _resolve_local_root(event_spec, source_cfg.get("local_root"))
        files = [Path(item) for item in source_cfg.get("files", [])]
        if files:
            resolved_files = []
            for raw_path in files:
                path = raw_path if raw_path.is_absolute() else (event_spec.project_root / raw_path).resolve()
                resolved_files.append(path)
        else:
            resolved_files = [
                local_root / f"omni_bz_1min_{event_spec.event_id}.csv",
                local_root / f"omni_dst_kp_hourly_{event_spec.event_id}.csv",
                local_root / f"omni_kp_3hour_{event_spec.event_id}.csv",
            ]
        assets: list[SourceAsset] = []
        for path in resolved_files:
            if not path.exists():
                raise FileNotFoundError(f"Local OMNI file does not exist: {path}")
            if local_root is not None and path.resolve().is_relative_to(local_root):
                staged = path.resolve()
            else:
                staged = stage_local_file(path, cache_dir / path.name)
            assets.append(
                SourceAsset(
                    event_id=event_spec.event_id,
                    source_kind="omni",
                    provider="local",
                    asset_id=file_asset_id(staged),
                    local_path=staged,
                    status="ready",
                )
            )
        write_source_assets(manifest_path, assets)
        return assets

    bz_frame = _fetch_bz_window(event_spec)
    indices_frame = _fetch_index_window(event_spec)
    kp_frame = _reduce_kp_to_3hour(indices_frame)

    bz_csv = cache_dir / f"omni_bz_1min_{event_spec.event_id}.csv"
    hourly_csv = cache_dir / f"omni_dst_kp_hourly_{event_spec.event_id}.csv"
    kp_csv = cache_dir / f"omni_kp_3hour_{event_spec.event_id}.csv"
    _save_bz_csv(bz_frame, bz_csv)
    _save_indices_csv(indices_frame, kp_frame, hourly_csv, kp_csv)

    assets = [
        SourceAsset(event_id=event_spec.event_id, source_kind="omni", provider="remote", asset_id=file_asset_id(bz_csv), local_path=bz_csv, status="ready"),
        SourceAsset(event_id=event_spec.event_id, source_kind="omni", provider="remote", asset_id=file_asset_id(hourly_csv), local_path=hourly_csv, status="ready"),
        SourceAsset(event_id=event_spec.event_id, source_kind="omni", provider="remote", asset_id=file_asset_id(kp_csv), local_path=kp_csv, status="ready"),
    ]
    write_source_assets(manifest_path, assets)
    return assets


def process_omni_assets(event_spec: EventSpec) -> OmniSeries:
    asset_manifest = event_spec.storage.manifests_dir / "omni_assets.csv"
    assets = read_source_assets(asset_manifest)
    if not assets:
        assets = discover_omni_assets(event_spec)
    if not assets:
        raise FileNotFoundError("No local OMNI cache files were discovered for the configured event.")

    path_map = {asset.local_path.name: asset.local_path for asset in assets}
    bz_path = next(path for name, path in path_map.items() if "bz_1min" in name)
    hourly_path = next(path for name, path in path_map.items() if "hourly" in name)
    kp_path = next(path for name, path in path_map.items() if "kp_3hour" in name)

    processed_dir = event_spec.storage.processed_omni_dir
    bz_processed = stage_local_file(bz_path, processed_dir / bz_path.name)
    hourly_processed = stage_local_file(hourly_path, processed_dir / hourly_path.name)
    kp_processed = stage_local_file(kp_path, processed_dir / kp_path.name)

    series = OmniSeries(
        event_id=event_spec.event_id,
        start_utc=event_spec.start_utc,
        end_utc=event_spec.end_utc,
        bz_csv_path=bz_processed,
        hourly_csv_path=hourly_processed,
        kp_csv_path=kp_processed,
    )
    write_omni_series(event_spec.storage.manifests_dir / "omni_series.csv", series)
    return series


def _fetch_bz_window(event_spec: EventSpec) -> pd.DataFrame:
    url = _hapi_query(BZ_DATASET_ID, "BZ_GSM", event_spec.start_utc.isoformat(), event_spec.end_utc.isoformat())
    frame = _fetch_csv(url, ["Time", "IMF_Bz_nT"])
    frame["Time"] = pd.to_datetime(frame["Time"], utc=True)
    frame["IMF_Bz_nT"] = frame["IMF_Bz_nT"].replace(9999.99, np.nan)
    return frame.sort_values("Time").reset_index(drop=True)


def _fetch_index_window(event_spec: EventSpec) -> pd.DataFrame:
    url = _hapi_query(INDEX_DATASET_ID, "KP1800,DST1800", event_spec.start_utc.isoformat(), event_spec.end_utc.isoformat())
    frame = _fetch_csv(url, ["Time", "Kp_code", "Dst_nT"])
    frame["Time"] = pd.to_datetime(frame["Time"], utc=True)
    frame = frame.sort_values("Time").reset_index(drop=True)
    frame["PlotTime"] = frame["Time"] - pd.Timedelta(minutes=30)
    frame["Dst_nT"] = frame["Dst_nT"].replace(99999, np.nan)
    frame["Kp_code"] = frame["Kp_code"].replace(99, np.nan)
    frame["Kp"] = frame["Kp_code"].map(_kp_code_to_decimal)
    return frame


def _reduce_kp_to_3hour(indices_frame: pd.DataFrame) -> pd.DataFrame:
    kp_frame = indices_frame.loc[:, ["PlotTime", "Kp"]].dropna().copy()
    kp_frame["KpStart"] = kp_frame["PlotTime"].dt.floor("3h")
    kp_frame = kp_frame.groupby("KpStart", as_index=False)["Kp"].first()
    kp_frame["KpEnd"] = kp_frame["KpStart"] + pd.Timedelta(hours=3)
    return kp_frame


def _save_bz_csv(frame: pd.DataFrame, output_path: Path) -> None:
    export_frame = frame.copy()
    export_frame["Time"] = export_frame["Time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    export_frame.to_csv(output_path, index=False, encoding="utf-8-sig")


def _save_indices_csv(frame: pd.DataFrame, kp_frame: pd.DataFrame, hourly_path: Path, kp_path: Path) -> None:
    hourly_export = frame.copy()
    hourly_export["Time"] = hourly_export["Time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    hourly_export["PlotTime"] = hourly_export["PlotTime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    hourly_export.to_csv(hourly_path, index=False, encoding="utf-8-sig")

    kp_export = kp_frame.copy()
    kp_export["KpStart"] = kp_export["KpStart"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    kp_export["KpEnd"] = kp_export["KpEnd"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    kp_export.to_csv(kp_path, index=False, encoding="utf-8-sig")


def _kp_code_to_decimal(value):
    if value is None or pd.isna(value):
        return np.nan
    value = int(value)
    remainder_map = {0: 0.0, 3: 1 / 3, 7: 2 / 3}
    remainder = value % 10
    if remainder not in remainder_map:
        return value / 10.0
    return value // 10 + remainder_map[remainder]


def _hapi_query(dataset_id: str, parameters: str, start: str, end: str) -> str:
    query = {
        "id": dataset_id,
        "parameters": parameters,
        "time.min": start,
        "time.max": end,
        "format": "csv",
    }
    return f"{DATA_URL}?{urlencode(query)}"


def _fetch_csv(url: str, column_names: list[str]) -> pd.DataFrame:
    with urlopen(url, timeout=90) as response:
        payload = response.read().decode("utf-8", errors="replace")
    if "HAPI error" in payload:
        raise RuntimeError(payload)
    return pd.read_csv(StringIO(payload), names=column_names)


def _resolve_local_root(event_spec: EventSpec, raw_root: str | None) -> Path:
    if raw_root is None:
        direct_root = event_spec.storage.cache_root / "omni" / event_spec.event_id
        if direct_root.exists():
            return direct_root
        return event_spec.storage.cache_root / "omni" / "local" / event_spec.event_id
    path = Path(raw_root)
    if path.is_absolute():
        return path.resolve()
    return (event_spec.project_root / path).resolve()
