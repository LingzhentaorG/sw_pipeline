from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..models import (
    GnssDownloadAsset,
    GnssGridProduct,
    GnssStationCandidate,
    GoldScene,
    OmniSeries,
    SourceAsset,
)


SOURCE_ASSET_FIELDS = [
    "event_id",
    "source_kind",
    "provider",
    "asset_id",
    "local_path",
    "status",
    "metadata_json",
]

GNSS_PRODUCT_FIELDS = [
    "event_id",
    "producer",
    "source_kind",
    "path",
    "metrics_json",
    "time_start",
    "time_end",
    "metadata_json",
]

GNSS_STATION_CANDIDATE_FIELDS = [
    "event_id",
    "provider",
    "station_id",
    "station_code4",
    "observation_date",
    "sampling_sec",
    "lat",
    "lon",
    "height_m",
    "obs_url",
    "nav_url",
    "status",
    "metadata_json",
]

GNSS_DOWNLOAD_ASSET_FIELDS = [
    "event_id",
    "source_kind",
    "provider",
    "protocol",
    "station_id",
    "station_code4",
    "observation_date",
    "url",
    "local_path",
    "status",
    "attempts",
    "error",
    "auth_ref",
    "metadata_json",
]

GOLD_SCENE_FIELDS = [
    "event_id",
    "tar_path",
    "midpoint",
    "cha_member",
    "chb_member",
    "cha_time",
    "chb_time",
    "delta_minutes",
]

OMNI_SERIES_FIELDS = [
    "event_id",
    "start_utc",
    "end_utc",
    "bz_csv_path",
    "hourly_csv_path",
    "kp_csv_path",
]

MIGRATION_FIELDS = [
    "project",
    "source_kind",
    "event_id",
    "cached_path",
    "original_relpath",
    "size_bytes",
]

STAGE_STATUS_FIELDS = [
    "stage",
    "target",
    "status",
    "detail",
]

PANEL_OUTPUT_FIELDS = [
    "panel_name",
    "slot_index",
    "kind",
    "requested_time",
    "resolved_time",
    "status",
    "detail",
    "output_path",
]


def write_source_assets(path: Path, assets: Iterable[SourceAsset]) -> None:
    rows = [
        {
            "event_id": asset.event_id,
            "source_kind": asset.source_kind,
            "provider": asset.provider,
            "asset_id": asset.asset_id,
            "local_path": str(asset.local_path),
            "status": asset.status,
            "metadata_json": _dump_json(asset.metadata),
        }
        for asset in assets
    ]
    _write_csv(path, SOURCE_ASSET_FIELDS, rows)


def read_source_assets(path: Path) -> list[SourceAsset]:
    if not _has_rows(path):
        return []
    frame = pd.read_csv(path)
    assets: list[SourceAsset] = []
    for row in frame.to_dict("records"):
        assets.append(
            SourceAsset(
                event_id=str(row["event_id"]),
                source_kind=str(row["source_kind"]),
                provider=str(row["provider"]),
                asset_id=str(row["asset_id"]),
                local_path=Path(str(row["local_path"])),
                status=str(row["status"]),
                metadata=_load_json(row.get("metadata_json")),
            )
        )
    return assets


def write_gnss_grid_products(path: Path, products: Iterable[GnssGridProduct]) -> None:
    rows = [
        {
            "event_id": product.event_id,
            "producer": product.producer,
            "source_kind": product.source_kind,
            "path": str(product.path),
            "metrics_json": _dump_json(product.metrics),
            "time_start": pd.Timestamp(product.time_start).isoformat(),
            "time_end": pd.Timestamp(product.time_end).isoformat(),
            "metadata_json": _dump_json(product.metadata),
        }
        for product in products
    ]
    _write_csv(path, GNSS_PRODUCT_FIELDS, rows)


def write_gnss_station_candidates(path: Path, candidates: Iterable[GnssStationCandidate]) -> None:
    rows = [
        {
            "event_id": candidate.event_id,
            "provider": candidate.provider,
            "station_id": candidate.station_id,
            "station_code4": candidate.station_code4,
            "observation_date": candidate.observation_date,
            "sampling_sec": candidate.sampling_sec,
            "lat": candidate.lat,
            "lon": candidate.lon,
            "height_m": candidate.height_m,
            "obs_url": candidate.obs_url,
            "nav_url": candidate.nav_url,
            "status": candidate.status,
            "metadata_json": _dump_json(candidate.metadata),
        }
        for candidate in candidates
    ]
    _write_csv(path, GNSS_STATION_CANDIDATE_FIELDS, rows)


def read_gnss_station_candidates(path: Path) -> list[GnssStationCandidate]:
    if not _has_rows(path):
        return []
    frame = pd.read_csv(path)
    candidates: list[GnssStationCandidate] = []
    for row in frame.to_dict("records"):
        candidates.append(
            GnssStationCandidate(
                event_id=str(row["event_id"]),
                provider=str(row["provider"]),
                station_id=str(row["station_id"]),
                station_code4=str(row["station_code4"]),
                observation_date=str(row["observation_date"]),
                sampling_sec=int(row["sampling_sec"]),
                lat=_nullable_float(row.get("lat")),
                lon=_nullable_float(row.get("lon")),
                height_m=_nullable_float(row.get("height_m")),
                obs_url=str(row.get("obs_url", "")),
                nav_url=str(row.get("nav_url", "")),
                status=str(row.get("status", "candidate")),
                metadata=_load_json(row.get("metadata_json")),
            )
        )
    return candidates


def write_gnss_download_assets(path: Path, assets: Iterable[GnssDownloadAsset]) -> None:
    rows = [
        {
            "event_id": asset.event_id,
            "source_kind": asset.source_kind,
            "provider": asset.provider,
            "protocol": asset.protocol,
            "station_id": asset.station_id,
            "station_code4": asset.station_code4,
            "observation_date": asset.observation_date,
            "url": asset.url,
            "local_path": str(asset.local_path) if asset.local_path else "",
            "status": asset.status,
            "attempts": asset.attempts,
            "error": asset.error or "",
            "auth_ref": asset.auth_ref or "",
            "metadata_json": _dump_json(asset.metadata),
        }
        for asset in assets
    ]
    _write_csv(path, GNSS_DOWNLOAD_ASSET_FIELDS, rows)


def read_gnss_download_assets(path: Path) -> list[GnssDownloadAsset]:
    if not _has_rows(path):
        return []
    frame = pd.read_csv(path)
    assets: list[GnssDownloadAsset] = []
    for row in frame.to_dict("records"):
        local_path_raw = row.get("local_path")
        assets.append(
            GnssDownloadAsset(
                event_id=str(row["event_id"]),
                source_kind=str(row["source_kind"]),
                provider=str(row["provider"]),
                protocol=str(row["protocol"]),
                station_id=str(row["station_id"]),
                station_code4=str(row["station_code4"]),
                observation_date=str(row["observation_date"]),
                url=str(row["url"]),
                local_path=Path(str(local_path_raw)) if _has_text(local_path_raw) else None,
                status=str(row["status"]),
                attempts=int(row.get("attempts", 0)),
                error=_nullable_text(row.get("error")),
                auth_ref=_nullable_text(row.get("auth_ref")),
                metadata=_load_json(row.get("metadata_json")),
            )
        )
    return assets


def upsert_gnss_grid_products(path: Path, producer: str, products: Iterable[GnssGridProduct]) -> list[GnssGridProduct]:
    existing = read_gnss_grid_products(path)
    merged = [item for item in existing if item.producer != producer]
    merged.extend(products)
    merged.sort(key=lambda item: (item.producer, item.time_start, item.path.name))
    write_gnss_grid_products(path, merged)
    return merged


def read_gnss_grid_products(path: Path) -> list[GnssGridProduct]:
    if not _has_rows(path):
        return []
    frame = pd.read_csv(path)
    products: list[GnssGridProduct] = []
    for row in frame.to_dict("records"):
        metrics = tuple(str(item).lower() for item in _load_json(row.get("metrics_json"), default=[]))
        products.append(
            GnssGridProduct(
                event_id=str(row["event_id"]),
                producer=str(row["producer"]),
                source_kind=str(row["source_kind"]),
                path=Path(str(row["path"])),
                metrics=metrics,
                time_start=pd.Timestamp(row["time_start"]).to_pydatetime(),
                time_end=pd.Timestamp(row["time_end"]).to_pydatetime(),
                metadata=_load_json(row.get("metadata_json")),
            )
        )
    return products


def write_gold_scenes(path: Path, scenes: Iterable[GoldScene]) -> None:
    rows = [
        {
            "event_id": scene.event_id,
            "tar_path": str(scene.tar_path),
            "midpoint": pd.Timestamp(scene.midpoint).isoformat(),
            "cha_member": scene.cha_member,
            "chb_member": scene.chb_member,
            "cha_time": pd.Timestamp(scene.cha_time).isoformat() if scene.cha_time is not None else "",
            "chb_time": pd.Timestamp(scene.chb_time).isoformat() if scene.chb_time is not None else "",
            "delta_minutes": scene.delta_minutes,
        }
        for scene in scenes
    ]
    _write_csv(path, GOLD_SCENE_FIELDS, rows)


def read_gold_scenes(path: Path) -> list[GoldScene]:
    if not _has_rows(path):
        return []
    frame = pd.read_csv(path)
    scenes: list[GoldScene] = []
    for row in frame.to_dict("records"):
        scenes.append(
            GoldScene(
                event_id=str(row["event_id"]),
                tar_path=Path(str(row["tar_path"])),
                midpoint=pd.Timestamp(row["midpoint"]).to_pydatetime(),
                cha_member=str(row["cha_member"]),
                chb_member=str(row["chb_member"]),
                cha_time=pd.Timestamp(row["cha_time"]).to_pydatetime() if _has_text(row.get("cha_time")) else None,
                chb_time=pd.Timestamp(row["chb_time"]).to_pydatetime() if _has_text(row.get("chb_time")) else None,
                delta_minutes=float(row["delta_minutes"]),
            )
        )
    return scenes


def write_omni_series(path: Path, series: OmniSeries) -> None:
    _write_csv(
        path,
        OMNI_SERIES_FIELDS,
        [
            {
                "event_id": series.event_id,
                "start_utc": pd.Timestamp(series.start_utc).isoformat(),
                "end_utc": pd.Timestamp(series.end_utc).isoformat(),
                "bz_csv_path": str(series.bz_csv_path),
                "hourly_csv_path": str(series.hourly_csv_path),
                "kp_csv_path": str(series.kp_csv_path),
            }
        ],
    )


def read_omni_series(path: Path) -> OmniSeries | None:
    if not _has_rows(path):
        return None
    row = pd.read_csv(path).iloc[0].to_dict()
    return OmniSeries(
        event_id=str(row["event_id"]),
        start_utc=pd.Timestamp(row["start_utc"]).to_pydatetime(),
        end_utc=pd.Timestamp(row["end_utc"]).to_pydatetime(),
        bz_csv_path=Path(str(row["bz_csv_path"])),
        hourly_csv_path=Path(str(row["hourly_csv_path"])),
        kp_csv_path=Path(str(row["kp_csv_path"])),
    )


def write_overlay_pairs(path: Path, rows: Iterable[dict[str, object]]) -> None:
    fieldnames = [
        "overlay_name",
        "metric",
        "gnss_time",
        "gold_time",
        "delta_seconds",
        "status",
        "detail",
        "output_path",
    ]
    normalized = []
    for row in rows:
        normalized.append(
            {
                "overlay_name": row.get("overlay_name", ""),
                "metric": row.get("metric", ""),
                "gnss_time": _stringify_value(row.get("gnss_time")),
                "gold_time": _stringify_value(row.get("gold_time")),
                "delta_seconds": row.get("delta_seconds", ""),
                "status": row.get("status", ""),
                "detail": row.get("detail", ""),
                "output_path": _stringify_value(row.get("output_path")),
            }
        )
    _write_csv(path, fieldnames, normalized)


def write_migration_manifest(path: Path, rows: Iterable[dict[str, object]]) -> None:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "project": row.get("project", ""),
                "source_kind": row.get("source_kind", ""),
                "event_id": row.get("event_id", ""),
                "cached_path": _stringify_value(row.get("cached_path")),
                "original_relpath": row.get("original_relpath", ""),
                "size_bytes": row.get("size_bytes", 0),
            }
        )
    _write_csv(path, MIGRATION_FIELDS, normalized)


def write_stage_status(path: Path, rows: Iterable[dict[str, object]]) -> None:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "stage": row.get("stage", ""),
                "target": row.get("target", ""),
                "status": row.get("status", ""),
                "detail": row.get("detail", ""),
            }
        )
    _write_csv(path, STAGE_STATUS_FIELDS, normalized)


def write_panel_outputs(path: Path, rows: Iterable[dict[str, object]]) -> None:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "panel_name": row.get("panel_name", ""),
                "slot_index": row.get("slot_index", ""),
                "kind": row.get("kind", ""),
                "requested_time": _stringify_value(row.get("requested_time")),
                "resolved_time": _stringify_value(row.get("resolved_time")),
                "status": row.get("status", ""),
                "detail": row.get("detail", ""),
                "output_path": _stringify_value(row.get("output_path")),
            }
        )
    _write_csv(path, PANEL_OUTPUT_FIELDS, normalized)


def _has_rows(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dump_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_json(value: object, default: object | None = None):
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return {} if default is None else default
    return json.loads(str(value))


def _stringify_value(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _nullable_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return None
    return float(value)


def _nullable_text(value: object) -> str | None:
    if not _has_text(value):
        return None
    return str(value)


def _has_text(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return str(value) != ""
