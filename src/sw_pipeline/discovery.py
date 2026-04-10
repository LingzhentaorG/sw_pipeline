from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import EventSpec, GoldScene, OmniSeries, SourceAsset
from .utils import file_asset_id


def discover_isee_grid_assets(event_spec: EventSpec) -> list[SourceAsset]:
    source_cfg = event_spec.sources.get("gnss_grid", {})
    metrics = tuple(str(item).upper() for item in source_cfg.get("metrics", ["VTEC", "ROTI"]))
    root = event_spec.storage.cache_root / "gnss_grid" / "isee"
    assets: list[SourceAsset] = []

    for day in event_spec.event_days():
        year = day.strftime("%Y")
        doy = day.strftime("%j")
        for metric in metrics:
            for path in sorted((root / metric / year / doy).glob("*.nc")):
                assets.append(
                    SourceAsset(
                        event_id=event_spec.event_id,
                        source_kind="gnss_grid",
                        provider="isee_cache",
                        asset_id=file_asset_id(path),
                        local_path=path,
                        status="ready",
                        metadata={"metric": metric, "year": year, "doy": doy},
                    )
                )
    return assets


def discover_gold_assets(event_spec: EventSpec) -> list[SourceAsset]:
    root = event_spec.storage.cache_root / "gold"
    assets: list[SourceAsset] = []
    for path in sorted(root.glob("*.tar")):
        assets.append(
            SourceAsset(
                event_id=event_spec.event_id,
                source_kind="gold",
                provider="cache",
                asset_id=file_asset_id(path),
                local_path=path,
                status="ready",
                metadata={},
            )
        )
    return assets


def discover_gold_scenes(event_spec: EventSpec) -> list[GoldScene]:
    from .internal import gold_core

    allowed_days = {pd.Timestamp(day).date() for day in event_spec.event_days()}
    scenes: list[GoldScene] = []
    for asset in discover_gold_assets(event_spec):
        entries = gold_core.discover_entries(asset.local_path)
        pairs, _ = gold_core.match_pairs(entries, float(event_spec.runtime.get("gold_max_pair_minutes", 5)))
        for pair in pairs:
            midpoint = pd.Timestamp(pair.midpoint, tz="UTC").to_pydatetime()
            if pd.Timestamp(midpoint).date() not in allowed_days:
                continue
            scenes.append(
                GoldScene(
                    event_id=event_spec.event_id,
                    tar_path=asset.local_path,
                    midpoint=midpoint,
                    cha_member=pair.cha.member_name,
                    chb_member=pair.chb.member_name,
                    cha_time=pd.Timestamp(pair.cha.obs_time, tz="UTC").to_pydatetime(),
                    chb_time=pd.Timestamp(pair.chb.obs_time, tz="UTC").to_pydatetime(),
                    delta_minutes=pair.delta.total_seconds() / 60.0,
                )
            )
    scenes.sort(key=lambda item: (item.midpoint, item.tar_path.name, item.cha_member, item.chb_member))
    return scenes


def discover_omni_assets(event_spec: EventSpec) -> list[SourceAsset]:
    assets: list[SourceAsset] = []
    for path in _discover_omni_local_files(event_spec):
        assets.append(
            SourceAsset(
                event_id=event_spec.event_id,
                source_kind="omni",
                provider="cache",
                asset_id=file_asset_id(path),
                local_path=path,
                status="ready",
                metadata={},
            )
        )
    return assets


def discover_omni_series(event_spec: EventSpec) -> OmniSeries:
    paths = _discover_omni_local_files(event_spec)
    path_map = {path.name: path for path in paths}
    bz_path = next(path for name, path in path_map.items() if "bz_1min" in name)
    hourly_path = next(path for name, path in path_map.items() if "hourly" in name)
    kp_path = next(path for name, path in path_map.items() if "kp_3hour" in name)
    return OmniSeries(
        event_id=event_spec.event_id,
        start_utc=event_spec.start_utc,
        end_utc=event_spec.end_utc,
        bz_csv_path=bz_path,
        hourly_csv_path=hourly_path,
        kp_csv_path=kp_path,
    )


def _discover_omni_local_files(event_spec: EventSpec) -> list[Path]:
    source_cfg = event_spec.sources.get("omni", {})
    files = source_cfg.get("files", [])
    if files:
        resolved_paths: list[Path] = []
        for raw in files:
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                path = (event_spec.project_root / path).resolve()
            resolved_paths.append(path)
        _ensure_paths_exist(resolved_paths)
        return resolved_paths

    candidate_roots = _candidate_omni_roots(event_spec)
    for root in candidate_roots:
        resolved_paths = [
            root / f"omni_bz_1min_{event_spec.event_id}.csv",
            root / f"omni_dst_kp_hourly_{event_spec.event_id}.csv",
            root / f"omni_kp_3hour_{event_spec.event_id}.csv",
        ]
        if all(path.exists() for path in resolved_paths):
            return resolved_paths

    attempted = ", ".join(str(path) for path in candidate_roots)
    raise FileNotFoundError(f"Local OMNI files do not exist under: {attempted}")


def _candidate_omni_roots(event_spec: EventSpec) -> list[Path]:
    source_cfg = event_spec.sources.get("omni", {})
    raw_root = source_cfg.get("local_root")
    roots: list[Path] = []
    if raw_root is not None:
        root = Path(str(raw_root)).expanduser()
        if not root.is_absolute():
            root = (event_spec.project_root / root).resolve()
        roots.append(root)

    defaults = [
        event_spec.storage.cache_root / "omni" / event_spec.event_id,
        event_spec.storage.cache_root / "omni" / "local" / event_spec.event_id,
    ]
    for root in defaults:
        if root not in roots:
            roots.append(root)
    return roots


def _ensure_paths_exist(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Local OMNI file does not exist: {missing[0]}")
