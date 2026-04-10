from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..discovery import discover_gold_assets
from ..models import EventSpec, GoldScene, SourceAsset
from ..registry.manifests import read_source_assets, write_gold_scenes, write_source_assets
from ..utils import file_asset_id, stage_local_file


def fetch_gold_assets(event_spec: EventSpec) -> list[SourceAsset]:
    if not event_spec.sources["gold"].get("enabled", False):
        return []

    manifest_path = event_spec.storage.manifests_dir / "gold_assets.csv"
    cached = read_source_assets(manifest_path)
    if cached:
        return cached

    source_cfg = event_spec.sources["gold"]
    mode = str(source_cfg.get("mode", "local")).lower()
    if mode != "local":
        raise NotImplementedError("Only local GOLD inputs are implemented in V1.")

    assets: list[SourceAsset] = []
    for raw_input in source_cfg.get("inputs", []):
        path = Path(str(raw_input)).expanduser()
        if not path.is_absolute():
            path = (event_spec.project_root / path).resolve()
        tar_paths = sorted(path.rglob("*.tar")) if path.is_dir() else [path]
        for tar_path in tar_paths:
            staged = stage_local_file(tar_path, event_spec.storage.cache_root / "gold" / tar_path.name)
            assets.append(
                SourceAsset(
                    event_id=event_spec.event_id,
                    source_kind="gold",
                    provider="local",
                    asset_id=file_asset_id(staged),
                    local_path=staged,
                    status="ready",
                    metadata={},
                )
            )

    if not assets:
        raise FileNotFoundError("No GOLD tar archives were discovered for the configured inputs.")
    write_source_assets(manifest_path, assets)
    return assets


def process_gold_assets(event_spec: EventSpec) -> list[GoldScene]:
    if not event_spec.sources["gold"].get("enabled", False):
        return []

    asset_manifest = event_spec.storage.manifests_dir / "gold_assets.csv"
    assets = read_source_assets(asset_manifest)
    if not assets:
        assets = discover_gold_assets(event_spec)
    if not assets:
        raise FileNotFoundError("No GOLD tar archives were discovered under storage/cache/gold.")

    from ..internal import gold_core

    allowed_days = {pd.Timestamp(day).date() for day in event_spec.event_days()}
    scenes: list[GoldScene] = []
    for asset in assets:
        tar_path = asset.local_path
        entries = gold_core.discover_entries(tar_path)
        pairs, _ = gold_core.match_pairs(entries, float(event_spec.runtime.get("gold_max_pair_minutes", 5)))
        for pair in pairs:
            midpoint = pd.Timestamp(pair.midpoint, tz="UTC").to_pydatetime()
            if pd.Timestamp(midpoint).date() not in allowed_days:
                continue
            scenes.append(
                GoldScene(
                    event_id=event_spec.event_id,
                    tar_path=tar_path,
                    midpoint=midpoint,
                    cha_member=pair.cha.member_name,
                    chb_member=pair.chb.member_name,
                    cha_time=pd.Timestamp(pair.cha.obs_time, tz="UTC").to_pydatetime(),
                    chb_time=pd.Timestamp(pair.chb.obs_time, tz="UTC").to_pydatetime(),
                    delta_minutes=pair.delta.total_seconds() / 60.0,
                )
            )

    if not scenes:
        raise FileNotFoundError("No GOLD scene pairs were available inside the staged tar archives for this event.")
    write_gold_scenes(event_spec.storage.manifests_dir / "gold_scenes.csv", scenes)
    return scenes
