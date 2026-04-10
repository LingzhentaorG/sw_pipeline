from __future__ import annotations

from pathlib import Path

from ..discovery import discover_gold_scenes
from ..models import EventSpec, GoldScene, SourceAsset
from ..providers.gold import fetch_gold_assets, process_gold_assets
from ..registry.manifests import read_gold_scenes


def fetch_gold(event_spec: EventSpec) -> list[SourceAsset]:
    return fetch_gold_assets(event_spec)


def process_gold(event_spec: EventSpec) -> list[GoldScene]:
    manifest_path = event_spec.storage.manifests_dir / "gold_scenes.csv"
    scenes = read_gold_scenes(manifest_path)
    if scenes and discover_gold_scenes(event_spec):
        return scenes
    return process_gold_assets(event_spec)
