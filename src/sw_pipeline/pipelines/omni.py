from __future__ import annotations

from ..discovery import discover_omni_assets
from ..models import EventSpec, OmniSeries, SourceAsset
from ..providers.omni import fetch_omni_assets, process_omni_assets
from ..registry.manifests import read_omni_series


def fetch_omni(event_spec: EventSpec) -> list[SourceAsset]:
    return fetch_omni_assets(event_spec)


def process_omni(event_spec: EventSpec) -> OmniSeries:
    manifest_path = event_spec.storage.manifests_dir / "omni_series.csv"
    series = read_omni_series(manifest_path)
    if series is not None and discover_omni_assets(event_spec):
        return series
    return process_omni_assets(event_spec)
