from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin

from ..discovery import discover_isee_grid_assets
from ..downloaders import create_retry_session, download_to_path, fetch_text
from ..models import EventSpec, GnssGridProduct, SourceAsset
from ..normalizers.gnss import normalize_isee_products
from ..registry.manifests import read_source_assets, upsert_gnss_grid_products, write_source_assets
from ..storage import reset_generated_directory
from ..utils import daterange_days, file_asset_id, stage_local_file


SOURCE_TEMPLATES = {
    "VTEC": "https://stdb2.isee.nagoya-u.ac.jp/GPS/shinbori/AGRID2/nc/{year}/",
    "ROTI": "https://stdb2.isee.nagoya-u.ac.jp/GPS/shinbori/RGRID2/nc/{year}/",
}


def fetch_gnss_grid_assets(event_spec: EventSpec) -> list[SourceAsset]:
    if not event_spec.sources["gnss_grid"].get("enabled", False):
        return []

    manifest_path = event_spec.storage.manifests_dir / "gnss_grid_assets.csv"
    cached = read_source_assets(manifest_path)
    if cached:
        return cached

    source_cfg = event_spec.sources["gnss_grid"]
    mode = str(source_cfg.get("mode", "local")).lower()
    metrics = tuple(str(item).upper() for item in source_cfg.get("metrics", ["VTEC", "ROTI"]))
    assets: list[SourceAsset] = []

    if mode == "local":
        root = _resolve_local_root(event_spec, source_cfg.get("local_root"))
        for day in daterange_days(event_spec.start_utc, event_spec.end_utc):
            year = day.strftime("%Y")
            doy = day.strftime("%j")
            for metric in metrics:
                for file_path in _local_metric_paths(root, metric, year, doy):
                    staged = stage_local_file(
                        file_path,
                        event_spec.storage.cache_root / "gnss_grid" / "isee" / metric / year / doy / file_path.name,
                    )
                    assets.append(
                        SourceAsset(
                            event_id=event_spec.event_id,
                            source_kind="gnss_grid",
                            provider="isee_local",
                            asset_id=file_asset_id(staged),
                            local_path=staged,
                            status="ready",
                            metadata={"metric": metric, "year": year, "doy": doy},
                        )
                    )
    elif mode == "remote":
        session = create_retry_session(total=int(event_spec.runtime.get("gnss_grid_download_retries", 4)))
        for day in daterange_days(event_spec.start_utc, event_spec.end_utc):
            year = day.strftime("%Y")
            doy = day.strftime("%j")
            for metric in metrics:
                assets.extend(_download_remote_metric(event_spec, metric, year, doy, session))
    else:
        raise ValueError(f"Unsupported gnss_grid mode: {mode}")

    if not assets:
        raise FileNotFoundError("No ISEE GNSS grid assets were discovered for the configured event window.")
    write_source_assets(manifest_path, assets)
    return assets


def process_gnss_grid_assets(event_spec: EventSpec) -> list[GnssGridProduct]:
    asset_manifest = event_spec.storage.manifests_dir / "gnss_grid_assets.csv"
    assets = read_source_assets(asset_manifest)
    if not assets:
        assets = discover_isee_grid_assets(event_spec)
    if not assets:
        raise FileNotFoundError("No ISEE GNSS grid cache files were discovered for the configured event days.")

    reset_generated_directory(event_spec.storage.grids_dir / "isee", event_spec.storage)
    paths: list[Path] = []
    for asset in assets:
        if not asset.local_path.exists():
            continue
        metric = str(asset.metadata.get("metric", "unknown")).lower()
        year = str(asset.metadata.get("year", "unknown"))
        doy = str(asset.metadata.get("doy", "unknown"))
        staged = stage_local_file(
            asset.local_path,
            event_spec.storage.grids_dir / "isee" / metric / year / doy / asset.local_path.name,
        )
        paths.append(staged)
    if not paths:
        raise FileNotFoundError("No local ISEE GNSS grid files were available for normalization.")
    products = normalize_isee_products(event_spec.event_id, paths)
    upsert_gnss_grid_products(event_spec.storage.manifests_dir / "gnss_grid_products.csv", "isee", products)
    return products


def _download_remote_metric(event_spec: EventSpec, metric: str, year: str, doy: str, session) -> list[SourceAsset]:
    from html.parser import HTMLParser

    class NcLinkParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.links: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag.lower() != "a":
                return
            href = dict(attrs).get("href")
            if href and href.lower().endswith(".nc"):
                self.links.append(href)

    base_url = SOURCE_TEMPLATES[metric].format(year=year)
    day_url = urljoin(base_url.rstrip("/") + "/", f"{doy}/")
    payload = fetch_text(
        day_url,
        session=session,
        timeout=int(event_spec.runtime.get("gnss_grid_download_timeout_sec", 90)),
        verify=True,
    )
    parser = NcLinkParser()
    parser.feed(payload)

    assets: list[SourceAsset] = []
    for name in sorted(set(parser.links)):
        file_url = urljoin(day_url, name)
        target = event_spec.storage.cache_root / "gnss_grid" / "isee" / metric / year / doy / name
        result = download_to_path(
            file_url,
            target,
            session=session,
            timeout=int(event_spec.runtime.get("gnss_grid_download_timeout_sec", 120)),
            verify=True,
            max_retries=int(event_spec.runtime.get("gnss_grid_download_retries", 4)),
            temp_suffix=".part",
        )
        if result.status != "ok":
            raise FileNotFoundError(f"Failed to download ISEE grid asset {file_url}: {result.error}")
        assets.append(
            SourceAsset(
                event_id=event_spec.event_id,
                source_kind="gnss_grid",
                provider="isee_remote",
                asset_id=file_asset_id(target),
                local_path=target,
                status="ready",
                metadata={"metric": metric, "year": year, "doy": doy},
            )
        )
    return assets


def _local_metric_paths(root: Path, metric: str, year: str, doy: str) -> list[Path]:
    folders = [
        root / f"{metric}_data" / year / doy,
        root / metric / year / doy,
    ]
    paths: list[Path] = []
    for folder in folders:
        if folder.exists():
            paths.extend(sorted(folder.glob("*.nc")))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def _resolve_local_root(event_spec: EventSpec, raw_root: str | None) -> Path:
    if raw_root is None:
        return event_spec.storage.cache_root / "gnss_grid" / "isee"
    root = Path(raw_root)
    if root.is_absolute():
        return root.resolve()
    return (event_spec.project_root / root).resolve()
