from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any

import pandas as pd
import requests
import yaml

from ..downloaders import create_earthdata_session, create_retry_session, download_to_path, infer_protocol
from ..models import EventSpec, GnssDownloadAsset, GnssGridProduct, GnssStationCandidate, SourceAsset
from ..normalizers.gnss import normalize_internal_products
from ..registry.manifests import (
    read_source_assets,
    upsert_gnss_grid_products,
    write_gnss_download_assets,
    write_gnss_station_candidates,
    write_source_assets,
)
from ..storage import reset_generated_directory
from ..utils import file_asset_id, glob_event_netcdf, stage_local_file

LOGGER = logging.getLogger(__name__)

LONG_RINEX_OBS_PATTERN = re.compile(
    r"^(?P<station_id>[A-Za-z0-9]{9})_R_\d{11}_01D_(?P<sampling>\d{2,3})S_.*\.(?:crx|rnx)\.gz$",
    re.IGNORECASE,
)
SHORT_RINEX_OBS_PATTERN = re.compile(
    r"^(?P<station_code4>[A-Za-z0-9]{4})\d{3}0\.\d{2}d(?:\.gz|\.Z)?$",
    re.IGNORECASE,
)


def fetch_gnss_raw_assets(event_spec: EventSpec) -> list[SourceAsset]:
    if not event_spec.sources["gnss_raw"].get("enabled", False):
        return []

    manifest_path = event_spec.storage.manifests_dir / "gnss_raw_assets.csv"
    cached = read_source_assets(manifest_path)
    if cached:
        return cached

    mode = event_spec.internal_gnss_mode()
    if mode == "internal_pipeline":
        assets = _discover_cached_raw_source_assets(event_spec)
        if not assets:
            pipeline_config_path = build_internal_pipeline_config(event_spec)
            config_module, _ = _load_internal_pipeline_modules()
            pipeline_config = config_module.load_pipeline_config(pipeline_config_path)
            assets = _fetch_internal_pipeline_assets(event_spec, pipeline_config)
    elif mode in {"workspace_snapshot", "local_workspace"}:
        workspace_root = resolve_workspace_root(event_spec)
        assets = _source_assets_from_workspace(event_spec, workspace_root)
    else:
        raise ValueError(f"Unsupported gnss_raw mode: {mode}")

    if not assets:
        raise FileNotFoundError("No GNSS raw assets were discovered for the configured event.")
    write_source_assets(manifest_path, assets)
    return assets


def process_gnss_raw_assets(event_spec: EventSpec) -> list[GnssGridProduct]:
    if not event_spec.sources["gnss_raw"].get("enabled", False):
        return []

    mode = event_spec.internal_gnss_mode()
    if mode == "internal_pipeline":
        pipeline_config_path = build_internal_pipeline_config(event_spec)
        config_module, _, preprocess_module, processing_module = _load_internal_pipeline_modules(include_processing=True)
        pipeline_config = config_module.load_pipeline_config(pipeline_config_path)
        reset_generated_directory(event_spec.storage.gnss_workspace_dir, event_spec.storage)
        reset_generated_directory(event_spec.storage.grids_dir / "internal", event_spec.storage)
        _rebuild_internal_workspace_manifests_from_cache(event_spec)
        workspace_netcdf_dir = event_spec.storage.gnss_workspace_dir / "products" / "netcdf"
        staged_netcdf_dir = event_spec.storage.grids_dir / "internal"
        watcher_stop = threading.Event()
        watcher = threading.Thread(
            target=_watch_internal_netcdf_outputs,
            args=(event_spec.event_id, workspace_netcdf_dir, staged_netcdf_dir, watcher_stop),
            daemon=True,
        )
        watcher.start()
        preprocess_module.preprocess_records(pipeline_config)
        try:
            processing_module.execute_processing_stage(pipeline_config)
        finally:
            watcher_stop.set()
            watcher.join(timeout=10)
            _sync_internal_netcdf_outputs(event_spec.event_id, workspace_netcdf_dir, staged_netcdf_dir, require_stable=False)
        workspace_root = event_spec.storage.gnss_workspace_dir
    else:
        workspace_root = resolve_workspace_root(event_spec)

    netcdf_dir = event_spec.storage.grids_dir / "internal" if mode == "internal_pipeline" else workspace_root / "products" / "netcdf"
    product_paths = glob_event_netcdf(netcdf_dir, event_spec.event_id)
    if not product_paths:
        raise FileNotFoundError(f"No internal GNSS NetCDF products were found under {netcdf_dir}")
    products = normalize_internal_products(event_spec.event_id, product_paths)
    upsert_gnss_grid_products(event_spec.storage.manifests_dir / "gnss_grid_products.csv", "internal", products)
    return products


def build_internal_pipeline_config(event_spec: EventSpec) -> Path:
    template_path = event_spec.project_root / "config" / "templates" / "internal_gnss_runtime.yaml"
    if template_path.exists():
        with template_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    else:
        raw = {}

    internal_start = pd.Timestamp(event_spec.start_utc)
    internal_end = pd.Timestamp(event_spec.end_utc)
    if event_spec.station_series_presets() and "internal" not in event_spec.gnss_map_producers():
        internal_start = min(pd.Timestamp(preset.start_utc) for preset in event_spec.station_series_presets())
        internal_end = max(pd.Timestamp(preset.end_utc) for preset in event_spec.station_series_presets())
    raw["events"] = [
        {
            "id": event_spec.event_id,
            "start": internal_start.isoformat().replace("+00:00", "Z"),
            "end": internal_end.isoformat().replace("+00:00", "Z"),
        }
    ]
    raw["bbox"] = dict(event_spec.bbox)
    raw["auth"] = {}
    raw["sources"] = {
        "observations": dict(event_spec.sources["gnss_raw"].get("providers", {})),
        "auxiliary": dict(event_spec.sources["gnss_raw"].get("auxiliary", {})),
    }
    raw.setdefault("outputs", {})
    raw["outputs"]["root"] = str(event_spec.storage.gnss_workspace_dir)
    raw["outputs"]["legacy_root"] = str(event_spec.storage.gnss_workspace_dir / "_legacy")
    raw.setdefault("download", {})
    raw["download"]["noaa_workers"] = int(event_spec.runtime.get("gnss_download_workers", 4))
    raw["download"]["other_workers"] = int(event_spec.runtime.get("gnss_download_workers", 8))
    raw.setdefault("processing", {})
    priority_station_codes = sorted({preset.station_code.lower() for preset in event_spec.station_series_presets()})
    if priority_station_codes:
        raw["processing"]["priority_station_codes"] = priority_station_codes
    raw["processing"].setdefault("progress_log_interval", 10)
    raw["processing"].setdefault("checkpoint_chunk_size", 10)
    if (
        priority_station_codes
        and "internal" not in event_spec.gnss_map_producers()
        and "max_station_days_per_event" not in raw["processing"]
    ):
        raw["processing"]["max_station_days_per_event"] = len(priority_station_codes) * len(event_spec.event_days())

    source_overrides = event_spec.sources["gnss_raw"].get("pipeline_overrides")
    if isinstance(source_overrides, dict):
        raw = _deep_merge(raw, source_overrides)

    output_path = event_spec.storage.processed_gnss_dir / f"{event_spec.event_id}_pipeline.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False, allow_unicode=True)
    return output_path


def resolve_workspace_root(event_spec: EventSpec) -> Path:
    return event_spec.internal_gnss_workspace_root()


def _fetch_internal_pipeline_assets(event_spec: EventSpec, pipeline_config) -> list[SourceAsset]:
    from ..internal.gnss_core.models import DownloadRecord
    from ..internal.gnss_core.sources import build_cddis_obs_url, make_adapters

    event = pipeline_config.events[0]
    observations_root = event_spec.storage.cache_root / "gnss_raw" / "observations"
    navigation_root = event_spec.storage.cache_root / "gnss_raw" / "navigation"
    auxiliary_root = event_spec.storage.cache_root / "gnss_aux"
    manifests_dir = event_spec.storage.manifests_dir

    http_session = create_retry_session(total=int(pipeline_config.download.get("max_retries", 4)))
    cddis_session = _build_cddis_session(event_spec, pipeline_config)

    primary_sources = {
        name: settings
        for name, settings in pipeline_config.observation_sources.items()
        if name in {"noaa", "rbmc", "ramsac"} and settings.enabled
    }
    noaa_settings = pipeline_config.observation_sources.get("noaa")
    base_nav_url = str(noaa_settings.params["base_obs_url"]) if noaa_settings else ""
    adapters = make_adapters(primary_sources, pipeline_config.bbox, base_nav_url)

    primary_records: list[DownloadRecord] = []
    for adapter in adapters:
        try:
            primary_records.extend(adapter.discover(event))
        except Exception as exc:
            LOGGER.exception("GNSS discovery failed for %s: %s", adapter.settings.name, exc)

    cddis_settings = pipeline_config.observation_sources.get("cddis")
    explicit_cddis_records = _build_explicit_cddis_records(event, cddis_settings, base_nav_url)
    candidate_rows = _candidate_rows_from_records(primary_records, explicit_cddis_records)
    write_gnss_station_candidates(manifests_dir / "gnss_station_candidates.csv", candidate_rows)

    observation_assets: list[GnssDownloadAsset] = []
    navigation_assets: list[GnssDownloadAsset] = []
    auxiliary_assets: list[GnssDownloadAsset] = []
    failure_assets: list[GnssDownloadAsset] = []
    selected_records: dict[tuple[str, str], DownloadRecord] = {}
    final_records: dict[tuple[str, str], DownloadRecord] = {}

    grouped_primary: dict[tuple[str, str], list[DownloadRecord]] = defaultdict(list)
    for record in primary_records:
        grouped_primary[(record.observation_date, record.station_code4.upper())].append(record)
    grouped_explicit_cddis = {
        (record.observation_date, record.station_code4.upper()): record for record in explicit_cddis_records
    }
    all_keys = sorted(set(grouped_primary) | set(grouped_explicit_cddis))

    for key in all_keys:
        primary_options = sorted(grouped_primary.get(key, []), key=lambda item: item.source_priority)
        chosen_asset: GnssDownloadAsset | None = None
        chosen_record: DownloadRecord | None = None

        for record in primary_options:
            settings = pipeline_config.observation_sources[record.source]
            asset = _download_observation_record(
                event_spec,
                pipeline_config,
                record,
                settings,
                observations_root,
                http_session=http_session,
                cddis_session=cddis_session,
            )
            observation_assets.append(asset)
            if asset.status == "ok":
                chosen_asset = asset
                chosen_record = record
                break
            failure_assets.append(asset)

        if chosen_asset is None and cddis_settings and cddis_settings.enabled:
            fallback_record = grouped_explicit_cddis.get(key)
            if fallback_record is None and primary_options:
                primary_seed = primary_options[0]
                fallback_record = DownloadRecord(
                    event_id=event.event_id,
                    source="cddis",
                    source_priority=cddis_settings.priority,
                    observation_date=primary_seed.observation_date,
                    station_id=primary_seed.station_id,
                    station_code4=primary_seed.station_code4,
                    sampling_sec=primary_seed.sampling_sec,
                    obs_url=build_cddis_obs_url(
                        date.fromisoformat(primary_seed.observation_date),
                        primary_seed.station_code4,
                        str(cddis_settings.params["obs_url_template"]),
                    ),
                    nav_url="",
                    lat=primary_seed.lat,
                    lon=primary_seed.lon,
                    height_m=primary_seed.height_m,
                )
            if fallback_record is not None:
                fallback_asset = _download_observation_record(
                    event_spec,
                    pipeline_config,
                    fallback_record,
                    cddis_settings,
                    observations_root,
                    http_session=http_session,
                    cddis_session=cddis_session,
                )
                observation_assets.append(fallback_asset)
                if fallback_asset.status == "ok":
                    chosen_asset = fallback_asset
                    chosen_record = fallback_record
                else:
                    failure_assets.append(fallback_asset)

        if chosen_asset is not None and chosen_record is not None:
            selected_records[key] = chosen_record
            final_records[key] = chosen_record
        elif primary_options:
            final_records[key] = primary_options[0]
        elif key in grouped_explicit_cddis:
            final_records[key] = grouped_explicit_cddis[key]

    navigation_by_date: dict[str, GnssDownloadAsset] = {}
    for day in event_spec.event_days():
        nav_asset, nav_failures = _download_daily_aux_product(
            event_spec=event_spec,
            pipeline_config=pipeline_config,
            product_type="broadcast",
            current_day=day.date(),
            cache_root=navigation_root,
            station_id="",
            station_code4="",
            http_session=http_session,
            cddis_session=cddis_session,
        )
        navigation_assets.append(nav_asset)
        failure_assets.extend(nav_failures)
        navigation_by_date[nav_asset.observation_date] = nav_asset

    for product_type in ("sp3", "dcb"):
        if product_type not in pipeline_config.auxiliary_sources:
            continue
        for day in event_spec.event_days():
            aux_asset, aux_failures = _download_daily_aux_product(
                event_spec=event_spec,
                pipeline_config=pipeline_config,
                product_type=product_type,
                current_day=day.date(),
                cache_root=auxiliary_root,
                station_id="",
                station_code4="",
                http_session=http_session,
                cddis_session=cddis_session,
            )
            auxiliary_assets.append(aux_asset)
            failure_assets.extend(aux_failures)

    if "antex" in pipeline_config.auxiliary_sources:
        first_day = event_spec.event_days()[0].date()
        antex_asset, antex_failures = _download_daily_aux_product(
            event_spec=event_spec,
            pipeline_config=pipeline_config,
            product_type="antex",
            current_day=first_day,
            cache_root=auxiliary_root,
            station_id="",
            station_code4="",
            http_session=http_session,
            cddis_session=cddis_session,
        )
        auxiliary_assets.append(antex_asset)
        failure_assets.extend(antex_failures)

    if "station_logs" in pipeline_config.auxiliary_sources:
        station_first_dates: dict[str, str] = {}
        station_ids: dict[str, str] = {}
        for key, record in selected_records.items():
            station_code4 = record.station_code4.upper()
            station_first_dates.setdefault(station_code4, record.observation_date)
            station_ids.setdefault(station_code4, record.station_id)
        for station_code4 in sorted(station_first_dates):
            station_log_asset, station_log_failures = _download_daily_aux_product(
                event_spec=event_spec,
                pipeline_config=pipeline_config,
                product_type="station_logs",
                current_day=date.fromisoformat(station_first_dates[station_code4]),
                cache_root=auxiliary_root,
                station_id=station_ids.get(station_code4, station_code4),
                station_code4=station_code4,
                http_session=http_session,
                cddis_session=cddis_session,
                allow_warning=True,
            )
            auxiliary_assets.append(station_log_asset)
            failure_assets.extend(station_log_failures)

    write_gnss_download_assets(manifests_dir / "gnss_observation_assets.csv", observation_assets)
    write_gnss_download_assets(manifests_dir / "gnss_navigation_assets.csv", navigation_assets)
    write_gnss_download_assets(manifests_dir / "gnss_aux_assets.csv", auxiliary_assets)
    write_gnss_download_assets(manifests_dir / "gnss_download_failures.csv", failure_assets)

    _write_legacy_observation_manifest(
        event_spec,
        selected_records=selected_records,
        fallback_records=final_records,
        navigation_by_date=navigation_by_date,
        observation_assets=observation_assets,
    )
    _write_legacy_aux_manifest(event_spec, auxiliary_assets)

    assets = _source_assets_from_workspace(event_spec, event_spec.storage.gnss_workspace_dir)
    if not assets:
        assets = [
            SourceAsset(
                event_id=event_spec.event_id,
                source_kind="gnss_raw",
                provider=asset.provider,
                asset_id=file_asset_id(asset.local_path or Path(asset.url)),
                local_path=asset.local_path or Path(asset.url),
                status=asset.status,
                metadata={
                    "station_id": asset.station_id,
                    "station_code4": asset.station_code4,
                    "observation_date": asset.observation_date,
                },
            )
            for asset in observation_assets
            if asset.status == "ok" and asset.local_path is not None
        ]

    missing_nav_days = [asset.observation_date for asset in navigation_assets if asset.status != "ok"]
    if missing_nav_days:
        raise FileNotFoundError(f"Broadcast navigation download failed for event days: {', '.join(missing_nav_days)}")
    if not any(asset.status == "ok" for asset in observation_assets):
        raise FileNotFoundError("No GNSS observation files were downloaded successfully.")
    return assets


def _source_assets_from_workspace(event_spec: EventSpec, workspace_root: Path) -> list[SourceAsset]:
    manifest_candidates = [
        workspace_root / "manifests" / "normalized_manifest.csv",
        workspace_root / "manifests" / "observation_manifest.csv",
    ]
    manifest_path = next((path for path in manifest_candidates if path.exists()), None)
    if manifest_path is None:
        return []

    frame = pd.read_csv(manifest_path)
    if "event_id" in frame.columns:
        frame = frame[frame["event_id"].astype(str) == event_spec.event_id].copy()

    assets: list[SourceAsset] = []
    for row in frame.to_dict("records"):
        obs_path_raw = row.get("obs_path") or row.get("local_path")
        local_path = manifest_path
        status = "indexed"
        if obs_path_raw:
            obs_path = Path(str(obs_path_raw))
            if obs_path.exists():
                local_path = obs_path
                status = "ready"
        asset_key = f"{row.get('station_id', '')}-{row.get('observation_date', '')}-{manifest_path.name}"
        assets.append(
            SourceAsset(
                event_id=event_spec.event_id,
                source_kind="gnss_raw",
                provider=str(row.get("source", "internal_snapshot")),
                asset_id=file_asset_id(local_path.with_name(asset_key)),
                local_path=local_path,
                status=status,
                metadata={
                    "station_id": str(row.get("station_id", "")),
                    "station_code4": str(row.get("station_code4", "")),
                    "observation_date": str(row.get("observation_date", "")),
                },
            )
        )
    return assets


def _build_explicit_cddis_records(event, cddis_settings, base_nav_url: str):
    from ..internal.gnss_core.sources import make_adapters

    if cddis_settings is None or not cddis_settings.enabled:
        return []
    adapters = make_adapters({"cddis": cddis_settings}, {}, base_nav_url)
    if not adapters:
        return []
    try:
        return adapters[0].discover(event)
    except Exception as exc:
        LOGGER.exception("Explicit CDDIS discovery failed: %s", exc)
        return []


def _candidate_rows_from_records(primary_records, explicit_cddis_records) -> list[GnssStationCandidate]:
    rows: list[GnssStationCandidate] = []
    for record in list(primary_records) + list(explicit_cddis_records):
        status = "candidate" if record.source != "cddis" else "supplemental"
        rows.append(
            GnssStationCandidate(
                event_id=record.event_id,
                provider=record.source,
                station_id=record.station_id,
                station_code4=record.station_code4.upper(),
                observation_date=record.observation_date,
                sampling_sec=int(record.sampling_sec or 0),
                lat=float(record.lat) if record.lat else None,
                lon=float(record.lon) if record.lon else None,
                height_m=float(record.height_m) if record.height_m else None,
                obs_url=record.obs_url,
                nav_url=record.nav_url,
                status=status,
                metadata={"source_priority": record.source_priority},
            )
        )
    rows.sort(key=lambda item: (item.observation_date, item.station_code4, item.provider))
    return rows


def _build_cddis_session(event_spec: EventSpec, pipeline_config) -> requests.Session:
    username = str(event_spec.auth.get("cddis", {}).get("username", "") or "")
    password = str(event_spec.auth.get("cddis", {}).get("password", "") or "")
    if not username or not password:
        return create_retry_session(total=int(pipeline_config.download.get("max_retries", 4)))
    verify = True
    cddis_settings = pipeline_config.observation_sources.get("cddis")
    if cddis_settings is not None:
        verify = bool(cddis_settings.params.get("verify_ssl", True))
    return create_earthdata_session(
        username=username,
        password=password,
        total=int(pipeline_config.download.get("max_retries", 4)),
        verify=verify,
    )


def _download_observation_record(
    event_spec: EventSpec,
    pipeline_config,
    record,
    settings,
    observations_root: Path,
    *,
    http_session: requests.Session,
    cddis_session: requests.Session,
) -> GnssDownloadAsset:
    observation_date = str(record.observation_date)
    provider = str(record.source)
    protocol = infer_protocol(record.obs_url, str(settings.params.get("transport", "auto")))
    target = observations_root / event_spec.event_id / observation_date / provider / Path(record.obs_url).name
    verify = bool(settings.params.get("verify_ssl", True))
    session = cddis_session if provider == "cddis" else http_session
    auth_ref = "cddis" if provider == "cddis" else None
    urls = [record.obs_url]
    fallback_base = settings.params.get("fallback_base_obs_url")
    primary_base = settings.params.get("base_obs_url")
    if provider == "noaa" and primary_base and fallback_base and str(record.obs_url).startswith(str(primary_base)):
        urls.append(str(record.obs_url).replace(str(primary_base).rstrip("/"), str(fallback_base).rstrip("/"), 1))

    last_error: str | None = None
    total_attempts = 0
    for url in urls:
        result = download_to_path(
            url,
            target,
            transport=protocol,
            session=session,
            timeout=int(settings.timeout_sec),
            verify=verify,
            max_retries=int(pipeline_config.download.get("max_retries", 4)),
            temp_suffix=str(pipeline_config.download.get("temp_suffix", ".part")),
        )
        total_attempts += result.attempts
        if result.status == "ok":
            return GnssDownloadAsset(
                event_id=event_spec.event_id,
                source_kind="gnss_observation",
                provider=provider,
                protocol=result.protocol,
                station_id=record.station_id,
                station_code4=record.station_code4.upper(),
                observation_date=observation_date,
                url=url,
                local_path=result.path,
                status="ok",
                attempts=total_attempts,
                auth_ref=auth_ref,
                metadata={"sampling_sec": int(record.sampling_sec or 0)},
            )
        last_error = result.error

    return GnssDownloadAsset(
        event_id=event_spec.event_id,
        source_kind="gnss_observation",
        provider=provider,
        protocol=protocol,
        station_id=record.station_id,
        station_code4=record.station_code4.upper(),
        observation_date=observation_date,
        url=urls[-1],
        local_path=None,
        status="error",
        attempts=total_attempts,
        error=last_error,
        auth_ref=auth_ref,
        metadata={"sampling_sec": int(record.sampling_sec or 0)},
    )


def _download_daily_aux_product(
    *,
    event_spec: EventSpec,
    pipeline_config,
    product_type: str,
    current_day: date,
    cache_root: Path,
    station_id: str,
    station_code4: str,
    http_session: requests.Session,
    cddis_session: requests.Session,
    allow_warning: bool = False,
) -> tuple[GnssDownloadAsset, list[GnssDownloadAsset]]:
    from ..internal.gnss_core.utils import date_to_doy, gps_week_and_dow

    settings = pipeline_config.auxiliary_sources.get(product_type)
    if settings is None or not settings.enabled:
        return (
            GnssDownloadAsset(
                event_id=event_spec.event_id,
                source_kind="gnss_aux" if product_type != "broadcast" else "gnss_navigation",
                provider="disabled",
                protocol="https",
                station_id=station_id,
                station_code4=station_code4,
                observation_date=current_day.isoformat(),
                url="",
                local_path=None,
                status="disabled",
            ),
            [],
        )

    _, doy_str, yy = date_to_doy(current_day)
    gps_week, dow = gps_week_and_dow(current_day)
    providers = sorted(settings.params.get("providers", []), key=lambda item: int(item.get("priority", settings.priority)))
    failures: list[GnssDownloadAsset] = []
    source_kind = "gnss_navigation" if product_type == "broadcast" else "gnss_aux"

    for provider in providers:
        priority = int(provider.get("priority", settings.priority))
        verify = bool(provider.get("verify_ssl", True))
        auth_ref = provider.get("auth")
        url = str(provider["url_template"]).format(
            yyyy=current_day.year,
            year=current_day.year,
            ddd=doy_str,
            doy=doy_str,
            yy=yy,
            gps_week=gps_week,
            dow=dow,
            station_id=station_id.lower(),
            station_code4=station_code4.lower(),
        )
        target = _aux_target_path(
            cache_root=cache_root,
            product_type=product_type,
            event_id=event_spec.event_id,
            current_day=current_day,
            station_code4=station_code4,
            filename=Path(url).name,
        )
        session = cddis_session if auth_ref == "cddis" else http_session
        result = download_to_path(
            url,
            target,
            transport=str(provider.get("transport", "auto")),
            session=session,
            timeout=int(settings.timeout_sec),
            verify=verify,
            max_retries=int(pipeline_config.download.get("aux_retries", pipeline_config.download.get("max_retries", 4))),
            temp_suffix=str(pipeline_config.download.get("temp_suffix", ".part")),
        )
        if result.status == "ok":
            return (
                GnssDownloadAsset(
                    event_id=event_spec.event_id,
                    source_kind=source_kind,
                    provider=str(provider["name"]),
                    protocol=result.protocol,
                    station_id=station_id,
                    station_code4=station_code4,
                    observation_date=current_day.isoformat(),
                    url=url,
                    local_path=result.path,
                    status="ok",
                    attempts=result.attempts,
                    auth_ref=str(auth_ref) if auth_ref else None,
                    metadata={
                        "product_type": product_type,
                        "priority": priority,
                        "verify_ssl": verify,
                    },
                ),
                failures,
            )
        failures.append(
            GnssDownloadAsset(
                event_id=event_spec.event_id,
                source_kind=source_kind,
                provider=str(provider["name"]),
                protocol=result.protocol,
                station_id=station_id,
                station_code4=station_code4,
                observation_date=current_day.isoformat(),
                url=url,
                local_path=None,
                status="warning" if allow_warning else "error",
                attempts=result.attempts,
                error=result.error,
                auth_ref=str(auth_ref) if auth_ref else None,
                metadata={
                    "product_type": product_type,
                    "priority": priority,
                    "verify_ssl": verify,
                },
            )
        )

    final_status = "warning" if allow_warning else "error"
    unresolved = GnssDownloadAsset(
        event_id=event_spec.event_id,
        source_kind=source_kind,
        provider="unresolved",
        protocol="https",
        station_id=station_id,
        station_code4=station_code4,
        observation_date=current_day.isoformat(),
        url="",
        local_path=None,
        status=final_status,
        attempts=0,
        error=f"No provider succeeded for {product_type}",
        metadata={"product_type": product_type},
    )
    return unresolved, failures


def _aux_target_path(
    *,
    cache_root: Path,
    product_type: str,
    event_id: str,
    current_day: date,
    station_code4: str,
    filename: str,
) -> Path:
    year = f"{current_day.year}"
    day_key = current_day.isoformat()
    if product_type == "broadcast":
        return cache_root / event_id / day_key / filename
    if product_type in {"sp3", "dcb"}:
        return cache_root / product_type / year / filename
    if product_type == "antex":
        return cache_root / "antex" / filename
    if product_type == "station_logs":
        return cache_root / "station_logs" / station_code4.lower() / filename
    return cache_root / product_type / filename


def _rebuild_internal_workspace_manifests_from_cache(event_spec: EventSpec) -> None:
    candidates = _discover_cached_observation_candidates(event_spec)
    if not candidates:
        raise FileNotFoundError("No GNSS raw cache files were discovered for the configured event days.")

    selected_rows = _select_cached_observation_rows(event_spec, candidates)
    if not selected_rows:
        raise FileNotFoundError("No GNSS raw cache files were selected for processing.")

    workspace_manifests = event_spec.storage.gnss_workspace_dir / "manifests"
    workspace_manifests.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_rows).to_csv(workspace_manifests / "observation_manifest.csv", index=False)
    failure_rows = [row for row in selected_rows if row["obs_status"] != "ok" or row["nav_status"] != "ok"]
    pd.DataFrame(failure_rows).to_csv(workspace_manifests / "observation_failures.csv", index=False)

    candidate_models = [
        GnssStationCandidate(
            event_id=event_spec.event_id,
            provider=str(candidate["source"]),
            station_id=str(candidate["station_id"]),
            station_code4=str(candidate["station_code4"]),
            observation_date=str(candidate["observation_date"]),
            sampling_sec=int(candidate["sampling_sec"]),
            status="cached",
            metadata={"source_priority": int(candidate["source_priority"])},
        )
        for candidate in candidates
    ]
    write_gnss_station_candidates(event_spec.storage.manifests_dir / "gnss_station_candidates.csv", candidate_models)

    source_assets = _discover_cached_raw_source_assets(event_spec)
    write_source_assets(event_spec.storage.manifests_dir / "gnss_raw_assets.csv", source_assets)

    observation_assets = [
        GnssDownloadAsset(
            event_id=event_spec.event_id,
            source_kind="gnss_observation",
            provider=str(row["source"]),
            protocol="file",
            station_id=str(row["station_id"]),
            station_code4=str(row["station_code4"]),
            observation_date=str(row["observation_date"]),
            url="",
            local_path=Path(str(row["obs_path"])) if str(row["obs_path"]).strip() else None,
            status=str(row["obs_status"]),
            error=str(row["obs_error"]) if str(row["obs_error"]).strip() else None,
            metadata={"source_priority": int(row["source_priority"])},
        )
        for row in selected_rows
    ]
    navigation_assets = _discover_cached_navigation_assets(event_spec, selected_rows)
    auxiliary_assets = _discover_cached_aux_assets(
        event_spec,
        {str(row["station_code4"]).upper() for row in selected_rows},
    )
    write_gnss_download_assets(event_spec.storage.manifests_dir / "gnss_observation_assets.csv", observation_assets)
    write_gnss_download_assets(event_spec.storage.manifests_dir / "gnss_navigation_assets.csv", navigation_assets)
    write_gnss_download_assets(event_spec.storage.manifests_dir / "gnss_aux_assets.csv", auxiliary_assets)
    _write_legacy_aux_manifest(event_spec, auxiliary_assets)


def _discover_cached_raw_source_assets(event_spec: EventSpec) -> list[SourceAsset]:
    candidates = _discover_cached_observation_candidates(event_spec)
    rows = _select_cached_observation_rows(event_spec, candidates)
    assets: list[SourceAsset] = []
    for row in rows:
        obs_path = Path(str(row["obs_path"]))
        if not obs_path.exists():
            continue
        assets.append(
            SourceAsset(
                event_id=event_spec.event_id,
                source_kind="gnss_raw",
                provider=str(row["source"]),
                asset_id=file_asset_id(obs_path),
                local_path=obs_path,
                status="ready",
                metadata={
                    "station_id": str(row["station_id"]),
                    "station_code4": str(row["station_code4"]),
                    "observation_date": str(row["observation_date"]),
                },
            )
        )
    return assets


def _discover_cached_observation_candidates(event_spec: EventSpec) -> list[dict[str, object]]:
    obs_root = event_spec.storage.cache_root / "gnss_raw" / "observations" / event_spec.event_id
    if not obs_root.exists():
        return []

    station_id_overrides = {
        preset.station_code.upper(): (preset.station_id or preset.station_code.upper())
        for preset in event_spec.station_series_presets()
    }
    provider_cfg = event_spec.sources.get("gnss_raw", {}).get("providers", {})
    candidates: list[dict[str, object]] = []

    for day in event_spec.event_days():
        day_key = day.strftime("%Y-%m-%d")
        day_root = obs_root / day_key
        if not day_root.exists():
            continue
        for provider_dir in sorted(path for path in day_root.iterdir() if path.is_dir()):
            provider = provider_dir.name.lower()
            priority = int(provider_cfg.get(provider, {}).get("priority", 999))
            default_sampling = int(provider_cfg.get(provider, {}).get("requested_interval_sec", 30) or 30)
            for path in sorted(file_path for file_path in provider_dir.iterdir() if file_path.is_file()):
                parsed = _parse_cached_observation_path(path, default_sampling, station_id_overrides)
                if parsed is None:
                    continue
                parsed["event_id"] = event_spec.event_id
                parsed["source"] = provider
                parsed["source_priority"] = priority
                parsed["observation_date"] = day_key
                candidates.append(parsed)
    return candidates


def _select_cached_observation_rows(event_spec: EventSpec, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    navigation_by_date = _discover_navigation_paths(event_spec)
    selected: dict[tuple[str, str], dict[str, object]] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: (
            str(item["observation_date"]),
            str(item["station_code4"]),
            int(item["source_priority"]),
            str(item["source"]),
            str(item["obs_path"]),
        ),
    ):
        key = (str(candidate["observation_date"]), str(candidate["station_code4"]))
        selected.setdefault(key, candidate)

    rows: list[dict[str, object]] = []
    for (_, _), candidate in sorted(selected.items()):
        day_key = str(candidate["observation_date"])
        nav_path = navigation_by_date.get(day_key)
        row = {
            "event_id": event_spec.event_id,
            "source": str(candidate["source"]),
            "source_priority": int(candidate["source_priority"]),
            "observation_date": day_key,
            "station_id": str(candidate["station_id"]),
            "station_code4": str(candidate["station_code4"]),
            "sampling_sec": int(candidate["sampling_sec"]),
            "obs_url": "",
            "nav_url": "",
            "lat": 0.0,
            "lon": 0.0,
            "height_m": 0.0,
            "obs_path": str(candidate["obs_path"]),
            "obs_status": "ok" if Path(str(candidate["obs_path"])).exists() else "error",
            "obs_error": "" if Path(str(candidate["obs_path"])).exists() else "observation_file_missing",
            "nav_path": str(nav_path) if nav_path is not None else "",
            "nav_status": "ok" if nav_path is not None and nav_path.exists() else "error",
            "nav_error": "" if nav_path is not None and nav_path.exists() else "navigation_file_missing",
        }
        rows.append(row)
    return rows


def _parse_cached_observation_path(
    path: Path,
    default_sampling: int,
    station_id_overrides: dict[str, str],
) -> dict[str, object] | None:
    match = LONG_RINEX_OBS_PATTERN.match(path.name)
    if match is not None:
        station_id = match.group("station_id").upper()
        station_code4 = station_id[:4]
        sampling_sec = int(match.group("sampling"))
    else:
        short_match = SHORT_RINEX_OBS_PATTERN.match(path.name)
        if short_match is None:
            return None
        station_code4 = short_match.group("station_code4").upper()
        station_id = station_id_overrides.get(station_code4, station_code4)
        sampling_sec = default_sampling

    station_id = station_id_overrides.get(station_code4, station_id)
    return {
        "station_id": station_id,
        "station_code4": station_code4,
        "sampling_sec": sampling_sec,
        "obs_path": path,
    }


def _discover_navigation_paths(event_spec: EventSpec) -> dict[str, Path]:
    nav_root = event_spec.storage.cache_root / "gnss_raw" / "navigation" / event_spec.event_id
    paths: dict[str, Path] = {}
    if not nav_root.exists():
        return paths
    for day in event_spec.event_days():
        day_key = day.strftime("%Y-%m-%d")
        day_dir = nav_root / day_key
        if not day_dir.exists():
            continue
        candidates = sorted(path for path in day_dir.iterdir() if path.is_file())
        if candidates:
            paths[day_key] = candidates[0]
    return paths


def _discover_cached_navigation_assets(
    event_spec: EventSpec,
    selected_rows: list[dict[str, object]],
) -> list[GnssDownloadAsset]:
    assets: list[GnssDownloadAsset] = []
    seen_dates: set[str] = set()
    for row in selected_rows:
        day_key = str(row["observation_date"])
        if day_key in seen_dates:
            continue
        seen_dates.add(day_key)
        nav_path_raw = str(row["nav_path"])
        nav_path = Path(nav_path_raw) if nav_path_raw else None
        assets.append(
            GnssDownloadAsset(
                event_id=event_spec.event_id,
                source_kind="gnss_navigation",
                provider="cache",
                protocol="file",
                station_id="",
                station_code4="",
                observation_date=day_key,
                url="",
                local_path=nav_path,
                status="ok" if nav_path is not None and nav_path.exists() else "error",
                error=None if nav_path is not None and nav_path.exists() else "navigation_file_missing",
            )
        )
    return assets


def _discover_cached_aux_assets(event_spec: EventSpec, station_codes: set[str]) -> list[GnssDownloadAsset]:
    aux_root = event_spec.storage.cache_root / "gnss_aux"
    assets: list[GnssDownloadAsset] = []

    for day in event_spec.event_days():
        year = day.strftime("%Y")
        doy = day.strftime("%j")
        day_key = day.strftime("%Y-%m-%d")
        for product_type in ("sp3", "dcb"):
            product_root = aux_root / product_type / year
            if not product_root.exists():
                continue
            for path in sorted(product_root.glob(f"*{year}{doy}*")):
                assets.append(
                    GnssDownloadAsset(
                        event_id=event_spec.event_id,
                        source_kind="gnss_aux",
                        provider="cache",
                        protocol="file",
                        station_id="",
                        station_code4="",
                        observation_date=day_key,
                        url="",
                        local_path=path,
                        status="ok",
                        metadata={"product_type": product_type},
                    )
                )

    for path in sorted((aux_root / "antex").glob("*")):
        if path.is_file():
            assets.append(
                GnssDownloadAsset(
                    event_id=event_spec.event_id,
                    source_kind="gnss_aux",
                    provider="cache",
                    protocol="file",
                    station_id="",
                    station_code4="",
                    observation_date=event_spec.event_days()[0].strftime("%Y-%m-%d"),
                    url="",
                    local_path=path,
                    status="ok",
                    metadata={"product_type": "antex"},
                )
            )

    for station_code in sorted(station_codes):
        station_root = aux_root / "station_logs" / station_code.lower()
        if not station_root.exists():
            continue
        for path in sorted(station_root.glob("*")):
            if not path.is_file():
                continue
            assets.append(
                GnssDownloadAsset(
                    event_id=event_spec.event_id,
                    source_kind="gnss_aux",
                    provider="cache",
                    protocol="file",
                    station_id=station_code,
                    station_code4=station_code,
                    observation_date=event_spec.event_days()[0].strftime("%Y-%m-%d"),
                    url="",
                    local_path=path,
                    status="ok",
                    metadata={"product_type": "station_logs"},
                )
            )
    return assets


def _watch_internal_netcdf_outputs(
    event_id: str,
    workspace_netcdf_dir: Path,
    staged_netcdf_dir: Path,
    stop_event: threading.Event,
    poll_seconds: float = 10.0,
) -> None:
    while not stop_event.wait(poll_seconds):
        _sync_internal_netcdf_outputs(event_id, workspace_netcdf_dir, staged_netcdf_dir, require_stable=True)


def _sync_internal_netcdf_outputs(
    event_id: str,
    workspace_netcdf_dir: Path,
    staged_netcdf_dir: Path,
    *,
    require_stable: bool,
) -> None:
    if not workspace_netcdf_dir.exists():
        return
    staged_netcdf_dir.mkdir(parents=True, exist_ok=True)
    for path in glob_event_netcdf(workspace_netcdf_dir, event_id):
        if require_stable:
            try:
                first_stat = path.stat()
                time.sleep(0.2)
                second_stat = path.stat()
            except FileNotFoundError:
                continue
            if first_stat.st_size == 0 or first_stat.st_size != second_stat.st_size or first_stat.st_mtime_ns != second_stat.st_mtime_ns:
                continue
        target = staged_netcdf_dir / path.name
        shutil.copy2(path, target)


def _write_legacy_observation_manifest(
    event_spec: EventSpec,
    *,
    selected_records: dict[tuple[str, str], Any],
    fallback_records: dict[tuple[str, str], Any],
    navigation_by_date: dict[str, GnssDownloadAsset],
    observation_assets: list[GnssDownloadAsset],
) -> None:
    workspace_manifests = event_spec.storage.gnss_workspace_dir / "manifests"
    workspace_manifests.mkdir(parents=True, exist_ok=True)

    observation_by_key: dict[tuple[str, str], GnssDownloadAsset] = {}
    for asset in observation_assets:
        key = (asset.observation_date, asset.station_code4.upper())
        if asset.status == "ok":
            observation_by_key[key] = asset
        elif key not in observation_by_key:
            observation_by_key[key] = asset

    rows: list[dict[str, object]] = []
    for key in sorted(fallback_records):
        record = selected_records.get(key, fallback_records[key])
        obs_asset = observation_by_key.get(key)
        nav_asset = navigation_by_date.get(record.observation_date)
        rows.append(
            {
                "event_id": event_spec.event_id,
                "source": record.source,
                "source_priority": int(record.source_priority),
                "observation_date": record.observation_date,
                "station_id": record.station_id,
                "station_code4": record.station_code4,
                "sampling_sec": int(record.sampling_sec or 0),
                "obs_url": obs_asset.url if obs_asset else record.obs_url,
                "nav_url": nav_asset.url if nav_asset else "",
                "lat": float(record.lat) if record.lat else 0.0,
                "lon": float(record.lon) if record.lon else 0.0,
                "height_m": float(record.height_m) if record.height_m else 0.0,
                "obs_path": str(obs_asset.local_path) if obs_asset and obs_asset.local_path else "",
                "obs_status": obs_asset.status if obs_asset else "error",
                "obs_error": obs_asset.error if obs_asset else "missing",
                "nav_path": str(nav_asset.local_path) if nav_asset and nav_asset.local_path else "",
                "nav_status": nav_asset.status if nav_asset else "error",
                "nav_error": nav_asset.error if nav_asset else "missing",
            }
        )
    pd.DataFrame(rows).to_csv(workspace_manifests / "observation_manifest.csv", index=False)
    failure_rows = [row for row in rows if row["obs_status"] != "ok" or row["nav_status"] != "ok"]
    pd.DataFrame(failure_rows).to_csv(workspace_manifests / "observation_failures.csv", index=False)


def _write_legacy_aux_manifest(event_spec: EventSpec, auxiliary_assets: list[GnssDownloadAsset]) -> None:
    workspace_manifests = event_spec.storage.gnss_workspace_dir / "manifests"
    workspace_manifests.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for asset in auxiliary_assets:
        product_type = str(asset.metadata.get("product_type", ""))
        row = {
            "observation_date": asset.observation_date,
            "product_type": product_type,
            "provider": asset.provider,
            "priority": int(asset.metadata.get("priority", 0) or 0),
            "url": asset.url,
            "auth_ref": asset.auth_ref or "",
            "verify_ssl": bool(asset.metadata.get("verify_ssl", True)),
            "path": str(asset.local_path) if asset.local_path else "",
            "status": asset.status,
            "error": asset.error or "",
            "attempts": asset.attempts,
            "metadata": asset.metadata,
        }
        rows.append(row)
        if asset.status not in {"ok", "disabled"}:
            failures.append(row)
    pd.DataFrame(rows).to_csv(workspace_manifests / "aux_manifest.csv", index=False)
    pd.DataFrame(failures).to_csv(workspace_manifests / "aux_failures.csv", index=False)


def _load_internal_pipeline_modules(include_processing: bool = False):
    from ..internal.gnss_core import config as config_module
    from ..internal.gnss_core import download as download_module

    if not include_processing:
        return config_module, download_module

    from ..internal.gnss_core import preprocess as preprocess_module
    from ..internal.gnss_core import processing_v2 as processing_module

    return config_module, download_module, preprocess_module, processing_module


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
