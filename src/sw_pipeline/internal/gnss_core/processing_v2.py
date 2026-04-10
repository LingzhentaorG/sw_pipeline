from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import georinex
import numpy as np
import pandas as pd
import xarray as xr

from .config import PipelineConfig
from .models import EventWindow
from .preprocess import preprocess_records
from .processing import (
    GPSBroadcastStore,
    az_el_from_ecef,
    code_to_stec_tecu,
    compute_ipp,
    detect_cycle_slips,
    geometry_free_phase_m,
    melbourne_wubbena_cycles,
    normalize_to_interval,
    phase_to_stec_tecu,
)
from .timeseries import build_timeseries_processing_config, finalize_satellite_frame
from .utils import build_event_time_index, ecef_to_geodetic, geodetic_to_ecef, write_dataframe, write_dataset


LOGGER = logging.getLogger(__name__)

PHASE_L1_FIELDS = ("L1C", "L1W", "L1P", "L1X", "L1S", "L1L", "L1M", "L1", "L1N")
PHASE_L2_FIELDS = ("L2W", "L2P", "L2C", "L2X", "L2S", "L2L", "L2M", "L2", "L2N")
CODE_L1_FIELDS = ("C1C", "C1W", "C1P", "C1X", "C1S", "C1L", "C1M", "P1", "C1")
CODE_L2_FIELDS = ("C2W", "C2P", "C2C", "C2X", "C2S", "C2L", "C2M", "P2", "C2")
MEASURE_FIELDS = tuple(dict.fromkeys((*PHASE_L1_FIELDS, *PHASE_L2_FIELDS, *CODE_L1_FIELDS, *CODE_L2_FIELDS)))

LINK_COLUMNS = [
    "time",
    "event_id",
    "station_id",
    "station_code4",
    "sv",
    "az_deg",
    "elev_deg",
    "ipp_lat",
    "ipp_lon",
    "stec",
    "vtec",
    "roti",
    "arc_id",
]
GRID_COLUMNS = ["time", "lat", "lon", "vtec", "roti", "sample_count"]
PROCESS_FAILURE_COLUMNS = [
    "event_id",
    "observation_date",
    "source",
    "station_id",
    "stage",
    "reason",
    "detail",
    "obs_path",
    "nav_path",
]
SLIP_DIAGNOSTIC_COLUMNS = [
    "event_id",
    "observation_date",
    "station_id",
    "sv",
    "phase_pair",
    "code_pair",
    "code_pair_available",
    "mw_used",
    "gap_break_count",
    "mw_break_count",
    "gf_break_count",
    "dropped_epoch_count",
    "input_epoch_count",
    "accepted_epoch_count",
]


@dataclass
class StationProcessResult:
    data: pd.DataFrame
    diagnostics: list[dict[str, object]]
    reason: str | None = None
    detail: str | None = None


def execute_processing_stage(config: PipelineConfig) -> list[Path]:
    manifest_path = config.outputs.manifests_dir / "normalized_manifest.csv"
    if not manifest_path.exists():
        preprocess_records(config)
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise RuntimeError("Normalized manifest is empty; no GNSS records are available for processing.")

    processing_config = build_timeseries_processing_config(config)
    processing_config["workers"] = int(config.processing.get("workers", 8))
    processing_config["priority_station_codes"] = tuple(str(code).upper() for code in config.processing.get("priority_station_codes", []))
    processing_config["max_station_days_per_event"] = int(config.processing.get("max_station_days_per_event", 0) or 0)
    processing_config["progress_log_interval"] = int(config.processing.get("progress_log_interval", 100) or 100)
    processing_config["progress_path"] = str(config.outputs.manifests_dir / "process_progress.csv")

    failure_rows: list[dict[str, object]] = []
    slip_rows: list[dict[str, object]] = []
    netcdf_outputs: list[Path] = []

    for event in config.events:
        event_rows = manifest[manifest["event_id"].astype(str) == event.event_id].copy()
        if event_rows.empty:
            raise RuntimeError(f"No normalized observations were found for event {event.event_id}.")

        LOGGER.info("Processing internal GNSS event %s with %s station-day rows", event.event_id, len(event_rows))
        link_df, event_failures, event_slips = _process_event_rows(event_rows, processing_config, event)
        failure_rows.extend(event_failures)
        slip_rows.extend(event_slips)

        if link_df.empty:
            raise RuntimeError(f"No valid internal GNSS links were produced for {event.event_id}.")

        _write_station_series_inputs(config, event.event_id, link_df)
        grid_df = _build_grid_frame(link_df, config, event)
        write_dataframe(grid_df, config.outputs.grid_dir / f"{event.event_id}.parquet")

        event_outputs = _write_event_netcdf_chunks(config, event, grid_df)
        netcdf_outputs.extend(event_outputs)
        LOGGER.info(
            "Wrote internal GNSS event %s: %s links, %s grid rows, %s hourly NetCDF files",
            event.event_id,
            len(link_df),
            len(grid_df),
            len(event_outputs),
        )

    write_dataframe(pd.DataFrame(failure_rows, columns=PROCESS_FAILURE_COLUMNS), config.outputs.manifests_dir / "process_failures.csv")
    write_dataframe(pd.DataFrame(slip_rows, columns=SLIP_DIAGNOSTIC_COLUMNS), config.outputs.manifests_dir / "slip_diagnostics.csv")
    return netcdf_outputs


def _process_event_rows(
    event_rows: pd.DataFrame,
    processing_config: dict[str, object],
    event: EventWindow,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    selected_rows = _select_event_rows(event_rows, processing_config)
    records = selected_rows.to_dict("records")
    if not records:
        return _empty_link_frame(), [], []
    if len(selected_rows) != len(event_rows):
        LOGGER.info(
            "Internal GNSS event %s sampled %s/%s station-day rows",
            event.event_id,
            len(selected_rows),
            len(event_rows),
        )

    max_workers = max(1, min(int(processing_config.get("workers", 8)), len(records)))
    max_records_per_batch = max(1, int(processing_config.get("checkpoint_chunk_size", 10) or 10))
    batches = _build_record_batches(records, max_workers * 3, max_records_per_batch)
    rows: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    progress_log_interval = max(1, int(processing_config.get("progress_log_interval", 100) or 100))
    progress_path = Path(str(processing_config.get("progress_path", ""))) if processing_config.get("progress_path") else None
    _write_processing_progress(progress_path, event.event_id, 0, len(records), len(rows))
    LOGGER.info(
        "Processing internal GNSS event %s with %s station-day rows using %s worker threads across %s batches",
        event.event_id,
        len(records),
        max_workers,
        len(batches),
    )

    completed = 0
    next_progress_mark = progress_log_interval
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_process_record_batch, batch_records, nav_path, processing_config, event): (nav_path, batch_records)
            for nav_path, batch_records in batches
        }
        for future in as_completed(future_map):
            nav_path, batch_records = future_map[future]
            try:
                batch_frame, batch_failures, batch_diagnostics = future.result()
            except Exception as exc:
                detail = f"{nav_path}: {exc}"
                for record in batch_records:
                    failures.append(_build_failure_row(record, "batch_processing_exception", detail))
                completed += len(batch_records)
                next_progress_mark = _emit_processing_progress(
                    event.event_id,
                    completed,
                    len(records),
                    next_progress_mark,
                    progress_log_interval,
                    progress_path,
                    len(rows),
                )
                continue
            completed += len(batch_records)
            next_progress_mark = _emit_processing_progress(
                event.event_id,
                completed,
                len(records),
                next_progress_mark,
                progress_log_interval,
                progress_path,
                len(rows) + (0 if batch_frame.empty else 1),
            )
            failures.extend(batch_failures)
            diagnostics.extend(batch_diagnostics)
            if not batch_frame.empty:
                rows.append(batch_frame)

    _write_processing_progress(progress_path, event.event_id, completed, len(records), len(rows))
    if not rows:
        return _empty_link_frame(), failures, diagnostics
    frame = pd.concat(rows, ignore_index=True).sort_values(["time", "station_id", "sv"]).reset_index(drop=True)
    return frame, failures, diagnostics


def _process_record_batch(
    batch_records: list[dict[str, object]],
    nav_path: str,
    processing_config: dict[str, object],
    event: EventWindow,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    failures: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    rows: list[pd.DataFrame] = []

    try:
        nav_store = GPSBroadcastStore(nav_path, float(processing_config["max_ephemeris_age_hours"]))
    except Exception as exc:
        detail = f"navigation file load failed: {exc}"
        for record in batch_records:
            failures.append(_build_failure_row(record, "navigation_load_failed", detail))
        return _empty_link_frame(), failures, diagnostics

    for record in batch_records:
        try:
            result = _process_station_day(record, nav_store, processing_config, event)
        except Exception as exc:
            failures.append(_build_failure_row(record, "station_processing_exception", str(exc)))
            continue
        diagnostics.extend(result.diagnostics)
        if result.data.empty:
            failures.append(_build_failure_row(record, result.reason or "no_valid_links", result.detail))
            continue
        rows.append(result.data)

    if not rows:
        return _empty_link_frame(), failures, diagnostics
    return pd.concat(rows, ignore_index=True), failures, diagnostics


def _process_station_day(
    record: dict[str, object],
    nav_store: GPSBroadcastStore,
    processing_config: dict[str, object],
    event: EventWindow,
) -> StationProcessResult:
    slice_start, slice_end = _record_time_window(record, event)
    if slice_start >= slice_end:
        return StationProcessResult(_empty_link_frame(), [], "out_of_window", "record does not overlap the event window")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        obs = georinex.load(
            record["obs_path"],
            use="G",
            meas=list(MEASURE_FIELDS),
            tlim=(slice_start.replace(tzinfo=None), slice_end.replace(tzinfo=None)),
            fast=True,
        )

    available = {str(name) for name in obs.data_vars}
    phase_pair = _select_measurement_pair(available, PHASE_L1_FIELDS, PHASE_L2_FIELDS)
    if phase_pair is None:
        return StationProcessResult(_empty_link_frame(), [], "missing_phase_pair", "no compatible dual-frequency phase pair")
    code_pair = _select_measurement_pair(available, CODE_L1_FIELDS, CODE_L2_FIELDS)

    fields = list(dict.fromkeys((*phase_pair, *(code_pair or ()))))
    obs_df = obs[fields].to_dataframe().reset_index()
    if obs_df.empty or "sv" not in obs_df.columns:
        return StationProcessResult(_empty_link_frame(), [], "empty_observations", "georinex returned no GPS observation rows")

    obs_df["time"] = pd.to_datetime(obs_df["time"], utc=True)
    obs_df = obs_df[obs_df["sv"].astype(str).str.upper().str.startswith(str(processing_config["gnss_system"]).upper())].copy()
    if obs_df.empty:
        return StationProcessResult(_empty_link_frame(), [], "empty_gps_rows", "no GPS rows remained after filtering")

    station_position = _resolve_station_position(record, obs.attrs)
    if station_position is None:
        return StationProcessResult(_empty_link_frame(), [], "missing_station_position", "station coordinates unavailable in manifest and RINEX header")
    station_lat, station_lon, station_height_m = station_position

    rx_xyz = np.asarray(geodetic_to_ecef(station_lat, station_lon, station_height_m), dtype=float)
    lat_rad = math.radians(station_lat)
    lon_rad = math.radians(station_lon)
    target_interval = int(processing_config["target_interval_sec"])
    elevation_mask = float(processing_config["elevation_mask_deg"])
    gap_threshold = target_interval * float(processing_config["arc_gap_factor"])

    station_rows: list[pd.DataFrame] = []
    diagnostics_rows: list[dict[str, object]] = []
    failure_counts: Counter[str] = Counter()

    for sv, sv_frame in obs_df.groupby("sv", sort=True):
        sv_id = str(sv).upper()
        sat_df = sv_frame.sort_values("time").copy()
        sat_df = normalize_to_interval(sat_df, target_interval)
        sat_df = sat_df.dropna(subset=list(phase_pair)).reset_index(drop=True)
        if len(sat_df) < 2:
            failure_counts["insufficient_phase_samples"] += 1
            continue

        positions = nav_store.position_ecef(sv_id, pd.DatetimeIndex(sat_df["time"]))
        sat_df[["sat_x", "sat_y", "sat_z"]] = positions
        sat_df = sat_df.dropna(subset=["sat_x", "sat_y", "sat_z"]).copy()
        if len(sat_df) < 2:
            failure_counts["missing_navigation_solution"] += 1
            continue

        az_deg, elev_deg = az_el_from_ecef(
            rx_xyz=rx_xyz,
            sat_xyz=sat_df[["sat_x", "sat_y", "sat_z"]].to_numpy(dtype=float),
            lat_rad=lat_rad,
            lon_rad=lon_rad,
        )
        sat_df["az_deg"] = az_deg
        sat_df["elev_deg"] = elev_deg
        sat_df = sat_df[sat_df["elev_deg"] >= elevation_mask].copy()
        if len(sat_df) < 2:
            failure_counts["below_elevation_mask"] += 1
            continue

        phase_l1, phase_l2 = phase_pair
        l1_cycles = sat_df[phase_l1].to_numpy(dtype=float)
        l2_cycles = sat_df[phase_l2].to_numpy(dtype=float)
        phase_stec = phase_to_stec_tecu(l1_cycles, l2_cycles)
        gf_phase = geometry_free_phase_m(l1_cycles, l2_cycles)

        code_stec: np.ndarray | None = None
        mw_cycles: np.ndarray | None = None
        if code_pair is not None:
            code_l1, code_l2 = code_pair
            c1 = sat_df[code_l1].to_numpy(dtype=float)
            c2 = sat_df[code_l2].to_numpy(dtype=float)
            code_stec = code_to_stec_tecu(c1, c2)
            mw_cycles = melbourne_wubbena_cycles(l1_cycles, l2_cycles, c1, c2)

        detection = detect_cycle_slips(
            times=pd.DatetimeIndex(sat_df["time"]),
            gf_phase_m=gf_phase,
            mw_cycles=mw_cycles if bool(processing_config["enable_mw"]) else None,
            gap_threshold_seconds=gap_threshold,
            enable_gf=bool(processing_config.get("enable_gf", True)),
            mw_window_points=int(processing_config["mw_window_points"]),
            mw_slip_threshold_cycles=float(processing_config["mw_slip_threshold_cycles"]),
            gf_window_points=int(processing_config["gf_window_points"]),
            gf_poly_degree=int(processing_config["gf_poly_degree"]),
            gf_residual_threshold_m=float(processing_config["gf_residual_threshold_m"]),
            drop_detected_slip_epoch=bool(processing_config["drop_detected_slip_epoch"]),
        )

        diagnostics_rows.append(
            {
                "event_id": record["event_id"],
                "observation_date": record["observation_date"],
                "station_id": record["station_id"],
                "sv": sv_id,
                "phase_pair": "/".join(phase_pair),
                "code_pair": "/".join(code_pair) if code_pair is not None else "",
                "code_pair_available": bool(code_pair),
                "mw_used": detection.mw_used,
                "gap_break_count": detection.gap_break_count,
                "mw_break_count": detection.mw_break_count,
                "gf_break_count": detection.gf_break_count,
                "dropped_epoch_count": detection.dropped_epoch_count,
                "input_epoch_count": int(len(sat_df)),
                "accepted_epoch_count": int(detection.keep_mask.sum()),
            }
        )

        accepted = finalize_satellite_frame(
            sat_df=sat_df,
            phase_stec=phase_stec,
            code_stec=code_stec,
            detection=detection,
            processing_config=processing_config,
        )
        if accepted.empty:
            failure_counts["finalize_empty"] += 1
            continue

        accepted = accepted[(accepted["vtec"].notna()) | (accepted["roti"].notna())].copy()
        if accepted.empty:
            failure_counts["no_finite_vtec_or_roti"] += 1
            continue

        ipp_lat, ipp_lon = compute_ipp(
            lat_rad=lat_rad,
            lon_rad=lon_rad,
            az_deg=accepted["az_deg"].to_numpy(dtype=float),
            elev_deg=accepted["elev_deg"].to_numpy(dtype=float),
            shell_height_km=float(processing_config["shell_height_km"]),
        )

        station_rows.append(
            accepted.assign(
                event_id=record["event_id"],
                station_id=str(record["station_id"]),
                station_code4=str(record["station_code4"]).upper(),
                sv=sv_id,
                ipp_lat=ipp_lat,
                ipp_lon=ipp_lon,
            )[LINK_COLUMNS]
        )

    if not station_rows:
        reason, detail = _summarize_failures(failure_counts)
        return StationProcessResult(_empty_link_frame(), diagnostics_rows, reason, detail)

    return StationProcessResult(pd.concat(station_rows, ignore_index=True), diagnostics_rows)


def _record_time_window(record: dict[str, object], event: EventWindow) -> tuple[datetime, datetime]:
    day_start = datetime.fromisoformat(f"{record['observation_date']}T00:00:00+00:00")
    day_end = day_start + timedelta(days=1)
    slice_start = max(pd.Timestamp(event.start_utc).to_pydatetime(), day_start)
    slice_end = min(pd.Timestamp(event.end_utc).to_pydatetime(), day_end)
    return slice_start, slice_end


def _resolve_station_position(record: dict[str, object], attrs: dict[str, object]) -> tuple[float, float, float] | None:
    lat = float(record.get("lat", 0.0) or 0.0)
    lon = float(record.get("lon", 0.0) or 0.0)
    height_m = float(record.get("height_m", 0.0) or 0.0)
    has_manifest_position = np.isfinite(lat) and np.isfinite(lon) and not (
        abs(lat) < 1e-9 and abs(lon) < 1e-9 and abs(height_m) < 1e-9
    )
    if has_manifest_position:
        return lat, lon, height_m

    position = attrs.get("position")
    if position is None or len(position) != 3:
        return None
    try:
        return ecef_to_geodetic(float(position[0]), float(position[1]), float(position[2]))
    except Exception:
        return None


def _select_measurement_pair(
    available: set[str],
    first_candidates: tuple[str, ...],
    second_candidates: tuple[str, ...],
) -> tuple[str, str] | None:
    for first in first_candidates:
        if first not in available:
            continue
        for second in second_candidates:
            if second in available:
                return first, second
    return None


def _build_record_batches(
    records: list[dict[str, object]],
    target_batch_count: int,
    max_records_per_batch: int,
) -> list[tuple[str, list[dict[str, object]]]]:
    if not records:
        return []
    chunk_size = max(1, math.ceil(len(records) / max(1, target_batch_count)))
    chunk_size = min(chunk_size, max(1, max_records_per_batch))
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["nav_path"]), []).append(record)

    batches: list[tuple[str, list[dict[str, object]]]] = []
    for nav_path, items in sorted(grouped.items()):
        for index in range(0, len(items), chunk_size):
            batches.append((nav_path, items[index : index + chunk_size]))
    return batches


def _select_event_rows(event_rows: pd.DataFrame, processing_config: dict[str, object]) -> pd.DataFrame:
    limit = int(processing_config.get("max_station_days_per_event", 0) or 0)
    if limit <= 0 or len(event_rows) <= limit:
        return event_rows.copy()

    priority_codes = {str(code).upper() for code in processing_config.get("priority_station_codes", ())}
    station_codes = event_rows["station_code4"].astype(str).str.upper()
    station_ids = event_rows["station_id"].astype(str).str.upper()
    priority_mask = station_codes.isin(priority_codes) | station_ids.isin(priority_codes)

    priority_rows = event_rows[priority_mask].copy()
    if len(priority_rows) >= limit:
        return priority_rows.sort_values(["observation_date", "station_code4", "station_id"]).head(limit).reset_index(drop=True)

    remaining = limit - len(priority_rows)
    supplemental = event_rows[~priority_mask].copy()
    lat_values = supplemental["lat"].astype(float)
    lon_values = supplemental["lon"].astype(float)
    supplemental["has_position"] = lat_values.notna() & lon_values.notna() & (
        ~(lat_values.abs() < 1e-9) | ~(lon_values.abs() < 1e-9)
    )
    if "source_priority" not in supplemental.columns:
        supplemental["source_priority"] = 0
    supplemental = supplemental.sort_values(
        ["observation_date", "has_position", "lat", "lon", "source_priority", "station_code4", "station_id"],
        ascending=[True, False, True, True, True, True, True],
    )
    supplemental = _downsample_evenly(supplemental, remaining)

    combined = pd.concat([priority_rows, supplemental], ignore_index=True)
    combined = combined.drop(columns=["has_position"], errors="ignore")
    return combined.sort_values(["observation_date", "station_code4", "station_id"]).reset_index(drop=True)


def _downsample_evenly(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    if len(frame) <= limit:
        return frame.copy()
    indices = np.linspace(0, len(frame) - 1, num=limit, dtype=int)
    return frame.iloc[np.unique(indices)].copy()


def _write_station_series_inputs(config: PipelineConfig, event_id: str, link_df: pd.DataFrame) -> None:
    vtec_frame = (
        link_df.loc[link_df["vtec"].notna(), ["time", "station_id", "station_code4", "sv", "vtec"]]
        .sort_values(["station_id", "sv", "time"])
        .reset_index(drop=True)
    )
    roti_frame = (
        link_df.loc[link_df["roti"].notna(), ["time", "station_id", "station_code4", "sv", "roti"]]
        .sort_values(["station_id", "sv", "time"])
        .reset_index(drop=True)
    )
    write_dataframe(vtec_frame, config.outputs.vtec_dir / f"{event_id}.parquet")
    write_dataframe(roti_frame, config.outputs.roti_dir / f"{event_id}.parquet")


def _build_grid_frame(link_df: pd.DataFrame, config: PipelineConfig, event: EventWindow) -> pd.DataFrame:
    if link_df.empty:
        return pd.DataFrame(columns=GRID_COLUMNS)

    bbox = config.bbox
    lat_step = float(config.gridding["lat_step_deg"])
    lon_step = float(config.gridding["lon_step_deg"])
    cadence = int(config.gridding["time_step_min"])

    grid = link_df.copy()
    grid["time"] = pd.to_datetime(grid["time"], utc=True).dt.floor(f"{cadence}min")
    grid = grid.dropna(subset=["ipp_lat", "ipp_lon"]).copy()
    grid = grid[
        (grid["ipp_lat"] >= bbox["lat_min"])
        & (grid["ipp_lat"] <= bbox["lat_max"])
        & (grid["ipp_lon"] >= bbox["lon_min"])
        & (grid["ipp_lon"] <= bbox["lon_max"])
    ].copy()
    if grid.empty:
        return pd.DataFrame(columns=GRID_COLUMNS)

    grid["lat"] = bbox["lat_min"] + np.round((grid["ipp_lat"] - bbox["lat_min"]) / lat_step) * lat_step
    grid["lon"] = bbox["lon_min"] + np.round((grid["ipp_lon"] - bbox["lon_min"]) / lon_step) * lon_step
    grid["lat"] = grid["lat"].astype(float).round(6)
    grid["lon"] = grid["lon"].astype(float).round(6)

    grouped = (
        grid.groupby(["time", "lat", "lon"], as_index=False)
        .agg(
            vtec=("vtec", "median"),
            roti=("roti", "median"),
            sample_count=("sv", "count"),
        )
        .sort_values(["time", "lat", "lon"])
        .reset_index(drop=True)
    )
    return grouped[GRID_COLUMNS]


def _build_event_dataset(
    grid_df: pd.DataFrame,
    config: PipelineConfig,
    event: EventWindow,
    window_start: datetime | pd.Timestamp | None = None,
    window_end: datetime | pd.Timestamp | None = None,
) -> xr.Dataset:
    cadence = int(config.gridding["time_step_min"])
    lat_step = float(config.gridding["lat_step_deg"])
    lon_step = float(config.gridding["lon_step_deg"])
    effective_start = pd.Timestamp(window_start or event.start_utc)
    effective_end = pd.Timestamp(window_end or event.end_utc)
    time_index = build_event_time_index(effective_start.to_pydatetime(), effective_end.to_pydatetime(), cadence)
    lat_coords = np.arange(config.bbox["lat_min"], config.bbox["lat_max"] + lat_step * 0.5, lat_step, dtype=float)
    lon_coords = np.arange(config.bbox["lon_min"], config.bbox["lon_max"] + lon_step * 0.5, lon_step, dtype=float)

    shape = (len(time_index), len(lat_coords), len(lon_coords))
    vtec = np.full(shape, np.nan, dtype=np.float32)
    roti = np.full(shape, np.nan, dtype=np.float32)
    sample_count = np.zeros(shape, dtype=np.int16)

    if not grid_df.empty:
        time_lookup = {pd.Timestamp(value): idx for idx, value in enumerate(time_index)}
        lat_lookup = {round(float(value), 6): idx for idx, value in enumerate(lat_coords)}
        lon_lookup = {round(float(value), 6): idx for idx, value in enumerate(lon_coords)}

        for row in grid_df.itertuples(index=False):
            time_value = pd.Timestamp(row.time)
            if time_value.tzinfo is None:
                time_value = time_value.tz_localize(UTC)
            else:
                time_value = time_value.tz_convert(UTC)
            time_idx = time_lookup.get(time_value)
            lat_idx = lat_lookup.get(round(float(row.lat), 6))
            lon_idx = lon_lookup.get(round(float(row.lon), 6))
            if time_idx is None or lat_idx is None or lon_idx is None:
                continue
            if pd.notna(row.vtec):
                vtec[time_idx, lat_idx, lon_idx] = float(row.vtec)
            if pd.notna(row.roti):
                roti[time_idx, lat_idx, lon_idx] = float(row.roti)
            sample_count[time_idx, lat_idx, lon_idx] = int(row.sample_count)

    dataset = xr.Dataset(
        data_vars={
            "vtec": (("time", "lat", "lon"), vtec),
            "roti": (("time", "lat", "lon"), roti),
            "sample_count": (("time", "lat", "lon"), sample_count),
        },
        coords={
            "time": time_index.tz_convert(None),
            "lat": lat_coords,
            "lon": lon_coords,
        },
        attrs={
            "event_id": event.event_id,
            "producer": "internal",
            "start_time": effective_start.isoformat(),
            "end_time": effective_end.isoformat(),
            "time_step_min": cadence,
            "lat_step_deg": lat_step,
            "lon_step_deg": lon_step,
        },
    )
    dataset["vtec"].attrs["units"] = "TECu"
    dataset["roti"].attrs["units"] = "TECu/min"
    dataset["sample_count"].attrs["units"] = "count"
    return dataset


def _write_event_netcdf_chunks(config: PipelineConfig, event: EventWindow, grid_df: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    if grid_df.empty:
        return outputs

    time_values = pd.to_datetime(grid_df["time"], utc=True)
    chunk_starts = time_values.dt.floor("1h")
    chunk_frame = grid_df.copy()
    chunk_frame["chunk_start"] = chunk_starts
    cadence = int(config.gridding["time_step_min"])
    cadence_delta = pd.Timedelta(minutes=cadence)

    for chunk_start, subset in chunk_frame.groupby("chunk_start", sort=True):
        subset = subset.drop(columns=["chunk_start"]).copy()
        chunk_end = min(chunk_start + pd.Timedelta(hours=1) - cadence_delta, pd.Timestamp(event.end_utc))
        dataset = _build_event_dataset(
            subset,
            config,
            event,
            window_start=chunk_start,
            window_end=chunk_end,
        )
        output_path = config.outputs.netcdf_dir / f"{event.event_id}_{chunk_start.strftime('%Y%m%d_%H00')}.nc"
        write_dataset(dataset, output_path)
        outputs.append(output_path)
    return outputs


def _empty_link_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=LINK_COLUMNS)


def _emit_processing_progress(
    event_id: str,
    completed: int,
    total: int,
    next_progress_mark: int,
    progress_log_interval: int,
    progress_path: Path | None,
    completed_batches: int,
) -> int:
    if completed >= min(total, next_progress_mark):
        LOGGER.info("Processing progress for %s: %s/%s station-days completed", event_id, completed, total)
        _write_processing_progress(progress_path, event_id, completed, total, completed_batches)
        return total + progress_log_interval if completed >= total else next_progress_mark + progress_log_interval
    return next_progress_mark


def _write_processing_progress(
    path: Path | None,
    event_id: str,
    completed: int,
    total: int,
    completed_batches: int,
) -> None:
    if path is None:
        return
    frame = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "completed_station_days": completed,
                "total_station_days": total,
                "completed_batches": completed_batches,
                "updated_at_utc": pd.Timestamp.now(tz=UTC).isoformat(),
            }
        ]
    )
    write_dataframe(frame, path)


def _summarize_failures(failure_counts: Counter[str]) -> tuple[str, str]:
    if not failure_counts:
        return "no_valid_links", "no satellite produced accepted VTEC or ROTI samples"
    reason, _ = failure_counts.most_common(1)[0]
    detail = ", ".join(f"{name}={value}" for name, value in sorted(failure_counts.items()))
    return reason, detail


def _build_failure_row(record: dict[str, object], reason: str, detail: str | None) -> dict[str, object]:
    return {
        "event_id": record["event_id"],
        "observation_date": record["observation_date"],
        "source": record["source"],
        "station_id": record["station_id"],
        "stage": "process",
        "reason": reason,
        "detail": detail or "",
        "obs_path": record["obs_path"],
        "nav_path": record["nav_path"],
    }
