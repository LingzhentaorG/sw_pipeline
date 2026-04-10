from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import georinex
import numpy as np
import pandas as pd
from pyproj import Transformer

from .config import PipelineConfig
from .constants import EARTH_RADIUS_KM, GPS_EPOCH_ISO, GPS_L1_HZ, GPS_L2_HZ, LIGHT_SPEED_MPS
from .preprocess import preprocess_records
from .utils import build_event_time_index, load_dataframe, write_dataframe


LOGGER = logging.getLogger(__name__)
GPS_EPOCH_NS = pd.Timestamp(GPS_EPOCH_ISO).value
GEODETIC_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
LINK_COLUMNS = [
    "time",
    "event_id",
    "station_id",
    "sv",
    "elev_deg",
    "az_deg",
    "ipp_lat",
    "ipp_lon",
    "stec",
    "vtec",
    "rot",
    "roti",
]
GRID_COLUMNS = ["time", "event_id", "lat_bin", "lon_bin", "vtec_median", "roti_median", "sample_count"]
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
    "code_pair_available",
    "mw_used",
    "gap_break_count",
    "mw_break_count",
    "gf_break_count",
    "dropped_epoch_count",
    "input_epoch_count",
    "accepted_epoch_count",
]


def _emit_processing_progress(
    event_id: str,
    completed: int,
    total: int,
    next_progress_mark: int,
    progress_log_interval: int,
) -> int:
    if total <= 0:
        return next_progress_mark
    if completed >= total:
        if next_progress_mark <= total:
            LOGGER.info(
                "Processing progress for %s: %s/%s station-days completed",
                event_id,
                total,
                total,
            )
            return total + progress_log_interval
        return next_progress_mark
    while completed >= next_progress_mark:
        LOGGER.info(
            "Processing progress for %s: %s/%s station-days completed",
            event_id,
            min(completed, total),
            total,
        )
        next_progress_mark += progress_log_interval
    return next_progress_mark


@dataclass
class SlipDetectionResult:
    keep_mask: np.ndarray
    arc_ids: np.ndarray
    gap_break_count: int
    mw_break_count: int
    gf_break_count: int
    dropped_epoch_count: int
    code_pair_available: bool
    mw_used: bool


@dataclass
class StationProcessResult:
    data: pd.DataFrame
    diagnostics: list[dict[str, object]]
    reason: str | None = None
    detail: str | None = None


class GPSBroadcastStore:
    def __init__(self, nav_path: str | Path, max_age_hours: float) -> None:
        self.max_age_seconds = max_age_hours * 3600.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            nav = georinex.load(nav_path, use="G", fast=True)
        self.by_sv: dict[str, pd.DataFrame] = {}
        for sv in nav.sv.values:
            if not str(sv).startswith("G"):
                continue
            df = nav.sel(sv=sv).to_dataframe().reset_index()
            if "sqrtA" not in df.columns:
                continue
            df = df.dropna(subset=["sqrtA"]).sort_values("time")
            if not df.empty:
                self.by_sv[str(sv)] = df

    def position_ecef(self, sv: str, times: pd.DatetimeIndex) -> np.ndarray:
        eph = self.by_sv.get(sv)
        if eph is None or eph.empty:
            return np.full((len(times), 3), np.nan)

        times_index = pd.DatetimeIndex(times)
        eph_times = pd.DatetimeIndex(pd.to_datetime(eph["time"], utc=True))
        obs_ns = times_index.asi8
        eph_ns = eph_times.asi8
        indices = np.searchsorted(eph_ns, obs_ns, side="right") - 1
        valid = indices >= 0
        coords = np.full((len(times_index), 3), np.nan, dtype=float)
        if not valid.any():
            return coords

        selected_idx = np.clip(indices[valid], 0, len(eph) - 1)
        selected = eph.iloc[selected_idx].reset_index(drop=True)
        age_seconds = (obs_ns[valid] - eph_ns[selected_idx]) / 1e9
        young_enough = np.abs(age_seconds) <= self.max_age_seconds
        if not young_enough.any():
            return coords

        positions = _compute_gps_positions(selected, times_index[valid])
        coords[np.where(valid)[0][young_enough]] = positions[young_enough]
        return coords


def execute_processing_stage(config: PipelineConfig) -> tuple[list[Path], list[Path]]:
    processed_manifest = config.outputs.manifests_dir / "processed_manifest.csv"
    if not processed_manifest.exists():
        preprocess_records(config)
    manifest = load_dataframe(processed_manifest)
    if manifest.empty:
        raise RuntimeError("Processed manifest is empty; nothing to process.")

    link_paths: list[Path] = []
    grid_paths: list[Path] = []
    failure_rows: list[dict[str, object]] = []
    slip_rows: list[dict[str, object]] = []
    for event in config.events:
        kept_mask = manifest["kept"]
        if kept_mask.dtype == object:
            kept_mask = kept_mask.astype(str).str.lower().eq("true")
        event_rows = manifest[(manifest["event_id"] == event.event_id) & kept_mask].copy()
        link_path = config.outputs.link_dir / f"{event.event_id}.parquet"
        grid_path = config.outputs.grid_dir / f"{event.event_id}.parquet"
        link_df, event_failures, event_slips = process_event_rows(config, event_rows, event)
        write_dataframe(link_df, link_path)
        grid_df = grid_event_links(link_df, config, event)
        write_dataframe(grid_df, grid_path)
        link_paths.append(link_path)
        grid_paths.append(grid_path)
        failure_rows.extend(event_failures)
        slip_rows.extend(event_slips)
        LOGGER.info("Processed event %s -> %s links, %s grid rows", event.event_id, len(link_df), len(grid_df))

    failures_path = config.outputs.manifests_dir / "process_failures.csv"
    write_dataframe(pd.DataFrame(failure_rows, columns=PROCESS_FAILURE_COLUMNS), failures_path)
    diagnostics_path = config.outputs.manifests_dir / "slip_diagnostics.csv"
    write_dataframe(pd.DataFrame(slip_rows, columns=SLIP_DIAGNOSTIC_COLUMNS), diagnostics_path)
    LOGGER.info("Processing failures written to %s", failures_path)
    LOGGER.info("Slip diagnostics written to %s", diagnostics_path)
    return link_paths, grid_paths


def process_event_rows(
    config: PipelineConfig,
    event_rows: pd.DataFrame,
    event,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    if event_rows.empty:
        return _empty_link_df(), [], []

    rows: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    slip_rows: list[dict[str, object]] = []
    records = event_rows.to_dict("records")
    max_workers = max(1, min(int(config.processing.get("workers", 8)), len(records)))
    progress_log_interval = max(1, int(config.processing.get("progress_log_interval", 100)))
    batches = _build_record_batches(records, max_workers * 3)
    LOGGER.info(
        "Processing event %s with %s station-days using %s worker processes across %s batches",
        event.event_id,
        len(records),
        max_workers,
        len(batches),
    )

    completed = 0
    next_progress_mark = progress_log_interval
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(_process_record_batch, batch_records, nav_path, dict(config.processing), event): {
                "nav_path": nav_path,
                "records": batch_records,
            }
            for nav_path, batch_records in batches
        }
        for future in as_completed(future_to_batch):
            batch_info = future_to_batch[future]
            try:
                batch_df, batch_failures, batch_slips, batch_count = future.result()
            except Exception as exc:
                batch_count = len(batch_info["records"])
                for record in batch_info["records"]:
                    failures.append(
                        _build_failure_row(
                            record,
                            "batch_processing_exception",
                            f"{batch_info['nav_path']}: {exc}",
                        )
                    )
                completed += batch_count
                next_progress_mark = _emit_processing_progress(
                    event.event_id,
                    completed,
                    len(records),
                    next_progress_mark,
                    progress_log_interval,
                )
                continue
            completed += batch_count
            next_progress_mark = _emit_processing_progress(
                event.event_id,
                completed,
                len(records),
                next_progress_mark,
                progress_log_interval,
            )
            failures.extend(batch_failures)
            slip_rows.extend(batch_slips)
            if not batch_df.empty:
                rows.append(batch_df)

    if not rows:
        return _empty_link_df(), failures, slip_rows
    return (
        pd.concat(rows, ignore_index=True).sort_values(["time", "station_id", "sv"]).reset_index(drop=True),
        failures,
        slip_rows,
    )


def process_station_day(
    record: dict[str, object],
    nav_store: GPSBroadcastStore,
    processing_config: dict[str, object],
    event,
) -> StationProcessResult:
    meas = [str(record["phase_l1"]), str(record["phase_l2"])]
    if record.get("code_l1") and record.get("code_l2"):
        meas.extend([str(record["code_l1"]), str(record["code_l2"])])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        obs = georinex.load(
            record["obs_path"],
            use=set(str(processing_config["gnss_system"])),
            meas=meas,
            tlim=(event.start_utc.replace(tzinfo=None), event.end_utc.replace(tzinfo=None)),
            fast=True,
        )
    obs_df = obs[meas].to_dataframe().reset_index()
    obs_df = obs_df[obs_df["sv"].astype(str).str.startswith("G")].copy()
    if obs_df.empty:
        return StationProcessResult(_empty_link_df(), [], "no_gps_rows", "No GPS observations remained after loading.")

    target_interval = int(processing_config["target_interval_sec"])
    elev_mask = float(processing_config["elevation_mask_deg"])
    shell_height_km = float(processing_config["shell_height_km"])
    gap_threshold = target_interval * float(processing_config["arc_gap_factor"])
    roti_points = int(float(processing_config["roti_window_minutes"]) * 60 / target_interval)

    station_rows: list[pd.DataFrame] = []
    diagnostics_rows: list[dict[str, object]] = []
    failure_counts: Counter[str] = Counter()
    rx_xyz = _geodetic_to_ecef(float(record["lat"]), float(record["lon"]), float(record["height_m"]))
    rx_lat = math.radians(float(record["lat"]))
    rx_lon = math.radians(float(record["lon"]))

    for sv, sv_df in obs_df.groupby("sv", sort=True):
        sv_df = sv_df.sort_values("time").copy()
        sv_df["time"] = pd.to_datetime(sv_df["time"], utc=True)
        sv_df = normalize_to_interval(sv_df, target_interval)
        if sv_df.empty:
            failure_counts["off_grid_or_duplicate_epochs"] += 1
            continue
        sv_df = sv_df.dropna(subset=[str(record["phase_l1"]), str(record["phase_l2"])])
        if len(sv_df) < roti_points:
            failure_counts["insufficient_phase_samples"] += 1
            continue

        positions = nav_store.position_ecef(str(sv), pd.DatetimeIndex(sv_df["time"]))
        if not np.isfinite(positions).any():
            failure_counts["navigation_solution_missing"] += 1
            continue
        sv_df[["sat_x", "sat_y", "sat_z"]] = positions
        sv_df = sv_df.dropna(subset=["sat_x", "sat_y", "sat_z"])
        if sv_df.empty:
            failure_counts["navigation_solution_missing"] += 1
            continue

        az_deg, elev_deg = az_el_from_ecef(
            rx_xyz=np.asarray(rx_xyz),
            sat_xyz=sv_df[["sat_x", "sat_y", "sat_z"]].to_numpy(),
            lat_rad=rx_lat,
            lon_rad=rx_lon,
        )
        sv_df["az_deg"] = az_deg
        sv_df["elev_deg"] = elev_deg
        sv_df = sv_df[sv_df["elev_deg"] >= elev_mask].copy()
        if len(sv_df) < roti_points:
            failure_counts["below_elevation_mask"] += 1
            continue

        l1_cycles = sv_df[str(record["phase_l1"])].to_numpy(dtype=float)
        l2_cycles = sv_df[str(record["phase_l2"])].to_numpy(dtype=float)
        phase_stec = phase_to_stec_tecu(l1_cycles, l2_cycles)
        gf_phase_m = geometry_free_phase_m(l1_cycles, l2_cycles)

        code_pair_available = bool(record.get("code_l1") and record.get("code_l2"))
        if code_pair_available:
            c1_m = sv_df[str(record["code_l1"])].to_numpy(dtype=float)
            c2_m = sv_df[str(record["code_l2"])].to_numpy(dtype=float)
            mw_cycles = melbourne_wubbena_cycles(l1_cycles, l2_cycles, c1_m, c2_m)
            code_stec = code_to_stec_tecu(c1_m, c2_m)
        else:
            mw_cycles = None
            code_stec = None

        detection = detect_cycle_slips(
            times=pd.DatetimeIndex(sv_df["time"]),
            gf_phase_m=gf_phase_m,
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
                "sv": sv,
                "code_pair_available": detection.code_pair_available,
                "mw_used": detection.mw_used,
                "gap_break_count": detection.gap_break_count,
                "mw_break_count": detection.mw_break_count,
                "gf_break_count": detection.gf_break_count,
                "dropped_epoch_count": detection.dropped_epoch_count,
                "input_epoch_count": int(len(sv_df)),
                "accepted_epoch_count": int(detection.keep_mask.sum()),
            }
        )

        kept_df = sv_df.loc[detection.keep_mask].copy()
        if len(kept_df) < roti_points:
            failure_counts["short_arcs_after_slip_detection"] += 1
            continue
        kept_df["arc_id"] = detection.arc_ids[detection.keep_mask]
        kept_phase_stec = phase_stec[detection.keep_mask]
        kept_df["stec"] = normalize_stec_by_arc(kept_phase_stec, kept_df["arc_id"])

        if code_stec is not None:
            kept_code_stec = code_stec[detection.keep_mask]
            fallback = _fallback_vtec_source(kept_df["stec"].to_numpy(dtype=float))
            kept_df["vtec_source"] = np.where(np.isfinite(kept_code_stec), kept_code_stec, fallback)
        else:
            kept_df["vtec_source"] = _fallback_vtec_source(kept_df["stec"].to_numpy(dtype=float))

        arc_frames = [
            compute_rot_roti(frame.copy(), roti_points)
            for _, frame in kept_df.groupby("arc_id", sort=True)
            if len(frame) >= roti_points
        ]
        if not arc_frames:
            failure_counts["no_valid_roti"] += 1
            continue

        kept_df = pd.concat(arc_frames, ignore_index=True)
        kept_df = kept_df.dropna(subset=["rot", "roti"])
        if kept_df.empty:
            failure_counts["no_valid_roti"] += 1
            continue

        mapping = mapping_function(kept_df["elev_deg"].to_numpy(dtype=float), shell_height_km)
        kept_df["vtec"] = kept_df["vtec_source"] / mapping
        ipp_lat, ipp_lon = compute_ipp(
            lat_rad=rx_lat,
            lon_rad=rx_lon,
            az_deg=kept_df["az_deg"].to_numpy(dtype=float),
            elev_deg=kept_df["elev_deg"].to_numpy(dtype=float),
            shell_height_km=shell_height_km,
        )
        kept_df["ipp_lat"] = ipp_lat
        kept_df["ipp_lon"] = ipp_lon
        station_rows.append(
            kept_df.assign(
                event_id=record["event_id"],
                station_id=record["station_id"],
                sv=sv,
            )[LINK_COLUMNS]
        )

    if not station_rows:
        reason, detail = _summarize_failure_counts(failure_counts)
        return StationProcessResult(_empty_link_df(), diagnostics_rows, reason, detail)
    return StationProcessResult(pd.concat(station_rows, ignore_index=True), diagnostics_rows)


def _process_record_batch(
    batch_records: list[dict[str, object]],
    nav_path: str,
    processing_config: dict[str, object],
    event,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]], int]:
    rows: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    slip_rows: list[dict[str, object]] = []
    try:
        nav_store = GPSBroadcastStore(nav_path, float(processing_config["max_ephemeris_age_hours"]))
    except Exception as exc:
        for record in batch_records:
            failures.append(_build_failure_row(record, "navigation_load_failed", str(exc)))
        return _empty_link_df(), failures, slip_rows, len(batch_records)

    for record in batch_records:
        try:
            result = process_station_day(record, nav_store, processing_config, event)
        except Exception as exc:
            failures.append(_build_failure_row(record, "station_processing_exception", str(exc)))
            continue
        slip_rows.extend(result.diagnostics)
        if result.data.empty:
            failures.append(_build_failure_row(record, result.reason or "no_valid_links", result.detail))
            continue
        rows.append(result.data)

    if not rows:
        return _empty_link_df(), failures, slip_rows, len(batch_records)
    return (
        pd.concat(rows, ignore_index=True).sort_values(["time", "station_id", "sv"]).reset_index(drop=True),
        failures,
        slip_rows,
        len(batch_records),
    )


def _build_record_batches(
    records: list[dict[str, object]],
    target_batch_count: int,
) -> list[tuple[str, list[dict[str, object]]]]:
    if not records:
        return []
    chunk_size = max(1, math.ceil(len(records) / max(1, target_batch_count)))
    by_nav_path: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_nav_path.setdefault(str(record["nav_path"]), []).append(record)

    batches: list[tuple[str, list[dict[str, object]]]] = []
    for nav_path, items in sorted(by_nav_path.items()):
        for index in range(0, len(items), chunk_size):
            batches.append((nav_path, items[index : index + chunk_size]))
    return batches


def normalize_to_interval(df: pd.DataFrame, target_interval: int) -> pd.DataFrame:
    filtered = df.copy()
    filtered["time"] = pd.to_datetime(filtered["time"], utc=True)
    filtered = filtered.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    unix_seconds = (pd.DatetimeIndex(filtered["time"]).asi8 // 10**9).astype(np.int64)
    mask = (unix_seconds % target_interval) == 0
    return filtered.loc[mask].reset_index(drop=True)


def phase_to_stec_tecu(l1_cycles: np.ndarray, l2_cycles: np.ndarray) -> np.ndarray:
    lam1 = LIGHT_SPEED_MPS / GPS_L1_HZ
    lam2 = LIGHT_SPEED_MPS / GPS_L2_HZ
    factor = (GPS_L1_HZ**2 * GPS_L2_HZ**2) / (40.3 * (GPS_L1_HZ**2 - GPS_L2_HZ**2)) / 1e16
    return (l1_cycles * lam1 - l2_cycles * lam2) * factor


def geometry_free_phase_m(l1_cycles: np.ndarray, l2_cycles: np.ndarray) -> np.ndarray:
    lam1 = LIGHT_SPEED_MPS / GPS_L1_HZ
    lam2 = LIGHT_SPEED_MPS / GPS_L2_HZ
    return l1_cycles * lam1 - l2_cycles * lam2


def melbourne_wubbena_cycles(
    l1_cycles: np.ndarray,
    l2_cycles: np.ndarray,
    c1_m: np.ndarray,
    c2_m: np.ndarray,
) -> np.ndarray:
    wide_lane_wavelength_m = LIGHT_SPEED_MPS / (GPS_L1_HZ - GPS_L2_HZ)
    narrow_lane_code_m = (GPS_L1_HZ * c1_m + GPS_L2_HZ * c2_m) / (GPS_L1_HZ + GPS_L2_HZ)
    wide_lane_phase_m = wide_lane_wavelength_m * (l1_cycles - l2_cycles)
    return (wide_lane_phase_m - narrow_lane_code_m) / wide_lane_wavelength_m


def code_to_stec_tecu(c1_m: np.ndarray, c2_m: np.ndarray) -> np.ndarray:
    factor = (GPS_L1_HZ**2 * GPS_L2_HZ**2) / (40.3 * (GPS_L1_HZ**2 - GPS_L2_HZ**2)) / 1e16
    return (c2_m - c1_m) * factor


def detect_cycle_slips(
    times: pd.DatetimeIndex,
    gf_phase_m: np.ndarray,
    mw_cycles: np.ndarray | None,
    gap_threshold_seconds: float,
    enable_gf: bool,
    mw_window_points: int,
    mw_slip_threshold_cycles: float,
    gf_window_points: int,
    gf_poly_degree: int,
    gf_residual_threshold_m: float,
    drop_detected_slip_epoch: bool,
) -> SlipDetectionResult:
    times_index = pd.DatetimeIndex(times)
    keep_mask = np.zeros(len(times_index), dtype=bool)
    arc_ids = np.full(len(times_index), -1, dtype=int)
    accepted_indices: list[int] = []
    current_arc_id = 0
    gap_break_count = 0
    mw_break_count = 0
    gf_break_count = 0
    dropped_epoch_count = 0
    mw_used = False
    code_pair_available = mw_cycles is not None
    mw_window_points = max(1, int(mw_window_points))
    gf_window_points = max(1, int(gf_window_points))
    gf_poly_degree = max(0, int(gf_poly_degree))

    for idx in range(len(times_index)):
        if idx > 0:
            dt_seconds = (times_index[idx] - times_index[idx - 1]).total_seconds()
            if dt_seconds > gap_threshold_seconds:
                gap_break_count += 1
                accepted_indices.clear()
                current_arc_id += 1

        mw_triggered = False
        if mw_cycles is not None and np.isfinite(mw_cycles[idx]):
            mw_history = [mw_cycles[j] for j in accepted_indices if np.isfinite(mw_cycles[j])]
            if len(mw_history) >= mw_window_points:
                mw_used = True
                reference = float(np.median(mw_history[-mw_window_points:]))
                if abs(float(mw_cycles[idx]) - reference) > mw_slip_threshold_cycles:
                    mw_triggered = True

        if mw_triggered:
            mw_break_count += 1
            if drop_detected_slip_epoch:
                dropped_epoch_count += 1
                accepted_indices.clear()
                current_arc_id += 1
                continue

        gf_triggered = False
        if enable_gf and len(accepted_indices) >= gf_window_points:
            window_indices = accepted_indices[-gf_window_points:]
            history_times = (times_index[window_indices].asi8 - times_index[window_indices[0]].value) / 1e9
            current_time = (times_index[idx].value - times_index[window_indices[0]].value) / 1e9
            history_values = gf_phase_m[window_indices]
            degree = min(gf_poly_degree, len(window_indices) - 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coefficients = np.polyfit(history_times, history_values, degree)
            predicted = float(np.polyval(coefficients, current_time))
            if abs(float(gf_phase_m[idx]) - predicted) > gf_residual_threshold_m:
                gf_triggered = True

        if gf_triggered:
            gf_break_count += 1
            if drop_detected_slip_epoch:
                dropped_epoch_count += 1
                accepted_indices.clear()
                current_arc_id += 1
                continue

        keep_mask[idx] = True
        arc_ids[idx] = current_arc_id
        accepted_indices.append(idx)

    return SlipDetectionResult(
        keep_mask=keep_mask,
        arc_ids=arc_ids,
        gap_break_count=gap_break_count,
        mw_break_count=mw_break_count,
        gf_break_count=gf_break_count,
        dropped_epoch_count=dropped_epoch_count,
        code_pair_available=code_pair_available,
        mw_used=mw_used,
    )


def compute_arc_ids(
    times: pd.Series,
    stec: pd.Series,
    gap_threshold_seconds: float,
    slip_threshold_tecu: float,
    geometry_free: pd.Series | None = None,
    geometry_free_threshold_tecu: float | None = None,
) -> pd.Series:
    dt = pd.to_datetime(times, utc=True).diff().dt.total_seconds().fillna(0)
    dtec = pd.Series(stec, dtype=float).diff().abs().fillna(0)
    breaks = (dt > gap_threshold_seconds) | (dtec > slip_threshold_tecu)
    if geometry_free is not None and geometry_free_threshold_tecu is not None:
        dgf = pd.Series(geometry_free, dtype=float).diff().abs().fillna(0)
        breaks = breaks | (dgf > geometry_free_threshold_tecu)
    return breaks.cumsum().astype(int)


def normalize_stec_by_arc(stec: np.ndarray, arc_ids: pd.Series) -> np.ndarray:
    values = pd.Series(stec, dtype=float)
    baseline = values.groupby(arc_ids).transform("first")
    return (values - baseline).to_numpy(dtype=float)


def _fallback_vtec_source(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    offset = np.nanpercentile(finite, 5)
    return values - offset


def compute_rot_roti(frame: pd.DataFrame, window_points: int) -> pd.DataFrame:
    frame = frame.sort_values("time").copy()
    dt_min = frame["time"].diff().dt.total_seconds() / 60.0
    frame["rot"] = frame["stec"].diff() / dt_min
    frame["roti"] = frame["rot"].rolling(window_points, min_periods=window_points).std(ddof=0)
    return frame


def mapping_function(elev_deg: np.ndarray, shell_height_km: float) -> np.ndarray:
    elev_rad = np.radians(elev_deg)
    ratio = (EARTH_RADIUS_KM * np.cos(elev_rad)) / (EARTH_RADIUS_KM + shell_height_km)
    return 1.0 / np.sqrt(1.0 - np.clip(ratio, -0.999999, 0.999999) ** 2)


def compute_ipp(
    lat_rad: float,
    lon_rad: float,
    az_deg: np.ndarray,
    elev_deg: np.ndarray,
    shell_height_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    elev_rad = np.radians(elev_deg)
    az_rad = np.radians(az_deg)
    z = np.pi / 2.0 - elev_rad
    zp = np.arcsin((EARTH_RADIUS_KM / (EARTH_RADIUS_KM + shell_height_km)) * np.sin(z))
    psi = z - zp
    ipp_lat = np.arcsin(np.sin(lat_rad) * np.cos(psi) + np.cos(lat_rad) * np.sin(psi) * np.cos(az_rad))
    ipp_lon = lon_rad + np.arcsin(np.sin(psi) * np.sin(az_rad) / np.clip(np.cos(ipp_lat), 1e-6, None))
    return np.degrees(ipp_lat), ((np.degrees(ipp_lon) + 180.0) % 360.0) - 180.0


def az_el_from_ecef(
    rx_xyz: np.ndarray,
    sat_xyz: np.ndarray,
    lat_rad: float,
    lon_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    diff = sat_xyz - rx_xyz
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    transform = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )
    enu = diff @ transform.T
    east = enu[:, 0]
    north = enu[:, 1]
    up = enu[:, 2]
    az = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0
    el = np.degrees(np.arctan2(up, np.sqrt(east**2 + north**2)))
    return az, el


def grid_event_links(link_df: pd.DataFrame, config: PipelineConfig, event) -> pd.DataFrame:
    cadence = int(config.grid["cadence_minutes"])
    full_times = build_event_time_index(event.start_utc, event.end_utc, cadence)
    if link_df.empty:
        return pd.DataFrame(
            {
                "time": full_times,
                "event_id": event.event_id,
                "lat_bin": np.nan,
                "lon_bin": np.nan,
                "vtec_median": np.nan,
                "roti_median": np.nan,
                "sample_count": 0,
            },
            columns=GRID_COLUMNS,
        )

    lat_step = float(config.grid["lat_step_deg"])
    lon_step = float(config.grid["lon_step_deg"])
    grid = link_df.copy()
    grid["time"] = pd.to_datetime(grid["time"], utc=True).dt.floor(f"{cadence}min")
    grid["lat_bin"] = np.floor(grid["ipp_lat"] / lat_step) * lat_step
    grid["lon_bin"] = np.floor(grid["ipp_lon"] / lon_step) * lon_step
    grouped = (
        grid.groupby(["time", "event_id", "lat_bin", "lon_bin"], as_index=False)
        .agg(
            vtec_median=("vtec", "median"),
            roti_median=("roti", "median"),
            sample_count=("sv", "count"),
        )
        .sort_values(["time", "lat_bin", "lon_bin"])
        .reset_index(drop=True)
    )

    missing_times = full_times.difference(pd.DatetimeIndex(grouped["time"]))
    if len(missing_times) > 0:
        placeholders = pd.DataFrame(
            {
                "time": missing_times,
                "event_id": event.event_id,
                "lat_bin": np.nan,
                "lon_bin": np.nan,
                "vtec_median": np.nan,
                "roti_median": np.nan,
                "sample_count": 0,
            },
            columns=GRID_COLUMNS,
        )
        placeholders = placeholders.astype(
            {
                "event_id": grouped["event_id"].dtype,
                "lat_bin": "float64",
                "lon_bin": "float64",
                "vtec_median": "float64",
                "roti_median": "float64",
                "sample_count": grouped["sample_count"].dtype,
            }
        )
        grouped = pd.concat([grouped, placeholders], ignore_index=True)

    return grouped.sort_values(["time", "lat_bin", "lon_bin"], na_position="last").reset_index(drop=True)


def _compute_gps_positions(eph: pd.DataFrame, times: pd.DatetimeIndex) -> np.ndarray:
    times_index = pd.DatetimeIndex(times)
    obs_ns = times_index.asi8
    weeks = eph["GPSWeek"].to_numpy(dtype=float)
    toe = eph["Toe"].to_numpy(dtype=float)
    toe_ns = GPS_EPOCH_NS + ((weeks * 7 * 24 * 3600 + toe) * 1e9).astype(np.int64)
    tk = (obs_ns - toe_ns) / 1e9
    tk = ((tk + 302400.0) % 604800.0) - 302400.0

    sqrt_a = eph["sqrtA"].to_numpy(dtype=float)
    ecc = eph["Eccentricity"].to_numpy(dtype=float)
    a = sqrt_a**2
    mu = 3.986005e14
    omega_e = 7.2921151467e-5
    n0 = np.sqrt(mu / a**3)
    n = n0 + eph["DeltaN"].to_numpy(dtype=float)
    mk = eph["M0"].to_numpy(dtype=float) + n * tk

    ek = mk.copy()
    for _ in range(8):
        ek = mk + ecc * np.sin(ek)

    vk = np.arctan2(np.sqrt(1 - ecc**2) * np.sin(ek), np.cos(ek) - ecc)
    phi_k = vk + eph["omega"].to_numpy(dtype=float)
    du = eph["Cuc"].to_numpy(dtype=float) * np.cos(2 * phi_k) + eph["Cus"].to_numpy(dtype=float) * np.sin(2 * phi_k)
    dr = eph["Crc"].to_numpy(dtype=float) * np.cos(2 * phi_k) + eph["Crs"].to_numpy(dtype=float) * np.sin(2 * phi_k)
    di = eph["Cic"].to_numpy(dtype=float) * np.cos(2 * phi_k) + eph["Cis"].to_numpy(dtype=float) * np.sin(2 * phi_k)
    u = phi_k + du
    r = a * (1 - ecc * np.cos(ek)) + dr
    i = eph["Io"].to_numpy(dtype=float) + di + eph["IDOT"].to_numpy(dtype=float) * tk
    omega = eph["Omega0"].to_numpy(dtype=float) + (eph["OmegaDot"].to_numpy(dtype=float) - omega_e) * tk - omega_e * toe
    x_orb = r * np.cos(u)
    y_orb = r * np.sin(u)
    x = x_orb * np.cos(omega) - y_orb * np.cos(i) * np.sin(omega)
    y = x_orb * np.sin(omega) + y_orb * np.cos(i) * np.cos(omega)
    z = y_orb * np.sin(i)
    return np.column_stack([x, y, z])


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, height_m: float) -> tuple[float, float, float]:
    x, y, z = GEODETIC_TO_ECEF.transform(lon_deg, lat_deg, height_m)
    return float(x), float(y), float(z)


def _empty_link_df() -> pd.DataFrame:
    return pd.DataFrame(columns=LINK_COLUMNS)


def _summarize_failure_counts(failure_counts: Counter[str]) -> tuple[str, str]:
    if not failure_counts:
        return "no_valid_links", "No link survived quality control."
    reason, count = failure_counts.most_common(1)[0]
    detail = ", ".join(f"{name}={value}" for name, value in sorted(failure_counts.items()))
    return reason, f"{detail}; dominant={reason}:{count}"


def _build_failure_row(record: dict[str, object], reason: str, detail: str | None) -> dict[str, object]:
    return {
        "event_id": record["event_id"],
        "observation_date": record["observation_date"],
        "source": record["source"],
        "station_id": record["station_id"],
        "stage": "process",
        "reason": reason,
        "detail": detail,
        "obs_path": record["obs_path"],
        "nav_path": record["nav_path"],
    }
