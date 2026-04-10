from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Sequence

import georinex
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PipelineConfig, load_pipeline_config
from .models import EventWindow
from .processing import (
    GPSBroadcastStore,
    SlipDetectionResult,
    _fallback_vtec_source,
    _geodetic_to_ecef,
    az_el_from_ecef,
    code_to_stec_tecu,
    compute_rot_roti,
    detect_cycle_slips,
    geometry_free_phase_m,
    mapping_function,
    melbourne_wubbena_cycles,
    normalize_stec_by_arc,
    normalize_to_interval,
    phase_to_stec_tecu,
)
from .utils import ensure_directories, resolve_station_identifier, utc_date_range


matplotlib.use("Agg")

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/pipeline.local.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/custom/timeseries")
DEFAULT_ALL_SATELLITES_SUBDIR = Path("all_satellites")
DEFAULT_STATION_CODE4 = "BOAV"
DEFAULT_EVENT_WINDOWS = (
    EventWindow(
        event_id="event_20241010T230000_20241011T040000",
        start_utc=datetime(2024, 10, 10, 23, 0, 0, tzinfo=UTC),
        end_utc=datetime(2024, 10, 11, 4, 0, 0, tzinfo=UTC),
    ),
    EventWindow(
        event_id="event_20241231T230000_20250101T040000",
        start_utc=datetime(2024, 12, 31, 23, 0, 0, tzinfo=UTC),
        end_utc=datetime(2025, 1, 1, 4, 0, 0, tzinfo=UTC),
    ),
)
DEFAULT_EVENT_SATELLITE_SELECTIONS = {
    DEFAULT_EVENT_WINDOWS[0].event_id: ("G21", "G02"),
    DEFAULT_EVENT_WINDOWS[1].event_id: ("G17", "G14"),
}
DEFAULT_SATELLITE_COUNT = 2
DEFAULT_USE_CYCLE_SLIP_DETECTION = True
TIMESERIES_COLUMNS = [
    "time",
    "utc_hour",
    "station_id",
    "station_code4",
    "sv",
    "vtec",
    "roti",
    "arc_id",
]


@dataclass(frozen=True)
class TimeseriesEventOutput:
    event: EventWindow
    station_id: str
    station_code4: str
    satellites: tuple[str, ...]
    csv_path: Path
    png_path: Path


@dataclass(frozen=True)
class TimeseriesOutput:
    station_code4: str
    station_id: str
    output_dir: Path
    events: list[TimeseriesEventOutput]

    @property
    def csv_paths(self) -> list[Path]:
        return [item.csv_path for item in self.events]

    @property
    def png_paths(self) -> list[Path]:
        return [item.png_path for item in self.events]


@dataclass(frozen=True)
class SingleSatelliteProduct:
    satellite: str
    rank: int
    valid_vtec_count: int
    valid_roti_count: int
    longest_arc_length: int
    qualified: bool
    csv_path: Path
    png_path: Path


@dataclass(frozen=True)
class AllSatellitesEventOutput:
    event: EventWindow
    station_id: str
    station_code4: str
    event_dir: Path
    summary_csv_path: Path
    csv_dir: Path
    plot_dir: Path
    satellites: list[SingleSatelliteProduct]


@dataclass(frozen=True)
class AllSatellitesOutput:
    station_code4: str
    station_id: str
    output_dir: Path
    events: list[AllSatellitesEventOutput]


@dataclass(frozen=True)
class StationContext:
    station_id: str
    station_code4: str
    lat: float
    lon: float
    height_m: float
    phase_l1: str
    phase_l2: str
    code_l1: str | None
    code_l2: str | None
    records_by_date: dict[str, dict[str, object]]


@dataclass(frozen=True)
class SatelliteCandidate:
    satellite: str
    frame: pd.DataFrame
    valid_vtec_count: int
    valid_roti_count: int
    longest_arc_length: int

    @property
    def score(self) -> int:
        return self.valid_vtec_count + self.valid_roti_count

    @property
    def qualified(self) -> bool:
        return self.valid_vtec_count > 0 and self.valid_roti_count > 0


@dataclass(frozen=True)
class PreparedEventSeries:
    event: EventWindow
    station_id: str
    station_code4: str
    satellites: tuple[str, ...]
    frame: pd.DataFrame
    coverage: list[SatelliteCandidate]


def build_timeseries_processing_config(config: PipelineConfig) -> dict[str, object]:
    return {
        "gnss_system": str(config.processing["gnss_system"]),
        "target_interval_sec": int(config.processing["target_interval_sec"]),
        "elevation_mask_deg": float(config.processing["vtec"]["cutoff_elevation_deg"]),
        "shell_height_km": float(config.processing["shell_height_km"]),
        "arc_gap_factor": int(config.processing["arcs"]["max_gap_epochs"]) + 1,
        "roti_window_minutes": int(config.processing["roti"]["window_length_min"]),
        "enable_mw": bool(config.processing["cycle_slip"]["enable_mw"]),
        "enable_gf": bool(config.processing["cycle_slip"].get("enable_gf", True)),
        "mw_window_points": int(config.processing["cycle_slip"]["mw_window_points"]),
        "mw_slip_threshold_cycles": float(config.processing["cycle_slip"]["mw_slip_threshold_cycles"]),
        "gf_window_points": int(config.processing["cycle_slip"]["gf_window_points"]),
        "gf_poly_degree": int(config.processing["cycle_slip"]["gf_poly_degree"]),
        "gf_residual_threshold_m": float(config.processing["cycle_slip"]["gf_residual_threshold_m"]),
        "drop_detected_slip_epoch": bool(config.processing["cycle_slip"]["drop_detected_slip_epoch"]),
        "max_ephemeris_age_hours": float(config.processing["max_ephemeris_age_hours"]),
        "use_cycle_slip_detection": DEFAULT_USE_CYCLE_SLIP_DETECTION,
    }


def format_panel_title(satellite: str) -> str:
    return f"prn={satellite}"


def build_figure_title(
    event: EventWindow,
    station_code4: str,
    station_id: str,
    satellites: Sequence[str],
) -> str:
    satellites_text = ", ".join(satellites)
    return (
        f"Station {station_code4.upper()} ({station_id}) | "
        f"UTC {event.start_utc:%Y-%m-%d %H:%M} to {event.end_utc:%Y-%m-%d %H:%M} | "
        f"Satellites: {satellites_text}"
    )


def compute_utc_hour(
    time_values: pd.Series | pd.DatetimeIndex,
    event: EventWindow | str,
) -> pd.Series:
    window = coerce_event_window(event)
    anchor = pd.Timestamp(window.start_utc).tz_convert(UTC).normalize()
    values = pd.to_datetime(time_values, utc=True)
    delta = values - anchor
    if isinstance(delta, pd.Series):
        return delta.dt.total_seconds() / 3600.0
    return pd.Series(delta.total_seconds() / 3600.0)


def filter_panel_frame(
    frame: pd.DataFrame,
    event: EventWindow | str,
    satellites: Sequence[str] | None = None,
) -> pd.DataFrame:
    window = coerce_event_window(event)
    filtered = frame.copy()
    filtered["time"] = pd.to_datetime(filtered["time"], utc=True)
    filtered = filtered[
        (filtered["time"] >= pd.Timestamp(window.start_utc))
        & (filtered["time"] <= pd.Timestamp(window.end_utc))
    ].copy()
    if satellites is not None:
        filtered = filtered[filtered["sv"].isin(list(satellites))].copy()
    if "utc_hour" not in filtered.columns or filtered["utc_hour"].isna().any():
        filtered["utc_hour"] = compute_utc_hour(filtered["time"], window)
    return filtered.sort_values(["sv", "time"]).reset_index(drop=True)


def load_station_record(
    manifest: pd.DataFrame,
    observation_date: str,
    station_code4: str,
) -> dict[str, object]:
    candidates = resolve_station_identifier(station_code4)
    if not candidates:
        raise ValueError("station_code4 must not be empty")

    rows = manifest[manifest["observation_date"] == observation_date].copy()
    if rows.empty:
        raise ValueError(f"No normalized rows for observation_date={observation_date}")

    mask = rows["station_id"].astype(str).str.upper().isin(candidates) | rows["station_code4"].astype(str).str.upper().isin(candidates)
    rows = rows[mask].copy()
    if rows.empty:
        raise ValueError(f"No normalized row found for station={station_code4} date={observation_date}")

    rows["station_code4_upper"] = rows["station_code4"].astype(str).str.upper()
    rows = rows.sort_values(["station_code4_upper", "station_id"])
    return rows.iloc[0].to_dict()


def generate_satellite_timeseries_products(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    station_code4: str = DEFAULT_STATION_CODE4,
    outdir: str | Path | None = None,
    event_windows: Sequence[EventWindow] = DEFAULT_EVENT_WINDOWS,
    satellite_count: int = DEFAULT_SATELLITE_COUNT,
    event_satellite_selections: dict[str, Sequence[str]] | None = None,
) -> TimeseriesOutput:
    config = load_pipeline_config(config_path)
    output_dir = Path(outdir) if outdir is not None else DEFAULT_OUTPUT_DIR
    output_dir = output_dir.resolve()
    ensure_directories([output_dir])

    manifest_path = config.outputs.manifests_dir / "normalized_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    processing_config = build_timeseries_processing_config(config)

    prepared_events: list[PreparedEventSeries] = []
    for event in event_windows:
        selected_satellites = None
        if event_satellite_selections is not None:
            selected_satellites = event_satellite_selections.get(event.event_id)
        if selected_satellites is None:
            prepared_events.append(
                extract_station_event_timeseries(
                    manifest=manifest,
                    event=event,
                    station_code4=station_code4,
                    processing_config=processing_config,
                    satellite_count=satellite_count,
                )
            )
        else:
            prepared_events.append(
                extract_station_fixed_satellite_timeseries(
                    manifest=manifest,
                    event=event,
                    station_code4=station_code4,
                    processing_config=processing_config,
                    satellites=tuple(selected_satellites),
                )
            )

    outputs: list[TimeseriesEventOutput] = []
    for prepared in prepared_events:
        base_name = build_output_base_name(prepared.event, prepared.station_code4, prepared.satellites)
        csv_path = output_dir / f"{base_name}.csv"
        png_path = output_dir / f"{base_name}.png"
        prepared.frame.to_csv(csv_path, index=False)
        plot_daily_satellite_timeseries(
            frame=prepared.frame,
            event=prepared.event,
            output_path=png_path,
            station_code4=prepared.station_code4,
            station_id=prepared.station_id,
            satellites=prepared.satellites,
        )
        outputs.append(
            TimeseriesEventOutput(
                event=prepared.event,
                station_id=prepared.station_id,
                station_code4=prepared.station_code4,
                satellites=prepared.satellites,
                csv_path=csv_path,
                png_path=png_path,
            )
        )
        LOGGER.info("Wrote %s and %s", csv_path, png_path)

    station_id = outputs[0].station_id if outputs else ""
    return TimeseriesOutput(
        station_code4=station_code4.upper(),
        station_id=station_id,
        output_dir=output_dir,
        events=outputs,
    )


def generate_all_satellite_timeseries_products(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    station_code4: str = DEFAULT_STATION_CODE4,
    outdir: str | Path | None = None,
    event_windows: Sequence[EventWindow] = DEFAULT_EVENT_WINDOWS,
) -> AllSatellitesOutput:
    config = load_pipeline_config(config_path)
    base_output_dir = Path(outdir) if outdir is not None else DEFAULT_OUTPUT_DIR
    output_dir = (base_output_dir / DEFAULT_ALL_SATELLITES_SUBDIR / station_code4.upper()).resolve()
    ensure_directories([output_dir])

    manifest_path = config.outputs.manifests_dir / "normalized_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    processing_config = build_timeseries_processing_config(config)

    event_outputs: list[AllSatellitesEventOutput] = []
    station_id = ""
    for event in event_windows:
        prepared = prepare_all_satellite_event_series(
            manifest=manifest,
            event=event,
            station_code4=station_code4,
            processing_config=processing_config,
        )
        station_id = prepared.station_id
        event_dir = output_dir / build_event_directory_name(event)
        csv_dir = event_dir / "csv"
        plot_dir = event_dir / "plots"
        ensure_directories([event_dir, csv_dir, plot_dir])

        ranked_candidates = rank_satellite_candidates(
            [candidate for candidate in prepared.coverage if not candidate.frame.empty]
        )
        products: list[SingleSatelliteProduct] = []
        summary_rows: list[dict[str, object]] = []
        for index, candidate in enumerate(ranked_candidates, start=1):
            file_prefix = f"{index:02d}_{candidate.satellite.lower()}"
            csv_path = csv_dir / f"{file_prefix}.csv"
            png_path = plot_dir / f"{file_prefix}.png"
            candidate.frame.to_csv(csv_path, index=False)
            plot_daily_satellite_timeseries(
                frame=candidate.frame,
                event=event,
                output_path=png_path,
                station_code4=prepared.station_code4,
                station_id=prepared.station_id,
                satellites=(candidate.satellite,),
            )
            product = SingleSatelliteProduct(
                satellite=candidate.satellite,
                rank=index,
                valid_vtec_count=candidate.valid_vtec_count,
                valid_roti_count=candidate.valid_roti_count,
                longest_arc_length=candidate.longest_arc_length,
                qualified=candidate.qualified,
                csv_path=csv_path,
                png_path=png_path,
            )
            products.append(product)
            summary_rows.append(
                {
                    "rank": index,
                    "sv": candidate.satellite,
                    "qualified": candidate.qualified,
                    "valid_vtec_count": candidate.valid_vtec_count,
                    "valid_roti_count": candidate.valid_roti_count,
                    "score": candidate.score,
                    "longest_arc_length": candidate.longest_arc_length,
                    "csv_path": str(csv_path),
                    "png_path": str(png_path),
                }
            )

        skipped_candidates = rank_satellite_candidates(
            [candidate for candidate in prepared.coverage if candidate.frame.empty]
        )
        for candidate in skipped_candidates:
            summary_rows.append(
                {
                    "rank": np.nan,
                    "sv": candidate.satellite,
                    "qualified": candidate.qualified,
                    "valid_vtec_count": candidate.valid_vtec_count,
                    "valid_roti_count": candidate.valid_roti_count,
                    "score": candidate.score,
                    "longest_arc_length": candidate.longest_arc_length,
                    "csv_path": "",
                    "png_path": "",
                }
            )

        summary_csv_path = event_dir / "summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
        event_outputs.append(
            AllSatellitesEventOutput(
                event=event,
                station_id=prepared.station_id,
                station_code4=prepared.station_code4,
                event_dir=event_dir,
                summary_csv_path=summary_csv_path,
                csv_dir=csv_dir,
                plot_dir=plot_dir,
                satellites=products,
            )
        )
        LOGGER.info(
            "Wrote %s single-satellite products under %s",
            len(products),
            event_dir,
        )

    return AllSatellitesOutput(
        station_code4=station_code4.upper(),
        station_id=station_id,
        output_dir=output_dir,
        events=event_outputs,
    )


def extract_station_event_timeseries(
    manifest: pd.DataFrame,
    event: EventWindow,
    station_code4: str,
    processing_config: dict[str, object],
    satellite_count: int = DEFAULT_SATELLITE_COUNT,
) -> PreparedEventSeries:
    station = load_station_context(manifest, event, station_code4)
    nav_stores = {
        observation_date: GPSBroadcastStore(
            str(record["nav_path"]),
            float(processing_config["max_ephemeris_age_hours"]),
        )
        for observation_date, record in station.records_by_date.items()
    }
    obs_df = load_event_observations(station, event)
    if obs_df.empty:
        raise ValueError(
            f"No observation samples for station={station.station_code4} "
            f"window={event.start_utc.isoformat()} to {event.end_utc.isoformat()}"
        )

    candidate_satellites = sorted(obs_df["sv"].dropna().astype(str).unique().tolist())
    candidates: list[SatelliteCandidate] = []
    for satellite in candidate_satellites:
        frame = _extract_satellite_series(
            obs_df=obs_df,
            station=station,
            satellite=satellite,
            nav_stores=nav_stores,
            processing_config=processing_config,
        )
        candidates.append(build_satellite_candidate(frame, satellite))

    selected = select_top_satellite_candidates(candidates, satellite_count, event, station.station_code4)
    satellites = tuple(candidate.satellite for candidate in selected)
    frame = pd.concat([candidate.frame for candidate in selected], ignore_index=True)
    frame["utc_hour"] = compute_utc_hour(frame["time"], event)
    frame = filter_panel_frame(frame, event, satellites=satellites)
    return PreparedEventSeries(
        event=event,
        station_id=station.station_id,
        station_code4=station.station_code4,
        satellites=satellites,
        frame=frame[TIMESERIES_COLUMNS],
        coverage=candidates,
    )


def extract_station_fixed_satellite_timeseries(
    manifest: pd.DataFrame,
    event: EventWindow,
    station_code4: str,
    processing_config: dict[str, object],
    satellites: Sequence[str],
) -> PreparedEventSeries:
    if not satellites:
        raise ValueError("satellites must not be empty")
    station, nav_stores, obs_df = _prepare_event_inputs(
        manifest=manifest,
        event=event,
        station_code4=station_code4,
        processing_config=processing_config,
    )

    candidates: list[SatelliteCandidate] = []
    selected_frames: list[pd.DataFrame] = []
    for satellite in satellites:
        frame = _extract_satellite_series(
            obs_df=obs_df,
            station=station,
            satellite=str(satellite).upper(),
            nav_stores=nav_stores,
            processing_config=processing_config,
        )
        if not frame.empty:
            frame = filter_panel_frame(frame, event, satellites=(str(satellite).upper(),))
            selected_frames.append(frame)
        candidates.append(build_satellite_candidate(frame, str(satellite).upper()))

    missing = [candidate.satellite for candidate in candidates if candidate.frame.empty]
    if missing:
        raise ValueError(
            f"Station {station.station_code4} has no exportable data for satellites={','.join(missing)} "
            f"during {event.start_utc.isoformat()} to {event.end_utc.isoformat()}"
        )

    frame = pd.concat(selected_frames, ignore_index=True)
    frame = filter_panel_frame(frame, event, satellites=tuple(str(satellite).upper() for satellite in satellites))
    return PreparedEventSeries(
        event=event,
        station_id=station.station_id,
        station_code4=station.station_code4,
        satellites=tuple(str(satellite).upper() for satellite in satellites),
        frame=frame[TIMESERIES_COLUMNS],
        coverage=candidates,
    )


def prepare_all_satellite_event_series(
    manifest: pd.DataFrame,
    event: EventWindow,
    station_code4: str,
    processing_config: dict[str, object],
) -> PreparedEventSeries:
    station, nav_stores, obs_df = _prepare_event_inputs(
        manifest=manifest,
        event=event,
        station_code4=station_code4,
        processing_config=processing_config,
    )

    candidate_satellites = sorted(obs_df["sv"].dropna().astype(str).unique().tolist())
    candidates: list[SatelliteCandidate] = []
    for satellite in candidate_satellites:
        frame = _extract_satellite_series(
            obs_df=obs_df,
            station=station,
            satellite=satellite,
            nav_stores=nav_stores,
            processing_config=processing_config,
        )
        if not frame.empty:
            frame = filter_panel_frame(frame, event, satellites=(satellite,))
        candidates.append(build_satellite_candidate(frame, satellite))

    exported = [candidate for candidate in candidates if not candidate.frame.empty]
    if not exported:
        coverage_lines = [
            (
                f"{candidate.satellite}: score={candidate.score}, "
                f"vtec={candidate.valid_vtec_count}, "
                f"roti={candidate.valid_roti_count}, "
                f"longest_arc={candidate.longest_arc_length}"
            )
            for candidate in rank_satellite_candidates(candidates)[:8]
        ]
        coverage_text = "; ".join(coverage_lines) if coverage_lines else "no candidate satellites"
        raise ValueError(
            f"Station {station.station_code4} has no exportable satellites for "
            f"{event.start_utc.isoformat()} to {event.end_utc.isoformat()}. Coverage: {coverage_text}"
        )

    satellites = tuple(candidate.satellite for candidate in rank_satellite_candidates(exported))
    frame = pd.concat([candidate.frame for candidate in exported], ignore_index=True)
    frame = filter_panel_frame(frame, event, satellites=satellites)
    return PreparedEventSeries(
        event=event,
        station_id=station.station_id,
        station_code4=station.station_code4,
        satellites=satellites,
        frame=frame[TIMESERIES_COLUMNS],
        coverage=candidates,
    )


def _prepare_event_inputs(
    manifest: pd.DataFrame,
    event: EventWindow,
    station_code4: str,
    processing_config: dict[str, object],
) -> tuple[StationContext, dict[str, GPSBroadcastStore], pd.DataFrame]:
    station = load_station_context(manifest, event, station_code4)
    nav_stores = {
        observation_date: GPSBroadcastStore(
            str(record["nav_path"]),
            float(processing_config["max_ephemeris_age_hours"]),
        )
        for observation_date, record in station.records_by_date.items()
    }
    obs_df = load_event_observations(station, event)
    if obs_df.empty:
        raise ValueError(
            f"No observation samples for station={station.station_code4} "
            f"window={event.start_utc.isoformat()} to {event.end_utc.isoformat()}"
        )
    return station, nav_stores, obs_df


def plot_daily_satellite_timeseries(
    frame: pd.DataFrame,
    event: EventWindow | str,
    output_path: str | Path,
    station_code4: str,
    station_id: str,
    satellites: Sequence[str],
) -> Path:
    window = coerce_event_window(event)
    output = Path(output_path)
    ensure_directories([output.parent])

    frame = filter_panel_frame(frame, window, satellites=satellites)
    fig, axes = plt.subplots(2, len(satellites), figsize=(8.8, 7.2), dpi=180)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, 1)

    xmin, xmax = window_hour_bounds(window)
    xticks = build_utc_ticks(window)
    xtick_labels = [format_utc_tick_label(value) for value in xticks]

    fig.suptitle(
        build_figure_title(window, station_code4, station_id, satellites),
        fontsize=12,
        y=0.98,
    )

    for col, satellite in enumerate(satellites):
        sat_frame = frame[frame["sv"] == satellite].copy()

        tec_ax = axes[0, col]
        roti_ax = axes[1, col]

        tec_ax.set_title(format_panel_title(satellite), fontsize=12)
        tec_ax.set_xlim(xmin, xmax)
        tec_ax.set_xticks(xticks, xtick_labels)
        tec_ax.set_xlabel("UTC [h]")
        tec_ax.set_ylabel("TEC [TECU]")

        roti_ax.set_xlim(xmin, xmax)
        roti_ax.set_xticks(xticks, xtick_labels)
        roti_ax.set_xlabel("UTC [h]")
        roti_ax.set_ylabel("ROTI [TECU/min]")

        if sat_frame.empty:
            _draw_empty_axis(tec_ax, "No valid TEC")
            _draw_empty_axis(roti_ax, "No valid ROTI")
            continue

        tec_frame = sat_frame.dropna(subset=["vtec"]).copy()
        if tec_frame.empty:
            _draw_empty_axis(tec_ax, "No valid TEC")
        else:
            _plot_scatter_by_arc(tec_ax, tec_frame, "vtec", color="black", size=5.0)
            _set_axis_limits(tec_ax, tec_frame["vtec"], floor_zero=False)

        roti_frame = sat_frame.dropna(subset=["roti"]).copy()
        if roti_frame.empty:
            _draw_empty_axis(roti_ax, "No valid ROTI")
        else:
            _plot_scatter_by_arc(roti_ax, roti_frame, "roti", color="blue", size=5.0)
            _set_axis_limits(roti_ax, roti_frame["roti"], floor_zero=True)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def load_station_context(
    manifest: pd.DataFrame,
    event: EventWindow,
    station_code4: str,
) -> StationContext:
    records = []
    for observation_day in utc_date_range(event.start_utc, event.end_utc):
        records.append(load_station_record(manifest, observation_day.isoformat(), station_code4))

    if not records:
        raise ValueError(f"No records for station={station_code4} event={event.event_id}")

    station_id = str(records[0]["station_id"])
    station_code = str(records[0]["station_code4"]).upper()
    phase_l1 = _require_clean_field(records[0].get("phase_l1"), "phase_l1", station_code, event)
    phase_l2 = _require_clean_field(records[0].get("phase_l2"), "phase_l2", station_code, event)
    code_l1 = _clean_field(records[0].get("code_l1"))
    code_l2 = _clean_field(records[0].get("code_l2"))

    for record in records[1:]:
        if str(record["station_id"]) != station_id:
            raise ValueError(f"Station id changed within event window for {station_code}")
        if str(record["station_code4"]).upper() != station_code:
            raise ValueError(f"Station code changed within event window for {station_code}")
        current_phase_l1 = _require_clean_field(record.get("phase_l1"), "phase_l1", station_code, event)
        current_phase_l2 = _require_clean_field(record.get("phase_l2"), "phase_l2", station_code, event)
        current_code_l1 = _clean_field(record.get("code_l1"))
        current_code_l2 = _clean_field(record.get("code_l2"))
        if (current_phase_l1, current_phase_l2, current_code_l1, current_code_l2) != (phase_l1, phase_l2, code_l1, code_l2):
            raise ValueError(
                f"Observation field mismatch within event window for station={station_code}"
            )

    return StationContext(
        station_id=station_id,
        station_code4=station_code,
        lat=float(records[0]["lat"]),
        lon=float(records[0]["lon"]),
        height_m=float(records[0]["height_m"]),
        phase_l1=phase_l1,
        phase_l2=phase_l2,
        code_l1=code_l1,
        code_l2=code_l2,
        records_by_date={
            str(record["observation_date"]): record
            for record in sorted(records, key=lambda item: str(item["observation_date"]))
        },
    )


def load_event_observations(
    station: StationContext,
    event: EventWindow,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    measure_fields = [station.phase_l1, station.phase_l2]
    if station.code_l1 is not None:
        measure_fields.append(station.code_l1)
    if station.code_l2 is not None:
        measure_fields.append(station.code_l2)

    for observation_date, record in station.records_by_date.items():
        day_start = datetime.fromisoformat(f"{observation_date}T00:00:00+00:00")
        day_end = day_start + timedelta(days=1)
        slice_start = max(event.start_utc, day_start)
        slice_end = min(event.end_utc, day_end)
        if slice_start > slice_end:
            continue
        obs_frame = _load_record_observation_slice(
            record=record,
            start_utc=slice_start,
            end_utc=slice_end,
            measure_fields=measure_fields,
        )
        if not obs_frame.empty:
            frames.append(obs_frame)

    if not frames:
        return pd.DataFrame(columns=["time", "sv", *measure_fields])

    obs_df = pd.concat(frames, ignore_index=True)
    obs_df["time"] = pd.to_datetime(obs_df["time"], utc=True)
    obs_df = obs_df.drop_duplicates(subset=["time", "sv"], keep="last")
    return obs_df.sort_values(["time", "sv"]).reset_index(drop=True)


def build_satellite_candidate(frame: pd.DataFrame, satellite: str) -> SatelliteCandidate:
    valid_vtec_count = int(frame["vtec"].notna().sum()) if "vtec" in frame.columns else 0
    valid_roti_count = int(frame["roti"].notna().sum()) if "roti" in frame.columns else 0
    longest_arc_length = 0
    if not frame.empty and "arc_id" in frame.columns:
        longest_arc_length = int(frame.groupby("arc_id").size().max())
    return SatelliteCandidate(
        satellite=satellite,
        frame=frame,
        valid_vtec_count=valid_vtec_count,
        valid_roti_count=valid_roti_count,
        longest_arc_length=longest_arc_length,
    )


def select_top_satellite_candidates(
    candidates: Sequence[SatelliteCandidate],
    satellite_count: int,
    event: EventWindow,
    station_code4: str,
) -> list[SatelliteCandidate]:
    qualified = [candidate for candidate in candidates if candidate.qualified]
    ranked = rank_satellite_candidates(qualified)
    if len(ranked) < satellite_count:
        coverage_lines = [
            (
                f"{candidate.satellite}: score={candidate.score}, "
                f"vtec={candidate.valid_vtec_count}, "
                f"roti={candidate.valid_roti_count}, "
                f"longest_arc={candidate.longest_arc_length}"
            )
            for candidate in rank_satellite_candidates(candidates)[:8]
        ]
        coverage_text = "; ".join(coverage_lines) if coverage_lines else "no candidate satellites"
        raise ValueError(
            f"Station {station_code4.upper()} has only {len(ranked)} qualified satellites for "
            f"{event.start_utc.isoformat()} to {event.end_utc.isoformat()}. Coverage: {coverage_text}"
        )
    return ranked[:satellite_count]


def rank_satellite_candidates(candidates: Sequence[SatelliteCandidate]) -> list[SatelliteCandidate]:
    return sorted(
        candidates,
        key=lambda item: (-item.score, -item.longest_arc_length, item.satellite),
    )


def build_output_base_name(
    event: EventWindow,
    station_code4: str,
    satellites: Sequence[str],
) -> str:
    satellite_tag = "_".join(satellite.lower() for satellite in satellites)
    start_tag = event.start_utc.strftime("%Y%m%dT%H%M%S")
    end_tag = event.end_utc.strftime("%Y%m%dT%H%M%S")
    return f"{start_tag}_{end_tag}_{station_code4.upper()}_{satellite_tag}_timeseries"


def build_event_directory_name(event: EventWindow) -> str:
    start_tag = event.start_utc.strftime("%Y%m%dT%H%M%S")
    end_tag = event.end_utc.strftime("%Y%m%dT%H%M%S")
    return f"{start_tag}_{end_tag}"


def build_utc_ticks(event: EventWindow) -> list[int]:
    xmin, xmax = window_hour_bounds(event)
    start_tick = int(np.floor(xmin))
    end_tick = int(np.ceil(xmax))
    return list(range(start_tick, end_tick + 1))


def format_utc_tick_label(hour_value: int | float) -> str:
    return f"{int(hour_value) % 24:02d}"


def window_hour_bounds(event: EventWindow | str) -> tuple[float, float]:
    window = coerce_event_window(event)
    anchor = pd.Timestamp(window.start_utc).tz_convert(UTC).normalize()
    start_hour = (pd.Timestamp(window.start_utc) - anchor).total_seconds() / 3600.0
    end_hour = (pd.Timestamp(window.end_utc) - anchor).total_seconds() / 3600.0
    return start_hour, end_hour


def coerce_event_window(event: EventWindow | str) -> EventWindow:
    if isinstance(event, EventWindow):
        return event
    observation_date = str(event)
    return EventWindow(
        event_id=f"legacy_{observation_date}",
        start_utc=datetime.fromisoformat(f"{observation_date}T00:00:00+00:00"),
        end_utc=datetime.fromisoformat(f"{observation_date}T05:00:00+00:00"),
    )


def _draw_empty_axis(ax: plt.Axes, message: str) -> None:
    ax.set_ylim(0, 1)
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="gray",
    )


def _set_axis_limits(ax: plt.Axes, values: pd.Series, floor_zero: bool) -> None:
    data = pd.Series(values, dtype=float).dropna()
    if data.empty:
        return
    vmin = float(data.min())
    vmax = float(data.max())
    if floor_zero:
        lower = 0.0
        upper = vmax * 1.08 if vmax > 0 else 1.0
    else:
        if np.isclose(vmin, vmax):
            pad = max(abs(vmax) * 0.1, 1.0)
        else:
            pad = (vmax - vmin) * 0.12
        lower = vmin - pad
        upper = vmax + pad
    ax.set_ylim(lower, upper)


def _plot_scatter_by_arc(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value_column: str,
    color: str,
    size: float,
) -> None:
    for _, arc_frame in frame.groupby("arc_id", sort=True):
        ax.scatter(
            arc_frame["utc_hour"],
            arc_frame[value_column],
            color=color,
            s=size,
            linewidths=0,
        )


def _load_record_observation_slice(
    record: dict[str, object],
    start_utc: datetime,
    end_utc: datetime,
    measure_fields: Sequence[str],
) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        obs = georinex.load(
            record["obs_path"],
            use="G",
            meas=list(measure_fields),
            tlim=(start_utc.replace(tzinfo=None), end_utc.replace(tzinfo=None)),
            fast=True,
        )
    obs_df = obs[list(measure_fields)].to_dataframe().reset_index()
    obs_df["time"] = pd.to_datetime(obs_df["time"], utc=True)
    return obs_df


def _extract_satellite_series(
    obs_df: pd.DataFrame,
    station: StationContext,
    satellite: str,
    nav_stores: dict[str, GPSBroadcastStore],
    processing_config: dict[str, object],
) -> pd.DataFrame:
    sat_df = obs_df[obs_df["sv"].astype(str) == satellite].copy()
    if sat_df.empty:
        return pd.DataFrame(columns=TIMESERIES_COLUMNS)

    sat_df = sat_df.sort_values("time").reset_index(drop=True)
    sat_df = normalize_to_interval(sat_df, int(processing_config["target_interval_sec"]))
    sat_df = sat_df.dropna(subset=[station.phase_l1, station.phase_l2]).reset_index(drop=True)
    if sat_df.empty:
        return pd.DataFrame(columns=TIMESERIES_COLUMNS)

    sat_df = _append_satellite_positions(
        sat_df=sat_df,
        satellite=satellite,
        nav_stores=nav_stores,
    )
    sat_df = sat_df.dropna(subset=["sat_x", "sat_y", "sat_z"]).copy()
    if sat_df.empty:
        return pd.DataFrame(columns=TIMESERIES_COLUMNS)

    rx_xyz = _geodetic_to_ecef(station.lat, station.lon, station.height_m)
    rx_lat = np.radians(station.lat)
    rx_lon = np.radians(station.lon)

    az_deg, elev_deg = az_el_from_ecef(
        rx_xyz=np.asarray(rx_xyz),
        sat_xyz=sat_df[["sat_x", "sat_y", "sat_z"]].to_numpy(),
        lat_rad=rx_lat,
        lon_rad=rx_lon,
    )
    sat_df["az_deg"] = az_deg
    sat_df["elev_deg"] = elev_deg
    sat_df = sat_df[sat_df["elev_deg"] >= float(processing_config["elevation_mask_deg"])].copy()
    if sat_df.empty:
        return pd.DataFrame(columns=TIMESERIES_COLUMNS)

    phase_stec = phase_to_stec_tecu(
        sat_df[station.phase_l1].to_numpy(dtype=float),
        sat_df[station.phase_l2].to_numpy(dtype=float),
    )
    gf_phase = geometry_free_phase_m(
        sat_df[station.phase_l1].to_numpy(dtype=float),
        sat_df[station.phase_l2].to_numpy(dtype=float),
    )

    code_stec: np.ndarray | None = None
    mw_cycles: np.ndarray | None = None
    if station.code_l1 is not None and station.code_l2 is not None:
        c1 = sat_df[station.code_l1].to_numpy(dtype=float)
        c2 = sat_df[station.code_l2].to_numpy(dtype=float)
        code_stec = code_to_stec_tecu(c1, c2)
        mw_cycles = melbourne_wubbena_cycles(
            sat_df[station.phase_l1].to_numpy(dtype=float),
            sat_df[station.phase_l2].to_numpy(dtype=float),
            c1,
            c2,
        )

    gap_threshold = float(processing_config["target_interval_sec"]) * float(processing_config["arc_gap_factor"])
    if bool(processing_config["use_cycle_slip_detection"]):
        detection = detect_cycle_slips(
            times=pd.DatetimeIndex(sat_df["time"]),
            gf_phase_m=gf_phase,
            mw_cycles=mw_cycles if bool(processing_config["enable_mw"]) else None,
            gap_threshold_seconds=gap_threshold,
            mw_window_points=int(processing_config["mw_window_points"]),
            mw_slip_threshold_cycles=float(processing_config["mw_slip_threshold_cycles"]),
            gf_window_points=int(processing_config["gf_window_points"]),
            gf_poly_degree=int(processing_config["gf_poly_degree"]),
            gf_residual_threshold_m=float(processing_config["gf_residual_threshold_m"]),
            drop_detected_slip_epoch=bool(processing_config["drop_detected_slip_epoch"]),
        )
    else:
        detection = _build_gap_only_detection_result(
            times=pd.DatetimeIndex(sat_df["time"]),
            gap_threshold_seconds=gap_threshold,
            mw_cycles=mw_cycles if bool(processing_config["enable_mw"]) else None,
        )

    accepted = finalize_satellite_frame(
        sat_df=sat_df,
        phase_stec=phase_stec,
        code_stec=code_stec,
        detection=detection,
        processing_config=processing_config,
    )
    if accepted.empty:
        return pd.DataFrame(columns=TIMESERIES_COLUMNS)

    accepted["utc_hour"] = np.nan
    accepted["station_id"] = station.station_id
    accepted["station_code4"] = station.station_code4
    accepted["sv"] = satellite
    return accepted[TIMESERIES_COLUMNS]


def finalize_satellite_frame(
    sat_df: pd.DataFrame,
    phase_stec: np.ndarray,
    code_stec: np.ndarray | None,
    detection: SlipDetectionResult,
    processing_config: dict[str, object],
) -> pd.DataFrame:
    accepted = sat_df.loc[detection.keep_mask].copy().reset_index(drop=True)
    if accepted.empty:
        return pd.DataFrame(columns=["time", "vtec", "roti", "arc_id"])

    accepted_phase = np.asarray(phase_stec[detection.keep_mask], dtype=float)
    accepted_code = np.asarray(code_stec[detection.keep_mask], dtype=float) if code_stec is not None else None
    arc_ids = pd.Series(detection.arc_ids[detection.keep_mask], dtype=int)

    accepted["arc_id"] = arc_ids.to_numpy(dtype=int)
    accepted["stec"] = normalize_stec_by_arc(accepted_phase, arc_ids)
    accepted["vtec_source"] = _build_vtec_source(accepted["stec"].to_numpy(dtype=float), accepted_code)
    accepted["vtec"] = accepted["vtec_source"] / mapping_function(
        accepted["elev_deg"].to_numpy(dtype=float),
        float(processing_config["shell_height_km"]),
    )
    accepted["roti"] = np.nan

    roti_points = max(
        1,
        int(
            float(processing_config["roti_window_minutes"]) * 60
            / int(processing_config["target_interval_sec"])
        ),
    )
    roti_frame = _compute_roti_series(accepted, roti_points)
    if not roti_frame.empty:
        accepted = accepted.merge(roti_frame[["time", "arc_id", "roti"]], on=["time", "arc_id"], how="left")
        accepted["roti"] = accepted["roti_y"].combine_first(accepted["roti_x"])
        accepted = accepted.drop(columns=["roti_x", "roti_y"])

    return accepted


def _build_gap_only_detection_result(
    times: pd.DatetimeIndex,
    gap_threshold_seconds: float,
    mw_cycles: np.ndarray | None,
) -> SlipDetectionResult:
    times_index = pd.DatetimeIndex(times)
    keep_mask = np.ones(len(times_index), dtype=bool)
    if len(times_index) == 0:
        arc_ids = np.array([], dtype=int)
        gap_break_count = 0
    else:
        dt_seconds = pd.Series(times_index).diff().dt.total_seconds().fillna(0.0)
        arc_ids = dt_seconds.gt(gap_threshold_seconds).cumsum().to_numpy(dtype=int)
        gap_break_count = int(dt_seconds.gt(gap_threshold_seconds).sum())
    return SlipDetectionResult(
        keep_mask=keep_mask,
        arc_ids=arc_ids,
        gap_break_count=gap_break_count,
        mw_break_count=0,
        gf_break_count=0,
        dropped_epoch_count=0,
        code_pair_available=mw_cycles is not None,
        mw_used=False,
    )


def _append_satellite_positions(
    sat_df: pd.DataFrame,
    satellite: str,
    nav_stores: dict[str, GPSBroadcastStore],
) -> pd.DataFrame:
    enriched = sat_df.copy()
    enriched["record_date"] = enriched["time"].dt.strftime("%Y-%m-%d")
    positions = np.full((len(enriched), 3), np.nan, dtype=float)
    for record_date, group in enriched.groupby("record_date", sort=False):
        if record_date not in nav_stores:
            continue
        positions[group.index.to_numpy()] = nav_stores[record_date].position_ecef(
            satellite,
            pd.DatetimeIndex(group["time"]),
        )
    enriched[["sat_x", "sat_y", "sat_z"]] = positions
    return enriched.drop(columns=["record_date"])


def _compute_roti_series(frame: pd.DataFrame, roti_points: int) -> pd.DataFrame:
    roti_frames: list[pd.DataFrame] = []
    for _, arc_frame in frame.groupby("arc_id", sort=True):
        if len(arc_frame) < roti_points:
            continue
        roti_frames.append(compute_rot_roti(arc_frame.copy(), roti_points))
    if not roti_frames:
        return pd.DataFrame(columns=["time", "arc_id", "roti"])
    result = pd.concat(roti_frames, ignore_index=True)
    return result.dropna(subset=["roti"])[["time", "arc_id", "roti"]]


def _build_vtec_source(stec_values: np.ndarray, code_stec: np.ndarray | None) -> np.ndarray:
    fallback = _fallback_vtec_source(stec_values)
    if code_stec is None:
        return fallback
    code_array = np.asarray(code_stec, dtype=float)
    return np.where(np.isfinite(code_array), code_array, fallback)


def _clean_field(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _require_clean_field(
    value: object,
    field_name: str,
    station_code4: str,
    event: EventWindow,
) -> str:
    text = _clean_field(value)
    if text is None:
        raise ValueError(
            f"Missing {field_name} for station={station_code4} "
            f"event={event.start_utc.isoformat()} to {event.end_utc.isoformat()}"
        )
    return text
