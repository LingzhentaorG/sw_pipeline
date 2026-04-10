from __future__ import annotations

from pathlib import Path
import string

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..models import EventSpec, StationSeriesPreset
from ..utils import read_partitioned_parquet
from .style import figure_style, set_axis_labels, set_axis_title, style_axis_ticks


STATION_TITLE_FONT_SIZE = 15.5
STATION_LABEL_FONT_SIZE = 14.0
STATION_TICK_FONT_SIZE = 13.0


def render_station_series(event_spec: EventSpec, preset: StationSeriesPreset, workspace_root: Path) -> Path:
    vtec_frame = read_partitioned_parquet(workspace_root / "intermediate" / "vtec", event_spec.event_id)
    roti_frame = read_partitioned_parquet(workspace_root / "intermediate" / "roti", event_spec.event_id)
    series_color = "black"

    vtec_frame["time"] = pd.to_datetime(vtec_frame["time"], utc=True)
    roti_frame["time"] = pd.to_datetime(roti_frame["time"], utc=True)

    station_id = preset.station_id or _resolve_station_id(event_spec, workspace_root, preset.station_code)
    start = pd.Timestamp(preset.start_utc).tz_convert("UTC")
    end = pd.Timestamp(preset.end_utc).tz_convert("UTC")
    satellites = tuple(preset.satellites[:2])

    with figure_style(event_spec.plot_defaults.font_family):
        fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=True)
        fig.subplots_adjust(hspace=0.18, wspace=0.12, top=0.95)

        for column, satellite in enumerate(satellites):
            tec_ax = axes[0, column]
            roti_ax = axes[1, column]

            tec_series = _filter_series(vtec_frame, station_id, preset.station_code, satellite, start, end, "vtec")
            roti_series = _filter_series(roti_frame, station_id, preset.station_code, satellite, start, end, "roti")

            tec_ax.scatter(
                tec_series["time"].dt.tz_convert("UTC").dt.tz_localize(None),
                tec_series["vtec"],
                s=14,
                color=series_color,
                alpha=0.8,
                linewidths=0,
            )
            set_axis_title(
                tec_ax,
                _panel_title(0, column, preset.station_code, satellite, "VTEC"),
                font_family=event_spec.plot_defaults.font_family,
                fontsize=STATION_TITLE_FONT_SIZE,
                pad=8,
            )
            set_axis_labels(
                tec_ax,
                font_family=event_spec.plot_defaults.font_family,
                ylabel="VTEC (TECU)",
                fontsize=STATION_LABEL_FONT_SIZE,
            )
            tec_ax.grid(True, linestyle="--", alpha=0.4)
            style_axis_ticks(tec_ax, font_family=event_spec.plot_defaults.font_family, labelsize=STATION_TICK_FONT_SIZE)

            roti_ax.scatter(
                roti_series["time"].dt.tz_convert("UTC").dt.tz_localize(None),
                roti_series["roti"],
                s=14,
                color=series_color,
                alpha=0.8,
                linewidths=0,
            )
            set_axis_title(
                roti_ax,
                _panel_title(1, column, preset.station_code, satellite, "ROTI"),
                font_family=event_spec.plot_defaults.font_family,
                fontsize=STATION_TITLE_FONT_SIZE,
                pad=8,
            )
            set_axis_labels(
                roti_ax,
                font_family=event_spec.plot_defaults.font_family,
                ylabel="ROTI",
                xlabel="UTC",
                fontsize=STATION_LABEL_FONT_SIZE,
            )
            roti_ax.grid(True, linestyle="--", alpha=0.4)
            style_axis_ticks(roti_ax, font_family=event_spec.plot_defaults.font_family, labelsize=STATION_TICK_FONT_SIZE)

        output_path = event_spec.storage.figures_station_series_dir / f"{preset.name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=event_spec.plot_defaults.dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path


def _panel_title(row_index: int, column_index: int, station_code: str, satellite: str, metric: str) -> str:
    label_index = row_index * 2 + column_index
    label = string.ascii_lowercase[label_index]
    return f"({label}) {station_code} {satellite} {metric}"


def _filter_series(
    frame: pd.DataFrame,
    station_id: str,
    station_code: str,
    satellite: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    value_column: str,
) -> pd.DataFrame:
    if "station_id" in frame.columns:
        station_id_mask = frame["station_id"].astype(str).str.upper() == station_id.upper()
    else:
        station_id_mask = pd.Series(False, index=frame.index)

    if station_id_mask.any():
        station_mask = station_id_mask
    elif "station_code4" in frame.columns:
        station_mask = frame["station_code4"].astype(str).str.upper() == station_code.upper()
    else:
        station_mask = station_id_mask

    filtered = frame[
        station_mask
        & (frame["sv"].astype(str).str.upper() == satellite.upper())
        & (frame["time"] >= start)
        & (frame["time"] <= end)
    ].copy()
    filtered = filtered.sort_values("time")
    return filtered.loc[:, ["time", value_column]]


def _resolve_station_id(event_spec: EventSpec, workspace_root: Path, station_code: str) -> str:
    manifest = workspace_root / "manifests" / "normalized_manifest.csv"
    frame = pd.read_csv(manifest)
    matches = frame[frame["station_code4"].astype(str).str.upper() == station_code.upper()]
    if matches.empty:
        raise ValueError(f"Station code not found in normalized manifest: {station_code}")
    return str(matches.iloc[0]["station_id"])
