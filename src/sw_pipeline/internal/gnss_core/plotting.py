from __future__ import annotations

import logging
from datetime import UTC, datetime
from importlib import import_module

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .utils import build_event_time_index, load_dataframe


matplotlib.use("Agg")
LOGGER = logging.getLogger(__name__)


def execute_plot_stage(config: PipelineConfig) -> list[str]:
    outputs: list[str] = []
    for event in config.events:
        grid_path = config.outputs.grid_dir / f"{event.event_id}.parquet"
        if not grid_path.exists():
            LOGGER.info("Grid table missing for %s, skipping plot stage for this event.", event.event_id)
            continue
        grid_df = load_dataframe(grid_path)
        if grid_df.empty:
            continue
        grid_df["time"] = pd.to_datetime(grid_df["time"], utc=True)
        out_dir = config.outputs.map_dir / event.event_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for timestamp in build_event_time_index(event.start_utc, event.end_utc, int(config.grid["cadence_minutes"])):
            frame = grid_df[grid_df["time"] == timestamp].copy()
            dt = pd.Timestamp(timestamp).to_pydatetime().astimezone(UTC)
            tec_path = out_dir / f"{dt:%Y%m%d_%H%M}_tec.png"
            roti_path = out_dir / f"{dt:%Y%m%d_%H%M}_roti.png"
            _plot_field(frame, config, dt, "vtec_median", "VTEC (TECU)", float(config.plot["tec_vmin"]), float(config.plot["tec_vmax"]), str(config.plot["tec_cmap"]), tec_path)
            _plot_field(frame, config, dt, "roti_median", "ROTI (TECU/min)", float(config.plot["roti_vmin"]), float(config.plot["roti_vmax"]), str(config.plot["roti_cmap"]), roti_path)
            outputs.extend([str(tec_path), str(roti_path)])
    LOGGER.info("Generated %s PNG frames", len(outputs))
    return outputs


def _plot_field(
    frame: pd.DataFrame,
    config: PipelineConfig,
    timestamp: datetime,
    value_column: str,
    colorbar_label: str,
    vmin: float,
    vmax: float,
    cmap: str,
    output_path,
) -> None:
    valid_frame = frame.dropna(subset=["lat_bin", "lon_bin"]).copy()
    lon_step = float(config.grid["lon_step_deg"])
    lat_step = float(config.grid["lat_step_deg"])
    lon_bins = np.arange(config.bbox["lon_min"], config.bbox["lon_max"], lon_step)
    lat_bins = np.arange(config.bbox["lat_min"], config.bbox["lat_max"], lat_step)
    matrix = np.full((len(lat_bins), len(lon_bins)), np.nan)
    lon_lookup = {round(value, 6): idx for idx, value in enumerate(lon_bins)}
    lat_lookup = {round(value, 6): idx for idx, value in enumerate(lat_bins)}
    for row in valid_frame.itertuples(index=False):
        lon_idx = lon_lookup.get(round(float(row.lon_bin), 6))
        lat_idx = lat_lookup.get(round(float(row.lat_bin), 6))
        if lon_idx is not None and lat_idx is not None:
            matrix[lat_idx, lon_idx] = getattr(row, value_column)

    fig = plt.figure(figsize=(12, 7), dpi=int(config.plot["dpi"]))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(
        [config.bbox["lon_min"], config.bbox["lon_max"], config.bbox["lat_min"], config.bbox["lat_max"]],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbeef7", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f3eee2", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#dbeef7", edgecolor="none", alpha=0.7)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.2, edgecolor="gray", alpha=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    mesh = ax.pcolormesh(
        np.append(lon_bins, lon_bins[-1] + lon_step),
        np.append(lat_bins, lat_bins[-1] + lat_step),
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    lon_eq, lat_eq = magnetic_equator_for_day(timestamp.date().isoformat(), config)
    valid_eq = np.isfinite(lon_eq) & np.isfinite(lat_eq)
    if valid_eq.any():
        ax.plot(
            lon_eq[valid_eq],
            lat_eq[valid_eq],
            color="cyan",
            linewidth=1.2,
            linestyle="-",
            transform=ccrs.PlateCarree(),
            label="Magnetic equator",
        )
        ax.legend(loc="lower left")
    if valid_frame.empty:
        ax.text(
            0.5,
            0.52,
            "No valid GNSS samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    ax.set_title(f"{colorbar_label} | {timestamp.astimezone(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.78, pad=0.03)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


_MAGNETIC_EQUATOR_CACHE: dict[tuple[str, float, float, float, float, float, float], tuple[np.ndarray, np.ndarray]] = {}


def magnetic_equator_for_day(day_iso: str, config: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    cache_key = (
        day_iso,
        float(config.bbox["lon_min"]),
        float(config.bbox["lon_max"]),
        float(config.bbox["lat_min"]),
        float(config.bbox["lat_max"]),
        float(config.plot["magnetic_equator_step_deg"]),
        float(config.processing["shell_height_km"]),
    )
    if cache_key in _MAGNETIC_EQUATOR_CACHE:
        return _MAGNETIC_EQUATOR_CACHE[cache_key]
    day = datetime.fromisoformat(day_iso)
    apex = _load_apex()(day, refh=float(config.processing["shell_height_km"]))
    step = float(config.plot["magnetic_equator_step_deg"])
    longitudes = np.arange(config.bbox["lon_min"], config.bbox["lon_max"] + step, step)
    latitudes: list[float] = []
    lat_grid = np.arange(config.bbox["lat_min"], config.bbox["lat_max"] + 0.25, 0.25)
    for lon in longitudes:
        qd = np.array([apex.geo2qd(float(lat), float(lon), float(config.processing["shell_height_km"]))[0] for lat in lat_grid])
        crossings = np.where(np.signbit(qd[:-1]) != np.signbit(qd[1:]))[0]
        if len(crossings) == 0:
            latitudes.append(np.nan)
            continue
        idx = crossings[np.argmin(np.abs(lat_grid[crossings]))]
        low = float(lat_grid[idx])
        high = float(lat_grid[idx + 1])
        low_val = float(qd[idx])
        high_val = float(qd[idx + 1])
        for _ in range(20):
            mid = 0.5 * (low + high)
            mid_val = float(apex.geo2qd(mid, float(lon), float(config.processing["shell_height_km"]))[0])
            if np.signbit(mid_val) == np.signbit(low_val):
                low, low_val = mid, mid_val
            else:
                high, high_val = mid, mid_val
        latitudes.append(0.5 * (low + high))
    result = (longitudes, np.asarray(latitudes))
    _MAGNETIC_EQUATOR_CACHE[cache_key] = result
    return result


def _load_apex():
    try:
        return import_module("apexpy").Apex
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("apexpy is required for magnetic equator overlay in plot stage.") from exc
