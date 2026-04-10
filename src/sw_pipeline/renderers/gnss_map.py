from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..models import EventSpec, GnssGridProduct, GnssGridSlice
from .style import TICK_FONT_SIZE, figure_style, set_axis_title, style_axis_ticks, style_colorbar

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
except Exception:  # pragma: no cover - optional fallback
    ccrs = None
    cfeature = None
    LatitudeFormatter = None
    LongitudeFormatter = None


DEFAULT_LON_TICKS = (-140.0, -100.0, -60.0, -20.0)
DEFAULT_LAT_TICKS = (-60.0, -20.0, 20.0, 60.0)
MAP_TICK_FONT_SIZE = TICK_FONT_SIZE


def render_gnss_maps(event_spec: EventSpec, products: list[GnssGridProduct], metric: str) -> list[Path]:
    outputs: list[Path] = []
    for slice_data in iter_gnss_slices(products, metric):
        prepared = prepare_gnss_slice(slice_data, event_spec)
        if prepared is None:
            continue
        output_path = (
            event_spec.storage.figures_gnss_dir
            / prepared.producer
            / metric
            / f"{metric}_{prepared.timestamp.strftime('%Y%m%dT%H%MZ')}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        render_gnss_slice(prepared, event_spec, output_path)
        outputs.append(output_path)
    return outputs


def iter_gnss_slices(products: list[GnssGridProduct], metric: str):
    for product in sorted(products, key=lambda item: item.time_start):
        if metric not in product.metrics:
            continue
        with xr.open_dataset(product.path) as dataset:
            if product.producer == "internal":
                variable = metric
            else:
                variable = _detect_external_var(dataset, metric)
            if variable is None:
                continue

            time_values = pd.to_datetime(dataset["time"].values, utc=True)
            lat = np.asarray(dataset[_find_coord_name(dataset, ("lat", "latitude"))].values, dtype=float)
            lon = np.asarray(dataset[_find_coord_name(dataset, ("lon", "longitude"))].values, dtype=float)
            units = dataset[variable].attrs.get("units")

            for index, timestamp in enumerate(time_values):
                values = np.asarray(dataset[variable].isel(time=index).values, dtype=float)
                yield GnssGridSlice(
                    event_id=product.event_id,
                    metric=metric,
                    producer=product.producer,
                    source_path=product.path,
                    timestamp=timestamp.tz_convert(None),
                    lat=lat,
                    lon=lon,
                    values=values,
                    units=units,
                )


def prepare_gnss_slice(slice_data: GnssGridSlice, event_spec: EventSpec) -> GnssGridSlice | None:
    lat = np.asarray(slice_data.lat, dtype=float)
    lon = np.asarray(slice_data.lon, dtype=float)
    values = np.asarray(slice_data.values, dtype=float)

    if lon.ndim != 1 or lat.ndim != 1 or values.ndim != 2:
        return None

    if np.nanmax(lon) > 180:
        lon = ((lon + 180.0) % 360.0) - 180.0
        order = np.argsort(lon)
        lon = lon[order]
        values = values[:, order]

    lon_min, lon_max, lat_min, lat_max = event_spec.map_extent()
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    if not np.any(lon_mask) or not np.any(lat_mask):
        return None

    return GnssGridSlice(
        event_id=slice_data.event_id,
        metric=slice_data.metric,
        producer=slice_data.producer,
        source_path=slice_data.source_path,
        timestamp=slice_data.timestamp,
        lat=lat[lat_mask],
        lon=lon[lon_mask],
        values=values[np.ix_(lat_mask, lon_mask)],
        units=slice_data.units,
    )


def render_gnss_slice(slice_data: GnssGridSlice, event_spec: EventSpec, output_path: Path) -> None:
    with figure_style(event_spec.plot_defaults.font_family):
        if ccrs is not None and event_spec.plot_defaults.use_cartopy:
            fig = plt.figure(figsize=event_spec.plot_defaults.figure_size)
            ax = plt.axes(projection=ccrs.PlateCarree())
        else:  # pragma: no cover - fallback path
            fig, ax = plt.subplots(figsize=event_spec.plot_defaults.figure_size)
        mesh = plot_gnss_slice(ax, slice_data, event_spec, draw_labels=True, top_labels=True, bottom_labels=False)

        title = f"{slice_data.metric.upper()} | {slice_data.timestamp.strftime('%Y-%m-%d %H:%M')} UTC | {slice_data.producer}"
        set_axis_title(ax, title, font_family=event_spec.plot_defaults.font_family)
        colorbar = fig.colorbar(mesh, ax=ax, shrink=0.86, pad=0.04)
        label = slice_data.units or slice_data.metric.upper()
        style_colorbar(colorbar, label, font_family=event_spec.plot_defaults.font_family)
        fig.tight_layout()
        fig.savefig(output_path, dpi=event_spec.plot_defaults.dpi, bbox_inches="tight")
        plt.close(fig)


def plot_gnss_slice(
    ax,
    slice_data: GnssGridSlice,
    event_spec: EventSpec,
    *,
    draw_labels: bool = False,
    top_labels: bool = False,
    left_labels: bool = True,
    bottom_labels: bool = True,
):
    style = event_spec.plot_defaults.gnss_styles[slice_data.metric]
    if ccrs is not None and event_spec.plot_defaults.use_cartopy:
        ax.set_extent(event_spec.map_extent(), crs=ccrs.PlateCarree())
        mesh = ax.pcolormesh(
            slice_data.lon,
            slice_data.lat,
            np.ma.masked_invalid(slice_data.values),
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap=style.cmap,
            vmin=style.vmin,
            vmax=style.vmax,
        )
        ax.coastlines(resolution="110m", linewidth=0.8)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.4)
        gl = ax.gridlines(draw_labels=draw_labels, linestyle="--", linewidth=0.4, alpha=0.5)
        gl.xlocator = mticker.FixedLocator(DEFAULT_LON_TICKS)
        gl.ylocator = mticker.FixedLocator(DEFAULT_LAT_TICKS)
        if LongitudeFormatter is not None:
            gl.xformatter = LongitudeFormatter(degree_symbol="°")
        if LatitudeFormatter is not None:
            gl.yformatter = LatitudeFormatter(degree_symbol="°")
        if draw_labels:
            gl.top_labels = top_labels
            gl.bottom_labels = bottom_labels
            gl.right_labels = False
            gl.left_labels = left_labels
            gl.xlabel_style = {"size": MAP_TICK_FONT_SIZE, "family": event_spec.plot_defaults.font_family}
            gl.ylabel_style = {"size": MAP_TICK_FONT_SIZE, "family": event_spec.plot_defaults.font_family}
        style_axis_ticks(ax, font_family=event_spec.plot_defaults.font_family)
        _draw_magnetic_equator(ax, event_spec, slice_data.timestamp, use_cartopy=True)
        return mesh

    mesh = ax.pcolormesh(
        slice_data.lon,
        slice_data.lat,
        np.ma.masked_invalid(slice_data.values),
        shading="auto",
        cmap=style.cmap,
        vmin=style.vmin,
        vmax=style.vmax,
    )
    lon_min, lon_max, lat_min, lat_max = event_spec.map_extent()
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xticks(DEFAULT_LON_TICKS)
    ax.set_yticks(DEFAULT_LAT_TICKS)
    ax.grid(True, linestyle="--", alpha=0.5)
    style_axis_ticks(ax, font_family=event_spec.plot_defaults.font_family)
    if not left_labels:
        ax.set_yticklabels([])
    if not bottom_labels:
        ax.set_xticklabels([])
    style_axis_ticks(ax, font_family=event_spec.plot_defaults.font_family)
    _draw_magnetic_equator(ax, event_spec, slice_data.timestamp, use_cartopy=False)
    return mesh


def _find_coord_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    lowered = {name.lower(): name for name in dataset.variables}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(f"Missing coordinate in dataset, expected one of {candidates}")


def _detect_external_var(dataset: xr.Dataset, metric: str) -> str | None:
    lowered = {name.lower(): name for name in dataset.data_vars}
    if metric == "roti":
        return lowered.get("roti")
    for candidate in ("atec", "vtec", "tec"):
        if candidate in lowered:
            return lowered[candidate]
    return None


def _draw_magnetic_equator(ax, event_spec: EventSpec, timestamp: pd.Timestamp, *, use_cartopy: bool) -> None:
    if not event_spec.plot_defaults.show_magnetic_equator:
        return
    try:
        from ..internal import gold_core

        lon_eq, lat_eq = gold_core.compute_magnetic_equator(
            event_spec.map_extent(),
            pd.Timestamp(timestamp).to_pydatetime(),
        )
    except Exception:
        return
    if len(lon_eq) <= 1:
        return

    plot_kwargs = {
        "color": event_spec.plot_defaults.magnetic_equator_color,
        "linewidth": event_spec.plot_defaults.magnetic_equator_linewidth,
        "linestyle": "--",
        "zorder": 5,
    }
    if use_cartopy and ccrs is not None:
        plot_kwargs["transform"] = ccrs.PlateCarree()
    ax.plot(lon_eq, lat_eq, **plot_kwargs)
