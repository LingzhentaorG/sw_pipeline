from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from ..models import EventSpec, GoldScene
from .style import GOLD_RADIANCE_LABEL, LEGEND_FONT_SIZE, figure_style, set_axis_labels, set_axis_title, style_colorbar


def _resolve_max_emission_angle_deg(event_spec: EventSpec):
    raw_value = event_spec.runtime.get("gold_max_emission_angle_deg", None)
    if raw_value is None:
        return None
    return float(raw_value)


def render_gold_maps(event_spec: EventSpec, scenes: list[GoldScene]) -> list[Path]:
    outputs: list[Path] = []
    for scene in scenes:
        output_path = event_spec.storage.figures_gold_dir / f"gold_{scene.midpoint.strftime('%Y%m%dT%H%MZ')}.png"
        render_gold_scene(event_spec, scene, output_path)
        outputs.append(output_path)
    return outputs


def render_gold_scene(
    event_spec: EventSpec,
    scene: GoldScene,
    output_path: Path,
    overlay: dict[str, object] | None = None,
) -> None:
    from ..internal import gold_core

    pair = resolve_scene_pair(scene, float(event_spec.runtime.get("gold_max_pair_minutes", 5)))
    if pair is None:
        raise ValueError(f"Could not resolve GOLD pair for {scene.tar_path.name} @ {scene.midpoint}")

    with figure_style(event_spec.plot_defaults.font_family):
        fig, ax = plt.subplots(
            figsize=event_spec.plot_defaults.figure_size,
            subplot_kw={"projection": gold_core.ccrs.PlateCarree()},
        )
        mesh = plot_gold_pair(ax, event_spec, pair, overlay=overlay)

        colorbar = fig.colorbar(mesh, ax=ax, shrink=0.83, pad=0.05)
        style_colorbar(colorbar, GOLD_RADIANCE_LABEL, font_family=event_spec.plot_defaults.font_family)
        title = overlay["title"] if overlay and "title" in overlay else gold_core.format_pair_title(pair)
        set_axis_title(ax, str(title), font_family=event_spec.plot_defaults.font_family)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=event_spec.plot_defaults.dpi, bbox_inches="tight")
        plt.close(fig)


def resolve_scene_pair(scene: GoldScene, max_pair_minutes: float):
    from ..internal import gold_core

    entries = gold_core.discover_entries(scene.tar_path)
    pairs, _ = gold_core.match_pairs(entries, max_pair_minutes)
    for pair in pairs:
        if pair.cha.member_name == scene.cha_member and pair.chb.member_name == scene.chb_member:
            return pair
    return None


def plot_gold_pair(
    ax,
    event_spec: EventSpec,
    pair,
    *,
    overlay: dict[str, object] | None = None,
    decorate_overlay: bool = True,
    draw_labels: bool = True,
    left_labels: bool = True,
    bottom_labels: bool = True,
):
    from ..internal import gold_core

    max_emission_angle_deg = _resolve_max_emission_angle_deg(event_spec)
    if max_emission_angle_deg is None:
        max_emission_angle_deg = gold_core.DEFAULT_MAX_EMISSION_ANGLE_DEG

    cha_lon, cha_lat, cha_rad = gold_core.read_geo_grid(
        pair.cha,
        135.6,
        "all",
        max_emission_angle_deg=max_emission_angle_deg,
    )
    chb_lon, chb_lat, chb_rad = gold_core.read_geo_grid(
        pair.chb,
        135.6,
        "all",
        max_emission_angle_deg=max_emission_angle_deg,
    )
    extent = tuple(overlay["extent"]) if overlay and overlay.get("extent") else event_spec.map_extent()

    gold_core.add_map_background(
        ax,
        draw_labels=draw_labels,
        left_labels=left_labels,
        bottom_labels=bottom_labels,
        font_family=event_spec.plot_defaults.font_family,
    )
    ax.set_extent(extent, crs=gold_core.ccrs.PlateCarree())

    mesh = None
    for lon, lat, radiance in ((cha_lon, cha_lat, cha_rad), (chb_lon, chb_lat, chb_rad)):
        if radiance.count() == 0:
            continue
        mesh = gold_core.plot_swath(
            ax,
            lon,
            lat,
            radiance,
            vmin=0.0,
            vmax=300.0,
            point_size=9.0,
            gap_factor=6.0,
        )

    if mesh is None:
        raise RuntimeError("No drawable GOLD pixels were available.")

    if event_spec.plot_defaults.show_magnetic_equator:
        gold_core.add_magnetic_equator(
            ax,
            extent,
            pair.midpoint,
            color=event_spec.plot_defaults.magnetic_equator_color,
            linewidth=event_spec.plot_defaults.magnetic_equator_linewidth,
        )

    if overlay is not None:
        _draw_overlay(ax, overlay, event_spec)
        if decorate_overlay:
            _decorate_overlay(ax, overlay, font_family=event_spec.plot_defaults.font_family)
    return mesh


def _draw_overlay(ax, overlay: dict[str, object], event_spec: EventSpec) -> None:
    from ..internal import gold_core

    if overlay.get("draw_mode") == "grid":
        _draw_overlay_grid(ax, overlay, gold_core)
        return
    lon_points, lat_points = _resolve_overlay_points(overlay)
    if lon_points.size == 0 or lat_points.size == 0:
        return
    ax.scatter(
        lon_points,
        lat_points,
        s=float(overlay.get("size", 14.0)),
        c=str(overlay.get("color", "red")),
        alpha=0.9,
        linewidths=0,
        marker=str(overlay.get("marker", "o")),
        transform=gold_core.ccrs.PlateCarree(),
        zorder=6,
    )


def _resolve_overlay_points(overlay: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    if "lon_points" in overlay and "lat_points" in overlay:
        return (
            np.asarray(overlay["lon_points"], dtype=float),
            np.asarray(overlay["lat_points"], dtype=float),
        )
    lat = np.asarray(overlay["lat"], dtype=float)
    lon = np.asarray(overlay["lon"], dtype=float)
    # Preserve masked-array thresholds from the overlay preparation step.
    values = np.ma.masked_invalid(np.ma.asarray(overlay["values"], dtype=float))
    if lon.ndim == 1 and lat.ndim == 1 and values.ndim == 2:
        lon_grid, lat_grid = np.meshgrid(lon, lat)
    else:
        lon_grid = np.asarray(lon, dtype=float)
        lat_grid = np.asarray(lat, dtype=float)
    mask = ~np.ma.getmaskarray(values)
    if not np.any(mask):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return np.asarray(lon_grid[mask], dtype=float), np.asarray(lat_grid[mask], dtype=float)


def _draw_overlay_grid(ax, overlay: dict[str, object], gold_core) -> None:
    lon = np.asarray(overlay["lon"], dtype=float)
    lat = np.asarray(overlay["lat"], dtype=float)
    values = np.ma.masked_invalid(np.ma.asarray(overlay["values"], dtype=float))
    if lon.ndim != 1 or lat.ndim != 1 or values.ndim != 2 or np.ma.count(values) == 0:
        return

    lon_edges = _center_to_axis_edges(lon)
    lat_edges = _center_to_axis_edges(lat)
    filled = np.ma.masked_where(np.ma.getmaskarray(values), np.ones(values.shape, dtype=float))
    ax.pcolormesh(
        lon_edges,
        lat_edges,
        filled,
        cmap=ListedColormap([str(overlay.get("color", "red"))]),
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        alpha=0.95,
        transform=gold_core.ccrs.PlateCarree(),
        zorder=6,
        rasterized=True,
    )


def _center_to_axis_edges(center: np.ndarray) -> np.ndarray:
    if center.ndim != 1 or center.size == 0:
        return np.asarray([], dtype=float)
    if center.size == 1:
        delta = 0.25
        return np.asarray([center[0] - delta, center[0] + delta], dtype=float)
    edge = np.empty(center.size + 1, dtype=float)
    edge[1:-1] = 0.5 * (center[:-1] + center[1:])
    edge[0] = center[0] - 0.5 * (center[1] - center[0])
    edge[-1] = center[-1] + 0.5 * (center[-1] - center[-2])
    return edge


def _decorate_overlay(ax, overlay: dict[str, object], *, font_family: str) -> None:
    legend_label = overlay.get("legend_label")
    if legend_label:
        color = str(overlay.get("color", "red"))
        handle = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=5,
            linewidth=0,
        )
        ax.legend(
            [handle],
            [str(legend_label)],
            loc="upper right",
            frameon=True,
            fontsize=LEGEND_FONT_SIZE,
            prop={"family": font_family, "size": LEGEND_FONT_SIZE},
        )
    ylabel = overlay.get("ylabel")
    if ylabel:
        set_axis_labels(
            ax,
            font_family=font_family,
            ylabel=str(ylabel),
            fontsize=12.5,
            fontweight="bold",
        )
