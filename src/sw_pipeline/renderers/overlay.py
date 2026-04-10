from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..models import EventSpec, GnssGridProduct, GoldScene, OverlaySpec
from ..registry.manifests import write_overlay_pairs
from ..registry.pairing import pair_nearest_times
from .gnss_map import iter_gnss_slices, prepare_gnss_slice
from .gold_map import render_gold_scene
from .style import metric_threshold_label, overlay_ylabel


def render_overlays(
    event_spec: EventSpec,
    products: list[GnssGridProduct],
    scenes: list[GoldScene],
    overlay_spec: OverlaySpec,
) -> list[Path]:
    gnss_slices = [prepare_gnss_slice(slice_data, event_spec) for slice_data in iter_gnss_slices(products, overlay_spec.metric)]
    gnss_slices = [slice_data for slice_data in gnss_slices if slice_data is not None]
    if not gnss_slices or not scenes:
        write_overlay_pairs(
            event_spec.storage.manifests_dir / f"{overlay_spec.name}_pairs.csv",
            [
                {
                    "overlay_name": overlay_spec.name,
                    "metric": overlay_spec.metric,
                    "status": "skipped",
                    "detail": "missing_gnss_or_gold",
                }
            ],
        )
        return []

    rows, pairs = _resolve_overlay_pairs(gnss_slices, scenes, overlay_spec)
    outputs: list[Path] = []
    for pair in pairs:
        slice_data = pair["slice_data"]
        scene = pair["scene"]
        delta_seconds = float(pair["delta"].total_seconds())
        overlay_payload = build_overlay_payload(slice_data, overlay_spec, event_spec)
        if int(overlay_payload["count"]) == 0:
            rows.append(
                {
                    "overlay_name": overlay_spec.name,
                    "metric": overlay_spec.metric,
                    "gnss_time": slice_data.timestamp,
                    "gold_time": scene.midpoint,
                    "delta_seconds": delta_seconds,
                    "status": "skipped",
                    "detail": "threshold_removed_all_pixels",
                }
            )
            continue
        output_path = (
            event_spec.storage.figures_overlays_dir
            / overlay_spec.name
            / overlay_spec.producer
            / f"{overlay_spec.name}_{slice_data.timestamp.strftime('%Y%m%dT%H%MZ')}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        render_gold_scene(
            event_spec,
            scene,
            output_path,
            overlay=overlay_payload,
        )
        rows.append(
            {
                "overlay_name": overlay_spec.name,
                "metric": overlay_spec.metric,
                "gnss_time": slice_data.timestamp,
                "gold_time": scene.midpoint,
                "delta_seconds": delta_seconds,
                "status": "rendered",
                "detail": "",
                "output_path": output_path,
            }
        )
        outputs.append(output_path)

    if not rows:
        rows.append(
            {
                "overlay_name": overlay_spec.name,
                "metric": overlay_spec.metric,
                "status": "skipped",
                "detail": "no_pairs_within_tolerance",
            }
        )
    write_overlay_pairs(event_spec.storage.manifests_dir / f"{overlay_spec.name}_pairs.csv", rows)
    return outputs


def _resolve_overlay_pairs(gnss_slices, scenes: list[GoldScene], overlay_spec: OverlaySpec) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if overlay_spec.pairs:
        return _resolve_explicit_overlay_pairs(gnss_slices, scenes, overlay_spec)

    slice_times = [_as_utc_naive(slice_data.timestamp) for slice_data in gnss_slices]
    scene_times = [_as_utc_naive(scene.midpoint) for scene in scenes]
    time_pairs = pair_nearest_times(
        slice_times,
        scene_times,
        timedelta(minutes=overlay_spec.max_pair_delta_minutes),
    )
    if not time_pairs:
        return (
            [
                {
                    "overlay_name": overlay_spec.name,
                    "metric": overlay_spec.metric,
                    "status": "skipped",
                    "detail": "no_pairs_within_tolerance",
                }
            ],
            [],
        )

    resolved_pairs: list[dict[str, object]] = []
    for pair in time_pairs:
        resolved_pairs.append(
            {
                "slice_data": gnss_slices[pair.left_index],
                "scene": scenes[pair.right_index],
                "delta": pair.delta,
            }
        )
    return [], resolved_pairs


def _resolve_explicit_overlay_pairs(gnss_slices, scenes: list[GoldScene], overlay_spec: OverlaySpec) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    slice_lookup = {_as_utc_naive(slice_data.timestamp): slice_data for slice_data in gnss_slices}
    scene_lookup = {
        (_as_utc_naive(scene.cha_time), _as_utc_naive(scene.chb_time)): scene
        for scene in scenes
        if scene.cha_time is not None and scene.chb_time is not None
    }

    rows: list[dict[str, object]] = []
    resolved_pairs: list[dict[str, object]] = []
    for pair_spec in overlay_spec.pairs:
        gnss_time = _as_utc_naive(pair_spec.gnss_time)
        gold_key = (_as_utc_naive(pair_spec.gold_cha_time), _as_utc_naive(pair_spec.gold_chb_time))
        slice_data = slice_lookup.get(gnss_time)
        scene = scene_lookup.get(gold_key)
        if slice_data is None or scene is None:
            rows.append(
                {
                    "overlay_name": overlay_spec.name,
                    "metric": overlay_spec.metric,
                    "gnss_time": pair_spec.gnss_time,
                    "gold_time": _overlay_pair_midpoint(pair_spec.gold_cha_time, pair_spec.gold_chb_time),
                    "status": "skipped",
                    "detail": _explicit_pair_missing_detail(slice_data is None, scene is None),
                }
            )
            continue
        resolved_pairs.append(
            {
                "slice_data": slice_data,
                "scene": scene,
                "delta": abs(_as_utc_naive(slice_data.timestamp) - _as_utc_naive(scene.midpoint)),
            }
        )
    return rows, resolved_pairs


def _mask_overlay_values(values, threshold: float):
    masked = np.ma.masked_invalid(np.asarray(values, dtype=float))
    keep = masked > threshold
    return np.ma.masked_where(~keep, masked)


def build_overlay_payload(slice_data, overlay_spec: OverlaySpec, event_spec: EventSpec) -> dict[str, object]:
    plot_extent = event_spec.map_extent()
    lat, lon, thresholded = _crop_overlay_grid(
        np.asarray(slice_data.lat, dtype=float),
        np.asarray(slice_data.lon, dtype=float),
        _mask_overlay_values(slice_data.values, overlay_spec.threshold),
        plot_extent,
    )
    count = int(np.ma.count(thresholded))
    native_step = _estimate_native_grid_step(lat, lon)
    if overlay_spec.bin_size_deg <= native_step + 1e-9:
        return {
            "lat": lat,
            "lon": lon,
            "values": thresholded,
            "metric": overlay_spec.metric,
            "color": overlay_spec.color,
            "count": count,
            "extent": plot_extent,
            "draw_mode": "grid",
            "legend_label": metric_threshold_label(overlay_spec.metric, overlay_spec.threshold, count),
            "title": "",
            "ylabel": overlay_ylabel(overlay_spec.metric),
        }

    lon_points, lat_points = _aggregate_overlay_points(
        lon,
        lat,
        thresholded,
        plot_extent,
        overlay_spec.bin_size_deg,
    )
    count = int(len(lon_points))
    return {
        "lat": lat,
        "lon": lon,
        "values": thresholded,
        "lon_points": lon_points,
        "lat_points": lat_points,
        "metric": overlay_spec.metric,
        "color": overlay_spec.color,
        "marker": "s",
        "size": 36.0,
        "count": count,
        "extent": plot_extent,
        "legend_label": metric_threshold_label(overlay_spec.metric, overlay_spec.threshold, count),
        "title": "",
        "ylabel": overlay_ylabel(overlay_spec.metric),
    }


def _crop_overlay_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ma.MaskedArray,
    extent: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    west, east, south, north = extent
    lon_mask = (lon >= west) & (lon <= east)
    lat_mask = (lat >= south) & (lat <= north)
    if not np.any(lon_mask) or not np.any(lat_mask):
        empty = np.asarray([], dtype=float)
        return empty, empty, np.ma.masked_all((0, 0), dtype=float)
    cropped = np.ma.asarray(values)[np.ix_(lat_mask, lon_mask)]
    return lat[lat_mask], lon[lon_mask], cropped


def _aggregate_overlay_points(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ma.MaskedArray,
    extent: tuple[float, float, float, float],
    bin_size_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    if lon.size == 0 or lat.size == 0 or np.ma.count(values) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    mask = ~np.ma.getmaskarray(values)
    if not np.any(mask):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    lon_values = np.asarray(lon_grid[mask], dtype=float)
    lat_values = np.asarray(lat_grid[mask], dtype=float)
    if bin_size_deg <= 0:
        return lon_values, lat_values

    west, _, south, _ = extent
    binned: dict[tuple[int, int], tuple[float, float]] = {}
    x_index = np.floor((lon_values - west) / bin_size_deg).astype(int)
    y_index = np.floor((lat_values - south) / bin_size_deg).astype(int)
    for xi, yi in zip(x_index.tolist(), y_index.tolist()):
        binned.setdefault(
            (xi, yi),
            (
                west + (xi + 0.5) * bin_size_deg,
                south + (yi + 0.5) * bin_size_deg,
            ),
        )
    if not binned:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    points = np.asarray(list(binned.values()), dtype=float)
    return points[:, 0], points[:, 1]


def _estimate_native_grid_step(lat: np.ndarray, lon: np.ndarray) -> float:
    steps: list[float] = []
    if lat.size > 1:
        lat_diff = np.diff(lat)
        lat_diff = np.abs(lat_diff[np.isfinite(lat_diff) & (lat_diff != 0)])
        if lat_diff.size:
            steps.append(float(np.nanmedian(lat_diff)))
    if lon.size > 1:
        lon_diff = np.diff(lon)
        lon_diff = np.abs(lon_diff[np.isfinite(lon_diff) & (lon_diff != 0)])
        if lon_diff.size:
            steps.append(float(np.nanmedian(lon_diff)))
    if not steps:
        return 0.5
    return min(steps)


def _as_utc_naive(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        return stamp.tz_convert(None)
    return stamp


def _overlay_pair_midpoint(left_time, right_time) -> pd.Timestamp:
    left = pd.Timestamp(left_time)
    right = pd.Timestamp(right_time)
    return left + (right - left) / 2


def _explicit_pair_missing_detail(missing_gnss: bool, missing_gold: bool) -> str:
    if missing_gnss and missing_gold:
        return "explicit_pair_missing_gnss_and_gold"
    if missing_gnss:
        return "explicit_pair_missing_gnss"
    return "explicit_pair_missing_gold"
