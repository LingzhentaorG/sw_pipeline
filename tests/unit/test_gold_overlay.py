from __future__ import annotations

import numpy as np

from sw_pipeline.renderers.gold_map import _resolve_overlay_points
from sw_pipeline.renderers.overlay import _aggregate_overlay_points, _estimate_native_grid_step, _mask_overlay_values


def test_resolve_overlay_points_preserves_threshold_mask():
    values = np.ma.masked_where(
        np.array([[0.5, 2.0], [1.0, 3.0]], dtype=float) <= 1.5,
        np.array([[0.5, 2.0], [1.0, 3.0]], dtype=float),
    )
    lon_points, lat_points = _resolve_overlay_points(
        {
            "lat": np.array([10.0, 20.0]),
            "lon": np.array([-70.0, -60.0]),
            "values": values,
        }
    )

    assert lon_points.tolist() == [-60.0, -60.0]
    assert lat_points.tolist() == [10.0, 20.0]


def test_aggregate_overlay_points_merges_half_degree_cells_into_one_degree_bins():
    values = np.ma.masked_where(
        np.array(
            [
                [False, False],
                [False, False],
            ]
        ),
        np.ones((2, 2), dtype=float),
    )
    lon_points, lat_points = _aggregate_overlay_points(
        np.array([-70.0, -69.5], dtype=float),
        np.array([10.0, 10.5], dtype=float),
        values,
        (-150.0, 10.0, -80.0, 80.0),
        1.0,
    )

    assert lon_points.tolist() == [-69.5]
    assert lat_points.tolist() == [10.5]


def test_estimate_native_grid_step_prefers_half_degree_spacing():
    step = _estimate_native_grid_step(
        np.array([10.0, 10.5, 11.0], dtype=float),
        np.array([-70.0, -69.5, -69.0], dtype=float),
    )

    assert step == 0.5


def test_mask_overlay_values_uses_strict_greater_than_threshold():
    masked = _mask_overlay_values(
        np.array([[1.0, 1.01], [0.99, 1.5]], dtype=float),
        1.0,
    )

    assert np.ma.count(masked) == 2
    assert masked.mask.tolist() == [[True, False], [True, False]]
