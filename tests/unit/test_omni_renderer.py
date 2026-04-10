from __future__ import annotations

import numpy as np
import pytest

from sw_pipeline.config import load_app_config
from sw_pipeline.renderers.omni_series import _kp_bar_colors

from tests.fixtures.helpers import write_yaml


def test_kp_bar_colors_follow_storm_thresholds():
    colors = _kp_bar_colors(np.array([2.0, 4.33, 7.0, np.nan], dtype=float))

    assert colors == ["#2e8b57", "#f59e0b", "#c1121f", "#6b7280"]


def test_load_app_config_rejects_invalid_omni_highlight_alpha(tmp_path):
    base_path = tmp_path / "base.yaml"
    event_path = tmp_path / "event.yaml"
    write_yaml(
        base_path,
        {
            "paths": {"storage_root": str(tmp_path / "storage")},
            "bbox": {"lon_min": -150, "lon_max": 10, "lat_min": -80, "lat_max": 80},
            "runtime": {"gold_max_pair_minutes": 5},
        },
    )
    write_yaml(
        event_path,
        {
            "event": {"id": "bad_omni_alpha", "start": "2024-10-10T00:00:00Z", "end": "2024-10-10T01:00:00Z"},
            "sources": {
                "gnss_raw": {"enabled": False},
                "gnss_grid": {"enabled": False},
                "gold": {"enabled": False},
                "omni": {"enabled": True},
            },
            "products": {"gnss_grid": {"map_producers": ["isee"]}},
            "figures": {
                "overlays": [],
                "station_series": [],
                "omni_series": {
                    "highlight_windows": [
                        {
                            "start": "2024-10-10T00:10:00Z",
                            "end": "2024-10-10T00:40:00Z",
                            "alpha": 1.2,
                        }
                    ]
                },
            },
            "runtime": {},
        },
    )

    with pytest.raises(ValueError, match="figures.omni_series.highlight_windows\\[\\].alpha"):
        load_app_config("bad_omni_alpha", base_path, event_path)
