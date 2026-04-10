from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sw_pipeline.config import load_app_config
from sw_pipeline.models import GoldScene
from sw_pipeline.renderers import overlay as overlay_module

from tests.fixtures.helpers import write_yaml


def test_render_overlays_prefers_explicit_pairs(tmp_path, monkeypatch):
    base_path = tmp_path / "base.yaml"
    event_path = tmp_path / "overlay_evt.yaml"
    write_yaml(
        base_path,
        {
            "paths": {"storage_root": str(tmp_path / "storage")},
            "bbox": {"lon_min": -150, "lon_max": 10, "lat_min": -80, "lat_max": 80},
            "runtime": {"gold_max_pair_minutes": 5},
            "plot_defaults": {
                "use_cartopy": False,
                "gnss_styles": {
                    "vtec": {"cmap": "viridis", "vmin": 0, "vmax": 80},
                    "roti": {"cmap": "viridis", "vmin": 0, "vmax": 1},
                },
            },
        },
    )
    write_yaml(
        event_path,
        {
            "event": {
                "id": "overlay_evt",
                "start": "2024-10-10T00:00:00Z",
                "end": "2024-10-10T23:59:59Z",
            },
            "sources": {
                "gnss_raw": {"enabled": False},
                "gnss_grid": {"enabled": False},
                "gold": {"enabled": False},
                "omni": {"enabled": False},
            },
            "products": {"gnss_grid": {"map_producers": ["isee"]}},
            "figures": {
                "overlays": [
                    {
                        "name": "roti_on_gold",
                        "threshold": 1.0,
                        "producer": "isee",
                        "pairs": [
                            {
                                "gold_cha_time": "2024-10-10T00:09:00Z",
                                "gold_chb_time": "2024-10-10T00:11:00Z",
                                "gnss_time": "2024-10-10T00:10:00Z",
                            },
                            {
                                "gold_cha_time": "2024-10-10T00:21:00Z",
                                "gold_chb_time": "2024-10-10T00:24:00Z",
                                "gnss_time": "2024-10-10T00:20:00Z",
                            },
                        ],
                    }
                ],
                "station_series": [],
            },
        },
    )

    spec = load_app_config("overlay_evt", base_path, event_path)
    overlay_spec = spec.overlay_specs()[0]
    slices = [
        SimpleNamespace(
            timestamp=pd.Timestamp("2024-10-10T00:10:00Z"),
            lat=np.array([0.0], dtype=float),
            lon=np.array([-70.0], dtype=float),
            values=np.array([[1.2]], dtype=float),
        ),
        SimpleNamespace(
            timestamp=pd.Timestamp("2024-10-10T00:20:00Z"),
            lat=np.array([0.0], dtype=float),
            lon=np.array([-70.0], dtype=float),
            values=np.array([[1.4]], dtype=float),
        ),
        SimpleNamespace(
            timestamp=pd.Timestamp("2024-10-10T00:25:00Z"),
            lat=np.array([0.0], dtype=float),
            lon=np.array([-70.0], dtype=float),
            values=np.array([[1.6]], dtype=float),
        ),
    ]
    scenes = [
        GoldScene(
            event_id=spec.event_id,
            tar_path=tmp_path / "gold_scene.tar",
            midpoint=datetime(2024, 10, 10, 0, 10, tzinfo=UTC),
            cha_member="CHA_001.nc",
            chb_member="CHB_001.nc",
            cha_time=datetime(2024, 10, 10, 0, 9, tzinfo=UTC),
            chb_time=datetime(2024, 10, 10, 0, 11, tzinfo=UTC),
            delta_minutes=2.0,
        ),
        GoldScene(
            event_id=spec.event_id,
            tar_path=tmp_path / "gold_scene.tar",
            midpoint=datetime(2024, 10, 10, 0, 22, 30, tzinfo=UTC),
            cha_member="CHA_002.nc",
            chb_member="CHB_002.nc",
            cha_time=datetime(2024, 10, 10, 0, 21, tzinfo=UTC),
            chb_time=datetime(2024, 10, 10, 0, 24, tzinfo=UTC),
            delta_minutes=3.0,
        ),
    ]

    monkeypatch.setattr(overlay_module, "iter_gnss_slices", lambda products, metric: slices)
    monkeypatch.setattr(overlay_module, "prepare_gnss_slice", lambda slice_data, event_spec: slice_data)

    def fake_render_gold_scene(event_spec, scene, output_path, overlay=None):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(scene.cha_member, encoding="utf-8")

    monkeypatch.setattr(overlay_module, "render_gold_scene", fake_render_gold_scene)

    outputs = overlay_module.render_overlays(spec, [], scenes, overlay_spec)

    assert len(outputs) == 2
    assert {path.name for path in outputs} == {
        "roti_on_gold_20241010T0010Z.png",
        "roti_on_gold_20241010T0020Z.png",
    }
    manifest = pd.read_csv(spec.storage.manifests_dir / "roti_on_gold_pairs.csv")
    rendered = manifest[manifest["status"] == "rendered"]
    assert len(rendered) == 2
    assert not rendered["gnss_time"].astype(str).str.contains("00:25").any()
