from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from sw_pipeline.config import load_app_config
from sw_pipeline.renderers import panel as panel_module

from tests.fixtures.helpers import write_yaml


def _build_event_spec(tmp_path, panels: list[dict], *, use_cartopy: bool = False):
    base_path = tmp_path / "base.yaml"
    event_path = tmp_path / "event.yaml"
    write_yaml(
        base_path,
        {
            "paths": {"storage_root": str(tmp_path / "storage")},
            "bbox": {"lon_min": -150, "lon_max": 10, "lat_min": -80, "lat_max": 80},
            "runtime": {"gold_max_pair_minutes": 5},
            "plot_defaults": {
                "use_cartopy": use_cartopy,
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
                "id": "panel_evt",
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
                "overlays": [],
                "station_series": [],
                "panels": panels,
            },
            "runtime": {},
        },
    )
    return load_app_config("panel_evt", base_path, event_path)


def _gnss_slot(timestamp: str, title: str, *, kind: str = "gnss_roti") -> dict[str, object]:
    return {
        "kind": kind,
        "producer": "isee",
        "timestamp": timestamp,
        "title": title,
    }


def test_load_app_config_parses_panel_specs(tmp_path):
    spec = _build_event_spec(
        tmp_path,
        [
            {
                "name": "sample_panel",
                "layout": {"rows": 1, "cols": 1},
                "shared_colorbar": "gnss_roti",
                "slots": [_gnss_slot("2024-10-10T00:10:00Z", "2024-10-10 00:10 UTC")],
            }
        ],
    )

    panel_spec = spec.panel_specs()[0]
    assert panel_spec.name == "sample_panel"
    assert panel_spec.rows == 1
    assert panel_spec.cols == 1
    assert panel_spec.shared_colorbar == "gnss_roti"
    assert panel_spec.slots[0].kind == "gnss_roti"
    assert panel_spec.slots[0].producer == "isee"
    assert str(panel_spec.slots[0].timestamp) == "2024-10-10 00:10:00+00:00"


def test_load_app_config_rejects_missing_panel_slot_title(tmp_path):
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
            "event": {"id": "bad_panel", "start": "2024-10-10T00:00:00Z", "end": "2024-10-10T01:00:00Z"},
            "sources": {
                "gnss_raw": {"enabled": False},
                "gnss_grid": {"enabled": False},
                "gold": {"enabled": False},
                "omni": {"enabled": False},
            },
            "products": {"gnss_grid": {"map_producers": ["isee"]}},
            "figures": {
                "overlays": [],
                "station_series": [],
                "panels": [
                    {
                        "name": "bad_panel",
                        "layout": {"rows": 1, "cols": 1},
                        "shared_colorbar": "gnss_roti",
                        "slots": [{"kind": "gnss_roti", "producer": "isee", "timestamp": "2024-10-10T00:10:00Z"}],
                    }
                ],
            },
            "runtime": {},
        },
    )

    with pytest.raises(ValueError, match="figures.panels\\[\\].slots\\[\\].title"):
        load_app_config("bad_panel", base_path, event_path)


def test_load_app_config_rejects_invalid_panel_kind(tmp_path):
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
            "event": {"id": "bad_panel_kind", "start": "2024-10-10T00:00:00Z", "end": "2024-10-10T01:00:00Z"},
            "sources": {
                "gnss_raw": {"enabled": False},
                "gnss_grid": {"enabled": False},
                "gold": {"enabled": False},
                "omni": {"enabled": False},
            },
            "products": {"gnss_grid": {"map_producers": ["isee"]}},
            "figures": {
                "overlays": [],
                "station_series": [],
                "panels": [
                    {
                        "name": "bad_panel",
                        "layout": {"rows": 1, "cols": 1},
                        "shared_colorbar": "gnss_roti",
                        "slots": [{"kind": "bad", "title": "oops"}],
                    }
                ],
            },
            "runtime": {},
        },
    )

    with pytest.raises(ValueError, match="figures.panels\\[\\].slots\\[\\].kind"):
        load_app_config("bad_panel_kind", base_path, event_path)


def test_build_panel_figure_supports_target_layouts(tmp_path):
    spec = _build_event_spec(
        tmp_path,
        [
            {
                "name": "panel_2x3",
                "layout": {"rows": 2, "cols": 3},
                "shared_colorbar": "gnss_roti",
                "slots": [_gnss_slot("2024-10-10T00:10:00Z", f"slot {index}") for index in range(6)],
            },
            {
                "name": "panel_3x3",
                "layout": {"rows": 3, "cols": 3},
                "shared_colorbar": "gnss_roti",
                "slots": [_gnss_slot("2024-10-10T00:10:00Z", f"slot {index}") for index in range(9)],
            },
            {
                "name": "panel_3x2",
                "layout": {"rows": 3, "cols": 2},
                "shared_colorbar": "gnss_roti",
                "slots": [_gnss_slot("2024-10-10T00:10:00Z", f"slot {index}") for index in range(6)],
            },
        ],
    )

    expected_shapes = [(2, 3), (3, 3), (3, 2)]
    for panel_spec, expected in zip(spec.panel_specs(), expected_shapes, strict=True):
        fig, axes = panel_module._build_panel_figure(spec, panel_spec)
        assert axes.shape == expected
        plt.close(fig)


def test_set_slot_title_applies_letter_prefix():
    fig, ax = plt.subplots()
    panel_module._set_slot_title(ax, "Example Panel", 5)
    assert ax.get_title() == "(f) Example Panel"
    plt.close(fig)


def test_render_panels_uses_single_shared_colorbar_for_roti_panel(tmp_path, monkeypatch):
    spec = _build_event_spec(
        tmp_path,
        [
            {
                "name": "roti_panel",
                "layout": {"rows": 2, "cols": 3},
                "shared_colorbar": "gnss_roti",
                "slots": [_gnss_slot("2024-10-10T00:10:00Z", f"ROTI {index}") for index in range(6)],
            }
        ],
    )

    dummy_slice = SimpleNamespace(
        metric="roti",
        timestamp=pd.Timestamp("2024-10-10T00:10:00Z"),
        lat=np.array([0.0], dtype=float),
        lon=np.array([0.0], dtype=float),
        values=np.array([[0.5]], dtype=float),
    )

    monkeypatch.setattr(panel_module._GnssSliceResolver, "resolve", lambda self, producer, metric, timestamp: dummy_slice)
    monkeypatch.setattr(panel_module, "plot_gnss_slice", lambda ax, slice_data, event_spec, **kwargs: ax.imshow([[0.5]], vmin=0, vmax=1, cmap="viridis"))

    colorbar_calls: list[int] = []
    original_colorbar = Figure.colorbar

    def counting_colorbar(self, *args, **kwargs):
        colorbar_calls.append(1)
        return original_colorbar(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "colorbar", counting_colorbar)

    outputs = panel_module.render_panels(spec, {"isee": []})

    assert len(outputs) == 1
    assert outputs[0].exists()
    assert len(colorbar_calls) == 1
    manifest = pd.read_csv(spec.storage.manifests_dir / "panel_outputs.csv")
    assert len(manifest) == 6
    assert set(manifest["status"]) == {"rendered"}


def test_render_panels_supports_single_shared_colorbar_for_vtec_panel(tmp_path, monkeypatch):
    spec = _build_event_spec(
        tmp_path,
        [
            {
                "name": "vtec_panel",
                "layout": {"rows": 2, "cols": 3},
                "shared_colorbar": "gnss_vtec",
                "slots": [_gnss_slot("2024-10-10T00:10:00Z", f"VTEC {index}", kind="gnss_vtec") for index in range(6)],
            }
        ],
    )

    dummy_slice = SimpleNamespace(
        metric="vtec",
        timestamp=pd.Timestamp("2024-10-10T00:10:00Z"),
        lat=np.array([0.0], dtype=float),
        lon=np.array([0.0], dtype=float),
        values=np.array([[25.0]], dtype=float),
    )

    resolver_calls: list[tuple[str, str, object]] = []
    colorbar_calls: list[int] = []
    original_colorbar = Figure.colorbar

    def fake_resolve(self, producer, metric, timestamp):
        resolver_calls.append((producer, metric, timestamp))
        return dummy_slice

    def counting_colorbar(self, *args, **kwargs):
        colorbar_calls.append(1)
        return original_colorbar(self, *args, **kwargs)

    monkeypatch.setattr(panel_module._GnssSliceResolver, "resolve", fake_resolve)
    monkeypatch.setattr(panel_module, "plot_gnss_slice", lambda ax, slice_data, event_spec, **kwargs: ax.imshow([[25.0]], vmin=0, vmax=80, cmap="viridis"))
    monkeypatch.setattr(Figure, "colorbar", counting_colorbar)

    outputs = panel_module.render_panels(spec, {"isee": []})

    assert len(outputs) == 1
    assert outputs[0].exists()
    assert len(colorbar_calls) == 1
    assert resolver_calls
    assert {metric for _, metric, _ in resolver_calls} == {"vtec"}


def test_render_panels_uses_single_shared_colorbar_for_gold_panel(tmp_path, monkeypatch):
    pytest.importorskip("cartopy", reason="cartopy required for GOLD panel rendering")
    spec = _build_event_spec(
        tmp_path,
        [
            {
                "name": "gold_panel",
                "layout": {"rows": 2, "cols": 3},
                "shared_colorbar": "gold",
                "slots": [
                    {
                        "kind": "gold",
                        "gold_cha_time": "2024-10-10T00:09:00Z",
                        "gold_chb_time": "2024-10-10T00:11:00Z",
                        "title": f"GOLD {index}",
                    }
                    for index in range(6)
                ],
            }
        ],
        use_cartopy=True,
    )

    dummy_pair = SimpleNamespace(
        midpoint=pd.Timestamp("2024-10-10T00:10:00Z").to_pydatetime(),
        cha=SimpleNamespace(obs_time=pd.Timestamp("2024-10-10T00:09:00Z").to_pydatetime()),
        chb=SimpleNamespace(obs_time=pd.Timestamp("2024-10-10T00:11:00Z").to_pydatetime()),
    )

    monkeypatch.setattr(panel_module._GoldPairResolver, "resolve", lambda self, cha_time, chb_time: dummy_pair)
    monkeypatch.setattr(
        panel_module,
        "plot_gold_pair",
        lambda ax, event_spec, pair, overlay=None, decorate_overlay=True, **kwargs: ax.scatter(
            [-70.0],
            [0.0],
            c=[100.0],
            cmap="viridis",
            vmin=0,
            vmax=300,
            transform=panel_module.ccrs.PlateCarree(),
        ),
    )

    colorbar_calls: list[int] = []
    original_colorbar = Figure.colorbar

    def counting_colorbar(self, *args, **kwargs):
        colorbar_calls.append(1)
        return original_colorbar(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "colorbar", counting_colorbar)

    outputs = panel_module.render_panels(spec, {})

    assert len(outputs) == 1
    assert outputs[0].exists()
    assert len(colorbar_calls) == 1


def test_render_panels_uses_single_shared_colorbar_for_overlay_panel(tmp_path, monkeypatch):
    pytest.importorskip("cartopy", reason="cartopy required for overlay panel rendering")
    spec = _build_event_spec(
        tmp_path,
        [
            {
                "name": "overlay_panel",
                "layout": {"rows": 3, "cols": 2},
                "shared_colorbar": "gold",
                "slots": [
                    {
                        "kind": "overlay" if index >= 4 else "gold" if index < 2 else "gnss_roti",
                        "gold_cha_time": "2024-10-10T00:09:00Z" if index in {0, 1, 4, 5} else None,
                        "gold_chb_time": "2024-10-10T00:11:00Z" if index in {0, 1, 4, 5} else None,
                        "producer": "isee" if index in {2, 3} else None,
                        "timestamp": "2024-10-10T00:10:00Z" if index in {2, 3} else None,
                        "gnss_timestamp": "2024-10-10T00:10:00Z" if index in {4, 5} else None,
                        "title": f"Slot {index}",
                    }
                    for index in range(6)
                ],
                "footer_note": "ROTI row uses viridis 0-1; no separate colorbar.",
            }
        ],
        use_cartopy=True,
    )

    dummy_pair = SimpleNamespace(
        midpoint=pd.Timestamp("2024-10-10T00:10:00Z").to_pydatetime(),
        cha=SimpleNamespace(obs_time=pd.Timestamp("2024-10-10T00:09:00Z").to_pydatetime()),
        chb=SimpleNamespace(obs_time=pd.Timestamp("2024-10-10T00:11:00Z").to_pydatetime()),
    )
    dummy_slice = SimpleNamespace(
        metric="roti",
        timestamp=pd.Timestamp("2024-10-10T00:10:00Z"),
        lat=np.array([0.0], dtype=float),
        lon=np.array([0.0], dtype=float),
        values=np.array([[0.5]], dtype=float),
    )

    monkeypatch.setattr(panel_module._GoldPairResolver, "resolve", lambda self, cha_time, chb_time: dummy_pair)
    monkeypatch.setattr(panel_module._GnssSliceResolver, "resolve", lambda self, producer, metric, timestamp: dummy_slice)
    monkeypatch.setattr(panel_module, "build_overlay_payload", lambda slice_data, overlay_spec, event_spec: {"extent": event_spec.map_extent(), "count": 1, "legend_label": "", "ylabel": ""})
    monkeypatch.setattr(
        panel_module,
        "plot_gold_pair",
        lambda ax, event_spec, pair, overlay=None, decorate_overlay=True, **kwargs: ax.scatter(
            [-70.0],
            [0.0],
            c=[120.0],
            cmap="viridis",
            vmin=0,
            vmax=300,
            transform=panel_module.ccrs.PlateCarree(),
        ),
    )
    monkeypatch.setattr(panel_module, "plot_gnss_slice", lambda ax, slice_data, event_spec, **kwargs: ax.scatter([-70.0], [0.0], c=[0.5], cmap="viridis", vmin=0, vmax=1, transform=panel_module.ccrs.PlateCarree()))

    colorbar_calls: list[int] = []
    original_colorbar = Figure.colorbar

    def counting_colorbar(self, *args, **kwargs):
        colorbar_calls.append(1)
        return original_colorbar(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "colorbar", counting_colorbar)

    outputs = panel_module.render_panels(spec, {"isee": []})

    assert len(outputs) == 1
    assert outputs[0].exists()
    assert len(colorbar_calls) == 2


def test_gold_pair_resolver_finds_explicit_pair_outside_event_days():
    spec = load_app_config("storm_20241010_11")
    resolver = panel_module._GoldPairResolver(spec)

    pair = resolver.resolve(
        pd.Timestamp("2024-10-12T00:21:00Z"),
        pd.Timestamp("2024-10-12T00:24:00Z"),
    )

    assert pair is not None
    assert pd.Timestamp(pair.cha.obs_time).strftime("%Y-%m-%d %H:%M") == "2024-10-12 00:21"
    assert pd.Timestamp(pair.chb.obs_time).strftime("%Y-%m-%d %H:%M") == "2024-10-12 00:24"
