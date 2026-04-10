from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sw_pipeline import app as app_module


def test_run_event_skips_fetch_by_default(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(app_module, "fetch_target", lambda *args, **kwargs: calls.append("fetch"))
    monkeypatch.setattr(app_module, "process_target", lambda *args, **kwargs: calls.append("process"))
    monkeypatch.setattr(app_module, "plot_target", lambda *args, **kwargs: calls.append("plot"))
    monkeypatch.setattr(app_module, "write_stage_status", lambda *args, **kwargs: None)

    class StubSpec:
        def __init__(self) -> None:
            self.sources = {
                "gnss_raw": {"enabled": True},
                "gnss_grid": {"enabled": True},
                "gold": {"enabled": True},
                "omni": {"enabled": True},
            }
            self.figures = {"gnss_maps": {"metrics": ["vtec"]}}
            self.storage = SimpleNamespace(manifests_dir=Path("."))

        def station_series_presets(self):
            return ()

        def gnss_map_producers(self):
            return ("internal", "isee")

        def overlay_specs(self):
            return ()

        def panel_specs(self):
            return ()

    app_module.run_event(StubSpec())

    assert "fetch" not in calls


def test_run_event_can_include_fetch(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(app_module, "fetch_target", lambda *args, **kwargs: calls.append("fetch"))
    monkeypatch.setattr(app_module, "process_target", lambda *args, **kwargs: calls.append("process"))
    monkeypatch.setattr(app_module, "plot_target", lambda *args, **kwargs: calls.append("plot"))
    monkeypatch.setattr(app_module, "write_stage_status", lambda *args, **kwargs: None)

    class StubSpec:
        def __init__(self) -> None:
            self.sources = {
                "gnss_raw": {"enabled": True},
                "gnss_grid": {"enabled": False},
                "gold": {"enabled": False},
                "omni": {"enabled": False},
            }
            self.figures = {"gnss_maps": {"metrics": ["vtec"]}}
            self.storage = SimpleNamespace(manifests_dir=Path("."))

        def station_series_presets(self):
            return ()

        def gnss_map_producers(self):
            return ("internal",)

        def overlay_specs(self):
            return ()

        def panel_specs(self):
            return ()

    app_module.run_event(StubSpec(), include_fetch=True)

    assert "fetch" in calls


def test_process_target_runs_internal_when_station_series_exists(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(app_module, "process_gnss_raw", lambda event_spec: calls.append("internal") or ["internal"])
    monkeypatch.setattr(app_module, "process_gnss_grid", lambda event_spec: calls.append("isee") or ["isee"])

    class StubSpec:
        def station_series_presets(self):
            return ("boav",)

        def gnss_map_producers(self):
            return ("isee",)

    products = app_module.process_target(StubSpec(), "gnss")

    assert calls == ["internal", "isee"]
    assert products == ["internal", "isee"]
