from __future__ import annotations

import numpy as np
import xarray as xr

from sw_pipeline.internal.gnss_core import processing as processing_module


def test_gps_broadcast_store_loads_only_gps_navigation(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_load(path, *args, **kwargs):
        calls.append({"path": path, **kwargs})
        return xr.Dataset(
            {"sqrtA": (("time", "sv"), np.array([[1.0]], dtype=float))},
            coords={"time": np.array(["2024-10-10T00:00:00"], dtype="datetime64[ns]"), "sv": ["G01"]},
        )

    monkeypatch.setattr(processing_module.georinex, "load", fake_load)

    processing_module.GPSBroadcastStore("nav.rnx.gz", 12.0)

    assert calls == [{"path": "nav.rnx.gz", "use": "G", "fast": True}]
