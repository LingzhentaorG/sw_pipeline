from __future__ import annotations

import pandas as pd
import xarray as xr

from sw_pipeline.normalizers.gnss import normalize_internal_products, normalize_isee_products


def test_normalize_internal_and_isee_products(tmp_path):
    time = pd.to_datetime(["2024-10-10T20:10:00Z"], utc=True).tz_convert(None)
    lat = [0.0, 5.0]
    lon = [-70.0, -60.0]

    internal_path = tmp_path / "internal.nc"
    xr.Dataset(
        {
            "vtec": (("time", "lat", "lon"), [[[10.0, 11.0], [12.0, 13.0]]]),
            "roti": (("time", "lat", "lon"), [[[0.1, 0.2], [0.3, 0.4]]]),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    ).to_netcdf(internal_path)

    isee_path = tmp_path / "isee.nc"
    xr.Dataset(
        {"atec": (("time", "lat", "lon"), [[[20.0, 21.0], [22.0, 23.0]]])},
        coords={"time": time, "lat": lat, "lon": lon},
    ).to_netcdf(isee_path)

    internal = normalize_internal_products("evt", [internal_path])
    external = normalize_isee_products("evt", [isee_path])

    assert internal[0].metrics == ("vtec", "roti")
    assert external[0].metrics == ("vtec",)
