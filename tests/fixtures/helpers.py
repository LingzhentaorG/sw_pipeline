from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from netCDF4 import Dataset


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def create_isee_gnss_root(root: Path, timestamp: str = "2024-10-10T20:10:00Z") -> Path:
    lat = np.array([-10.0, 0.0, 10.0], dtype=float)
    lon = np.array([-80.0, -70.0, -60.0], dtype=float)
    time = pd.to_datetime([timestamp, "2024-10-10T20:25:00Z"], utc=True).tz_convert(None)
    values = np.arange(18, dtype=float).reshape(2, 3, 3)

    vtec_dir = root / "VTEC_data" / "2024" / "284"
    roti_dir = root / "ROTI_data" / "2024" / "284"
    vtec_dir.mkdir(parents=True, exist_ok=True)
    roti_dir.mkdir(parents=True, exist_ok=True)

    xr.Dataset(
        {"atec": (("time", "lat", "lon"), values + 10)},
        coords={"time": time, "lat": lat, "lon": lon},
    ).to_netcdf(vtec_dir / "2024101020_atec.nc")
    xr.Dataset(
        {"roti": (("time", "lat", "lon"), values / 5.0)},
        coords={"time": time, "lat": lat, "lon": lon},
    ).to_netcdf(roti_dir / "2024101020_roti.nc")
    return root


def create_gold_tarball(root: Path, midpoint: str = "2024-10-10T20:10:00Z") -> Path:
    midpoint_ts = pd.Timestamp(midpoint, tz="UTC")
    year = midpoint_ts.strftime("%Y")
    doy = midpoint_ts.strftime("%j")
    hour = midpoint_ts.strftime("%H")
    minute = midpoint_ts.strftime("%M")
    tar_path = root / "gold_scene.tar"
    temp_dir = root / "gold_nc"
    temp_dir.mkdir(parents=True, exist_ok=True)

    created_files: list[tuple[Path, str]] = []
    for hemisphere in ("CHA", "CHB"):
        nc_path = temp_dir / f"{hemisphere}.nc"
        with Dataset(nc_path, "w") as dataset:
            dataset.createDimension("y", 2)
            dataset.createDimension("x", 2)
            dataset.createDimension("w", 1)

            lat_var = dataset.createVariable("REFERENCE_POINT_LAT", "f4", ("y", "x"))
            lon_var = dataset.createVariable("REFERENCE_POINT_LON", "f4", ("y", "x"))
            wavelength_var = dataset.createVariable("WAVELENGTH", "f4", ("y", "x", "w"))
            radiance_var = dataset.createVariable("RADIANCE", "f4", ("y", "x", "w"))
            quality_var = dataset.createVariable("QUALITY_FLAG", "u4", ("y", "x"))
            emission_var = dataset.createVariable("EMISSION_ANGLE", "f4", ("y", "x"))

            lat_var[:] = np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)
            lon_var[:] = np.array([[-70.0, -66.0], [-62.0, -58.0]], dtype=np.float32)
            wavelength_var[:] = np.full((2, 2, 1), 135.6, dtype=np.float32)
            radiance_var[:] = np.full((2, 2, 1), 100.0 if hemisphere == "CHA" else 130.0, dtype=np.float32)
            quality_var[:] = np.zeros((2, 2), dtype=np.uint32)
            emission_var[:] = np.full((2, 2), 45.0, dtype=np.float32)

        member_name = f"tmp/archive_L1C/{year}/{doy}/GOLD_L1C_{hemisphere}_NI1_{year}_{doy}_{hour}_{minute}_v05_r01_c01.nc"
        created_files.append((nc_path, member_name))

    with tarfile.open(tar_path, "w") as archive:
        for nc_path, member_name in created_files:
            archive.add(nc_path, arcname=member_name)
    return tar_path


def create_local_omni_files(root: Path, event_id: str) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    bz_path = root / f"omni_bz_1min_{event_id}.csv"
    hourly_path = root / f"omni_dst_kp_hourly_{event_id}.csv"
    kp_path = root / f"omni_kp_3hour_{event_id}.csv"

    pd.DataFrame(
        {
            "Time": ["2024-10-10T20:00:00Z", "2024-10-10T20:01:00Z", "2024-10-10T20:02:00Z"],
            "IMF_Bz_nT": [-5.0, -3.0, 1.0],
        }
    ).to_csv(bz_path, index=False)

    pd.DataFrame(
        {
            "Time": ["2024-10-10T20:00:00Z"],
            "Kp_code": [43],
            "Dst_nT": [-90],
            "PlotTime": ["2024-10-10T19:30:00Z"],
            "Kp": [4.3333333333],
        }
    ).to_csv(hourly_path, index=False)

    pd.DataFrame(
        {
            "KpStart": ["2024-10-10T18:00:00Z"],
            "KpEnd": ["2024-10-10T21:00:00Z"],
            "Kp": [4.3333333333],
        }
    ).to_csv(kp_path, index=False)
    return bz_path, hourly_path, kp_path


def create_internal_workspace(root: Path, event_id: str) -> Path:
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "event_id": event_id,
                "station_id": "41636M001",
                "station_code4": "BOAV",
                "obs_path": str(root / "dummy.obs"),
            }
        ]
    ).to_csv(manifests / "normalized_manifest.csv", index=False)

    vtec_dir = root / "intermediate" / "vtec"
    roti_dir = root / "intermediate" / "roti"
    vtec_dir.mkdir(parents=True, exist_ok=True)
    roti_dir.mkdir(parents=True, exist_ok=True)

    times = pd.date_range("2024-10-10T23:00:00Z", periods=8, freq="30min", tz="UTC")
    rows_vtec = []
    rows_roti = []
    for satellite in ("G21", "G02"):
        for index, timestamp in enumerate(times):
            rows_vtec.append({"time": timestamp, "station_id": "41636M001", "sv": satellite, "vtec": 10 + index})
            rows_roti.append({"time": timestamp, "station_id": "41636M001", "sv": satellite, "roti": index / 10})

    pd.DataFrame(rows_vtec).to_parquet(vtec_dir / f"{event_id}.parquet", index=False)
    pd.DataFrame(rows_roti).to_parquet(roti_dir / f"{event_id}.parquet", index=False)

    product_dir = root / "products" / "netcdf"
    product_dir.mkdir(parents=True, exist_ok=True)
    grid_time = pd.to_datetime(["2024-10-10T23:00:00Z", "2024-10-10T23:30:00Z"], utc=True).tz_convert(None)
    lat = np.array([-10.0, 0.0, 10.0], dtype=float)
    lon = np.array([-80.0, -70.0, -60.0], dtype=float)
    xr.Dataset(
        {
            "vtec": (("time", "lat", "lon"), np.full((2, 3, 3), 12.0, dtype=float)),
            "roti": (("time", "lat", "lon"), np.full((2, 3, 3), 0.6, dtype=float)),
        },
        coords={"time": grid_time, "lat": lat, "lon": lon},
    ).to_netcdf(product_dir / f"{event_id}_20241010_2300.nc")
    return root
