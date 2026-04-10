from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from sw_pipeline.internal import gold_core


def _create_gold_tarball_with_emission(root: Path) -> Path:
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

            lat_var[:] = np.array([[10.0, 72.0], [4.0, 6.0]], dtype=np.float32)
            lon_var[:] = np.array([[-70.0, -66.0], [-62.0, -58.0]], dtype=np.float32)
            wavelength_var[:] = np.full((2, 2, 1), 135.6, dtype=np.float32)
            radiance_var[:] = np.full((2, 2, 1), 100.0 if hemisphere == "CHA" else 130.0, dtype=np.float32)
            quality_var[:] = np.zeros((2, 2), dtype=np.uint32)
            emission_var[:] = np.array([[45.0, 89.5], [46.0, 47.0]], dtype=np.float32)

        member_name = f"tmp/archive_L1C/2024/284/GOLD_L1C_{hemisphere}_NI1_2024_284_23_21_v05_r01_c01.nc"
        created_files.append((nc_path, member_name))

    with tarfile.open(tar_path, "w") as archive:
        for nc_path, member_name in created_files:
            archive.add(nc_path, arcname=member_name)
    return tar_path


def test_read_geo_grid_masks_high_emission_angle_pixels_by_default(tmp_path):
    tar_path = _create_gold_tarball_with_emission(tmp_path)
    entry = next(item for item in gold_core.discover_entries(tar_path) if item.hemisphere == "CHA")

    _, _, masked = gold_core.read_geo_grid(entry, 135.6, "all")
    _, _, unmasked = gold_core.read_geo_grid(entry, 135.6, "all", max_emission_angle_deg=None)

    assert bool(masked.mask[0, 1])
    assert not bool(masked.mask[0, 0])
    assert not bool(unmasked.mask[0, 1])
