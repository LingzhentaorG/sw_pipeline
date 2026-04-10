from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


def daterange_days(start_utc, end_utc) -> tuple[pd.Timestamp, ...]:
    start = pd.Timestamp(start_utc).floor("D")
    end = pd.Timestamp(end_utc).floor("D")
    return tuple(pd.date_range(start, end, freq="D"))


def stage_local_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def manifest_to_records(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def file_asset_id(path: Path) -> str:
    digest = hashlib.md5(str(path).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"{path.stem}-{digest}"


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def list_existing_paths(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def ensure_utc_naive(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value, tz="UTC")
    return stamp.tz_convert(None)


def read_partitioned_parquet(root: Path, stem: str) -> pd.DataFrame:
    full_path = root / f"{stem}.parquet"
    if full_path.exists():
        return pd.read_parquet(full_path)

    part_paths = sorted(root.glob(f"{stem}.part*.parquet"))
    if not part_paths:
        raise FileNotFoundError(f"No parquet data found for {stem} under {root}")
    frames = [pd.read_parquet(path) for path in part_paths]
    return pd.concat(frames, ignore_index=True)


def glob_event_netcdf(netcdf_dir: Path, event_id: str) -> list[Path]:
    return sorted(netcdf_dir.glob(f"{event_id}*.nc"))
