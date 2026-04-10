from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from ..models import GnssGridProduct


def normalize_internal_products(event_id: str, paths: list[Path]) -> list[GnssGridProduct]:
    products: list[GnssGridProduct] = []
    for path in sorted(paths):
        with xr.open_dataset(path) as dataset:
            metrics = tuple(_internal_metrics(dataset))
            times = pd.to_datetime(dataset["time"].values, utc=True)
            products.append(
                GnssGridProduct(
                    event_id=event_id,
                    producer="internal",
                    source_kind="gnss_grid",
                    path=path,
                    metrics=metrics,
                    time_start=times.min().to_pydatetime(),
                    time_end=times.max().to_pydatetime(),
                    metadata={"path_name": path.name},
                )
            )
    return products


def normalize_isee_products(event_id: str, paths: list[Path]) -> list[GnssGridProduct]:
    products: list[GnssGridProduct] = []
    for path in sorted(paths):
        with xr.open_dataset(path) as dataset:
            metric = _detect_isee_metric(dataset, path)
            times = pd.to_datetime(dataset["time"].values, utc=True)
            products.append(
                GnssGridProduct(
                    event_id=event_id,
                    producer="isee",
                    source_kind="gnss_grid",
                    path=path,
                    metrics=(metric,),
                    time_start=times.min().to_pydatetime(),
                    time_end=times.max().to_pydatetime(),
                    metadata={"path_name": path.name},
                )
            )
    return products


def _internal_metrics(dataset: xr.Dataset) -> list[str]:
    metrics: list[str] = []
    for candidate in ("vtec", "roti"):
        if candidate in dataset.data_vars:
            metrics.append(candidate)
    return metrics


def _detect_isee_metric(dataset: xr.Dataset, path: Path) -> str:
    lower_map = {name.lower(): name for name in dataset.data_vars}
    for candidate in ("roti", "vtec", "tec", "atec"):
        if candidate in lower_map:
            return "roti" if candidate == "roti" else "vtec"
    if "roti" in path.name.lower():
        return "roti"
    return "vtec"
