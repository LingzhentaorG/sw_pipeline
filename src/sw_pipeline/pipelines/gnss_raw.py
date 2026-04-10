from __future__ import annotations

from ..models import EventSpec, GnssGridProduct, SourceAsset
from ..providers.gnss_raw import fetch_gnss_raw_assets, process_gnss_raw_assets
from ..registry.manifests import read_gnss_grid_products


def fetch_gnss_raw(event_spec: EventSpec) -> list[SourceAsset]:
    return fetch_gnss_raw_assets(event_spec)


def process_gnss_raw(event_spec: EventSpec) -> list[GnssGridProduct]:
    manifest_path = event_spec.storage.manifests_dir / "gnss_grid_products.csv"
    products = [product for product in read_gnss_grid_products(manifest_path) if product.producer == "internal"]
    if _products_are_current(event_spec, products, "internal"):
        return products
    return process_gnss_raw_assets(event_spec)


def resolve_internal_workspace_root(event_spec: EventSpec):
    return event_spec.storage.gnss_workspace_dir if event_spec.internal_gnss_mode() == "internal_pipeline" else event_spec.internal_gnss_workspace_root()


def _products_are_current(event_spec: EventSpec, products: list[GnssGridProduct], producer: str) -> bool:
    expected_root = (event_spec.storage.grids_dir / producer).resolve()
    for product in products:
        if not product.path.exists():
            return False
        if not product.path.resolve().is_relative_to(expected_root):
            return False
    return bool(products)
