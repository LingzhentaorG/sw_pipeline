from __future__ import annotations

from ..discovery import discover_isee_grid_assets
from ..models import EventSpec, GnssGridProduct, SourceAsset
from ..providers.gnss_grid import fetch_gnss_grid_assets, process_gnss_grid_assets
from ..registry.manifests import read_gnss_grid_products


def fetch_gnss_grid(event_spec: EventSpec) -> list[SourceAsset]:
    return fetch_gnss_grid_assets(event_spec)


def process_gnss_grid(event_spec: EventSpec) -> list[GnssGridProduct]:
    manifest_path = event_spec.storage.manifests_dir / "gnss_grid_products.csv"
    products = [product for product in read_gnss_grid_products(manifest_path) if product.producer == "isee"]
    if _products_are_current(event_spec, products, "isee") and discover_isee_grid_assets(event_spec):
        return products
    return process_gnss_grid_assets(event_spec)


def _products_are_current(event_spec: EventSpec, products: list[GnssGridProduct], producer: str) -> bool:
    expected_root = (event_spec.storage.grids_dir / producer).resolve()
    for product in products:
        if not product.path.exists():
            return False
        if not product.path.resolve().is_relative_to(expected_root):
            return False
    return bool(products)
