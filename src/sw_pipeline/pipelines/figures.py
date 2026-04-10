from __future__ import annotations

from ..discovery import discover_gold_scenes, discover_omni_series, discover_isee_grid_assets
from ..models import EventSpec
from ..registry.manifests import read_gnss_grid_products, read_gold_scenes, read_omni_series
from ..renderers.gnss_map import render_gnss_maps
from ..renderers.gold_map import render_gold_maps
from ..renderers.omni_series import render_omni_series
from ..renderers.overlay import render_overlays
from ..renderers.panel import render_panels
from ..renderers.station_series import render_station_series
from .gnss_grid import process_gnss_grid
from .gnss_raw import process_gnss_raw, resolve_internal_workspace_root
from .gold import process_gold
from .omni import process_omni
from ..storage import remove_generated_tree, reset_generated_directory


def plot_event_target(event_spec: EventSpec, target: str, producer: str | None = None) -> None:
    if target == "gnss-map":
        errors: list[str] = []
        producers = (producer,) if producer is not None else event_spec.gnss_map_producers()
        _remove_obsolete_gnss_dirs(event_spec)
        for gnss_producer in producers:
            reset_generated_directory(event_spec.storage.figures_gnss_dir / gnss_producer, event_spec.storage)
            try:
                products = _load_gnss_products(event_spec, gnss_producer)
                outputs = []
                for metric in event_spec.figures.get("gnss_maps", {}).get("metrics", ["vtec", "roti"]):
                    outputs.extend(render_gnss_maps(event_spec, products, str(metric).lower()))
                if not outputs:
                    raise ValueError(f"No GNSS map figures were rendered for producer '{gnss_producer}'.")
            except Exception as exc:
                errors.append(f"{gnss_producer}: {exc}")
        if errors:
            raise ValueError("; ".join(errors))
        return

    if target == "gold-map":
        scenes = _load_gold_scenes(event_spec)
        render_gold_maps(event_spec, scenes)
        return

    if target == "omni-series":
        series = read_omni_series(event_spec.storage.manifests_dir / "omni_series.csv")
        if series is None:
            series = discover_omni_series(event_spec)
            if series is None:
                series = process_omni(event_spec)
        render_omni_series(event_spec, series)
        return

    if target == "overlay":
        scenes = _load_gold_scenes(event_spec)
        _remove_obsolete_overlay_dirs(event_spec)
        for overlay_spec in event_spec.overlay_specs():
            products = _load_gnss_products(event_spec, overlay_spec.producer)
            reset_generated_directory(event_spec.storage.figures_overlays_dir / overlay_spec.name, event_spec.storage)
            outputs = render_overlays(event_spec, products, scenes, overlay_spec)
            if not outputs:
                raise ValueError(f"No overlay figures were rendered for '{overlay_spec.name}'.")
        return

    if target == "station-series":
        workspace_root = resolve_internal_workspace_root(event_spec)
        for preset in event_spec.station_series_presets():
            render_station_series(event_spec, preset, workspace_root)
        return

    if target == "panel":
        panel_specs = event_spec.panel_specs()
        if not panel_specs:
            raise ValueError("No panel presets are configured for this event.")
        producers_needed = {
            slot.producer
            for panel_spec in panel_specs
            for slot in panel_spec.slots
            if slot.kind in {"gnss_roti", "gnss_vtec"} and slot.producer is not None
        }
        if any(slot.kind == "overlay" for panel_spec in panel_specs for slot in panel_spec.slots):
            overlay_specs = event_spec.overlay_specs()
            overlay_producer = overlay_specs[0].producer if overlay_specs else "isee"
            producers_needed.add(overlay_producer)
        gnss_products_by_producer = {
            producer: _load_gnss_products(event_spec, producer)
            for producer in sorted(producers_needed)
        }
        reset_generated_directory(event_spec.storage.figures_panels_dir, event_spec.storage)
        outputs = render_panels(
            event_spec,
            gnss_products_by_producer=gnss_products_by_producer,
            overlay_spec=event_spec.overlay_specs()[0] if event_spec.overlay_specs() else None,
        )
        if not outputs:
            raise ValueError("No panel figures were rendered.")
        return

    raise ValueError(f"Unsupported plot target: {target}")


def _load_gnss_products(event_spec: EventSpec, producer: str):
    products = [product for product in read_gnss_grid_products(event_spec.storage.manifests_dir / "gnss_grid_products.csv") if product.producer == producer]
    if not products:
        if producer == "isee":
            if not discover_isee_grid_assets(event_spec):
                raise ValueError(f"GNSS grid products for producer '{producer}' are unavailable.")
            products = [product for product in process_gnss_grid(event_spec) if product.producer == producer]
        elif producer == "internal":
            products = [product for product in process_gnss_raw(event_spec) if product.producer == producer]
    if not products:
        raise ValueError(f"GNSS grid products for producer '{producer}' are unavailable.")
    return products


def _load_gold_scenes(event_spec: EventSpec):
    scenes = read_gold_scenes(event_spec.storage.manifests_dir / "gold_scenes.csv")
    if not scenes:
        scenes = discover_gold_scenes(event_spec)
        if scenes:
            return scenes
        scenes = process_gold(event_spec)
    if not scenes:
        raise ValueError("GOLD scenes are unavailable for this event.")
    return scenes


def _remove_obsolete_gnss_dirs(event_spec: EventSpec) -> None:
    for metric in ("vtec", "roti"):
        legacy_dir = event_spec.storage.figures_gnss_dir / metric
        if legacy_dir.exists():
            remove_generated_tree(legacy_dir, event_spec.storage)


def _remove_obsolete_overlay_dirs(event_spec: EventSpec) -> None:
    legacy_dir = event_spec.storage.figures_overlays_dir / "vtec_on_gold"
    if legacy_dir.exists():
        remove_generated_tree(legacy_dir, event_spec.storage)
