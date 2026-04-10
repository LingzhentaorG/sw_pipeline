from __future__ import annotations

from .models import EventSpec
from .pipelines.figures import plot_event_target
from .pipelines.gnss_grid import fetch_gnss_grid, process_gnss_grid
from .pipelines.gnss_raw import fetch_gnss_raw, process_gnss_raw
from .pipelines.gold import fetch_gold, process_gold
from .pipelines.omni import fetch_omni, process_omni
from .registry.manifests import write_stage_status


def fetch_target(event_spec: EventSpec, target: str):
    if target == "gnss-raw":
        return fetch_gnss_raw(event_spec)
    if target == "gnss-grid":
        return fetch_gnss_grid(event_spec)
    if target == "gold":
        return fetch_gold(event_spec)
    if target == "omni":
        return fetch_omni(event_spec)
    raise ValueError(f"Unsupported fetch target: {target}")


def process_target(event_spec: EventSpec, target: str):
    if target == "gnss":
        errors: list[str] = []
        products = []
        needs_internal = bool(event_spec.station_series_presets()) or "internal" in event_spec.gnss_map_producers()
        if needs_internal:
            try:
                products.extend(process_gnss_raw(event_spec))
            except Exception as exc:
                errors.append(f"internal: {exc}")
        if "isee" in event_spec.gnss_map_producers():
            try:
                products.extend(process_gnss_grid(event_spec))
            except Exception as exc:
                errors.append(f"isee: {exc}")
        if errors:
            raise ValueError("; ".join(errors))
        return products
    if target == "gold":
        return process_gold(event_spec)
    if target == "omni":
        return process_omni(event_spec)
    raise ValueError(f"Unsupported process target: {target}")


def plot_target(event_spec: EventSpec, target: str, **kwargs) -> None:
    plot_event_target(event_spec, target, **kwargs)


def run_event(event_spec: EventSpec, include_fetch: bool = False) -> None:
    stage_rows: list[dict[str, object]] = []
    stage_status_path = event_spec.storage.manifests_dir / "event_stage_status.csv"

    needs_station_series = bool(event_spec.station_series_presets())
    map_producers = event_spec.gnss_map_producers()
    gnss_raw_enabled = bool(event_spec.sources["gnss_raw"].get("enabled", False))
    gnss_grid_enabled = bool(event_spec.sources["gnss_grid"].get("enabled", False))

    if include_fetch and gnss_raw_enabled and ("internal" in map_producers or needs_station_series):
        _run_stage(stage_status_path, stage_rows, "fetch", "gnss-raw", lambda: fetch_target(event_spec, "gnss-raw"))
    if include_fetch and gnss_grid_enabled and "isee" in map_producers:
        _run_stage(stage_status_path, stage_rows, "fetch", "gnss-grid", lambda: fetch_target(event_spec, "gnss-grid"))
    if map_producers:
        _run_stage(stage_status_path, stage_rows, "process", "gnss", lambda: process_target(event_spec, "gnss"))

    if event_spec.sources["gold"].get("enabled", False):
        if include_fetch:
            _run_stage(stage_status_path, stage_rows, "fetch", "gold", lambda: fetch_target(event_spec, "gold"))
        _run_stage(stage_status_path, stage_rows, "process", "gold", lambda: process_target(event_spec, "gold"))

    if event_spec.sources["omni"].get("enabled", False):
        if include_fetch:
            _run_stage(stage_status_path, stage_rows, "fetch", "omni", lambda: fetch_target(event_spec, "omni"))
        _run_stage(stage_status_path, stage_rows, "process", "omni", lambda: process_target(event_spec, "omni"))

    if event_spec.figures.get("gnss_maps", {}).get("metrics", ["vtec", "roti"]):
        for producer in map_producers:
            _run_stage(
                stage_status_path,
                stage_rows,
                "plot",
                f"gnss-map:{producer}",
                lambda producer=producer: plot_target(event_spec, "gnss-map", producer=producer),
            )
    if event_spec.sources["gold"].get("enabled", False):
        _run_stage(stage_status_path, stage_rows, "plot", "gold-map", lambda: plot_target(event_spec, "gold-map"))
    if event_spec.sources["omni"].get("enabled", False):
        _run_stage(stage_status_path, stage_rows, "plot", "omni-series", lambda: plot_target(event_spec, "omni-series"))
    if event_spec.sources["gold"].get("enabled", False) and event_spec.overlay_specs():
        _run_stage(stage_status_path, stage_rows, "plot", "overlay:isee", lambda: plot_target(event_spec, "overlay"))
    if event_spec.station_series_presets():
        _run_stage(stage_status_path, stage_rows, "plot", "station-series", lambda: plot_target(event_spec, "station-series"))
    if event_spec.panel_specs():
        _run_stage(stage_status_path, stage_rows, "plot", "panel", lambda: plot_target(event_spec, "panel"))

    write_stage_status(stage_status_path, stage_rows)


def _run_stage(
    stage_status_path,
    stage_rows: list[dict[str, object]],
    stage: str,
    target: str,
    action,
) -> None:
    row = {
        "stage": stage,
        "target": target,
        "status": "running",
        "detail": "",
    }
    stage_rows.append(row)
    write_stage_status(stage_status_path, stage_rows)
    try:
        action()
    except Exception as exc:
        row["status"] = "failed"
        row["detail"] = str(exc)
        write_stage_status(stage_status_path, stage_rows)
        return

    row["status"] = "ok"
    write_stage_status(stage_status_path, stage_rows)
