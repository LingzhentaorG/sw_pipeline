from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .models import ALLOWED_GNSS_PRODUCERS, FIXED_MAP_BBOX, EventSpec, GnssMapStyle, PlotDefaults
from .storage import build_storage_layout


def load_app_config(
    event_id: str,
    base_config_path: str | Path | None = None,
    event_config_path: str | Path | None = None,
) -> EventSpec:
    project_root = Path(__file__).resolve().parents[2]
    base_path = (
        Path(base_config_path).expanduser().resolve()
        if base_config_path is not None
        else project_root / "config" / "base.yaml"
    )
    event_path = (
        Path(event_config_path).expanduser().resolve()
        if event_config_path is not None
        else project_root / "config" / "events" / f"{event_id}.yaml"
    )
    if not base_path.exists():
        raise FileNotFoundError(f"Base config does not exist: {base_path}")
    if not event_path.exists():
        raise FileNotFoundError(f"Event config does not exist: {event_path}")

    base_raw = _load_yaml(base_path)
    event_raw = _load_yaml(event_path)
    merged = _deep_merge(base_raw, event_raw)
    _apply_download_defaults(merged)
    merged.setdefault("figures", {}).setdefault("panels", [])
    merged.setdefault("figures", {}).setdefault("omni_series", {})
    _validate_required_sections(merged)

    event_cfg = merged["event"]
    resolved_event_id = str(event_cfg.get("id", event_id))
    storage_root = _resolve_path(merged.get("paths", {}).get("storage_root", "storage"), project_root)
    storage = build_storage_layout(project_root, storage_root, resolved_event_id)
    plot_defaults = _build_plot_defaults(merged.get("plot_defaults", {}))

    event_spec = EventSpec(
        event_id=resolved_event_id,
        start_utc=_parse_utc(event_cfg["start"]),
        end_utc=_parse_utc(event_cfg["end"]),
        bbox={key: float(value) for key, value in merged["bbox"].items()},
        sources=merged["sources"],
        products=merged["products"],
        figures=merged["figures"],
        storage=storage,
        plot_defaults=plot_defaults,
        auth=_resolve_auth(merged.get("auth", {})),
        runtime=dict(merged.get("runtime", {})),
        project_root=project_root,
        base_config_path=base_path,
        event_config_path=event_path,
    )
    return event_spec


def export_event_spec_summary(event_spec: EventSpec) -> dict[str, Any]:
    summary = asdict(event_spec)
    summary["project_root"] = str(event_spec.project_root)
    summary["base_config_path"] = str(event_spec.base_config_path)
    summary["event_config_path"] = str(event_spec.event_config_path)
    return summary


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_required_sections(raw: dict[str, Any]) -> None:
    required = [
        "event",
        "bbox",
        "sources",
        "products",
        "figures",
        "runtime",
    ]
    for key in required:
        if key not in raw:
            raise ValueError(f"Missing required top-level config section: {key}")

    event_cfg = raw["event"]
    for key in ("start", "end"):
        if key not in event_cfg:
            raise ValueError(f"Missing required event config: event.{key}")

    bbox = raw["bbox"]
    for key in ("lon_min", "lon_max", "lat_min", "lat_max"):
        if key not in bbox:
            raise ValueError(f"Missing required bbox config: bbox.{key}")
    for key, expected in FIXED_MAP_BBOX.items():
        actual = float(bbox[key])
        if actual != expected:
            raise ValueError(
                "bbox must use the fixed map extent "
                f"{FIXED_MAP_BBOX}, got {key}={actual:g}."
            )

    required_sources = ("gnss_raw", "gnss_grid", "gold", "omni")
    for key in required_sources:
        if key not in raw["sources"]:
            raise ValueError(f"Missing required source config: sources.{key}")
    gnss_raw_cfg = raw["sources"].get("gnss_raw", {})
    providers = gnss_raw_cfg.get("providers", {})
    auxiliary = gnss_raw_cfg.get("auxiliary", {})
    if not isinstance(providers, dict):
        raise ValueError("sources.gnss_raw.providers must be a mapping.")
    if not isinstance(auxiliary, dict):
        raise ValueError("sources.gnss_raw.auxiliary must be a mapping.")

    gnss_grid_cfg = raw["products"].get("gnss_grid", {})
    map_producers = gnss_grid_cfg.get("map_producers")
    if not isinstance(map_producers, list) or not map_producers:
        raise ValueError("products.gnss_grid.map_producers must be a non-empty list.")
    normalized_producers = [str(item).lower() for item in map_producers]
    invalid = [item for item in normalized_producers if item not in ALLOWED_GNSS_PRODUCERS]
    if invalid:
        allowed = ", ".join(ALLOWED_GNSS_PRODUCERS)
        raise ValueError(f"products.gnss_grid.map_producers only allows: {allowed}.")

    if "overlays" not in raw["figures"]:
        raise ValueError("Missing required figure config: figures.overlays")
    if "station_series" not in raw["figures"]:
        raise ValueError("Missing required figure config: figures.station_series")
    if "panels" not in raw["figures"]:
        raise ValueError("Missing required figure config: figures.panels")
    omni_series_cfg = raw["figures"].get("omni_series", {})
    if not isinstance(raw["figures"]["overlays"], list):
        raise ValueError("figures.overlays must be a list.")
    if not isinstance(raw["figures"]["station_series"], list):
        raise ValueError("figures.station_series must be a list.")
    if not isinstance(raw["figures"]["panels"], list):
        raise ValueError("figures.panels must be a list.")
    if omni_series_cfg is None:
        omni_series_cfg = {}
    if not isinstance(omni_series_cfg, dict):
        raise ValueError("figures.omni_series must be a mapping.")

    overlays = raw["figures"]["overlays"]
    if len(overlays) > 1:
        raise ValueError("figures.overlays may contain at most one entry.")
    for overlay in overlays:
        if "name" not in overlay:
            raise ValueError("Missing required overlay config: figures.overlays[].name")
        if str(overlay["name"]).lower() != "roti_on_gold":
            raise ValueError("figures.overlays only supports the 'roti_on_gold' overlay.")
        if "plot_extent" in overlay:
            raise ValueError("figures.overlays[].plot_extent is no longer supported; use the fixed bbox only.")
        pairs = overlay.get("pairs", [])
        if pairs is None:
            pairs = []
        if not isinstance(pairs, list):
            raise ValueError("figures.overlays[].pairs must be a list when provided.")
        for pair in pairs:
            for key in ("gold_cha_time", "gold_chb_time", "gnss_time"):
                if key not in pair:
                    raise ValueError(f"Missing required overlay pair config: figures.overlays[].pairs[].{key}")

    for preset in raw["figures"]["station_series"]:
        for key in ("name", "station_code", "window", "satellites"):
            if key not in preset:
                raise ValueError(f"Missing required station series config: figures.station_series[].{key}")
        window = preset["window"]
        for key in ("start", "end"):
            if key not in window:
                raise ValueError(f"Missing required station series window config: window.{key}")
        satellites = preset["satellites"]
        if not isinstance(satellites, list) or len(satellites) < 2:
            raise ValueError("figures.station_series[].satellites must provide at least two satellites.")

    highlight_windows = omni_series_cfg.get("highlight_windows", [])
    if highlight_windows is None:
        highlight_windows = []
    if not isinstance(highlight_windows, list):
        raise ValueError("figures.omni_series.highlight_windows must be a list.")
    for window in highlight_windows:
        if not isinstance(window, dict):
            raise ValueError("figures.omni_series.highlight_windows[] must be a mapping.")
        for key in ("start", "end"):
            if key not in window:
                raise ValueError(f"Missing required omni highlight config: figures.omni_series.highlight_windows[].{key}")
        alpha = float(window.get("alpha", 0.35))
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("figures.omni_series.highlight_windows[].alpha must be between 0 and 1.")

    allowed_panel_kinds = {"gnss_roti", "gnss_vtec", "gold", "overlay"}
    allowed_colorbars = {"gold", "gnss_roti", "gnss_vtec"}
    for panel in raw["figures"]["panels"]:
        for key in ("name", "layout", "shared_colorbar", "slots"):
            if key not in panel:
                raise ValueError(f"Missing required panel config: figures.panels[].{key}")
        layout = panel["layout"]
        if not isinstance(layout, dict):
            raise ValueError("figures.panels[].layout must be a mapping.")
        for key in ("rows", "cols"):
            if key not in layout:
                raise ValueError(f"Missing required panel layout config: figures.panels[].layout.{key}")
            if int(layout[key]) <= 0:
                raise ValueError(f"figures.panels[].layout.{key} must be greater than 0.")
        if str(panel["shared_colorbar"]).lower() not in allowed_colorbars:
            allowed = ", ".join(sorted(allowed_colorbars))
            raise ValueError(f"figures.panels[].shared_colorbar only allows: {allowed}.")
        slots = panel["slots"]
        if not isinstance(slots, list):
            raise ValueError("figures.panels[].slots must be a list.")
        expected_slots = int(layout["rows"]) * int(layout["cols"])
        if len(slots) != expected_slots:
            raise ValueError("figures.panels[].slots must match layout.rows * layout.cols.")
        for slot in slots:
            for key in ("kind", "title"):
                if key not in slot:
                    raise ValueError(f"Missing required panel slot config: figures.panels[].slots[].{key}")
            kind = str(slot["kind"]).lower()
            if kind not in allowed_panel_kinds:
                allowed = ", ".join(sorted(allowed_panel_kinds))
                raise ValueError(f"figures.panels[].slots[].kind only allows: {allowed}.")
            if kind in {"gnss_roti", "gnss_vtec"}:
                for key in ("producer", "timestamp"):
                    if key not in slot:
                        raise ValueError(f"Missing required {kind} slot config: figures.panels[].slots[].{key}")
                producer = str(slot["producer"]).lower()
                if producer not in ALLOWED_GNSS_PRODUCERS:
                    allowed = ", ".join(ALLOWED_GNSS_PRODUCERS)
                    raise ValueError(f"figures.panels[].slots[].producer only allows: {allowed}.")
            elif kind == "gold":
                for key in ("gold_cha_time", "gold_chb_time"):
                    if key not in slot:
                        raise ValueError(f"Missing required gold slot config: figures.panels[].slots[].{key}")
            elif kind == "overlay":
                for key in ("gold_cha_time", "gold_chb_time", "gnss_timestamp"):
                    if key not in slot:
                        raise ValueError(f"Missing required overlay slot config: figures.panels[].slots[].{key}")

def _build_plot_defaults(raw: dict[str, Any]) -> PlotDefaults:
    styles_raw = raw.get("gnss_styles", {})
    styles = {
        key.lower(): GnssMapStyle(
            cmap=str(value.get("cmap", "viridis")),
            vmin=float(value.get("vmin", 0.0)),
            vmax=float(value.get("vmax", 1.0)),
        )
        for key, value in styles_raw.items()
    }
    for metric, defaults in {
        "vtec": ("viridis", 0.0, 80.0),
        "roti": ("viridis", 0.0, 1.0),
    }.items():
        styles.setdefault(
            metric,
            GnssMapStyle(cmap=defaults[0], vmin=defaults[1], vmax=defaults[2]),
        )

    figure_size = raw.get("figure_size", [12.0, 6.75])
    return PlotDefaults(
        dpi=int(raw.get("dpi", 220)),
        figure_size=(float(figure_size[0]), float(figure_size[1])),
        font_family=str(raw.get("font_family", "Times New Roman")),
        use_cartopy=bool(raw.get("use_cartopy", True)),
        show_magnetic_equator=bool(raw.get("show_magnetic_equator", True)),
        magnetic_equator_color=str(raw.get("magnetic_equator_color", "red")),
        magnetic_equator_linewidth=float(raw.get("magnetic_equator_linewidth", 1.2)),
        gnss_styles=styles,
    )


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _parse_utc(value: str) -> Any:
    import pandas as pd

    return pd.Timestamp(value, tz="UTC").to_pydatetime()


def _apply_download_defaults(raw: dict[str, Any]) -> None:
    sources = raw.setdefault("sources", {})
    gnss_raw = sources.setdefault("gnss_raw", {})
    gnss_raw["providers"] = _deep_merge(_default_gnss_raw_providers(), dict(gnss_raw.get("providers", {})))
    gnss_raw["auxiliary"] = _deep_merge(_default_gnss_raw_auxiliary(), dict(gnss_raw.get("auxiliary", {})))


def _resolve_auth(raw_auth: dict[str, Any]) -> dict[str, Any]:
    auth = dict(raw_auth or {})
    cddis = dict(auth.get("cddis", {})) if isinstance(auth.get("cddis"), dict) else {}
    username = (
        os.getenv("CDDIS_USERNAME")
        or os.getenv("EARTHDATA_USERNAME")
        or os.getenv("NASA_EARTHDATA_USERNAME")
        or cddis.get("username", "")
    )
    password = (
        os.getenv("CDDIS_PASSWORD")
        or os.getenv("EARTHDATA_PASSWORD")
        or os.getenv("NASA_EARTHDATA_PASSWORD")
        or cddis.get("password", "")
    )
    cddis["username"] = str(username or "")
    cddis["password"] = str(password or "")
    auth["cddis"] = cddis
    return auth


def _default_gnss_raw_providers() -> dict[str, Any]:
    return {
        "noaa": {
            "enabled": True,
            "priority": 10,
            "timeout_sec": 90,
            "transport": "https",
            "network_kmz_url": "https://geodesy.noaa.gov/CORS/data/kmz/NOAA_CORS_Network.kmz",
            "base_obs_url": "https://geodesy.noaa.gov/corsdata/rinex",
            "fallback_base_obs_url": "https://noaa-cors-pds.s3.amazonaws.com/rinex",
        },
        "rbmc": {
            "enabled": True,
            "priority": 20,
            "timeout_sec": 90,
            "transport": "https",
            "base_dir_url": "https://geoftp.ibge.gov.br/informacoes_sobre_posicionamento_geodesico/rbmc/dados_RINEX3",
        },
        "ramsac": {
            "enabled": True,
            "priority": 30,
            "timeout_sec": 90,
            "transport": "https",
            "verify_ssl": False,
            "stations_api_url": "https://ramsac.ign.gob.ar/api/v1/estaciones",
            "download_base_url": "https://ramsac.ign.gob.ar/api/v1/rinex/download",
            "requested_interval_sec": 15,
        },
        "cddis": {
            "enabled": True,
            "priority": 100,
            "timeout_sec": 120,
            "transport": "https",
            "station_codes": [],
            "obs_url_template": "https://cddis.nasa.gov/archive/gnss/data/daily/{year}/{doy}/{yy}d/{station}{doy}0.{yy}d.Z",
        },
    }


def _default_gnss_raw_auxiliary() -> dict[str, Any]:
    return {
        "broadcast": {
            "enabled": True,
            "priority": 10,
            "timeout_sec": 120,
            "providers": [
                {
                    "name": "cddis_daily_mixed_nav",
                    "priority": 10,
                    "auth": "cddis",
                    "transport": "https",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/data/daily/{yyyy}/{ddd}/{yy}p/BRDM00DLR_S_{yyyy}{ddd}0000_01D_MN.rnx.gz",
                },
                {
                    "name": "cddis_daily_gps_nav_legacy",
                    "priority": 20,
                    "auth": "cddis",
                    "transport": "https",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/data/daily/{yyyy}/{ddd}/{yy}n/brdc{ddd}0.{yy}n.Z",
                },
            ],
        },
        "sp3": {
            "enabled": True,
            "priority": 20,
            "timeout_sec": 120,
            "providers": [
                {
                    "name": "cddis_igs_fin_long",
                    "priority": 10,
                    "auth": "cddis",
                    "transport": "https",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/products/{gps_week}/IGS0OPSFIN_{yyyy}{ddd}0000_01D_15M_ORB.SP3.gz",
                },
                {
                    "name": "cddis_igs_rap_long",
                    "priority": 20,
                    "auth": "cddis",
                    "transport": "https",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/products/{gps_week}/IGS0OPSRAP_{yyyy}{ddd}0000_01D_15M_ORB.SP3.gz",
                },
                {
                    "name": "cddis_igs_fin_short",
                    "priority": 30,
                    "auth": "cddis",
                    "transport": "https",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/products/{gps_week}/igs{gps_week}{dow}.sp3.Z",
                },
            ],
        },
        "dcb": {
            "enabled": True,
            "priority": 30,
            "timeout_sec": 120,
            "providers": [
                {
                    "name": "cddis_cas1_ops_bia",
                    "priority": 10,
                    "transport": "https",
                    "url_template": "https://data.bdsmart.cn/pub/product/bias/{yyyy}/CAS1OPSRAP_{yyyy}{ddd}0000_01D_01D_DCB.BIA.gz",
                },
                {
                    "name": "cddis_cas0_ops_bia",
                    "priority": 20,
                    "transport": "https",
                    "url_template": "https://data.bdsmart.cn/pub/product/bias/{yyyy}/CAS0OPSRAP_{yyyy}{ddd}0000_01D_01D_DCB.BIA.gz",
                },
            ],
        },
        "antex": {
            "enabled": True,
            "priority": 40,
            "timeout_sec": 120,
            "providers": [
                {
                    "name": "cddis_igs20_atx",
                    "priority": 10,
                    "transport": "https",
                    "url_template": "https://files.igs.org/pub/station/general/igs20.atx.gz",
                }
            ],
        },
        "station_logs": {
            "enabled": True,
            "priority": 50,
            "timeout_sec": 90,
            "best_effort": True,
            "providers": [
                {
                    "name": "cddis_station_log",
                    "priority": 10,
                    "auth": "cddis",
                    "transport": "https",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/data/station/{station_code4}.log",
                }
            ],
        },
    }
