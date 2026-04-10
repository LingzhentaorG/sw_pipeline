from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


FIXED_MAP_BBOX = {
    "lon_min": -150.0,
    "lon_max": 10.0,
    "lat_min": -80.0,
    "lat_max": 80.0,
}

ALLOWED_GNSS_PRODUCERS = ("internal", "isee")


@dataclass(frozen=True)
class StorageLayout:
    project_root: Path
    storage_root: Path
    cache_root: Path
    archive_root: Path
    pre_refactor_archive_root: Path
    runs_root: Path
    run_root: Path
    manifests_dir: Path
    processed_root: Path
    processed_gnss_dir: Path
    processed_gold_dir: Path
    processed_omni_dir: Path
    gnss_workspace_dir: Path
    grids_dir: Path
    figures_root: Path
    figures_gnss_dir: Path
    figures_gold_dir: Path
    figures_omni_dir: Path
    figures_overlays_dir: Path
    figures_panels_dir: Path
    figures_station_series_dir: Path

    @property
    def event_root(self) -> Path:
        return self.run_root


@dataclass(frozen=True)
class SourceAsset:
    event_id: str
    source_kind: str
    provider: str
    asset_id: str
    local_path: Path
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GnssStationCandidate:
    event_id: str
    provider: str
    station_id: str
    station_code4: str
    observation_date: str
    sampling_sec: int
    lat: float | None = None
    lon: float | None = None
    height_m: float | None = None
    obs_url: str = ""
    nav_url: str = ""
    status: str = "candidate"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GnssDownloadAsset:
    event_id: str
    source_kind: str
    provider: str
    protocol: str
    station_id: str
    station_code4: str
    observation_date: str
    url: str
    local_path: Path | None
    status: str
    attempts: int = 0
    error: str | None = None
    auth_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GnssGridProduct:
    event_id: str
    producer: str
    source_kind: str
    path: Path
    metrics: tuple[str, ...]
    time_start: datetime
    time_end: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoldScene:
    event_id: str
    tar_path: Path
    midpoint: datetime
    cha_member: str
    chb_member: str
    delta_minutes: float
    cha_time: datetime | None = None
    chb_time: datetime | None = None


@dataclass(frozen=True)
class OmniSeries:
    event_id: str
    start_utc: datetime
    end_utc: datetime
    bz_csv_path: Path
    hourly_csv_path: Path
    kp_csv_path: Path


@dataclass(frozen=True)
class StationSeriesPreset:
    name: str
    station_code: str
    station_id: str | None
    start_utc: datetime
    end_utc: datetime
    satellites: tuple[str, ...]


@dataclass(frozen=True)
class OverlayPairSpec:
    gold_cha_time: datetime
    gold_chb_time: datetime
    gnss_time: datetime


@dataclass(frozen=True)
class OverlaySpec:
    name: str
    metric: str = "roti"
    threshold: float = 1.0
    producer: str = "isee"
    color: str = "red"
    max_pair_delta_minutes: int = 15
    bin_size_deg: float = 0.5
    pairs: tuple[OverlayPairSpec, ...] = ()


@dataclass(frozen=True)
class PanelSlotSpec:
    kind: str
    title: str
    producer: str | None = None
    timestamp: datetime | None = None
    gold_cha_time: datetime | None = None
    gold_chb_time: datetime | None = None
    gnss_time: datetime | None = None


@dataclass(frozen=True)
class PanelSpec:
    name: str
    rows: int
    cols: int
    shared_colorbar: str
    slots: tuple[PanelSlotSpec, ...]
    footer_note: str = ""


@dataclass(frozen=True)
class OmniHighlightWindow:
    start_utc: datetime
    end_utc: datetime
    color: str = "#dbeafe"
    alpha: float = 0.35


@dataclass(frozen=True)
class GnssMapStyle:
    cmap: str
    vmin: float
    vmax: float


@dataclass(frozen=True)
class PlotDefaults:
    dpi: int
    figure_size: tuple[float, float]
    font_family: str
    use_cartopy: bool
    show_magnetic_equator: bool
    magnetic_equator_color: str
    magnetic_equator_linewidth: float
    gnss_styles: dict[str, GnssMapStyle]


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    start_utc: datetime
    end_utc: datetime
    bbox: dict[str, float]
    sources: dict[str, Any]
    products: dict[str, Any]
    figures: dict[str, Any]
    storage: StorageLayout
    plot_defaults: PlotDefaults
    auth: dict[str, Any]
    runtime: dict[str, Any]
    project_root: Path
    base_config_path: Path
    event_config_path: Path

    def event_days(self) -> tuple[pd.Timestamp, ...]:
        start = pd.Timestamp(self.start_utc).floor("D")
        end = pd.Timestamp(self.end_utc).floor("D")
        days = pd.date_range(start, end, freq="D")
        return tuple(days)

    def needs_internal_gnss(self) -> bool:
        return "internal" in self.gnss_map_producers() or bool(self.station_series_presets())

    def gnss_map_producers(self) -> tuple[str, ...]:
        raw_producers = self.products.get("gnss_grid", {}).get("map_producers", ["internal"])
        normalized: list[str] = []
        for value in raw_producers:
            producer = str(value).lower()
            if producer in ALLOWED_GNSS_PRODUCERS and producer not in normalized:
                normalized.append(producer)
        return tuple(normalized)

    def map_extent(self) -> tuple[float, float, float, float]:
        return (
            self.bbox["lon_min"],
            self.bbox["lon_max"],
            self.bbox["lat_min"],
            self.bbox["lat_max"],
        )

    def overlay_specs(self) -> tuple[OverlaySpec, ...]:
        specs: list[OverlaySpec] = []
        for raw in self.figures.get("overlays", []):
            pair_specs: list[OverlayPairSpec] = []
            for pair in raw.get("pairs", []):
                pair_specs.append(
                    OverlayPairSpec(
                        gold_cha_time=pd.Timestamp(pair["gold_cha_time"], tz="UTC").to_pydatetime(),
                        gold_chb_time=pd.Timestamp(pair["gold_chb_time"], tz="UTC").to_pydatetime(),
                        gnss_time=pd.Timestamp(pair["gnss_time"], tz="UTC").to_pydatetime(),
                    )
                )
            specs.append(
                OverlaySpec(
                    name=str(raw["name"]),
                    metric=str(raw.get("metric", "roti")).lower(),
                    threshold=float(raw.get("threshold", 1.0)),
                    producer=str(raw.get("producer", "isee")).lower(),
                    color=str(raw.get("color", "red")),
                    max_pair_delta_minutes=int(raw.get("max_pair_delta_minutes", 15)),
                    bin_size_deg=float(raw.get("bin_size_deg", 0.5)),
                    pairs=tuple(pair_specs),
                )
            )
        return tuple(specs)

    def station_series_presets(self) -> tuple[StationSeriesPreset, ...]:
        presets: list[StationSeriesPreset] = []
        for raw in self.figures.get("station_series", []):
            window = raw["window"]
            presets.append(
                StationSeriesPreset(
                    name=str(raw["name"]),
                    station_code=str(raw["station_code"]).upper(),
                    station_id=str(raw["station_id"]).upper() if raw.get("station_id") else None,
                    start_utc=pd.Timestamp(window["start"], tz="UTC").to_pydatetime(),
                    end_utc=pd.Timestamp(window["end"], tz="UTC").to_pydatetime(),
                    satellites=tuple(str(item).upper() for item in raw["satellites"]),
                )
            )
        return tuple(presets)

    def panel_specs(self) -> tuple[PanelSpec, ...]:
        specs: list[PanelSpec] = []
        for raw in self.figures.get("panels", []):
            layout = raw["layout"]
            slot_specs: list[PanelSlotSpec] = []
            for slot in raw["slots"]:
                slot_specs.append(
                    PanelSlotSpec(
                        kind=str(slot["kind"]).lower(),
                        title=str(slot["title"]),
                        producer=str(slot["producer"]).lower() if slot.get("producer") else None,
                        timestamp=(
                            pd.Timestamp(slot["timestamp"], tz="UTC").to_pydatetime()
                            if slot.get("timestamp")
                            else None
                        ),
                        gold_cha_time=(
                            pd.Timestamp(slot["gold_cha_time"], tz="UTC").to_pydatetime()
                            if slot.get("gold_cha_time")
                            else None
                        ),
                        gold_chb_time=(
                            pd.Timestamp(slot["gold_chb_time"], tz="UTC").to_pydatetime()
                            if slot.get("gold_chb_time")
                            else None
                        ),
                        gnss_time=(
                            pd.Timestamp(slot["gnss_timestamp"], tz="UTC").to_pydatetime()
                            if slot.get("gnss_timestamp")
                            else None
                        ),
                    )
                )
            specs.append(
                PanelSpec(
                    name=str(raw["name"]),
                    rows=int(layout["rows"]),
                    cols=int(layout["cols"]),
                    shared_colorbar=str(raw["shared_colorbar"]).lower(),
                    slots=tuple(slot_specs),
                    footer_note=str(raw.get("footer_note", "")),
                )
            )
        return tuple(specs)

    def omni_highlight_windows(self) -> tuple[OmniHighlightWindow, ...]:
        raw_windows = self.figures.get("omni_series", {}).get("highlight_windows", [])
        windows: list[OmniHighlightWindow] = []
        for raw in raw_windows:
            windows.append(
                OmniHighlightWindow(
                    start_utc=pd.Timestamp(raw["start"], tz="UTC").to_pydatetime(),
                    end_utc=pd.Timestamp(raw["end"], tz="UTC").to_pydatetime(),
                    color=str(raw.get("color", "#dbeafe")),
                    alpha=float(raw.get("alpha", 0.35)),
                )
            )
        return tuple(windows)

    def internal_gnss_mode(self) -> str:
        return str(self.sources.get("gnss_raw", {}).get("mode", "workspace_snapshot")).lower()

    def internal_gnss_workspace_root(self) -> Path:
        raw = self.sources.get("gnss_raw", {})
        workspace_root = raw.get("workspace_root")
        if workspace_root is None:
            return self.storage.cache_root / "gnss_raw" / "internal" / self.event_id
        path = Path(str(workspace_root)).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.project_root / path).resolve()


@dataclass(frozen=True)
class GnssGridSlice:
    event_id: str
    metric: str
    producer: str
    source_path: Path
    timestamp: pd.Timestamp
    lat: Any
    lon: Any
    values: Any
    units: str | None


@dataclass(frozen=True)
class TimePair:
    left_time: pd.Timestamp
    right_time: pd.Timestamp
    delta: timedelta
    left_index: int
    right_index: int
