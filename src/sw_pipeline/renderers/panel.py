from __future__ import annotations

import string
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..discovery import discover_gold_assets
from ..models import EventSpec, GnssGridProduct, OverlaySpec, PanelSpec, PanelSlotSpec
from ..registry.manifests import write_panel_outputs
from .gnss_map import iter_gnss_slices, plot_gnss_slice, prepare_gnss_slice
from .gold_map import plot_gold_pair
from .overlay import build_overlay_payload
from .style import GOLD_RADIANCE_LABEL, metric_colorbar_label


try:  # pragma: no cover - optional dependency
    import cartopy.crs as ccrs
except Exception:  # pragma: no cover - optional fallback
    ccrs = None


PANEL_TITLE_FONT_SIZE = 12.5
PANEL_TICK_FONT_SIZE = 11.0
PANEL_COLORBAR_FONT_SIZE = 11.5
PANEL_FONT_FAMILY = "Times New Roman"
GNSS_PANEL_METRICS = {
    "gnss_roti": "roti",
    "gnss_vtec": "vtec",
}


def render_panels(
    event_spec: EventSpec,
    gnss_products_by_producer: dict[str, list[GnssGridProduct]],
    overlay_spec: OverlaySpec | None = None,
) -> list[Path]:
    panel_specs = event_spec.panel_specs()
    if not panel_specs:
        return []

    gnss_resolver = _GnssSliceResolver(event_spec, gnss_products_by_producer)
    gold_resolver = _GoldPairResolver(event_spec)
    overlay_style = overlay_spec or OverlaySpec(name="roti_on_gold")

    rows: list[dict[str, object]] = []
    outputs: list[Path] = []
    for panel_spec in panel_specs:
        output_path = event_spec.storage.figures_panels_dir / f"{panel_spec.name}.png"
        panel_rows = _render_single_panel(
            event_spec,
            panel_spec,
            gnss_resolver,
            gold_resolver,
            overlay_style,
            output_path,
        )
        rows.extend(panel_rows)
        outputs.append(output_path)

    write_panel_outputs(event_spec.storage.manifests_dir / "panel_outputs.csv", rows)
    return outputs


def _render_single_panel(
    event_spec: EventSpec,
    panel_spec: PanelSpec,
    gnss_resolver: "_GnssSliceResolver",
    gold_resolver: "_GoldPairResolver",
    overlay_style: OverlaySpec,
    output_path: Path,
) -> list[dict[str, object]]:
    with matplotlib.rc_context({"font.family": [PANEL_FONT_FAMILY], "font.size": PANEL_TICK_FONT_SIZE}):
        fig, axes = _build_panel_figure(event_spec, panel_spec)
        rows: list[dict[str, object]] = []
        rendered_count = 0

        for index, (ax, slot) in enumerate(zip(axes.flat, panel_spec.slots, strict=True)):
            row_index = index // panel_spec.cols
            col_index = index % panel_spec.cols
            show_left_labels = col_index == 0
            show_bottom_labels = row_index == panel_spec.rows - 1
            try:
                row = _render_panel_slot(
                    ax,
                    event_spec,
                    panel_spec,
                    slot,
                    index,
                    gnss_resolver,
                    gold_resolver,
                    overlay_style,
                    show_left_labels=show_left_labels,
                    show_bottom_labels=show_bottom_labels,
                )
            except Exception as exc:
                _draw_placeholder(ax, f"Missing {slot.kind}", str(exc))
                row = {
                    "panel_name": panel_spec.name,
                    "slot_index": index,
                    "kind": slot.kind,
                    "requested_time": _slot_requested_time(slot),
                    "resolved_time": "",
                    "status": "placeholder",
                    "detail": str(exc),
                    "output_path": output_path,
                }
            if row["status"] == "rendered":
                rendered_count += 1
            _set_slot_title(ax, slot.title, index)
            row["output_path"] = output_path
            rows.append(row)

        fig.subplots_adjust(left=0.01, right=0.865, bottom=0.055, top=0.958, wspace=0.002, hspace=0.08)
        if rendered_count > 0:
            _add_shared_colorbar(fig, axes, event_spec, panel_spec, overlay_style.metric)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=event_spec.plot_defaults.dpi, bbox_inches="tight")
        plt.close(fig)
    return rows


def _render_panel_slot(
    ax,
    event_spec: EventSpec,
    panel_spec: PanelSpec,
    slot: PanelSlotSpec,
    index: int,
    gnss_resolver: "_GnssSliceResolver",
    gold_resolver: "_GoldPairResolver",
    overlay_style: OverlaySpec,
    *,
    show_left_labels: bool,
    show_bottom_labels: bool,
) -> dict[str, object]:
    if slot.kind in GNSS_PANEL_METRICS:
        metric = GNSS_PANEL_METRICS[slot.kind]
        slice_data = gnss_resolver.resolve(slot.producer or "isee", metric, slot.timestamp)
        if slice_data is None:
            _draw_placeholder(ax, "Missing GNSS", f"Target {_format_timestamp(slot.timestamp)}")
            return {
                "panel_name": panel_spec.name,
                "slot_index": index,
                "kind": slot.kind,
                "requested_time": slot.timestamp,
                "resolved_time": "",
                "status": "placeholder",
                "detail": "missing_gnss_slice",
            }
        plot_gnss_slice(
            ax,
            slice_data,
            event_spec,
            draw_labels=show_left_labels or show_bottom_labels,
            top_labels=False,
            left_labels=show_left_labels,
            bottom_labels=show_bottom_labels,
        )
        return {
            "panel_name": panel_spec.name,
            "slot_index": index,
            "kind": slot.kind,
            "requested_time": slot.timestamp,
            "resolved_time": slice_data.timestamp,
            "status": "rendered",
            "detail": "",
        }

    if slot.kind == "gold":
        pair = gold_resolver.resolve(slot.gold_cha_time, slot.gold_chb_time)
        if pair is None:
            _draw_placeholder(ax, "Missing GOLD", _gold_target_label(slot))
            return {
                "panel_name": panel_spec.name,
                "slot_index": index,
                "kind": slot.kind,
                "requested_time": _pair_midpoint(slot.gold_cha_time, slot.gold_chb_time),
                "resolved_time": "",
                "status": "placeholder",
                "detail": "missing_gold_pair",
            }
        plot_gold_pair(
            ax,
            event_spec,
            pair,
            decorate_overlay=False,
            draw_labels=show_left_labels or show_bottom_labels,
            left_labels=show_left_labels,
            bottom_labels=show_bottom_labels,
        )
        return {
            "panel_name": panel_spec.name,
            "slot_index": index,
            "kind": slot.kind,
            "requested_time": _pair_midpoint(slot.gold_cha_time, slot.gold_chb_time),
            "resolved_time": pd.Timestamp(pair.midpoint),
            "status": "rendered",
            "detail": f"CHA={_format_timestamp(pair.cha.obs_time)}; CHB={_format_timestamp(pair.chb.obs_time)}",
        }

    if slot.kind == "overlay":
        pair = gold_resolver.resolve(slot.gold_cha_time, slot.gold_chb_time)
        slice_data = gnss_resolver.resolve(overlay_style.producer, overlay_style.metric, slot.gnss_time)
        if pair is None or slice_data is None:
            _draw_placeholder(ax, "Missing Overlay", _overlay_target_label(slot))
            detail = "missing_gold_pair_and_gnss_slice" if pair is None and slice_data is None else "missing_gold_pair" if pair is None else "missing_gnss_slice"
            return {
                "panel_name": panel_spec.name,
                "slot_index": index,
                "kind": slot.kind,
                "requested_time": slot.gnss_time,
                "resolved_time": "",
                "status": "placeholder",
                "detail": detail,
            }
        overlay_payload = build_overlay_payload(slice_data, overlay_style, event_spec)
        overlay_payload["legend_label"] = ""
        overlay_payload["ylabel"] = ""
        plot_gold_pair(
            ax,
            event_spec,
            pair,
            overlay=overlay_payload,
            decorate_overlay=False,
            draw_labels=show_left_labels or show_bottom_labels,
            left_labels=show_left_labels,
            bottom_labels=show_bottom_labels,
        )
        detail = ""
        if int(overlay_payload.get("count", 0)) == 0:
            detail = "threshold_removed_all_pixels"
        return {
            "panel_name": panel_spec.name,
            "slot_index": index,
            "kind": slot.kind,
            "requested_time": slot.gnss_time,
            "resolved_time": slot.gnss_time,
            "status": "rendered",
            "detail": detail,
        }

    raise ValueError(f"Unsupported panel slot kind: {slot.kind}")


def _build_panel_figure(event_spec: EventSpec, panel_spec: PanelSpec):
    figsize = _panel_figure_size(event_spec, panel_spec)
    needs_geo = any(slot.kind in {"gold", "overlay"} for slot in panel_spec.slots)
    use_geo = (needs_geo or event_spec.plot_defaults.use_cartopy) and ccrs is not None
    if needs_geo and ccrs is None:
        raise RuntimeError("Cartopy is required to render GOLD or overlay panels.")
    if use_geo:
        fig, axes = plt.subplots(
            panel_spec.rows,
            panel_spec.cols,
            figsize=figsize,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        fig, axes = plt.subplots(panel_spec.rows, panel_spec.cols, figsize=figsize)
    axes = np.asarray(axes, dtype=object).reshape(panel_spec.rows, panel_spec.cols)
    return fig, axes


def _panel_figure_size(event_spec: EventSpec, panel_spec: PanelSpec) -> tuple[float, float]:
    base_width, base_height = event_spec.plot_defaults.figure_size
    width = max(base_width * 0.64, 3.88 * panel_spec.cols)
    height = max(base_height * 0.84, 3.36 * panel_spec.rows)
    return (width, height)


def _add_shared_colorbar(fig, axes, event_spec: EventSpec, panel_spec: PanelSpec, overlay_metric: str) -> None:
    has_overlay_slots = any(slot.kind == "overlay" for slot in panel_spec.slots)
    if has_overlay_slots:
        _add_overlay_colorbars(fig, event_spec, overlay_metric)
        return
    if panel_spec.shared_colorbar == "gold":
        mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=300.0), cmap="viridis")
        label = GOLD_RADIANCE_LABEL
    else:
        metric = GNSS_PANEL_METRICS.get(panel_spec.shared_colorbar, "roti")
        style = event_spec.plot_defaults.gnss_styles[metric]
        mappable = ScalarMappable(norm=Normalize(vmin=style.vmin, vmax=style.vmax), cmap=style.cmap)
        label = metric_colorbar_label(metric)
    mappable.set_array([])
    cax = fig.add_axes([0.87, 0.15, 0.0105, 0.7])
    colorbar = fig.colorbar(mappable, cax=cax)
    colorbar.set_label(label, fontsize=PANEL_COLORBAR_FONT_SIZE, fontfamily=PANEL_FONT_FAMILY)
    colorbar.ax.tick_params(labelsize=PANEL_TICK_FONT_SIZE)
    plt.setp(colorbar.ax.get_yticklabels(), fontfamily=PANEL_FONT_FAMILY)


def _add_overlay_colorbars(fig, event_spec: EventSpec, overlay_metric: str) -> None:
    gold_mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=300.0), cmap="viridis")
    gold_mappable.set_array([])
    gold_cax = fig.add_axes([0.87, 0.56, 0.0105, 0.28])
    gold_colorbar = fig.colorbar(gold_mappable, cax=gold_cax)
    gold_colorbar.set_label(GOLD_RADIANCE_LABEL, fontsize=PANEL_COLORBAR_FONT_SIZE, fontfamily=PANEL_FONT_FAMILY)
    gold_colorbar.ax.tick_params(labelsize=PANEL_TICK_FONT_SIZE)
    plt.setp(gold_colorbar.ax.get_yticklabels(), fontfamily=PANEL_FONT_FAMILY)

    metric_style = event_spec.plot_defaults.gnss_styles.get(overlay_metric, event_spec.plot_defaults.gnss_styles["roti"])
    metric_mappable = ScalarMappable(norm=Normalize(vmin=metric_style.vmin, vmax=metric_style.vmax), cmap=metric_style.cmap)
    metric_mappable.set_array([])
    metric_cax = fig.add_axes([0.87, 0.16, 0.0105, 0.28])
    metric_colorbar = fig.colorbar(metric_mappable, cax=metric_cax)
    metric_colorbar.set_label(metric_colorbar_label(overlay_metric), fontsize=PANEL_COLORBAR_FONT_SIZE, fontfamily=PANEL_FONT_FAMILY)
    metric_colorbar.ax.tick_params(labelsize=PANEL_TICK_FONT_SIZE)
    plt.setp(metric_colorbar.ax.get_yticklabels(), fontfamily=PANEL_FONT_FAMILY)


def _set_slot_title(ax, title: str, index: int) -> None:
    label = string.ascii_lowercase[index]
    ax.set_title(f"({label}) {title}", loc="center", fontsize=PANEL_TITLE_FONT_SIZE, pad=3, fontfamily=PANEL_FONT_FAMILY)


def _draw_placeholder(ax, heading: str, detail: str) -> None:
    ax.set_axis_off()
    ax.set_facecolor("#f3f3f3")
    ax.text(
        0.5,
        0.56,
        heading,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=PANEL_TITLE_FONT_SIZE + 1.0,
        fontweight="bold",
        fontfamily=PANEL_FONT_FAMILY,
    )
    ax.text(
        0.5,
        0.44,
        detail,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=PANEL_COLORBAR_FONT_SIZE,
        fontfamily=PANEL_FONT_FAMILY,
    )


def _slot_requested_time(slot: PanelSlotSpec):
    if slot.kind in GNSS_PANEL_METRICS:
        return slot.timestamp
    if slot.kind == "gold":
        return _pair_midpoint(slot.gold_cha_time, slot.gold_chb_time)
    if slot.kind == "overlay":
        return slot.gnss_time
    return ""


def _gold_target_label(slot: PanelSlotSpec) -> str:
    return f"Target {_format_timestamp(_pair_midpoint(slot.gold_cha_time, slot.gold_chb_time))}"


def _overlay_target_label(slot: PanelSlotSpec) -> str:
    return (
        f"GOLD {_format_timestamp(_pair_midpoint(slot.gold_cha_time, slot.gold_chb_time))}\n"
        f"GNSS {_format_timestamp(slot.gnss_time)}"
    )


def _format_timestamp(value) -> str:
    if value is None:
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M UTC")


def _pair_midpoint(left_time, right_time):
    if left_time is None or right_time is None:
        return None
    left = pd.Timestamp(left_time)
    right = pd.Timestamp(right_time)
    return left + (right - left) / 2


def _as_utc_naive(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        return stamp.tz_convert(None)
    return stamp


class _GnssSliceResolver:
    def __init__(self, event_spec: EventSpec, products_by_producer: dict[str, list[GnssGridProduct]]) -> None:
        self.event_spec = event_spec
        self.products_by_producer = products_by_producer
        self._cache: dict[tuple[str, str], dict[pd.Timestamp, object]] = {}

    def resolve(self, producer: str, metric: str, timestamp) -> object | None:
        producer_key = str(producer).lower()
        metric_key = str(metric).lower()
        cache_key = (producer_key, metric_key)
        if cache_key not in self._cache:
            self._cache[cache_key] = self._load_slices(producer_key, metric_key)
        return self._cache[cache_key].get(_as_utc_naive(timestamp))

    def _load_slices(self, producer: str, metric: str) -> dict[pd.Timestamp, object]:
        slices: dict[pd.Timestamp, object] = {}
        for slice_data in iter_gnss_slices(self.products_by_producer.get(producer, []), metric):
            prepared = prepare_gnss_slice(slice_data, self.event_spec)
            if prepared is None:
                continue
            slices[_as_utc_naive(prepared.timestamp)] = prepared
        return slices


class _GoldPairResolver:
    def __init__(self, event_spec: EventSpec) -> None:
        self.event_spec = event_spec
        self._pair_lookup: dict[tuple[pd.Timestamp, pd.Timestamp], object] | None = None

    def resolve(self, cha_time, chb_time):
        if cha_time is None or chb_time is None:
            return None
        if self._pair_lookup is None:
            self._pair_lookup = self._build_lookup()
        return self._pair_lookup.get((_as_utc_naive(cha_time), _as_utc_naive(chb_time)))

    def _build_lookup(self) -> dict[tuple[pd.Timestamp, pd.Timestamp], object]:
        from ..internal import gold_core

        lookup: dict[tuple[pd.Timestamp, pd.Timestamp], object] = {}
        for asset in discover_gold_assets(self.event_spec):
            entries = gold_core.discover_entries(asset.local_path)
            pairs, _ = gold_core.match_pairs(entries, float(self.event_spec.runtime.get("gold_max_pair_minutes", 5)))
            for pair in pairs:
                key = (_as_utc_naive(pair.cha.obs_time), _as_utc_naive(pair.chb.obs_time))
                lookup.setdefault(key, pair)
        return lookup
