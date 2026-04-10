from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..models import EventSpec, OmniSeries
from .style import figure_style, set_axis_labels, set_axis_title, style_axis_ticks


OMNI_TITLE_FONT_SIZE = 16.0
OMNI_LABEL_FONT_SIZE = 14.5
OMNI_TICK_FONT_SIZE = 13.5
OMNI_SHADE_COLOR = "#dbeafe"
OMNI_SHADE_ALPHA = 0.35


def render_omni_series(event_spec: EventSpec, series: OmniSeries) -> Path:
    bz_frame = pd.read_csv(series.bz_csv_path)
    hourly_frame = pd.read_csv(series.hourly_csv_path)
    kp_frame = pd.read_csv(series.kp_csv_path)

    bz_frame["Time"] = pd.to_datetime(bz_frame["Time"], utc=True)
    hourly_frame["PlotTime"] = pd.to_datetime(hourly_frame["PlotTime"], utc=True)
    kp_frame["KpStart"] = pd.to_datetime(kp_frame["KpStart"], utc=True)

    with figure_style(event_spec.plot_defaults.font_family):
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12.0, 8.85),
            sharex=True,
            gridspec_kw={"height_ratios": [1.15, 1.0, 0.85]},
        )
        fig.subplots_adjust(top=0.985, bottom=0.09, left=0.125, right=0.985, hspace=0.19)

        ax_bz, ax_dst, ax_kp = axes
        _apply_highlight_windows(axes, event_spec)

        ax_bz.plot(
            bz_frame["Time"].dt.tz_convert("UTC").dt.tz_localize(None),
            bz_frame["IMF_Bz_nT"],
            color="#1f4e79",
            linewidth=0.95,
        )
        ax_bz.axhline(0, color="black", linewidth=0.8)
        set_axis_labels(
            ax_bz,
            font_family=event_spec.plot_defaults.font_family,
            ylabel="IMF Bz (nT)",
            fontsize=OMNI_LABEL_FONT_SIZE,
        )
        set_axis_title(
            ax_bz,
            "(a) IMF Bz (GSM, 1 min)",
            font_family=event_spec.plot_defaults.font_family,
            loc="left",
            pad=8,
            fontsize=OMNI_TITLE_FONT_SIZE,
        )

        ax_dst.plot(
            hourly_frame["PlotTime"].dt.tz_convert("UTC").dt.tz_localize(None),
            hourly_frame["Dst_nT"],
            color="#2f6f3e",
            linewidth=1.05,
        )
        ax_dst.axhline(-50, color="#7f1d1d", linewidth=0.9, linestyle="--")
        set_axis_labels(
            ax_dst,
            font_family=event_spec.plot_defaults.font_family,
            ylabel="Dst (nT)",
            fontsize=OMNI_LABEL_FONT_SIZE,
        )
        set_axis_title(
            ax_dst,
            "(b) Disturbance Storm Time Index",
            font_family=event_spec.plot_defaults.font_family,
            loc="left",
            pad=8,
            fontsize=OMNI_TITLE_FONT_SIZE,
        )

        kp_start = kp_frame["KpStart"].dt.tz_convert("UTC").dt.tz_localize(None)
        kp_values = kp_frame["Kp"].to_numpy(dtype=float)
        ax_kp.bar(
            kp_start,
            kp_values,
            width=3 / 24,
            align="edge",
            color=_kp_bar_colors(kp_values),
            edgecolor="black",
            linewidth=0.45,
            zorder=2.2,
        )
        set_axis_labels(
            ax_kp,
            font_family=event_spec.plot_defaults.font_family,
            ylabel="Kp",
            xlabel="Time (UTC)",
            fontsize=OMNI_LABEL_FONT_SIZE,
        )
        set_axis_title(
            ax_kp,
            "(c) Planetary Kp Index (3 h)",
            font_family=event_spec.plot_defaults.font_family,
            loc="left",
            pad=8,
            fontsize=OMNI_TITLE_FONT_SIZE,
        )
        ax_kp.set_ylim(0, max(9.2, float(np.nanmax(kp_values)) + 0.5))

        for axis in axes:
            axis.set_axisbelow(True)
            axis.grid(True, axis="y", linestyle="--", linewidth=0.55, color="#d0d0d0")
            axis.grid(True, axis="x", linestyle=":", linewidth=0.45, color="#e0e0e0")
            axis.tick_params(
                which="major",
                direction="in",
                top=True,
                right=True,
                length=6,
                width=0.9,
                labelsize=OMNI_TICK_FONT_SIZE,
            )
            axis.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.7)
            axis.minorticks_on()
            axis.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
            axis.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
            style_axis_ticks(
                axis,
                font_family=event_spec.plot_defaults.font_family,
                labelsize=OMNI_TICK_FONT_SIZE,
            )

        fig.align_ylabels(axes)

        output_path = event_spec.storage.figures_omni_dir / f"omni_timeseries_{event_spec.event_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=event_spec.plot_defaults.dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path


def _apply_highlight_windows(axes, event_spec: EventSpec) -> None:
    for window in event_spec.omni_highlight_windows():
        start = pd.Timestamp(window.start_utc).tz_convert("UTC").tz_localize(None)
        end = pd.Timestamp(window.end_utc).tz_convert("UTC").tz_localize(None)
        for axis in axes:
            axis.axvspan(
                start,
                end,
                color=window.color or OMNI_SHADE_COLOR,
                alpha=window.alpha if window.alpha is not None else OMNI_SHADE_ALPHA,
                linewidth=0,
                zorder=0.1,
            )


def _kp_bar_colors(values: np.ndarray) -> list[str]:
    colors: list[str] = []
    for value in values:
        if np.isnan(value):
            colors.append("#6b7280")
        elif value < 4.0:
            colors.append("#2e8b57")
        elif value < 5.0:
            colors.append("#f59e0b")
        else:
            colors.append("#c1121f")
    return colors
