from __future__ import annotations

from contextlib import contextmanager

import matplotlib
import matplotlib.pyplot as plt


TITLE_FONT_SIZE = 12.5
LABEL_FONT_SIZE = 11.5
TICK_FONT_SIZE = 11.0
SUPTITLE_FONT_SIZE = 16.0
SUBTITLE_FONT_SIZE = 12.5
LEGEND_FONT_SIZE = 10.5
GOLD_RADIANCE_LABEL = "135.6 nm Radiance (Rayleigh)"
VTEC_UNIT_LABEL = "TECU"
ROTI_UNIT_LABEL = "10^16 el/m^2/min"


@contextmanager
def figure_style(font_family: str):
    with matplotlib.rc_context({"font.family": [font_family], "font.size": TICK_FONT_SIZE}):
        yield


def style_colorbar(colorbar, label: str, *, font_family: str) -> None:
    colorbar.set_label(label, fontsize=LABEL_FONT_SIZE, fontfamily=font_family)
    colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    plt.setp(colorbar.ax.get_yticklabels(), fontfamily=font_family)


def style_axis_ticks(ax, *, font_family: str, labelsize: float = TICK_FONT_SIZE) -> None:
    ax.tick_params(labelsize=labelsize)
    plt.setp(ax.get_xticklabels(), fontfamily=font_family)
    plt.setp(ax.get_yticklabels(), fontfamily=font_family)


def set_axis_title(
    ax,
    title: str,
    *,
    font_family: str,
    loc: str = "center",
    pad: float = 6.0,
    fontsize: float = TITLE_FONT_SIZE,
) -> None:
    ax.set_title(title, loc=loc, fontsize=fontsize, pad=pad, fontfamily=font_family)


def set_axis_labels(
    ax,
    *,
    font_family: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    fontsize: float = LABEL_FONT_SIZE,
    fontweight: str | None = None,
) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily=font_family, fontweight=fontweight)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily=font_family, fontweight=fontweight)


def metric_unit_label(metric: str) -> str:
    lowered = str(metric).lower()
    if lowered == "vtec":
        return VTEC_UNIT_LABEL
    if lowered == "roti":
        return ROTI_UNIT_LABEL
    return ""


def metric_colorbar_label(metric: str) -> str:
    lowered = str(metric).lower()
    unit = metric_unit_label(lowered)
    if lowered == "vtec":
        return f"VTEC ({unit})" if unit else "VTEC"
    if lowered == "roti":
        return f"ROTI ({unit})" if unit else "ROTI"
    upper = str(metric).upper()
    return f"{upper} ({unit})" if unit else upper


def metric_threshold_label(metric: str, threshold: float, count: int) -> str:
    lowered = str(metric).lower()
    unit = metric_unit_label(lowered)
    upper = "VTEC" if lowered == "vtec" else "ROTI" if lowered == "roti" else str(metric).upper()
    if unit:
        return f"{upper} > {threshold:.1f} {unit} (n={count})"
    return f"{upper} > {threshold:.1f} (n={count})"


def overlay_ylabel(metric: str) -> str:
    return f"GOLD + High {metric_colorbar_label(metric)}"
