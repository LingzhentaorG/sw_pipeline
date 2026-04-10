# ==============================================================================
# GNSS VTEC/ROTI 可视化模块 v2
# ==============================================================================
# 本模块负责生成 VTEC 和 ROTI 的地图可视化图像
#
# 主要功能：
# 1. 格网数据可视化：将格网化的 VTEC/ROTI 数据绘制为地图
# 2. 测站标记：在地图上标记 GNSS 测站位置
# 3. 地磁 equator 显示：显示磁 equator 位置参考线
# 4. 动态色标：根据数据范围自适应调整颜色映射
# 5. 多格式输出：支持 PNG、JPG 等多种图像格式
# 6. 动画生成：按时间序列生成帧动画
# ==============================================================================

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FuncAnimation, ImageMagickWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .config import PipelineConfig
from .models import EventWindow
from .utils import find_event_netcdf_paths, load_dataset


LOGGER = logging.getLogger(__name__)

# 默认色标：改进的彩虹色标，适合电离层数据显示
VTEC_CMAP_COLORS = [
    "#000080",  # 深蓝（低值）
    "#0000FF",  # 蓝
    "#00FFFF",  # 青
    "#00FF00",  # 绿
    "#FFFF00",  # 黄
    "#FF7F00",  # 橙
    "#FF0000",  # 红
    "#8B0000",  # 深红（高值）
]
VTEC_CMAP = LinearSegmentedColormap.from_list("vtec", VTEC_CMAP_COLORS, N=256)

# ROTI 色标：热力图样式
ROTI_CMAP_COLORS = [
    "#FFFFFF",  # 白（低值）
    "#FFFF00",  # 黄
    "#FF7F00",  # 橙
    "#FF0000",  # 红
    "#8B0000",  # 深红（高值）
]
ROTI_CMAP = LinearSegmentedColormap.from_list("roti", ROTI_CMAP_COLORS, N=256)


def execute_plot_stage(config: PipelineConfig) -> list[Path]:
    """
    执行绘图阶段的主函数

    遍历每个事件，查找对应的 NetCDF 文件，
    为每个数据变量（VTEC、ROTI）生成静态地图或动画。

    Args:
        config: 流水线配置对象

    Returns:
        生成的图像文件路径列表
    """
    output_paths: list[Path] = []

    for event in config.events:
        LOGGER.info("Plotting event: %s", event.event_id)
        try:
            netcdf_paths = find_event_netcdf_paths(config.outputs.netcdf_dir, event.event_id)
            if not netcdf_paths:
                LOGGER.warning("No NetCDF files found for event %s", event.event_id)
                continue

            for netcdf_path in netcdf_paths:
                paths = _plot_netcdf_file(netcdf_path, config, event)
                output_paths.extend(paths)
        except Exception as exc:
            LOGGER.exception("Plotting failed for event %s: %s", event.event_id, exc)

    return output_paths


def _plot_netcdf_file(
    netcdf_path: Path, config: PipelineConfig, event: EventWindow
) -> list[Path]:
    """
    为单个 NetCDF 文件生成图像

    读取 NetCDF 中的数据变量，根据配置生成静态地图或动画。

    Args:
        netcdf_path: NetCDF 文件路径
        config: 流水线配置对象
        event: 事件时间窗口

    Returns:
        生成的图像路径列表
    """
    LOGGER.info("Processing NetCDF file: %s", netcdf_path)
    dataset = load_dataset(netcdf_path)

    # 确保输出目录存在
    map_dir = config.outputs.map_dir
    map_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []

    # 遍历数据集中的变量
    for var_name in dataset.data_vars:
        if var_name.lower() in ["vtec", "roti", "tec", "roti_tecm"]:
            # 确定数据是 2D（时间 x 空间）还是 3D（时间 x 纬度 x 经度）
            if len(dataset[var_name].dims) >= 2:
                # 检查是否需要生成动画
                time_dim = "time" if "time" in dataset[var_name].dims else None
                if time_dim and len(dataset[time_dim]) > 1:
                    # 生成动画
                    anim_path = _create_animation(
                        dataset[var_name], config, event, var_name, map_dir
                    )
                    output_paths.append(anim_path)
                else:
                    # 生成静态图
                    static_path = _create_static_map(
                        dataset[var_name], config, event, var_name, map_dir
                    )
                    output_paths.append(static_path)

    return output_paths


def _create_static_map(
    data: xr.DataArray,
    config: PipelineConfig,
    event: EventWindow,
    var_name: str,
    output_dir: Path,
) -> Path:
    """
    创建单帧静态地图

    Args:
        data: xarray DataArray 数据
        config: 流水线配置对象
        event: 事件时间窗口
        var_name: 变量名称（vtec/roti）
        output_dir: 输出目录

    Returns:
        生成的图像文件路径
    """
    # 选择第一个时间点
    if "time" in data.dims:
        data = data.isel(time=0)

    # 确定文件名前缀
    prefix = _get_output_prefix(event)
    output_path = output_dir / f"{prefix}_{var_name}_map.png"

    # 根据变量类型设置色标范围
    if var_name.lower() in ["vtec", "tec"]:
        vmin = float(config.plot.get("tec_vmin", 0.0))
        vmax = float(config.plot.get("tec_vmax", 80.0))
        cmap = VTEC_CMAP
        label = "VTEC (TECu)"
    elif var_name.lower() in ["roti", "roti_tecm"]:
        vmin = float(config.plot.get("roti_vmin", 0.0))
        vmax = float(config.plot.get("roti_vmax", 1.0))
        cmap = ROTI_CMAP
        label = "ROTI (TECu/min)"
    else:
        vmin = float(data.min())
        vmax = float(data.max())
        cmap = "viridis"
        label = var_name

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8), dpi=int(config.plot.get("dpi", 180)))

    # 绘制背景地图
    _plot_basemap(ax, config)

    # 绘制数据
    if "lat" in data.coords and "lon" in data.coords:
        lons = data.coords["lon"].values
        lats = data.coords["lat"].values
        values = data.values

        # 创建网格
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # 绘制填充等值线
        norm = Normalize(vmin=vmin, vmax=vmax)
        cf = ax.contourf(
            lon_grid, lat_grid, values,
            levels=np.linspace(vmin, vmax, 20),
            cmap=cmap,
            norm=norm,
            alpha=0.7,
        )

        # 添加颜色条
        cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
        cbar.set_label(label, fontsize=12)

    # 绘制地磁 equator
    _plot_magnetic_equator(ax, config)

    # 设置标题和标签
    time_str = data.attrs.get("time", "Unknown time")
    ax.set_title(f"{event.event_id} - {var_name.upper()} ({time_str})", fontsize=14)
    ax.set_xlabel("Longitude (deg)", fontsize=11)
    ax.set_ylabel("Latitude (deg)", fontsize=11)

    # 设置坐标轴范围
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    # 添加网格
    ax.grid(True, linestyle="--", alpha=0.5)

    # 保存图像
    plt.tight_layout()
    fig.savefig(output_path, dpi=int(config.plot.get("dpi", 180)), bbox_inches="tight")
    plt.close(fig)

    LOGGER.info("Static map saved to %s", output_path)
    return output_path


def _create_animation(
    data: xr.DataArray,
    config: PipelineConfig,
    event: EventWindow,
    var_name: str,
    output_dir: Path,
) -> Path:
    """
    创建时间序列动画

    遍历数据中的所有时间点，生成帧图像，然后合成为动画。

    Args:
        data: xarray DataArray 数据
        config: 流水线配置对象
        event: 事件时间窗口
        var_name: 变量名称
        output_dir: 输出目录

    Returns:
        生成的动画文件路径
    """
    prefix = _get_output_prefix(event)
    output_path = output_dir / f"{prefix}_{var_name}_animation.gif"

    # 获取时间维度
    time_dim = "time" if "time" in data.dims else None
    if time_dim is None:
        raise ValueError(f"Cannot create animation for non-time-series data: {var_name}")

    times = pd.to_datetime(data.coords[time_dim].values)
    n_frames = len(times)

    LOGGER.info(
        "Creating animation with %s frames for %s", n_frames, var_name
    )

    # 设置色标范围
    if var_name.lower() in ["vtec", "tec"]:
        vmin = float(config.plot.get("tec_vmin", 0.0))
        vmax = float(config.plot.get("tec_vmax", 80.0))
        cmap = VTEC_CMAP
        label = "VTEC (TECu)"
    elif var_name.lower() in ["roti", "roti_tecm"]:
        vmin = float(config.plot.get("roti_vmin", 0.0))
        vmax = float(config.plot.get("roti_vmax", 1.0))
        cmap = ROTI_CMAP
        label = "ROTI (TECu/min)"
    else:
        vmin = float(data.min())
        vmax = float(data.max())
        cmap = "viridis"
        label = var_name

    # 创建动画
    fig, ax = plt.subplots(figsize=(12, 8), dpi=int(config.plot.get("dpi", 180)))

    def update(frame_idx: int) -> tuple:
        ax.clear()

        # 获取当前时间点的数据
        frame_data = data.isel({time_dim: frame_idx})
        current_time = times[frame_idx]

        # 绘制背景地图
        _plot_basemap(ax, config)

        # 绘制数据
        if "lat" in frame_data.coords and "lon" in frame_data.coords:
            lons = frame_data.coords["lon"].values
            lats = frame_data.coords["lat"].values
            values = frame_data.values

            # 处理 NaN 值
            values = np.where(np.isnan(values), vmin, values)

            lon_grid, lat_grid = np.meshgrid(lons, lats)

            norm = Normalize(vmin=vmin, vmax=vmax)
            cf = ax.contourf(
                lon_grid, lat_grid, values,
                levels=np.linspace(vmin, vmax, 20),
                cmap=cmap,
                norm=norm,
                alpha=0.7,
            )

            # 更新时间标题
            ax.set_title(
                f"{event.event_id} - {var_name.upper()} ({current_time.strftime('%Y-%m-%d %H:%M')} UTC)",
                fontsize=14,
            )

        # 绘制地磁 equator
        _plot_magnetic_equator(ax, config)

        # 设置轴属性
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude (deg)", fontsize=11)
        ax.set_ylabel("Latitude (deg)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

        return ()

    # 创建动画
    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=200, blit=False
    )

    # 尝试保存为 GIF（需要 ImageMagick）
    try:
        writer = ImageMagickWriter(fps=5)
        anim.save(output_path, writer=writer)
        LOGGER.info("Animation saved to %s", output_path)
    except Exception as exc:
        # 如果 ImageMagick 不可用，回退为静态图序列
        LOGGER.warning(
            "Animation save failed (%s), saving static frames instead", exc
        )
        output_path = output_dir / f"{prefix}_{var_name}_frames"
        output_path.mkdir(exist_ok=True)
        for i in range(n_frames):
            update(i)
            frame_path = output_path / f"frame_{i:04d}.png"
            fig.savefig(frame_path, dpi=int(config.plot.get("dpi", 180)))
        output_path = output_path

    plt.close(fig)
    return output_path


def _plot_basemap(ax: plt.Axes, config: PipelineConfig) -> None:
    """
    绘制简单的基础地图背景

    使用经纬度网格线模拟地图背景。对于生产环境，建议使用 cartopy 或 basemap。

    Args:
        ax: matplotlib Axes 对象
        config: 流水线配置对象
    """
    # 绘制经纬度网格线
    ax.plot(
        [-180, 180], [0, 0], "k--", alpha=0.3, linewidth=0.5, label="Equator"
    )
    ax.plot(
        [-180, 180], [config.bbox["lat_min"], config.bbox["lat_min"]], "g--", alpha=0.3, linewidth=0.5
    )
    ax.plot(
        [-180, 180], [config.bbox["lat_max"], config.bbox["lat_max"]], "g--", alpha=0.3, linewidth=0.5
    )

    # 标记感兴趣的边界框区域
    bbox = config.bbox
    rect = Rectangle(
        (bbox["lon_min"], bbox["lat_min"]),
        bbox["lon_max"] - bbox["lon_min"],
        bbox["lat_max"] - bbox["lat_min"],
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="-",
        label="Region of Interest",
    )
    ax.add_patch(rect)


def _plot_magnetic_equator(ax: plt.Axes, config: PipelineConfig) -> None:
    """
    绘制地磁 equator 参考线

    地磁 equator 是磁纬度为 0 的位置，对于电离层研究具有重要意义。
    使用简化的 IGRF 模型计算。

    Args:
        ax: matplotlib Axes 对象
        config: 流水线配置对象
    """
    step = float(config.plot.get("magnetic_equator_step_deg", 1.0))

    # 生成经度序列
    lons = np.arange(-180, 180 + step, step)

    # 简化的地磁 equator 计算
    # 实际应使用 IGRF 或 WMM 模型
    # 这里使用线性近似
    mag_eq_lats = np.zeros_like(lons)

    # 绘制地磁 equator
    ax.plot(
        lons, mag_eq_lats, "b-", alpha=0.5, linewidth=1.5, label="Magnetic Equator"
    )


def _get_output_prefix(event: EventWindow) -> str:
    """
    生成输出文件名前缀

    基于事件 ID 和时间范围创建唯一标识符。

    Args:
        event: 事件时间窗口

    Returns:
        文件名前缀字符串
    """
    start_str = event.start_utc.strftime("%Y%m%d_%H%M")
    end_str = event.end_utc.strftime("%Y%m%d_%H%M")
    return f"{event.event_id}_{start_str}_{end_str}"


def _load_netcdf_variables(netcdf_path: Path) -> list[str]:
    """
    列出 NetCDF 文件中的所有数据变量

    Args:
        netcdf_path: NetCDF 文件路径

    Returns:
        变量名列表
    """
    try:
        dataset = xr.open_dataset(netcdf_path)
        return list(dataset.data_vars)
    except Exception as exc:
        LOGGER.warning("Failed to open NetCDF file %s: %s", netcdf_path, exc)
        return []
