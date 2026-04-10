# ==============================================================================
# GNSS VTEC/ROTI 流水线调度模块
# ==============================================================================
# 本模块负责流水线的阶段调度和流程控制
# 根据用户命令（download/process/plot/all）调用相应的处理函数
# 并在各阶段之间进行必要的依赖检查和状态验证
# ==============================================================================

from __future__ import annotations

import logging

from .config import load_pipeline_config
from .download import execute_download_stage
from .plotting_v2 import execute_plot_stage
from .preprocess import preprocess_records
from .processing_v2 import execute_processing_stage
from .utils import configure_logging, find_event_netcdf_paths


LOGGER = logging.getLogger(__name__)


def run_pipeline_command(command: str, config_path: str) -> None:
    """
    根据命令执行相应的流水线阶段

    本函数是流水线的核心调度器，根据传入的命令参数执行不同的处理阶段：
    - download: 下载 GNSS 观测数据和辅助产品
    - process: 预处理 + VTEC/ROTI 处理（如果缺少下载清单会自动触发下载）
    - plot: 生成地图图像（如果缺少 NetCDF 会自动触发完整处理）
    - all: 执行完整流程

    Args:
        command: 流水线命令，支持 download/process/plot/all
        config_path: YAML 配置文件路径
    """
    # 加载配置文件
    config = load_pipeline_config(config_path)
    # 配置日志系统，输出到指定日志文件
    configure_logging(config.outputs.log_dir / "pipeline.log")

    # 根据命令执行相应阶段
    if command == "download":
        # 下载阶段：发现测站并下载观测文件和辅助产品
        execute_download_stage(config)
        return

    if command == "process":
        # 处理阶段：确保已下载数据，执行预处理和处理
        _ensure_downloads(config)
        preprocess_records(config)
        execute_processing_stage(config)
        return

    if command == "plot":
        # 绘图阶段：确保已处理数据，执行绘图
        _ensure_processed(config)
        execute_plot_stage(config)
        return

    if command == "all":
        # 全流程：依次执行下载、预处理、处理、绘图
        execute_download_stage(config)
        preprocess_records(config)
        execute_processing_stage(config)
        execute_plot_stage(config)
        return

    # 如果命令不支持，抛出异常
    raise ValueError(f"Unsupported command: {command}")


def _ensure_downloads(config) -> None:
    """
    确保下载阶段已完成

    检查是否存在观测数据下载清单（observation_manifest.csv），
    如果不存在则自动触发下载阶段。这是处理阶段的前置检查，
    确保在进行数据处理之前已经有可用的数据。

    Args:
        config: 流水线配置对象
    """
    # 下载清单文件路径
    manifest = config.outputs.manifests_dir / "observation_manifest.csv"
    if not manifest.exists():
        LOGGER.info("Download manifest missing, executing download stage first.")
        # 清单不存在，触发下载阶段
        execute_download_stage(config)


def _ensure_processed(config) -> None:
    """
    确保处理阶段已完成

    检查所有事件是否都有对应的 NetCDF 输出文件，
    如果有任何事件的 NetCDF 缺失，则自动触发完整的数据处理流程：
    下载 -> 预处理 -> 处理。这是绘图阶段的前置检查，
    确保在生成地图之前已经有可用的处理结果。

    Args:
        config: 流水线配置对象
    """
    # 检查每个事件是否都有 NetCDF 文件
    netcdf_missing = any(
        not find_event_netcdf_paths(config.outputs.netcdf_dir, event.event_id)
        for event in config.events
    )
    if netcdf_missing:
        LOGGER.info("NetCDF missing for some events, executing processing stage first.")
        # 触发完整的处理流程
        _ensure_downloads(config)
        preprocess_records(config)
        execute_processing_stage(config)
