# ==============================================================================
# GNSS 流水线数据模型模块
# ==============================================================================
# 本模块定义了 GNSS 数据处理管道中使用的数据模型（dataclass）
# 包括：事件窗口、测站信息、下载记录、处理结果等核心数据结构
# 这些模型提供了类型安全的数据封装，便于在模块间传递和处理
# ==============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class EventWindow:
    """
    事件时间窗口数据模型

    表示一个监测事件的时间范围，包含唯一标识符和起止时间。
    用于定义 GNSS 数据处理的分析时段。

    Attributes:
        event_id: 事件的唯一标识符（如 "storm_2024_001"）
        start_utc: 事件开始时间（UTC 时区）
        end_utc: 事件结束时间（UTC 时区）
    """
    event_id: str
    start_utc: datetime
    end_utc: datetime


@dataclass
class StationInfo:
    """
    GNSS 测站信息数据模型

    存储测站的基本信息，包括标识符、地理位置、天线配置等。
    从 RINEX 文件头或数据源元数据中提取。

    Attributes:
        station_id: 测站唯一标识符（如 "ABPO"）
        station_code4: 4 位测站代码
        lat: 测站纬度（度）
        lon: 测站经度（度）
        height_m: 测站椭球高（米）
        antenna_delta_h: 天线高（米，相对于测站标石）
        antenna_delta_e: 天线东向偏移（米）
        antenna_delta_n: 天线北向偏移（米）
        antenna_type: 天线类型描述
        receiver_type: 接收机类型描述
    """
    station_id: str
    station_code4: str
    lat: float
    lon: float
    height_m: float
    antenna_delta_h: float = 0.0
    antenna_delta_e: float = 0.0
    antenna_delta_n: float = 0.0
    antenna_type: str = ""
    receiver_type: str = ""


@dataclass
class SourceSettings:
    """
    数据源配置数据模型

    定义 GNSS 数据源的连接参数和行为设置，
    包括 URL、认证、超时、优先级等配置项。

    Attributes:
        name: 数据源名称（如 "noaa", "rbmc"）
        enabled: 是否启用该数据源
        priority: 优先级（数值越小优先级越高）
        timeout_sec: 请求超时时间（秒）
        params: 其他数据源特定参数字典
    """
    name: str
    enabled: bool = True
    priority: int = 100
    timeout_sec: int = 60
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DownloadRecord:
    """
    下载记录数据模型

    表示一次 GNSS 观测文件或导航文件的下载任务。
    包含源信息、URL、目标路径、状态和坐标等。

    Attributes:
        event_id: 所属事件的唯一标识符
        source: 数据源名称
        source_priority: 数据源优先级
        observation_date: 观测日期（ISO 格式字符串）
        station_id: 测站标识符
        station_code4: 4 位测站代码
        sampling_sec: 采样间隔（秒）
        obs_url: 观测文件下载 URL
        nav_url: 导航文件下载 URL
        lat: 测站纬度（度，可选）
        lon: 测站经度（度，可选）
        height_m: 测站高程（米，可选）
        obs_path: 观测文件本地路径（下载后填充）
        obs_status: 观测文件下载状态（ok/error/skipped）
        obs_error: 观测文件下载错误信息
        nav_path: 导航文件本地路径（下载后填充）
        nav_status: 导航文件下载状态
        nav_error: 导航文件下载错误信息
    """
    event_id: str
    source: str
    source_priority: int
    observation_date: str
    station_id: str
    station_code4: str
    sampling_sec: int
    obs_url: str
    nav_url: str
    lat: float = 0.0
    lon: float = 0.0
    height_m: float = 0.0
    obs_path: Optional[str] = None
    obs_status: str = "pending"
    obs_error: Optional[str] = None
    nav_path: Optional[str] = None
    nav_status: str = "pending"
    nav_error: Optional[str] = None


@dataclass
class AuxDownloadRecord:
    """
    辅助产品下载记录数据模型

    表示一次辅助产品（如 SP3、DCB、IONEX）的下载任务。
    包含产品类型、提供商、优先级和状态等信息。

    Attributes:
        observation_date: 观测日期
        product_type: 产品类型（sp3/dcb/ionex）
        provider: 提供商名称
        priority: 下载优先级
        url: 下载 URL
        path: 本地文件路径
        status: 下载状态（ok/error）
        error: 错误信息（如果失败）
        attempts: 下载尝试次数
        metadata: 附加元数据字典（如 GPS 周、周内日）
    """
    observation_date: str
    product_type: str
    provider: str
    priority: int
    url: str
    auth_ref: Optional[str] = None
    verify_ssl: bool = True
    path: Optional[str] = None
    status: str = "pending"
    error: Optional[str] = None
    attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedObservationFile:
    """
    标准化观测文件数据模型

    表示经过预处理后的 GNSS 观测文件信息。
    包含文件路径、测站信息、解析结果等。

    Attributes:
        event_id: 事件 ID
        observation_date: 观测日期
        source: 数据来源
        station_id: 测站标识符
        station_code4: 4 位测站代码
        sampling_sec: 采样间隔
        obs_path: 观测文件路径
        nav_path: 导航文件路径
        obs_result: 观测文件解析结果
        nav_result: 导航文件解析结果
        approx_pos_ecef: 近似位置（ECEF 坐标）
    """
    event_id: str
    observation_date: str
    source: str
    station_id: str
    station_code4: str
    sampling_sec: int
    obs_path: str
    nav_path: str
    obs_result: Any = None
    nav_result: Any = None
    approx_pos_ecef: Optional[tuple[float, float, float]] = None


@dataclass
class PreprocessFailure:
    """
    预处理失败记录数据模型

    记录在 GNSS 数据预处理阶段失败的案例，
    包括文件路径、错误类型和详细信息。

    Attributes:
        event_id: 事件 ID
        observation_date: 观测日期
        station_id: 测站标识符
        obs_path: 观测文件路径
        nav_path: 导航文件路径
        error: 错误信息描述
    """
    event_id: str
    observation_date: str
    station_id: str
    obs_path: str
    nav_path: str
    error: str


@dataclass
class ArcSegmentation:
    """
    弧段分割数据模型

    表示 GNSS 卫星观测的连续弧段信息。
    弧段是卫星在测站可视范围内连续观测的时间序列。

    Attributes:
        station_id: 测站标识符
        sat_id: 卫星标识符
        arc_id: 弧段序号
        start_time: 弧段开始时间
        end_time: 弧段结束时间
        start_gps_second: 弧段开始 GPS 秒数
        end_gps_second: 弧段结束 GPS 秒数
        num_epochs: 弧段内的观测历元数
        mean_elevation: 平均卫星仰角
        is_gap_truncated: 是否因数据间隙被截断
        gap_indices: 数据间隙所在位置索引
    """
    station_id: str
    sat_id: str
    arc_id: int
    start_time: datetime
    end_time: datetime
    start_gps_second: float
    end_gps_second: float
    num_epochs: int
    mean_elevation: float
    is_gap_truncated: bool = False
    gap_indices: list[int] = field(default_factory=list)


@dataclass
class CycleSlipRecord:
    """
    周跳记录数据模型

    记录检测到的周跳事件，包括位置信息和检测方法。

    Attributes:
        station_id: 测站标识符
        sat_id: 卫星标识符
        timestamp: 周跳发生时间
        gps_second: GPS 秒数
        epoch_index: 弧段内历元索引
        method: 检测方法（mw/gf/combined）
        mw_combination: MW 组合值变化
        gf_combination: GF 组合值变化
        is_repaired: 是否已修复
    """
    station_id: str
    sat_id: str
    timestamp: datetime
    gps_second: float
    epoch_index: int
    method: str
    mw_combination: float = 0.0
    gf_combination: float = 0.0
    is_repaired: bool = False


@dataclass
class StationVTEC:
    """
    测站 VTEC 数据模型

    表示单个测站的垂直总电子含量（VTEC）测量或建模结果。
    包含时间、位置、STEC、VTEC 及相关参数。

    Attributes:
        event_id: 事件 ID
        timestamp: 观测时间戳
        station_id: 测站标识符
        station_lat: 测站纬度
        station_lon: 测站经度
        station_height_m: 测站高程
        ipp_lat: 电离层穿透点纬度
        ipp_lon: 电离层穿透点经度
        ipp_height_m: 穿透点高程
        sat_id: 卫星标识符
        azimuth_deg: 卫星方位角
        elevation_deg: 卫星仰角
        stec_tecu: 斜向总电子含量（TECu）
        vtec_tecu: 垂直总电子含量（TECu）
        vtec_model: VTEC 模型名称
        is_valid: 数据有效性标志
    """
    event_id: str
    timestamp: pd.Timestamp
    station_id: str
    station_lat: float
    station_lon: float
    station_height_m: float
    ipp_lat: float
    ipp_lon: float
    ipp_height_m: float
    sat_id: str
    azimuth_deg: float
    elevation_deg: float
    stec_tecu: float
    vtec_tecu: float
    vtec_model: str
    is_valid: bool = True


@dataclass
class GridCellStatistics:
    """
    格网单元统计数据模型

    表示某个时空格网单元内的 VTEC/ROTI 统计结果。

    Attributes:
        time: 格网中心时间
        lat_min: 纬度下界
        lat_max: 纬度上界
        lon_min: 经度下界
        lon_max: 经度上界
        vtec_mean: VTEC 平均值
        vtec_median: VTEC 中位数
        vtec_std: VTEC 标准差
        vtec_count: 有效观测数
        arc_count: 参与统计的弧段数
    """
    time: pd.Timestamp
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    vtec_mean: float = float("nan")
    vtec_median: float = float("nan")
    vtec_std: float = 0.0
    vtec_count: int = 0
    arc_count: int = 0


@dataclass
class OutputPaths:
    """
    输出路径管理数据模型

    集中管理 GNSS 流水线所有输出文件的目录路径。
    按照数据类型和处理阶段组织目录结构。

    Attributes:
        root: 根输出目录
        raw_dir: 原始下载文件目录
        normalized_dir: 标准化数据目录
        aux_dir: 辅助产品目录
        manifests_dir: 下载清单目录
        intermediate_dir: 中间结果目录
        availability_dir: 数据可用性分析目录
        arc_dir: 弧段数据目录
        stec_dir: STEC 数据目录
        vtec_dir: VTEC 数据目录
        roti_dir: ROTI 数据目录
        grid_dir: 格网化数据目录
        validation_dir: 验证结果目录
        product_dir: 最终产品目录
        netcdf_dir: NetCDF 输出目录
        map_dir: 图像输出目录
        log_dir: 日志文件目录
        legacy_root: 旧版输出根目录（兼容）
        legacy_raw_dir: 旧版原始数据目录
        legacy_normalized_dir: 旧版标准化数据目录
    """
    root: Path
    raw_dir: Path
    normalized_dir: Path
    aux_dir: Path
    manifests_dir: Path
    intermediate_dir: Path
    availability_dir: Path
    arc_dir: Path
    stec_dir: Path
    vtec_dir: Path
    roti_dir: Path
    grid_dir: Path
    validation_dir: Path
    product_dir: Path
    netcdf_dir: Path
    map_dir: Path
    log_dir: Path
    legacy_root: Path
    legacy_raw_dir: Path
    legacy_normalized_dir: Path
