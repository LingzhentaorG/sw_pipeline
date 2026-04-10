# ==============================================================================
# GNSS 流水线辅助数据处理模块
# ==============================================================================
# 本模块负责处理 GNSS 数据管线所需的辅助数据产品
#
# 支持的辅助数据类型：
# 1. SP3：精密星历产品（卫星位置和时钟）
# 2. DCB：差分码偏差产品（用于伪距校正）
# 3. IONEX：电离层格网产品（GIM，用于 VTEC 建模）
#
# 主要功能：
# - 从各种来源下载和缓存辅助产品
# - 解析 SP3、DCB、IONEX 格式文件
# - 提供统一的辅助数据访问接口
# ==============================================================================

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import BinaryIO, Iterator

import pandas as pd

from .utils import (
    GPS_EPOCH_START,
    SECONDS_IN_DAY,
    date_to_doy,
    gps_week_and_dow,
    load_bytes_maybe_compressed,
    open_text_maybe_compressed,
)


LOGGER = logging.getLogger(__name__)


# GPS 卫星的标称频率（Hz）
GPS_L1_FREQ = 1575.42e6
GPS_L2_FREQ = 1227.60e6
GPS_L5_FREQ = 1176.45e6

# Galileo 卫星频率（Hz）
GALILEO_E1_FREQ = 1575.42e6
GALILEO_E5A_FREQ = 1176.45e6
GALILEO_E6_FREQ = 1278.75e6

# BeiDou 卫星频率（Hz）
BEIDOU_B1_FREQ = 1561.098e6
BEIDOU_B2_FREQ = 1207.14e6
BEIDOU_B3_FREQ = 1268.52e6

# 电离层薄壳模型高度（km）
IONOSPHERIC_SHELL_HEIGHT_KM = 350.0

# 地球椭球长半轴（米），WGS84
EARTH_EQUATORIAL_RADIUS_M = 6378137.0

# 地球椭球扁率
EARTH_FLATTENING = 1.0 / 298.257223563

# WGS84 第一偏心率平方
WGS84_E2 = 2 * EARTH_FLATTENING - EARTH_FLATTENING ** 2

# 太阳光速（米/秒）
SPEED_OF_LIGHT = 299792458.0

# 电离层常数 K，用于 STEC 计算
# K = 40.308 * 10^16 （当频率单位为 Hz 时）
IONOSPHERIC_K = 40.308e16


def get_frequency_for_satellite(sat_sys: str, band: str) -> float:
    """
    根据卫星系统和频段获取标称频率

    Args:
        sat_sys: 卫星系统标识（G=GPS, E=Galileo, C=BeiDou, R=GLONASS）
        band: 频段标识（L1, L2, L5, E1, E5a, B1, B2, ...）

    Returns:
        频率值（Hz）
    """
    band = band.upper()
    if sat_sys == "G":  # GPS
        if band in ("L1", "L1C", "L1P"):
            return GPS_L1_FREQ
        elif band in ("L2", "L2C", "L2P"):
            return GPS_L2_FREQ
        elif band == "L5":
            return GPS_L5_FREQ
    elif sat_sys == "E":  # Galileo
        if band in ("E1", "L1"):
            return GALILEO_E1_FREQ
        elif band in ("E5A", "L5"):
            return GALILEO_E5A_FREQ
        elif band == "E6":
            return GALILEO_E6_FREQ
    elif sat_sys == "C":  # BeiDou
        if band == "B1":
            return BEIDOU_B1_FREQ
        elif band == "B2":
            return BEIDOU_B2_FREQ
        elif band == "B3":
            return BEIDOU_B3_FREQ
    return GPS_L1_FREQ  # 默认返回 L1 频率


def calculate_stec_from_geometry_free(
    P1: float, P2: float, f1: float, f2: float
) -> float:
    """
    使用几何无关组合计算 STEC

    公式：STEC = (P2 - P1) / K * (f1^2 * f2^2) / (f1^2 - f2^2)

    Args:
        P1: P1 伪距观测值（米）
        P2: P2 伪距观测值（米）
        f1: L1 频率（Hz）
        f2: L2 频率（Hz）

    Returns:
        STEC 值（TECu，即 10^16 电子/平方米）
    """
    delta_P = P2 - P1
    factor = (f1 ** 2 * f2 ** 2) / (f1 ** 2 - f2 ** 2)
    stec = delta_P / IONOSPHERIC_K * factor
    return stec


def calculate_mw_combination(L1: float, L2: float, P1: float, P2: float) -> float:
    """
    计算 Melbourne-Wübbena (MW) 组合

    MW 组合用于周跳检测和模糊度固定：
    MW = (L1 - L2) - (P1 + P2) / 2 * (f1 - f2) / (f1 + f2)

    Args:
        L1: L1 载波相位观测值（周）
        L2: L2 载波相位观测值（周）
        P1: P1 伪距观测值（米）
        P2: P2 伪距观测值（米）

    Returns:
        MW 组合值（米）
    """
    lambda_w = SPEED_OF_LIGHT / (f1 - f2)
    mw = (L1 - L2) - (P1 + P2) / 2 * (f1 - f2) / (f1 + f2)
    return mw * lambda_w


def calculate_gf_combination(L1: float, L2: float, f1: float, f2: float) -> float:
    """
    计算几何无关（GF）组合

    GF 组合用于周跳检测，对伪距噪声不敏感：
    GF = L1 - L2 = (lambda1 * N1 - lambda2 * N2)

    Args:
        L1: L1 载波相位观测值（米）
        L2: L2 载波相位观测值（米）

    Returns:
        GF 组合值（米）
    """
    lambda1 = SPEED_OF_LIGHT / f1
    lambda2 = SPEED_OF_LIGHT / f2
    gf = L1 - L2
    return gf


@dataclass
class SP3Record:
    """
    SP3 精密星历记录数据类

    表示一个历元的卫星位置和时钟信息。

    Attributes:
        timestamp: 记录时间戳
        gps_week: GPS 周数
        gps_second: GPS 周内秒数
        sat_id: 卫星标识符（如 "G01"）
        x: X 坐标（米，ECEF）
        y: Y 坐标（米，ECEF）
        z: Z 坐标（米，ECEF）
        clock: 卫星钟差（米）
        type: 记录类型（P=精确位置，E=估计位置，-=不可用）
    """
    timestamp: datetime
    gps_week: int
    gps_second: float
    sat_id: str
    x: float
    y: float
    z: float
    clock: float
    type: str = "P"


@dataclass
class DCBRecord:
    """
    DCB（差分码偏差）记录数据类

    表示卫星或接收机的差分码偏差，用于伪距校正。

    Attributes:
        sat_id: 卫星标识符（如 "G01"）
        code1: 第一个码类型
        code2: 第二个码类型
        dcb_ns: DCB 值（纳秒）
        dcb_m: DCB 值（米）
        source: 数据来源
    """
    sat_id: str
    code1: str
    code2: str
    dcb_ns: float
    dcb_m: float
    source: str = ""


@dataclass
class IonexHeader:
    """
    IONEX 文件头数据类

    包含 IONEX 文件的元信息和控制参数。

    Attributes:
        base_radius: 地球基准半径（公里）
        map_width: 地图宽度（经度格点数）
        map_height: 地图高度（纬度格点数）
        lat_min: 最低纬度（度）
        lat_max: 最高纬度（度）
        lon_min: 最低经度（度）
        lon_max: 最高经度（度）
        dlat: 纬度分辨率（度）
        dlon: 经度分辨率（度）
        hgt1: 第一个高度层（公里）
        hgt2: 第二个高度层（公里）
        dhgt: 高度分辨率（公里）
        yyyy: 起始年
        mm: 起始月
        dd: 起始日
        hh: 起始小时
        min: 起始分钟
        sec: 起始秒
        num_epochs: 地图 epoch 数
        num_maps: 地图数量
        maps_contained: 包含的地图标识列表
    """
    base_radius: float
    map_width: int
    map_height: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    dlat: float
    dlon: float
    hgt1: float
    hgt2: float
    dhgt: float
    yyyy: int
    mm: int
    dd: int
    hh: int
    min: int
    sec: int
    num_epochs: int
    num_maps: int
    maps_contained: list[int]


@dataclass
class IonexMap:
    """
    IONEX 电离层格网地图数据类

    表示一个历元的全球电离层 VTEC 格网数据。

    Attributes:
        epoch: 地图历元时间
        map_id: 地图标识符
        latitude: 纬度数组
        longitude: 经度数组
        vtec: VTEC 数据数组（TECu），shape = (map_height, map_width)
        rms: VTEC RMS 数组（可选）
    """
    epoch: datetime
    map_id: int
    latitude: "np.ndarray"
    longitude: "np.ndarray"
    vtec: "np.ndarray"
    rms: "np.ndarray | None" = None


def parse_sp3_file(path: Path) -> Iterator[SP3Record]:
    """
    解析 SP3 精密星历文件

    SP3 格式（标准产品 3）包含卫星位置和时钟信息。
    支持 SP3-a、SP3-b、SP3-c、SP3-d 等版本。

    Args:
        path: SP3 文件路径

    Yields:
        SP3Record 对象迭代器
    """
    content = open_text_maybe_compressed(path)
    lines = content.splitlines()

    current_epoch_records: list[SP3Record] = []
    header_found = False
    epoch_timestamp: datetime | None = None
    gps_week = 0
    gps_sec = 0.0

    for line in lines:
        if not line.strip():
            continue

        if line.startswith("#"):
            # 解析 Epoch 头部行
            # 格式: # 2024 01 15 00 00 00.000000  yr mo day hr min sec
            parts = line[1:].split()
            if len(parts) >= 7:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    hour = int(parts[3])
                    minute = int(parts[4])
                    sec = float(parts[5])
                    epoch_timestamp = datetime(year, month, day, hour, minute, int(sec), tzinfo=UTC)
                    gps_week, gps_sec = _datetime_to_gps_time(epoch_timestamp)
                except (ValueError, IndexError):
                    continue
            header_found = True
            current_epoch_records = []

        elif header_found and line.startswith("EP"):
            # 解析卫星位置行
            # 格式: EP  1 G01   .00000000000000   .00000000000000   .00000000000000    .00000000000000 P
            parts = line.split()
            if len(parts) >= 8:
                sat_id = parts[2]
                try:
                    x = float(parts[3]) * 1000  # SP3 单位是公里，转换为米
                    y = float(parts[4]) * 1000
                    z = float(parts[5]) * 1000
                    clock = float(parts[6]) * 1000  # 转换为米
                    rec_type = parts[7] if len(parts) > 7 else "P"

                    yield SP3Record(
                        timestamp=epoch_timestamp,
                        gps_week=gps_week,
                        gps_second=gps_sec,
                        sat_id=sat_id,
                        x=x,
                        y=y,
                        z=z,
                        clock=clock,
                        type=rec_type,
                    )
                except (ValueError, IndexError):
                    continue

        elif line.startswith("EOF"):
            # 文件结束
            break


def parse_dcb_file(path: Path) -> list[DCBRecord]:
    """
    解析 DCB（差分码偏差）文件

    支持 CAS（中国科学院）格式的 DCB 文件。

    Args:
        path: DCB 文件路径

    Returns:
        DCBRecord 对象列表
    """
    content = open_text_maybe_compressed(path)
    lines = content.splitlines()

    records: list[DCBRecord] = []
    current_source = ""

    for line in lines:
        if not line.strip():
            continue

        if line.startswith("SOURCE"):
            # 解析数据来源标识
            parts = line.split()
            if len(parts) >= 2:
                current_source = parts[1]

        elif line.startswith("C1P"):
            # 解析 C1-P1 DCB
            parts = line.split()
            if len(parts) >= 3:
                try:
                    sat_id = parts[0]
                    dcb_ns = float(parts[1])
                    dcb_m = dcb_ns * 1e-9 * SPEED_OF_LIGHT
                    records.append(
                        DCBRecord(
                            sat_id=sat_id,
                            code1="C1",
                            code2="P1",
                            dcb_ns=dcb_ns,
                            dcb_m=dcb_m,
                            source=current_source,
                        )
                    )
                except ValueError:
                    continue

        elif line.startswith("C1C"):
            # 解析 C1-C2 DCB
            parts = line.split()
            if len(parts) >= 3:
                try:
                    sat_id = parts[0]
                    dcb_ns = float(parts[1])
                    dcb_m = dcb_ns * 1e-9 * SPEED_OF_LIGHT
                    records.append(
                        DCBRecord(
                            sat_id=sat_id,
                            code1="C1",
                            code2="C2",
                            dcb_ns=dcb_ns,
                            dcb_m=dcb_m,
                            source=current_source,
                        )
                    )
                except ValueError:
                    continue

    return records


def parse_ionex_file(path: Path) -> tuple[IonexHeader, list[IonexMap]]:
    """
    解析 IONEX（电离层格网交换）文件

    IONEX 格式用于交换全球电离层地图（GIM）数据。

    Args:
        path: IONEX 文件路径

    Returns:
        (IONEX 文件头, IONEX 地图列表) 元组
    """
    content = open_text_maybe_compressed(path)
    lines = content.splitlines()

    header: IonexHeader | None = None
    maps: list[IonexMap] = []
    current_map: IonexMap | None = None
    in_data_block = False
    data_lines: list[str] = []

    for line in lines:
        if not line.strip():
            continue

        if "START OF HEADER" in line:
            # 开始解析文件头
            header_lines = []
            continue

        if "END OF HEADER" in line:
            # 解析文件头并创建 Header 对象
            header = _parse_ionex_header(header_lines)
            in_data_block = False
            continue

        if header is None:
            # 收集文件头行
            header_lines.append(line)

        if "START OF MAP" in line:
            # 开始新的地图数据块
            parts = line.split()
            map_id = int(parts[2]) if len(parts) >= 3 else 0
            current_map = IonexMap(
                epoch=datetime.now(UTC),
                map_id=map_id,
                latitude=header.latitude if header else _default_ionex_latitudes(),
                longitude=header.longitude if header else _default_ionex_longitudes(),
                vtec=np.zeros((header.map_height if header else 71, header.map_width if header else 71)),
                rms=None,
            )
            in_data_block = True
            data_lines = []
            continue

        if "END OF MAP" in line and current_map is not None:
            # 解析数据块
            current_map.vtec = _parse_ionex_data(data_lines, current_map.vtec.shape)
            maps.append(current_map)
            current_map = None
            in_data_block = False
            continue

        if in_data_block and current_map is not None:
            # 收集数据行
            data_lines.append(line)

    if header is None:
        raise ValueError(f"Invalid IONEX file: {path}")

    return header, maps


def _parse_ionex_header(header_lines: list[str]) -> IonexHeader:
    """
    解析 IONEX 文件头

    Args:
        header_lines: 文件头的所有行

    Returns:
        IonexHeader 对象
    """
    header_map: dict[str, str] = {}

    for line in header_lines:
        if len(line) < 60:
            continue
        label = line[60:].strip()
        value = line[:60].strip()
        header_map[label] = value

    # 提取关键参数
    base_radius = float(header_map.get("BASE RADIUS", "6371.0"))
    map_width = int(header_map.get("MAP WIDTH", "71"))
    map_height = int(header_map.get("MAP HEIGHT", "71"))

    lat_min = float(header_map.get("LAT1", "-87.5"))
    lat_max = float(header_map.get("LAT2", "87.5"))
    lon_min = float(header_map.get("LON1", "-180.0"))
    lon_max = float(header_map.get("LON2", "180.0"))
    dlat = float(header_map.get("DLAT", "2.5"))
    dlon = float(header_map.get("DLON", "5.0"))

    hgt1 = float(header_map.get("HGT1", "0.0"))
    hgt2 = float(header_map.get("HGT2", "0.0"))
    dhgt = float(header_map.get("DHGT", "0.0"))

    # 解析时间
    time_str = header_map.get("EPOCH OF CURRENT MAP", "")
    if time_str:
        parts = time_str.split()
        yyyy = int(parts[0]) if len(parts) > 0 else 2024
        mm = int(parts[1]) if len(parts) > 1 else 1
        dd = int(parts[2]) if len(parts) > 2 else 1
        hh = int(parts[3]) if len(parts) > 3 else 0
        mn = int(parts[4]) if len(parts) > 4 else 0
        sec = int(parts[5]) if len(parts) > 5 else 0
    else:
        yyyy, mm, dd, hh, mn, sec = 2024, 1, 1, 0, 0, 0

    num_epochs = int(header_map.get("NUM_epochs", "1"))
    num_maps = int(header_map.get("NUM_MAPS", "1"))

    return IonexHeader(
        base_radius=base_radius,
        map_width=map_width,
        map_height=map_height,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        dlat=dlat,
        dlon=dlon,
        hgt1=hgt1,
        hgt2=hgt2,
        dhgt=dhgt,
        yyyy=yyyy,
        mm=mm,
        dd=dd,
        hh=hh,
        min=mn,
        sec=sec,
        num_epochs=num_epochs,
        num_maps=num_maps,
        maps_contained=list(range(1, num_maps + 1)),
    )


def _parse_ionex_data(data_lines: list[str], shape: tuple[int, int]) -> "np.ndarray":
    """
    解析 IONEX 数据块

    Args:
        data_lines: 数据行的文本列表
        shape: 数据形状 (height, width)

    Returns:
        VTEC numpy 数组
    """
    data = np.zeros(shape)
    row_idx = 0

    for line in data_lines:
        if line.strip().startswith("LAT"):
            # 纬度行头部，跳过
            continue
        if line.strip().startswith("END"):
            break

        # 解析数据行，每行包含多个空格分隔的浮点数
        values = []
        for val_str in line.split():
            try:
                values.append(float(val_str))
            except ValueError:
                continue

        # 填充数据数组
        for col_idx, val in enumerate(values):
            if col_idx < shape[1]:
                data[row_idx, col_idx] = val

        row_idx += 1
        if row_idx >= shape[0]:
            break

    return data


def _default_ionex_latitudes() -> "np.ndarray":
    """返回默认的 IONEX 纬度数组"""
    return np.arange(87.5, -88, -2.5)


def _default_ionex_longitudes() -> "np.ndarray":
    """返回默认的 IONEX 经度数组"""
    return np.arange(-180, 185, 5)


def _datetime_to_gps_time(dt: datetime) -> tuple[int, float]:
    """
    将 datetime 转换为 GPS 时间

    Args:
        dt: datetime 对象

    Returns:
        (GPS 周数, GPS 周内秒数) 元组
    """
    delta = dt - GPS_EPOCH_START
    total_days = delta.days
    total_seconds = delta.total_seconds()

    gps_week = total_days // 7
    gps_second = (total_days % 7) * SECONDS_IN_DAY + (total_seconds - total_days * SECONDS_IN_DAY)

    return gps_week, gps_second
