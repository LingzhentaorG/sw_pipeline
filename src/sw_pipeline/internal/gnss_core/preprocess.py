# ==============================================================================
# GNSS 数据预处理模块
# ==============================================================================
# 本模块负责 GNSS 观测数据的预处理，将 RINEX 格式的原始数据解析并转换为标准化格式
#
# 主要功能：
# 1. RINEX 文件解析：读取 RINEX 3/2 格式的观测文件和导航文件
# 2. 数据标准化：将不同来源的数据统一转换为标准化格式
# 3. 测站信息提取：从观测文件头中提取测站坐标、天线高等信息
# 4. 周跳标记：识别并标记观测数据中的周跳位置
# 5. 数据验证：检测零字节文件、损坏文件等异常情况
# 6. 失败记录：记录预处理失败的文件和错误原因
# ==============================================================================

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .models import EventWindow, NormalizedObservationFile, PreprocessFailure, StationInfo
from .utils import (
    GPS_WEEK_START,
    SECONDS_IN_DAY,
    build_event_time_index,
    ecef_to_geodetic,
    geodetic_to_ecef,
    load_bytes_maybe_compressed,
    open_text_maybe_compressed,
    write_dataframe,
)


LOGGER = logging.getLogger(__name__)

# GNSS 频段标识符映射表，将字符标识转换为标准频段名称
BAND_CODES = {
    "1": "L1",
    "2": "L2",
    "3": "L3",
    "4": "L4",
    "5": "L5",
    "6": "L6",
    "7": "L7",
    "8": "L8",
    "9": "L9",
    "A": "L1C",
    "B": "L1C",
    "C": "L1C",
    "D": "L2C",
    "E": "L5",
    "L": "L1",
    "M": "L1C",
    "S": "L1C",
    "X": "L1C",
    "Y": "L1C",
    "Z": "L1C",
}

# RINEX 3 观测类型模式，用于从 RINEX 3 文件头中提取观测类型列表
RINEX3_OBS_PATTERN = re.compile(r"^\s*OBS\s+(\d+)\s*$", re.IGNORECASE)
# RINEX 3.05 格式的时间系统标识行
RINEX3_TIME_SYSTEM = re.compile(r"^\s*TIME OF FIRST OBS\s+.*\s+(\w+)\s*$", re.IGNORECASE)


def preprocess_records(config: PipelineConfig) -> Path:
    """
    预处理所有观测记录的主函数

    读取下载清单，解析每个观测文件和导航文件，
    提取标准化的观测数据、天线信息等，并输出标准化的 CSV 文件。

    Args:
        config: 流水线配置对象

    Returns:
        标准化观测清单文件路径
    """
    manifest_path = config.outputs.manifests_dir / "observation_manifest.csv"
    # 如果清单不存在，先执行下载阶段
    if not manifest_path.exists():
        from .download import execute_download_stage

        execute_download_stage(config)
        manifest_path = config.outputs.manifests_dir / "observation_manifest.csv"

    manifest = pd.read_csv(manifest_path)
    valid_mask = manifest["obs_status"].astype(str).eq("ok") & manifest["nav_status"].astype(str).eq("ok")
    valid = manifest[valid_mask].copy()

    LOGGER.info("Normalizing %s observation records (%s total rows)", len(valid), len(manifest))

    failures: list[PreprocessFailure] = []
    records: list[NormalizedObservationFile] = []

    for _, row in valid.iterrows():
        obs_path = Path(str(row["obs_path"]))
        nav_path = Path(str(row["nav_path"])) if pd.notna(row["nav_path"]) and str(row["nav_path"]).strip() else None
        if not obs_path.exists():
            failures.append(
                PreprocessFailure(
                    event_id=str(row["event_id"]),
                    observation_date=str(row["observation_date"]),
                    station_id=str(row.get("station_id", "")),
                    obs_path=str(row.get("obs_path", "")),
                    nav_path=str(row.get("nav_path", "")),
                    error="observation_file_missing",
                )
            )
            continue
        if nav_path is None or not nav_path.exists():
            failures.append(
                PreprocessFailure(
                    event_id=str(row["event_id"]),
                    observation_date=str(row["observation_date"]),
                    station_id=str(row.get("station_id", "")),
                    obs_path=str(row.get("obs_path", "")),
                    nav_path=str(row.get("nav_path", "")),
                    error="navigation_file_missing",
                )
            )
            continue

        records.append(
            NormalizedObservationFile(
                event_id=str(row["event_id"]),
                observation_date=str(row["observation_date"]),
                source=str(row["source"]),
                station_id=str(row["station_id"]),
                station_code4=str(row["station_code4"]).upper(),
                sampling_sec=int(row.get("sampling_sec", 30) or 30),
                obs_path=str(obs_path),
                nav_path=str(nav_path),
            )
        )

    invalid = manifest[~valid_mask].copy()
    for _, row in invalid.iterrows():
        failures.append(
            PreprocessFailure(
                event_id=str(row.get("event_id", "")),
                observation_date=str(row.get("observation_date", "")),
                station_id=str(row.get("station_id", "")),
                obs_path=str(row.get("obs_path", "")),
                nav_path=str(row.get("nav_path", "")),
                error=f"obs_status={row.get('obs_status', '')}; nav_status={row.get('nav_status', '')}",
            )
        )

    norm_manifest_path = config.outputs.manifests_dir / "normalized_manifest.csv"
    normalized_rows = []
    for rec in records:
        row = _norm_record_to_dict(rec)
        source_row = valid[
            (valid["event_id"].astype(str) == rec.event_id)
            & (valid["observation_date"].astype(str) == rec.observation_date)
            & (valid["station_id"].astype(str) == rec.station_id)
        ]
        if not source_row.empty:
            row["lat"] = float(source_row.iloc[0].get("lat", 0.0) or 0.0)
            row["lon"] = float(source_row.iloc[0].get("lon", 0.0) or 0.0)
            row["height_m"] = float(source_row.iloc[0].get("height_m", 0.0) or 0.0)
        else:
            row["lat"] = 0.0
            row["lon"] = 0.0
            row["height_m"] = 0.0
        normalized_rows.append(row)
    write_dataframe(pd.DataFrame(normalized_rows), norm_manifest_path)

    failures_path = config.outputs.manifests_dir / "preprocess_failures.csv"
    write_dataframe(pd.DataFrame([asdict(f) for f in failures]), failures_path)

    LOGGER.info("Preprocessing complete: %s records written, %s failures", len(records), len(failures))
    return norm_manifest_path


def parse_rinex_obs(path: Path, time_resolution: int) -> "RinexObservationResult":
    """
    解析 RINEX 观测文件

    自动检测 RINEX 版本（2.x 或 3.x），调用相应的解析器。

    Args:
        path: RINEX 观测文件路径
        time_resolution: 时间分辨率（秒）

    Returns:
        RINEX 观测解析结果对象
    """
    # 检测 RINEX 版本
    first_line = open_text_maybe_compressed(path).splitlines()[0]
    version = float(first_line[0:9].strip())
    if version >= 3.0:
        return _parse_rinex3_obs(path, time_resolution)
    else:
        return _parse_rinex2_obs(path, time_resolution)


def parse_rinex_nav(path: Path | None) -> "RinexNavResult | None":
    """
    解析 RINEX 导航文件（广播星历）

    Args:
        path: RINEX 导航文件路径

    Returns:
        导航文件解析结果，如果文件不存在或解析失败则返回 None
    """
    if path is None or not Path(path).exists():
        return None
    try:
        return _parse_rinex_nav_impl(path)
    except Exception as exc:
        LOGGER.warning("Failed to parse navigation file %s: %s", path, exc)
        return None


def _parse_rinex3_obs(path: Path, time_resolution: int) -> "RinexObservationResult":
    """
    解析 RINEX 3.x 格式观测文件

    RINEX 3 格式特点：
    - 第一行以 "3.0x" 开头（x 为子版本号）
    - 观测类型使用两字符代码（如 C1C, L1C, L5Q）
    - 波长因子在独立行中指定
    - 卫星标识使用 PRN + satellite type 格式（如 G01, E11）

    Args:
        path: RINEX 3 文件路径
        time_resolution: 时间分辨率（秒）

    Returns:
        解析结果
    """
    lines = open_text_maybe_compressed(path).splitlines()
    header: dict[str, object] = {}
    data_lines: list[str] = []

    # 解析文件头
    for i, line in enumerate(lines):
        if i < 3:
            # RINEX 3 文件的前几行包含版本和系统信息
            _parse_rinex3_header_line(line, header)
        if "END OF HEADER" in line:
            # 头结束，进入数据部分
            data_lines = lines[i + 1 :]
            break

    # 解析观测数据
    obs_data = _parse_rinex3_data(data_lines, header, time_resolution)
    return RinexObservationResult(header=header, data=obs_data)


def _parse_rinex2_obs(path: Path, time_resolution: int) -> "RinexObservationResult":
    """
    解析 RINEX 2.x 格式观测文件

    RINEX 2 格式特点：
    - 第一行以 "2.1x" 或 "2.xx" 开头
    - 观测类型使用单字符代码（如 C1, L1, P2）
    - 波长因子在头中以每颗卫星为单位指定
    - 测站标识为 4 字符

    Args:
        path: RINEX 2 文件路径
        time_resolution: 时间分辨率（秒）

    Returns:
        解析结果
    """
    lines = open_text_maybe_compressed(path).splitlines()
    header: dict[str, object] = {}
    data_lines: list[str] = []

    # 解析文件头
    for i, line in enumerate(lines):
        if "END OF HEADER" in line:
            data_lines = lines[i + 1 :]
            break
        _parse_rinex2_header_line(line, header)

    # 解析观测数据
    obs_data = _parse_rinex2_data(data_lines, header, time_resolution)
    return RinexObservationResult(header=header, data=obs_data)


def _parse_rinex3_data(
    data_lines: list[str], header: dict[str, object], time_resolution: int
) -> pd.DataFrame:
    """
    解析 RINEX 3 格式的观测数据块

    Args:
        data_lines: 数据部分的所有行
        header: 已解析的文件头
        time_resolution: 时间分辨率（秒）

    Returns:
        包含解析后观测数据的 DataFrame
    """
    # 获取观测类型列表
    obs_types: list[tuple[str, str]] = header.get("obs_types", [])
    num_obs_types = len(obs_types)

    records: list[dict[str, object]] = []
    i = 0
    while i < len(data_lines):
        line = data_lines[i]
        # 跳过空行
        if not line.strip():
            i += 1
            continue

        # 解析 epoch 行：格式为 ">" + 年 月 日 时 分 秒 + 标志 + 卫星数
        if line.startswith(">"):
            # 解析时间信息
            parts = line[1:].split()
            year, month, day, hour, minute = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            sec = float(parts[6]) if len(parts) > 6 else 0.0
            flag = int(parts[7]) if len(parts) > 7 else 0
            num_sats = int(parts[8]) if len(parts) > 8 else 0

            # 构建 UTC 时间戳
            timestamp = datetime(year, month, day, hour, minute, int(sec), tzinfo=UTC)
            epoch_gps_sec = _datetime_to_gps_seconds(timestamp)

            # 读取后续卫星行
            sat_lines = data_lines[i + 1 : i + 1 + num_sats]
            i += 1 + num_sats

            # 跳过质量标志异常的 epoch（0=正常，1=power failure，6=end of usable data）
            if flag not in {0, 6}:
                continue

            # 解析每颗卫星的观测值
            for sat_line in sat_lines:
                # 解析卫星标识（格式：系统 + PRN，如 G01, E11, C19）
                sat_id = sat_line[:3].strip()
                if len(sat_line) < 3:
                    continue
                sys = sat_id[0]
                prn = sat_id[1:]

                # 读取该卫星的观测值（每行 80 字符，包含多个观测值）
                sat_obs: list[str] = []
                remaining = sat_line[3:]
                while len(sat_obs) < num_obs_types:
                    sat_obs.extend(_split_rinex_obs_line(remaining))
                    if len(sat_obs) < num_obs_types and i < len(data_lines):
                        i += 1
                        remaining = data_lines[i]
                    else:
                        break

                # 构建记录字典
                rec: dict[str, object] = {
                    "timestamp": timestamp,
                    "gps_second": epoch_gps_sec,
                    "sat_sys": sys,
                    "sat_prn": prn,
                    "sat_id": sat_id,
                    "epoch_flag": flag,
                }

                # 填充观测值
                for j, (band, obs_type) in enumerate(obs_types):
                    val = float(sat_obs[j]) if j < len(sat_obs) and sat_obs[j].strip() else float("nan")
                    rec[f"{band}_{obs_type}"] = val

                records.append(rec)
        else:
            i += 1

    # 创建 DataFrame
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)

    # 提取频段和伪距/载波相位类型
    df["band"] = df["sat_id"].str[0].map({"G": "G", "E": "E", "C": "C", "R": "R", "J": "J"})
    df["lband"] = df.apply(lambda row: _rnx3_band_to_lband(str(row["sat_id"][1:])), axis=1)

    # 解析伪距和载波相位观测值
    for col in df.columns:
        if col.endswith("_L1C") or col.endswith("_L1P") or col.endswith("_L1"):
            df["C1"] = df.get(f"GC1C") or df.get(f"GC1P") or df.get(f"GC1")
            df["L1"] = df[col]
        elif col.endswith("_L2C") or col.endswith("_L2P") or col.endswith("_L2"):
            df["C2"] = df.get(f"GC2C") or df.get(f"GC2P") or df.get(f"GC2")
            df["L2"] = df[col]

    # 筛选指定 GNSS 系统和时间分辨率
    sys_filter = str(header.get("gnss_system", "G"))
    df = df[df["sat_sys"].isin(list(sys_filter))]
    return df


def _parse_rinex2_data(
    data_lines: list[str], header: dict[str, object], time_resolution: int
) -> pd.DataFrame:
    """
    解析 RINEX 2 格式的观测数据块

    Args:
        data_lines: 数据部分的所有行
        header: 已解析的文件头
        time_resolution: 时间分辨率（秒）

    Returns:
        包含解析后观测数据的 DataFrame
    """
    obs_types = header.get("obs_types", [])
    records: list[dict[str, object]] = []
    i = 0

    while i < len(data_lines):
        line = data_lines[i]
        if not line.strip():
            i += 1
            continue

        # RINEX 2 epoch 行：年 月 日 时 分 秒 标志 卫星数
        if line[0:1].isdigit():
            parts = line[:30].split()
            year = int(parts[0])
            # RINEX 2 使用两位年份，需要处理千年问题
            if year < 80:
                year += 2000
            else:
                year += 1900
            month, day, hour, minute = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            sec = float(parts[5])
            flag = int(parts[6])
            num_sats = int(parts[7])

            timestamp = datetime(year, month, day, hour, minute, int(sec), tzinfo=UTC)
            epoch_gps_sec = _datetime_to_gps_seconds(timestamp)

            # 读取后续卫星行
            sat_lines = data_lines[i + 1 : i + 1 + math.ceil(num_sats * 3 / 80)]
            i += 1 + math.ceil(num_sats * 3 / 80)

            if flag not in {0, 6}:
                continue

            # 解析卫星观测值
            sat_idx = 0
            for sat_line in sat_lines:
                for j in range(0, min(len(sat_line), 80), 3):
                    if sat_idx >= num_sats:
                        break
                    sat_str = sat_line[j : j + 3]
                    if len(sat_str) < 3:
                        break
                    sat_id = f"G{sat_str[1:3]}"
                    records.append(
                        {
                            "timestamp": timestamp,
                            "gps_second": epoch_gps_sec,
                            "sat_sys": "G",
                            "sat_prn": sat_str[1:3],
                            "sat_id": sat_id,
                            "C1": float(sat_str[:1] + "1" + sat_line[j + 3 : j + 14].strip()),
                            "L1": float(sat_line[j + 14 : j + 26].strip()),
                            "flag": flag,
                        }
                    )
                    sat_idx += 1
        else:
            i += 1

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df


def _parse_rinex_nav_impl(path: Path) -> "RinexNavResult":
    """
    解析 RINEX 导航文件（广播星历）

    Args:
        path: 导航文件路径

    Returns:
        导航解析结果
    """
    content = open_text_maybe_compressed(path)
    lines = content.splitlines()

    header: dict[str, object] = {}
    nav_data: list[dict[str, float]] = []

    # 解析头
    for line in lines:
        if "END OF HEADER" in line:
            break
        # 解析版本
        if line.startswith("3.0"):
            header["version"] = float(line[:9].strip())
            header["sat_sys"] = line[20:23].strip()

    # 解析星历数据块（每块 7 或 8 行，对应一颗卫星）
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.startswith("3.0") or "END OF HEADER" in line:
            i += 1
            continue

        # 检查是否为卫星标识行（如 "G 1" 或 "G01"）
        if len(line) >= 3 and line[0] in "GRECJ":
            # 解析卫星标识
            sat_sys = line[0]
            prn_str = line[1:3].strip()
            if prn_str.isdigit():
                prn = int(prn_str)
            else:
                i += 1
                continue

            # 解析星历参数（8 行）
            block = [line] + lines[i + 1 : i + 9]
            if len(block) < 8:
                i += 1
                continue

            # 提取关键参数
            try:
                epoch = _parse_nav_epoch(block[1])
                record: dict[str, object] = {
                    "sat_sys": sat_sys,
                    "sat_prn": prn,
                    "timestamp": epoch,
                    "gps_week": int(float(block[2][23:33])),
                    "sv_clock_bias": float(block[2][3:23]),
                    "sv_clock_drift": float(block[3][3:23]),
                    "sv_clock_drift_rate": float(block[3][23:43]),
                    "broadcast_epoch_1": float(block[1][3:23]),
                    "broadcast_epoch_2": float(block[1][23:43]),
                    "broadcast_epoch_3": float(block[2][3:23]),
                    "broadcast_epoch_4": float(block[3][23:43]),
                    "broadcast_epoch_5": float(block[4][:23]),
                    "broadcast_epoch_6": float(block[4][23:43]),
                    "broadcast_epoch_7": float(block[5][:23]),
                    "broadcast_epoch_8": float(block[5][23:43]),
                    "broadcast_epoch_9": float(block[6][:23]),
                    "broadcast_epoch_10": float(block[6][23:43]),
                    "broadcast_epoch_11": float(block[7][:23]),
                    "broadcast_epoch_12": float(block[7][23:43]),
                }
                nav_data.append(record)
            except (ValueError, IndexError):
                pass
            i += 8
        else:
            i += 1

    return RinexNavResult(header=header, data=pd.DataFrame(nav_data) if nav_data else pd.DataFrame())


def _datetime_to_gps_seconds(dt: datetime) -> float:
    """
    将 datetime 转换为 GPS 时间（秒）

    GPS 时间以 1980 年 1 月 6 日 00:00:00 UTC 为起点

    Args:
        dt: datetime 对象（UTC 时区）

    Returns:
        GPS 秒数
    """
    delta = dt - GPS_WEEK_START
    return delta.total_seconds()


def _rnx3_band_to_lband(sat_id_tail: str) -> int:
    """
    将 RINEX 3 卫星标识后缀转换为 L-波段号

    Args:
        sat_id_tail: 卫星标识去掉系统前缀后的部分（如 "01", "11"）

    Returns:
        L-波段号（1-9）
    """
    # GPS: L1 = 1, L2 = 2, L5 = 5
    # Galileo: E1 = 1, E5 = 5, E6 = 6
    # BeiDou: B1 = 1, B2 = 2, B3 = 3
    tail = sat_id_tail.upper()
    if len(tail) >= 2:
        first_char = tail[0]
        if first_char == "1":
            return 1
        elif first_char == "2":
            return 2
        elif first_char == "5":
            return 5
        elif first_char == "6":
            return 6
        elif first_char == "7":
            return 7
        elif first_char == "8":
            return 8
    return 1


def _split_rinex_obs_line(line: str) -> list[str]:
    """
    切分 RINEX 观测行为观测值列表

    RINEX 观测值每个字段固定宽度 16 字符

    Args:
        line: 观测行字符串

    Returns:
        观测值字符串列表
    """
    result = []
    for j in range(0, len(line), 16):
        val = line[j : j + 16].strip()
        result.append(val)
    return result


def _parse_nav_epoch(block: list[str]) -> datetime:
    """
    解析导航文件中的星历历元时间

    Args:
        block: 星历数据块

    Returns:
        datetime 对象
    """
    year = int(float(block[0:6].strip()))
    if year < 80:
        year += 2000
    month = int(float(block[6:12].strip()))
    day = int(float(block[12:18].strip()))
    hour = int(float(block[18:24].strip()))
    minute = int(float(block[24:30].strip()))
    sec = float(block[30:38].strip())
    return datetime(year, month, day, hour, minute, int(sec), tzinfo=UTC)


def _parse_rinex3_header_line(line: str, header: dict[str, object]) -> None:
    """
    解析 RINEX 3 文件头行

    Args:
        line: 文件头的一行
        header: 解析结果存储字典
    """
    if "RINEX VERSION" in line:
        header["version"] = float(line[0:9].strip())
        header["sat_sys"] = line[20:23].strip()
    elif "MARKER NAME" in line:
        header["station_id"] = line[0:20].strip()
    elif "MARKER NUMBER" in line:
        header["marker_number"] = line[0:20].strip()
    elif "APPROX POSITION XYZ" in line:
        try:
            x = float(line[0:14].strip())
            y = float(line[14:28].strip())
            z = float(line[28:42].strip())
            header["approx_pos_ecef"] = (x, y, z)
        except ValueError:
            pass
    elif "ANTENNA: DELTA H/E/N" in line:
        try:
            header["antenna_delta_h"] = float(line[0:14].strip())
            header["antenna_delta_e"] = float(line[14:28].strip())
            header["antenna_delta_n"] = float(line[28:42].strip())
        except ValueError:
            pass
    elif "OBS TYPES" in line:
        # 解析观测类型列表（格式：num_types  TYPE1 TYPE2 ...）
        match = RINEX3_OBS_PATTERN.search(line)
        if match:
            num_obs = int(match.group(1))
            types = []
            rest = line[line.index("OBS TYPES") + 10 :]
            # 收集所有观测类型
            for i in range(num_obs):
                if i * 4 + 7 <= len(rest):
                    types.append(rest[i * 4 : i * 4 + 4].strip())
            header["obs_types"] = [(t[0], t[1:]) if len(t) > 1 else ("?", t) for t in types]


def _parse_rinex2_header_line(line: str, header: dict[str, object]) -> None:
    """
    解析 RINEX 2 文件头行

    Args:
        line: 文件头的一行
        header: 解析结果存储字典
    """
    if "RINEX VERSION" in line:
        header["version"] = float(line[0:9].strip())
        header["sat_sys"] = line[20:23].strip()
    elif "MARKER NAME" in line:
        header["station_id"] = line[0:20].strip()
    elif "MARKER NUMBER" in line:
        header["marker_number"] = line[0:20].strip()
    elif "APPROX POSITION XYZ" in line:
        try:
            x = float(line[0:14].strip())
            y = float(line[14:28].strip())
            z = float(line[28:42].strip())
            header["approx_pos_ecef"] = (x, y, z)
        except ValueError:
            pass
    elif "ANTENNA: DELTA H/E/N" in line:
        try:
            header["antenna_delta_h"] = float(line[0:14].strip())
            header["antenna_delta_e"] = float(line[14:28].strip())
            header["antenna_delta_n"] = float(line[28:42].strip())
        except ValueError:
            pass
    elif "INTERVAL" in line:
        try:
            header["interval"] = float(line[0:10].strip())
        except ValueError:
            pass
    elif "OBS TYPES" in line:
        # RINEX 2 格式的观测类型行
        pass


def _norm_record_to_dict(rec: NormalizedObservationFile) -> dict[str, object]:
    """
    将标准化记录转换为字典格式

    Args:
        rec: 标准化记录对象

    Returns:
        可序列化为 DataFrame 的字典
    """
    return {
        "event_id": rec.event_id,
        "observation_date": rec.observation_date,
        "source": rec.source,
        "station_id": rec.station_id,
        "station_code4": rec.station_code4,
        "sampling_sec": rec.sampling_sec,
        "obs_path": rec.obs_path,
        "nav_path": rec.nav_path,
        "lat": math.nan,
        "lon": math.nan,
        "height_m": math.nan,
    }


@dataclass
class RinexObservationResult:
    """
    RINEX 观测文件解析结果数据类

    Attributes:
        header: 从文件头提取的元信息字典
        data: 解析后的观测数据 DataFrame
    """
    header: dict[str, object]
    data: pd.DataFrame


@dataclass
class RinexNavResult:
    """
    RINEX 导航文件解析结果数据类

    Attributes:
        header: 从文件头提取的元信息字典
        data: 解析后的星历数据 DataFrame
    """
    header: dict[str, object]
    data: pd.DataFrame
