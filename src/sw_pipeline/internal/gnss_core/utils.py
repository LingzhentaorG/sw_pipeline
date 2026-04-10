# ==============================================================================
# GNSS 数据处理管道工具函数模块
# ==============================================================================
# 本文件提供了 GNSS 数据处理管道所需的各类通用工具函数
# 主要包括：
# - 日期时间解析与范围生成（UTC 时区处理、GPS 周/年积日计算）
# - 目录创建与文件系统操作
# - 带自动重试机制的 HTTP 会话
# - 坐标系统转换（ECEF 与大地坐标互转）
# - 压缩文件读写支持（gzip、compress 格式）
# - DataFrame/Dataset 的读写封装（CSV、Parquet、NetCDF）
# - NOAA KMZ 站点数据解析
# - 日志系统配置
# - 统计工具（中位数绝对偏差）
# ==============================================================================

from __future__ import annotations

import io
import logging
import math
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import gzip
import hatanaka
import ncompress
import pandas as pd
import requests
import urllib3
import xarray as xr
from pyproj import Transformer
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


# 模块级日志记录器
LOGGER = logging.getLogger(__name__)
# GPS 时间系统常量，供预处理与辅助产品解析复用
GPS_WEEK_START = datetime(1980, 1, 6, tzinfo=UTC)
GPS_EPOCH_START = GPS_WEEK_START
SECONDS_IN_DAY = 24 * 60 * 60
# 禁用不安全的 HTTPS 请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# ECEF（地心地固坐标系）到大地坐标系的转换器：EPSG:4978 -> EPSG:4979
ECEF_TO_GEODETIC = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
# 大地坐标系到 ECEF 的转换器：EPSG:4979 -> EPSG:4978
GEODETIC_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)


def parse_utc_datetime(value: str) -> datetime:
    """
    将字符串解析为带 UTC 时区的 datetime 对象

    支持 ISO 8601 格式的时间字符串，自动处理 'Z' 后缀和缺失时区信息的情况，
    确保返回的 datetime 始终使用 UTC 时区。

    Args:
        value: 时间字符串，如 '2024-01-15T08:30:00Z' 或 '2024-01-15T08:30:00+08:00'

    Returns:
        带 UTC 时区的 datetime 对象
    """
    # 去除首尾空格，将 'Z' 后缀替换为 '+00:00' 以兼容 fromisoformat
    value = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    # 如果解析后没有时区信息，则手动添加 UTC 时区
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    # 如果已有时区信息，则转换为 UTC 时区
    return dt.astimezone(UTC)


def utc_date_range(start: datetime, end: datetime) -> list[date]:
    """
    生成起止时间之间的所有日期列表（包含首尾）

    从 start 的日期开始，逐日递增直到 end 的日期，返回完整的日期序列。

    Args:
        start: 起始 datetime 对象
        end: 结束 datetime 对象

    Returns:
        包含起止范围内所有 date 对象的列表
    """
    current = start.date()
    dates: list[date] = []
    # 逐日递增，直到超过结束日期
    while current <= end.date():
        dates.append(current)
        current += timedelta(days=1)
    return dates


def build_event_time_index(
    start: datetime, end: datetime, cadence_minutes: int
) -> pd.DatetimeIndex:
    """
    构建事件时间索引，按指定时间间隔生成等间距的时间戳序列

    将起止时间向下取整到指定间隔的倍数，然后生成均匀分布的时间索引，
    适用于事件数据的时间对齐和重采样。

    Args:
        start: 起始 datetime 对象
        end: 结束 datetime 对象
        cadence_minutes: 时间间隔（分钟），最小为 1

    Returns:
        带 UTC 时区的 pandas DatetimeIndex
    """
    # 确保间隔至少为 1 分钟
    cadence = max(int(cadence_minutes), 1)
    # 将起止时间转换为 UTC 并向下取整到指定间隔
    start_ts = pd.Timestamp(start).tz_convert(UTC).floor(f"{cadence}min")
    end_ts = pd.Timestamp(end).tz_convert(UTC).floor(f"{cadence}min")
    return pd.date_range(start_ts, end_ts, freq=f"{cadence}min", tz=UTC)


def date_to_doy(day: date) -> tuple[int, str, str]:
    """
    将日期转换为年积日（Day of Year）及相关格式

    Args:
        day: date 对象

    Returns:
        三元组：
            - int: 年积日数值（1-366）
            - str: 三位数年积日字符串（如 '045'）
            - str: 两位数年号（年份对 100 取模，如 2024 -> '24'）
    """
    doy = day.timetuple().tm_yday
    return doy, f"{doy:03d}", f"{day.year % 100:02d}"


def gps_week_and_dow(day: date) -> tuple[int, int]:
    """
    计算指定日期对应的 GPS 周数和周内天数

    GPS 时间以 1980 年 1 月 6 日（周日）为纪元起点，每周从周日开始计数。

    Args:
        day: date 对象

    Returns:
        二元组：
            - int: GPS 周数（自 GPS 纪元以来的完整周数）
            - int: 周内天数（0=周日，1=周一，..., 6=周六）
    """
    # GPS 时间纪元：1980 年 1 月 6 日
    gps_epoch = date(1980, 1, 6)
    delta_days = (day - gps_epoch).days
    week = delta_days // 7
    dow = delta_days % 7
    return week, dow


def ensure_directories(paths: Iterable[Path]) -> None:
    """
    确保指定的所有目录路径存在，不存在则递归创建

    使用 parents=True 创建所有缺失的父目录，exist_ok=True 避免已存在时报错。

    Args:
        paths: 需要确保存在的 Path 对象可迭代对象
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def create_retry_session(total: int = 4, backoff: float = 0.8) -> requests.Session:
    """
    创建带有自动重试机制的 HTTP 请求会话

    配置了指数退避重试策略，适用于网络不稳定或服务器临时不可用的场景。
    针对 GET 和 HEAD 请求，在遇到 429（限流）或 5xx（服务器错误）状态码时自动重试。

    Args:
        total: 最大重试次数，默认为 4 次
        backoff: 退避因子，用于计算重试间隔（第 n 次重试等待 backoff * 2^(n-1) 秒）

    Returns:
        配置好重试策略的 requests.Session 对象
    """
    session = requests.Session()
    retry = Retry(
        total=total,
        backoff_factor=backoff,
        # 遇到这些 HTTP 状态码时触发重试
        status_forcelist=(429, 500, 502, 503, 504),
        # 仅对 GET 和 HEAD 方法进行重试（幂等操作）
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    # 将重试适配器挂载到 http 和 https 协议上
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    # 设置自定义 User-Agent 标识
    session.headers.update({"User-Agent": "gnss-pipeline/0.1"})
    return session


def in_bbox(lat: float, lon: float, bbox: dict[str, float]) -> bool:
    """
    判断给定的经纬度坐标是否在指定的矩形边界框内

    Args:
        lat: 纬度（度）
        lon: 经度（度）
        bbox: 边界框字典，包含 'lat_min'、'lat_max'、'lon_min'、'lon_max' 四个键

    Returns:
        如果坐标在边界框内返回 True，否则返回 False
    """
    return (
        bbox["lat_min"] <= lat <= bbox["lat_max"]
        and bbox["lon_min"] <= lon <= bbox["lon_max"]
    )


def parse_noaa_kmz(kmz_bytes: bytes) -> list[dict[str, float | str]]:
    """
    解析 NOAA GNSS 站点 KMZ 文件，提取站点信息列表

    KMZ 是压缩的 KML 格式，本函数解压后解析其中的 Placemark 元素，
    提取站点名称、经纬度和高程信息。

    Args:
        kmz_bytes: KMZ 文件的原始字节数据

    Returns:
        站点信息字典列表，每个字典包含：
            - station_id: 站点 ID（大写）
            - station_code4: 站点 4 位代码（小写）
            - lat: 纬度
            - lon: 经度
            - height_m: 高程（米）
    """
    stations: list[dict[str, float | str]] = []
    # 从字节流中解压 KMZ（ZIP 格式）文件
    with zipfile.ZipFile(io.BytesIO(kmz_bytes)) as zf:
        # 查找第一个 .kml 文件（忽略大小写）
        xml_name = next(name for name in zf.namelist() if name.lower().endswith(".kml"))
        root = ET.fromstring(zf.read(xml_name))
    # KML 2.2 命名空间
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    # 遍历所有 Placemark 元素提取站点信息
    for placemark in root.findall(".//kml:Placemark", ns):
        name = (placemark.findtext("kml:name", default="", namespaces=ns) or "").strip()
        coord_text = placemark.findtext(".//kml:coordinates", default="", namespaces=ns)
        # 跳过名称或坐标缺失的无效条目
        if not name or not coord_text:
            continue
        # 坐标格式为 "经度,纬度,高度"
        lon_str, lat_str, alt_str = coord_text.split(",")
        stations.append(
            {
                "station_id": name.upper(),
                "station_code4": name[:4].lower(),
                "lat": float(lat_str),
                "lon": float(lon_str),
                "height_m": float(alt_str),
            }
        )
    return stations


def parse_anchor_hrefs(html: str) -> list[str]:
    """
    从 HTML 文本中提取所有 <a> 标签的 href 属性值

    用于解析目录索引页面中的文件链接列表。

    Args:
        html: HTML 文本字符串

    Returns:
        所有有效的 href 属性值列表
    """
    # 包裹在 <root> 标签中使其成为合法的 XML
    root = ET.fromstring(f"<root>{html}</root>")
    return [
        elem.attrib["href"] for elem in root.findall(".//a") if "href" in elem.attrib
    ]


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    """
    将 ECEF（地心地固直角坐标）转换为大地坐标（纬度、经度、高程）

    ECEF 坐标系（EPSG:4978）使用 X,Y,Z 表示地球上的位置，
    大地坐标系（EPSG:4979）使用纬度、经度、椭球高表示。

    Args:
        x: ECEF X 坐标（米）
        y: ECEF Y 坐标（米）
        z: ECEF Z 坐标（米）

    Returns:
        三元组：
            - float: 纬度（度）
            - float: 经度（度）
            - float: 椭球高（米）
    """
    lon, lat, height = ECEF_TO_GEODETIC.transform(x, y, z)
    return float(lat), float(lon), float(height)


def geodetic_to_ecef(
    lat_deg: float, lon_deg: float, height_m: float
) -> tuple[float, float, float]:
    """
    将大地坐标（纬度、经度、高程）转换为 ECEF（地心地固直角坐标）

    Args:
        lat_deg: 纬度（度）
        lon_deg: 经度（度）
        height_m: 椭球高（米）

    Returns:
        三元组：
            - float: ECEF X 坐标（米）
            - float: ECEF Y 坐标（米）
            - float: ECEF Z 坐标（米）
    """
    x, y, z = GEODETIC_TO_ECEF.transform(lon_deg, lat_deg, height_m)
    return float(x), float(y), float(z)


def open_text_maybe_compressed(path: Path) -> str:
    """
    读取文本文件，自动处理 gzip、compress 和 Hatanaka 压缩格式

    根据文件后缀名判断压缩类型：
    - .gz / .Z: 优先使用 hatanaka 处理（兼容 CRINEX），失败时回退到常规解压
    - 其他: 直接读取

    Args:
        path: 文件路径

    Returns:
        文件内容的文本字符串（ASCII 编码，忽略错误字符）
    """
    return load_bytes_maybe_compressed(path).decode("ascii", errors="ignore")


def load_bytes_maybe_compressed(path: Path) -> bytes:
    """
    读取文件字节数据，自动处理 gzip、compress 和 Hatanaka 压缩格式

    与 open_text_maybe_compressed 类似，但返回原始字节而非解码后的文本。

    Args:
        path: 文件路径

    Returns:
        文件的字节数据（如为压缩文件则返回解压后的字节）
    """
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] in {".gz", ".z"}:
        raw = path.read_bytes()
        try:
            return hatanaka.decompress(raw)
        except Exception:
            if suffixes[-1] == ".gz":
                return gzip.decompress(raw)
            return ncompress.decompress(raw)
    return path.read_bytes()


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    """
    将 pandas DataFrame 写入文件，支持 CSV 和 Parquet 格式

    自动创建父目录，根据文件扩展名选择写入格式。

    Args:
        df: 要写入的 pandas DataFrame 对象
        path: 输出文件路径，扩展名决定格式（.csv 或 .parquet）

    Raises:
        ValueError: 当文件扩展名不是 .csv 或 .parquet 时抛出
    """
    ensure_directories([path.parent])
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def load_dataframe(path: Path) -> pd.DataFrame:
    """
    从文件加载 pandas DataFrame，支持 CSV 和 Parquet 格式

    Args:
        path: 输入文件路径，扩展名决定格式（.csv 或 .parquet）

    Returns:
        加载的 pandas DataFrame 对象

    Raises:
        ValueError: 当文件扩展名不是 .csv 或 .parquet 时抛出
    """
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path}")


def write_dataset(dataset: xr.Dataset, path: Path) -> None:
    """
    将 xarray Dataset 写入 NetCDF 文件，自动配置压缩参数

    对浮点型和整型变量启用 zlib 压缩（压缩级别 2），以减小文件体积。

    Args:
        dataset: 要写入的 xarray Dataset 对象
        path: 输出 NetCDF 文件路径（.nc）
    """
    ensure_directories([path.parent])
    encoding = {}
    # 为数值型变量（浮点数、有符号整数、无符号整数）配置压缩
    for name, variable in dataset.data_vars.items():
        if variable.dtype.kind in {"f", "i", "u"}:
            encoding[name] = {"zlib": True, "complevel": 2}
    dataset.to_netcdf(path, encoding=encoding)


def load_dataset(path: Path) -> xr.Dataset:
    """
    从 NetCDF 文件加载 xarray Dataset

    Args:
        path: NetCDF 文件路径

    Returns:
        加载的 xarray Dataset 对象
    """
    return xr.load_dataset(path)


def find_event_netcdf_paths(netcdf_dir: Path, event_id: str) -> list[Path]:
    """
    查找指定事件 ID 对应的 NetCDF 数据文件路径

    优先查找分段文件（格式：{event_id}_*.nc），如果没有则回退到单文件模式（{event_id}.nc）。

    Args:
        netcdf_dir: NetCDF 文件所在目录
        event_id: 事件唯一标识符

    Returns:
        匹配的 NetCDF 文件路径列表，未找到时返回空列表
    """
    # 优先查找分段文件（按名称排序）
    segmented = sorted(netcdf_dir.glob(f"{event_id}_*.nc"))
    if segmented:
        return segmented
    # 回退到单文件模式
    legacy = netcdf_dir / f"{event_id}.nc"
    return [legacy] if legacy.exists() else []


def resolve_station_identifier(station_id: str | None) -> set[str]:
    """
    解析站点标识符，生成所有可能的匹配候选集合

    同时包含完整 ID 和前 4 位字符，用于兼容不同格式的站点引用。

    Args:
        station_id: 站点 ID 字符串，可为 None

    Returns:
        站点标识符候选集合（已转为大写），如果输入为空则返回空集合
    """
    text = str(station_id or "").strip().upper()
    if not text:
        return set()
    candidates = {text}
    # 如果长度足够，添加前 4 位字符作为候选（兼容 4 位站点代码）
    if len(text) >= 4:
        candidates.add(text[:4])
    return candidates


def median_abs_deviation(values: pd.Series | list[float] | tuple[float, ...]) -> float:
    """
    计算数值序列的中位数绝对偏差（Median Absolute Deviation, MAD）

    MAD 是衡量数据离散程度的稳健统计量，对异常值不敏感。
    计算公式：MAD = median(|Xi - median(X)|)

    Args:
        values: 数值序列，支持 pandas Series、list 或 tuple

    Returns:
        中位数绝对偏差值，如果输入为空则返回 NaN
    """
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return math.nan
    median = float(series.median())
    return float((series - median).abs().median())


def configure_logging(log_path: Path) -> None:
    """
    配置全局日志系统，同时输出到控制台和文件

    设置根日志记录器为 INFO 级别，配置统一的日志格式，
    添加控制台处理器和文件处理器（UTF-8 编码）。
    如果根日志记录器已有处理器则跳过配置，避免重复添加。

    Args:
        log_path: 日志文件输出路径
    """
    ensure_directories([log_path.parent])
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # 如果已有处理器则跳过，防止重复配置
    if root.handlers:
        return
    # 定义统一的日志格式：时间 [级别] 记录器名称: 消息
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # 控制台处理器
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    # 文件处理器（UTF-8 编码）
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(console)
    root.addHandler(file_handler)
    LOGGER.info("Logging initialized at %s", log_path)
