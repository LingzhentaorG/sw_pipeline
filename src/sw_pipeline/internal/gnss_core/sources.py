# ==============================================================================
# GNSS 数据源适配器模块
# ==============================================================================
# 本模块定义了从不同 GNSS 数据源发现测站并构建下载记录的适配器
#
# 支持的数据源：
# - NOAA CORS（美国国家海洋和大气管理局连续运行参考站网络）
# - RBMC（巴西连续监测 GNSS 网络）
# - RAMSAC（阿根廷连续 GNSS 监测网络）
# - CDDIS（地壳动力学数据信息系统，NASA）
#
# 每个数据源都有独立的适配器类，负责：
# 1. 从远程服务器获取可用测站列表
# 2. 根据地理边界框（bbox）过滤测站
# 3. 为每个测站和日期构建观测文件和导航文件的下载 URL
# 4. 生成 DownloadRecord 对象供后续下载流程使用
# ==============================================================================

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date
from typing import Iterable

import requests

from .models import DownloadRecord, EventWindow, SourceSettings
from .utils import create_retry_session, date_to_doy, in_bbox, parse_noaa_kmz


LOGGER = logging.getLogger(__name__)


def build_global_nav_url(day: date, base_obs_url: str) -> str:
    """
    构建全球导航星历文件的下载 URL

    根据指定日期和基础观测 URL，生成对应日期的广播星历（broadcast ephemeris）文件地址。
    导航文件包含所有卫星的轨道参数，用于后续定位解算。

    Args:
        day: 目标日期，用于确定年份和年积日（doy）
        base_obs_url: 数据源的基础观测文件 URL 前缀

    Returns:
        完整的导航文件下载 URL，格式如: {base_obs_url}/2024/001/brdc0010.24n.gz
    """
    doy, doy_str, yy = date_to_doy(day)
    return f"{base_obs_url}/{day.year}/{doy_str}/brdc{doy_str}0.{yy}n.gz"


def build_noaa_obs_url(day: date, station_code4: str, base_obs_url: str) -> str:
    """
    构建 NOAA 数据源的观测文件下载 URL

    NOAA 的观测文件按 {base_url}/{year}/{doy}/{station}/{station}{doy}0.{yy}d.gz 格式组织。

    Args:
        day: 目标观测日期
        station_code4: 4 位测站代码（如 "ALB1"）
        base_obs_url: NOAA 数据源的基础观测 URL

    Returns:
        完整的 NOAA 观测文件下载 URL
    """
    _, doy_str, yy = date_to_doy(day)
    return f"{base_obs_url}/{day.year}/{doy_str}/{station_code4.lower()}/{station_code4.lower()}{doy_str}0.{yy}d.gz"


def build_ramsac_obs_url(
    day: date, station_code4: str, interval_sec: int, base_url: str
) -> str:
    """
    构建 RAMSAC 数据源的观测文件下载 URL

    RAMSAC 的观测文件按 {base_url}/{interval}/{station}{doy}0.{yy}d.Z 格式组织，
    其中 interval 为采样间隔（秒）。

    Args:
        day: 目标观测日期
        station_code4: 4 位测站代码
        interval_sec: 采样间隔（秒），如 15 秒
        base_url: RAMSAC 数据源的基础下载 URL

    Returns:
        完整的 RAMSAC 观测文件下载 URL
    """
    _, doy_str, yy = date_to_doy(day)
    return f"{base_url}/{interval_sec}/{station_code4.lower()}{doy_str}0.{yy}d.Z"


def build_cddis_obs_url(day: date, station_code4: str, template: str) -> str:
    """
    构建 CDDIS 数据源的观测文件下载 URL

    CDDIS 使用模板字符串来生成 URL，模板中包含 {year}、{doy}、{yy}、{station} 占位符。

    Args:
        day: 目标观测日期
        station_code4: 4 位测站代码
        template: URL 模板字符串，例如 "{year}/{doy}/{station}{doy}0.{yy}d.gz"

    Returns:
        填充占位符后的完整 CDDIS 观测文件下载 URL
    """
    _, doy_str, yy = date_to_doy(day)
    return template.format(
        year=day.year, doy=doy_str, yy=yy, station=station_code4.lower()
    )


class SourceAdapter(ABC):
    """
    数据源适配器抽象基类

    所有具体的数据源适配器（NOAA、RBMC、RAMSAC、CDDIS）都必须继承此类，
    并实现 discover() 方法来发现可用测站并构建下载记录。

    Attributes:
        settings: 数据源的配置参数，包含 URL 模板、优先级、超时等
        bbox: 地理边界框字典，包含 min_lat、max_lat、min_lon、max_lon，用于过滤测站
        base_nav_url: 导航星历文件的基础 URL
    """

    def __init__(
        self, settings: SourceSettings, bbox: dict[str, float], base_nav_url: str
    ) -> None:
        """
        初始化适配器

        Args:
            settings: 数据源配置对象
            bbox: 地理边界框，用于筛选指定区域内的测站
            base_nav_url: 导航文件的基础 URL
        """
        self.settings = settings
        self.bbox = bbox
        self.base_nav_url = base_nav_url

    @abstractmethod
    def discover(self, event: EventWindow) -> list[DownloadRecord]:
        """
        发现可用测站并构建下载记录列表

        子类必须实现此方法，从远程数据源获取测站信息，
        为事件时间窗口内的每一天生成观测文件和导航文件的下载记录。

        Args:
            event: 事件时间窗口，包含事件 ID 和起止时间

        Returns:
            DownloadRecord 对象列表，每个记录包含一个测站在某一天的下载信息
        """
        raise NotImplementedError


class NOAAAdapter(SourceAdapter):
    """
    NOAA（美国国家海洋和大气管理局）CORS 数据源适配器

    从 NOAA CORS 网络发现测站，流程如下：
    1. 下载 KMZ 格式的测站元数据文件
    2. 解析 KMZ 并根据边界框过滤测站
    3. 对每一天，获取当日可用的观测文件列表
    4. 将文件列表中的测站与元数据匹配，构建下载记录
    """

    def discover(self, event: EventWindow) -> list[DownloadRecord]:
        """
        发现 NOAA CORS 测站并构建下载记录

        首先从 KMZ 文件获取测站元数据并按地理边界框过滤，
        然后逐日查询当日可用的观测文件列表，
        将可用文件与元数据匹配后生成 DownloadRecord。

        Args:
            event: 事件时间窗口

        Returns:
            NOAA 数据源的 DownloadRecord 列表
        """
        session = create_retry_session()
        kmz_url = str(self.settings.params["network_kmz_url"])
        base_obs_url = str(self.settings.params["base_obs_url"])

        # 下载并解析 KMZ 文件，获取所有测站元数据
        response = session.get(kmz_url, timeout=self.settings.timeout_sec)
        response.raise_for_status()

        # 解析 KMZ 内容，按边界框过滤测站，建立测站代码到元数据的映射
        station_lookup = {
            str(station["station_code4"]).lower(): station
            for station in parse_noaa_kmz(response.content)
            if in_bbox(float(station["lat"]), float(station["lon"]), self.bbox)
        }
        LOGGER.info("NOAA discovered %s stations inside bbox", len(station_lookup))

        records: list[DownloadRecord] = []
        # 遍历事件时间窗口内的每一天
        for day in _event_days(event):
            # 构建当日的全球导航文件 URL
            nav_url = build_global_nav_url(day, base_obs_url)
            _, doy_str, yy = date_to_doy(day)

            # 获取当日观测文件列表，用于确认哪些测站有可用数据
            file_list_url = (
                f"{base_obs_url}/{day.year}/{doy_str}/{day.year}.{doy_str}.files.list"
            )
            file_list_response = session.get(
                file_list_url, timeout=self.settings.timeout_sec
            )
            file_list_response.raise_for_status()

            # 从文件列表中提取所有可用的测站代码（去重并排序）
            available_codes = sorted(
                {
                    line.split()[0].split("/")[3].lower()
                    for line in file_list_response.text.splitlines()[1:]
                    if line.split()[0].endswith(f".{yy}d.gz")
                }
            )
            LOGGER.info(
                "NOAA %s returned %s daily observation files before bbox filter",
                day.isoformat(),
                len(available_codes),
            )

            # 遍历可用测站，与元数据匹配后生成下载记录
            for station_code in available_codes:
                station = station_lookup.get(station_code.lower())
                if station is None:
                    continue
                records.append(
                    DownloadRecord(
                        event_id=event.event_id,
                        source="noaa",
                        source_priority=self.settings.priority,
                        observation_date=day.isoformat(),
                        station_id=str(station["station_id"]),
                        station_code4=str(station["station_code4"]),
                        sampling_sec=30,  # NOAA CORS 标准采样率为 30 秒
                        obs_url=build_noaa_obs_url(day, station_code, base_obs_url),
                        nav_url=nav_url,
                        lat=float(station["lat"]),
                        lon=float(station["lon"]),
                        height_m=float(station["height_m"]),
                    )
                )
        return records


class RBMCAdapter(SourceAdapter):
    """
    RBMC（巴西连续监测网络）数据源适配器

    从 IBGE/RBMC 数据源发现测站，通过目录列表页面获取当日可用的 .crx.gz 压缩观测文件。
    RBMC 不依赖边界框过滤，因为所有列出的文件都已属于该网络。
    """

    def discover(self, event: EventWindow) -> list[DownloadRecord]:
        """
        发现 RBMC 测站并构建下载记录

        逐日访问 RBMC 数据源的目录列表页面，
        从 HTML 响应中提取 .crx.gz 文件名，
        解析文件名获取测站代码和采样间隔，生成下载记录。

        Args:
            event: 事件时间窗口

        Returns:
            RBMC 数据源的 DownloadRecord 列表
        """
        session = create_retry_session()
        base_dir_url = str(self.settings.params["base_dir_url"]).rstrip("/")
        base_nav_url = str(self.base_nav_url)

        records: list[DownloadRecord] = []
        for day in _event_days(event):
            _, doy_str, _ = date_to_doy(day)
            # 构建当日目录 URL，该目录包含当天所有观测文件的列表
            day_url = f"{base_dir_url}/{day.year}/{doy_str}/"

            # 获取目录列表页面（HTML 格式）
            response = session.get(
                day_url,
                timeout=self.settings.timeout_sec,
                verify=bool(self.settings.params.get("verify_ssl", True)),
            )
            response.raise_for_status()

            # 从 HTML 响应中提取所有 .crx.gz 文件名
            hrefs = set(response.text.split('"'))
            filenames = sorted(name for name in hrefs if name.endswith(".crx.gz"))

            # 构建当日的全球导航文件 URL
            nav_url = build_global_nav_url(day, base_nav_url)
            LOGGER.info(
                "RBMC %s returned %s observation files", day.isoformat(), len(filenames)
            )

            # 遍历每个观测文件，解析文件名并生成下载记录
            for filename in filenames:
                # 文件名格式: {station}_{year}{doy}_{time}_{type}_{sampling}S.{format}.gz
                parts = filename.split("_")
                sampling_sec = (
                    int(parts[4].replace("S", "")) if len(parts) > 4 else None
                )
                records.append(
                    DownloadRecord(
                        event_id=event.event_id,
                        source="rbmc",
                        source_priority=self.settings.priority,
                        observation_date=day.isoformat(),
                        station_id=filename[:9],
                        station_code4=filename[:4].lower(),
                        sampling_sec=sampling_sec,
                        obs_url=f"{day_url}{filename}",
                        nav_url=nav_url,
                    )
                )
        return records


class RAMSACAdapter(SourceAdapter):
    """
    RAMSAC（阿根廷连续 GNSS 监测网络）数据源适配器

    通过 REST API 获取测站元数据，根据配置的采样间隔要求筛选可用测站，
    然后为每个测站和日期构建下载记录。
    """

    def discover(self, event: EventWindow) -> list[DownloadRecord]:
        """
        发现 RAMSAC 测站并构建下载记录

        从 RAMSAC 的 API 获取所有测站信息，
        筛选出采样间隔满足要求的测站，
        为每个测站和每一天生成下载记录。

        Args:
            event: 事件时间窗口

        Returns:
            RAMSAC 数据源的 DownloadRecord 列表
        """
        session = create_retry_session()
        stations_url = str(self.settings.params["stations_api_url"])
        download_base_url = str(self.settings.params["download_base_url"]).rstrip("/")
        requested_interval = int(self.settings.params.get("requested_interval_sec", 15))

        # 从 API 获取所有测站元数据（JSON 格式）
        response = session.get(
            stations_url,
            timeout=self.settings.timeout_sec,
            verify=bool(self.settings.params.get("verify_ssl", True)),
        )
        response.raise_for_status()
        stations = response.json()

        # 筛选出采样间隔小于等于期望值的测站
        usable = [
            station
            for station in stations
            if int(station["intervalo_observacion"]) <= requested_interval
        ]
        LOGGER.info(
            "RAMSAC discovered %s stations with <= %ss interval",
            len(usable),
            requested_interval,
        )

        records: list[DownloadRecord] = []
        for day in _event_days(event):
            nav_url = build_global_nav_url(day, self.base_nav_url)
            for station in usable:
                code = str(station["cod_estacion"]).lower()
                records.append(
                    DownloadRecord(
                        event_id=event.event_id,
                        source="ramsac",
                        source_priority=self.settings.priority,
                        observation_date=day.isoformat(),
                        station_id=code.upper(),
                        station_code4=code[:4],
                        sampling_sec=requested_interval,
                        obs_url=build_ramsac_obs_url(
                            day, code, requested_interval, download_base_url
                        ),
                        nav_url=nav_url,
                    )
                )
        return records


class CDDISAdapter(SourceAdapter):
    """
    CDDIS（地壳动力学数据信息系统）数据源适配器

    CDDIS 是一个归档型数据源，不主动发现测站，
    而是根据配置中预设的测站列表生成下载记录。
    通常用作其他数据源的补充。
    """

    def discover(self, event: EventWindow) -> list[DownloadRecord]:
        """
        根据预设测站列表构建 CDDIS 下载记录

        CDDIS 不进行远程测站发现，直接使用配置中的 station_codes 列表。
        如果列表为空则跳过。

        Args:
            event: 事件时间窗口

        Returns:
            CDDIS 数据源的 DownloadRecord 列表，如果未配置测站则返回空列表
        """
        template = str(self.settings.params["obs_url_template"])
        stations = [
            str(code).lower() for code in self.settings.params.get("station_codes", [])
        ]
        base_nav_url = str(self.base_nav_url)

        records: list[DownloadRecord] = []
        if not stations:
            LOGGER.info(
                "CDDIS enabled but station_codes is empty; skipping supplemental discovery."
            )
            return records

        for day in _event_days(event):
            nav_url = build_global_nav_url(day, base_nav_url)
            for station in stations:
                records.append(
                    DownloadRecord(
                        event_id=event.event_id,
                        source="cddis",
                        source_priority=self.settings.priority,
                        observation_date=day.isoformat(),
                        station_id=station.upper(),
                        station_code4=station[:4],
                        sampling_sec=30,
                        obs_url=build_cddis_obs_url(day, station[:4], template),
                        nav_url=nav_url,
                    )
                )
        return records


def make_adapters(
    sources: dict[str, SourceSettings],
    bbox: dict[str, float],
    base_nav_url: str,
) -> list[SourceAdapter]:
    """
    工厂函数：根据配置创建并返回所有已启用的数据源适配器

    按照固定顺序（noaa -> rbmc -> ramsac -> cddis）检查每个数据源是否启用，
    为已启用的数据源实例化对应的适配器。

    Args:
        sources: 数据源配置字典，键为数据源名称（如 "noaa"），值为 SourceSettings 对象
        bbox: 地理边界框字典，传递给所有适配器用于测站过滤
        base_nav_url: 导航文件的基础 URL

    Returns:
        已启用的 SourceAdapter 实例列表，按固定顺序排列
    """
    adapter_map: dict[str, type[SourceAdapter]] = {
        "noaa": NOAAAdapter,
        "rbmc": RBMCAdapter,
        "ramsac": RAMSACAdapter,
        "cddis": CDDISAdapter,
    }
    adapters: list[SourceAdapter] = []
    for name in ("noaa", "rbmc", "ramsac", "cddis"):
        settings = sources.get(name)
        if settings and settings.enabled:
            adapters.append(adapter_map[name](settings, bbox, base_nav_url))
    return adapters


def _event_days(event: EventWindow) -> Iterable[date]:
    """
    生成事件时间窗口内所有日期的迭代器

    从事件的起始日期（UTC）开始，逐日递增，直到结束日期

    Args:
        event: 事件时间窗口对象

    Yields:
        事件时间窗口内的每一个日期（date 对象）
    """
    current = event.start_utc.date()
    end = event.end_utc.date()
    while current <= end:
        yield current
        current = current.fromordinal(current.toordinal() + 1)
