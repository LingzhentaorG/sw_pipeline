# ==============================================================================
# GNSS 数据下载模块
# ==============================================================================
# 本模块负责 GNSS 观测数据和辅助产品的下载管理
#
# 主要功能：
# 1. 测站发现：从各数据源（NOAA、RBMC、RAMSAC、CDDIS）发现可用测站
# 2. 观测数据下载：下载 RINEX 格式的 GNSS 观测文件
# 3. 导航数据下载：下载广播星历和精密星历文件
# 4. 辅助产品下载：下载 SP3、DCB、IONEX 等电离层处理所需产品
# 5. 清单管理：生成和管理下载记录清单（manifest）
# 6. 断点续传：支持跳过已下载文件和增量下载
# 7. 重试机制：处理网络错误和临时失败
# ==============================================================================

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from .config import PipelineConfig
from .models import AuxDownloadRecord, DownloadRecord
from .sources import make_adapters
from .utils import create_retry_session, date_to_doy, gps_week_and_dow, write_dataframe


LOGGER = logging.getLogger(__name__)

# 可重试的预处理错误标记列表，这些错误通常是由于网络或临时问题导致的
RETRYABLE_PREPROCESS_ERROR_MARKERS = (
    "Compressed file ended before the end-of-stream marker was reached",
    "could not find first valid header line",
    "unsupported compressed payload",
    "empty compressed payload",
    "Downloaded file is empty",
)


def discover_download_records(config: PipelineConfig) -> list[DownloadRecord]:
    """
    发现所有需要下载的 GNSS 数据记录

    遍历配置中的所有事件和数据源适配器，
    生成完整的下载记录列表，包含每个测站在每一天的观测文件和导航文件下载信息。

    Args:
        config: 流水线配置对象

    Returns:
        DownloadRecord 对象列表，每个记录代表一个测站某一天的下载任务
    """
    # 从配置中获取 NOAA 基础 URL 作为导航文件下载的基础 URL
    base_nav_url = str(config.observation_sources["noaa"].params["base_obs_url"])
    # 创建所有已启用的数据源适配器
    adapters = make_adapters(config.observation_sources, config.bbox, base_nav_url)
    records: list[DownloadRecord] = []

    # 遍历所有事件和数据源，发现可用测站
    for event in config.events:
        for adapter in adapters:
            try:
                records.extend(adapter.discover(event))
            except Exception as exc:
                LOGGER.exception("Discovery failed for %s on %s", adapter.settings.name, event.event_id)
                LOGGER.error("%s", exc)
    return records


def execute_download_stage(config: PipelineConfig) -> Path:
    """
    执行下载阶段主函数

    负责协调整个下载流程：
    1. 清理过时的临时文件和零字节文件
    2. 发现需要下载的记录
    3. 检查并复用已有的下载文件
    4. 并行下载缺失的文件
    5. 生成下载清单和失败记录

    Args:
        config: 流水线配置对象

    Returns:
        观测数据下载清单文件路径
    """
    temp_suffix = str(config.download["temp_suffix"])

    # 清理可重试的预处理失败对应的原始文件，以便重新下载
    _purge_retryable_preprocess_failures(config.outputs.manifests_dir)
    # 清理过时的临时文件和零字节原始文件
    _cleanup_stale_raw_files(
        config.outputs.raw_dir,
        temp_suffix=temp_suffix,
        drop_zero_byte_raw=bool(config.processing["drop_zero_byte_raw"]),
    )

    # 发现所有需要下载的记录
    records = discover_download_records(config)
    LOGGER.info("Prepared %s download candidates", len(records))

    # 获取 CDDIS 认证信息
    cddis_auth = config.auth.get("cddis", {})
    nav_settings = config.observation_sources["noaa"]

    # 构建下载任务字典，按 URL+认证+验证方式聚合
    jobs: dict[tuple[str, str, str, bool], dict[str, object]] = {}

    # 遍历所有记录，检查已存在的文件或创建下载任务
    for record in records:
        # 处理观测文件
        obs_basename = record.obs_url.rstrip("/").split("/")[-1]
        record.obs_path = _find_existing_observation_file(
            config,
            record.event_id,
            record.observation_date,
            record.source,
            obs_basename,
        )
        if record.obs_path:
            # 文件已存在，标记为 ok
            record.obs_status = "ok"
            record.obs_error = None
        else:
            # 需要下载，创建下载任务
            raw_dir = config.outputs.raw_dir / record.event_id / record.observation_date / record.source
            raw_dir.mkdir(parents=True, exist_ok=True)
            obs_auth = _auth_for_source(record.source, cddis_auth)
            obs_verify = bool(config.observation_sources[record.source].params.get("verify_ssl", True))
            obs_key = _job_key(record.obs_url, obs_auth, obs_verify)
            jobs.setdefault(
                obs_key,
                {
                    "url": record.obs_url,
                    "source": record.source,
                    "output_dir": raw_dir,
                    "auth": obs_auth,
                    "timeout": config.observation_sources[record.source].timeout_sec,
                    "verify": obs_verify,
                    "worker_group": "noaa" if record.source == "noaa" else "other",
                },
            )

        # 处理导航文件
        record.nav_path = _find_existing_navigation_file(config, record.event_id, record.observation_date, nav_basename)
        if record.nav_path:
            record.nav_status = "ok"
            record.nav_error = None
        else:
            nav_dir = config.outputs.raw_dir / record.event_id / record.observation_date / "nav"
            nav_dir.mkdir(parents=True, exist_ok=True)
            nav_key = _job_key(record.nav_url, None, bool(nav_settings.params.get("verify_ssl", True)))
            jobs.setdefault(
                nav_key,
                {
                    "url": record.nav_url,
                    "source": "nav",
                    "output_dir": nav_dir,
                    "auth": None,
                    "timeout": nav_settings.timeout_sec,
                    "verify": bool(nav_settings.params.get("verify_ssl", True)),
                    "worker_group": "noaa",
                },
            )

    # 执行下载任务
    results: dict[tuple[str, str, str, bool], dict[str, object]] = {}
    if jobs:
        # 分组执行：NOAA 和其他源使用不同的线程数
        noaa_results = _execute_job_group(
            jobs=jobs,
            worker_group="noaa",
            max_workers=int(config.download["noaa_workers"]),
            max_retries=int(config.download["max_retries"]),
            temp_suffix=temp_suffix,
        )
        other_results = _execute_job_group(
            jobs=jobs,
            worker_group="other",
            max_workers=int(config.download["other_workers"]),
            max_retries=int(config.download["max_retries"]),
            temp_suffix=temp_suffix,
        )
        results.update(noaa_results)
        results.update(other_results)

    # 更新记录状态并写入清单
    for record in records:
        if not record.obs_path:
            obs_auth = _auth_for_source(record.source, cddis_auth)
            obs_verify = bool(config.observation_sources[record.source].params.get("verify_ssl", True))
            obs_key = _job_key(record.obs_url, obs_auth, obs_verify)
            obs_result = results.get(obs_key, {"path": None, "status": "error", "error": "missing"})
            record.obs_path = obs_result["path"]
            record.obs_status = str(obs_result["status"])
            record.obs_error = obs_result["error"]

        if not record.nav_path:
            nav_key = _job_key(record.nav_url, None, bool(nav_settings.params.get("verify_ssl", True)))
            nav_result = results.get(nav_key, {"path": None, "status": "error", "error": "missing"})
            record.nav_path = nav_result["path"]
            record.nav_status = str(nav_result["status"])
            record.nav_error = nav_result["error"]

    # 写入观测数据下载清单
    manifest_path = config.outputs.manifests_dir / "observation_manifest.csv"
    manifest_df = pd.DataFrame(asdict(record) for record in records)
    write_dataframe(manifest_df, manifest_path)

    # 写入下载失败记录
    failures_path = config.outputs.manifests_dir / "observation_failures.csv"
    failure_rows = [
        {
            "url": job["url"],
            "source": job["source"],
            "worker_group": job["worker_group"],
            "output_dir": str(job["output_dir"]),
            "path": result["path"],
            "status": result["status"],
            "attempts": result["attempts"],
            "error": result["error"],
        }
        for key, job in jobs.items()
        for result in [results[key]]
        if result["status"] != "ok"
    ]
    write_dataframe(pd.DataFrame(failure_rows), failures_path)

    # 下载辅助产品（SP3、DCB、IONEX）
    aux_manifest_path, aux_failures_path = _download_auxiliary_products(config, temp_suffix=temp_suffix)

    LOGGER.info("Observation manifest written to %s", manifest_path)
    LOGGER.info("Observation failures written to %s", failures_path)
    LOGGER.info("Auxiliary manifest written to %s", aux_manifest_path)
    LOGGER.info("Auxiliary failures written to %s", aux_failures_path)
    return manifest_path


def _download_auxiliary_products(config: PipelineConfig, temp_suffix: str) -> tuple[Path, Path]:
    """
    下载辅助产品（SP3、DCB、IONEX）

    遍历所有事件日期，为每一天下载所需的辅助产品。
    按优先级尝试多个提供商，确保至少有一个成功。

    Args:
        config: 流水线配置对象
        temp_suffix: 临时文件后缀

    Returns:
        (辅助产品清单路径, 辅助产品失败记录路径) 元组
    """
    # 收集所有事件涉及的唯一日期
    unique_days = sorted({current_day for event in config.events for current_day in _event_days(event)})
    records: list[AuxDownloadRecord] = []
    failures: list[dict[str, object]] = []

    # 遍历每一天和每种辅助产品类型
    for current_day in unique_days:
        for product_type, settings in config.auxiliary_sources.items():
            if not settings.enabled:
                continue
            record, failed_candidates = _download_aux_product_for_day(config, current_day, product_type, settings, temp_suffix)
            records.append(record)
            failures.extend(failed_candidates)

    # 写入清单和失败记录
    manifest_path = config.outputs.manifests_dir / "aux_manifest.csv"
    failures_path = config.outputs.manifests_dir / "aux_failures.csv"
    write_dataframe(pd.DataFrame(asdict(record) for record in records), manifest_path)
    write_dataframe(pd.DataFrame(failures), failures_path)
    return manifest_path, failures_path


def _download_aux_product_for_day(
    config: PipelineConfig,
    current_day: date,
    product_type: str,
    settings,
    temp_suffix: str,
) -> tuple[AuxDownloadRecord, list[dict[str, object]]]:
    """
    下载某一天指定类型的辅助产品

    按优先级遍历提供商列表，尝试下载产品文件。
    如果已有有效文件则直接复用。

    Args:
        config: 流水线配置对象
        current_day: 目标日期
        product_type: 产品类型（sp3/dcb/ionex）
        settings: 辅助源设置
        temp_suffix: 临时文件后缀

    Returns:
        (下载记录, 失败记录列表) 元组
    """
    year = current_day.year
    _, doy_str, yy = date_to_doy(current_day)
    gps_week, dow = gps_week_and_dow(current_day)
    failures: list[dict[str, object]] = []

    # 按优先级排序提供商
    providers = sorted(settings.params.get("providers", []), key=lambda item: int(item.get("priority", 100)))

    for provider in providers:
        # 构建产品 URL（替换模板中的占位符）
        url = str(provider["url_template"]).format(
            yyyy=year,
            ddd=doy_str,
            yy=yy,
            gps_week=gps_week,
            dow=dow,
        )
        filename = url.rstrip("/").split("/")[-1]
        output_dir = config.outputs.aux_dir / product_type / str(year)
        existing = output_dir / filename

        # 如果文件已存在且大小大于 0，直接复用
        if existing.exists() and existing.stat().st_size > 0:
            return (
                AuxDownloadRecord(
                    observation_date=current_day.isoformat(),
                    product_type=product_type,
                    provider=str(provider["name"]),
                    priority=int(provider.get("priority", settings.priority)),
                    url=url,
                    auth_ref=provider.get("auth"),
                    verify_ssl=bool(provider.get("verify_ssl", True)),
                    path=str(existing),
                    status="ok",
                    metadata={"gps_week": gps_week, "day_of_week": dow},
                ),
                failures,
            )

        # 获取认证信息并执行下载
        auth = _auth_for_ref(provider.get("auth"), config.auth)
        result = _download_one(
            url=url,
            output_dir=output_dir,
            auth=auth,
            timeout=settings.timeout_sec,
            verify=bool(provider.get("verify_ssl", True)),
            max_retries=int(config.download["aux_retries"]),
            temp_suffix=temp_suffix,
        )

        if result["status"] == "ok":
            return (
                AuxDownloadRecord(
                    observation_date=current_day.isoformat(),
                    product_type=product_type,
                    provider=str(provider["name"]),
                    priority=int(provider.get("priority", settings.priority)),
                    url=url,
                    auth_ref=provider.get("auth"),
                    verify_ssl=bool(provider.get("verify_ssl", True)),
                    path=result["path"],
                    status="ok",
                    error=None,
                    attempts=int(result["attempts"]),
                    metadata={"gps_week": gps_week, "day_of_week": dow},
                ),
                failures,
            )

        # 记录失败信息
        failures.append(
            {
                "observation_date": current_day.isoformat(),
                "product_type": product_type,
                "provider": str(provider["name"]),
                "url": url,
                "status": result["status"],
                "error": result["error"],
                "attempts": result["attempts"],
            }
        )

    # 所有提供商都失败
    return (
        AuxDownloadRecord(
            observation_date=current_day.isoformat(),
            product_type=product_type,
            provider="unresolved",
            priority=int(settings.priority),
            url="",
            status="error",
            error=f"No provider succeeded for {product_type}",
            metadata={"gps_week": gps_week, "day_of_week": dow},
        ),
        failures,
    )


def _execute_job_group(
    jobs: dict[tuple[str, str, str, bool], dict[str, object]],
    worker_group: str,
    max_workers: int,
    max_retries: int,
    temp_suffix: str,
) -> dict[tuple[str, str, str, bool], dict[str, object]]:
    """
    执行一组下载任务

    使用线程池并行下载同一组的文件，并定期输出进度。

    Args:
        jobs: 下载任务字典
        worker_group: 工作组名称（用于日志）
        max_workers: 最大工作线程数
        max_retries: 最大重试次数
        temp_suffix: 临时文件后缀

    Returns:
        下载结果字典
    """
    # 筛选属于该组的任务
    selected = {key: job for key, job in jobs.items() if job["worker_group"] == worker_group}
    if not selected:
        return {}

    workers = max(1, max_workers)
    LOGGER.info("Downloading %s %s URLs with %s workers", len(selected), worker_group, workers)
    completed = 0
    results: dict[tuple[str, str, str, bool], dict[str, object]] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _download_one,
                str(job["url"]),
                Path(job["output_dir"]),
                job["auth"],
                int(job["timeout"]),
                bool(job["verify"]),
                max_retries,
                temp_suffix,
            ): key
            for key, job in selected.items()
        }
        for future in as_completed(future_map):
            key = future_map[future]
            results[key] = future.result()
            completed += 1
            # 每 100 个文件或完成时输出进度
            if completed % 100 == 0 or completed == len(future_map):
                LOGGER.info(
                    "Download progress [%s]: %s/%s unique URLs",
                    worker_group,
                    completed,
                    len(future_map),
                )
    return results


def _cleanup_stale_raw_files(raw_dir: Path, temp_suffix: str, drop_zero_byte_raw: bool) -> tuple[int, int]:
    """
    清理过时的原始数据文件

    删除：
    1. 未完成的临时下载文件（.part 后缀）
    2. 零字节的原始文件（可选）

    Args:
        raw_dir: 原始数据目录
        temp_suffix: 临时文件后缀
        drop_zero_byte_raw: 是否删除零字节文件

    Returns:
        (删除的临时文件数, 删除的零字节文件数) 元组
    """
    removed_temp = 0
    removed_zero = 0
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        # 删除临时文件
        if path.name.endswith(temp_suffix):
            path.unlink(missing_ok=True)
            removed_temp += 1
            continue
        # 删除零字节文件
        if drop_zero_byte_raw and path.stat().st_size == 0:
            path.unlink(missing_ok=True)
            removed_zero += 1
    LOGGER.info("Cleaned raw directory: removed %s temp files and %s zero-byte files", removed_temp, removed_zero)
    return removed_temp, removed_zero


def _purge_retryable_preprocess_failures(manifests_dir: Path) -> int:
    """
    删除因可重试错误而失败的原始文件

    读取预处理失败记录，对于标记为可重试错误的文件，
    删除其原始文件以便后续重新下载和处理。

    Args:
        manifests_dir: 清单文件目录

    Returns:
        删除的文件数量
    """
    failures_path = manifests_dir / "preprocess_failures.csv"
    if not failures_path.exists():
        return 0
    try:
        failures = pd.read_csv(failures_path)
    except Exception as exc:
        LOGGER.warning("Failed to read preprocess failures manifest %s: %s", failures_path, exc)
        return 0

    removed = 0
    for row in failures.to_dict("records"):
        error = str(row.get("error", "") or "")
        # 检查是否为可重试错误
        if not any(marker in error for marker in RETRYABLE_PREPROCESS_ERROR_MARKERS):
            continue
        # 删除对应的原始文件
        for key in ("obs_path", "nav_path"):
            raw_path = row.get(key)
            if not raw_path:
                continue
            path = Path(str(raw_path))
            if path.exists():
                path.unlink(missing_ok=True)
                removed += 1
    if removed:
        LOGGER.info("Removed %s raw files flagged by preprocess failures for re-download", removed)
    return removed


def _download_one(
    url: str,
    output_dir: Path,
    auth: tuple[str, str] | None,
    timeout: int,
    verify: bool,
    max_retries: int,
    temp_suffix: str,
) -> dict[str, object]:
    """
    执行单个文件下载

    使用流式下载避免内存占用，支持断点续传和自动重试。
    下载时先写入临时文件，成功后重命名为正式文件名。

    Args:
        url: 下载 URL
        output_dir: 输出目录
        auth: 认证信息元组（用户名, 密码）
        timeout: 超时时间（秒）
        verify: 是否验证 SSL 证书
        max_retries: 最大重试次数
        temp_suffix: 临时文件后缀

    Returns:
        包含下载结果的字典（path, status, error, attempts）
    """
    # 计算输出路径和临时文件路径
    output_path = output_dir / url.rstrip("/").split("/")[-1]
    temp_path = output_path.with_name(output_path.name + temp_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 如果文件已存在且大小大于 0，跳过下载
    if output_path.exists() and output_path.stat().st_size > 0:
        return {"path": str(output_path), "status": "ok", "error": None, "attempts": 0}
    # 清理可能存在的残留文件
    if output_path.exists():
        output_path.unlink(missing_ok=True)
    temp_path.unlink(missing_ok=True)

    last_error: str | None = None
    # 尝试下载，带重试机制
    for attempt in range(1, max_retries + 1):
        session = create_retry_session(total=max_retries, backoff=0.5)
        try:
            with session.get(
                url,
                stream=True,
                timeout=timeout,
                auth=auth,
                allow_redirects=True,
                verify=verify,
            ) as response:
                response.raise_for_status()
                # 流式写入临时文件
                with temp_path.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1 << 20):  # 1MB chunks
                        if chunk:
                            stream.write(chunk)

            # 验证下载文件
            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise OSError("Downloaded file is empty")

            # 下载成功，重命名临时文件为正式文件
            temp_path.replace(output_path)
            return {"path": str(output_path), "status": "ok", "error": None, "attempts": attempt}

        except (requests.RequestException, OSError) as exc:
            last_error = str(exc)
            # 清理失败的临时文件
            temp_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            # 等待后重试（指数退避）
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 16))
        finally:
            session.close()

    LOGGER.warning("Download failed for %s after %s attempts: %s", url, max_retries, last_error)
    return {"path": None, "status": "error", "error": last_error, "attempts": max_retries}


def _job_key(
    url: str,
    auth: tuple[str, str] | None,
    verify: bool,
) -> tuple[str, str, str, bool]:
    """
    生成下载任务的唯一键

    用于聚合相同 URL、认证和验证方式的下载任务。

    Args:
        url: 下载 URL
        auth: 认证信息
        verify: SSL 验证标志

    Returns:
        用于字典键的元组
    """
    if auth is None:
        return (url, "", "", verify)
    return (url, auth[0], auth[1], verify)


def _auth_for_source(source: str, auth_cfg: object) -> tuple[str, str] | None:
    """
    获取数据源的认证信息

    Args:
        source: 数据源名称
        auth_cfg: 认证配置

    Returns:
        认证元组（用户名, 密码）或 None
    """
    if source != "cddis":
        return None
    if not isinstance(auth_cfg, dict):
        return None
    username = str(auth_cfg.get("username", "") or "")
    password = str(auth_cfg.get("password", "") or "")
    if not username or not password:
        return None
    return username, password


def _auth_for_ref(auth_ref: object, auth_map: dict[str, object]) -> tuple[str, str] | None:
    """
    根据认证引用获取认证信息

    Args:
        auth_ref: 认证引用标识
        auth_map: 认证映射字典

    Returns:
        认证元组（用户名, 密码）或 None
    """
    if not auth_ref:
        return None
    auth_cfg = auth_map.get(str(auth_ref), {})
    return _auth_for_source(str(auth_ref), auth_cfg)


def _find_existing_observation_file(
    config: PipelineConfig,
    event_id: str,
    observation_date: str,
    source: str,
    filename: str,
) -> str | None:
    """
    查找已存在的观测文件

    依次检查新缓存目录、旧缓存目录，找到则返回路径。

    Args:
        config: 流水线配置
        event_id: 事件 ID
        observation_date: 观测日期
        source: 数据源
        filename: 文件名

    Returns:
        存在的文件路径或 None
    """
    candidates = [
        config.outputs.raw_dir / event_id / observation_date / source / filename,
        config.outputs.legacy_raw_dir / event_id / observation_date / source / filename,
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return str(path)
    return None


def _find_existing_navigation_file(
    config: PipelineConfig,
    event_id: str,
    observation_date: str,
    filename: str,
) -> str | None:
    """
    查找已存在的导航文件

    首先检查标准位置，然后递归搜索目录。

    Args:
        config: 流水线配置
        event_id: 事件 ID
        observation_date: 观测日期
        filename: 文件名

    Returns:
        存在的文件路径或 None
    """
    # 首先检查标准位置
    direct_candidates = [
        config.outputs.raw_dir / event_id / observation_date / "nav" / filename,
        config.outputs.legacy_raw_dir / event_id / observation_date / "nav" / filename,
    ]
    for path in direct_candidates:
        if path.exists() and path.stat().st_size > 0:
            return str(path)

    # 如果没找到，递归搜索父目录
    parent_roots = [
        config.outputs.raw_dir / event_id / observation_date,
        config.outputs.legacy_raw_dir / event_id / observation_date,
    ]
    for root in parent_roots:
        if not root.exists():
            continue
        for path in root.rglob(filename):
            if path.is_file() and path.stat().st_size > 0:
                return str(path)
    return None


def _event_days(event) -> list[date]:
    """
    生成事件时间窗口内的所有日期

    Args:
        event: 事件对象

    Returns:
        日期列表
    """
    current = event.start_utc.date()
    result: list[date] = []
    while current <= event.end_utc.date():
        result.append(current)
        current = current.fromordinal(current.toordinal() + 1)
    return result
