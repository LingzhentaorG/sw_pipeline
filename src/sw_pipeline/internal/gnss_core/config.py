# ==============================================================================
# GNSS 管道配置加载模块
# ==============================================================================
# 本模块负责解析和加载 YAML 配置文件
# 将用户定义的配置转换为 PipelineConfig 数据类
# 包含配置验证、默认值设置、辅助数据源默认配置等功能
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import EventWindow, OutputPaths, SourceSettings
from .utils import ensure_directories, parse_utc_datetime


@dataclass
class PipelineConfig:
    """
    流水线配置数据类

    存储从 YAML 文件加载的所有配置信息，包括项目设置、认证信息、
    事件定义、数据源配置、处理参数、输出路径等。

    Attributes:
        config_path: 配置文件路径
        project: 项目基本信息字典
        auth: 认证信息字典（如 CDDIS 用户名密码）
        events: 事件时间窗口列表
        observation_sources: 观测数据源配置字典
        auxiliary_sources: 辅助数据源配置字典
        bbox: 地理边界框配置字典
        download: 下载阶段参数字典
        processing: 处理阶段参数字典
        gridding: 格网化参数字典
        plot: 绘图参数字典
        validation: 验证参数字典
        outputs: 输出路径管理对象
    """
    config_path: Path
    project: dict[str, object]
    auth: dict[str, object]
    events: list[EventWindow]
    observation_sources: dict[str, SourceSettings]
    auxiliary_sources: dict[str, SourceSettings]
    bbox: dict[str, float]
    download: dict[str, float | int | str | bool]
    processing: dict[str, object]
    gridding: dict[str, float | int]
    plot: dict[str, float | int | str]
    validation: dict[str, object]
    outputs: OutputPaths


def load_pipeline_config(config_path: str | Path) -> PipelineConfig:
    """
    从 YAML 文件加载完整的流水线配置

    解析配置文件，验证必填字段，设置默认值，
    创建输出目录结构，返回包含所有配置的 PipelineConfig 对象。

    Args:
        config_path: YAML 配置文件路径

    Returns:
        完整的流水线配置对象（PipelineConfig）
    """
    # 将路径转换为 Path 对象并解析为绝对路径
    path = Path(config_path).resolve()
    # 打开并解析 YAML 文件
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    # 解析下载配置，设置默认值
    raw_download = raw.get("download", {})
    download = {
        "max_retries": int(raw_download.get("max_retries", 5)),
        "noaa_workers": int(raw_download.get("noaa_workers", 4)),
        "other_workers": int(raw_download.get("other_workers", 8)),
        "temp_suffix": str(raw_download.get("temp_suffix", ".part")),
        "aux_retries": int(raw_download.get("aux_retries", raw_download.get("max_retries", 5))),
    }

    # 解析处理、格网、验证、绘图配置
    processing = _build_processing_config(raw.get("processing", {}))
    gridding = _build_gridding_config(raw.get("gridding", raw.get("grid", {})))
    validation = _build_validation_config(raw.get("validation", {}))
    plot = _build_plot_config(raw.get("plot", {}))

    # 解析事件时间窗口列表
    events = [
        EventWindow(
            event_id=entry["id"],
            start_utc=parse_utc_datetime(entry["start"]),
            end_utc=parse_utc_datetime(entry["end"]),
        )
        for entry in raw["events"]
    ]

    # 解析观测数据源配置
    observation_settings = raw.get("sources", {}).get("observations")
    legacy_sources = raw.get("sources", {})
    # 兼容旧版配置格式
    if observation_settings is None:
        observation_settings = {
            key: value
            for key, value in legacy_sources.items()
            if isinstance(value, dict) and key not in {"auxiliary"}
        }

    # 解析辅助数据源配置，如果未指定则使用默认值
    auxiliary_settings = raw.get("sources", {}).get("auxiliary") or _default_auxiliary_sources()

    # 构建观测数据源设置字典
    observation_sources = {
        name: SourceSettings(
            name=name,
            enabled=bool(settings.get("enabled", True)),
            priority=int(settings["priority"]),
            timeout_sec=int(settings.get("timeout_sec", 60)),
            params={k: v for k, v in settings.items() if k not in {"enabled", "priority", "timeout_sec"}},
        )
        for name, settings in observation_settings.items()
    }

    # 构建辅助数据源设置字典
    auxiliary_sources = {
        name: SourceSettings(
            name=name,
            enabled=bool(settings.get("enabled", True)),
            priority=int(settings.get("priority", 100)),
            timeout_sec=int(settings.get("timeout_sec", 90)),
            params={k: v for k, v in settings.items() if k not in {"enabled", "priority", "timeout_sec"}},
        )
        for name, settings in auxiliary_settings.items()
    }

    # 解析输出路径配置
    outputs_cfg = raw.get("outputs", {})
    root = (path.parent.parent / outputs_cfg.get("root", "outputs/v2")).resolve()
    legacy_root = (path.parent.parent / outputs_cfg.get("legacy_root", "outputs")).resolve()

    # 创建输出路径管理对象
    outputs = OutputPaths(
        root=root,
        raw_dir=root / "raw",
        normalized_dir=root / "normalized",
        aux_dir=root / "aux",
        manifests_dir=root / "manifests",
        intermediate_dir=root / "intermediate",
        availability_dir=root / "intermediate" / "availability",
        arc_dir=root / "intermediate" / "arcs",
        stec_dir=root / "intermediate" / "stec",
        vtec_dir=root / "intermediate" / "vtec",
        roti_dir=root / "intermediate" / "roti",
        grid_dir=root / "intermediate" / "grids",
        validation_dir=root / "intermediate" / "validation",
        product_dir=root / "products",
        netcdf_dir=root / "products" / "netcdf",
        map_dir=root / "products" / "maps",
        log_dir=root / "logs",
        legacy_root=legacy_root,
        legacy_raw_dir=legacy_root / "raw",
        legacy_normalized_dir=legacy_root / "normalized",
    )

    # 确保所有必需的输出目录存在
    ensure_directories(
        [
            outputs.root,
            outputs.raw_dir,
            outputs.normalized_dir,
            outputs.aux_dir,
            outputs.manifests_dir,
            outputs.intermediate_dir,
            outputs.availability_dir,
            outputs.arc_dir,
            outputs.stec_dir,
            outputs.vtec_dir,
            outputs.roti_dir,
            outputs.grid_dir,
            outputs.validation_dir,
            outputs.product_dir,
            outputs.netcdf_dir,
            outputs.map_dir,
            outputs.log_dir,
        ]
    )

    # 返回完整的配置对象
    return PipelineConfig(
        config_path=path,
        project=dict(raw.get("project", {})),
        auth=raw.get("auth", {}),
        events=events,
        observation_sources=observation_sources,
        auxiliary_sources=auxiliary_sources,
        bbox={k: float(v) for k, v in raw["bbox"].items()},
        download=download,
        processing=processing,
        gridding=gridding,
        plot=plot,
        validation=validation,
        outputs=outputs,
    )


def _build_processing_config(raw_processing: dict[str, object]) -> dict[str, object]:
    """
    构建处理阶段配置，设置默认值

    Args:
        raw_processing: 从 YAML 加载的原始处理配置字典

    Returns:
        包含所有处理参数及默认值的字典
    """
    processing = dict(raw_processing)
    processing.setdefault("gnss_system", "G")
    processing.setdefault("target_interval_sec", 30)
    processing.setdefault("shell_height_km", 350.0)
    processing.setdefault("workers", 12)
    processing.setdefault("progress_log_interval", 100)
    processing.setdefault("checkpoint_chunk_size", 10)
    processing.setdefault("max_ephemeris_age_hours", 4.0)
    processing.setdefault("orbit_preference", ["sp3", "broadcast"])
    processing.setdefault("drop_zero_byte_raw", True)

    # VTEC 处理默认配置
    processing.setdefault(
        "vtec",
        {
            "cutoff_elevation_deg": 15.0,
            "leveling_method": "hatch",
            "rx_dcb_method": "external_then_gim_then_minimize_negative",
            "minimum_vtec_tecu": -2.0,
            "maximum_vtec_tecu": 200.0,
        },
    )

    # ROTI 处理默认配置
    processing.setdefault(
        "roti",
        {
            "cutoff_elevation_deg": 15.0,
            "window_length_min": 5,
            "min_valid_points_in_window": 6,
            "quality_low_elevation_deg": 20.0,
        },
    )

    # 弧段管理默认配置
    processing.setdefault(
        "arcs",
        {
            "max_gap_epochs": 2,
            "min_arc_length_min": 10,
            "post_cycleslip_buffer_epochs": 2,
        },
    )

    # 周跳检测默认配置
    processing.setdefault(
        "cycle_slip",
        {
            "enable_mw": True,
            "enable_gf": True,
            "mw_window_points": 10,
            "mw_slip_threshold_cycles": 4.0,
            "gf_window_points": 10,
            "gf_poly_degree": 2,
            "gf_residual_threshold_m": 0.12,
            "drop_detected_slip_epoch": True,
        },
    )
    return processing


def _build_gridding_config(raw_grid: dict[str, object]) -> dict[str, float | int]:
    """
    构建格网化配置，设置默认值

    Args:
        raw_grid: 从 YAML 加载的原始格网配置字典

    Returns:
        包含所有格网参数及默认值的字典
    """
    return {
        "lon_step_deg": float(raw_grid.get("lon_step_deg", 1.0)),
        "lat_step_deg": float(raw_grid.get("lat_step_deg", 1.0)),
        "time_step_min": int(raw_grid.get("time_step_min", raw_grid.get("cadence_minutes", 15))),
        "min_points_per_cell": int(raw_grid.get("min_points_per_cell", 3)),
        "netcdf_segment_hours": int(raw_grid.get("netcdf_segment_hours", 0)),
    }


def _build_plot_config(raw_plot: dict[str, object]) -> dict[str, float | int | str]:
    """
    构建绘图配置，设置默认值

    Args:
        raw_plot: 从 YAML 加载的原始绘图配置字典

    Returns:
        包含所有绘图参数及默认值的字典
    """
    return {
        "tec_vmin": float(raw_plot.get("tec_vmin", 0.0)),
        "tec_vmax": float(raw_plot.get("tec_vmax", 80.0)),
        "roti_vmin": float(raw_plot.get("roti_vmin", 0.0)),
        "roti_vmax": float(raw_plot.get("roti_vmax", 1.0)),
        "tec_cmap": str(raw_plot.get("tec_cmap", "jet")),
        "roti_cmap": str(raw_plot.get("roti_cmap", "jet")),
        "dpi": int(raw_plot.get("dpi", 180)),
        "magnetic_equator_step_deg": float(raw_plot.get("magnetic_equator_step_deg", 1.0)),
    }


def _build_validation_config(raw_validation: dict[str, object]) -> dict[str, object]:
    """
    构建验证配置，设置默认值

    Args:
        raw_validation: 从 YAML 加载的原始验证配置字典

    Returns:
        包含所有验证参数及默认值的字典
    """
    return {
        "enabled": bool(raw_validation.get("enabled", True)),
        "minimum_points": int(raw_validation.get("minimum_points", 10)),
    }


def _default_auxiliary_sources() -> dict[str, object]:
    """
    返回辅助数据源的默认配置

    辅助数据源包括：
    - SP3：精密星历产品
    - DCB：差分码偏差产品
    - IONEX：电离层格网产品

    Returns:
        辅助数据源配置字典
    """
    return {
        "sp3": {
            "enabled": True,
            "priority": 10,
            "timeout_sec": 120,
            "providers": [
                {
                    "name": "cddis_igs_fin_long",
                    "priority": 10,
                    "auth": "cddis",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/products/{gps_week}/IGS0OPSFIN_{yyyy}{ddd}0000_01D_15M_ORB.SP3.gz",
                },
                {
                    "name": "cddis_igs_rap_long",
                    "priority": 20,
                    "auth": "cddis",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/products/{gps_week}/IGS0OPSRAP_{yyyy}{ddd}0000_01D_15M_ORB.SP3.gz",
                },
                {
                    "name": "cddis_igs_fin_short",
                    "priority": 30,
                    "auth": "cddis",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/products/{gps_week}/igs{gps_week}{dow}.sp3.Z",
                },
                {
                    "name": "cddis_igs_rap_short",
                    "priority": 40,
                    "auth": "cddis",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/products/{gps_week}/igr{gps_week}{dow}.sp3.Z",
                },
            ],
        },
        "dcb": {
            "enabled": True,
            "priority": 20,
            "timeout_sec": 120,
            "providers": [
                {
                    "name": "cddis_cas1_ops_bia",
                    "priority": 10,
                    "auth": "cddis",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/products/bias/{yyyy}/CAS1OPSRAP_{yyyy}{ddd}0000_01D_01D_DCB.BIA.gz",
                },
                {
                    "name": "cddis_cas0_ops_bia",
                    "priority": 20,
                    "auth": "cddis",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/products/bias/{yyyy}/CAS0OPSRAP_{yyyy}{ddd}0000_01D_01D_DCB.BIA.gz",
                },
                {
                    "name": "cddis_cas0_mgx_bsx",
                    "priority": 30,
                    "auth": "cddis",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/products/bias/{yyyy}/CAS0MGXRAP_{yyyy}{ddd}0000_01D_01D_DCB.BSX.gz",
                },
            ],
        },
        "ionex": {
            "enabled": True,
            "priority": 30,
            "timeout_sec": 120,
            "providers": [
                {
                    "name": "cddis_igs_fin_long",
                    "priority": 10,
                    "auth": "cddis",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/products/ionex/{yyyy}/{ddd}/IGS0OPSFIN_{yyyy}{ddd}0000_01D_02H_GIM.INX.gz",
                },
                {
                    "name": "cddis_igs_rap_long",
                    "priority": 20,
                    "auth": "cddis",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/products/ionex/{yyyy}/{ddd}/IGS0OPSRAP_{yyyy}{ddd}0000_01D_02H_GIM.INX.gz",
                },
                {
                    "name": "cddis_igs_fin_short",
                    "priority": 30,
                    "auth": "cddis",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/products/ionosphere/{yyyy}/{ddd}/igsg{ddd}0.{yy}i.Z",
                },
                {
                    "name": "cddis_igs_rap_short",
                    "priority": 40,
                    "auth": "cddis",
                    "url_template": "https://cddis.nasa.gov/archive/gnss/products/ionosphere/{yyyy}/{ddd}/igrg{ddd}0.{yy}i.Z",
                },
            ],
        },
    }
