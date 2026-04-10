from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd
import yaml

from .manifests import write_migration_manifest


OMNI_RANGE_PATTERN = re.compile(
    r"^(?P<prefix>omni(?:_bz_1min|_dst_kp_hourly|_kp_3hour)?)_(?P<start>\d{8})_(?P<end>\d{8})\.csv$",
    re.IGNORECASE,
)


def migrate_legacy_project(source_path: Path) -> Path:
    if not source_path.exists():
        raise FileNotFoundError(f"Legacy project does not exist: {source_path}")

    project_root = Path(__file__).resolve().parents[3]
    cache_root = project_root / "storage" / "cache"
    migration_root = cache_root / "migrations"
    migration_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    candidates = [source_path]
    for child_name in ("lzt_prj", "lzt_thesis_code"):
        child = source_path / child_name
        if child.exists():
            candidates.append(child)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        rows.extend(_migrate_candidate(candidate, cache_root, project_root))

    manifest_path = migration_root / f"{source_path.name}_import.csv"
    write_migration_manifest(manifest_path, rows)
    return manifest_path


def _migrate_candidate(source_path: Path, cache_root: Path, project_root: Path) -> list[dict[str, object]]:
    if (source_path / "outputs" / "v2").exists():
        return _migrate_internal_gnss_workspace(source_path, cache_root)
    if _is_omni_data_source(source_path):
        return _migrate_omni_outputs(source_path, cache_root, project_root)
    if (
        (source_path / "GNSSdraw").exists()
        or (source_path / "GOLDdraw").exists()
        or source_path.name in {"GNSSdraw", "GOLDdraw", "OMNIdarw", "Data_download", "ROTI_data", "VTEC_data"}
    ):
        return _migrate_thesis_assets(source_path, cache_root, project_root)
    return _migrate_generic_files(source_path, cache_root)


def _migrate_internal_gnss_workspace(source_path: Path, cache_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    outputs_root = source_path / "outputs" / "v2"
    manifests_dir = outputs_root / "manifests"
    normalized_manifest = manifests_dir / "normalized_manifest.csv"
    observation_manifest = manifests_dir / "observation_manifest.csv"

    event_ids = _discover_internal_event_ids(outputs_root, normalized_manifest)
    for event_id in event_ids:
        workspace_root = cache_root / "gnss_raw" / "internal" / event_id
        (workspace_root / "manifests").mkdir(parents=True, exist_ok=True)

        if normalized_manifest.exists():
            frame = pd.read_csv(normalized_manifest)
            if "event_id" in frame.columns:
                frame = frame[frame["event_id"].astype(str) == event_id].copy()
            frame = _sanitize_legacy_paths(frame)
            target = workspace_root / "manifests" / "normalized_manifest.csv"
            frame.to_csv(target, index=False)
            rows.append(_migration_row(source_path.name, "gnss_raw", event_id, target, normalized_manifest))

        if observation_manifest.exists():
            frame = pd.read_csv(observation_manifest)
            if "event_id" in frame.columns:
                frame = frame[frame["event_id"].astype(str) == event_id].copy()
            frame = _sanitize_legacy_paths(frame)
            target = workspace_root / "manifests" / "observation_manifest.csv"
            frame.to_csv(target, index=False)
            rows.append(_migration_row(source_path.name, "gnss_raw", event_id, target, observation_manifest))

        for relative_root, pattern in (
            ("intermediate/vtec", f"{event_id}*.parquet"),
            ("intermediate/roti", f"{event_id}*.parquet"),
            ("intermediate/arcs", f"{event_id}*.parquet"),
            ("intermediate/stec", f"{event_id}*.parquet"),
            ("products/netcdf", f"{event_id}*.nc"),
        ):
            source_dir = outputs_root / relative_root
            if not source_dir.exists():
                continue
            target_dir = workspace_root / relative_root
            for path in sorted(source_dir.glob(pattern)):
                copied = _copy_file(path, target_dir / path.name)
                rows.append(_migration_row(source_path.name, "gnss_raw", event_id, copied, path))
    return rows


def _migrate_thesis_assets(source_path: Path, cache_root: Path, project_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    gnss_root = _resolve_gnssdraw_root(source_path)
    if gnss_root.exists():
        target_root = cache_root / "gnss_grid" / "isee"
        for path in sorted(gnss_root.rglob("*.nc")):
            relative = path.relative_to(gnss_root)
            if not relative.parts or relative.parts[0] not in {"ROTI_data", "VTEC_data"}:
                continue
            copied = _copy_file(path, target_root / path.relative_to(gnss_root))
            rows.append(_migration_row(source_path.name, "gnss_grid", "", copied, path, gnss_root.parent))

    gold_root = source_path if source_path.name == "GOLDdraw" else source_path / "GOLDdraw"
    if gold_root.exists():
        target_root = cache_root / "gold"
        for path in sorted(gold_root.glob("*.tar")):
            copied = _copy_file(path, target_root / path.name)
            rows.append(_migration_row(source_path.name, "gold", "", copied, path, gold_root))

    omni_root = source_path if source_path.name == "OMNIdarw" else source_path / "OMNIdarw"
    if omni_root.exists():
        rows.extend(_migrate_omni_outputs(omni_root, cache_root, project_root))

    return rows


def _migrate_omni_outputs(source_path: Path, cache_root: Path, project_root: Path) -> list[dict[str, object]]:
    data_root = _resolve_omni_data_root(source_path)
    if not data_root.exists():
        return []

    event_windows = _load_event_windows(project_root)
    source_files = sorted(data_root.glob("*.csv"))
    rows: list[dict[str, object]] = []
    for event_id, window in event_windows.items():
        matched = _match_omni_files_for_event(source_files, window["start"], window["end"])
        for kind, source_file in matched.items():
            target_name = _canonical_omni_name(kind, event_id)
            target_path = cache_root / "omni" / "local" / event_id / target_name
            copied = _copy_file(source_file, target_path)
            rows.append(_migration_row(source_path.name, "omni", event_id, copied, source_file, data_root))
    return rows


def _migrate_generic_files(source_path: Path, cache_root: Path) -> list[dict[str, object]]:
    target_root = cache_root / "legacy_imports" / source_path.name
    rows: list[dict[str, object]] = []
    for pattern, source_kind in (
        ("**/*.nc", "netcdf"),
        ("**/*.parquet", "parquet"),
        ("**/*.tar", "tar"),
        ("**/*.csv", "csv"),
    ):
        for file_path in sorted(source_path.glob(pattern)):
            copied = _copy_file(file_path, target_root / file_path.relative_to(source_path))
            rows.append(_migration_row(source_path.name, source_kind, "", copied, file_path, source_path))
    return rows


def _discover_internal_event_ids(outputs_root: Path, normalized_manifest: Path) -> list[str]:
    event_ids: set[str] = set()
    if normalized_manifest.exists():
        frame = pd.read_csv(normalized_manifest)
        if "event_id" in frame.columns:
            event_ids.update(frame["event_id"].dropna().astype(str).tolist())
    if event_ids:
        return sorted(event_ids)

    for path in sorted((outputs_root / "products" / "netcdf").glob("*.nc")):
        stem = path.stem
        parts = stem.split("_")
        if len(parts) >= 5:
            event_ids.add("_".join(parts[:-2]))
    return sorted(event_ids)


def _sanitize_legacy_paths(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sanitized = frame.copy()
    for column in sanitized.columns:
        if not column.endswith("_path"):
            continue
        sanitized[column] = sanitized[column].map(lambda value: _sanitize_path_value(value))
    return sanitized


def _sanitize_path_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    path = Path(str(value))
    if path.is_absolute():
        return ""
    return str(path)


def _copy_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def _migration_row(
    project: str,
    source_kind: str,
    event_id: str,
    cached_path: Path,
    source_path: Path,
    source_root: Path | None = None,
) -> dict[str, object]:
    original_relpath = source_path.name
    if source_root is not None:
        try:
            original_relpath = str(source_path.relative_to(source_root))
        except ValueError:
            original_relpath = source_path.name
    return {
        "project": project,
        "source_kind": source_kind,
        "event_id": event_id,
        "cached_path": cached_path,
        "original_relpath": original_relpath,
        "size_bytes": cached_path.stat().st_size,
    }


def _resolve_gnssdraw_root(source_path: Path) -> Path:
    if source_path.name in {"ROTI_data", "VTEC_data"}:
        return source_path.parent
    if source_path.name == "Data_download":
        return source_path
    if source_path.name == "GNSSdraw":
        return source_path / "Data_download"
    return source_path / "GNSSdraw" / "Data_download"


def _is_omni_data_source(source_path: Path) -> bool:
    if source_path.name == "data" and source_path.parent.name == "outputs":
        return True
    if source_path.name == "OMNIdarw":
        return True
    if (source_path / "outputs" / "data").exists():
        return True
    return any(path.name.lower().startswith("omni_bz_1min_") for path in source_path.glob("*.csv"))


def _resolve_omni_data_root(source_path: Path) -> Path:
    if source_path.name == "data" and source_path.parent.name == "outputs":
        return source_path
    if source_path.name == "OMNIdarw":
        return source_path / "outputs" / "data"
    return source_path / "outputs" / "data"


def _load_event_windows(project_root: Path) -> dict[str, dict[str, pd.Timestamp]]:
    event_windows: dict[str, dict[str, pd.Timestamp]] = {}
    events_root = project_root / "config" / "events"
    for path in sorted(events_root.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        event = payload.get("event", {})
        event_id = str(event.get("id", path.stem))
        if "start" not in event or "end" not in event:
            continue
        event_windows[event_id] = {
            "start": pd.Timestamp(event["start"], tz="UTC"),
            "end": pd.Timestamp(event["end"], tz="UTC"),
        }
    return event_windows


def _match_omni_files_for_event(
    source_files: list[Path],
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
) -> dict[str, Path]:
    matches: dict[str, list[tuple[pd.Timedelta, Path]]] = {"bz": [], "hourly": [], "kp": []}
    for path in source_files:
        parsed = _parse_omni_file(path)
        if parsed is None:
            continue
        coverage_start, coverage_end, kind = parsed
        if coverage_start > event_end or coverage_end < event_start:
            continue
        coverage_width = coverage_end - coverage_start
        matches[kind].append((coverage_width, path))

    selected: dict[str, Path] = {}
    for kind, candidates in matches.items():
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1].name))
            selected[kind] = candidates[0][1]
    return selected


def _parse_omni_file(path: Path) -> tuple[pd.Timestamp, pd.Timestamp, str] | None:
    match = OMNI_RANGE_PATTERN.match(path.name)
    if match is None:
        return None
    start = pd.Timestamp(match.group("start"), tz="UTC")
    end = pd.Timestamp(match.group("end"), tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    prefix = match.group("prefix").lower()
    if "bz_1min" in prefix:
        kind = "bz"
    elif "dst_kp_hourly" in prefix:
        kind = "hourly"
    elif "kp_3hour" in prefix:
        kind = "kp"
    else:
        return None
    return start, end, kind


def _canonical_omni_name(kind: str, event_id: str) -> str:
    if kind == "bz":
        return f"omni_bz_1min_{event_id}.csv"
    if kind == "hourly":
        return f"omni_dst_kp_hourly_{event_id}.csv"
    if kind == "kp":
        return f"omni_kp_3hour_{event_id}.csv"
    raise ValueError(f"Unsupported OMNI kind: {kind}")
