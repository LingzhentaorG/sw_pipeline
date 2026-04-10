from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .app import fetch_target, plot_target, process_target, run_event
from .cleanup import clean_run_outputs, clean_workspace
from .config import load_app_config
from .registry.legacy_import import migrate_legacy_project
from .storage import ensure_storage_layout


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified space weather event pipeline")
    parser.add_argument(
        "--base-config",
        default=None,
        help="Optional override for config/base.yaml",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an event workflow")
    run_subparsers = run_parser.add_subparsers(dest="run_command", required=True)
    run_event_parser = run_subparsers.add_parser("event", help="Run a full event")
    run_event_parser.add_argument("--event", required=True, help="Event id")
    run_event_parser.add_argument(
        "--include-fetch",
        action="store_true",
        help="Also run fetch stages before process/plot stages.",
    )

    fetch_parser = subparsers.add_parser("fetch", help="Fetch source assets")
    fetch_parser.add_argument("target", choices=("gnss-raw", "gnss-grid", "gold", "omni"))
    fetch_parser.add_argument("--event", required=True, help="Event id")

    process_parser = subparsers.add_parser("process", help="Process source assets")
    process_parser.add_argument("target", choices=("gnss", "gold", "omni"))
    process_parser.add_argument("--event", required=True, help="Event id")

    plot_parser = subparsers.add_parser("plot", help="Render event figures")
    plot_parser.add_argument(
        "target",
        choices=("gnss-map", "gold-map", "omni-series", "overlay", "station-series", "panel"),
    )
    plot_parser.add_argument("--event", required=True, help="Event id")

    migrate_parser = subparsers.add_parser("migrate-legacy", help="Import existing project data into storage/cache")
    migrate_parser.add_argument("--from", dest="source_path", required=True, help="Source project or parent workspace path")

    clean_parser = subparsers.add_parser("clean", help="Clean reproducible outputs without touching storage/cache")
    clean_subparsers = clean_parser.add_subparsers(dest="clean_command", required=True)

    clean_workspace_parser = clean_subparsers.add_parser("workspace", help="Remove repo caches and empty log directories")
    clean_workspace_parser.add_argument(
        "--project-root",
        default=None,
        help="Optional override for the project root. Defaults to the current package root.",
    )

    clean_run_parser = clean_subparsers.add_parser("run", help="Remove generated outputs for one event run")
    clean_run_parser.add_argument("--event", required=True, help="Event id")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "migrate-legacy":
            migrate_legacy_project(Path(args.source_path).expanduser().resolve())
            return 0

        if args.command == "clean":
            if args.clean_command == "workspace":
                root = (
                    Path(args.project_root).expanduser().resolve()
                    if args.project_root is not None
                    else Path(__file__).resolve().parents[2]
                )
                clean_workspace(root)
                return 0

            event_spec = load_app_config(args.event, args.base_config)
            ensure_storage_layout(event_spec.storage)
            clean_run_outputs(event_spec.storage)
            return 0

        event_spec = load_app_config(args.event, args.base_config)
        ensure_storage_layout(event_spec.storage)
        if args.command == "run":
            run_event(event_spec, include_fetch=bool(args.include_fetch))
            return 0

        if args.command == "fetch":
            fetch_target(event_spec, args.target)
            return 0

        if args.command == "process":
            process_target(event_spec, args.target)
            return 0

        if args.command == "plot":
            plot_target(event_spec, args.target)
            return 0
    except Exception as exc:
        LOGGER.exception("%s", exc)
        return 1

    parser.error("Unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
