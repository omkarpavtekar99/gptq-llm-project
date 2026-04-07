"""Phase 2 speech benchmark entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from config.settings import get_settings
from mizan.logging_setup import get_logger
from mizan.speech.benchmarking import run_benchmark

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the Phase 2 benchmark."""

    parser = argparse.ArgumentParser(description="Run the Phase 2 speech benchmark.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional path to a benchmark manifest JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and log its outputs."""

    args = parse_args()
    settings = get_settings()
    settings.ensure_directories()
    manifest_path = None if args.manifest is None else Path(args.manifest)
    if manifest_path is not None and not manifest_path.is_absolute():
        manifest_path = settings.paths.project_root / manifest_path
    report = run_benchmark(settings, manifest_path=manifest_path)
    LOGGER.info(
        "phase2_benchmark_complete",
        extra={
            "record_count": len(report.records),
            "csv_path": str(settings.paths.phase2_benchmark_csv),
            "summary_path": str(settings.paths.phase2_summary_md),
        },
    )


if __name__ == "__main__":
    main()
