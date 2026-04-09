"""Tests for the golden set store."""

from __future__ import annotations

from pathlib import Path

from config.settings import Settings
from mizan.eval.golden_sets import GoldenSetStore
from mizan.eval.models import GoldenSetEntry


def test_publish_and_load_round_trip(tmp_path: Path) -> None:
    """Published golden sets should be loadable by version."""

    settings = Settings()
    settings.paths.golden_set_dir = tmp_path
    store = GoldenSetStore(settings)
    entry = GoldenSetEntry(
        id="g1",
        prompt="hello",
        expected_output="world",
        tags={"domain": "test"},
        version="1.0.0",
    )

    store.publish("demo", "1.0.0", [entry])
    loaded = store.load_by_version("demo", "1.0.0")

    assert loaded.entries[0].id == "g1"


def test_export_to_csv_creates_file(tmp_path: Path) -> None:
    """CSV export should materialize the requested bundle."""

    settings = Settings()
    settings.paths.golden_set_dir = tmp_path / "golden"
    store = GoldenSetStore(settings)
    store.publish(
        "demo",
        "1.0.0",
        [
            GoldenSetEntry(
                id="g1",
                prompt="hello",
                expected_output="world",
                tags={"domain": "test"},
                version="1.0.0",
            )
        ],
    )
    output = store.export_to_csv(tmp_path / "demo.csv", "demo", "1.0.0")

    assert output.exists()
