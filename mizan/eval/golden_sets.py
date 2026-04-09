"""JSON-backed golden set storage for Phase 3."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from config.settings import Settings
from mizan.eval.models import GoldenSetBundle, GoldenSetEntry


class GoldenSetStore:
    """Read and publish immutable golden-set versions."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the store."""

        self._settings = settings
        self._root = settings.paths.golden_set_dir

    def publish(self, name: str, version: str, entries: list[GoldenSetEntry]) -> GoldenSetBundle:
        """Publish a new immutable golden-set version."""

        bundle = GoldenSetBundle(name=name, version=version, entries=entries)
        path = self._bundle_path(name, version)
        if path.exists():
            raise FileExistsError(f"Golden set already published: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bundle.model_dump(mode="json"), indent=2), encoding="utf-8")
        return bundle.model_copy(update={"source_path": path})

    def load_by_version(self, name: str, version: str) -> GoldenSetBundle:
        """Load one published golden-set version."""

        path = self._bundle_path(name, version)
        bundle = GoldenSetBundle.model_validate_json(path.read_text(encoding="utf-8"))
        return bundle.model_copy(update={"source_path": path})

    def load_by_tag(
        self, tag_key: str, tag_value: str, name: str | None = None
    ) -> list[GoldenSetEntry]:
        """Load all entries matching one tag constraint."""

        matches: list[GoldenSetEntry] = []
        for bundle in self._iter_bundles():
            if name is not None and bundle.name != name:
                continue
            matches.extend(
                entry
                for entry in bundle.entries
                if entry.tags.get(tag_key) == tag_value
            )
        return matches

    def export_to_csv(self, output_path: Path, name: str, version: str) -> Path:
        """Export one published golden set to CSV."""

        bundle = self.load_by_version(name, version)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["id", "prompt", "expected_output", "tags", "version", "created_at"],
            )
            writer.writeheader()
            for entry in bundle.entries:
                writer.writerow(
                    {
                        "id": entry.id,
                        "prompt": entry.prompt,
                        "expected_output": entry.expected_output,
                        "tags": json.dumps(entry.tags, sort_keys=True),
                        "version": entry.version,
                        "created_at": entry.created_at.isoformat(),
                    }
                )
        return output_path

    def _iter_bundles(self) -> list[GoldenSetBundle]:
        """Load every stored bundle under the golden-set root."""

        bundles: list[GoldenSetBundle] = []
        for path in sorted(self._root.glob("*.json")):
            bundle = GoldenSetBundle.model_validate_json(path.read_text(encoding="utf-8"))
            bundles.append(bundle.model_copy(update={"source_path": path}))
        return bundles

    def _bundle_path(self, name: str, version: str) -> Path:
        """Return the JSON path for one published bundle."""

        return self._root / f"{name}__{version}.json"
