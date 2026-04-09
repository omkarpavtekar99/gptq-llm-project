"""YAML-backed prompt registry for Phase 3."""

from __future__ import annotations

from difflib import unified_diff
from pathlib import Path

import yaml

from config.settings import Settings
from mizan.eval.models import PromptTemplate


class PromptRegistry:
    """Store and retrieve versioned prompt templates."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the registry."""

        self._root = settings.paths.prompt_dir

    def save(self, template: PromptTemplate) -> Path:
        """Save a new prompt template version."""

        path = self._template_path(template.name, template.version)
        if path.exists():
            raise FileExistsError(f"Prompt template already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(template.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
        return path

    def get(self, name: str, version: str) -> PromptTemplate:
        """Load one prompt template version."""

        return PromptTemplate.model_validate(
            yaml.safe_load(self._template_path(name, version).read_text(encoding="utf-8"))
        )

    def list_versions(self, name: str) -> list[str]:
        """List known versions for one prompt name."""

        pattern = self._root / f"{name}__*.yaml"
        versions: list[str] = []
        for path in sorted(self._root.glob(pattern.name)):
            versions.append(path.stem.removeprefix(f"{name}__"))
        return versions

    def diff(self, name: str, left_version: str, right_version: str) -> str:
        """Return a clean unified diff of two prompt contents."""

        left = self.get(name, left_version)
        right = self.get(name, right_version)
        return "\n".join(
            unified_diff(
                left.content.splitlines(),
                right.content.splitlines(),
                fromfile=f"{name}:{left_version}",
                tofile=f"{name}:{right_version}",
                lineterm="",
            )
        )

    def _template_path(self, name: str, version: str) -> Path:
        """Return the YAML path for one prompt template version."""

        return self._root / f"{name}__{version}.yaml"
