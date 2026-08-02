"""Dataset versioning — lightweight metadata for reproducibility.

This module provides pure data classes for dataset version tracking.
It does NOT manage any filesystem operations — versioning is the
responsibility of fusion-model-hub, not Fusion-Trainer.
"""

from __future__ import annotations

from typing import Any


class DatasetVersion:
    """Represents a specific version of a dataset.

    This is a pure data class with no filesystem coupling.
    Version metadata is typically provided by fusion-model-hub.
    """

    def __init__(
        self,
        name: str = "",
        version: str = "v1",
        path: str = "",
        hash: str = "",
        num_samples: int = 0,
    ):
        self.name = name
        self.version = version
        self.path = path
        self.hash = hash
        self.num_samples = num_samples

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "hash": self.hash,
            "num_samples": self.num_samples,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetVersion:
        return cls(
            name=d.get("dataset", d.get("name", "")),
            version=d.get("version", "v1"),
            path=d.get("file_path", d.get("path", "")),
            hash=d.get("file_hash", d.get("hash", "")),
            num_samples=d.get("num_samples", 0),
        )
