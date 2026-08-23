"""Crash-safe, path-confined artifact writing for a QuantPlan run."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class ArtifactError(RuntimeError):
    """An artifact path or overwrite policy was violated."""


class ArtifactStore:
    """Write immutable-by-default artifacts below one output directory."""

    def __init__(self, root: str | Path, *, overwrite: bool = False) -> None:
        self.root = Path(root).resolve()
        self.overwrite = bool(overwrite)
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ArtifactError(f"artifact root is not a directory: {self.root}")

    def path(self, relative: str | Path) -> Path:
        candidate = (self.root / Path(relative)).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactError(f"artifact path escapes output directory: {relative}") from exc
        if candidate == self.root:
            raise ArtifactError("artifact path must identify a file")
        return candidate

    def write_json(self, relative: str | Path, value: Any) -> Path:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        return self.write_text(relative, text)

    def write_jsonl(
        self,
        relative: str | Path,
        records: Iterable[Mapping[str, Any]],
    ) -> Path:
        lines = [
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for record in records
        ]
        return self.write_text(relative, "\n".join(lines) + ("\n" if lines else ""))

    def write_text(self, relative: str | Path, text: str) -> Path:
        if not isinstance(text, str):
            raise TypeError("artifact text must be a string")
        destination = self.path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not self.overwrite:
            raise ArtifactError(
                f"artifact already exists: {destination}; rerun with overwrite enabled"
            )

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return destination
