import json
from pathlib import Path

import pytest

from quantplan.artifacts import ArtifactError, ArtifactStore


def test_artifact_store_writes_atomic_json_and_refuses_overwrite(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    destination = store.write_json("nested/report.json", {"status": "complete"})

    assert json.loads(destination.read_text(encoding="utf-8"))["status"] == "complete"
    with pytest.raises(ArtifactError, match="already exists"):
        store.write_json("nested/report.json", {"status": "different"})


def test_artifact_store_confines_paths_to_output_root(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)

    with pytest.raises(ArtifactError, match="escapes"):
        store.write_text("../outside.txt", "unsafe")


def test_artifact_store_can_replace_only_when_explicit(tmp_path: Path) -> None:
    ArtifactStore(tmp_path).write_text("value.txt", "first")
    ArtifactStore(tmp_path, overwrite=True).write_text("value.txt", "second")

    assert (tmp_path / "value.txt").read_text(encoding="utf-8") == "second"
