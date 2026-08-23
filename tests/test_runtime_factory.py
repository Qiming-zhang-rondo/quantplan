from pathlib import Path
from unittest.mock import patch

import pytest

from quantplan.run_config import loads_run_config
from quantplan.runtime.factory import (
    build_torch_runtime,
    expand_block_patterns,
    uniformly_select,
)


def test_expand_block_patterns_requires_every_pattern_to_match() -> None:
    with pytest.raises(ValueError, match="matched no"):
        expand_block_patterns(("model.layers.9",), ("model.layers.0",))


def test_expand_block_patterns_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="matched by both"):
        expand_block_patterns(
            ("model.layers.*", "model.layers.0"),
            ("model.layers.0", "model.layers.1"),
        )


def test_uniform_selection_covers_depth_boundaries() -> None:
    values = tuple(f"layer.{index}" for index in range(40))

    selected = uniformly_select(values, 8)

    assert len(selected) == 8
    assert selected[0] == "layer.0"
    assert selected[-1] == "layer.39"


def test_build_runtime_forwards_explicit_image_text_fallback(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "data.jsonl"
    data.write_text('{"text":"hello"}\n', encoding="utf-8")
    source_yaml = tmp_path / "source.yaml"
    source_yaml.write_text("apiversion: modelslim_v1\nspec: {}\n", encoding="utf-8")
    config = loads_run_config(
        f"""
schema_version: quantplan_v1
model_path: {model}
data_path: {data}
original_yaml: {source_yaml}
output_dir: {tmp_path / 'out'}
device: cpu
runtime:
  allow_image_text_fallback: true
"""
    )
    with patch(
        "quantplan.runtime.factory.load_causal_lm",
        side_effect=RuntimeError("stop after loader invocation"),
    ) as loader, pytest.raises(RuntimeError, match="stop after loader invocation"):
        build_torch_runtime(config)

    assert loader.call_args.kwargs["allow_image_text_fallback"] is True
