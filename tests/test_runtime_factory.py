import pytest

from quantplan.runtime.factory import expand_block_patterns, uniformly_select


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
