from pathlib import Path

import pytest

from quantplan.adapters.modelslim import (
    SourcePlanError,
    profile_linear_source,
    resolve_precision_templates,
)
from quantplan.domain import PrecisionFormat


def test_source_profile_preserves_recipe_and_changes_only_weight_dtype(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.yaml"
    source.write_text(
        """\
apiversion: modelslim_v1
spec:
  process:
    - type: linear_quant
      qconfig:
        act: {dtype: int8, scope: per_token, symmetric: true, method: minmax}
        weight:
          dtype: int8
          scope: per_channel
          symmetric: true
          method: ssz
          ext: {group_size: 128}
      include: ['model.layers.*']
      exclude: ['*o_proj']
""",
        encoding="utf-8",
    )
    modules = (
        "model.layers.0.q_proj",
        "model.layers.0.o_proj",
        "model.layers.1.mlp.experts",
    )

    profile = profile_linear_source(source, modules)
    resolution = resolve_precision_templates(
        profile,
        (PrecisionFormat.W4A8, PrecisionFormat.W8A8),
    )

    assert profile.eligible_modules == (modules[0], modules[2])
    assert resolution.templates[PrecisionFormat.W8A8]["weight"]["dtype"] == "int8"
    w4_weight = resolution.templates[PrecisionFormat.W4A8]["weight"]
    assert w4_weight == {
        "dtype": "int4",
        "scope": "per_channel",
        "symmetric": True,
        "method": "ssz",
        "ext": {"group_size": 128},
    }
    assert "weight.dtype_only" in resolution.origins[PrecisionFormat.W4A8]


def test_source_profile_rejects_multiple_recipes_for_one_precision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.yaml"
    source.write_text(
        """\
apiversion: modelslim_v1
spec:
  process:
    - type: linear_quant
      qconfig:
        act: {dtype: int8, scope: per_token, symmetric: true, method: minmax}
        weight: {dtype: int8, scope: per_channel, symmetric: true, method: minmax}
      include: ['model.layers.0.*']
    - type: linear_quant
      qconfig:
        act: {dtype: int8, scope: per_token, symmetric: true, method: minmax}
        weight: {dtype: int8, scope: per_channel, symmetric: true, method: ssz}
      include: ['model.layers.1.*']
""",
        encoding="utf-8",
    )

    with pytest.raises(SourcePlanError, match="multiple w8a8 qconfig recipes"):
        profile_linear_source(
            source,
            ("model.layers.0.q_proj", "model.layers.1.q_proj"),
        )
