from pathlib import Path

import pytest

from quantplan.domain import PrecisionFormat
from quantplan.runtime.capability import (
    CAPABILITY_SCHEMA,
    CapabilityManifestError,
    load_capability_manifest,
)


def _manifest(tmp_path: Path, *, hardware: str = "a5") -> Path:
    path = tmp_path / "capability.yaml"
    path.write_text(
        f"""\
schema_version: {CAPABILITY_SCHEMA}
backend: {{name: vllm_ascend, version: 0.27.1, operator: prolog_v3}}
hardware: {{profile: {hardware}}}
domains:
  - id: mla_preprocess
    status: verified
    legal_modes: [native, w4a8_mxfp]
    allowed_precisions: [w4a8, w8a8, bf16]
    owners: ['*q_proj*']
    contract: {{weight_uk: {{dtype: bf16, layout: ND}}}}
    fallback: [native]
""",
        encoding="utf-8",
    )
    return path


def test_capability_manifest_is_hardware_bound_and_matches_owner(tmp_path: Path) -> None:
    snapshot = load_capability_manifest(_manifest(tmp_path), hardware_profile="a5")

    assert snapshot.known is True
    assert snapshot.domain_for_owner("model.layers.0.q_proj").allowed_precisions == (
        PrecisionFormat.W4A8,
        PrecisionFormat.W8A8,
        PrecisionFormat.BF16,
    )
    assert len(snapshot.sha256) == 64


def test_capability_manifest_rejects_hardware_mismatch(tmp_path: Path) -> None:
    with pytest.raises(CapabilityManifestError, match="does not match"):
        load_capability_manifest(_manifest(tmp_path, hardware="a3"), hardware_profile="a5")
