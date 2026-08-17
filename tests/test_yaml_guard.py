from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from quantplan.adapters.modelslim.capability import (
    UnknownFieldStatus,
    default_modelslim_manifest,
)
from quantplan.adapters.modelslim.yaml_guard import (
    UnsafeModelSlimYamlError,
    inspect_modelslim_yaml,
)


def _write_yaml(path: Path, document: object) -> Path:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _linear_document() -> dict:
    return {
        "apiversion": "modelslim_v1",
        "metadata": {"config_id": "test"},
        "spec": {
            "process": [
                {
                    "type": "linear_quant",
                    "qconfig": {
                        "act": {
                            "dtype": "int8",
                            "scope": "per_token",
                            "symmetric": True,
                            "method": "minmax",
                        },
                        "weight": {
                            "dtype": "int8",
                            "scope": "per_channel",
                            "symmetric": True,
                            "method": "minmax",
                            "ext": {"group_size": 128},
                        },
                    },
                }
            ]
        },
    }


def test_guard_is_read_only_hashes_exact_bytes_and_keeps_metadata_opaque(
    tmp_path: Path,
) -> None:
    source = tmp_path / "practice.yaml"
    original = b"""\
apiversion: modelslim_v1
helper_anchor: &shared
  note: do-not-rewrite
metadata:
  config_id: qwen-test
  deployment_ticket: OPS-7
  label:
    w_bit: 8
    private_badge: canary
spec:
  process:
    - type: linear_quant
      qconfig:
        act: {dtype: int8, scope: per_token, symmetric: true, method: minmax}
        weight: {dtype: int8, scope: per_channel, symmetric: true, method: minmax}
      include:
        - 'model.layers.*.self_attn.q_proj'
      exclude:
        - 'model.layers.0.self_attn.q_proj'
"""
    source.write_bytes(original)

    result = inspect_modelslim_yaml(
        source,
        module_names=(
            "model.layers.0.self_attn.q_proj",
            "model.layers.1.self_attn.q_proj",
            "model.layers.1.self_attn.k_proj",
        ),
    )

    assert source.read_bytes() == original
    assert result.source_sha256 == hashlib.sha256(original).hexdigest()
    assert result.sha256 == result.source_sha256
    assert result.source_size_bytes == len(original)
    assert result.static_policy_passed
    assert not result.execution_ready
    assert {finding.path for finding in result.opaque_fields} == {
        "$.helper_anchor",
        "$.metadata.deployment_ticket",
        "$.metadata.label.private_badge",
    }
    assert all(
        finding.status is UnknownFieldStatus.OPAQUE for finding in result.opaque_fields
    )
    assert [expansion.pattern for expansion in result.pattern_expansions] == [
        "model.layers.*.self_attn.q_proj",
        "model.layers.0.self_attn.q_proj",
    ]
    assert all(expansion.matched for expansion in result.pattern_expansions)


@pytest.mark.parametrize(
    ("mutate", "blocked_path"),
    [
        (lambda doc: doc["spec"].__setitem__("future_mode", True), "$.spec.future_mode"),
        (
            lambda doc: doc["spec"]["process"][0].__setitem__("future_mode", True),
            "$.spec.process[0].future_mode",
        ),
        (
            lambda doc: doc["spec"]["process"][0]["qconfig"].__setitem__(
                "future_branch", {}
            ),
            "$.spec.process[0].qconfig.future_branch",
        ),
        (
            lambda doc: doc["spec"]["process"][0]["qconfig"]["weight"].__setitem__(
                "future_scale", 1
            ),
            "$.spec.process[0].qconfig.weight.future_scale",
        ),
        (
            lambda doc: doc["spec"]["process"][0]["qconfig"]["weight"][
                "ext"
            ].__setitem__("future_ext", 1),
            "$.spec.process[0].qconfig.weight.ext.future_ext",
        ),
    ],
)
def test_unknown_execution_semantics_fail_closed(
    tmp_path: Path,
    mutate,
    blocked_path: str,
) -> None:
    document = _linear_document()
    mutate(document)
    source = _write_yaml(tmp_path / "unknown.yaml", document)

    result = inspect_modelslim_yaml(source)

    assert result.blocked
    assert blocked_path in {finding.path for finding in result.blocked_fields}
    assert any(
        issue.code == "unknown_field_blocked" and issue.path == blocked_path
        for issue in result.blocking_issues
    )
    with pytest.raises(UnsafeModelSlimYamlError):
        result.raise_if_blocked()


def test_process_type_must_be_registered_and_custom_registration_is_local(
    tmp_path: Path,
) -> None:
    document = {
        "apiversion": "modelslim_v1",
        "spec": {
            "process": [
                {
                    "type": "vendor_quant",
                    "mode": "safe",
                    "include": ["layers.*.mlp.down_proj"],
                }
            ]
        },
    }
    source = _write_yaml(tmp_path / "vendor.yaml", document)

    default_result = inspect_modelslim_yaml(
        source,
        module_names=("layers.0.mlp.down_proj",),
    )
    assert default_result.blocked
    assert any(
        issue.code == "process_type_unregistered"
        for issue in default_result.blocking_issues
    )

    manifest = default_modelslim_manifest()
    manifest.register_process_type(
        "vendor_quant",
        allowed_fields={"mode", "include"},
        required_fields={"mode", "include"},
        pattern_fields={"include"},
    )
    registered_result = inspect_modelslim_yaml(
        source,
        manifest=manifest,
        module_names=("layers.0.mlp.down_proj",),
    )

    assert registered_result.static_policy_passed
    assert registered_result.pattern_expansions[0].matched_modules == (
        "layers.0.mlp.down_proj",
    )
    # Registration does not broaden a separately-created default manifest.
    assert "vendor_quant" not in default_modelslim_manifest().process_types


def test_every_include_and_exclude_pattern_must_match(tmp_path: Path) -> None:
    document = _linear_document()
    process = document["spec"]["process"][0]
    process["include"] = ["layers.*.q_proj", "layers.99.*"]
    process["exclude"] = ["layers.*.k_proj"]
    source = _write_yaml(tmp_path / "patterns.yaml", document)

    result = inspect_modelslim_yaml(
        source,
        module_names=("layers.0.q_proj", "layers.0.k_proj"),
    )

    assert result.blocked
    assert len(result.pattern_expansions) == 3
    assert result.pattern_expansions[0].matched_modules == ("layers.0.q_proj",)
    assert result.pattern_expansions[1].matched_modules == ()
    # ModelSlim applies exclude after include, so k_proj is outside this
    # processor's candidate universe and the exclusion is treated as unmatched.
    assert result.pattern_expansions[2].matched_modules == ()
    assert any(
        issue.code == "pattern_unmatched"
        and issue.path == "$.spec.process[0].include[1]"
        for issue in result.blocking_issues
    )


def test_patterns_require_module_inventory(tmp_path: Path) -> None:
    document = _linear_document()
    document["spec"]["process"][0]["include"] = ["layers.*"]
    source = _write_yaml(tmp_path / "no-inventory.yaml", document)

    result = inspect_modelslim_yaml(source)

    assert result.blocked
    assert any(
        issue.code == "module_inventory_missing" for issue in result.blocking_issues
    )


def test_duplicate_mapping_keys_are_blocked_without_changing_source(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.yaml"
    original = b"""\
apiversion: modelslim_v1
spec:
  process: []
  process: []
"""
    source.write_bytes(original)

    result = inspect_modelslim_yaml(source)

    assert result.blocked
    assert any(issue.code == "yaml_parse_error" for issue in result.blocking_issues)
    assert result.sha256 == hashlib.sha256(original).hexdigest()
    assert source.read_bytes() == original


def test_direct_dynamic_cache_qconfig_is_not_treated_as_linear(tmp_path: Path) -> None:
    document = {
        "apiversion": "modelslim_v1",
        "spec": {
            "process": [
                {
                    "type": "dynamic_cache",
                    "qconfig": {
                        "dtype": "int8",
                        "scope": "per_channel",
                        "symmetric": True,
                        "method": "minmax",
                    },
                    "include": ["model.layers.*.self_attn"],
                }
            ]
        },
    }
    source = _write_yaml(tmp_path / "dynamic.yaml", document)

    result = inspect_modelslim_yaml(
        source, module_names=("model.layers.0.self_attn",)
    )

    assert result.static_policy_passed
    assert not result.execution_ready


def test_dynamic_cache_rejects_linear_qconfig_shape(tmp_path: Path) -> None:
    document = {
        "apiversion": "modelslim_v1",
        "spec": {
            "process": [
                {
                    "type": "dynamic_cache",
                    "qconfig": _linear_document()["spec"]["process"][0]["qconfig"],
                    "include": ["model.layers.*.self_attn"],
                }
            ]
        },
    }
    source = _write_yaml(tmp_path / "wrong-dynamic.yaml", document)

    result = inspect_modelslim_yaml(
        source, module_names=("model.layers.0.self_attn",)
    )

    assert result.blocked
    assert any(
        issue.code == "unknown_field_blocked" for issue in result.blocking_issues
    )


def test_fa3_qconfig_and_details_are_mutually_exclusive(tmp_path: Path) -> None:
    quantizer = {
        "dtype": "fp8_e4m3",
        "scope": "per_head",
        "symmetric": True,
        "method": "minmax",
    }
    document = {
        "apiversion": "modelslim_v1",
        "spec": {
            "process": [
                {
                    "type": "fa3_quant",
                    "qconfig": quantizer,
                    "details": {"fa_q": quantizer},
                }
            ]
        },
    }
    source = _write_yaml(tmp_path / "fa3-conflict.yaml", document)

    result = inspect_modelslim_yaml(source)

    assert result.blocked
    assert any(
        issue.code == "fa3_qconfig_details_conflict"
        for issue in result.blocking_issues
    )


@pytest.mark.parametrize(
    ("field", "value", "issue_code"),
    [
        ("dtype", "future_int3", "quantizer_dtype_unsupported"),
        ("scope", "nonsense", "quantizer_scope_unsupported"),
        ("symmetric", "yes", "quantizer_symmetric_invalid"),
        ("method", "future_method", "quantizer_method_unsupported"),
    ],
)
def test_known_quantizer_fields_validate_values(
    tmp_path: Path, field: str, value: object, issue_code: str
) -> None:
    document = _linear_document()
    document["spec"]["process"][0]["include"] = ["layers.*.mlp"]
    document["spec"]["process"][0]["qconfig"]["weight"][field] = value
    source = _write_yaml(tmp_path / f"invalid-{field}.yaml", document)

    result = inspect_modelslim_yaml(source, module_names=("layers.0.mlp",))

    assert result.blocked
    assert any(issue.code == issue_code for issue in result.blocking_issues)


def test_linear_weight_qconfig_is_required(tmp_path: Path) -> None:
    document = _linear_document()
    document["spec"]["process"][0]["include"] = ["layers.*.mlp"]
    document["spec"]["process"][0]["qconfig"]["weight"] = None
    source = _write_yaml(tmp_path / "missing-weight.yaml", document)

    result = inspect_modelslim_yaml(source, module_names=("layers.0.mlp",))

    assert result.blocked
    assert any(
        issue.code == "linear_weight_qconfig_required"
        for issue in result.blocking_issues
    )


@pytest.mark.parametrize("patterns", ["layers.*", []])
def test_patterns_must_be_non_empty_lists(tmp_path: Path, patterns: object) -> None:
    document = _linear_document()
    document["spec"]["process"][0]["include"] = patterns
    source = _write_yaml(tmp_path / "bad-patterns.yaml", document)

    result = inspect_modelslim_yaml(source, module_names=("layers.0.mlp",))

    assert result.blocked
    assert any(
        issue.code in {"patterns_not_list", "patterns_empty"}
        for issue in result.blocking_issues
    )


def test_recursive_process_alias_is_blocked(tmp_path: Path) -> None:
    source = tmp_path / "recursive.yaml"
    source.write_text(
        """\
apiversion: modelslim_v1
spec:
  process: &loop
    - type: group
      configs: *loop
""",
        encoding="utf-8",
    )

    result = inspect_modelslim_yaml(source)

    assert result.blocked
    assert any(
        issue.code == "process_cycle_blocked" for issue in result.blocking_issues
    )


def test_manifest_fingerprint_changes_after_local_registration() -> None:
    manifest = default_modelslim_manifest()
    original = manifest.fingerprint

    manifest.register_process_type("vendor_quant", allowed_fields={"mode"})

    assert manifest.fingerprint != original


@pytest.mark.parametrize(
    ("mutate", "issue_code"),
    [
        (lambda doc: doc.__setitem__("metadata", 42), "metadata_not_mapping"),
        (lambda doc: doc["spec"].__setitem__("runner", 123), "runner_unsupported"),
        (lambda doc: doc["spec"].__setitem__("dataset", ["x"]), "dataset_invalid"),
        (
            lambda doc: doc["spec"].__setitem__(
                "save", [{"type": "ascendv1_saver", "part_file_size": -1}]
            ),
            "saver_part_file_size_invalid",
        ),
    ],
)
def test_known_container_values_fail_closed(
    tmp_path: Path, mutate, issue_code: str
) -> None:
    document = _linear_document()
    document["spec"]["process"][0]["include"] = ["layers.*.mlp"]
    mutate(document)
    source = _write_yaml(tmp_path / f"{issue_code}.yaml", document)

    result = inspect_modelslim_yaml(source, module_names=("layers.0.mlp",))

    assert result.blocked
    assert any(issue.code == issue_code for issue in result.blocking_issues)


def test_brace_patterns_require_pinned_backend_dialect(tmp_path: Path) -> None:
    document = _linear_document()
    document["spec"]["process"][0]["include"] = ["layers.{0,1}.mlp"]
    source = _write_yaml(tmp_path / "brace.yaml", document)

    result = inspect_modelslim_yaml(
        source, module_names=("layers.0.mlp", "layers.1.mlp")
    )

    assert result.blocked
    assert any(
        issue.code == "pattern_dialect_unsupported"
        for issue in result.blocking_issues
    )
