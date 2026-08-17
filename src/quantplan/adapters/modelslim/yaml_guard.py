from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatchcase
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from .capability import (
    CapabilityManifest,
    CapabilityStatus,
    ProcessCapability,
    UnknownFieldStatus,
    default_modelslim_manifest,
)


class IssueSeverity(str, Enum):
    WARNING = "warning"
    BLOCKING = "blocking"


@dataclass(frozen=True)
class FieldFinding:
    """Classification of one YAML mapping key at a capability boundary."""

    path: str
    status: UnknownFieldStatus

    @property
    def field_path(self) -> str:
        return self.path


@dataclass(frozen=True)
class GuardIssue:
    code: str
    path: str
    message: str
    severity: IssueSeverity

    @property
    def blocking(self) -> bool:
        return self.severity is IssueSeverity.BLOCKING


@dataclass(frozen=True)
class PatternExpansion:
    """Auditable expansion of one include/exclude pattern."""

    field_path: str
    pattern: str
    matched_modules: tuple[str, ...]

    @property
    def matches(self) -> tuple[str, ...]:
        return self.matched_modules

    @property
    def matched(self) -> bool:
        return bool(self.matched_modules)


@dataclass(frozen=True)
class YamlGuardResult:
    """Read-only static-policy result tied to the exact source bytes.

    Passing this guard never means the YAML is executable on a particular
    model, adapter, runtime or device.
    """

    source_path: str
    source_sha256: str
    source_size_bytes: int
    manifest_id: str
    findings: tuple[FieldFinding, ...]
    issues: tuple[GuardIssue, ...]
    pattern_expansions: tuple[PatternExpansion, ...]

    @property
    def sha256(self) -> str:
        return self.source_sha256

    @property
    def size_bytes(self) -> int:
        return self.source_size_bytes

    @property
    def expansions(self) -> tuple[PatternExpansion, ...]:
        return self.pattern_expansions

    @property
    def blocking_issues(self) -> tuple[GuardIssue, ...]:
        return tuple(issue for issue in self.issues if issue.blocking)

    @property
    def opaque_fields(self) -> tuple[FieldFinding, ...]:
        return tuple(
            finding
            for finding in self.findings
            if finding.status is UnknownFieldStatus.OPAQUE
        )

    @property
    def blocked_fields(self) -> tuple[FieldFinding, ...]:
        return tuple(
            finding
            for finding in self.findings
            if finding.status is UnknownFieldStatus.BLOCKING
        )

    @property
    def schema_safe(self) -> bool:
        return not self.blocking_issues

    @property
    def is_safe(self) -> bool:
        return self.schema_safe

    @property
    def static_policy_passed(self) -> bool:
        return self.schema_safe

    @property
    def execution_ready(self) -> bool:
        return False

    @property
    def execution_readiness(self) -> str:
        return "not_assessed"

    @property
    def blocked(self) -> bool:
        return not self.schema_safe

    def raise_if_blocked(self) -> None:
        if self.blocking_issues:
            raise UnsafeModelSlimYamlError(self)


class UnsafeModelSlimYamlError(ValueError):
    def __init__(self, result: YamlGuardResult) -> None:
        self.result = result
        summary = "; ".join(
            f"{issue.path}: {issue.message}" for issue in result.blocking_issues
        )
        super().__init__(f"ModelSlim YAML is blocked by the capability guard: {summary}")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects literal duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: MappingNode, deep: bool = False
) -> Mapping[Any, Any]:
    # Check the literal mapping before SafeLoader expands YAML merge keys. This
    # rejects ambiguous duplicates but still permits a mapping to override a key
    # inherited through ``<<`` as required by the YAML merge convention.
    seen: set[Any] = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in seen
            seen.add(key)
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _render_path(parts: Sequence[str | int]) -> str:
    rendered = "$"
    for part in parts:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif part.isidentifier():
            rendered += f".{part}"
        else:
            rendered += f"[{json.dumps(part, ensure_ascii=False)}]"
    return rendered


def _child_path(parts: tuple[str | int, ...], key: Any) -> str:
    if isinstance(key, (str, int)):
        return _render_path((*parts, key))
    return f"{_render_path(parts)}[{key!r}]"


class YamlGuard:
    """Fail-closed, non-mutating validator for a pinned ModelSlim YAML subset."""

    def __init__(self, manifest: CapabilityManifest | None = None) -> None:
        self.manifest = manifest or default_modelslim_manifest()

    def inspect(
        self,
        source: str | Path,
        *,
        module_names: Iterable[str] | None = None,
    ) -> YamlGuardResult:
        source_path = Path(source)
        source_bytes = source_path.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()

        inventory: tuple[str, ...] | None
        if module_names is None:
            inventory = None
        else:
            inventory = tuple(dict.fromkeys(module_names))

        inspector = _Inspection(self.manifest, inventory)
        try:
            document = yaml.load(source_bytes, Loader=_UniqueKeySafeLoader)
        except (yaml.YAMLError, UnicodeError) as exc:
            inspector.block(
                "yaml_parse_error",
                "$",
                f"YAML could not be parsed safely: {exc}",
            )
        else:
            inspector.inspect_document(document)

        return YamlGuardResult(
            source_path=str(source_path),
            source_sha256=source_sha256,
            source_size_bytes=len(source_bytes),
            manifest_id=self.manifest.fingerprint,
            findings=tuple(inspector.findings),
            issues=tuple(inspector.issues),
            pattern_expansions=tuple(inspector.expansions),
        )

    # ``validate`` is intentionally an alias returning evidence rather than
    # throwing; callers can opt into an exception with result.raise_if_blocked().
    validate = inspect
    check = inspect


class _Inspection:
    def __init__(
        self,
        manifest: CapabilityManifest,
        module_names: tuple[str, ...] | None,
    ) -> None:
        self.manifest = manifest
        self.module_names = module_names
        self.findings: list[FieldFinding] = []
        self.issues: list[GuardIssue] = []
        self.expansions: list[PatternExpansion] = []
        self._active_process_lists: set[int] = set()
        self._process_depth = 0

    def block(self, code: str, path: str, message: str) -> None:
        self.issues.append(
            GuardIssue(
                code=code,
                path=path,
                message=message,
                severity=IssueSeverity.BLOCKING,
            )
        )

    def warn(self, code: str, path: str, message: str) -> None:
        self.issues.append(
            GuardIssue(
                code=code,
                path=path,
                message=message,
                severity=IssueSeverity.WARNING,
            )
        )

    def inspect_document(self, document: Any) -> None:
        if not isinstance(document, Mapping):
            self.block("root_not_mapping", "$", "the YAML document root must be a mapping")
            return

        self._classify_fields(
            document,
            (),
            self.manifest.root_fields,
            self.manifest.unknown_root_status,
        )

        api_version = document.get("apiversion")
        if not isinstance(api_version, str):
            self.block(
                "apiversion_missing_or_invalid",
                "$.apiversion",
                "apiversion must be a string",
            )
        elif api_version not in self.manifest.supported_apiversions:
            self.block(
                "apiversion_unsupported",
                "$.apiversion",
                f"apiversion {api_version!r} is not declared by this capability manifest",
            )

        if "metadata" in document:
            self._inspect_metadata(document["metadata"], ("metadata",))

        if "spec" not in document:
            self.block("spec_missing", "$.spec", "spec is required")
            return
        self._inspect_spec(document["spec"], ("spec",))

    def _inspect_metadata(self, value: Any, path: tuple[str | int, ...]) -> None:
        if not isinstance(value, Mapping):
            self.block(
                "metadata_not_mapping",
                _render_path(path),
                "metadata must be a mapping when present",
            )
            return
        self._classify_fields(
            value,
            path,
            self.manifest.metadata_fields,
            self.manifest.unknown_metadata_status,
        )
        label = value.get("label")
        config_id = value.get("config_id")
        if config_id is not None and not isinstance(config_id, str):
            self.block(
                "metadata_config_id_invalid",
                _render_path((*path, "config_id")),
                "metadata.config_id must be a string",
            )
        score = value.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not isfinite(score)
        ):
            self.block(
                "metadata_score_invalid",
                _render_path((*path, "score")),
                "metadata.score must be numeric",
            )
        for field_name in ("verified_model_types", "verified_tags"):
            field_value = value.get(field_name)
            if field_value is not None and (
                not isinstance(field_value, list)
                or not all(isinstance(item, str) for item in field_value)
            ):
                self.block(
                    "metadata_list_invalid",
                    _render_path((*path, field_name)),
                    f"metadata.{field_name} must be a list of strings",
                )
        if label is None:
            return
        if not isinstance(label, Mapping):
            self.warn(
                "metadata_label_shape_opaque",
                _render_path((*path, "label")),
                "metadata.label is not a mapping and will remain opaque",
            )
            return
        self._classify_fields(
            label,
            (*path, "label"),
            self.manifest.metadata_label_fields,
            self.manifest.unknown_metadata_status,
        )
        for field_name, field_value in label.items():
            if field_name in {"is_sparse", "kv_cache", "fa_quant"}:
                valid = isinstance(field_value, bool)
            else:
                valid = isinstance(field_value, int) and not isinstance(field_value, bool)
            if field_name in self.manifest.metadata_label_fields and not valid:
                self.block(
                    "metadata_label_value_invalid",
                    _render_path((*path, "label", field_name)),
                    f"metadata.label.{field_name} has an invalid value type",
                )

    def _inspect_spec(self, value: Any, path: tuple[str | int, ...]) -> None:
        if not isinstance(value, Mapping):
            self.block("spec_not_mapping", _render_path(path), "spec must be a mapping")
            return
        self._classify_fields(
            value,
            path,
            self.manifest.spec_fields,
            self.manifest.unknown_semantic_status,
        )

        if "runner" in value:
            runner = value["runner"]
            if not isinstance(runner, str) or runner not in self.manifest.runner_values:
                self.block(
                    "runner_unsupported",
                    _render_path((*path, "runner")),
                    "runner is not declared by this pinned schema manifest",
                )
        if "dataset" in value:
            self._inspect_dataset(value["dataset"], (*path, "dataset"))
        if "default_text" in value and not isinstance(value["default_text"], str):
            self.block(
                "default_text_invalid",
                _render_path((*path, "default_text")),
                "default_text must be a string",
            )

        if "process" in value:
            self._inspect_process_list(value["process"], (*path, "process"))
        if "prior" in value:
            self._inspect_prior(value["prior"], (*path, "prior"))
        if "save" in value:
            self._inspect_savers(value["save"], (*path, "save"))

    def _inspect_prior(self, value: Any, path: tuple[str | int, ...]) -> None:
        if not isinstance(value, list):
            self.block("prior_not_list", _render_path(path), "spec.prior must be a list")
            return
        for index, stage in enumerate(value):
            stage_path = (*path, index)
            if not isinstance(stage, Mapping):
                self.block(
                    "prior_stage_not_mapping",
                    _render_path(stage_path),
                    "each prior stage must be a mapping",
                )
                continue
            self._classify_fields(
                stage,
                stage_path,
                self.manifest.prior_stage_fields,
                self.manifest.unknown_semantic_status,
            )
            if "process" in stage:
                self._inspect_process_list(
                    stage["process"],
                    (*stage_path, "process"),
                )
            if "dataset" in stage:
                self._inspect_dataset(stage["dataset"], (*stage_path, "dataset"))

    def _inspect_dataset(self, value: Any, path: tuple[str | int, ...]) -> None:
        if not isinstance(value, str) or not value or len(value) > 4096:
            self.block(
                "dataset_invalid",
                _render_path(path),
                "dataset must be a non-empty string of at most 4096 characters",
            )

    def _inspect_process_list(self, value: Any, path: tuple[str | int, ...]) -> None:
        if not isinstance(value, list):
            self.block(
                "process_not_list",
                _render_path(path),
                "process/configs must be a list",
            )
            return
        identity = id(value)
        if identity in self._active_process_lists:
            self.block(
                "process_cycle_blocked",
                _render_path(path),
                "recursive YAML aliases are not allowed in process lists",
            )
            return
        if self._process_depth >= 32:
            self.block(
                "process_depth_exceeded",
                _render_path(path),
                "nested process depth exceeds the static guard limit",
            )
            return
        self._active_process_lists.add(identity)
        self._process_depth += 1
        try:
            for index, process in enumerate(value):
                self._inspect_process(process, (*path, index))
        finally:
            self._process_depth -= 1
            self._active_process_lists.remove(identity)

    def _inspect_process(self, value: Any, path: tuple[str | int, ...]) -> None:
        if not isinstance(value, Mapping):
            self.block(
                "process_not_mapping",
                _render_path(path),
                "each process must be a mapping",
            )
            return

        type_name = value.get("type")
        capability: ProcessCapability | None = None
        if not isinstance(type_name, str) or not type_name:
            self.block(
                "process_type_missing_or_invalid",
                _render_path((*path, "type")),
                "process type must be a non-empty string",
            )
        else:
            capability = self.manifest.process_types.get(type_name)
            if capability is None:
                self.block(
                    "process_type_unregistered",
                    _render_path((*path, "type")),
                    f"process type {type_name!r} is not registered",
                )
            else:
                self._inspect_capability_status(
                    capability.status,
                    _render_path((*path, "type")),
                    f"process type {type_name!r}",
                )

        allowed_fields = capability.allowed_fields if capability else frozenset({"type"})
        self._classify_fields(
            value,
            path,
            allowed_fields,
            self.manifest.unknown_semantic_status,
        )
        if capability is None:
            return

        for required_field in sorted(capability.required_fields):
            if required_field not in value or value[required_field] is None:
                self.block(
                    "process_required_field_missing",
                    _render_path((*path, required_field)),
                    f"{capability.type_name!r} requires {required_field!r}",
                )

        for field_name in value:
            if field_name not in capability.qconfig_fields:
                continue
            if field_name in value and value[field_name] is not None:
                self._inspect_linear_qconfig(
                    value[field_name],
                    (*path, field_name),
                )
        for field_name in value:
            if field_name not in capability.direct_qconfig_fields:
                continue
            if field_name in value and value[field_name] is not None:
                self._inspect_quantizer(value[field_name], (*path, field_name))
        for field_name in value:
            if field_name not in capability.qconfig_maps:
                continue
            branches = capability.qconfig_maps[field_name]
            if field_name in value and value[field_name] is not None:
                self._inspect_qconfig_map(
                    value[field_name],
                    (*path, field_name),
                    frozenset(branches),
                )
        for field_name in value:
            if field_name not in capability.nested_process_fields:
                continue
            if field_name in value:
                self._inspect_process_list(value[field_name], (*path, field_name))
        if type_name == "fa3_quant" and value.get("qconfig") is not None and value.get(
            "details"
        ) is not None:
            self.block(
                "fa3_qconfig_details_conflict",
                _render_path(path),
                "fa3_quant qconfig and details are mutually exclusive",
            )
        if type_name == "dynamic_cache" and isinstance(value.get("qconfig"), Mapping):
            if value["qconfig"].get("scope") != "per_channel":
                self.block(
                    "dynamic_cache_scope_invalid",
                    _render_path((*path, "qconfig", "scope")),
                    "dynamic_cache requires scope 'per_channel' in this manifest",
                )

        include_matches: tuple[str, ...] | None = None
        if "include" in capability.pattern_fields and "include" in value:
            include_matches = self._inspect_patterns(
                value["include"], (*path, "include"), universe=self.module_names
            )
        elif type_name == "linear_quant":
            self.block(
                "linear_include_required",
                _render_path((*path, "include")),
                "linear_quant requires an explicit non-empty include list in this guard",
            )

        exclude_matches: tuple[str, ...] = ()
        if "exclude" in capability.pattern_fields and "exclude" in value:
            exclude_universe = include_matches if include_matches is not None else self.module_names
            exclude_matches = self._inspect_patterns(
                value["exclude"], (*path, "exclude"), universe=exclude_universe
            )
        for field_name in capability.pattern_fields - {"include", "exclude"}:
            if field_name in value:
                self._inspect_patterns(value[field_name], (*path, field_name))
        if include_matches is not None and not set(include_matches) - set(exclude_matches):
            self.block(
                "pattern_net_target_empty",
                _render_path(path),
                "include/exclude resolution leaves no target module",
            )

    def _inspect_linear_qconfig(
        self, value: Any, path: tuple[str | int, ...]
    ) -> None:
        if not isinstance(value, Mapping):
            self.block("qconfig_not_mapping", _render_path(path), "qconfig must be a mapping")
            return
        self._classify_fields(
            value,
            path,
            self.manifest.qconfig_fields,
            self.manifest.unknown_semantic_status,
        )
        if "weight" not in value or value["weight"] is None:
            self.block(
                "linear_weight_qconfig_required",
                _render_path((*path, "weight")),
                "linear qconfig requires a non-null weight quantizer",
            )
        for branch_name in self.manifest.qconfig_fields:
            if branch_name not in value or value[branch_name] is None:
                continue
            self._inspect_quantizer(value[branch_name], (*path, branch_name))

    def _inspect_qconfig_map(
        self,
        value: Any,
        path: tuple[str | int, ...],
        allowed_branches: frozenset[str],
    ) -> None:
        if not isinstance(value, Mapping):
            self.block(
                "qconfig_map_not_mapping",
                _render_path(path),
                "named qconfig details must be a mapping",
            )
            return
        self._classify_fields(
            value,
            path,
            allowed_branches,
            self.manifest.unknown_semantic_status,
        )
        if not any(value.get(branch_name) is not None for branch_name in allowed_branches):
            self.block(
                "qconfig_map_empty",
                _render_path(path),
                "named qconfig details must contain at least one quantizer",
            )
        for branch_name in allowed_branches:
            if branch_name not in value or value[branch_name] is None:
                continue
            self._inspect_quantizer(value[branch_name], (*path, branch_name))

    def _inspect_quantizer(self, value: Any, path: tuple[str | int, ...]) -> None:
        if not isinstance(value, Mapping):
            self.block(
                "quantizer_not_mapping",
                _render_path(path),
                "quantizer configuration must be a mapping or null",
            )
            return
        self._classify_fields(
            value,
            path,
            self.manifest.quantizer_fields,
            self.manifest.unknown_semantic_status,
        )
        for field_name in self.manifest.quantizer_required_fields:
            if field_name not in value or value[field_name] is None:
                self.block(
                    "quantizer_required_field_missing",
                    _render_path((*path, field_name)),
                    f"quantizer requires {field_name!r}",
                )
        dtype = value.get("dtype")
        if dtype is not None and (
            not isinstance(dtype, str) or dtype not in self.manifest.quantizer_dtypes
        ):
            self.block(
                "quantizer_dtype_unsupported",
                _render_path((*path, "dtype")),
                f"dtype {dtype!r} is not declared by this manifest",
            )
        scope = value.get("scope")
        if scope is not None and (
            not isinstance(scope, str) or scope not in self.manifest.quantizer_scopes
        ):
            self.block(
                "quantizer_scope_unsupported",
                _render_path((*path, "scope")),
                f"scope {scope!r} is not declared by this manifest",
            )
        symmetric = value.get("symmetric")
        if symmetric is not None and not isinstance(symmetric, bool):
            self.block(
                "quantizer_symmetric_invalid",
                _render_path((*path, "symmetric")),
                "symmetric must be a boolean",
            )
        method = value.get("method")
        if method is not None and (
            not isinstance(method, str) or method not in self.manifest.quantizer_methods
        ):
            self.block(
                "quantizer_method_unsupported",
                _render_path((*path, "method")),
                f"method {method!r} is not declared by this manifest",
            )
        if "ext" not in value or value["ext"] is None:
            return
        ext = value["ext"]
        ext_path = (*path, "ext")
        if not isinstance(ext, Mapping):
            self.block(
                "qconfig_ext_not_mapping",
                _render_path(ext_path),
                "quantizer ext must be a mapping",
            )
            return
        self._classify_fields(
            ext,
            ext_path,
            self.manifest.qconfig_ext_fields,
            self.manifest.unknown_semantic_status,
        )

        for field_name, field_value in ext.items():
            if field_name not in self.manifest.qconfig_ext_fields:
                continue
            field_path = _render_path((*ext_path, field_name))
            if field_name == "aggregation_type":
                valid = isinstance(field_value, str) and bool(field_value)
            elif field_name == "axes":
                valid = (
                    isinstance(field_value, int)
                    and not isinstance(field_value, bool)
                ) or (
                    isinstance(field_value, list)
                    and bool(field_value)
                    and all(
                        isinstance(item, int) and not isinstance(item, bool)
                        for item in field_value
                    )
                )
            else:
                valid = (
                    isinstance(field_value, (int, float))
                    and not isinstance(field_value, bool)
                    and isfinite(field_value)
                    and field_value > 0
                )
            if not valid:
                self.block(
                    "qconfig_ext_value_invalid",
                    field_path,
                    f"qconfig ext field {field_name!r} has an invalid value",
                )

    def _inspect_patterns(
        self,
        value: Any,
        path: tuple[str | int, ...],
        *,
        universe: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        if isinstance(value, list):
            patterns = value
        else:
            self.block(
                "patterns_not_list",
                _render_path(path),
                "include/exclude must be a list of strings",
            )
            return ()
        if not patterns:
            self.block(
                "patterns_empty",
                _render_path(path),
                "include/exclude lists must not be empty",
            )
            return ()

        all_matches: list[str] = []
        for index, pattern in enumerate(patterns):
            pattern_path = (*path, index)
            if not isinstance(pattern, str) or not pattern:
                self.block(
                    "pattern_invalid",
                    _render_path(pattern_path),
                    "each include/exclude pattern must be a non-empty string",
                )
                continue
            if "{" in pattern or "}" in pattern:
                self.block(
                    "pattern_dialect_unsupported",
                    _render_path(pattern_path),
                    "brace expansion requires a pinned wcmatch-compatible adapter",
                )
                continue
            matches = (
                ()
                if universe is None
                else tuple(
                    module_name
                    for module_name in universe
                    if fnmatchcase(module_name, pattern)
                )
            )
            self.expansions.append(
                PatternExpansion(
                    field_path=_render_path(path),
                    pattern=pattern,
                    matched_modules=matches,
                )
            )
            all_matches.extend(matches)
            if universe is None:
                self.block(
                    "module_inventory_missing",
                    _render_path(pattern_path),
                    "a module inventory is required to verify every include/exclude pattern",
                )
            elif not matches:
                self.block(
                    "pattern_unmatched",
                    _render_path(pattern_path),
                    f"pattern {pattern!r} matched no module",
                )
        return tuple(dict.fromkeys(all_matches))

    def _inspect_savers(self, value: Any, path: tuple[str | int, ...]) -> None:
        if not isinstance(value, list):
            self.block("save_not_list", _render_path(path), "spec.save must be a list")
            return
        for index, saver in enumerate(value):
            saver_path = (*path, index)
            if not isinstance(saver, Mapping):
                self.block(
                    "saver_not_mapping",
                    _render_path(saver_path),
                    "each saver must be a mapping",
                )
                continue
            type_name = saver.get("type")
            capability = (
                self.manifest.saver_types.get(type_name)
                if isinstance(type_name, str)
                else None
            )
            if not isinstance(type_name, str) or not type_name:
                self.block(
                    "saver_type_missing_or_invalid",
                    _render_path((*saver_path, "type")),
                    "saver type must be a non-empty string",
                )
            elif capability is None:
                self.block(
                    "saver_type_unregistered",
                    _render_path((*saver_path, "type")),
                    f"saver type {type_name!r} is not registered",
                )
            else:
                self._inspect_capability_status(
                    capability.status,
                    _render_path((*saver_path, "type")),
                    f"saver type {type_name!r}",
                )
            self._classify_fields(
                saver,
                saver_path,
                capability.allowed_fields if capability else frozenset({"type"}),
                self.manifest.unknown_semantic_status,
            )
            if "part_file_size" in saver:
                part_size = saver["part_file_size"]
                if (
                    not isinstance(part_size, int)
                    or isinstance(part_size, bool)
                    or part_size <= 0
                ):
                    self.block(
                        "saver_part_file_size_invalid",
                        _render_path((*saver_path, "part_file_size")),
                        "part_file_size must be a positive integer",
                    )
            if "save_directory" in saver and (
                not isinstance(saver["save_directory"], str)
                or not saver["save_directory"]
            ):
                self.block(
                    "saver_directory_invalid",
                    _render_path((*saver_path, "save_directory")),
                    "save_directory must be a non-empty string",
                )

    def _inspect_capability_status(
        self,
        status: CapabilityStatus,
        path: str,
        description: str,
    ) -> None:
        if status is CapabilityStatus.SUPPORTED_EXPERIMENTAL:
            self.warn(
                "schema_fragment_experimental",
                path,
                f"{description} has experimental static schema coverage",
            )
        elif status in {
            CapabilityStatus.KNOWN_UNSUPPORTED,
            CapabilityStatus.UNKNOWN,
            CapabilityStatus.OUT_OF_SCOPE,
        }:
            self.block(
                "schema_fragment_not_supported",
                path,
                f"{description} has static schema status {status.value!r}",
            )

    def _classify_fields(
        self,
        mapping: Mapping[Any, Any],
        path: tuple[str | int, ...],
        known_fields: frozenset[str],
        unknown_status: UnknownFieldStatus,
    ) -> None:
        for key in mapping:
            field_path = _child_path(path, key)
            if not isinstance(key, str):
                status = UnknownFieldStatus.BLOCKING
                self.findings.append(FieldFinding(path=field_path, status=status))
                self.block(
                    "non_string_field_blocked",
                    field_path,
                    "mapping keys at a capability boundary must be strings",
                )
                continue

            status = (
                UnknownFieldStatus.KNOWN if key in known_fields else unknown_status
            )
            self.findings.append(FieldFinding(path=field_path, status=status))
            if status is UnknownFieldStatus.OPAQUE:
                self.warn(
                    "unknown_field_opaque",
                    field_path,
                    "field is not interpreted and remains opaque",
                )
            elif status is UnknownFieldStatus.BLOCKING:
                self.block(
                    "unknown_field_blocked",
                    field_path,
                    "field is not declared at this semantic capability boundary",
                )


def inspect_modelslim_yaml(
    source: str | Path,
    *,
    manifest: CapabilityManifest | None = None,
    module_names: Iterable[str] | None = None,
) -> YamlGuardResult:
    return YamlGuard(manifest).inspect(source, module_names=module_names)


def validate_modelslim_yaml(
    source: str | Path,
    *,
    manifest: CapabilityManifest | None = None,
    module_names: Iterable[str] | None = None,
) -> YamlGuardResult:
    return inspect_modelslim_yaml(
        source,
        manifest=manifest,
        module_names=module_names,
    )


# Integration-facing aliases; all are evidence-returning and never rewrite the
# source file.
guard_yaml = inspect_modelslim_yaml
check_yaml = inspect_modelslim_yaml
