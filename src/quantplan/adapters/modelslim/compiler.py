from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

from quantplan.domain.models import PrecisionFormat

from .capability import CapabilityManifest, default_modelslim_manifest
from .yaml_guard import YamlGuardResult, inspect_modelslim_yaml


class CompilationBlockCode(str, Enum):
    SOURCE_YAML_BLOCKED = "source_yaml_blocked"
    PLAN_INVALID = "plan_invalid"
    TEMPLATE_INVALID = "template_invalid"
    TEMPLATE_MISSING = "template_missing"
    UNSAFE_PATTERN_NAME = "unsafe_pattern_name"
    UNSAFE_SOURCE_STRUCTURE = "unsafe_source_structure"
    DERIVED_YAML_BLOCKED = "derived_yaml_blocked"
    DERIVED_YAML_MISMATCH = "derived_yaml_mismatch"


class PlanCompilationBlocked(ValueError):
    """Raised when a plan cannot be represented without guessing semantics."""

    def __init__(
        self,
        code: CompilationBlockCode,
        message: str,
        *,
        details: Sequence[str] = (),
        source_sha256: str | None = None,
    ) -> None:
        self.code = code
        self.details = tuple(details)
        self.source_sha256 = source_sha256
        suffix = f" ({'; '.join(self.details)})" if self.details else ""
        super().__init__(f"{code.value}: {message}{suffix}")


# A shorter public alias reads naturally at call sites.
ModelSlimPlanBlocked = PlanCompilationBlocked


@dataclass(frozen=True)
class PathDiff:
    operation: str
    path: str
    before: Any
    after: Any
    before_sha256: str | None
    after_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "path": self.path,
            "before": copy.deepcopy(self.before),
            "after": copy.deepcopy(self.after),
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
        }


@dataclass(frozen=True)
class AssignmentProvenance:
    group_id: str
    precision: PrecisionFormat
    module_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "precision": self.precision.value,
            "module_names": list(self.module_names),
        }


@dataclass(frozen=True)
class TemplateProvenance:
    precision: PrecisionFormat
    qconfig_sha256: str
    qconfig: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision.value,
            "qconfig_sha256": self.qconfig_sha256,
            "qconfig": copy.deepcopy(dict(self.qconfig)),
        }


@dataclass(frozen=True)
class CompilationSidecar:
    schema_version: str
    compiler: str
    source_path: str
    source_sha256: str
    derived_sha256: str
    capability_manifest_id: str
    execution_readiness: str
    quant_target_inventory_sha256: str
    quant_target_count: int
    model_module_inventory_sha256: str
    model_module_count: int
    assignments: tuple[AssignmentProvenance, ...]
    templates: tuple[TemplateProvenance, ...]
    bf16_modules: tuple[str, ...]
    quantized_modules: tuple[str, ...]
    diff: tuple[PathDiff, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "compiler": self.compiler,
            "source": {
                "path": self.source_path,
                "sha256": self.source_sha256,
            },
            "derived": {"sha256": self.derived_sha256},
            "capability_manifest_id": self.capability_manifest_id,
            "execution_readiness": self.execution_readiness,
            "inventories": {
                "quant_targets": {
                    "sha256": self.quant_target_inventory_sha256,
                    "count": self.quant_target_count,
                },
                "model_modules": {
                    "sha256": self.model_module_inventory_sha256,
                    "count": self.model_module_count,
                },
            },
            "assignments": [item.to_dict() for item in self.assignments],
            "templates": [item.to_dict() for item in self.templates],
            "bf16_modules": list(self.bf16_modules),
            "quantized_modules": list(self.quantized_modules),
            "diff": [item.to_dict() for item in self.diff],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )


@dataclass(frozen=True)
class PlanCompilationResult:
    """Pure compiler output; callers decide where derived artifacts are written."""

    derived_yaml: str
    sidecar: CompilationSidecar
    output_guard: YamlGuardResult

    @property
    def yaml_text(self) -> str:
        return self.derived_yaml

    @property
    def yaml_bytes(self) -> bytes:
        return self.derived_yaml.encode("utf-8")

    @property
    def sidecar_dict(self) -> dict[str, Any]:
        return self.sidecar.to_dict()

    def sidecar_json(self, *, indent: int = 2) -> str:
        return self.sidecar.to_json(indent=indent)


@dataclass
class _LinearSlot:
    parent: list[Any]
    indices: tuple[int, ...]
    path: str


class _IndentedSafeDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False):
        return super().increase_indent(flow, False)


class ModelSlimPlanCompiler:
    """Compile an assignment into a guarded ModelSlim V1 linear-quant plan.

    The compiler owns only the existing linear-quant slot (or an empty/missing
    process list). Other processors and all non-process fields are preserved.
    Runtime executability remains deliberately unassessed.
    """

    COMPILER_ID = "quantplan.modelslim-plan-compiler/v1"
    _QUANT_PRECISIONS = (PrecisionFormat.W4A8, PrecisionFormat.W8A8)
    _ALLOWED_PRECISIONS = frozenset(
        {PrecisionFormat.BF16, PrecisionFormat.W4A8, PrecisionFormat.W8A8}
    )
    _PATTERN_META = frozenset("*?[]{}")

    def __init__(self, manifest: CapabilityManifest | None = None) -> None:
        self.manifest = manifest or default_modelslim_manifest()

    def compile(
        self,
        source: str | Path,
        group_modules: Mapping[str, Iterable[str]],
        assignments: Mapping[str, PrecisionFormat | str],
        precision_templates: Mapping[PrecisionFormat | str, Mapping[str, Any]],
        module_names: Iterable[str],
        *,
        model_module_names: Iterable[str] | None = None,
        module_patterns: Mapping[str, str] | None = None,
    ) -> PlanCompilationResult:
        """Compile without writing either the source or a derived artifact.

        ``module_names`` is the exact quantization-target inventory and must be
        partitioned by ``group_modules``. ``model_module_names`` is the broader
        model inventory used only to prove source/derived include and exclude
        patterns; when omitted, the target inventory is assumed to be complete.
        """
        source_path = Path(source)
        source_bytes = source_path.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()

        inventory = self._normalize_inventory(module_names, source_sha256)
        guard_inventory = self._normalize_model_inventory(
            model_module_names,
            target_inventory=inventory,
            source_sha256=source_sha256,
        )
        groups = self._normalize_groups(group_modules, inventory, source_sha256)
        patterns = self._normalize_module_patterns(
            module_patterns,
            inventory,
            source_sha256,
        )
        normalized_assignments = self._normalize_assignments(
            assignments,
            groups,
            source_sha256,
        )
        templates = self._normalize_templates(
            precision_templates,
            normalized_assignments,
            source_sha256,
        )

        source_guard = inspect_modelslim_yaml(
            source_path,
            manifest=self.manifest,
            module_names=guard_inventory,
        )
        if source_guard.source_sha256 != source_sha256:
            raise PlanCompilationBlocked(
                CompilationBlockCode.SOURCE_YAML_BLOCKED,
                "the source YAML changed while it was being inspected",
                details=(
                    f"snapshot={source_sha256}",
                    f"guard={source_guard.source_sha256}",
                ),
                source_sha256=source_sha256,
            )
        if source_guard.blocked:
            raise PlanCompilationBlocked(
                CompilationBlockCode.SOURCE_YAML_BLOCKED,
                "the original YAML did not pass the static capability policy",
                details=tuple(
                    f"{issue.code}@{issue.path}" for issue in source_guard.blocking_issues
                ),
                source_sha256=source_sha256,
            )

        try:
            source_text = source_bytes.decode("utf-8")
            document = yaml.safe_load(source_text)
        except (UnicodeError, yaml.YAMLError) as exc:
            raise PlanCompilationBlocked(
                CompilationBlockCode.SOURCE_YAML_BLOCKED,
                "the original YAML cannot be decoded and parsed as UTF-8 YAML",
                details=(str(exc),),
                source_sha256=source_sha256,
            ) from exc

        if not isinstance(document, dict) or not isinstance(document.get("spec"), dict):
            raise PlanCompilationBlocked(
                CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                "the source must contain a mapping at $.spec",
                source_sha256=source_sha256,
            )

        spec = document["spec"]
        process_was_present = "process" in spec
        old_process = spec.get("process", [])
        if not isinstance(old_process, list):
            raise PlanCompilationBlocked(
                CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                "$.spec.process must be a list",
                source_sha256=source_sha256,
            )
        new_process = copy.deepcopy(old_process)
        generated_configs = self._build_linear_configs(
            inventory,
            groups,
            normalized_assignments,
            templates,
            patterns,
        )
        self._install_linear_configs(
            new_process,
            generated_configs,
            source_sha256,
        )

        derived_text = self._splice_process(
            source_text,
            new_process,
            process_was_present=process_was_present,
            source_sha256=source_sha256,
        )
        derived_bytes = derived_text.encode("utf-8")
        derived_sha256 = hashlib.sha256(derived_bytes).hexdigest()

        try:
            derived_document = yaml.safe_load(derived_text)
        except yaml.YAMLError as exc:
            raise PlanCompilationBlocked(
                CompilationBlockCode.DERIVED_YAML_BLOCKED,
                "the minimal text patch did not produce valid YAML",
                details=(str(exc),),
                source_sha256=source_sha256,
            ) from exc
        self._verify_only_process_changed(
            document,
            derived_document,
            new_process,
            source_sha256,
        )

        output_guard = self._inspect_derived(derived_bytes, guard_inventory)
        if output_guard.blocked:
            raise PlanCompilationBlocked(
                CompilationBlockCode.DERIVED_YAML_BLOCKED,
                "the derived YAML did not pass the same static capability policy",
                details=tuple(
                    f"{issue.code}@{issue.path}" for issue in output_guard.blocking_issues
                ),
                source_sha256=source_sha256,
            )

        assignments_provenance = tuple(
            AssignmentProvenance(
                group_id=group_id,
                precision=normalized_assignments[group_id],
                module_names=tuple(
                    module_name for module_name in inventory if module_name in groups[group_id]
                ),
            )
            for group_id in sorted(groups)
        )
        used_quant_precisions = tuple(
            precision
            for precision in self._QUANT_PRECISIONS
            if precision in normalized_assignments.values()
        )
        template_provenance = tuple(
            TemplateProvenance(
                precision=precision,
                qconfig_sha256=_content_sha256(templates[precision]),
                qconfig=copy.deepcopy(templates[precision]),
            )
            for precision in used_quant_precisions
        )
        bf16_modules = tuple(
            module_name
            for module_name in inventory
            if self._module_precision(module_name, groups, normalized_assignments)
            is PrecisionFormat.BF16
        )
        quantized_modules = tuple(
            module_name for module_name in inventory if module_name not in bf16_modules
        )
        before_plain = _plain_data(old_process, "$.spec.process")
        after_plain = _plain_data(new_process, "$.spec.process")
        path_diff = PathDiff(
            operation="replace" if process_was_present else "add",
            path="$.spec.process",
            before=before_plain if process_was_present else None,
            after=after_plain,
            before_sha256=(_content_sha256(before_plain) if process_was_present else None),
            after_sha256=_content_sha256(after_plain),
        )
        sidecar = CompilationSidecar(
            schema_version="quantplan.modelslim-plan-sidecar/v1",
            compiler=self.COMPILER_ID,
            source_path=str(source_path),
            source_sha256=source_sha256,
            derived_sha256=derived_sha256,
            capability_manifest_id=self.manifest.fingerprint,
            execution_readiness="not_assessed",
            quant_target_inventory_sha256=_content_sha256(list(inventory)),
            quant_target_count=len(inventory),
            model_module_inventory_sha256=_content_sha256(list(guard_inventory)),
            model_module_count=len(guard_inventory),
            assignments=assignments_provenance,
            templates=template_provenance,
            bf16_modules=bf16_modules,
            quantized_modules=quantized_modules,
            diff=(path_diff,),
        )
        return PlanCompilationResult(
            derived_yaml=derived_text,
            sidecar=sidecar,
            output_guard=replace(output_guard, source_path="<derived-yaml>"),
        )

    def _normalize_inventory(
        self,
        module_names: Iterable[str],
        source_sha256: str,
    ) -> tuple[str, ...]:
        if isinstance(module_names, (str, bytes)):
            raise PlanCompilationBlocked(
                CompilationBlockCode.PLAN_INVALID,
                "the quantization-target inventory must contain module names",
                source_sha256=source_sha256,
            )
        try:
            inventory = tuple(module_names)
        except TypeError as exc:
            raise PlanCompilationBlocked(
                CompilationBlockCode.PLAN_INVALID,
                "the quantization-target inventory must contain module names",
                source_sha256=source_sha256,
            ) from exc
        if not inventory:
            raise PlanCompilationBlocked(
                CompilationBlockCode.PLAN_INVALID,
                "the module inventory must not be empty",
                source_sha256=source_sha256,
            )
        if any(not isinstance(name, str) or not name for name in inventory):
            raise PlanCompilationBlocked(
                CompilationBlockCode.PLAN_INVALID,
                "module inventory entries must be non-empty strings",
                source_sha256=source_sha256,
            )
        if len(set(inventory)) != len(inventory):
            raise PlanCompilationBlocked(
                CompilationBlockCode.PLAN_INVALID,
                "the module inventory contains duplicate names",
                source_sha256=source_sha256,
            )
        unsafe = tuple(
            name for name in inventory if any(char in name for char in self._PATTERN_META)
        )
        if unsafe:
            raise PlanCompilationBlocked(
                CompilationBlockCode.UNSAFE_PATTERN_NAME,
                "module names containing pattern metacharacters cannot be emitted exactly",
                details=unsafe,
                source_sha256=source_sha256,
            )
        return inventory

    def _normalize_model_inventory(
        self,
        model_module_names: Iterable[str] | None,
        *,
        target_inventory: tuple[str, ...],
        source_sha256: str,
    ) -> tuple[str, ...]:
        if model_module_names is None:
            return target_inventory
        if isinstance(model_module_names, (str, bytes)):
            raise PlanCompilationBlocked(
                CompilationBlockCode.PLAN_INVALID,
                "model_module_names must contain the complete model module inventory",
                source_sha256=source_sha256,
            )
        try:
            inventory = tuple(model_module_names)
        except TypeError as exc:
            raise PlanCompilationBlocked(
                CompilationBlockCode.PLAN_INVALID,
                "model_module_names must contain the complete model module inventory",
                source_sha256=source_sha256,
            ) from exc
        if not inventory or any(not isinstance(name, str) for name in inventory):
            raise PlanCompilationBlocked(
                CompilationBlockCode.PLAN_INVALID,
                "model_module_names must contain strings (the empty root name is allowed)",
                source_sha256=source_sha256,
            )
        if len(set(inventory)) != len(inventory):
            raise PlanCompilationBlocked(
                CompilationBlockCode.PLAN_INVALID,
                "model_module_names contains duplicate names",
                source_sha256=source_sha256,
            )
        # ``torch.nn.Module.named_modules()`` includes the root as ``""``.
        # It is not addressable by ModelSlim include/exclude patterns, so omit it
        # from the guard universe while accepting the direct API output.
        inventory = tuple(name for name in inventory if name)
        missing_targets = set(target_inventory) - set(inventory)
        if missing_targets:
            raise PlanCompilationBlocked(
                CompilationBlockCode.PLAN_INVALID,
                "the complete model inventory must contain every quantization target",
                details=(f"missing_targets={sorted(missing_targets)}",),
                source_sha256=source_sha256,
            )
        return inventory

    def _normalize_groups(
        self,
        group_modules: Mapping[str, Iterable[str]],
        inventory: tuple[str, ...],
        source_sha256: str,
    ) -> dict[str, frozenset[str]]:
        if not isinstance(group_modules, Mapping) or not group_modules:
            raise PlanCompilationBlocked(
                CompilationBlockCode.PLAN_INVALID,
                "group_modules must be a non-empty mapping",
                source_sha256=source_sha256,
            )
        normalized: dict[str, frozenset[str]] = {}
        owner: dict[str, str] = {}
        for group_id, names in group_modules.items():
            if not isinstance(group_id, str) or not group_id:
                raise PlanCompilationBlocked(
                    CompilationBlockCode.PLAN_INVALID,
                    "decision group IDs must be non-empty strings",
                    source_sha256=source_sha256,
                )
            if isinstance(names, (str, bytes)) or not isinstance(names, Iterable):
                raise PlanCompilationBlocked(
                    CompilationBlockCode.PLAN_INVALID,
                    f"decision group {group_id!r} must contain module names",
                    source_sha256=source_sha256,
                )
            try:
                group_names = tuple(names)
            except TypeError as exc:
                raise PlanCompilationBlocked(
                    CompilationBlockCode.PLAN_INVALID,
                    f"decision group {group_id!r} must contain module names",
                    source_sha256=source_sha256,
                ) from exc
            if not group_names or any(
                not isinstance(name, str) or not name for name in group_names
            ):
                raise PlanCompilationBlocked(
                    CompilationBlockCode.PLAN_INVALID,
                    f"decision group {group_id!r} contains invalid module names",
                    source_sha256=source_sha256,
                )
            if len(set(group_names)) != len(group_names):
                raise PlanCompilationBlocked(
                    CompilationBlockCode.PLAN_INVALID,
                    f"decision group {group_id!r} contains duplicate module names",
                    source_sha256=source_sha256,
                )
            normalized[group_id] = frozenset(group_names)
            for name in group_names:
                if name in owner:
                    raise PlanCompilationBlocked(
                        CompilationBlockCode.PLAN_INVALID,
                        f"module {name!r} belongs to more than one decision group",
                        details=(owner[name], group_id),
                        source_sha256=source_sha256,
                    )
                owner[name] = group_id

        inventory_set = set(inventory)
        group_set = set(owner)
        if inventory_set != group_set:
            raise PlanCompilationBlocked(
                CompilationBlockCode.PLAN_INVALID,
                "decision groups must partition the complete module inventory",
                details=(
                    f"missing={sorted(inventory_set - group_set)}",
                    f"extra={sorted(group_set - inventory_set)}",
                ),
                source_sha256=source_sha256,
            )
        return normalized

    def _normalize_assignments(
        self,
        assignments: Mapping[str, PrecisionFormat | str],
        groups: Mapping[str, frozenset[str]],
        source_sha256: str,
    ) -> dict[str, PrecisionFormat]:
        if not isinstance(assignments, Mapping) or set(assignments) != set(groups):
            actual = set(assignments) if isinstance(assignments, Mapping) else set()
            raise PlanCompilationBlocked(
                CompilationBlockCode.PLAN_INVALID,
                "assignments must contain exactly one entry for every decision group",
                details=(
                    f"missing={sorted(set(groups) - actual)}",
                    f"extra={sorted(actual - set(groups))}",
                ),
                source_sha256=source_sha256,
            )
        normalized: dict[str, PrecisionFormat] = {}
        for group_id, raw_precision in assignments.items():
            try:
                precision = (
                    raw_precision
                    if isinstance(raw_precision, PrecisionFormat)
                    else PrecisionFormat(str(raw_precision).lower())
                )
            except ValueError as exc:
                raise PlanCompilationBlocked(
                    CompilationBlockCode.PLAN_INVALID,
                    f"assignment for {group_id!r} has an unknown precision",
                    details=(str(raw_precision),),
                    source_sha256=source_sha256,
                ) from exc
            if precision not in self._ALLOWED_PRECISIONS:
                raise PlanCompilationBlocked(
                    CompilationBlockCode.PLAN_INVALID,
                    f"precision {precision.value!r} is not safely expressible by this compiler",
                    source_sha256=source_sha256,
                )
            normalized[group_id] = precision
        return normalized

    def _normalize_module_patterns(
        self,
        module_patterns: Mapping[str, str] | None,
        inventory: tuple[str, ...],
        source_sha256: str,
    ) -> dict[str, str]:
        if module_patterns is None:
            return {name: name for name in inventory}
        if not isinstance(module_patterns, Mapping) or set(module_patterns) != set(
            inventory
        ):
            actual = set(module_patterns) if isinstance(module_patterns, Mapping) else set()
            raise PlanCompilationBlocked(
                CompilationBlockCode.PLAN_INVALID,
                "module_patterns must contain exactly one backend pattern per target",
                details=(
                    f"missing={sorted(set(inventory) - actual)}",
                    f"extra={sorted(actual - set(inventory))}",
                ),
                source_sha256=source_sha256,
            )
        normalized: dict[str, str] = {}
        owners: dict[str, str] = {}
        for module_name in inventory:
            pattern = module_patterns[module_name]
            if not isinstance(pattern, str) or not pattern:
                raise PlanCompilationBlocked(
                    CompilationBlockCode.PLAN_INVALID,
                    f"backend pattern for {module_name!r} must be a non-empty string",
                    source_sha256=source_sha256,
                )
            previous = owners.setdefault(pattern, module_name)
            if previous != module_name:
                raise PlanCompilationBlocked(
                    CompilationBlockCode.PLAN_INVALID,
                    f"backend pattern {pattern!r} is shared by multiple targets",
                    details=(previous, module_name),
                    source_sha256=source_sha256,
                )
            normalized[module_name] = pattern
        return normalized

    def _normalize_templates(
        self,
        precision_templates: Mapping[PrecisionFormat | str, Mapping[str, Any]],
        assignments: Mapping[str, PrecisionFormat],
        source_sha256: str,
    ) -> dict[PrecisionFormat, dict[str, Any]]:
        if not isinstance(precision_templates, Mapping):
            raise PlanCompilationBlocked(
                CompilationBlockCode.TEMPLATE_INVALID,
                "precision_templates must be a mapping",
                source_sha256=source_sha256,
            )
        normalized: dict[PrecisionFormat, dict[str, Any]] = {}
        for raw_precision, raw_template in precision_templates.items():
            try:
                precision = (
                    raw_precision
                    if isinstance(raw_precision, PrecisionFormat)
                    else PrecisionFormat(str(raw_precision).lower())
                )
            except ValueError as exc:
                raise PlanCompilationBlocked(
                    CompilationBlockCode.TEMPLATE_INVALID,
                    f"template key {raw_precision!r} is not a known precision",
                    source_sha256=source_sha256,
                ) from exc
            if precision in normalized:
                raise PlanCompilationBlocked(
                    CompilationBlockCode.TEMPLATE_INVALID,
                    f"precision {precision.value!r} has duplicate templates",
                    source_sha256=source_sha256,
                )
            if not isinstance(raw_template, Mapping):
                raise PlanCompilationBlocked(
                    CompilationBlockCode.TEMPLATE_INVALID,
                    f"template for {precision.value!r} must be a qconfig mapping",
                    source_sha256=source_sha256,
                )
            try:
                normalized[precision] = _plain_data(
                    raw_template,
                    f"precision_templates.{precision.value}",
                )
            except (TypeError, ValueError) as exc:
                raise PlanCompilationBlocked(
                    CompilationBlockCode.TEMPLATE_INVALID,
                    f"template for {precision.value!r} is not plain YAML data",
                    details=(str(exc),),
                    source_sha256=source_sha256,
                ) from exc
            self._validate_template_signature(
                precision,
                normalized[precision],
                source_sha256,
            )

        used = set(assignments.values()) & set(self._QUANT_PRECISIONS)
        missing = used - set(normalized)
        if missing:
            raise PlanCompilationBlocked(
                CompilationBlockCode.TEMPLATE_MISSING,
                "a qconfig template is required for every emitted quantized precision",
                details=tuple(sorted(item.value for item in missing)),
                source_sha256=source_sha256,
            )
        return normalized

    @staticmethod
    def _validate_template_signature(
        precision: PrecisionFormat,
        template: Mapping[str, Any],
        source_sha256: str,
    ) -> None:
        if precision not in {PrecisionFormat.W4A8, PrecisionFormat.W8A8}:
            return
        expected_weight_dtype = (
            "int4" if precision is PrecisionFormat.W4A8 else "int8"
        )
        expected = {"act": "int8", "weight": expected_weight_dtype}
        mismatches: list[str] = []
        for branch_name, dtype in expected.items():
            branch = template.get(branch_name)
            if not isinstance(branch, Mapping):
                mismatches.append(f"{branch_name}=missing_or_not_mapping")
                continue
            if branch.get("dtype") != dtype:
                mismatches.append(
                    f"{branch_name}.dtype={branch.get('dtype')!r}, expected={dtype!r}"
                )
        if mismatches:
            raise PlanCompilationBlocked(
                CompilationBlockCode.TEMPLATE_INVALID,
                f"template does not implement the {precision.value} candidate signature",
                details=tuple(mismatches),
                source_sha256=source_sha256,
            )

    def _build_linear_configs(
        self,
        inventory: tuple[str, ...],
        groups: Mapping[str, frozenset[str]],
        assignments: Mapping[str, PrecisionFormat],
        templates: Mapping[PrecisionFormat, Mapping[str, Any]],
        module_patterns: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        configs: list[dict[str, Any]] = []
        for precision in self._QUANT_PRECISIONS:
            if precision not in assignments.values():
                continue
            config: dict[str, Any] = {
                "type": "linear_quant",
                "qconfig": copy.deepcopy(templates[precision]),
                "include": [
                    module_patterns[name]
                    for name in inventory
                    if self._module_precision(name, groups, assignments) is precision
                ],
            }
            configs.append(config)
        return configs

    @staticmethod
    def _module_precision(
        module_name: str,
        groups: Mapping[str, frozenset[str]],
        assignments: Mapping[str, PrecisionFormat],
    ) -> PrecisionFormat:
        for group_id, names in groups.items():
            if module_name in names:
                return assignments[group_id]
        raise AssertionError(f"unowned inventory module: {module_name}")

    def _install_linear_configs(
        self,
        process: list[Any],
        generated_configs: list[dict[str, Any]],
        source_sha256: str,
    ) -> None:
        slots: list[_LinearSlot] = []
        self._find_linear_slots(process, "$.spec.process", slots)
        if len(slots) > 1:
            raise PlanCompilationBlocked(
                CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                "linear_quant processors occur in multiple containers",
                details=tuple(slot.path for slot in slots),
                source_sha256=source_sha256,
            )
        if not slots:
            if process and generated_configs:
                raise PlanCompilationBlocked(
                    CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                    "a non-empty process list has no unambiguous linear_quant insertion slot",
                    source_sha256=source_sha256,
                )
            if generated_configs:
                process.append({"type": "group", "configs": generated_configs})
            return

        slot = slots[0]
        first, last = slot.indices[0], slot.indices[-1]
        if slot.indices != tuple(range(first, last + 1)):
            raise PlanCompilationBlocked(
                CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                "linear_quant processors are separated by order-sensitive processors",
                details=(slot.path,),
                source_sha256=source_sha256,
            )
        replacement_configs: list[Any]
        if slot.parent is process:
            replacement_configs = (
                [{"type": "group", "configs": generated_configs}] if generated_configs else []
            )
        else:
            replacement_configs = generated_configs
        slot.parent[first : last + 1] = replacement_configs

    def _find_linear_slots(
        self,
        processes: list[Any],
        path: str,
        slots: list[_LinearSlot],
    ) -> None:
        direct_indices = tuple(
            index
            for index, process in enumerate(processes)
            if isinstance(process, Mapping) and process.get("type") == "linear_quant"
        )
        if direct_indices:
            slots.append(_LinearSlot(processes, direct_indices, path))
        for index, process in enumerate(processes):
            if not isinstance(process, Mapping) or process.get("type") != "group":
                continue
            configs = process.get("configs")
            if isinstance(configs, list):
                self._find_linear_slots(
                    configs,
                    f"{path}[{index}].configs",
                    slots,
                )

    def _splice_process(
        self,
        source_text: str,
        new_process: list[Any],
        *,
        process_was_present: bool,
        source_sha256: str,
    ) -> str:
        try:
            root = yaml.compose(source_text, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            raise PlanCompilationBlocked(
                CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                "unable to locate $.spec.process in the original text",
                details=(str(exc),),
                source_sha256=source_sha256,
            ) from exc
        if not isinstance(root, MappingNode):
            raise PlanCompilationBlocked(
                CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                "the YAML root is not a mapping node",
                source_sha256=source_sha256,
            )
        spec_entry = _mapping_entry(root, "spec")
        if spec_entry is None:
            raise PlanCompilationBlocked(
                CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                "$.spec cannot be located in the original text",
                source_sha256=source_sha256,
            )
        spec_key, spec_node = spec_entry
        if not isinstance(spec_node, MappingNode):
            raise PlanCompilationBlocked(
                CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                "$.spec must be a concrete mapping, not an alias or scalar",
                source_sha256=source_sha256,
            )
        if spec_node.start_mark.index < spec_key.end_mark.index:
            raise PlanCompilationBlocked(
                CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                "an aliased $.spec mapping cannot be patched safely",
                source_sha256=source_sha256,
            )

        process_entry = _mapping_entry(spec_node, "process")
        if process_was_present:
            if process_entry is None:
                raise PlanCompilationBlocked(
                    CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                    "parsed $.spec.process cannot be located in the source text",
                    source_sha256=source_sha256,
                )
            process_key, process_node = process_entry
            if not isinstance(process_node, SequenceNode):
                raise PlanCompilationBlocked(
                    CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                    "$.spec.process is not a concrete sequence node",
                    source_sha256=source_sha256,
                )
            if process_node.start_mark.index < process_key.end_mark.index:
                raise PlanCompilationBlocked(
                    CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                    "an aliased $.spec.process sequence cannot be patched safely",
                    source_sha256=source_sha256,
                )
            newline = "\r\n" if "\r\n" in source_text else "\n"
            block = _dump_block(new_process)
            if process_node.start_mark.line == process_key.start_mark.line:
                if new_process:
                    indent = process_key.start_mark.column + 2
                    replacement_text = newline + _indent_all(block, indent).replace("\n", newline)
                else:
                    replacement_text = "[]"
            else:
                replacement_text = _indent_after_first(
                    block,
                    process_node.start_mark.column,
                ).replace("\n", newline)
                suffix = source_text[process_node.end_mark.index :]
                # For block collections PyYAML's end mark consumes the indentation
                # before the next key, but not the key itself. Restore that boundary.
                if suffix and not suffix.startswith(("\r", "\n")):
                    replacement_text += newline + (" " * process_node.end_mark.column)
            return (
                source_text[: process_node.start_mark.index]
                + replacement_text
                + source_text[process_node.end_mark.index :]
            )

        if process_entry is not None:
            raise PlanCompilationBlocked(
                CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                "$.spec.process presence differs between parsed and composed YAML",
                source_sha256=source_sha256,
            )
        newline = "\r\n" if "\r\n" in source_text else "\n"
        child_indent = spec_key.start_mark.column + 2
        entry_text = _render_process_entry(new_process, child_indent, newline)
        if spec_node.flow_style:
            if spec_node.value:
                raise PlanCompilationBlocked(
                    CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                    "a non-empty flow-style $.spec mapping cannot be extended safely",
                    source_sha256=source_sha256,
                )
            replacement_text = newline + entry_text.rstrip("\r\n")
            return (
                source_text[: spec_node.start_mark.index]
                + replacement_text
                + source_text[spec_node.end_mark.index :]
            )

        root_entries = root.value
        spec_position = next(
            index for index, (key_node, _) in enumerate(root_entries) if key_node is spec_key
        )
        if spec_position + 1 < len(root_entries):
            next_key = root_entries[spec_position + 1][0]
            insertion_index = _line_start(source_text, next_key.start_mark.index)
        else:
            insertion_index = len(source_text)
        prefix = source_text[:insertion_index]
        separator = "" if not prefix or prefix.endswith(("\n", "\r")) else newline
        return prefix + separator + entry_text + source_text[insertion_index:]

    def _verify_only_process_changed(
        self,
        original: Any,
        derived: Any,
        expected_process: list[Any],
        source_sha256: str,
    ) -> None:
        if not isinstance(derived, dict) or not isinstance(derived.get("spec"), dict):
            raise PlanCompilationBlocked(
                CompilationBlockCode.DERIVED_YAML_MISMATCH,
                "derived YAML lost the $.spec mapping",
                source_sha256=source_sha256,
            )
        if derived["spec"].get("process") != expected_process:
            raise PlanCompilationBlocked(
                CompilationBlockCode.DERIVED_YAML_MISMATCH,
                "derived $.spec.process differs from the compiled process tree",
                source_sha256=source_sha256,
            )
        original_rest = copy.deepcopy(original)
        derived_rest = copy.deepcopy(derived)
        original_rest["spec"].pop("process", None)
        derived_rest["spec"].pop("process", None)
        try:
            same = original_rest == derived_rest
        except RecursionError as exc:
            raise PlanCompilationBlocked(
                CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE,
                "recursive aliases outside $.spec.process cannot be compared safely",
                source_sha256=source_sha256,
            ) from exc
        if not same:
            raise PlanCompilationBlocked(
                CompilationBlockCode.DERIVED_YAML_MISMATCH,
                "the derived YAML changed semantics outside $.spec.process",
                source_sha256=source_sha256,
            )

    def _inspect_derived(
        self,
        derived_bytes: bytes,
        inventory: tuple[str, ...],
    ) -> YamlGuardResult:
        with tempfile.TemporaryDirectory(prefix="quantplan-modelslim-") as directory:
            path = Path(directory) / "derived.yaml"
            path.write_bytes(derived_bytes)
            return inspect_modelslim_yaml(
                path,
                manifest=self.manifest,
                module_names=inventory,
            )


def compile_modelslim_plan(
    source: str | Path,
    group_modules: Mapping[str, Iterable[str]],
    assignments: Mapping[str, PrecisionFormat | str],
    precision_templates: Mapping[PrecisionFormat | str, Mapping[str, Any]],
    module_names: Iterable[str],
    *,
    model_module_names: Iterable[str] | None = None,
    module_patterns: Mapping[str, str] | None = None,
    manifest: CapabilityManifest | None = None,
) -> PlanCompilationResult:
    return ModelSlimPlanCompiler(manifest).compile(
        source,
        group_modules,
        assignments,
        precision_templates,
        module_names,
        model_module_names=model_module_names,
        module_patterns=module_patterns,
    )


# Explicit YAML-oriented alias for integrations that have several plan compilers.
compile_modelslim_yaml_plan = compile_modelslim_plan


def _mapping_entry(
    mapping: MappingNode,
    field_name: str,
) -> tuple[ScalarNode, Any] | None:
    for key_node, value_node in mapping.value:
        if isinstance(key_node, ScalarNode) and key_node.value == field_name:
            return key_node, value_node
    return None


def _dump_block(value: Any) -> str:
    return yaml.dump(
        value,
        Dumper=_IndentedSafeDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=4096,
    ).rstrip("\n")


def _indent_all(text: str, indent: int) -> str:
    prefix = " " * indent
    return "\n".join(prefix + line for line in text.splitlines())


def _indent_after_first(text: str, indent: int) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    prefix = " " * indent
    return lines[0] + "".join("\n" + prefix + line for line in lines[1:])


def _render_process_entry(value: list[Any], indent: int, newline: str) -> str:
    prefix = " " * indent
    if not value:
        return f"{prefix}process: []{newline}"
    dumped = _dump_block(value)
    indented = _indent_all(dumped, indent + 2).replace("\n", newline)
    return f"{prefix}process:{newline}{indented}{newline}"


def _line_start(text: str, index: int) -> int:
    newline_index = text.rfind("\n", 0, index)
    return 0 if newline_index < 0 else newline_index + 1


def _plain_data(value: Any, path: str, active: set[int] | None = None) -> Any:
    active = set() if active is None else active
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} contains a recursive mapping")
        active.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} contains a non-string mapping key")
                result[key] = _plain_data(item, f"{path}.{key}", active)
            return result
        finally:
            active.remove(identity)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} contains a recursive sequence")
        active.add(identity)
        try:
            return [
                _plain_data(item, f"{path}[{index}]", active) for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


def _content_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
