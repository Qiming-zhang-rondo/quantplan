from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from quantplan.domain.models import PrecisionFormat

SCHEMA_VERSION = "quantplan_v1"
_PLACEHOLDER_MARKER = "REPLACE_WITH_"
_CUSTOM_LOADER_PATTERN = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$"
)
_QUANTIZED_PRECISIONS = frozenset(
    {PrecisionFormat.W8A8, PrecisionFormat.W4A8, PrecisionFormat.W4A4}
)


class RunConfigError(ValueError):
    """A structurally invalid or non-runnable ``quantplan_v1`` configuration."""


@dataclass(frozen=True)
class RuntimeConfig:
    loader: str = "huggingface"
    revision: str | None = None
    torch_dtype: str = "bf16"
    analysis_only: bool = True
    trust_remote_code: bool = False
    strict_target_coverage: bool = True
    allow_image_text_fallback: bool = False
    hardware_profile: str = "unknown"
    backend_capability_manifest: Path | None = None
    inference_framework: str = "auto"

    def __post_init__(self) -> None:
        if not _valid_loader(self.loader):
            raise RunConfigError(
                "runtime.loader must be 'huggingface' or a Python 'module:function' reference"
            )
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise RunConfigError("$.runtime.revision must be null or a non-empty string")
        if self.torch_dtype not in {
            "auto",
            "bf16",
            "bfloat16",
        }:
            raise RunConfigError(
                "$.runtime.torch_dtype must be bf16/bfloat16 or auto; "
                "the P0 protection anchor cannot be labeled BF16 when loaded as FP16/FP32"
            )
        _require_bool(self.analysis_only, "$.runtime.analysis_only")
        _require_bool(self.trust_remote_code, "$.runtime.trust_remote_code")
        _require_bool(
            self.allow_image_text_fallback,
            "$.runtime.allow_image_text_fallback",
        )
        _require_bool(
            self.strict_target_coverage, "$.runtime.strict_target_coverage"
        )
        if not isinstance(self.hardware_profile, str) or self.hardware_profile.strip().lower() not in {
            "unknown",
            "a3",
            "a5",
        }:
            raise RunConfigError(
                "$.runtime.hardware_profile must be one of unknown, a3 or a5"
            )
        object.__setattr__(self, "hardware_profile", self.hardware_profile.strip().lower())
        if self.backend_capability_manifest is not None and not isinstance(
            self.backend_capability_manifest, Path
        ):
            raise RunConfigError("$.runtime.backend_capability_manifest must be a path or null")
        if not isinstance(self.inference_framework, str) or not self.inference_framework.strip():
            raise RunConfigError("$.runtime.inference_framework must be a non-empty string")
        object.__setattr__(self, "inference_framework", self.inference_framework.strip().lower())


@dataclass(frozen=True)
class ModuleSelectionConfig:
    block_patterns: tuple[str, ...] = ()
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = (
        "*vision*",
        "*visual*",
        "*mtp*",
        "*.mlp.gate",
        "*.mlp.gate.weight",
        "*router*",
    )

    def __post_init__(self) -> None:
        for name in ("block_patterns", "include", "exclude"):
            value = getattr(self, name)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise RunConfigError(f"$.modules.{name} must be a sequence")
            normalized = tuple(value)
            object.__setattr__(self, name, normalized)
            _validate_patterns(normalized, f"$.modules.{name}")
        overlap = sorted(set(self.include) & set(self.exclude))
        if overlap:
            raise RunConfigError(
                "$.modules include/exclude contain the same patterns: " + ", ".join(overlap)
            )


@dataclass(frozen=True)
class CandidateConfig:
    precisions: tuple[PrecisionFormat, ...] = (
        PrecisionFormat.W4A8,
        PrecisionFormat.W8A8,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.precisions, Sequence) or isinstance(
            self.precisions, (str, bytes)
        ):
            raise RunConfigError("$.candidates.precisions must be a sequence")
        object.__setattr__(self, "precisions", tuple(self.precisions))
        if not self.precisions:
            raise RunConfigError("$.candidates.precisions must not be empty")
        if any(not isinstance(item, PrecisionFormat) for item in self.precisions):
            raise RunConfigError(
                "$.candidates.precisions must contain PrecisionFormat values"
            )
        if len(set(self.precisions)) != len(self.precisions):
            raise RunConfigError("$.candidates.precisions must not contain duplicates")
        unsupported = set(self.precisions) - _QUANTIZED_PRECISIONS
        if unsupported:
            values = ", ".join(sorted(item.value for item in unsupported))
            raise RunConfigError(
                f"$.candidates.precisions contains non-candidate precision(s): {values}"
            )


@dataclass(frozen=True)
class AnalysisConfig:
    max_samples: int = 16
    max_length: int = 512
    batch_size: int = 1
    max_groups: int = 8
    compute_kl: bool = False
    text_field: str = "text"
    id_field: str = "id"
    seed: int = 20260817

    def __post_init__(self) -> None:
        _require_positive_int(self.max_samples, "$.analysis.max_samples")
        _require_positive_int(self.max_length, "$.analysis.max_length")
        if self.max_length > 4096:
            raise RunConfigError(
                "$.analysis.max_length exceeds the quantplan_v1 evidence boundary of 4096"
            )
        _require_positive_int(self.batch_size, "$.analysis.batch_size")
        _require_positive_int(self.max_groups, "$.analysis.max_groups")
        _require_bool(self.compute_kl, "$.analysis.compute_kl")
        for name in ("text_field", "id_field"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RunConfigError(f"$.analysis.{name} must be a non-empty string")
        if self.text_field == self.id_field:
            raise RunConfigError("$.analysis.text_field and id_field must be different")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise RunConfigError("$.analysis.seed must be a non-negative integer")


@dataclass(frozen=True)
class SearchConfig:
    base_precision: PrecisionFormat = PrecisionFormat.W4A8
    high_precision_fraction: float = 0.25
    max_candidates: int = 8
    max_delta_nll: float = 0.02
    costs: Mapping[PrecisionFormat, float] = field(
        default_factory=lambda: {
            PrecisionFormat.W4A4: 0.75,
            PrecisionFormat.W4A8: 1.0,
            PrecisionFormat.W8A8: 1.5,
            PrecisionFormat.BF16: 2.5,
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.base_precision, PrecisionFormat):
            raise RunConfigError("$.search.base_precision must be a PrecisionFormat")
        if self.base_precision not in _QUANTIZED_PRECISIONS:
            raise RunConfigError("$.search.base_precision must be a quantized candidate")
        fraction = self.high_precision_fraction
        if (
            not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not isfinite(float(fraction))
            or not 0 <= float(fraction) <= 1
        ):
            raise RunConfigError(
                "$.search.high_precision_fraction must be finite and in [0, 1]"
            )
        object.__setattr__(self, "high_precision_fraction", float(fraction))
        _require_positive_int(self.max_candidates, "$.search.max_candidates")
        if (
            not isinstance(self.max_delta_nll, (int, float))
            or isinstance(self.max_delta_nll, bool)
            or not isfinite(float(self.max_delta_nll))
            or float(self.max_delta_nll) < 0
        ):
            raise RunConfigError("$.search.max_delta_nll must be finite and non-negative")
        object.__setattr__(self, "max_delta_nll", float(self.max_delta_nll))

        if not isinstance(self.costs, Mapping):
            raise RunConfigError("$.search.costs must be a mapping")
        normalized: dict[PrecisionFormat, float] = {}
        for precision, cost in self.costs.items():
            if not isinstance(precision, PrecisionFormat):
                raise RunConfigError("$.search.costs keys must be PrecisionFormat values")
            if (
                not isinstance(cost, (int, float))
                or isinstance(cost, bool)
                or not isfinite(float(cost))
                or float(cost) <= 0
            ):
                raise RunConfigError(
                    f"$.search.costs.{precision.value} must be finite and positive"
                )
            normalized[precision] = float(cost)
        object.__setattr__(self, "costs", MappingProxyType(normalized))


@dataclass(frozen=True)
class LatencyConfig:
    """Fast latency-estimation settings.

    ``off`` preserves the historical normalized parameter-count cost.  The
    static and calibrated modes are proposal signals; only the backend engine
    can provide a final deployment measurement.
    """

    mode: str = "off"
    deployment_mode: str = "decode_bs1"
    memory_bandwidth_gbps: float | None = None
    compute_tflops: float | None = None
    memory_efficiency: float = 0.70
    compute_efficiency: float = 0.50
    launch_overhead_us: float = 2.0
    qdq_overhead_us: float = 0.0
    layout_overhead_us: float = 0.0
    fallback_overhead_us: float = 0.0
    uncertainty: float = 0.25
    calibration_path: Path | None = None

    def __post_init__(self) -> None:
        normalized = self.mode.strip().lower()
        if normalized not in {"off", "static", "calibrated"}:
            raise RunConfigError("$.latency.mode must be off, static or calibrated")
        object.__setattr__(self, "mode", normalized)
        if not isinstance(self.deployment_mode, str) or not self.deployment_mode.strip():
            raise RunConfigError("$.latency.deployment_mode must be a non-empty string")
        object.__setattr__(self, "deployment_mode", self.deployment_mode.strip())
        for name in ("memory_bandwidth_gbps", "compute_tflops"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isfinite(float(value))
                or float(value) <= 0
            ):
                raise RunConfigError(f"$.latency.{name} must be null or positive")
            if value is not None:
                object.__setattr__(self, name, float(value))
        for name in ("memory_efficiency", "compute_efficiency"):
            value = getattr(self, name)
            if not isfinite(float(value)) or not 0 < float(value) <= 1:
                raise RunConfigError(f"$.latency.{name} must be in (0, 1]")
            object.__setattr__(self, name, float(value))
        for name in (
            "launch_overhead_us",
            "qdq_overhead_us",
            "layout_overhead_us",
            "fallback_overhead_us",
        ):
            value = getattr(self, name)
            if not isfinite(float(value)) or float(value) < 0:
                raise RunConfigError(f"$.latency.{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))
        if not isfinite(float(self.uncertainty)) or not 0 <= float(self.uncertainty) < 1:
            raise RunConfigError("$.latency.uncertainty must be in [0, 1)")
        object.__setattr__(self, "uncertainty", float(self.uncertainty))
        if self.calibration_path is not None and not isinstance(self.calibration_path, Path):
            raise RunConfigError("$.latency.calibration_path must be a path or null")
        if self.mode == "calibrated" and self.calibration_path is None:
            raise RunConfigError("$.latency.calibration_path is required in calibrated mode")


@dataclass(frozen=True)
class OutputConfig:
    model_inventory: Path = Path("model-inventory.json")
    risk_evidence: Path = Path("risk-evidence.jsonl")
    candidate_plan: Path = Path("candidate-plan.json")
    validation_report: Path = Path("validation-report.json")
    derived_yaml: Path = Path("derived-modelslim.yaml")
    provenance: Path = Path("provenance.json")
    diff: Path = Path("plan.diff")
    yaml_audit: Path = Path("yaml-audit.json")
    run_manifest: Path = Path("run-manifest.json")
    latency_report: Path = Path("latency-report.json")

    def __post_init__(self) -> None:
        paths = {
            "model_inventory": self.model_inventory,
            "risk_evidence": self.risk_evidence,
            "candidate_plan": self.candidate_plan,
            "validation_report": self.validation_report,
            "derived_yaml": self.derived_yaml,
            "provenance": self.provenance,
            "diff": self.diff,
            "yaml_audit": self.yaml_audit,
            "run_manifest": self.run_manifest,
            "latency_report": self.latency_report,
        }
        for name, path in paths.items():
            _validate_output_path(path, f"$.output.{name}")
        if len(set(paths.values())) != len(paths):
            raise RunConfigError("$.output paths must be unique")


@dataclass(frozen=True)
class RunConfig:
    """Strict, backend-neutral user configuration for one QuantPlan run.

    Loading checks schema and value semantics. ``validate_for_run`` is a
    separate boundary because checked-in templates intentionally contain
    placeholders and a configuration may be prepared before data is mounted.
    """

    model_path: Path
    data_path: Path
    original_yaml: Path
    output_dir: Path
    device: str
    schema_version: str = SCHEMA_VERSION
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    modules: ModuleSelectionConfig = field(default_factory=ModuleSelectionConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise RunConfigError(
                f"$.schema_version must be {SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        for name in ("model_path", "data_path", "original_yaml", "output_dir"):
            if not isinstance(getattr(self, name), Path):
                raise RunConfigError(f"$.{name} must be a path")
        if not isinstance(self.device, str) or not self.device.strip():
            raise RunConfigError("$.device must be a non-empty string")

        sections = {
            "runtime": (self.runtime, RuntimeConfig),
            "modules": (self.modules, ModuleSelectionConfig),
            "candidates": (self.candidates, CandidateConfig),
            "analysis": (self.analysis, AnalysisConfig),
            "search": (self.search, SearchConfig),
            "latency": (self.latency, LatencyConfig),
            "output": (self.output, OutputConfig),
        }
        for name, (value, expected_type) in sections.items():
            if not isinstance(value, expected_type):
                raise RunConfigError(f"$.{name} must be a {expected_type.__name__}")

        if self.search.costs:
            required = {PrecisionFormat.BF16, *self.candidates.precisions}
            missing = sorted(item.value for item in required - set(self.search.costs))
            if missing:
                raise RunConfigError(
                    "$.search.costs must define bf16 and every candidate when non-empty; "
                    f"missing={missing}"
                )
        if self.search.base_precision not in self.candidates.precisions:
            raise RunConfigError(
                "$.search.base_precision must be present in $.candidates.precisions"
            )

    @property
    def blocking_placeholders(self) -> tuple[str, ...]:
        fields = {
            "model_path": str(self.model_path),
            "data_path": str(self.data_path),
            "original_yaml": str(self.original_yaml),
            "output_dir": str(self.output_dir),
            "device": self.device,
            "backend_capability_manifest": (
                "" if self.runtime.backend_capability_manifest is None
                else str(self.runtime.backend_capability_manifest)
            ),
            "latency_calibration_path": (
                "" if self.latency.calibration_path is None else str(self.latency.calibration_path)
            ),
        }
        return tuple(
            name for name, value in fields.items() if _PLACEHOLDER_MARKER in value
        )

    @property
    def resolved_output_paths(self) -> Mapping[str, Path]:
        return MappingProxyType(
            {
                "model_inventory": self.output_dir / self.output.model_inventory,
                "risk_evidence": self.output_dir / self.output.risk_evidence,
                "candidate_plan": self.output_dir / self.output.candidate_plan,
                "validation_report": self.output_dir / self.output.validation_report,
                "derived_yaml": self.output_dir / self.output.derived_yaml,
                "provenance": self.output_dir / self.output.provenance,
                "diff": self.output_dir / self.output.diff,
                "yaml_audit": self.output_dir / self.output.yaml_audit,
                "run_manifest": self.output_dir / self.output.run_manifest,
                "latency_report": self.output_dir / self.output.latency_report,
            }
        )

    def validate_for_run(self, *, check_paths: bool = True) -> None:
        """Reject templates and, optionally, paths unavailable to the runner."""

        if self.blocking_placeholders:
            raise RunConfigError(
                "configuration still contains blocking placeholders: "
                + ", ".join(self.blocking_placeholders)
            )
        if not check_paths:
            return

        for name in ("model_path", "data_path"):
            path = getattr(self, name)
            if not path.exists():
                raise RunConfigError(f"$.{name} does not exist: {path}")
        if self.runtime.backend_capability_manifest is not None and not self.runtime.backend_capability_manifest.is_file():
            raise RunConfigError(
                "$.runtime.backend_capability_manifest is not a file: "
                f"{self.runtime.backend_capability_manifest}"
            )
        if self.latency.calibration_path is not None and not self.latency.calibration_path.is_file():
            raise RunConfigError(
                "$.latency.calibration_path is not a file: "
                f"{self.latency.calibration_path}"
            )
        if not self.original_yaml.is_file():
            raise RunConfigError(f"$.original_yaml is not a file: {self.original_yaml}")
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise RunConfigError(f"$.output_dir exists but is not a directory: {self.output_dir}")
        if self.model_path.is_dir():
            try:
                self.output_dir.relative_to(self.model_path)
            except ValueError:
                pass
            else:
                raise RunConfigError("$.output_dir must not be inside $.model_path")
        protected_inputs = {self.data_path.resolve(), self.original_yaml.resolve()}
        overlap = sorted(
            str(path)
            for path in self.resolved_output_paths.values()
            if path.resolve() in protected_inputs
        )
        if overlap:
            raise RunConfigError(
                "output artifacts must not overwrite input files: " + ", ".join(overlap)
            )


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys before merge expansion."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: MappingNode, deep: bool = False
) -> Mapping[Any, Any]:
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


def parse_run_config(
    document: Mapping[str, Any], *, base_dir: str | Path | None = None
) -> RunConfig:
    """Parse an already-decoded mapping using the strict ``quantplan_v1`` schema."""

    root = _mapping(document, "$")
    _reject_unknown(
        root,
        {
            "schema_version",
            "model_path",
            "data_path",
            "original_yaml",
            "output_dir",
            "device",
            "runtime",
            "modules",
            "candidates",
            "analysis",
            "search",
            "latency",
            "output",
        },
        "$",
    )
    _require_keys(
        root,
        {"schema_version", "model_path", "data_path", "original_yaml", "output_dir", "device"},
        "$",
    )
    root_dir = Path.cwd() if base_dir is None else Path(base_dir)

    runtime_doc = _optional_mapping(root, "runtime", "$.runtime")
    _reject_unknown(
        runtime_doc,
        {
            "loader",
            "revision",
            "torch_dtype",
            "analysis_only",
            "trust_remote_code",
            "allow_image_text_fallback",
            "strict_target_coverage",
            "hardware_profile",
            "backend_capability_manifest",
            "inference_framework",
        },
        "$.runtime",
    )
    runtime = RuntimeConfig(
        loader=_string(runtime_doc.get("loader", "huggingface"), "$.runtime.loader"),
        revision=_optional_string(runtime_doc.get("revision"), "$.runtime.revision"),
        torch_dtype=_string(runtime_doc.get("torch_dtype", "bf16"), "$.runtime.torch_dtype"),
        analysis_only=_bool(runtime_doc.get("analysis_only", True), "$.runtime.analysis_only"),
        trust_remote_code=_bool(
            runtime_doc.get("trust_remote_code", False), "$.runtime.trust_remote_code"
        ),
        allow_image_text_fallback=_bool(
            runtime_doc.get("allow_image_text_fallback", False),
            "$.runtime.allow_image_text_fallback",
        ),
        strict_target_coverage=_bool(
            runtime_doc.get("strict_target_coverage", True),
            "$.runtime.strict_target_coverage",
        ),
        hardware_profile=_string(
            runtime_doc.get("hardware_profile", "unknown"),
            "$.runtime.hardware_profile",
        ),
        backend_capability_manifest=(
            None
            if runtime_doc.get("backend_capability_manifest") is None
            else _input_path(
                runtime_doc["backend_capability_manifest"],
                "$.runtime.backend_capability_manifest",
                root_dir,
            )
        ),
        inference_framework=_string(
            runtime_doc.get("inference_framework", "auto"),
            "$.runtime.inference_framework",
        ),
    )

    modules_doc = _optional_mapping(root, "modules", "$.modules")
    _reject_unknown(modules_doc, {"block_patterns", "include", "exclude"}, "$.modules")
    modules = ModuleSelectionConfig(
        block_patterns=_strings(
            modules_doc.get("block_patterns", []), "$.modules.block_patterns"
        ),
        include=_strings(modules_doc.get("include", []), "$.modules.include"),
        exclude=_strings(
            modules_doc.get(
                "exclude",
                [
                    "*vision*",
                    "*visual*",
                    "*mtp*",
                    "*.mlp.gate",
                    "*.mlp.gate.weight",
                    "*router*",
                ],
            ),
            "$.modules.exclude",
        ),
    )

    candidates_doc = _optional_mapping(root, "candidates", "$.candidates")
    _reject_unknown(candidates_doc, {"precisions"}, "$.candidates")
    precisions = tuple(
        _precision(item, f"$.candidates.precisions[{index}]")
        for index, item in enumerate(
            _sequence(
                candidates_doc.get("precisions", ["w4a8", "w8a8"]),
                "$.candidates.precisions",
            )
        )
    )
    candidates = CandidateConfig(precisions=precisions)

    analysis_doc = _optional_mapping(root, "analysis", "$.analysis")
    _reject_unknown(
        analysis_doc,
        {
            "max_samples",
            "max_length",
            "batch_size",
            "max_groups",
            "compute_kl",
            "text_field",
            "id_field",
            "seed",
        },
        "$.analysis",
    )
    analysis = AnalysisConfig(
        max_samples=_int(analysis_doc.get("max_samples", 16), "$.analysis.max_samples"),
        max_length=_int(analysis_doc.get("max_length", 512), "$.analysis.max_length"),
        batch_size=_int(analysis_doc.get("batch_size", 1), "$.analysis.batch_size"),
        max_groups=_int(analysis_doc.get("max_groups", 8), "$.analysis.max_groups"),
        compute_kl=_bool(
            analysis_doc.get("compute_kl", False), "$.analysis.compute_kl"
        ),
        text_field=_string(analysis_doc.get("text_field", "text"), "$.analysis.text_field"),
        id_field=_string(analysis_doc.get("id_field", "id"), "$.analysis.id_field"),
        seed=_int(analysis_doc.get("seed", 20260817), "$.analysis.seed"),
    )

    search_doc = _optional_mapping(root, "search", "$.search")
    _reject_unknown(
        search_doc,
        {
            "base_precision",
            "high_precision_fraction",
            "max_candidates",
            "max_delta_nll",
            "costs",
        },
        "$.search",
    )
    search_defaults = SearchConfig()
    costs_doc = (
        _optional_mapping(search_doc, "costs", "$.search.costs")
        if "costs" in search_doc
        else {precision.value: cost for precision, cost in search_defaults.costs.items()}
    )
    costs: dict[PrecisionFormat, float] = {}
    for key, value in costs_doc.items():
        precision = _precision(key, f"$.search.costs.{key}", allow_bf16=True)
        costs[precision] = _number(value, f"$.search.costs.{key}")
    search = SearchConfig(
        base_precision=_precision(
            search_doc.get("base_precision", "w4a8"), "$.search.base_precision"
        ),
        high_precision_fraction=_number(
            search_doc.get("high_precision_fraction", 0.25),
            "$.search.high_precision_fraction",
        ),
        max_candidates=_int(
            search_doc.get("max_candidates", 8), "$.search.max_candidates"
        ),
        max_delta_nll=_number(
            search_doc.get("max_delta_nll", 0.02), "$.search.max_delta_nll"
        ),
        costs=costs,
    )

    latency_doc = _optional_mapping(root, "latency", "$.latency")
    _reject_unknown(
        latency_doc,
        {
            "mode",
            "deployment_mode",
            "memory_bandwidth_gbps",
            "compute_tflops",
            "memory_efficiency",
            "compute_efficiency",
            "launch_overhead_us",
            "qdq_overhead_us",
            "layout_overhead_us",
            "fallback_overhead_us",
            "uncertainty",
            "calibration_path",
        },
        "$.latency",
    )
    raw_latency_mode = latency_doc.get("mode", "off")
    # YAML 1.1 parses an unquoted ``off`` as boolean false.  Accept that
    # spelling so hand-written configs remain unsurprising.
    if raw_latency_mode is False:
        raw_latency_mode = "off"
    latency = LatencyConfig(
        mode=_string(raw_latency_mode, "$.latency.mode"),
        deployment_mode=_string(
            latency_doc.get("deployment_mode", "decode_bs1"),
            "$.latency.deployment_mode",
        ),
        memory_bandwidth_gbps=(
            None
            if latency_doc.get("memory_bandwidth_gbps") is None
            else _number(
                latency_doc["memory_bandwidth_gbps"],
                "$.latency.memory_bandwidth_gbps",
            )
        ),
        compute_tflops=(
            None
            if latency_doc.get("compute_tflops") is None
            else _number(latency_doc["compute_tflops"], "$.latency.compute_tflops")
        ),
        memory_efficiency=_number(
            latency_doc.get("memory_efficiency", 0.70),
            "$.latency.memory_efficiency",
        ),
        compute_efficiency=_number(
            latency_doc.get("compute_efficiency", 0.50),
            "$.latency.compute_efficiency",
        ),
        launch_overhead_us=_number(
            latency_doc.get("launch_overhead_us", 2.0),
            "$.latency.launch_overhead_us",
        ),
        qdq_overhead_us=_number(
            latency_doc.get("qdq_overhead_us", 0.0),
            "$.latency.qdq_overhead_us",
        ),
        layout_overhead_us=_number(
            latency_doc.get("layout_overhead_us", 0.0),
            "$.latency.layout_overhead_us",
        ),
        fallback_overhead_us=_number(
            latency_doc.get("fallback_overhead_us", 0.0),
            "$.latency.fallback_overhead_us",
        ),
        uncertainty=_number(
            latency_doc.get("uncertainty", 0.25), "$.latency.uncertainty"
        ),
        calibration_path=(
            None
            if latency_doc.get("calibration_path") is None
            else _input_path(
                latency_doc["calibration_path"],
                "$.latency.calibration_path",
                root_dir,
            )
        ),
    )

    output_doc = _optional_mapping(root, "output", "$.output")
    output_names = {
        "model_inventory",
        "risk_evidence",
        "candidate_plan",
        "validation_report",
        "derived_yaml",
        "provenance",
        "diff",
        "yaml_audit",
        "run_manifest",
        "latency_report",
    }
    _reject_unknown(output_doc, output_names, "$.output")
    defaults = OutputConfig()
    output = OutputConfig(
        **{
            name: _relative_path(output_doc.get(name, str(getattr(defaults, name))), f"$.output.{name}")
            for name in output_names
        }
    )

    return RunConfig(
        schema_version=_string(root["schema_version"], "$.schema_version"),
        model_path=_input_path(root["model_path"], "$.model_path", root_dir),
        data_path=_input_path(root["data_path"], "$.data_path", root_dir),
        original_yaml=_input_path(root["original_yaml"], "$.original_yaml", root_dir),
        output_dir=_input_path(root["output_dir"], "$.output_dir", root_dir),
        device=_string(root["device"], "$.device"),
        runtime=runtime,
        modules=modules,
        candidates=candidates,
        analysis=analysis,
        search=search,
        latency=latency,
        output=output,
    )


def loads_run_config(source: str | bytes, *, base_dir: str | Path | None = None) -> RunConfig:
    """Load a run configuration from YAML text without executing YAML objects."""

    try:
        document = yaml.load(source, Loader=_UniqueKeySafeLoader)
    except (yaml.YAMLError, UnicodeError) as exc:
        raise RunConfigError(f"invalid run-config YAML: {exc}") from exc
    return parse_run_config(document, base_dir=base_dir)


def load_run_config(path: str | Path) -> RunConfig:
    """Load a run config; relative paths are anchored to the config directory."""

    source_path = Path(path).resolve()
    try:
        source = source_path.read_bytes()
    except OSError as exc:
        raise RunConfigError(f"could not read run config {source_path}: {exc}") from exc
    return loads_run_config(source, base_dir=source_path.parent)


def render_example_run_config(
    *,
    model_path: str | Path | None = None,
    data_path: str | Path | None = None,
    original_yaml: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
) -> str:
    """Render a starter config, filling any of the five required user values."""

    replacements = {
        "REPLACE_WITH_MODEL_PATH": model_path,
        "REPLACE_WITH_DATA_PATH": data_path,
        "REPLACE_WITH_ORIGINAL_MODELSLIM_YAML": original_yaml,
        "REPLACE_WITH_OUTPUT_DIRECTORY": output_dir,
        "REPLACE_WITH_DEVICE": device,
    }
    rendered = _EXAMPLE_RUN_CONFIG
    for placeholder, value in replacements.items():
        if value is None:
            continue
        text = str(value)
        if not text.strip():
            raise RunConfigError(f"value for {placeholder} must not be empty")
        rendered = rendered.replace(placeholder, json.dumps(text, ensure_ascii=False))
    return rendered


def init_run_config(
    path: str | Path,
    *,
    model_path: str | Path | None = None,
    data_path: str | Path | None = None,
    original_yaml: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a starter config without silently replacing an existing file."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing config: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_example_run_config(
            model_path=model_path,
            data_path=data_path,
            original_yaml=original_yaml,
            output_dir=output_dir,
            device=device,
        ),
        encoding="utf-8",
    )
    return destination


write_example_run_config = init_run_config


def _valid_loader(value: Any) -> bool:
    return isinstance(value, str) and (
        value == "huggingface" or _CUSTOM_LOADER_PATTERN.fullmatch(value) is not None
    )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunConfigError(f"{path} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise RunConfigError(f"{path} keys must be strings")
    return value


def _optional_mapping(parent: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    return _mapping(parent.get(key, {}), path)


def _reject_unknown(document: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise RunConfigError(f"{path} contains unknown field(s): {unknown}")


def _require_keys(document: Mapping[str, Any], required: set[str], path: str) -> None:
    missing = sorted(required - set(document))
    if missing:
        raise RunConfigError(f"{path} is missing required field(s): {missing}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunConfigError(f"{path} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _bool(value: Any, path: str) -> bool:
    _require_bool(value, path)
    return value


def _require_bool(value: Any, path: str) -> None:
    if not isinstance(value, bool):
        raise RunConfigError(f"{path} must be a boolean")


def _int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RunConfigError(f"{path} must be an integer")
    return value


def _require_positive_int(value: Any, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RunConfigError(f"{path} must be a positive integer")


def _number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RunConfigError(f"{path} must be a number")
    return float(value)


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RunConfigError(f"{path} must be a sequence")
    return value


def _strings(value: Any, path: str) -> tuple[str, ...]:
    return tuple(
        _string(item, f"{path}[{index}]")
        for index, item in enumerate(_sequence(value, path))
    )


def _validate_patterns(patterns: tuple[str, ...], path: str) -> None:
    if any(not isinstance(pattern, str) or not pattern.strip() for pattern in patterns):
        raise RunConfigError(f"{path} must contain non-empty strings")
    if len(set(patterns)) != len(patterns):
        raise RunConfigError(f"{path} must not contain duplicates")


def _precision(value: Any, path: str, *, allow_bf16: bool = False) -> PrecisionFormat:
    text = _string(value, path)
    try:
        precision = PrecisionFormat(text)
    except ValueError as exc:
        raise RunConfigError(f"{path} has unknown precision {text!r}") from exc
    if precision is PrecisionFormat.BF16 and not allow_bf16:
        raise RunConfigError(f"{path} cannot use bf16 as a quantized candidate")
    return precision


def _input_path(value: Any, path: str, base_dir: Path) -> Path:
    raw = _string(value, path)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve(strict=False)


def _relative_path(value: Any, path: str) -> Path:
    candidate = Path(_string(value, path))
    _validate_output_path(candidate, path)
    return candidate


def _validate_output_path(path: Path, field_path: str) -> None:
    if not isinstance(path, Path):
        raise RunConfigError(f"{field_path} must be a path")
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise RunConfigError(
            f"{field_path} must stay inside output_dir and use a relative file path"
        )


_EXAMPLE_RUN_CONFIG = """\
schema_version: quantplan_v1

# Required: replace every REPLACE_WITH_* value before calling validate_for_run().
model_path: REPLACE_WITH_MODEL_PATH
data_path: REPLACE_WITH_DATA_PATH
original_yaml: REPLACE_WITH_ORIGINAL_MODELSLIM_YAML
output_dir: REPLACE_WITH_OUTPUT_DIRECTORY
device: REPLACE_WITH_DEVICE  # npu:0=single card; npu:auto=resident module sharding

# Optional conservative defaults. A custom loader must use module:function syntax.
runtime:
  loader: huggingface
  revision: null
  torch_dtype: bf16
  analysis_only: true
  trust_remote_code: false
  # Explicit opt-in only; this does not synthesize image inputs.
  allow_image_text_fallback: false
  strict_target_coverage: true
  # GLM-5.3 target binding: unknown, a3 or a5.
  hardware_profile: unknown
  # Optional versioned vLLM-Ascend/PrologV3 capability snapshot.
  backend_capability_manifest: null
  # auto uses the manifest backend; external plug-ins may provide other frameworks.
  inference_framework: auto

# Empty block/include selectors defer discovery to the runtime inventory. The
# conservative excludes remain locked, and unknown target coverage fails closed.
modules:
  block_patterns: []
  include: []
  exclude: ["*vision*", "*visual*", "*mtp*", "*.mlp.gate", "*.mlp.gate.weight", "*router*"]

# Candidate presence is a hypothesis, not a backend support claim.
candidates:
  precisions: [w4a8, w8a8]

analysis:
  max_samples: 16
  max_length: 512
  batch_size: 1
  max_groups: 8
  compute_kl: false
  text_field: text
  id_field: id
  seed: 20260817

search:
  base_precision: w4a8
  high_precision_fraction: 0.25
  max_candidates: 8
  max_delta_nll: 0.02
  # Normalized planning proxies, not measured hardware latency or memory.
  costs: {w4a4: 0.75, w4a8: 1.0, w8a8: 1.5, bf16: 2.5}

# Optional fast latency model. Set mode=calibrated and provide a JSON file
# with schema quantplan.latency-calibration/v1 for sparse operator corrections.
latency:
  mode: "off"  # off, static or calibrated
  deployment_mode: decode_bs1
  memory_bandwidth_gbps: null
  compute_tflops: null
  memory_efficiency: 0.70
  compute_efficiency: 0.50
  launch_overhead_us: 2.0
  qdq_overhead_us: 0.0
  layout_overhead_us: 0.0
  fallback_overhead_us: 0.0
  uncertainty: 0.25
  calibration_path: null

# Relative paths are confined under output_dir.
output:
  model_inventory: model-inventory.json
  risk_evidence: risk-evidence.jsonl
  candidate_plan: candidate-plan.json
  validation_report: validation-report.json
  derived_yaml: derived-modelslim.yaml
  provenance: provenance.json
  diff: plan.diff
  yaml_audit: yaml-audit.json
  run_manifest: run-manifest.json
  latency_report: latency-report.json
"""


__all__ = [
    "SCHEMA_VERSION",
    "AnalysisConfig",
    "CandidateConfig",
    "LatencyConfig",
    "ModuleSelectionConfig",
    "OutputConfig",
    "RunConfig",
    "RunConfigError",
    "RuntimeConfig",
    "SearchConfig",
    "init_run_config",
    "load_run_config",
    "loads_run_config",
    "parse_run_config",
    "render_example_run_config",
    "write_example_run_config",
]
