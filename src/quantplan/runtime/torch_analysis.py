"""Optional PyTorch runtime for analysis-only causal-LM interventions.

This module deliberately imports neither PyTorch nor Transformers at module
import time.  The runtime uses a simple symmetric fake-quantization proxy at
``torch.nn.Linear`` weights and inputs.  It is useful for controlled
interventions, but it is not ModelSlim-exact and must not be presented as a
backend capability or deployment result.

Qwen-style fused expert ``gate_up_proj``/``down_proj`` parameters are handled
explicitly; router modules remain excluded by default.  The default path computes
teacher-forced NLL only.  It retains aggregate
scalars, never a dataset-sized logits cache.  Exact teacher/candidate logits KL
is opt-in and is reduced one batch at a time.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from quantplan.domain.models import (
    GroupInventory,
    InterventionMeasurement,
    PlanEvaluation,
    PrecisionFormat,
)
from quantplan.runtime.architecture import (
    BASE_SAFETY_EXCLUDES,
    ArchitectureAdapter,
    ArchitectureAdapterError,
    resolve_architecture_adapter,
)

PROXY_SEMANTICS = (
    "analysis-only symmetric Linear/fused-expert fake-quant proxy; not ModelSlim-exact"
)
QUANTIZER_ID = "quantplan.symmetric_weight_activation_fake_quant/v1"
CONTEXT_SCHEMA = "quantplan.torch-analysis-context/v2"
PLACEMENT_SCHEMA = "quantplan.model-placement/v1"
DEFAULT_LINEAR_EXCLUDE = BASE_SAFETY_EXCLUDES
ProgressCallback = Callable[[str, int, int, str, str], None]

_PLACEMENT_STRATEGIES = frozenset({"single_device", "hf_device_map", "pre_sharded"})


class TorchAnalysisError(RuntimeError):
    """Base error for the optional analysis runtime."""


class TorchDependencyError(TorchAnalysisError):
    """Raised when an optional PyTorch operation is requested without PyTorch."""


class TransformersDependencyError(TorchAnalysisError):
    """Raised when the default Hugging Face loader lacks Transformers."""


class DiscoveryError(TorchAnalysisError):
    """Raised when Transformer blocks cannot be resolved safely."""


@dataclass(frozen=True)
class TextSample:
    sample_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ModelPlacement:
    """Declared one-process model placement contract.

    ``hf_device_map`` and ``pre_sharded`` mean that complete modules are
    resident on one of several accelerator devices and the loaded model owns
    activation transfers between them.  CPU/disk offload, meta parameters,
    tensor parallelism and multi-process execution are intentionally outside
    this contract.
    """

    strategy: str
    input_device: str | None = None
    device_map: tuple[tuple[str, str], ...] | Mapping[str, str] = ()

    def __post_init__(self) -> None:
        if self.strategy not in _PLACEMENT_STRATEGIES:
            allowed = ", ".join(sorted(_PLACEMENT_STRATEGIES))
            raise ValueError(f"placement strategy must be one of {allowed}; got {self.strategy!r}")
        if self.input_device is not None and (
            not isinstance(self.input_device, str) or not self.input_device.strip()
        ):
            raise ValueError("placement input_device must be null or a non-empty string")
        normalized_map: list[tuple[str, str]] = []
        entries = self.device_map.items() if isinstance(self.device_map, Mapping) else self.device_map
        for module_name, device in entries:
            if not isinstance(module_name, str) or not isinstance(device, str):
                raise TypeError("placement device_map entries must be string pairs")
            if not device.strip():
                raise ValueError("placement device_map devices must not be empty")
            normalized_map.append((module_name, device.strip()))
        if len({name for name, _ in normalized_map}) != len(normalized_map):
            raise ValueError("placement device_map module names must be unique")
        object.__setattr__(self, "device_map", tuple(sorted(normalized_map)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLACEMENT_SCHEMA,
            "strategy": self.strategy,
            "input_device": self.input_device,
            "device_map": dict(self.device_map),
        }


@dataclass(frozen=True)
class ModelBundle:
    """Objects returned by either the built-in HF loader or a custom loader."""

    model: Any
    tokenizer: Any
    model_id: str
    revision: str | None = None
    loader_name: str = "custom"
    warnings: tuple[str, ...] = ()
    placement: ModelPlacement | None = None


@dataclass(frozen=True)
class RuntimeProvenance:
    context_id: str
    context_schema: str
    runtime: str
    torch_version: str
    model_id: str
    model_revision: str | None
    loader_name: str
    loader_warnings: tuple[str, ...]
    device: str
    placement_schema: str
    placement_strategy: str
    model_devices: tuple[str, ...]
    target_devices: tuple[str, ...]
    placement_sha256: str
    dataset_sha256: str
    sample_count: int
    batch_size: int
    max_length: int | None
    compute_kl: bool
    selected_group_ids: tuple[str, ...]
    discovered_group_count: int
    full_group_coverage: bool
    backend_compilation_ready: bool
    backend_adapter: str
    group_discovery_warnings: tuple[str, ...]
    quantization_semantics: str
    target_inventory_sha256: str
    target_weight_dtypes: tuple[str, ...]
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "context_schema": self.context_schema,
            "runtime": self.runtime,
            "torch_version": self.torch_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "loader_name": self.loader_name,
            "loader_warnings": list(self.loader_warnings),
            "device": self.device,
            "placement_schema": self.placement_schema,
            "placement_strategy": self.placement_strategy,
            "model_devices": list(self.model_devices),
            "target_devices": list(self.target_devices),
            "placement_sha256": self.placement_sha256,
            "dataset_sha256": self.dataset_sha256,
            "sample_count": self.sample_count,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "compute_kl": self.compute_kl,
            "selected_group_ids": list(self.selected_group_ids),
            "discovered_group_count": self.discovered_group_count,
            "full_group_coverage": self.full_group_coverage,
            "backend_compilation_ready": self.backend_compilation_ready,
            "backend_adapter": self.backend_adapter,
            "group_discovery_warnings": list(self.group_discovery_warnings),
            "quantization_semantics": self.quantization_semantics,
            "target_inventory_sha256": self.target_inventory_sha256,
            "target_weight_dtypes": list(self.target_weight_dtypes),
            "created_at_utc": self.created_at_utc,
        }


@dataclass(frozen=True)
class AnalysisRunResult:
    groups: tuple[GroupInventory, ...]
    module_inventory: tuple[str, ...]
    provenance: RuntimeProvenance
    baseline_nll: float
    token_count: int
    measurements: tuple[InterventionMeasurement, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": [_group_inventory_to_dict(group) for group in self.groups],
            "module_inventory": list(self.module_inventory),
            "provenance": self.provenance.to_dict(),
            "baseline_nll": self.baseline_nll,
            "token_count": self.token_count,
            "measurements": [_measurement_to_dict(item) for item in self.measurements],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)


@dataclass(frozen=True)
class _ForwardAggregate:
    nll_sum: float
    token_count: int
    elapsed_seconds: float

    @property
    def mean_nll(self) -> float:
        if self.token_count <= 0:
            raise TorchAnalysisError("teacher-forced evaluation produced no target tokens")
        return self.nll_sum / self.token_count


@dataclass(frozen=True)
class _ResolvedPlacement:
    placement: ModelPlacement
    input_device: str
    model_devices: tuple[str, ...]
    target_devices: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class _RuntimeGroup:
    inventory: GroupInventory
    block_module_names: tuple[str, ...]
    linear_module_names: tuple[str, ...]
    expert_targets: tuple[_ExpertTarget, ...]


@dataclass(frozen=True)
class _ExpertTarget:
    module_name: str
    parameter_names: tuple[str, ...]
    backend_module_names: tuple[str, ...]
    backend_pattern: str
    activation_kwarg: str


@dataclass(frozen=True)
class _BoundFusedTarget:
    module: Any
    parameter_names: tuple[str, ...]
    activation_kwarg: str


def intervention_measurement_to_dict(
    measurement: InterventionMeasurement,
) -> dict[str, Any]:
    """Convert a shared-domain intervention measurement to JSON-safe values."""

    return _measurement_to_dict(measurement)


def plan_evaluation_to_dict(evaluation: PlanEvaluation) -> dict[str, Any]:
    """Convert a shared-domain plan evaluation to JSON-safe values."""

    return {
        "plan_id": evaluation.plan_id,
        "context_id": evaluation.context_id,
        "baseline_nll": evaluation.baseline_nll,
        "candidate_nll": evaluation.candidate_nll,
        "delta_nll": evaluation.delta_nll,
        "kl_divergence": evaluation.kl_divergence,
        "token_count": evaluation.token_count,
        "elapsed_seconds": evaluation.elapsed_seconds,
        "warnings": list(evaluation.warnings),
    }


def load_jsonl_texts(
    path: str | Path,
    *,
    text_field: str = "text",
    id_field: str = "id",
    limit: int | None = None,
) -> tuple[TextSample, ...]:
    """Load non-empty text records from a JSONL file.

    Blank lines are ignored.  IDs default to one-based source line numbers.
    All non-text fields are retained as JSON-compatible metadata.
    """

    if not text_field:
        raise ValueError("text_field must not be empty")
    if not id_field:
        raise ValueError("id_field must not be empty")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least one")

    source = Path(path)
    samples: list[TextSample] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise TypeError(f"{source}:{line_number}: each record must be a JSON object")
        text = record.get(text_field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"{source}:{line_number}: field {text_field!r} must be a non-empty string"
            )
        sample_id = str(record.get(id_field, line_number))
        if sample_id in seen_ids:
            raise ValueError(f"{source}:{line_number}: duplicate sample id {sample_id!r}")
        seen_ids.add(sample_id)
        metadata = {
            key: value for key, value in record.items() if key not in {text_field, id_field}
        }
        samples.append(TextSample(sample_id=sample_id, text=text, metadata=metadata))
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise ValueError(f"{source} contains no usable text records")
    return tuple(samples)


def load_causal_lm(
    model_name_or_path: str,
    *,
    revision: str | None = None,
    device: str | None = None,
    torch_dtype: Any = "bfloat16",
    trust_remote_code: bool = False,
    allow_image_text_fallback: bool = False,
    loader_entrypoint: str | Callable[..., Any] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
) -> ModelBundle:
    """Load a generic Hugging Face causal LM or invoke a custom loader.

    A custom loader receives keyword arguments matching this function and may
    return ``ModelBundle``, ``(model, tokenizer)``, or a mapping containing
    ``model`` and ``tokenizer``.  String entrypoints use ``"module:function"``.
    """

    torch = _require_torch()
    if isinstance(device, str) and _device_kind(device) == "npu":
        _require_torch_npu()
    auto_placement = _is_auto_device(device)
    resolved_dtype = _resolve_torch_dtype(torch, torch_dtype)
    model_options = dict(model_kwargs or {})
    tokenizer_options = dict(tokenizer_kwargs or {})

    if loader_entrypoint is not None:
        loader = _resolve_entrypoint(loader_entrypoint)
        loaded = loader(
            model_name_or_path=model_name_or_path,
            revision=revision,
            device=device,
            torch_dtype=resolved_dtype,
            trust_remote_code=trust_remote_code,
            allow_image_text_fallback=allow_image_text_fallback,
            model_kwargs=model_options,
            tokenizer_kwargs=tokenizer_options,
        )
        bundle = _coerce_model_bundle(
            loaded,
            model_id=model_name_or_path,
            revision=revision,
            loader_name=_entrypoint_name(loader_entrypoint),
        )
    else:
        transformers = _require_transformers()
        configured_device_map = model_options.get("device_map")
        if device is not None and not auto_placement and configured_device_map is not None:
            raise TorchAnalysisError(
                "an explicit device cannot be combined with model_kwargs.device_map; "
                "use device='auto' for resident module sharding"
            )
        if revision is not None:
            model_options.setdefault("revision", revision)
            tokenizer_options.setdefault("revision", revision)
        model_options.setdefault("trust_remote_code", trust_remote_code)
        tokenizer_options.setdefault("trust_remote_code", trust_remote_code)
        if resolved_dtype is not None:
            model_options.setdefault("torch_dtype", resolved_dtype)
        if auto_placement:
            model_options.setdefault("device_map", "auto")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, **tokenizer_options
        )
        loader_name = "transformers.AutoModelForCausalLM"
        loader_warnings: list[str] = []
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **model_options
            )
        except Exception as causal_error:
            image_text_loader = getattr(transformers, "AutoModelForImageTextToText", None)
            if image_text_loader is None or not allow_image_text_fallback:
                suggestion = (
                    " Transformers exposes AutoModelForImageTextToText; set "
                    "allow_image_text_fallback=True only after confirming text-only semantics."
                    if image_text_loader is not None
                    else " Provide a custom loader_entrypoint for this model architecture."
                )
                raise TorchAnalysisError(
                    "AutoModelForCausalLM failed and no automatic architecture fallback was "
                    f"taken:{suggestion} Original error: {type(causal_error).__name__}: "
                    f"{causal_error}"
                ) from causal_error
            try:
                model = image_text_loader.from_pretrained(model_name_or_path, **model_options)
            except Exception as image_text_error:  # noqa: BLE001 - third-party loader boundary
                raise TorchAnalysisError(
                    "AutoModelForCausalLM failed, then the explicitly enabled "
                    "AutoModelForImageTextToText fallback also failed. "
                    f"Causal error: {type(causal_error).__name__}: {causal_error}; "
                    f"image-text error: {type(image_text_error).__name__}: {image_text_error}"
                ) from causal_error
            loader_name = "transformers.AutoModelForImageTextToText"
            loader_warnings.append(
                "used explicit image-text architecture fallback for text-only analysis"
            )
        if device is not None and not auto_placement:
            model.to(device)
        if getattr(tokenizer, "pad_token_id", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is None:
                raise TorchAnalysisError(
                    "tokenizer has no pad_token or eos_token; provide tokenizer_kwargs "
                    "or a custom loader"
                )
            tokenizer.pad_token = eos_token
            loader_warnings.append("tokenizer pad_token was set to eos_token for batching")
        placement = _placement_from_loaded_model(
            model,
            requested_device=device,
            hf_auto=auto_placement or configured_device_map is not None,
        )
        if placement.strategy == "hf_device_map":
            loader_warnings.append(
                "Hugging Face device_map placement is accepted only for fully resident "
                "one-process module sharding; runtime validation rejects CPU/disk/meta "
                "offload and does not support TP/FSDP/EP"
            )
        bundle = ModelBundle(
            model=model,
            tokenizer=tokenizer,
            model_id=model_name_or_path,
            revision=revision,
            loader_name=loader_name,
            warnings=tuple(loader_warnings),
            placement=placement,
        )

    if hasattr(bundle.model, "eval"):
        bundle.model.eval()
    return bundle


def discover_transformer_blocks(
    model: Any,
    explicit_groups: Mapping[str, str | Sequence[str]] | Sequence[str] | None = None,
    *,
    linear_include: Sequence[str] = ("*",),
    linear_exclude: Sequence[str] = DEFAULT_LINEAR_EXCLUDE,
    max_groups: int | None = None,
    strict_targets: bool = False,
    architecture_adapter: ArchitectureAdapter | None = None,
) -> tuple[GroupInventory, ...]:
    """Discover Transformer blocks and supported quantization targets.

    Explicit mappings associate a stable group ID with one or more module names.
    A sequence treats every module name as both group ID and module name.
    Automatic discovery prefers common HF ``ModuleList`` paths, then falls back
    to the largest plausible ``ModuleList`` containing Linear descendants.
    """

    discovered = _discover_runtime_groups(
        model,
        explicit_groups,
        linear_include=linear_include,
        linear_exclude=linear_exclude,
        max_groups=None,
        strict_targets=strict_targets,
        architecture_adapter=architecture_adapter,
    )
    return tuple(group.inventory for group in _select_runtime_groups(discovered, max_groups))


def _discover_runtime_groups(
    model: Any,
    explicit_groups: Mapping[str, str | Sequence[str]] | Sequence[str] | None = None,
    *,
    linear_include: Sequence[str] = ("*",),
    linear_exclude: Sequence[str] = DEFAULT_LINEAR_EXCLUDE,
    max_groups: int | None = None,
    strict_targets: bool = False,
    architecture_adapter: ArchitectureAdapter | None = None,
) -> tuple[_RuntimeGroup, ...]:
    torch = _require_torch()
    modules = dict(model.named_modules())
    try:
        adapter = architecture_adapter or resolve_architecture_adapter(model)
    except ArchitectureAdapterError as exc:
        raise DiscoveryError(f"architecture adapter resolution failed: {exc}") from exc
    include_patterns = _validate_patterns("linear_include", linear_include, allow_empty=False)
    configured_excludes = _validate_patterns("linear_exclude", linear_exclude, allow_empty=True)
    # These patterns are safety invariants, not merely convenience defaults.  In
    # particular, a caller passing an empty configured exclude list must not make
    # router weights eligible for intervention.
    exclude_patterns = _effective_exclude_patterns(
        configured_excludes,
        adapter=adapter,
    )
    if max_groups is not None and max_groups < 1:
        raise ValueError("max_groups must be at least one")
    if explicit_groups is not None:
        normalized = _normalize_explicit_groups(explicit_groups)
        return _build_runtime_groups(
            torch,
            modules,
            normalized,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            common_warnings=(
                "explicit group discovery",
                f"architecture adapter={adapter.adapter_id}",
                "router exclusion is name-based only; verify the complete router path is locked",
            ),
            strict_targets=strict_targets,
            adapter=adapter,
        )

    container_path: str | None = None
    for path in adapter.preferred_block_paths:
        module = modules.get(path)
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            container_path = path
            break

    if container_path is None and adapter.allow_block_path_fallback:
        candidates: list[tuple[int, str]] = []
        for path, module in modules.items():
            if not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
                continue
            linear_count = sum(
                1 for child in module.modules() if isinstance(child, torch.nn.Linear)
            )
            if linear_count:
                candidates.append((len(module), path))
        if candidates:
            container_path = max(candidates, key=lambda item: (item[0], item[1]))[1]

    if container_path is None:
        raise DiscoveryError(
            f"architecture adapter {adapter.adapter_id!r} could not discover Transformer "
            "blocks; pass explicit_groups with named_modules paths"
        )

    container = modules[container_path]
    discovered = {
        f"{container_path}.{index}": (f"{container_path}.{index}",)
        for index in range(len(container))
    }
    discovered_count = len(discovered)
    selection_warning = f"automatic discovery selected all {discovered_count} blocks"
    if max_groups is not None and max_groups < discovered_count:
        selected_indexes = _uniform_indexes(discovered_count, max_groups)
        discovered = {
            key: value
            for index, (key, value) in enumerate(discovered.items())
            if index in selected_indexes
        }
        selection_warning = (
            f"automatic uniform selection chose {len(discovered)} of {discovered_count} "
            f"blocks at indexes {sorted(selected_indexes)}"
        )
    common_warnings = (
        f"architecture adapter={adapter.adapter_id}",
        f"automatic block container={container_path}",
        selection_warning,
        "router exclusion is name-based only; verify the complete router path is locked",
    )
    return _build_runtime_groups(
        torch,
        modules,
        discovered,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        common_warnings=common_warnings,
        strict_targets=strict_targets,
        adapter=adapter,
    )


class TorchAnalysisRuntime:
    """Run temporary block-level fake-quant interventions on a causal LM."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        samples: Sequence[TextSample | str],
        *,
        groups: Mapping[str, str | Sequence[str]] | Sequence[str] | None = None,
        linear_include: Sequence[str] = ("*",),
        linear_exclude: Sequence[str] = DEFAULT_LINEAR_EXCLUDE,
        max_groups: int | None = None,
        strict_targets: bool = False,
        batch_size: int = 1,
        max_length: int | None = None,
        device: str | None = None,
        compute_kl: bool = False,
        model_id: str = "provided-model",
        model_revision: str | None = None,
        loader_name: str = "provided-objects",
        loader_warnings: Sequence[str] = (),
        placement: ModelPlacement | None = None,
        context_id: str | None = None,
    ) -> None:
        self._torch = _require_torch()
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        if max_length is not None and max_length < 2:
            raise ValueError("max_length must be at least two")
        if not hasattr(model, "named_modules") or not callable(model):
            raise TypeError("model must be a callable torch module with named_modules()")
        if not callable(tokenizer):
            raise TypeError("tokenizer must be callable")

        self.model = model
        self.tokenizer = tokenizer
        self.samples = _normalize_samples(samples)
        try:
            self.architecture_adapter = resolve_architecture_adapter(model)
        except ArchitectureAdapterError as exc:
            raise DiscoveryError(f"architecture adapter resolution failed: {exc}") from exc
        discovered_groups = _discover_runtime_groups(
            model,
            groups,
            linear_include=linear_include,
            linear_exclude=linear_exclude,
            max_groups=None,
            strict_targets=strict_targets,
            architecture_adapter=self.architecture_adapter,
        )
        self.discovered_group_count = len(discovered_groups)
        self.all_groups = tuple(group.inventory for group in discovered_groups)
        self._runtime_groups = _select_runtime_groups(discovered_groups, max_groups)
        self.full_group_coverage = len(self._runtime_groups) == self.discovered_group_count
        self.groups = tuple(group.inventory for group in self._runtime_groups)
        self.module_inventory = tuple(name for name, _ in model.named_modules())
        self.backend_adapter = self.architecture_adapter.adapter_id
        backend_names = [name for name in self.module_inventory if name]
        unresolved_fused: list[str] = []
        for group in discovered_groups:
            for target in group.expert_targets:
                backend_names.extend(target.backend_module_names)
        self.backend_module_inventory = tuple(dict.fromkeys(backend_names))
        self.backend_compilation_ready = not unresolved_fused
        self.backend_compilation_blockers = tuple(
            f"no verified ModelSlim post-adapter target mapping for {name}"
            for name in dict.fromkeys(unresolved_fused)
        )
        self.batch_size = batch_size
        self.max_length = max_length
        self.compute_kl = bool(compute_kl)
        self.model_id = model_id
        self.model_revision = model_revision
        self.loader_name = loader_name
        self.loader_warnings = tuple(loader_warnings)
        if context_id is not None and (not isinstance(context_id, str) or not context_id.strip()):
            raise ValueError("context_id must be None or a non-empty string")
        target_inventory = _selected_target_inventory(model, self._runtime_groups)
        resolved_placement = _resolve_runtime_placement(
            self._torch,
            model,
            target_inventory,
            declared=placement,
            requested_device=device,
        )
        self.placement = resolved_placement.placement
        self.device = resolved_placement.input_device
        self.model_devices = resolved_placement.model_devices
        self.target_devices = resolved_placement.target_devices
        self.placement_sha256 = resolved_placement.sha256
        self.target_weight_dtypes = tuple(
            sorted(
                {
                    str(target["dtype"])
                    for group_inventory in target_inventory
                    for target in group_inventory["targets"]
                }
            )
        )
        _require_bf16_target_anchor(
            self._torch,
            target_inventory,
        )
        self.target_inventory_sha256 = _stable_json_hash(target_inventory)
        self._context_namespace = context_id or "torch-proxy"
        context_payload = self._build_context_payload(target_inventory, resolved_placement)
        self._context_payload_sha256 = _stable_json_hash(context_payload)
        self.context_id = f"{self._context_namespace}:{self._context_payload_sha256}"
        self._group_by_id = {group.inventory.group_id: group for group in self._runtime_groups}
        try:
            forward_parameters = inspect.signature(model.forward).parameters.values()
        except (AttributeError, TypeError, ValueError):
            self._disable_kv_cache = False
            self.loader_warnings = (
                *self.loader_warnings,
                "could not inspect model.forward; use_cache was not overridden",
            )
        else:
            self._disable_kv_cache = any(
                (
                    parameter.name == "use_cache"
                    and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
                )
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in forward_parameters
            )
        self._baseline: _ForwardAggregate | None = None

    @classmethod
    def from_bundle(
        cls,
        bundle: ModelBundle,
        samples: Sequence[TextSample | str],
        **kwargs: Any,
    ) -> TorchAnalysisRuntime:
        return cls(
            bundle.model,
            bundle.tokenizer,
            samples,
            model_id=bundle.model_id,
            model_revision=bundle.revision,
            loader_name=bundle.loader_name,
            loader_warnings=bundle.warnings,
            placement=bundle.placement,
            **kwargs,
        )

    @classmethod
    def from_jsonl(
        cls,
        model: Any,
        tokenizer: Any,
        path: str | Path,
        *,
        text_field: str = "text",
        id_field: str = "id",
        limit: int | None = None,
        **kwargs: Any,
    ) -> TorchAnalysisRuntime:
        samples = load_jsonl_texts(path, text_field=text_field, id_field=id_field, limit=limit)
        return cls(model, tokenizer, samples, **kwargs)

    def run(
        self,
        precisions: Sequence[PrecisionFormat | str] = (
            PrecisionFormat.W4A8,
            PrecisionFormat.W8A8,
            PrecisionFormat.BF16,
        ),
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> AnalysisRunResult:
        """Evaluate every group/precision pair.

        ``progress_callback`` receives ``(stage, current, total, group_id,
        precision)`` immediately before and after each measurement.  Stage is
        respectively ``"measurement_start"`` or ``"measurement_complete"``;
        ``current`` is one-based and precision is the serialized format value.
        Callback exceptions propagate while no fake-quant context is active.
        """

        normalized = tuple(_normalize_precision(value) for value in precisions)
        if not normalized:
            raise ValueError("precisions must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("precisions must not contain duplicates")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None")

        self._assert_context_integrity()
        baseline = self._ensure_baseline()
        measurements: list[InterventionMeasurement] = []
        total = len(self._runtime_groups) * len(normalized)
        current = 0
        for group in self._runtime_groups:
            for precision in normalized:
                current += 1
                if progress_callback is not None:
                    progress_callback(
                        "measurement_start",
                        current,
                        total,
                        group.inventory.group_id,
                        precision.value,
                    )
                    self._assert_context_integrity()
                measurement = self._evaluate_group(group, precision, baseline)
                self._assert_context_integrity()
                measurements.append(measurement)
                if progress_callback is not None:
                    progress_callback(
                        "measurement_complete",
                        current,
                        total,
                        group.inventory.group_id,
                        precision.value,
                    )
                    self._assert_context_integrity()

        return AnalysisRunResult(
            groups=self.groups,
            module_inventory=self.module_inventory,
            provenance=self._provenance(),
            baseline_nll=baseline.mean_nll,
            token_count=baseline.token_count,
            measurements=tuple(measurements),
        )

    def evaluate_plan(
        self,
        assignments: Mapping[str, PrecisionFormat | str],
        *,
        plan_id: str | None = None,
        context_id: str | None = None,
    ) -> PlanEvaluation:
        """Evaluate a multi-group assignment with guaranteed intervention cleanup."""

        if not assignments:
            raise ValueError("assignments must not be empty")
        if context_id is not None and context_id != self.context_id:
            raise ValueError("context_id override must equal the runtime experiment context")
        self._assert_context_integrity()
        unknown = sorted(set(assignments) - set(self._group_by_id))
        if unknown:
            raise ValueError(f"unknown group IDs: {unknown}")
        normalized = {
            group_id: _normalize_precision(precision) for group_id, precision in assignments.items()
        }
        selected_groups = tuple(self._group_by_id[group_id] for group_id in normalized)
        self._ensure_disjoint_linears(selected_groups)
        baseline = self._ensure_baseline()
        aggregate, kl = self._evaluate_assignment(normalized)
        warnings = list(self._measurement_warnings(normalized.values()))
        warnings.extend(
            warning for group in selected_groups for warning in group.inventory.warnings
        )
        if len(normalized) < len(self._group_by_id):
            warnings.append("groups absent from assignments remain at reference precision")
        assignment_text = ",".join(
            f"{group_id}={precision.value}" for group_id, precision in sorted(normalized.items())
        )
        warnings.append("plan result requires independent whole-model and backend validation")
        warnings.append(f"assignments: {assignment_text}")
        resolved_plan_id = plan_id or (
            "plan-" + hashlib.sha256(assignment_text.encode("utf-8")).hexdigest()[:16]
        )
        return PlanEvaluation(
            plan_id=resolved_plan_id,
            context_id=self.context_id,
            baseline_nll=baseline.mean_nll,
            candidate_nll=aggregate.mean_nll,
            kl_divergence=kl,
            token_count=aggregate.token_count,
            elapsed_seconds=aggregate.elapsed_seconds,
            warnings=tuple(warnings),
        )

    def _assert_context_integrity(self) -> None:
        current = _selected_target_inventory(self.model, self._runtime_groups)
        _require_bf16_target_anchor(self._torch, current)
        if _stable_json_hash(current) != self.target_inventory_sha256:
            raise TorchAnalysisError(
                "selected quantization target inventory changed after runtime "
                "construction; create a new runtime/context before measuring"
            )
        placement = _resolve_runtime_placement(
            self._torch,
            self.model,
            current,
            declared=self.placement,
            requested_device=self.device,
        )
        if placement.sha256 != self.placement_sha256:
            raise TorchAnalysisError(
                "model placement changed after runtime construction; create a new "
                "runtime/context before measuring"
            )
        current_payload_sha256 = _stable_json_hash(
            self._build_context_payload(current, placement)
        )
        expected_context_id = f"{self._context_namespace}:{self._context_payload_sha256}"
        if (
            current_payload_sha256 != self._context_payload_sha256
            or self.context_id != expected_context_id
        ):
            raise TorchAnalysisError(
                "experiment inputs or analysis settings changed after runtime construction; "
                "create a new runtime/context before measuring"
            )

    def _build_context_payload(
        self,
        target_inventory: Sequence[Mapping[str, Any]],
        placement: _ResolvedPlacement,
    ) -> dict[str, Any]:
        target_dtypes = sorted(
            {
                str(target["dtype"])
                for group_inventory in target_inventory
                for target in group_inventory["targets"]
            }
        )
        return {
            "schema": CONTEXT_SCHEMA,
            "namespace": self._context_namespace,
            "model": {
                "id": str(self.model_id),
                "revision": self.model_revision,
                "class": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
            },
            "tokenizer": _tokenizer_identity(self.tokenizer),
            "dataset_sha256": _dataset_hash(self.samples),
            "sample_count": len(self.samples),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "device": self.device,
            "placement": {
                **placement.placement.to_dict(),
                "model_devices": list(placement.model_devices),
                "target_devices": list(placement.target_devices),
                "sha256": placement.sha256,
            },
            "compute_kl": self.compute_kl,
            "architecture_adapter_id": self.architecture_adapter.adapter_id,
            "selected_targets": target_inventory,
            "target_weight_dtypes": target_dtypes,
            "architecture_adapter": self.backend_adapter,
            "quantizer_id": QUANTIZER_ID,
            "quantization_semantics": PROXY_SEMANTICS,
        }

    def _evaluate_group(
        self,
        group: _RuntimeGroup,
        precision: PrecisionFormat,
        baseline: _ForwardAggregate,
    ) -> InterventionMeasurement:
        if precision is PrecisionFormat.BF16:
            aggregate = baseline
            kl = 0.0 if self.compute_kl else None
        else:
            aggregate, kl = self._evaluate_assignment({group.inventory.group_id: precision})
        return InterventionMeasurement(
            group_id=group.inventory.group_id,
            precision=precision,
            context_id=self.context_id,
            module_names=group.inventory.module_names,
            parameter_count=group.inventory.parameter_count,
            baseline_nll=baseline.mean_nll,
            candidate_nll=aggregate.mean_nll,
            kl_divergence=kl,
            token_count=aggregate.token_count,
            elapsed_seconds=aggregate.elapsed_seconds,
            quantizer_id=QUANTIZER_ID,
            warnings=(
                *self._measurement_warnings((precision,)),
                *group.inventory.warnings,
            ),
        )

    def _evaluate_assignment(
        self,
        assignments: Mapping[str, PrecisionFormat],
    ) -> tuple[_ForwardAggregate, float | None]:
        effective = {
            group_id: precision
            for group_id, precision in assignments.items()
            if precision is not PrecisionFormat.BF16
        }
        if not effective:
            baseline = self._ensure_baseline()
            return baseline, 0.0 if self.compute_kl else None
        if self.compute_kl:
            return self._paired_forward(effective)
        with self._interventions(effective):
            return self._forward_aggregate(), None

    def _ensure_baseline(self) -> _ForwardAggregate:
        if self._baseline is None:
            self._baseline = self._forward_aggregate()
        return self._baseline

    def _forward_aggregate(self) -> _ForwardAggregate:
        started = time.perf_counter()
        nll_sum = 0.0
        token_count = 0
        with _evaluation_mode(self.model), self._torch.inference_mode():
            for batch in self._batches():
                logits = self._forward_logits(batch)
                batch_nll, batch_tokens = _nll_sum_and_count(
                    self._torch, logits, batch["input_ids"], batch["attention_mask"]
                )
                nll_sum += batch_nll
                token_count += batch_tokens
                del logits
        return _ForwardAggregate(nll_sum, token_count, time.perf_counter() - started)

    def _paired_forward(
        self,
        assignments: Mapping[str, PrecisionFormat],
    ) -> tuple[_ForwardAggregate, float]:
        """Stream exact KL without retaining logits beyond the current batch."""

        started = time.perf_counter()
        nll_sum = 0.0
        kl_sum = 0.0
        token_count = 0
        with _evaluation_mode(self.model), self._torch.inference_mode():
            for batch in self._batches():
                reference_logits = self._forward_logits(batch)
                with self._interventions(assignments):
                    candidate_logits = self._forward_logits(batch)
                batch_nll, batch_tokens = _nll_sum_and_count(
                    self._torch,
                    candidate_logits,
                    batch["input_ids"],
                    batch["attention_mask"],
                )
                kl_sum += _kl_sum(
                    self._torch,
                    reference_logits,
                    candidate_logits,
                    batch["attention_mask"],
                )
                nll_sum += batch_nll
                token_count += batch_tokens
                del reference_logits, candidate_logits
        if token_count <= 0:
            raise TorchAnalysisError("teacher-forced evaluation produced no target tokens")
        aggregate = _ForwardAggregate(nll_sum, token_count, time.perf_counter() - started)
        # Floating-point reduction can produce a tiny negative value for a
        # theoretically non-negative KL.
        return aggregate, max(0.0, kl_sum / token_count)

    def _batches(self) -> Iterator[dict[str, Any]]:
        for start in range(0, len(self.samples), self.batch_size):
            texts = [sample.text for sample in self.samples[start : start + self.batch_size]]
            kwargs: dict[str, Any] = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": self.max_length is not None,
            }
            if self.max_length is not None:
                kwargs["max_length"] = self.max_length
            encoded = self.tokenizer(texts, **kwargs)
            if "input_ids" not in encoded:
                raise TorchAnalysisError("tokenizer output must contain input_ids")
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = self._torch.ones_like(input_ids)
            if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise TorchAnalysisError(
                    "tokenizer input_ids and attention_mask must have matching "
                    "[batch, sequence] shape"
                )
            yield {
                "input_ids": input_ids.to(self.device),
                "attention_mask": attention_mask.to(self.device),
            }

    def _forward_logits(self, batch: Mapping[str, Any]) -> Any:
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        if self._disable_kv_cache:
            kwargs["use_cache"] = False
        output = self.model(**kwargs)
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, (tuple, list)) and output:
            logits = output[0]
        if logits is None or getattr(logits, "ndim", 0) != 3:
            raise TorchAnalysisError("causal LM output must expose [batch, sequence, vocab] logits")
        if not bool(self._torch.isfinite(logits).all()):
            raise TorchAnalysisError("model produced non-finite logits")
        return logits

    @contextmanager
    def _interventions(self, assignments: Mapping[str, PrecisionFormat]) -> Iterator[None]:
        groups = tuple(self._group_by_id[group_id] for group_id in assignments)
        self._ensure_disjoint_linears(groups)
        modules = dict(self.model.named_modules())
        with ExitStack() as stack:
            for group_id, precision in assignments.items():
                group = self._group_by_id[group_id]
                linears = tuple(modules[name] for name in group.linear_module_names)
                fused_targets = tuple(
                    _BoundFusedTarget(
                        module=modules[target.module_name],
                        parameter_names=target.parameter_names,
                        activation_kwarg=target.activation_kwarg,
                    )
                    for target in group.expert_targets
                )
                if linears:
                    stack.enter_context(
                        temporary_linear_fake_quant(
                            linears,
                            precision,
                            torch_module=self._torch,
                        )
                    )
                if fused_targets:
                    stack.enter_context(
                        _temporary_fused_targets_fake_quant(
                            fused_targets,
                            precision,
                            torch_module=self._torch,
                        )
                    )
            yield

    @staticmethod
    def _ensure_disjoint_linears(groups: Sequence[_RuntimeGroup]) -> None:
        owners: dict[str, str] = {}
        for group in groups:
            for name in group.inventory.module_names:
                previous = owners.setdefault(name, group.inventory.group_id)
                if previous != group.inventory.group_id:
                    raise ValueError(
                        f"groups {previous!r} and {group.inventory.group_id!r} "
                        f"overlap at Linear {name!r}"
                    )

    def _measurement_warnings(self, precisions: Sequence[PrecisionFormat]) -> tuple[str, ...]:
        warnings = [PROXY_SEMANTICS, *self.loader_warnings]
        if not self.compute_kl:
            warnings.append("KL disabled; kl_divergence is null")
        if all(precision is PrecisionFormat.BF16 for precision in precisions):
            warnings.append("BF16 candidate reuses the unmodified reference forward")
        return tuple(warnings)

    def _provenance(self) -> RuntimeProvenance:
        return RuntimeProvenance(
            context_id=self.context_id,
            context_schema=CONTEXT_SCHEMA,
            runtime="quantplan.torch_analysis/v1",
            torch_version=str(self._torch.__version__),
            model_id=self.model_id,
            model_revision=self.model_revision,
            loader_name=self.loader_name,
            loader_warnings=self.loader_warnings,
            device=self.device,
            placement_schema=PLACEMENT_SCHEMA,
            placement_strategy=self.placement.strategy,
            model_devices=self.model_devices,
            target_devices=self.target_devices,
            placement_sha256=self.placement_sha256,
            dataset_sha256=_dataset_hash(self.samples),
            sample_count=len(self.samples),
            batch_size=self.batch_size,
            max_length=self.max_length,
            compute_kl=self.compute_kl,
            selected_group_ids=tuple(group.group_id for group in self.groups),
            discovered_group_count=self.discovered_group_count,
            full_group_coverage=self.full_group_coverage,
            backend_compilation_ready=self.backend_compilation_ready,
            backend_adapter=self.backend_adapter,
            group_discovery_warnings=_ordered_unique(
                iter(warning for group in self.groups for warning in group.warnings)
            ),
            quantization_semantics=PROXY_SEMANTICS,
            target_inventory_sha256=self.target_inventory_sha256,
            target_weight_dtypes=self.target_weight_dtypes,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )


@contextmanager
def temporary_linear_fake_quant(
    linears: Sequence[Any],
    precision: PrecisionFormat | str,
    *,
    torch_module: Any | None = None,
) -> Iterator[None]:
    """Lazily fake-quantize Linear weights and inputs for each invocation.

    Hooks quantize one Linear immediately before its forward and restore it
    immediately afterwards.  A complete mixed plan therefore needs only one
    active Linear backup at a time instead of cloning every selected weight at
    context entry.  The outer cleanup remains authoritative when a forward
    raises before the post-hook runs.
    """

    torch = torch_module or _require_torch()
    normalized = _normalize_precision(precision)
    if normalized is PrecisionFormat.BF16:
        yield
        return
    weight_bits = 4 if normalized is PrecisionFormat.W4A8 else 8
    unique_linears: list[Any] = []
    seen: set[int] = set()
    for module in linears:
        if not isinstance(module, torch.nn.Linear):
            raise TypeError("temporary_linear_fake_quant accepts only torch.nn.Linear modules")
        if id(module) not in seen:
            seen.add(id(module))
            unique_linears.append(module)
    if not unique_linears:
        raise ValueError("at least one Linear module is required")

    handles: list[Any] = []
    active_backups: dict[int, tuple[Any, Any]] = {}

    def restore(module: Any) -> None:
        entry = active_backups.pop(id(module), None)
        if entry is None:
            return
        _, backup = entry
        with torch.no_grad():
            module.weight.copy_(backup)

    def activation_weight_hook(module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
        identity = id(module)
        if identity in active_backups:
            raise TorchAnalysisError("re-entrant Linear fake quantization is not supported")
        backup = module.weight.detach().clone()
        active_backups[identity] = (module, backup)
        try:
            with torch.no_grad():
                _fake_quantize_parameter_in_place(torch, module.weight, weight_bits)
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            quantized = _fake_quant_activation(torch, inputs[0], 8)
            return (quantized, *inputs[1:])
        except BaseException:
            restore(module)
            raise

    def restore_hook(module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        restore(module)
        return output

    try:
        for module in unique_linears:
            handles.append(module.register_forward_pre_hook(activation_weight_hook))
            handles.append(module.register_forward_hook(restore_hook, always_call=True))
        yield
    finally:
        try:
            for handle in reversed(handles):
                handle.remove()
        finally:
            for module, _backup in reversed(tuple(active_backups.values())):
                restore(module)


@contextmanager
def _temporary_fused_targets_fake_quant(
    targets: Sequence[_BoundFusedTarget],
    precision: PrecisionFormat | str,
    *,
    torch_module: Any | None = None,
) -> Iterator[None]:
    """Lazily QDQ adapter-declared fused weights and activation inputs."""

    torch = torch_module or _require_torch()
    normalized = _normalize_precision(precision)
    if normalized is PrecisionFormat.BF16:
        yield
        return
    weight_bits = 4 if normalized is PrecisionFormat.W4A8 else 8
    handles: list[Any] = []
    active_backups: dict[int, tuple[Any, tuple[tuple[str, Any], ...]]] = {}
    target_by_id: dict[int, _BoundFusedTarget] = {}
    for target in targets:
        identity = id(target.module)
        previous = target_by_id.get(identity)
        if previous is not None and previous != target:
            raise ValueError("one fused module received conflicting adapter mappings")
        target_by_id[identity] = target
    if not target_by_id:
        raise ValueError("at least one fused target is required")

    def restore(module: Any) -> None:
        entry = active_backups.pop(id(module), None)
        if entry is None:
            return
        _, backups = entry
        with torch.no_grad():
            for parameter_name, backup in backups:
                getattr(module, parameter_name).copy_(backup)

    def expert_activation_weight_hook(
        module: Any,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        identity = id(module)
        target = target_by_id[identity]
        if identity in active_backups:
            raise TorchAnalysisError("re-entrant fused expert fake quantization is not supported")
        backups: list[tuple[str, Any]] = []
        try:
            with torch.no_grad():
                for parameter_name in target.parameter_names:
                    parameter = getattr(module, parameter_name, None)
                    if parameter is None or getattr(parameter, "ndim", 0) < 2:
                        raise TypeError(
                            "adapter-declared fused parameters must exist and have "
                            "at least two dimensions"
                        )
                    backups.append((parameter_name, parameter.detach().clone()))
                active_backups[identity] = (module, tuple(backups))
                for parameter_name, _backup in backups:
                    _fake_quantize_parameter_in_place(
                        torch,
                        getattr(module, parameter_name),
                        weight_bits,
                    )

            if inputs and torch.is_tensor(inputs[0]):
                return (
                    (_fake_quant_activation(torch, inputs[0], 8), *inputs[1:]),
                    kwargs,
                )
            hidden_states = kwargs.get(target.activation_kwarg)
            if hidden_states is not None and torch.is_tensor(hidden_states):
                updated = dict(kwargs)
                updated[target.activation_kwarg] = _fake_quant_activation(torch, hidden_states, 8)
                return inputs, updated
            raise TorchAnalysisError(
                "fused expert intervention could not observe a tensor activation at the "
                f"first positional argument or keyword {target.activation_kwarg!r}; "
                "refusing to label a weight-only intervention as W4A8/W8A8"
            )
        except BaseException:
            if identity in active_backups:
                restore(module)
            raise

    def restore_hook(module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        restore(module)
        return output

    try:
        for target in target_by_id.values():
            module = target.module
            handles.append(
                module.register_forward_pre_hook(
                    expert_activation_weight_hook,
                    with_kwargs=True,
                )
            )
            handles.append(module.register_forward_hook(restore_hook, always_call=True))
        yield
    finally:
        try:
            for handle in reversed(handles):
                handle.remove()
        finally:
            for module, _backups in reversed(tuple(active_backups.values())):
                restore(module)


@contextmanager
def temporary_fused_expert_fake_quant(
    expert_modules: Sequence[Any],
    precision: PrecisionFormat | str,
    *,
    torch_module: Any | None = None,
) -> Iterator[None]:
    """Backward-compatible helper for Qwen-style fused expert modules."""

    targets = tuple(
        _BoundFusedTarget(
            module=module,
            parameter_names=("gate_up_proj", "down_proj"),
            activation_kwarg="hidden_states",
        )
        for module in expert_modules
    )
    with _temporary_fused_targets_fake_quant(
        targets,
        precision,
        torch_module=torch_module,
    ):
        yield


@contextmanager
def temporary_group_fake_quant(
    linears: Sequence[Any],
    expert_modules: Sequence[Any],
    precision: PrecisionFormat | str,
    *,
    torch_module: Any | None = None,
) -> Iterator[None]:
    """Apply all supported proxy targets in one group with atomic cleanup."""

    torch = torch_module or _require_torch()
    if not linears and not expert_modules:
        raise ValueError("a group must contain at least one supported quantization target")
    with ExitStack() as stack:
        if linears:
            stack.enter_context(temporary_linear_fake_quant(linears, precision, torch_module=torch))
        if expert_modules:
            stack.enter_context(
                temporary_fused_expert_fake_quant(expert_modules, precision, torch_module=torch)
            )
        yield


def _fake_quant_weight(torch: Any, value: Any, bits: int) -> Any:
    work = value.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(work).all()):
        raise TorchAnalysisError("Linear weight contains non-finite values")
    qmax = float((1 << (bits - 1)) - 1)
    # Linear [out, in] and fused expert [expert, out, in] weights both use a
    # scale per output row while reducing only the final input dimension.
    maximum = work.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum))
    quantized = torch.round(work / scale).clamp(-qmax, qmax) * scale
    return quantized.to(dtype=value.dtype)


def _fake_quantize_parameter_in_place(
    torch: Any,
    value: Any,
    bits: int,
    *,
    row_chunk_size: int = 4096,
) -> None:
    """QDQ a parameter by last-dimension rows without a full FP32 copy."""

    if value.ndim < 2:
        raise TorchAnalysisError("weight fake quantization requires at least two dimensions")
    if not value.is_contiguous():
        raise TorchAnalysisError("non-contiguous quantization parameters are not supported")
    rows = value.view(-1, value.shape[-1])
    for start in range(0, rows.shape[0], row_chunk_size):
        target = rows[start : start + row_chunk_size]
        target.copy_(_fake_quant_weight(torch, target, bits))


def _fake_quant_activation(torch: Any, value: Any, bits: int) -> Any:
    if not value.is_floating_point():
        return value
    work = value.to(dtype=torch.float32)
    if not bool(torch.isfinite(work).all()):
        raise TorchAnalysisError("Linear activation contains non-finite values")
    qmax = float((1 << (bits - 1)) - 1)
    maximum = work.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum))
    quantized = torch.round(work / scale).clamp(-qmax, qmax) * scale
    return quantized.to(dtype=value.dtype)


def _nll_sum_and_count(
    torch: Any, logits: Any, input_ids: Any, attention_mask: Any
) -> tuple[float, int]:
    if logits.shape[:2] != input_ids.shape:
        raise TorchAnalysisError("logits and input_ids batch/sequence dimensions must match")
    if input_ids.shape[1] < 2:
        return 0.0, 0
    prediction = logits[:, :-1, :].to(dtype=torch.float32)
    labels = input_ids[:, 1:].to(prediction.device)
    # A valid next-token target requires both a real context token and a real
    # target token.  Checking only the target side incorrectly scores the first
    # non-padding token when a tokenizer uses left padding.
    valid = attention_mask[:, :-1].to(device=prediction.device, dtype=torch.bool) & attention_mask[
        :, 1:
    ].to(device=prediction.device, dtype=torch.bool)
    losses = torch.nn.functional.cross_entropy(
        prediction.reshape(-1, prediction.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)
    return float(losses.masked_select(valid).sum().item()), int(valid.sum().item())


def _kl_sum(torch: Any, reference_logits: Any, candidate_logits: Any, mask: Any) -> float:
    if reference_logits.shape != candidate_logits.shape:
        raise TorchAnalysisError("reference and candidate logits shapes must match for KL")
    reference = reference_logits[:, :-1, :].to(dtype=torch.float32)
    if candidate_logits.device != reference_logits.device:
        raise TorchAnalysisError("reference and candidate logits must be on the same device for KL")
    candidate = candidate_logits[:, :-1, :].to(dtype=torch.float32)
    log_p = torch.nn.functional.log_softmax(reference, dim=-1)
    log_q = torch.nn.functional.log_softmax(candidate, dim=-1)
    token_kl = torch.sum(torch.exp(log_p) * (log_p - log_q), dim=-1)
    valid = mask[:, :-1].to(device=token_kl.device, dtype=torch.bool) & mask[:, 1:].to(
        device=token_kl.device, dtype=torch.bool
    )
    return float(token_kl.masked_select(valid).sum().item())


def _build_runtime_groups(
    torch: Any,
    modules: Mapping[str, Any],
    groups: Mapping[str, Sequence[str]],
    *,
    include_patterns: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
    common_warnings: tuple[str, ...],
    strict_targets: bool,
    adapter: ArchitectureAdapter,
) -> tuple[_RuntimeGroup, ...]:
    result: list[_RuntimeGroup] = []
    seen_module_names: set[str] = set()
    for group_id, module_names in groups.items():
        if not group_id:
            raise DiscoveryError("group IDs must not be empty")
        if not module_names:
            raise DiscoveryError(f"group {group_id!r} has no module names")
        linear_names: list[str] = []
        linear_ids: set[int] = set()
        expert_targets: list[_ExpertTarget] = []
        quantized_target_names: list[str] = []
        backend_patterns: list[str] = []
        excluded_target_names: list[str] = []
        group_warnings = list(common_warnings)
        expert_like_module_seen = False
        expert_coverage = False
        for module_name in module_names:
            if module_name not in modules:
                raise DiscoveryError(f"module {module_name!r} for group {group_id!r} was not found")
            if module_name in seen_module_names:
                raise DiscoveryError(f"module {module_name!r} belongs to more than one group")
            seen_module_names.add(module_name)
            block = modules[module_name]
            for relative_name, child in block.named_modules():
                full_name = module_name if not relative_name else f"{module_name}.{relative_name}"
                lowered = full_name.lower()
                class_name = type(child).__name__.lower()
                is_expert_like = "expert" in lowered or "expert" in class_name
                expert_like_module_seen = expert_like_module_seen or is_expert_like

                if isinstance(child, torch.nn.Linear) and id(child) not in linear_ids:
                    linear_ids.add(id(child))
                    if _target_selected(full_name, include_patterns, exclude_patterns):
                        linear_names.append(full_name)
                        quantized_target_names.append(full_name)
                        backend_patterns.append(full_name)
                        if is_expert_like:
                            expert_coverage = True
                    else:
                        excluded_target_names.append(full_name)
                    continue

                direct_parameters = dict(child.named_parameters(recurse=False))
                matrix_parameter_names = tuple(
                    name
                    for name, parameter in direct_parameters.items()
                    if getattr(parameter, "ndim", 0) >= 2
                )
                if not matrix_parameter_names:
                    continue
                module_selected = _target_selected(
                    full_name,
                    include_patterns,
                    exclude_patterns,
                )
                selected_matrix_parameters = tuple(
                    name
                    for name in matrix_parameter_names
                    if module_selected
                    or _target_selected(
                        f"{full_name}.{name}",
                        include_patterns,
                        exclude_patterns,
                    )
                )
                if not selected_matrix_parameters:
                    excluded_target_names.extend(
                        f"{full_name}.{name}" for name in matrix_parameter_names
                    )
                    continue
                try:
                    mapping = adapter.map_fused_target(
                        full_name,
                        child,
                        direct_parameters,
                    )
                except ArchitectureAdapterError as exc:
                    raise DiscoveryError(
                        f"architecture adapter {adapter.adapter_id!r} rejected target "
                        f"{full_name!r}: {exc}"
                    ) from exc
                if mapping is None:
                    raise DiscoveryError(
                        f"architecture adapter {adapter.adapter_id!r} does not support "
                        f"custom matrix target {full_name!r} with parameters "
                        f"{list(selected_matrix_parameters)}"
                    )
                parameter_names = tuple(
                    f"{full_name}.{name}" for name in mapping.runtime_parameter_names
                )
                missing = tuple(
                    name
                    for name in mapping.runtime_parameter_names
                    if name not in direct_parameters
                )
                if missing:
                    raise DiscoveryError(
                        f"architecture adapter {adapter.adapter_id!r} mapped missing "
                        f"parameters at {full_name!r}: {list(missing)}"
                    )
                selected = tuple(
                    name
                    for name in parameter_names
                    if (
                        module_selected
                        or any(fnmatchcase(name, pattern) for pattern in include_patterns)
                    )
                    and not any(fnmatchcase(name, pattern) for pattern in exclude_patterns)
                )
                if len(selected) != len(parameter_names):
                    excluded_target_names.extend(
                        name for name in parameter_names if name not in selected
                    )
                    if selected:
                        raise DiscoveryError(
                            f"architecture adapter {adapter.adapter_id!r} requires atomic "
                            f"fused target {full_name!r}; include/exclude selected only "
                            f"{list(selected)}"
                        )
                    continue
                expert_targets.append(
                    _ExpertTarget(
                        module_name=full_name,
                        parameter_names=mapping.runtime_parameter_names,
                        backend_module_names=mapping.backend_module_names,
                        backend_pattern=mapping.backend_pattern,
                        activation_kwarg=mapping.activation_kwarg,
                    )
                )
                quantized_target_names.append(full_name)
                backend_patterns.append(mapping.backend_pattern)
                group_warnings.append(f"fused_expert_parameters={','.join(parameter_names)}")
                group_warnings.append(
                    "fused_activation_proxy=module_input_a8_only; internal projection "
                    "activations such as down_proj input are not intercepted"
                )
                group_warnings.append(
                    f"backend_mapping={full_name}->{mapping.backend_pattern} "
                    f"({len(mapping.backend_module_names)} Linear modules)"
                )
                expert_like_module_seen = True
                expert_coverage = True

        if not quantized_target_names:
            raise DiscoveryError(
                f"group {group_id!r} contains no supported quantization targets after filters"
            )
        if not expert_coverage:
            if expert_like_module_seen:
                detail = (
                    f"group {group_id!r} contains expert-like modules but has no "
                    "confirmed routed-expert target coverage"
                )
                if strict_targets:
                    raise DiscoveryError(detail)
            else:
                detail = f"group {group_id!r} appears dense; no routed-expert target was expected"
            group_warnings.append(detail)
        elif expert_like_module_seen and not expert_targets:
            group_warnings.append(
                "expert coverage comes from nn.Linear modules; no fused 3D expert target found"
            )
        group_warnings.append(f"block_roots={','.join(module_names)}")
        if excluded_target_names:
            group_warnings.append(
                "excluded_targets=" + ",".join(_ordered_unique(iter(excluded_target_names)))
            )

        parameter_count = sum(int(modules[name].weight.numel()) for name in linear_names)
        for target in expert_targets:
            expert_module = modules[target.module_name]
            for parameter_name in target.parameter_names:
                parameter_count += int(getattr(expert_module, parameter_name).numel())
        result.append(
            _RuntimeGroup(
                inventory=GroupInventory(
                    group_id=group_id,
                    module_names=tuple(quantized_target_names),
                    parameter_count=parameter_count,
                    warnings=tuple(group_warnings),
                    backend_patterns=tuple(backend_patterns),
                ),
                block_module_names=tuple(module_names),
                linear_module_names=tuple(linear_names),
                expert_targets=tuple(expert_targets),
            )
        )
    if not result:
        raise DiscoveryError("no Transformer groups were provided or discovered")
    return tuple(result)


def _normalize_explicit_groups(
    explicit: Mapping[str, str | Sequence[str]] | Sequence[str],
) -> dict[str, tuple[str, ...]]:
    if isinstance(explicit, Mapping):
        normalized: dict[str, tuple[str, ...]] = {}
        for group_id, names in explicit.items():
            if isinstance(names, str):
                normalized[str(group_id)] = (names,)
            else:
                normalized[str(group_id)] = tuple(str(name) for name in names)
        return normalized
    if isinstance(explicit, str):
        return {explicit: (explicit,)}
    return {str(name): (str(name),) for name in explicit}


def _normalize_samples(samples: Sequence[TextSample | str]) -> tuple[TextSample, ...]:
    normalized: list[TextSample] = []
    for index, sample in enumerate(samples):
        if isinstance(sample, TextSample):
            normalized.append(sample)
        elif isinstance(sample, str) and sample.strip():
            normalized.append(TextSample(sample_id=str(index), text=sample))
        else:
            raise TypeError("samples must contain TextSample objects or non-empty strings")
    if not normalized:
        raise ValueError("samples must not be empty")
    if len({sample.sample_id for sample in normalized}) != len(normalized):
        raise ValueError("sample IDs must be unique")
    return tuple(normalized)


def _normalize_precision(value: PrecisionFormat | str) -> PrecisionFormat:
    raw = getattr(value, "value", value)
    try:
        normalized = PrecisionFormat(str(raw).lower())
    except ValueError as exc:
        allowed = ", ".join(
            item.value
            for item in (
                PrecisionFormat.BF16,
                PrecisionFormat.W4A8,
                PrecisionFormat.W8A8,
            )
        )
        raise ValueError(f"unsupported proxy precision {raw!r}; expected one of {allowed}") from exc
    if normalized not in {
        PrecisionFormat.BF16,
        PrecisionFormat.W4A8,
        PrecisionFormat.W8A8,
    }:
        raise ValueError(
            f"unsupported proxy precision {raw!r}; W4A4 activation proxy is not implemented"
        )
    return normalized


def _group_inventory_to_dict(group: GroupInventory) -> dict[str, Any]:
    return {
        "group_id": group.group_id,
        "module_names": list(group.module_names),
        "parameter_count": group.parameter_count,
        "kind": group.kind,
        "warnings": list(group.warnings),
        "backend_patterns": list(group.effective_backend_patterns),
    }


def _measurement_to_dict(measurement: InterventionMeasurement) -> dict[str, Any]:
    return {
        "group_id": measurement.group_id,
        "precision": measurement.precision.value,
        "context_id": measurement.context_id,
        "module_names": list(measurement.module_names),
        "parameter_count": measurement.parameter_count,
        "baseline_nll": measurement.baseline_nll,
        "candidate_nll": measurement.candidate_nll,
        "delta_nll": measurement.delta_nll,
        "kl_divergence": measurement.kl_divergence,
        "token_count": measurement.token_count,
        "elapsed_seconds": measurement.elapsed_seconds,
        "quantizer_id": measurement.quantizer_id,
        "warnings": list(measurement.warnings),
    }


def _dataset_hash(samples: Sequence[TextSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        payload = json.dumps(sample.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tokenizer_identity(tokenizer: Any) -> dict[str, str | None]:
    name_or_path = getattr(tokenizer, "name_or_path", None)
    return {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": str(name_or_path) if name_or_path is not None else None,
    }


def _selected_target_inventory(
    model: Any,
    groups: Sequence[_RuntimeGroup],
) -> tuple[dict[str, Any], ...]:
    """Return canonical group/target metadata used by the experiment fingerprint."""

    modules = dict(model.named_modules())
    inventory: list[dict[str, Any]] = []
    for group in groups:
        targets: list[dict[str, Any]] = []
        for module_name in group.linear_module_names:
            module = modules[module_name]
            weight = module.weight
            targets.append(
                {
                    "kind": "linear_weight",
                    "name": f"{module_name}.weight",
                    "module_class": f"{type(module).__module__}.{type(module).__qualname__}",
                    "shape": list(weight.shape),
                    "dtype": str(weight.dtype),
                    "device": str(weight.device),
                    "contiguous": bool(weight.is_contiguous()),
                }
            )
        for expert in group.expert_targets:
            module = modules[expert.module_name]
            for parameter_name in expert.parameter_names:
                parameter = getattr(module, parameter_name)
                targets.append(
                    {
                        "kind": "fused_expert_weight",
                        "name": f"{expert.module_name}.{parameter_name}",
                        "module_class": (f"{type(module).__module__}.{type(module).__qualname__}"),
                        "shape": list(parameter.shape),
                        "dtype": str(parameter.dtype),
                        "device": str(parameter.device),
                        "contiguous": bool(parameter.is_contiguous()),
                    }
                )
        inventory.append(
            {
                "group_id": group.inventory.group_id,
                "block_module_names": list(group.block_module_names),
                "quantized_module_names": list(group.inventory.module_names),
                "backend_patterns": list(group.inventory.effective_backend_patterns),
                "parameter_count": group.inventory.parameter_count,
                "targets": targets,
            }
        )
    return tuple(inventory)


def _require_bf16_target_anchor(
    torch: Any,
    target_inventory: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed unless every selected quantization target is truly BF16."""

    expected = str(torch.bfloat16)
    mismatches = [
        f"{target['name']}={target['dtype']}"
        for group in target_inventory
        for target in group["targets"]
        if target["dtype"] != expected
    ]
    if not mismatches:
        return
    preview = ", ".join(mismatches[:8])
    remaining = len(mismatches) - 8
    suffix = f", ... and {remaining} more" if remaining > 0 else ""
    raise TorchAnalysisError(
        "BF16 protection requires every selected quantization target weight to "
        f"be torch.bfloat16; found {preview}{suffix}. Load/cast the reference "
        "targets as BF16 before analysis instead of treating another dtype as BF16."
    )


def _validate_patterns(name: str, values: Sequence[str], *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(values, str):
        patterns = (values,)
    else:
        patterns = tuple(values)
    if not patterns and not allow_empty:
        raise ValueError(f"{name} must contain at least one pattern")
    if any(not isinstance(pattern, str) or not pattern for pattern in patterns):
        raise ValueError(f"{name} must contain only non-empty strings")
    return patterns


def _effective_exclude_patterns(
    configured: Sequence[str],
    *,
    adapter: ArchitectureAdapter | None = None,
) -> tuple[str, ...]:
    """Merge caller filters with non-removable runtime safety exclusions."""

    required = (
        adapter.nonremovable_exclude_patterns if adapter is not None else DEFAULT_LINEAR_EXCLUDE
    )
    return tuple(dict.fromkeys((*required, *configured)))


def _target_selected(
    name: str,
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in include_patterns) and not any(
        fnmatchcase(name, pattern) for pattern in exclude_patterns
    )


def _uniform_indexes(total: int, count: int) -> set[int]:
    if count >= total:
        return set(range(total))
    if count == 1:
        return {total // 2}
    return {round(position * (total - 1) / (count - 1)) for position in range(count)}


def _select_runtime_groups(
    groups: tuple[_RuntimeGroup, ...],
    max_groups: int | None,
) -> tuple[_RuntimeGroup, ...]:
    """Select a depth-spanning pilot while retaining total-coverage evidence."""

    if max_groups is None or max_groups >= len(groups):
        return groups
    if max_groups < 1:
        raise ValueError("max_groups must be at least one")
    selected_indexes = _uniform_indexes(len(groups), max_groups)
    selection_warning = (
        f"pilot selection chose {len(selected_indexes)} of {len(groups)} blocks "
        f"at indexes {sorted(selected_indexes)}; derived YAML is forbidden"
    )
    selected: list[_RuntimeGroup] = []
    for index, group in enumerate(groups):
        if index not in selected_indexes:
            continue
        warnings = tuple(
            warning
            for warning in group.inventory.warnings
            if not warning.startswith("automatic discovery selected all ")
        )
        selected.append(
            replace(
                group,
                inventory=replace(
                    group.inventory,
                    warnings=(*warnings, selection_warning),
                ),
            )
        )
    return tuple(selected)


def _ordered_unique(values: Iterator[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _placement_from_loaded_model(
    model: Any,
    *,
    requested_device: str | None,
    hf_auto: bool,
) -> ModelPlacement:
    input_device = str(_model_input_device(_require_torch(), model))
    raw_map = getattr(model, "hf_device_map", None)
    device_map = _normalize_device_map(raw_map)
    if hf_auto:
        if not device_map:
            raise TorchAnalysisError(
                "device='auto' requested Hugging Face module sharding, but the loaded "
                "model exposes no hf_device_map; use an explicit single device or a "
                "custom loader returning ModelBundle(..., placement=ModelPlacement(...))"
            )
        return ModelPlacement(
            strategy="hf_device_map",
            input_device=input_device,
            device_map=device_map,
        )
    return ModelPlacement(
        strategy="single_device",
        input_device=str(requested_device or input_device),
        device_map=device_map,
    )


def _resolve_runtime_placement(
    torch: Any,
    model: Any,
    target_inventory: Sequence[Mapping[str, Any]],
    *,
    declared: ModelPlacement | None,
    requested_device: str | None,
) -> _ResolvedPlacement:
    """Validate the resident one-process placement used by interventions."""

    distributed = getattr(torch, "distributed", None)
    is_initialized = getattr(distributed, "is_initialized", None)
    get_world_size = getattr(distributed, "get_world_size", None)
    if (
        callable(is_initialized)
        and bool(is_initialized())
        and callable(get_world_size)
        and int(get_world_size()) > 1
    ):
        raise TorchAnalysisError(
            "distributed parameter execution (TP/EP/FSDP/ZeRO or multi-process "
            "pipeline) is outside the P0 placement contract"
        )

    model_devices = tuple(sorted({str(parameter.device) for parameter in model.parameters()}))
    if not model_devices:
        model_devices = (str(_model_input_device(torch, model)),)
    requested_backend = _requested_auto_backend(requested_device)
    if requested_backend is not None:
        actual_backends = {
            _device_kind(device)
            for device in model_devices
            if _device_kind(device) not in {"", "cpu", "disk", "meta"}
        }
        if actual_backends != {requested_backend}:
            raise TorchAnalysisError(
                f"device={requested_device!r} requires the loaded model to reside on "
                f"{requested_backend}; found backends {sorted(actual_backends) or ['none']}"
            )
    target_devices = tuple(
        sorted(
            {
                str(target["device"])
                for group in target_inventory
                for target in group["targets"]
            }
        )
    )
    input_device = str(_model_input_device(torch, model))
    raw_map = getattr(model, "hf_device_map", None)
    inferred_map = _normalize_device_map(raw_map)

    placement = declared
    if placement is None:
        if len(model_devices) > 1:
            raise TorchAnalysisError(
                "the loaded model spans multiple devices but no placement contract was "
                "declared; a custom sharded loader must return ModelBundle with "
                "ModelPlacement(strategy='pre_sharded', input_device=...)"
            )
        placement = ModelPlacement(
            strategy="single_device",
            input_device=input_device,
            device_map=inferred_map,
        )
    elif placement.strategy == "hf_device_map" and not placement.device_map:
        if not inferred_map:
            raise TorchAnalysisError(
                "hf_device_map placement requires the loaded model to expose a non-empty "
                "hf_device_map"
            )
        placement = replace(placement, device_map=inferred_map)
    elif placement.strategy == "pre_sharded" and not placement.device_map:
        raise TorchAnalysisError(
            "pre_sharded placement requires a non-empty device_map that covers the "
            "loaded model parameters"
        )

    declared_input = placement.input_device or input_device
    if _canonical_device(declared_input) != _canonical_device(input_device):
        raise TorchAnalysisError(
            "placement input_device does not match the loaded input embedding device: "
            f"declared={declared_input!r}, embedding={input_device!r}"
        )
    if requested_device is not None and not _is_auto_device(requested_device) and (
        _canonical_device(str(requested_device)) != _canonical_device(input_device)
    ):
        raise TorchAnalysisError(
            "configured device must be the input embedding device for a resident model: "
            f"configured={requested_device!r}, embedding={input_device!r}"
        )

    forbidden_model_devices = tuple(
        device for device in model_devices if _device_kind(device) in {"meta"}
    )
    forbidden_map_devices = tuple(
        device for _name, device in placement.device_map if _device_kind(device) in {"disk", "meta"}
    )
    if forbidden_model_devices or forbidden_map_devices:
        raise TorchAnalysisError(
            "QuantPlan interventions require resident weights; meta/disk offload is "
            "unsupported. offending devices="
            f"{list(dict.fromkeys((*forbidden_model_devices, *forbidden_map_devices)))}"
        )

    if placement.device_map:
        _validate_resident_device_map(model, placement.device_map, model_devices)

    if placement.strategy == "single_device" and len(model_devices) != 1:
        raise TorchAnalysisError(
            "single_device placement found model parameters on multiple devices: "
            f"{list(model_devices)}"
        )
    if placement.strategy in {"hf_device_map", "pre_sharded"}:
        cpu_or_disk = tuple(
            device
            for device in (*model_devices, *(value for _, value in placement.device_map))
            if _device_kind(device) in {"cpu", "disk", "meta"}
        )
        if cpu_or_disk:
            raise TorchAnalysisError(
                "P0 sharded analysis supports accelerator-resident modules only; "
                "CPU/disk/meta offload is not supported. offending devices="
                f"{list(dict.fromkeys(cpu_or_disk))}"
            )
        accelerator_kinds = {
            _device_kind(device) for device in model_devices if _device_kind(device)
        }
        if len(accelerator_kinds) != 1:
            raise TorchAnalysisError(
                "sharded placement must use one accelerator backend; found "
                f"{sorted(accelerator_kinds)}"
            )

    if not target_devices:
        raise TorchAnalysisError("selected target inventory contains no weight devices")
    noncontiguous = tuple(
        str(target["name"])
        for group in target_inventory
        for target in group["targets"]
        if not bool(target["contiguous"])
    )
    if noncontiguous:
        raise TorchAnalysisError(
            "selected quantization target weights must be contiguous for in-place "
            f"fake quantization; found {list(noncontiguous[:8])}"
        )
    if any(_device_kind(device) in {"cpu", "disk", "meta"} for device in target_devices) and (
        placement.strategy != "single_device" or target_devices != ("cpu",)
    ):
        raise TorchAnalysisError(
            "selected quantization targets must be accelerator-resident for sharded "
            f"analysis; found {list(target_devices)}"
        )

    evidence = {
        "schema": PLACEMENT_SCHEMA,
        "strategy": placement.strategy,
        "input_device": input_device,
        "model_devices": list(model_devices),
        "target_devices": list(target_devices),
        "device_map": dict(placement.device_map),
    }
    return _ResolvedPlacement(
        placement=replace(placement, input_device=input_device),
        input_device=input_device,
        model_devices=model_devices,
        target_devices=target_devices,
        sha256=_stable_json_hash(evidence),
    )


def _normalize_device_map(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TorchAnalysisError("model.hf_device_map must be a mapping when present")
    return tuple(sorted((str(name), str(device)) for name, device in value.items()))


def _validate_resident_device_map(
    model: Any,
    device_map: Sequence[tuple[str, str]],
    model_devices: Sequence[str],
) -> None:
    """Verify a module device map against the parameters that were actually loaded."""

    module_names = {name for name, _module in model.named_modules()}
    unknown_modules = tuple(name for name, _device in device_map if name not in module_names)
    if unknown_modules:
        raise TorchAnalysisError(
            "placement device_map contains module paths absent from named_modules: "
            f"{list(unknown_modules[:8])}"
        )

    accelerator_kinds = {
        _device_kind(device)
        for device in model_devices
        if _device_kind(device) not in {"", "cpu", "disk", "meta"}
    }
    numeric_backend = next(iter(accelerator_kinds)) if len(accelerator_kinds) == 1 else None
    uncovered: list[str] = []
    mismatches: list[str] = []
    ordered_map = sorted(device_map, key=lambda item: len(item[0]), reverse=True)
    for parameter_name, parameter in model.named_parameters():
        match = next(
            (
                (module_name, declared_device)
                for module_name, declared_device in ordered_map
                if not module_name
                or parameter_name == module_name
                or parameter_name.startswith(f"{module_name}.")
            ),
            None,
        )
        if match is None:
            uncovered.append(parameter_name)
            continue
        module_name, declared_device = match
        declared_canonical = _canonical_device_map_value(declared_device, numeric_backend)
        actual_canonical = _canonical_device(str(parameter.device))
        if declared_canonical != actual_canonical:
            mismatches.append(
                f"{parameter_name}: map[{module_name!r}]={declared_device!r}, "
                f"actual={str(parameter.device)!r}"
            )
    if uncovered:
        raise TorchAnalysisError(
            "placement device_map does not cover every loaded parameter; missing "
            f"{list(uncovered[:8])}"
        )
    if mismatches:
        raise TorchAnalysisError(
            "placement device_map disagrees with loaded parameter devices: "
            f"{mismatches[:8]}"
        )


def _canonical_device_map_value(value: str, numeric_backend: str | None) -> str:
    normalized = value.strip().lower()
    if normalized.isdecimal():
        if numeric_backend is None:
            raise TorchAnalysisError(
                "numeric device_map entries require exactly one observable accelerator backend"
            )
        return f"{numeric_backend}:{int(normalized)}"
    return _canonical_device(normalized)


def _canonical_device(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"cuda", "npu", "xpu"}:
        return f"{normalized}:0"
    return normalized


def _is_auto_device(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"auto", "npu:auto", "cuda:auto", "xpu:auto"}


def _requested_auto_backend(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    backend, separator, policy = normalized.partition(":")
    if separator == ":" and policy == "auto" and backend in {"npu", "cuda", "xpu"}:
        return backend
    return None


def _device_kind(value: str) -> str:
    normalized = value.strip().lower()
    if normalized.isdecimal():
        # HF device maps may use integer accelerator ordinals. Actual parameter
        # devices remain authoritative for the backend kind.
        return ""
    return normalized.partition(":")[0]


def _model_input_device(torch: Any, model: Any) -> Any:
    try:
        embedding = model.get_input_embeddings()
        if embedding is not None:
            return embedding.weight.device
    except (AttributeError, NotImplementedError):
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@contextmanager
def _evaluation_mode(model: Any) -> Iterator[None]:
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise TorchDependencyError(
            "PyTorch is optional but required for quantplan.runtime; install a "
            "PyTorch build appropriate for the target CPU/NPU/GPU environment"
        ) from exc


def _require_transformers() -> Any:
    try:
        return importlib.import_module("transformers")
    except ModuleNotFoundError as exc:
        if exc.name != "transformers":
            raise
        raise TransformersDependencyError(
            "Transformers is required by the default Hugging Face loader; install "
            "transformers or provide loader_entrypoint='module:function'"
        ) from exc


def _require_torch_npu() -> Any:
    try:
        return importlib.import_module("torch_npu")
    except (ImportError, OSError) as exc:
        raise TorchDependencyError(
            "an NPU device was requested but torch_npu could not be imported; install "
            "the torch_npu build matched to PyTorch and CANN"
        ) from exc


def _resolve_torch_dtype(torch: Any, value: Any) -> Any:
    if value is None or not isinstance(value, str):
        return value
    normalized = value.lower()
    if normalized == "auto":
        # Transformers interprets this sentinel as the checkpoint/config dtype.
        # Omitting the argument instead commonly falls back to FP32 and can make
        # large-model loading unexpectedly expensive.
        return "auto"
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported torch_dtype {value!r}")
    return aliases[normalized]


def _resolve_entrypoint(value: str | Callable[..., Any]) -> Callable[..., Any]:
    if callable(value):
        return value
    if ":" not in value:
        raise ValueError("loader_entrypoint must use 'module:function' syntax")
    module_name, attribute_name = value.split(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("loader_entrypoint must use 'module:function' syntax")
    module = importlib.import_module(module_name)
    loader = getattr(module, attribute_name, None)
    if not callable(loader):
        raise TypeError(f"loader entrypoint {value!r} is not callable")
    return loader


def _entrypoint_name(value: str | Callable[..., Any]) -> str:
    if isinstance(value, str):
        return value
    return f"{value.__module__}:{value.__qualname__}"


def _coerce_model_bundle(
    value: Any,
    *,
    model_id: str,
    revision: str | None,
    loader_name: str,
) -> ModelBundle:
    if isinstance(value, ModelBundle):
        return value
    placement: ModelPlacement | None = None
    if isinstance(value, Mapping):
        model = value.get("model")
        tokenizer = value.get("tokenizer")
        raw_placement = value.get("placement")
        if isinstance(raw_placement, ModelPlacement):
            placement = raw_placement
        elif isinstance(raw_placement, Mapping):
            raw_map = raw_placement.get("device_map", {})
            placement = ModelPlacement(
                strategy=str(raw_placement.get("strategy", "")),
                input_device=(
                    str(raw_placement["input_device"])
                    if raw_placement.get("input_device") is not None
                    else None
                ),
                device_map=_normalize_device_map(raw_map),
            )
        elif raw_placement is not None:
            raise TypeError("custom loader placement must be ModelPlacement, a mapping, or null")
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        model, tokenizer = value
    else:
        raise TypeError("custom loader must return ModelBundle, (model, tokenizer), or a mapping")
    if model is None or tokenizer is None:
        raise TypeError("custom loader result must contain non-null model and tokenizer")
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        revision=revision,
        loader_name=loader_name,
        placement=placement,
    )
