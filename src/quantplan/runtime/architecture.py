"""Versioned architecture adapters for runtime discovery and backend mapping.

Adapters contain architecture knowledge only.  The analysis runtime remains
responsible for temporary interventions, metrics and cleanup; planning and YAML
compilation consume the architecture-neutral inventory produced by the runtime.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from importlib import metadata as importlib_metadata
from threading import RLock
from typing import Any, Protocol, runtime_checkable

BASE_SAFETY_EXCLUDES = (
    "*router*",
    "*.mlp.gate",
    "*mtp*",
    "*vision*",
    "*visual*",
)
GENERIC_ADAPTER_ID = "generic.nn_linear/v1"
QWEN35_MODELSLIM_COMMIT = "376e2fad43bc999f9a49b48bdbc3eef06ffdd83a"
QWEN35_MOE_ADAPTER_ID = f"modelslim.qwen3_5_moe/{QWEN35_MODELSLIM_COMMIT}"
ARCHITECTURE_ADAPTER_ENTRY_POINT = "quantplan.architecture_adapters"


class ArchitectureAdapterError(RuntimeError):
    """Base error for adapter registration, resolution and target mapping."""


class ArchitectureAdapterRegistrationError(ArchitectureAdapterError):
    """Raised when an adapter declaration is invalid or duplicated."""


class ArchitectureAdapterResolutionError(ArchitectureAdapterError):
    """Raised when adapter selection is unknown or ambiguous."""


class ArchitectureTargetMappingError(ArchitectureAdapterError):
    """Raised when a custom/fused target cannot be mapped safely."""


@dataclass(frozen=True)
class FusedTargetMapping:
    """Runtime parameters and their verified post-adapter ModelSlim names."""

    runtime_parameter_names: tuple[str, ...]
    backend_module_names: tuple[str, ...]
    backend_pattern: str
    activation_kwarg: str = "hidden_states"

    def __post_init__(self) -> None:
        if not self.runtime_parameter_names or any(
            not name for name in self.runtime_parameter_names
        ):
            raise ValueError("runtime_parameter_names must contain non-empty names")
        if not self.backend_module_names or any(
            not name for name in self.backend_module_names
        ):
            raise ValueError("backend_module_names must contain non-empty names")
        if not self.backend_pattern:
            raise ValueError("backend_pattern must not be empty")
        if not self.activation_kwarg:
            raise ValueError("activation_kwarg must not be empty")
        if len(set(self.runtime_parameter_names)) != len(self.runtime_parameter_names):
            raise ValueError("runtime_parameter_names must not contain duplicates")
        if len(set(self.backend_module_names)) != len(self.backend_module_names):
            raise ValueError("backend_module_names must not contain duplicates")
        if any(
            any(char in name for char in "*?[]{}")
            for name in self.backend_module_names
        ):
            raise ValueError("backend_module_names must be exact names, not patterns")
        if "{" in self.backend_pattern or "}" in self.backend_pattern:
            raise ValueError("backend_pattern cannot use an unpinned brace dialect")
        if not all(
            fnmatchcase(name, self.backend_pattern) for name in self.backend_module_names
        ):
            raise ValueError("backend_pattern must cover every backend_module_name")


@runtime_checkable
class ArchitectureAdapter(Protocol):
    """Minimal plug-in contract needed by the core analysis runtime."""

    adapter_id: str
    fallback: bool
    preferred_block_paths: tuple[str, ...]
    nonremovable_exclude_patterns: tuple[str, ...]
    allow_block_path_fallback: bool

    def matches(self, model: Any, config: Any | None = None) -> bool: ...

    def map_fused_target(
        self,
        module_name: str,
        module: Any,
        direct_parameters: Mapping[str, Any],
    ) -> FusedTargetMapping | None: ...


@dataclass(frozen=True)
class GenericLinearAdapter:
    """Fallback for dense models composed only of standard ``nn.Linear`` targets."""

    adapter_id: str = GENERIC_ADAPTER_ID
    fallback: bool = True
    preferred_block_paths: tuple[str, ...] = (
        "model.language_model.layers",
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
        "transformer.blocks",
        "blocks",
        "layers",
    )
    nonremovable_exclude_patterns: tuple[str, ...] = BASE_SAFETY_EXCLUDES
    allow_block_path_fallback: bool = True

    def matches(self, model: Any, config: Any | None = None) -> bool:
        del model, config
        return True

    def map_fused_target(
        self,
        module_name: str,
        module: Any,
        direct_parameters: Mapping[str, Any],
    ) -> FusedTargetMapping | None:
        del module_name, module, direct_parameters
        return None


@dataclass(frozen=True)
class Qwen35MoeModelSlimAdapter:
    """Qwen3.5-MoE mapping verified against one audited ModelSlim commit."""

    adapter_id: str = QWEN35_MOE_ADAPTER_ID
    fallback: bool = False
    preferred_block_paths: tuple[str, ...] = (
        "model.language_model.layers",
        "model.layers",
    )
    nonremovable_exclude_patterns: tuple[str, ...] = BASE_SAFETY_EXCLUDES
    allow_block_path_fallback: bool = False

    def matches(self, model: Any, config: Any | None = None) -> bool:
        resolved_config = config if config is not None else getattr(model, "config", None)
        model_type = str(getattr(resolved_config, "model_type", "") or "").lower()
        return model_type in ("qwen3_5_moe", "qwen3_5_moe_text")

    def map_fused_target(
        self,
        module_name: str,
        module: Any,
        direct_parameters: Mapping[str, Any],
    ) -> FusedTargetMapping | None:
        del module
        expected = ("gate_up_proj", "down_proj")
        present = tuple(name for name in expected if name in direct_parameters)
        if not present:
            return None
        missing = tuple(name for name in expected if name not in direct_parameters)
        wrong_rank = tuple(
            name
            for name in present
            if getattr(direct_parameters[name], "ndim", None) != 3
        )
        if missing or wrong_rank:
            raise ArchitectureTargetMappingError(
                f"unsupported Qwen3.5 fused target {module_name}: "
                f"missing={list(missing)}, non_3d={list(wrong_rank)}"
            )
        expert_counts = {
            int(direct_parameters[name].shape[0]) for name in expected
        }
        if len(expert_counts) != 1 or next(iter(expert_counts)) < 1:
            raise ArchitectureTargetMappingError(
                f"Qwen3.5 fused target {module_name} has inconsistent expert dimensions"
            )
        expert_count = next(iter(expert_counts))
        backend_names = tuple(
            f"{module_name}.{expert_index}.{projection}"
            for expert_index in range(expert_count)
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
        return FusedTargetMapping(
            runtime_parameter_names=expected,
            backend_module_names=backend_names,
            backend_pattern=f"{module_name}.*",
        )


class ArchitectureAdapterRegistry:
    """Thread-safe registry with exact-adapter precedence and one fallback."""

    def __init__(self, adapters: Iterable[ArchitectureAdapter] = ()) -> None:
        self._lock = RLock()
        self._adapters: dict[str, ArchitectureAdapter] = {}
        self._entry_points_loaded = False
        for adapter in adapters:
            self.register(adapter)

    def register(
        self,
        adapter: ArchitectureAdapter,
        *,
        replace: bool = False,
    ) -> None:
        _validate_adapter(adapter)
        with self._lock:
            if adapter.adapter_id in self._adapters and not replace:
                raise ArchitectureAdapterRegistrationError(
                    f"architecture adapter {adapter.adapter_id!r} is already registered"
                )
            self._adapters[adapter.adapter_id] = adapter

    def unregister(self, adapter_id: str) -> None:
        with self._lock:
            if adapter_id not in self._adapters:
                raise ArchitectureAdapterRegistrationError(
                    f"architecture adapter {adapter_id!r} is not registered"
                )
            del self._adapters[adapter_id]

    def adapters(self) -> tuple[ArchitectureAdapter, ...]:
        with self._lock:
            return tuple(self._adapters.values())

    def discover_entry_points(self, entries: Iterable[Any] | None = None) -> None:
        """Load installed adapter objects/factories exactly once and fail closed."""

        with self._lock:
            if entries is None and self._entry_points_loaded:
                return
            try:
                selected_entries = (
                    tuple(entries) if entries is not None else _installed_entry_points()
                )
            except Exception as exc:
                raise ArchitectureAdapterResolutionError(
                    "failed to enumerate architecture adapter entry points: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            staged: list[ArchitectureAdapter] = []
            staged_ids: set[str] = set()
            for entry in sorted(
                selected_entries,
                key=lambda item: (str(getattr(item, "name", "")), str(item)),
            ):
                label = str(getattr(entry, "name", entry))
                try:
                    loaded = entry.load()
                    adapter = (
                        loaded
                        if isinstance(loaded, ArchitectureAdapter)
                        and not isinstance(loaded, type)
                        else loaded()
                    )
                    _validate_adapter(adapter)
                except Exception as exc:
                    raise ArchitectureAdapterResolutionError(
                        f"failed to load architecture adapter entry point {label!r}: {exc}"
                    ) from exc
                if (
                    adapter.adapter_id in self._adapters
                    or adapter.adapter_id in staged_ids
                ):
                    raise ArchitectureAdapterResolutionError(
                        f"entry point {label!r} duplicates architecture adapter "
                        f"{adapter.adapter_id!r}"
                    )
                staged.append(adapter)
                staged_ids.add(adapter.adapter_id)
            for adapter in staged:
                self._adapters[adapter.adapter_id] = adapter
            if entries is None:
                self._entry_points_loaded = True

    def resolve(
        self,
        model: Any,
        config: Any | None = None,
        *,
        adapter_id: str | None = None,
    ) -> ArchitectureAdapter:
        adapters = self.adapters()
        if adapter_id is not None:
            selected = tuple(item for item in adapters if item.adapter_id == adapter_id)
            if not selected:
                raise ArchitectureAdapterResolutionError(
                    f"architecture adapter {adapter_id!r} is not registered"
                )
            adapter = selected[0]
            if not _adapter_matches(adapter, model, config):
                raise ArchitectureAdapterResolutionError(
                    f"architecture adapter {adapter_id!r} does not match the model/config"
                )
            return adapter

        matching: list[ArchitectureAdapter] = []
        for item in adapters:
            if _adapter_matches(item, model, config):
                matching.append(item)
        exact = tuple(item for item in matching if not item.fallback)
        if len(exact) > 1:
            ids = sorted(item.adapter_id for item in exact)
            raise ArchitectureAdapterResolutionError(
                f"ambiguous architecture adapters matched the model/config: {ids}"
            )
        if exact:
            return exact[0]
        fallbacks = tuple(item for item in matching if item.fallback)
        if len(fallbacks) > 1:
            ids = sorted(item.adapter_id for item in fallbacks)
            raise ArchitectureAdapterResolutionError(
                f"ambiguous fallback architecture adapters: {ids}"
            )
        if fallbacks:
            return fallbacks[0]
        raise ArchitectureAdapterResolutionError(
            "no registered architecture adapter matches the model/config"
        )


def _adapter_matches(
    adapter: ArchitectureAdapter,
    model: Any,
    config: Any | None,
) -> bool:
    try:
        matched = adapter.matches(model, config)
    except Exception as exc:
        raise ArchitectureAdapterResolutionError(
            f"architecture adapter {adapter.adapter_id!r} failed while matching: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(matched, bool):
        raise ArchitectureAdapterResolutionError(
            f"architecture adapter {adapter.adapter_id!r} returned a non-boolean "
            "match result"
        )
    return matched


def _validate_adapter(adapter: ArchitectureAdapter) -> None:
    if not isinstance(adapter, ArchitectureAdapter):
        raise ArchitectureAdapterRegistrationError(
            "adapter does not implement the ArchitectureAdapter protocol"
        )
    if not isinstance(adapter.adapter_id, str) or not adapter.adapter_id:
        raise ArchitectureAdapterRegistrationError("adapter_id must be a non-empty string")
    if not isinstance(adapter.fallback, bool):
        raise ArchitectureAdapterRegistrationError("fallback must be a boolean")
    if not isinstance(adapter.allow_block_path_fallback, bool):
        raise ArchitectureAdapterRegistrationError(
            "allow_block_path_fallback must be a boolean"
        )
    for field_name in ("preferred_block_paths", "nonremovable_exclude_patterns"):
        values = getattr(adapter, field_name)
        if not isinstance(values, tuple) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ArchitectureAdapterRegistrationError(
                f"{field_name} must be a tuple of non-empty strings"
            )


def _installed_entry_points() -> tuple[Any, ...]:
    discovered = importlib_metadata.entry_points()
    if hasattr(discovered, "select"):
        return tuple(discovered.select(group=ARCHITECTURE_ADAPTER_ENTRY_POINT))
    return tuple(discovered.get(ARCHITECTURE_ADAPTER_ENTRY_POINT, ()))


_GLOBAL_REGISTRY = ArchitectureAdapterRegistry(
    (Qwen35MoeModelSlimAdapter(), GenericLinearAdapter())
)


def register_architecture_adapter(
    adapter: ArchitectureAdapter,
    *,
    replace: bool = False,
) -> None:
    """Register a process-wide architecture plug-in."""

    _GLOBAL_REGISTRY.register(adapter, replace=replace)


def unregister_architecture_adapter(adapter_id: str) -> None:
    """Remove a process-wide plug-in, primarily for controlled integration tests."""

    _GLOBAL_REGISTRY.unregister(adapter_id)


def registered_architecture_adapters() -> tuple[ArchitectureAdapter, ...]:
    _GLOBAL_REGISTRY.discover_entry_points()
    return _GLOBAL_REGISTRY.adapters()


def resolve_architecture_adapter(
    model: Any,
    config: Any | None = None,
    *,
    adapter_id: str | None = None,
) -> ArchitectureAdapter:
    _GLOBAL_REGISTRY.discover_entry_points()
    return _GLOBAL_REGISTRY.resolve(model, config, adapter_id=adapter_id)


__all__ = [
    "ARCHITECTURE_ADAPTER_ENTRY_POINT",
    "BASE_SAFETY_EXCLUDES",
    "GENERIC_ADAPTER_ID",
    "QWEN35_MODELSLIM_COMMIT",
    "QWEN35_MOE_ADAPTER_ID",
    "ArchitectureAdapter",
    "ArchitectureAdapterError",
    "ArchitectureAdapterRegistrationError",
    "ArchitectureAdapterRegistry",
    "ArchitectureAdapterResolutionError",
    "ArchitectureTargetMappingError",
    "FusedTargetMapping",
    "GenericLinearAdapter",
    "Qwen35MoeModelSlimAdapter",
    "register_architecture_adapter",
    "registered_architecture_adapters",
    "resolve_architecture_adapter",
    "unregister_architecture_adapter",
]
