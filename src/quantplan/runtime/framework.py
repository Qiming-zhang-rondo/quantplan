"""Inference-framework plug-ins and operator resolution.

Architecture adapters describe model structure.  Framework adapters describe
which fused operator a concrete inference stack selects for that structure.
The core never assumes that a model family implies one kernel.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from quantplan.runtime.capability import CapabilitySnapshot

INFERENCE_FRAMEWORK_ADAPTER_ENTRY_POINT = "quantplan.inference_framework_adapters"
MANIFEST_FRAMEWORK_ADAPTER_ID = "manifest.inference_framework/v1"
ANALYSIS_ONLY_FRAMEWORK_ADAPTER_ID = "generic.analysis_only/v1"


class FrameworkAdapterError(RuntimeError):
    """Base error for framework adapter registration and resolution."""


class FrameworkAdapterRegistrationError(FrameworkAdapterError):
    """A framework adapter declaration is invalid or duplicated."""


class FrameworkAdapterResolutionError(FrameworkAdapterError):
    """No unique framework adapter can resolve the requested stack."""


@dataclass(frozen=True)
class FrameworkRequest:
    requested_framework: str
    architecture_adapter_id: str
    model_type: str
    hardware_profile: str
    capability_snapshot: CapabilitySnapshot | None = None

    def __post_init__(self) -> None:
        for name in (
            "requested_framework",
            "architecture_adapter_id",
            "model_type",
            "hardware_profile",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class OperatorResolution:
    framework_adapter_id: str
    framework_name: str
    framework_version: str | None
    framework_git_commit: str | None
    operator: str
    status: str
    selection_basis: str
    runtime_selected: bool
    domain_ids: tuple[str, ...] = ()
    fallback_modes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def verified(self) -> bool:
        return self.status == "verified" and self.runtime_selected

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework_adapter_id": self.framework_adapter_id,
            "framework_name": self.framework_name,
            "framework_version": self.framework_version,
            "framework_git_commit": self.framework_git_commit,
            "operator": self.operator,
            "status": self.status,
            "selection_basis": self.selection_basis,
            "runtime_selected": self.runtime_selected,
            "domain_ids": list(self.domain_ids),
            "fallback_modes": list(self.fallback_modes),
            "reasons": list(self.reasons),
        }


@runtime_checkable
class InferenceFrameworkAdapter(Protocol):
    adapter_id: str
    fallback: bool

    def matches(self, request: FrameworkRequest) -> bool: ...

    def resolve(self, request: FrameworkRequest) -> OperatorResolution: ...


@dataclass(frozen=True)
class ManifestInferenceFrameworkAdapter:
    """Resolve an operator from a target-environment capability snapshot."""

    adapter_id: str = MANIFEST_FRAMEWORK_ADAPTER_ID
    fallback: bool = False

    def matches(self, request: FrameworkRequest) -> bool:
        snapshot = request.capability_snapshot
        return snapshot is not None and request.requested_framework.lower() in {
            "auto",
            snapshot.backend_name.lower(),
        } and snapshot.matches_model(
            request.model_type,
            request.architecture_adapter_id,
        )

    def resolve(self, request: FrameworkRequest) -> OperatorResolution:
        snapshot = request.capability_snapshot
        if snapshot is None:
            raise FrameworkAdapterResolutionError("manifest adapter requires a snapshot")
        fallback_modes = tuple(
            dict.fromkeys(
                mode
                for domain in snapshot.domains
                for mode in domain.fallback_modes
            )
        )
        reasons = () if snapshot.known else ("capability snapshot is not fully verified",)
        return OperatorResolution(
            framework_adapter_id=self.adapter_id,
            framework_name=snapshot.backend_name,
            framework_version=snapshot.backend_version,
            framework_git_commit=snapshot.backend_git_commit,
            operator=snapshot.operator,
            status="capability_declared" if snapshot.known else "unknown",
            selection_basis="capability_manifest",
            runtime_selected=False,
            domain_ids=tuple(domain.domain_id for domain in snapshot.domains),
            fallback_modes=fallback_modes,
            reasons=reasons,
        )


@dataclass(frozen=True)
class AnalysisOnlyFrameworkAdapter:
    """Fallback that permits evidence collection but claims no kernel selection."""

    adapter_id: str = ANALYSIS_ONLY_FRAMEWORK_ADAPTER_ID
    fallback: bool = True

    def matches(self, request: FrameworkRequest) -> bool:
        del request
        return True

    def resolve(self, request: FrameworkRequest) -> OperatorResolution:
        framework = request.requested_framework.lower()
        return OperatorResolution(
            framework_adapter_id=self.adapter_id,
            framework_name="unknown" if framework == "auto" else framework,
            framework_version=None,
            framework_git_commit=None,
            operator="unknown",
            status="analysis_only",
            selection_basis="analysis_only_fallback",
            runtime_selected=False,
            reasons=("no verified inference-framework capability adapter matched",),
        )


class InferenceFrameworkAdapterRegistry:
    """Thread-safe exact-first registry for inference-framework plug-ins."""

    def __init__(self, adapters: Iterable[InferenceFrameworkAdapter] = ()) -> None:
        self._lock = RLock()
        self._adapters: dict[str, InferenceFrameworkAdapter] = {}
        self._entry_points_loaded = False
        for adapter in adapters:
            self.register(adapter)

    def register(
        self,
        adapter: InferenceFrameworkAdapter,
        *,
        replace: bool = False,
    ) -> None:
        _validate_adapter(adapter)
        with self._lock:
            if adapter.adapter_id in self._adapters and not replace:
                raise FrameworkAdapterRegistrationError(
                    f"framework adapter {adapter.adapter_id!r} is already registered"
                )
            self._adapters[adapter.adapter_id] = adapter

    def unregister(self, adapter_id: str) -> None:
        with self._lock:
            if adapter_id not in self._adapters:
                raise FrameworkAdapterRegistrationError(
                    f"framework adapter {adapter_id!r} is not registered"
                )
            del self._adapters[adapter_id]

    def adapters(self) -> tuple[InferenceFrameworkAdapter, ...]:
        with self._lock:
            return tuple(self._adapters.values())

    def discover_entry_points(self, entries: Iterable[Any] | None = None) -> None:
        with self._lock:
            if entries is None and self._entry_points_loaded:
                return
            selected = tuple(entries) if entries is not None else _installed_entry_points()
            staged: list[InferenceFrameworkAdapter] = []
            staged_ids: set[str] = set()
            for entry in selected:
                try:
                    loaded = entry.load()
                    adapter = (
                        loaded
                        if isinstance(loaded, InferenceFrameworkAdapter)
                        and not isinstance(loaded, type)
                        else loaded()
                    )
                    _validate_adapter(adapter)
                except Exception as exc:
                    raise FrameworkAdapterResolutionError(
                        f"failed to load framework adapter entry point {entry!s}: {exc}"
                    ) from exc
                if adapter.adapter_id in self._adapters or adapter.adapter_id in staged_ids:
                    raise FrameworkAdapterResolutionError(
                        f"duplicate framework adapter {adapter.adapter_id!r}"
                    )
                staged.append(adapter)
                staged_ids.add(adapter.adapter_id)
            for adapter in staged:
                self._adapters[adapter.adapter_id] = adapter
            if entries is None:
                self._entry_points_loaded = True

    def resolve(self, request: FrameworkRequest) -> OperatorResolution:
        matching = tuple(adapter for adapter in self.adapters() if _matches(adapter, request))
        exact = tuple(adapter for adapter in matching if not adapter.fallback)
        if len(exact) > 1:
            raise FrameworkAdapterResolutionError(
                "ambiguous inference-framework adapters: "
                f"{sorted(adapter.adapter_id for adapter in exact)}"
            )
        candidates = exact or tuple(adapter for adapter in matching if adapter.fallback)
        if len(candidates) != 1:
            raise FrameworkAdapterResolutionError(
                "no unique inference-framework adapter matched the request"
            )
        try:
            resolution = candidates[0].resolve(request)
        except FrameworkAdapterError:
            raise
        except Exception as exc:
            raise FrameworkAdapterResolutionError(
                f"framework adapter {candidates[0].adapter_id!r} failed: {exc}"
            ) from exc
        if resolution.framework_adapter_id != candidates[0].adapter_id:
            raise FrameworkAdapterResolutionError(
                "framework resolution adapter ID does not match the selected adapter"
            )
        return resolution


def _matches(adapter: InferenceFrameworkAdapter, request: FrameworkRequest) -> bool:
    try:
        matched = adapter.matches(request)
    except Exception as exc:
        raise FrameworkAdapterResolutionError(
            f"framework adapter {adapter.adapter_id!r} failed while matching: {exc}"
        ) from exc
    if not isinstance(matched, bool):
        raise FrameworkAdapterResolutionError(
            f"framework adapter {adapter.adapter_id!r} returned a non-boolean match"
        )
    return matched


def _validate_adapter(adapter: InferenceFrameworkAdapter) -> None:
    if not isinstance(adapter, InferenceFrameworkAdapter):
        raise FrameworkAdapterRegistrationError(
            "adapter does not implement the InferenceFrameworkAdapter protocol"
        )
    if not isinstance(adapter.adapter_id, str) or not adapter.adapter_id:
        raise FrameworkAdapterRegistrationError("adapter_id must be a non-empty string")
    if not isinstance(adapter.fallback, bool):
        raise FrameworkAdapterRegistrationError("fallback must be a boolean")


def _installed_entry_points() -> tuple[Any, ...]:
    discovered = importlib_metadata.entry_points()
    if hasattr(discovered, "select"):
        return tuple(discovered.select(group=INFERENCE_FRAMEWORK_ADAPTER_ENTRY_POINT))
    return tuple(discovered.get(INFERENCE_FRAMEWORK_ADAPTER_ENTRY_POINT, ()))


_GLOBAL_REGISTRY = InferenceFrameworkAdapterRegistry(
    (ManifestInferenceFrameworkAdapter(), AnalysisOnlyFrameworkAdapter())
)


def register_inference_framework_adapter(
    adapter: InferenceFrameworkAdapter,
    *,
    replace: bool = False,
) -> None:
    _GLOBAL_REGISTRY.register(adapter, replace=replace)


def unregister_inference_framework_adapter(adapter_id: str) -> None:
    _GLOBAL_REGISTRY.unregister(adapter_id)


def registered_inference_framework_adapters() -> tuple[InferenceFrameworkAdapter, ...]:
    _GLOBAL_REGISTRY.discover_entry_points()
    return _GLOBAL_REGISTRY.adapters()


def resolve_inference_framework(request: FrameworkRequest) -> OperatorResolution:
    _GLOBAL_REGISTRY.discover_entry_points()
    return _GLOBAL_REGISTRY.resolve(request)


__all__ = [
    "ANALYSIS_ONLY_FRAMEWORK_ADAPTER_ID",
    "INFERENCE_FRAMEWORK_ADAPTER_ENTRY_POINT",
    "MANIFEST_FRAMEWORK_ADAPTER_ID",
    "AnalysisOnlyFrameworkAdapter",
    "FrameworkAdapterError",
    "FrameworkAdapterRegistrationError",
    "FrameworkAdapterResolutionError",
    "FrameworkRequest",
    "InferenceFrameworkAdapter",
    "InferenceFrameworkAdapterRegistry",
    "ManifestInferenceFrameworkAdapter",
    "OperatorResolution",
    "register_inference_framework_adapter",
    "registered_inference_framework_adapters",
    "resolve_inference_framework",
    "unregister_inference_framework_adapter",
]
