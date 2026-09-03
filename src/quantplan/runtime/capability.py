"""Versioned backend capability snapshots for fusion-aware planning.

The model architecture adapter describes *which* runtime owners belong to a
quantization domain.  This module describes *which combinations* a particular
backend/hardware revision can execute.  It intentionally has no dependency on
vLLM-Ascend internals; an extractor may emit this schema, or a user may provide
the same data as a checked-in YAML file.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml

from quantplan.domain.models import PrecisionFormat

CAPABILITY_SCHEMA = "quantplan.capability/v1"
_PRECISIONS = frozenset(
    {PrecisionFormat.BF16, PrecisionFormat.W4A8, PrecisionFormat.W8A8}
)
_STATUSES = frozenset({"verified", "experimental", "unsupported", "unknown"})


class CapabilityManifestError(ValueError):
    """A capability manifest is malformed or incompatible with the request."""


@dataclass(frozen=True)
class QuantizationDomainCapability:
    domain_id: str
    legal_modes: tuple[str, ...]
    owners: tuple[str, ...]
    allowed_precisions: tuple[PrecisionFormat, ...]
    contract: Mapping[str, Any]
    fallback_modes: tuple[str, ...] = ()
    status: str = "unknown"

    def matches_owner(self, owner: str) -> bool:
        return any(fnmatchcase(owner, pattern) for pattern in self.owners)


@dataclass(frozen=True)
class CapabilitySnapshot:
    schema_version: str
    backend_name: str
    backend_version: str
    backend_git_commit: str | None
    operator: str
    hardware_profile: str
    model_types: tuple[str, ...]
    architecture_adapter_ids: tuple[str, ...]
    domains: tuple[QuantizationDomainCapability, ...]
    source: str
    sha256: str

    @property
    def known(self) -> bool:
        return bool(self.domains) and all(domain.status == "verified" for domain in self.domains)

    @property
    def summary(self) -> str:
        return (
            f"{self.backend_name}/{self.backend_version}"
            f"@{self.hardware_profile}:{self.operator}"
        )

    def domain_for_owner(self, owner: str) -> QuantizationDomainCapability | None:
        matches = tuple(domain for domain in self.domains if domain.matches_owner(owner))
        if len(matches) > 1:
            raise CapabilityManifestError(
                f"owner {owner!r} matches multiple capability domains: "
                f"{[domain.domain_id for domain in matches]}"
            )
        return matches[0] if matches else None

    def matches_model(self, model_type: str, architecture_adapter_id: str) -> bool:
        return (
            not self.model_types or model_type.lower() in self.model_types
        ) and (
            not self.architecture_adapter_ids
            or architecture_adapter_id in self.architecture_adapter_ids
        )


def load_capability_manifest(
    path: str | Path,
    *,
    hardware_profile: str = "unknown",
) -> CapabilitySnapshot:
    """Load and validate one immutable backend capability snapshot."""

    source_path = Path(path)
    raw = source_path.read_bytes()
    try:
        document = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise CapabilityManifestError(f"invalid capability YAML: {exc}") from exc
    if not isinstance(document, Mapping):
        raise CapabilityManifestError("capability manifest root must be a mapping")
    schema = document.get("schema_version")
    if schema != CAPABILITY_SCHEMA:
        raise CapabilityManifestError(
            f"unsupported capability schema {schema!r}; expected {CAPABILITY_SCHEMA!r}"
        )
    backend = _mapping(document.get("backend"), "backend")
    backend_name = _string(backend.get("name"), "backend.name")
    backend_version = _string(backend.get("version"), "backend.version")
    operator = _string(backend.get("operator"), "backend.operator")
    commit = backend.get("git_commit")
    if commit is not None:
        commit = _string(commit, "backend.git_commit")
    manifest_hardware = _string(
        _mapping(document.get("hardware"), "hardware").get("profile"),
        "hardware.profile",
    ).lower()
    requested_hardware = str(hardware_profile).strip().lower()
    if requested_hardware != "unknown" and manifest_hardware != requested_hardware:
        raise CapabilityManifestError(
            f"capability hardware profile {manifest_hardware!r} does not match "
            f"requested {requested_hardware!r}"
        )
    models = document.get("models", {})
    if not isinstance(models, Mapping):
        raise CapabilityManifestError("models must be a mapping")
    model_types = tuple(
        item.lower() for item in _strings(models.get("model_types", ()), "models.model_types")
    )
    architecture_adapter_ids = _strings(
        models.get("architecture_adapter_ids", ()),
        "models.architecture_adapter_ids",
    )
    domains_doc = document.get("domains")
    if not isinstance(domains_doc, Sequence) or isinstance(domains_doc, (str, bytes)):
        raise CapabilityManifestError("domains must be a sequence")
    domains: list[QuantizationDomainCapability] = []
    seen: set[str] = set()
    for index, raw_domain in enumerate(domains_doc):
        item = _mapping(raw_domain, f"domains[{index}]")
        domain_id = _string(item.get("id"), f"domains[{index}].id")
        if domain_id in seen:
            raise CapabilityManifestError(f"duplicate capability domain {domain_id!r}")
        seen.add(domain_id)
        legal_modes = _strings(item.get("legal_modes", ()), f"domains[{index}].legal_modes")
        owners = _strings(item.get("owners", ()), f"domains[{index}].owners")
        raw_precisions = item.get("allowed_precisions", ())
        if not isinstance(raw_precisions, Sequence) or isinstance(raw_precisions, (str, bytes)):
            raise CapabilityManifestError(
                f"domains[{index}].allowed_precisions must be a sequence"
            )
        precisions: list[PrecisionFormat] = []
        for precision in raw_precisions:
            try:
                parsed = PrecisionFormat(str(precision).lower())
            except ValueError as exc:
                raise CapabilityManifestError(
                    f"domains[{index}].allowed_precisions contains {precision!r}"
                ) from exc
            if parsed not in _PRECISIONS:
                raise CapabilityManifestError(f"unsupported capability precision {precision!r}")
            if parsed not in precisions:
                precisions.append(parsed)
        contract = item.get("contract", {})
        if not isinstance(contract, Mapping):
            raise CapabilityManifestError(f"domains[{index}].contract must be a mapping")
        fallback = _strings(item.get("fallback", ()), f"domains[{index}].fallback")
        status = _string(item.get("status", "unknown"), f"domains[{index}].status").lower()
        if status not in _STATUSES:
            raise CapabilityManifestError(
                f"domains[{index}].status must be one of {sorted(_STATUSES)}"
            )
        domains.append(
            QuantizationDomainCapability(
                domain_id=domain_id,
                legal_modes=legal_modes,
                owners=owners,
                allowed_precisions=tuple(precisions),
                contract=dict(contract),
                fallback_modes=fallback,
                status=status,
            )
        )
    return CapabilitySnapshot(
        schema_version=CAPABILITY_SCHEMA,
        backend_name=backend_name,
        backend_version=backend_version,
        backend_git_commit=commit,
        operator=operator,
        hardware_profile=manifest_hardware,
        model_types=model_types,
        architecture_adapter_ids=architecture_adapter_ids,
        domains=tuple(domains),
        source=str(source_path.resolve()),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CapabilityManifestError(f"{path} must be a mapping")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapabilityManifestError(f"{path} must be a non-empty string")
    return value.strip()


def _strings(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CapabilityManifestError(f"{path} must be a sequence")
    result = tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise CapabilityManifestError(f"{path} must not contain duplicates")
    return result


__all__ = [
    "CAPABILITY_SCHEMA",
    "CapabilityManifestError",
    "CapabilitySnapshot",
    "QuantizationDomainCapability",
    "load_capability_manifest",
]
