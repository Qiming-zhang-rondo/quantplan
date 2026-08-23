from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum


class UnknownFieldStatus(str, Enum):
    """How the guard treats a field at a pinned capability boundary.

    ``OPAQUE`` means the field is retained unchanged and recorded as evidence,
    but is never interpreted or synthesized by QuantPlan. ``BLOCKING`` means
    execution cannot be declared safe until a capability plug-in explicitly
    understands it.
    """

    KNOWN = "known"
    OPAQUE = "opaque"
    BLOCKING = "blocking"


# Kept as a public alias because callers often describe this value as a
# disposition rather than a status.
FieldDisposition = UnknownFieldStatus


class CapabilityStatus(str, Enum):
    """Confidence attached to a registered static schema fragment.

    This status never implies that a model/runtime/hardware combination can
    execute the corresponding processor.
    """

    SUPPORTED_VERIFIED = "supported_verified"
    SUPPORTED_EXPERIMENTAL = "supported_experimental"
    KNOWN_UNSUPPORTED = "known_unsupported"
    UNKNOWN = "unknown"
    OUT_OF_SCOPE = "out_of_scope"

    # Compact aliases keep the registration API pleasant while the serialized
    # values retain the architecture-level vocabulary.
    VERIFIED = SUPPORTED_VERIFIED
    EXPERIMENTAL = SUPPORTED_EXPERIMENTAL
    UNSUPPORTED = KNOWN_UNSUPPORTED


@dataclass(frozen=True)
class ProcessCapability:
    """Schema fragment for one ModelSlim ``spec.process[*].type``.

    The schema is intentionally small and explicit. A process may declare
    ordinary qconfig fields, nested process lists (for ``group``), and maps of
    named qconfigs (for FA3 ``details``). Anything else remains blocking.
    """

    type_name: str
    allowed_fields: frozenset[str]
    required_fields: frozenset[str] = frozenset({"type"})
    # Linear qconfig has ``act``/``weight`` branches. Direct qconfig is one
    # quantizer mapping, as used by FA3 and dynamic cache.
    qconfig_fields: frozenset[str] = frozenset()
    direct_qconfig_fields: frozenset[str] = frozenset()
    nested_process_fields: frozenset[str] = frozenset()
    pattern_fields: frozenset[str] = frozenset()
    qconfig_map_fields: tuple[tuple[str, tuple[str, ...]], ...] = ()
    status: CapabilityStatus = CapabilityStatus.VERIFIED

    def __post_init__(self) -> None:
        if not self.type_name:
            raise ValueError("process type_name must not be empty")
        if "type" not in self.allowed_fields:
            raise ValueError("process allowed_fields must include 'type'")
        if not self.required_fields <= self.allowed_fields:
            raise ValueError(
                f"required fields are absent from allowed_fields for {self.type_name!r}: "
                f"{sorted(self.required_fields - self.allowed_fields)}"
            )
        declared_special_fields = (
            self.qconfig_fields
            | self.direct_qconfig_fields
            | self.nested_process_fields
            | self.pattern_fields
        )
        declared_special_fields |= frozenset(name for name, _ in self.qconfig_map_fields)
        undeclared = declared_special_fields - self.allowed_fields
        if undeclared:
            raise ValueError(
                f"special fields are absent from allowed_fields for {self.type_name!r}: "
                f"{sorted(undeclared)}"
            )

    @property
    def qconfig_maps(self) -> Mapping[str, tuple[str, ...]]:
        return dict(self.qconfig_map_fields)


@dataclass(frozen=True)
class SaverCapability:
    type_name: str
    allowed_fields: frozenset[str]
    status: CapabilityStatus = CapabilityStatus.VERIFIED

    def __post_init__(self) -> None:
        if not self.type_name:
            raise ValueError("saver type_name must not be empty")
        if "type" not in self.allowed_fields:
            raise ValueError("saver allowed_fields must include 'type'")


@dataclass
class CapabilityManifest:
    """Versioned, runtime-independent ModelSlim YAML capability manifest.

    Registries are per-instance so a plug-in can add a process for one pinned
    ModelSlim environment without silently broadening all other environments.
    """

    manifest_id: str = "modelslim-v1-schema-prototype"
    supported_apiversions: frozenset[str] = frozenset(
        {"modelslim_v1", "multimodal_vlm_modelslim_v1"}
    )
    root_fields: frozenset[str] = frozenset({"apiversion", "metadata", "spec"})
    metadata_fields: frozenset[str] = frozenset(
        {"config_id", "score", "label", "verified_model_types", "verified_tags"}
    )
    metadata_label_fields: frozenset[str] = frozenset(
        {"w_bit", "a_bit", "is_sparse", "kv_cache", "fa_bit", "fa_quant"}
    )
    spec_fields: frozenset[str] = frozenset(
        {"runner", "prior", "process", "save", "dataset", "default_text"}
    )
    prior_stage_fields: frozenset[str] = frozenset({"process", "dataset"})
    qconfig_fields: frozenset[str] = frozenset({"act", "weight"})
    quantizer_fields: frozenset[str] = frozenset(
        {"dtype", "scope", "symmetric", "method", "ext"}
    )
    quantizer_required_fields: frozenset[str] = frozenset(
        {"dtype", "scope", "symmetric", "method"}
    )
    quantizer_dtypes: frozenset[str] = frozenset(
        {"float", "int8", "int4", "mxfp8", "mxfp4", "fp8_e4m3", "placeholder"}
    )
    quantizer_scopes: frozenset[str] = frozenset(
        {
            "per_tensor",
            "per_channel",
            "per_group",
            "per_block",
            "per_token",
            "pd_mix",
            "per_head",
            "dual_scale",
            "placeholder",
        }
    )
    # Only the conservative built-in combination is accepted by default.
    # A pinned plug-in manifest may extend this set after registry probing.
    quantizer_methods: frozenset[str] = frozenset({"minmax", "ssz"})
    runner_values: frozenset[str] = frozenset(
        {"auto", "model_wise", "layer_wise", "dp_layer_wise"}
    )
    qconfig_ext_fields: frozenset[str] = frozenset(
        {
            "aggregation_type",
            "axes",
            "block_size",
            "dual_block_size",
            "group_size",
            "highbound_factor",
            "lowbound_factor",
            "percdamp",
            "step",
        }
    )
    unknown_root_status: UnknownFieldStatus = UnknownFieldStatus.OPAQUE
    unknown_metadata_status: UnknownFieldStatus = UnknownFieldStatus.OPAQUE
    unknown_semantic_status: UnknownFieldStatus = UnknownFieldStatus.BLOCKING
    process_types: dict[str, ProcessCapability] = field(default_factory=dict)
    saver_types: dict[str, SaverCapability] = field(default_factory=dict)

    def register_process_type(
        self,
        capability: ProcessCapability | str,
        *,
        allowed_fields: Iterable[str] = (),
        required_fields: Iterable[str] = (),
        qconfig_fields: Iterable[str] = (),
        direct_qconfig_fields: Iterable[str] = (),
        nested_process_fields: Iterable[str] = (),
        pattern_fields: Iterable[str] = (),
        qconfig_map_fields: Mapping[str, Iterable[str]] | None = None,
        status: CapabilityStatus = CapabilityStatus.VERIFIED,
        replace: bool = False,
    ) -> ProcessCapability:
        """Register a process type understood by this exact manifest instance."""

        if isinstance(capability, str):
            qconfig_maps = tuple(
                (name, tuple(branches))
                for name, branches in (qconfig_map_fields or {}).items()
            )
            capability = ProcessCapability(
                type_name=capability,
                allowed_fields=frozenset({"type", *allowed_fields}),
                required_fields=frozenset({"type", *required_fields}),
                qconfig_fields=frozenset(qconfig_fields),
                direct_qconfig_fields=frozenset(direct_qconfig_fields),
                nested_process_fields=frozenset(nested_process_fields),
                pattern_fields=frozenset(pattern_fields),
                qconfig_map_fields=qconfig_maps,
                status=status,
            )
        if capability.type_name in self.process_types and not replace:
            raise ValueError(f"process type {capability.type_name!r} is already registered")
        self.process_types[capability.type_name] = capability
        return capability

    def register_saver_type(
        self,
        capability: SaverCapability | str,
        *,
        allowed_fields: Iterable[str] = (),
        status: CapabilityStatus = CapabilityStatus.VERIFIED,
        replace: bool = False,
    ) -> SaverCapability:
        if isinstance(capability, str):
            capability = SaverCapability(
                type_name=capability,
                allowed_fields=frozenset({"type", *allowed_fields}),
                status=status,
            )
        if capability.type_name in self.saver_types and not replace:
            raise ValueError(f"saver type {capability.type_name!r} is already registered")
        self.saver_types[capability.type_name] = capability
        return capability

    def register_qconfig_ext_fields(self, *field_names: str) -> None:
        """Extend the understood qconfig ``ext`` keys for this manifest only."""

        if any(not name for name in field_names):
            raise ValueError("qconfig ext field names must not be empty")
        self.qconfig_ext_fields = self.qconfig_ext_fields | frozenset(field_names)

    @property
    def fingerprint(self) -> str:
        """Content-addressed ID so local registrations remain auditable."""

        payload = {
            "manifest_id": self.manifest_id,
            "apiversions": sorted(self.supported_apiversions),
            "root_fields": sorted(self.root_fields),
            "metadata_fields": sorted(self.metadata_fields),
            "metadata_label_fields": sorted(self.metadata_label_fields),
            "spec_fields": sorted(self.spec_fields),
            "prior_stage_fields": sorted(self.prior_stage_fields),
            "linear_qconfig_fields": sorted(self.qconfig_fields),
            "quantizer_fields": sorted(self.quantizer_fields),
            "quantizer_required_fields": sorted(self.quantizer_required_fields),
            "dtypes": sorted(self.quantizer_dtypes),
            "scopes": sorted(self.quantizer_scopes),
            "methods": sorted(self.quantizer_methods),
            "runners": sorted(self.runner_values),
            "qconfig_ext_fields": sorted(self.qconfig_ext_fields),
            "unknown_statuses": {
                "root": self.unknown_root_status.value,
                "metadata": self.unknown_metadata_status.value,
                "semantic": self.unknown_semantic_status.value,
            },
            "processes": {
                name: {
                    "fields": sorted(item.allowed_fields),
                    "required": sorted(item.required_fields),
                    "linear_qconfig": sorted(item.qconfig_fields),
                    "direct_qconfig": sorted(item.direct_qconfig_fields),
                    "nested": sorted(item.nested_process_fields),
                    "patterns": sorted(item.pattern_fields),
                    "maps": sorted(
                        (field_name, sorted(branches))
                        for field_name, branches in item.qconfig_map_fields
                    ),
                    "status": item.status.value,
                }
                for name, item in sorted(self.process_types.items())
            },
            "savers": {
                name: {
                    "fields": sorted(item.allowed_fields),
                    "status": item.status.value,
                }
                for name, item in sorted(self.saver_types.items())
            },
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"{self.manifest_id}@sha256:{digest}"


def default_modelslim_manifest() -> CapabilityManifest:
    """Return a fresh manifest for the ModelSlim V1 subset used by QuantPlan."""

    manifest = CapabilityManifest()

    manifest.register_process_type(
        "linear_quant",
        allowed_fields={"qconfig", "include", "exclude"},
        required_fields={"qconfig"},
        qconfig_fields={"qconfig"},
        pattern_fields={"include", "exclude"},
    )
    manifest.register_process_type(
        "group",
        allowed_fields={"configs"},
        required_fields={"configs"},
        nested_process_fields={"configs"},
    )
    manifest.register_process_type(
        "quarot",
        allowed_fields={
            "online",
            "block_size",
            "down_proj_online_layers",
            "max_tp_size",
            "export_extra_info",
        },
    )
    manifest.register_process_type(
        "flex_smooth_quant",
        allowed_fields={"alpha", "beta", "enable_subgraph_type", "include", "exclude"},
        pattern_fields={"include", "exclude"},
    )
    manifest.register_process_type(
        "flex_awq_ssz",
        allowed_fields={
            "alpha",
            "beta",
            "enable_subgraph_type",
            "include",
            "exclude",
            "qconfig",
        },
        qconfig_fields={"qconfig"},
        required_fields={"qconfig"},
        pattern_fields={"include", "exclude"},
    )
    manifest.register_process_type(
        "fa3_quant",
        allowed_fields={"qconfig", "details", "include", "exclude"},
        direct_qconfig_fields={"qconfig"},
        qconfig_map_fields={"details": ("fa_q", "fa_k", "fa_v")},
        pattern_fields={"include", "exclude"},
    )
    manifest.register_process_type(
        "dynamic_cache",
        allowed_fields={"qconfig", "include", "exclude"},
        required_fields={"qconfig"},
        direct_qconfig_fields={"qconfig"},
        pattern_fields={"include", "exclude"},
    )

    manifest.register_saver_type(
        "ascendv1_saver",
        allowed_fields={"part_file_size", "save_directory", "ext"},
    )
    return manifest


# A shorter alias is convenient at integration boundaries.
modelslim_v1_manifest = default_modelslim_manifest
