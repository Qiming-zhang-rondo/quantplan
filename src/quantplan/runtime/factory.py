"""Construct the optional PyTorch runtime from a validated run config."""

from __future__ import annotations

from fnmatch import fnmatchcase

from quantplan.adapters.modelslim import (
    inspect_modelslim_yaml,
    profile_linear_source,
)
from quantplan.run_config import RunConfig
from quantplan.runtime.capability import load_capability_manifest
from quantplan.runtime.torch_analysis import (
    TorchAnalysisRuntime,
    discover_transformer_blocks,
    load_causal_lm,
    load_jsonl_texts,
)


def build_torch_runtime(config: RunConfig) -> TorchAnalysisRuntime:
    """Load the configured model/data and build a bounded analysis runtime."""

    config.validate_for_run()
    if not config.runtime.analysis_only:
        raise ValueError(
            "runtime.analysis_only=false is not implemented; QuantPlan does not execute "
            "ModelSlim or claim target-hardware readiness"
        )
    loader_entrypoint = (
        None if config.runtime.loader == "huggingface" else config.runtime.loader
    )
    bundle = load_causal_lm(
        str(config.model_path),
        revision=config.runtime.revision,
        device=config.device,
        torch_dtype=config.runtime.torch_dtype,
        trust_remote_code=config.runtime.trust_remote_code,
        allow_image_text_fallback=config.runtime.allow_image_text_fallback,
        loader_entrypoint=loader_entrypoint,
    )
    samples = load_jsonl_texts(
        config.data_path,
        text_field=config.analysis.text_field,
        id_field=config.analysis.id_field,
        limit=config.analysis.max_samples,
    )
    module_names = tuple(name for name, _ in bundle.model.named_modules())
    source_guard = inspect_modelslim_yaml(
        config.original_yaml,
        module_names=module_names,
    )
    if source_guard.blocked:
        codes = ", ".join(issue.code for issue in source_guard.blocking_issues)
        raise ValueError(f"source ModelSlim YAML failed static policy: {codes}")

    explicit_groups = None
    if config.modules.block_patterns:
        selected = expand_block_patterns(
            config.modules.block_patterns,
            module_names,
        )
        explicit_groups = {name: (name,) for name in selected}

    potential_groups = discover_transformer_blocks(
        bundle.model,
        explicit_groups,
        linear_include=config.modules.include or ("*",),
        linear_exclude=config.modules.exclude,
        max_groups=None,
        strict_targets=config.runtime.strict_target_coverage,
    )
    potential_targets = tuple(
        module_name
        for group in potential_groups
        for module_name in group.module_names
    )
    potential_aliases: dict[str, tuple[str, ...]] = {}
    for group in potential_groups:
        for module_name, backend_pattern in zip(
            group.module_names,
            group.effective_backend_patterns,
            strict=True,
        ):
            if backend_pattern.endswith(".*"):
                potential_aliases[module_name] = (
                    f"{backend_pattern[:-2]}0.gate_proj",
                    f"{backend_pattern[:-2]}0.up_proj",
                    f"{backend_pattern[:-2]}0.down_proj",
                )
    source_profile = profile_linear_source(
        config.original_yaml,
        potential_targets,
        candidate_aliases=potential_aliases,
    )
    eligible = set(source_profile.eligible_modules)
    # GLM discovery exposes one inventory per fused subgroup (``block:kind``),
    # while the explicit runtime API consumes block roots.  Collapse those
    # subgroup IDs back to unique roots; the runtime reapplies the adapter's
    # fusion-aware split after source-YAML filtering.
    source_groups: dict[str, tuple[str, ...]] = {}
    for group in potential_groups:
        if not eligible.intersection(group.module_names):
            continue
        root = group.group_id.rsplit(":", 1)[0] if ":" in group.group_id else group.group_id
        source_groups[root] = (root,)
    if not source_groups:
        raise ValueError("source YAML leaves no decision groups after target filtering")

    runtime = TorchAnalysisRuntime.from_bundle(
        bundle,
        samples,
        groups=source_groups,
        linear_include=source_profile.eligible_modules,
        linear_exclude=config.modules.exclude,
        max_groups=config.analysis.max_groups,
        strict_targets=config.runtime.strict_target_coverage,
        batch_size=config.analysis.batch_size,
        max_length=config.analysis.max_length,
        # An explicit ModelPlacement returned by the loader is authoritative for
        # the input ordinal. A typed auto request (for example npu:auto) remains
        # a backend constraint so a loader cannot silently return CUDA instead.
        # Legacy loaders without placement retain the full configured constraint.
        device=_runtime_device_constraint(config.device, bundle.placement is not None),
        compute_kl=config.analysis.compute_kl,
        hardware_profile=config.runtime.hardware_profile,
        inference_framework=config.runtime.inference_framework,
        capability_snapshot=(
            load_capability_manifest(
                config.runtime.backend_capability_manifest,
                hardware_profile=config.runtime.hardware_profile,
            )
            if config.runtime.backend_capability_manifest is not None
            else None
        ),
    )
    discovered_targets = {
        module_name
        for group in runtime.all_groups
        for module_name in group.module_names
    }
    missing_targets = eligible - discovered_targets
    if missing_targets:
        raise ValueError(
            "source target inventory was not preserved by runtime discovery: "
            f"{sorted(missing_targets)}"
        )
    runtime.source_linear_profile = source_profile
    runtime.source_guard = source_guard
    return runtime


def expand_block_patterns(
    patterns: tuple[str, ...],
    module_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Expand every block pattern and reject empty or overlapping selections."""

    if not patterns:
        raise ValueError("at least one block pattern is required")
    selected: list[str] = []
    owner: dict[str, str] = {}
    for pattern in patterns:
        raw_matches = tuple(name for name in module_names if fnmatchcase(name, pattern))
        matches = tuple(
            name
            for name in raw_matches
            if not any(
                name != other and name.startswith(f"{other}.")
                for other in raw_matches
            )
        )
        if not matches:
            raise ValueError(f"block pattern {pattern!r} matched no named_modules entry")
        for name in matches:
            previous = owner.setdefault(name, pattern)
            if previous != pattern:
                raise ValueError(
                    f"block module {name!r} is matched by both {previous!r} and {pattern!r}"
                )
            if name not in selected:
                selected.append(name)
    return tuple(selected)


def uniformly_select(values: tuple[str, ...], limit: int) -> tuple[str, ...]:
    """Select deterministic, evenly spaced entries while retaining order."""

    if limit < 1:
        raise ValueError("limit must be positive")
    if len(values) <= limit:
        return values
    if limit == 1:
        return (values[len(values) // 2],)
    indexes = {
        round(step * (len(values) - 1) / (limit - 1)) for step in range(limit)
    }
    return tuple(value for index, value in enumerate(values) if index in indexes)


def _runtime_device_constraint(configured_device: str, has_placement: bool) -> str | None:
    if not has_placement:
        return configured_device
    normalized = configured_device.strip().lower()
    if normalized in {"npu:auto", "cuda:auto", "xpu:auto"}:
        return configured_device
    return None


__all__ = ["build_torch_runtime", "expand_block_patterns", "uniformly_select"]
