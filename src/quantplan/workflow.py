"""End-to-end analysis, planning, validation and guarded YAML compilation."""

from __future__ import annotations

import difflib
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from quantplan.adapters.modelslim import (
    ModelSlimPlanCompiler,
    PlanCompilationResult,
    inspect_modelslim_yaml,
    profile_linear_source,
    resolve_precision_templates,
)
from quantplan.artifacts import ArtifactError, ArtifactStore
from quantplan.domain import (
    ApproximateMeasurement,
    DirectionalMeasurement,
    GroupInventory,
    InterventionMeasurement,
    PlanEvaluation,
    PrecisionFormat,
    QuantPlan,
    StatisticalMeasurement,
)
from quantplan.latency import build_latency_provider
from quantplan.planning import (
    PlanningPolicy,
    build_candidate_set,
    choose_lowest_cost_passing_plan,
    plan_id,
)
from quantplan.run_config import RunConfig

_DIRECTIONAL_EVIDENCE_ROLE = (
    "shared projection signed first-order; one extra local gradient projection per Linear/batch "
    "(possibly multiple chunk GEMMs), "
    "no candidate affine/block replay"
)


class AnalysisResultLike(Protocol):
    groups: Sequence[GroupInventory]
    module_inventory: Sequence[str]
    measurements: Sequence[
        InterventionMeasurement
        | ApproximateMeasurement
        | StatisticalMeasurement
        | DirectionalMeasurement
    ]
    provenance: Any


class AnalysisRuntimeLike(Protocol):
    groups: Sequence[GroupInventory]
    module_inventory: Sequence[str]
    discovered_group_count: int
    full_group_coverage: bool

    def run(
        self,
        precisions: Sequence[PrecisionFormat],
        *,
        progress_callback: Callable[[str, int, int, str, str], None] | None = None,
    ) -> AnalysisResultLike: ...

    def run_fast(
        self,
        precisions: Sequence[PrecisionFormat],
        *,
        progress_callback: Callable[[str, int, int, str, str], None] | None = None,
    ) -> AnalysisResultLike: ...

    def run_fast_stats(
        self,
        precisions: Sequence[PrecisionFormat],
        *,
        progress_callback: Callable[[str, int, int, str, str], None] | None = None,
    ) -> AnalysisResultLike: ...

    def run_fast_taylor(
        self,
        precisions: Sequence[PrecisionFormat],
        *,
        progress_callback: Callable[[str, int, int, str, str], None] | None = None,
    ) -> AnalysisResultLike: ...

    def evaluate_plan(
        self,
        assignments: Mapping[str, PrecisionFormat],
        *,
        plan_id: str,
    ) -> PlanEvaluation: ...


@dataclass(frozen=True)
class WorkflowResult:
    status: str
    selected_plan_id: str | None
    output_dir: Path
    derived_yaml: Path | None
    validation: PlanEvaluation | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "selected_plan_id": self.selected_plan_id,
            "output_dir": str(self.output_dir),
            "derived_yaml": str(self.derived_yaml) if self.derived_yaml else None,
            "validation": _jsonable(self.validation),
        }


def execute_workflow(
    config: RunConfig,
    runtime: AnalysisRuntimeLike,
    *,
    overwrite: bool = False,
    compiler: ModelSlimPlanCompiler | None = None,
    precision_templates: Mapping[PrecisionFormat, Mapping[str, Any]] | None = None,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
    analysis_mode: str = "exact",
) -> WorkflowResult:
    """Run the complete evidence loop with an already constructed runtime."""

    config.validate_for_run()
    if analysis_mode not in {"exact", "fast", "fast-stats", "fast-taylor"}:
        raise ValueError("analysis_mode must be 'exact', 'fast', 'fast-stats' or 'fast-taylor'")
    early_stop_on_pass = analysis_mode in {"fast-stats", "fast-taylor"}
    validation_cap = None if analysis_mode == "exact" else 3
    _prepare_output(config, overwrite=overwrite)
    store = ArtifactStore(config.output_dir, overwrite=overwrite)
    manifest_store = ArtifactStore(config.output_dir, overwrite=True)
    started_at = _now()
    _write_manifest(
        manifest_store,
        config,
        status="running",
        started_at=started_at,
        details={"analysis_mode": analysis_mode},
    )

    try:
        groups = tuple(runtime.groups)
        full_module_inventory = tuple(runtime.module_inventory)
        discovered_group_count = int(getattr(runtime, "discovered_group_count", len(groups)))
        full_group_coverage = bool(getattr(runtime, "full_group_coverage", True))
        if not groups or not full_module_inventory:
            raise RuntimeError("analysis runtime returned an empty model inventory")
        if discovered_group_count < len(groups):
            raise RuntimeError(
                "analysis runtime discovered_group_count is smaller than selected groups"
            )
        if full_group_coverage and discovered_group_count != len(groups):
            raise RuntimeError(
                "analysis runtime reported full coverage with an incomplete group inventory"
            )

        store.write_json(
            config.output.model_inventory,
            {
                "schema_version": "quantplan.model-inventory/v1",
                "groups": [_jsonable(item) for item in groups],
                "model_module_names": list(full_module_inventory),
                "coverage": {
                    "selected_group_count": len(groups),
                    "discovered_group_count": discovered_group_count,
                    "full_group_coverage": full_group_coverage,
                },
            },
        )
        source_guard = inspect_modelslim_yaml(
            config.original_yaml,
            module_names=full_module_inventory,
        )
        store.write_json(
            config.output.yaml_audit,
            _guard_payload(source_guard, guard_scope="source_yaml"),
        )
        if source_guard.blocked:
            codes = ", ".join(issue.code for issue in source_guard.blocking_issues)
            raise RuntimeError(f"source ModelSlim YAML failed static policy: {codes}")

        _progress(
            progress,
            "analysis_started",
            {"groups": len(runtime.groups), "analysis_mode": analysis_mode},
        )
        source_anchor_precisions = tuple(
            dict.fromkeys(
                group.source_precision
                for group in groups
                if group.source_precision is not None
            )
        )
        analysis_precisions = tuple(
            dict.fromkeys((*config.candidates.precisions, *source_anchor_precisions))
        )
        if analysis_mode == "exact":
            analysis_runner = runtime.run
        elif analysis_mode == "fast":
            analysis_runner = runtime.run_fast
        elif analysis_mode == "fast-stats":
            analysis_runner = runtime.run_fast_stats
        else:
            analysis_runner = runtime.run_fast_taylor
        result = analysis_runner(
            analysis_precisions,
            progress_callback=_measurement_progress(progress),
        )
        measurements = tuple(result.measurements)
        if tuple(result.groups) != groups:
            raise RuntimeError("analysis runtime group inventory changed during the run")
        if tuple(result.module_inventory) != full_module_inventory:
            raise RuntimeError("analysis runtime module inventory changed during the run")
        if not measurements:
            raise RuntimeError("analysis runtime returned no intervention measurements")
        _progress(
            progress,
            "analysis_completed",
            {
                "groups": len(groups),
                "measurements": len(measurements),
                "analysis_mode": analysis_mode,
            },
        )

        store.write_jsonl(
            config.output.risk_evidence,
            (_measurement_payload(item) for item in measurements),
        )
        latency_provider = build_latency_provider(
            mode=config.latency.mode,
            deployment_mode=config.latency.deployment_mode,
            hardware_profile=config.runtime.hardware_profile,
            memory_bandwidth_gbps=config.latency.memory_bandwidth_gbps,
            compute_tflops=config.latency.compute_tflops,
            memory_efficiency=config.latency.memory_efficiency,
            compute_efficiency=config.latency.compute_efficiency,
            launch_overhead_us=config.latency.launch_overhead_us,
            qdq_overhead_us=config.latency.qdq_overhead_us,
            layout_overhead_us=config.latency.layout_overhead_us,
            fallback_overhead_us=config.latency.fallback_overhead_us,
            uncertainty=config.latency.uncertainty,
            calibration_path=config.latency.calibration_path,
        )
        if latency_provider is not None:
            _progress(
                progress,
                "latency_model_ready",
                dict(latency_provider.summary()),
            )
            latency_payload = {
                "schema_version": "quantplan.latency-report/v1",
                "provider": dict(latency_provider.summary()),
                "candidates": [],
            }
            for inventory in groups:
                for precision in analysis_precisions:
                    if (
                        inventory.allowed_formats is not None
                        and precision not in inventory.allowed_formats
                    ):
                        continue
                    estimate = latency_provider.estimate(inventory, precision)
                    latency_payload["candidates"].append(
                        {
                            "group_id": inventory.group_id,
                            "precision": precision.value,
                            "seconds": estimate.seconds,
                            "lower_bound": estimate.lower_bound,
                            "upper_bound": estimate.upper_bound,
                            "confidence": estimate.confidence,
                            "model": estimate.model,
                            "components": dict(estimate.components),
                            "warnings": list(estimate.warnings),
                        }
                    )
            store.write_json(config.output.latency_report, latency_payload)
        else:
            store.write_json(
                config.output.latency_report,
                {
                    "schema_version": "quantplan.latency-report/v1",
                    "provider": {"mode": "off"},
                    "candidates": [],
                    "warnings": [
                        "latency model disabled; estimated_cost uses normalized parameter-count proxies"
                    ],
                },
            )
        planning_policy = PlanningPolicy(
            base_precision=config.search.base_precision,
            max_high_precision_fraction=config.search.high_precision_fraction,
            max_candidates=config.search.max_candidates,
            max_delta_nll=config.search.max_delta_nll,
        )
        candidate_set = build_candidate_set(
            groups,
            measurements,
            allowed_formats=analysis_precisions,
            policy=planning_policy,
            relative_costs=config.search.costs,
            latency_provider=latency_provider,
        )
        if not candidate_set.plans:
            reasons = "; ".join(candidate_set.rejected_reasons) or "no plan candidates"
            raise RuntimeError(f"planning produced no candidate: {reasons}")

        candidate_payload = {
            "schema_version": "quantplan.candidate-set/v1",
            "analysis_mode": analysis_mode,
            "claim": {
                "exact": "unvalidated proposals from candidate-specific whole-model interventions",
                "fast": "unvalidated proposals from gradient-weighted local approximations",
                "fast-stats": (
                    "unvalidated proposals from factorized gradient statistics; no block replay"
                ),
                "fast-taylor": f"unvalidated proposals from {_DIRECTIONAL_EVIDENCE_ROLE}",
            }[analysis_mode],
            "max_estimated_cost": candidate_set.max_estimated_cost,
            "source_yaml_safe_anchors": {
                group.group_id: group.source_precision.value
                for group in groups
                if group.source_precision is not None
            },
            "rejected_reasons": list(candidate_set.rejected_reasons),
            "plans": [_plan_payload(plan) for plan in candidate_set.plans],
            "latency_model": (
                dict(latency_provider.summary()) if latency_provider is not None else {"mode": "off"}
            ),
            "evidence": [_jsonable(item) for item in candidate_set.evidence],
        }
        store.write_json(config.output.candidate_plan, candidate_payload)

        evaluations: list[PlanEvaluation] = []
        ordered_plans = sorted(
            candidate_set.plans,
            key=lambda item: (item.estimated_cost, item.proposal_score),
        )
        if validation_cap is not None:
            ordered_plans = ordered_plans[:validation_cap]
        validated_plans: list[QuantPlan] = []
        for index, plan in enumerate(ordered_plans, start=1):
            identifier = plan_id(plan)
            _progress(
                progress,
                "plan_validation_started",
                {"current": index, "total": len(ordered_plans), "plan_id": identifier},
            )
            evaluation = runtime.evaluate_plan(
                plan.assignments,
                plan_id=identifier,
            )
            if evaluation.plan_id != identifier:
                raise RuntimeError("runtime returned a mismatched plan evaluation ID")
            if evaluation.context_id != plan.context_id:
                raise ValueError(f"evaluation context mismatch for {identifier}")
            validated_plans.append(plan)
            evaluations.append(evaluation)
            _progress(
                progress,
                "plan_validation_completed",
                {"plan_id": identifier, "delta_nll": evaluation.delta_nll},
            )
            if early_stop_on_pass and evaluation.passes(
                max_delta_nll=config.search.max_delta_nll
            ):
                break

        selected = choose_lowest_cost_passing_plan(
            validated_plans,
            evaluations,
            max_delta_nll=config.search.max_delta_nll,
        )
        validation_payload = {
            "schema_version": "quantplan.validation-report/v1",
            "analysis_mode": analysis_mode,
            "validation_cap": validation_cap,
            "early_stop_on_pass": early_stop_on_pass,
            "gate": {
                "metric": "teacher_forced_delta_nll_per_token",
                "maximum": config.search.max_delta_nll,
            },
            "evaluations": [_evaluation_payload(item) for item in evaluations],
            "selected_plan_id": plan_id(selected[0]) if selected else None,
            "passed": selected is not None,
            "scope": "full_group_inventory" if full_group_coverage else "pilot_subset",
            "limitations": [
                "fake-quant proxy evidence is not ModelSlim-exact",
                *(
                    [
                        "fast proposal scores are first-order local approximations, not delta NLL",
                        "fast mode exactly validates at most the three cheapest proposals",
                    ]
                    if analysis_mode == "fast"
                    else []
                ),
                *(
                    [
                        (
                            "fast-stats proposal scores use factorized gradient statistics, "
                            "not observed delta NLL or first-order replay effects; no block replay"
                        ),
                        (
                            "fast-stats validates at most the three cheapest proposals and "
                            "stops at the first passing plan"
                        ),
                    ]
                    if analysis_mode == "fast-stats"
                    else []
                ),
                *(
                    [
                        f"fast-taylor proposal scores use {_DIRECTIONAL_EVIDENCE_ROLE}",
                        (
                            "fast-taylor scores preserve signed effects with a shared per-group "
                            "ranking offset; they are not observed delta NLL"
                        ),
                        (
                            "cross-group sensitivity ranking must use mean_first_order_effect, "
                            "not offset-adjusted estimated_risk"
                        ),
                        (
                            "fast-taylor validates at most the three cheapest proposals and "
                            "stops at the first passing plan"
                        ),
                    ]
                    if analysis_mode == "fast-taylor"
                    else []
                ),
                "target-hardware execution readiness is not assessed",
                "task accuracy and hardware performance require independent validation",
                *(
                    []
                    if full_group_coverage
                    else ["pilot subset does not cover every discovered block and cannot emit YAML"]
                ),
            ],
        }
        validation_report_path = store.write_json(
            config.output.validation_report,
            validation_payload,
        )

        if selected is None:
            _write_provenance(
                store,
                config,
                result.provenance,
                analysis_mode=analysis_mode,
                validation_report_path=validation_report_path,
            )
            _write_manifest(
                manifest_store,
                config,
                status="plan_validation_failed",
                started_at=started_at,
                details={
                    "analysis_mode": analysis_mode,
                    "evaluated_plan_count": len(evaluations),
                },
            )
            return WorkflowResult(
                status="plan_validation_failed",
                selected_plan_id=None,
                output_dir=config.output_dir,
                derived_yaml=None,
                validation=None,
            )

        selected_plan, selected_evaluation = selected
        if not full_group_coverage:
            _write_provenance(
                store,
                config,
                result.provenance,
                analysis_mode=analysis_mode,
                validation_report_path=validation_report_path,
                selected_plan=selected_plan,
                selected_evaluation=selected_evaluation,
            )
            identifier = plan_id(selected_plan)
            _write_manifest(
                manifest_store,
                config,
                status="complete_partial_analysis",
                started_at=started_at,
                details={
                    "selected_plan_id": identifier,
                    "analysis_mode": analysis_mode,
                    "selected_group_count": len(groups),
                    "discovered_group_count": discovered_group_count,
                    "derived_yaml_emitted": False,
                    "reason": "pilot subset cannot represent a full-model assignment",
                },
            )
            _progress(
                progress,
                "partial_analysis_completed",
                {
                    "plan_id": identifier,
                    "selected_groups": len(groups),
                    "discovered_groups": discovered_group_count,
                },
            )
            return WorkflowResult(
                status="complete_partial_analysis",
                selected_plan_id=identifier,
                output_dir=config.output_dir,
                derived_yaml=None,
                validation=selected_evaluation,
            )
        blocker_resolver = getattr(runtime, "plan_compilation_blockers", None)
        if callable(blocker_resolver):
            blockers = tuple(blocker_resolver(selected_plan.assignments))
        elif not bool(getattr(runtime, "backend_compilation_ready", True)):
            blockers = tuple(getattr(runtime, "backend_compilation_blockers", ()))
        else:
            blockers = ()
        if blockers:
            _write_provenance(
                store,
                config,
                result.provenance,
                analysis_mode=analysis_mode,
                validation_report_path=validation_report_path,
                selected_plan=selected_plan,
                selected_evaluation=selected_evaluation,
            )
            _write_manifest(
                manifest_store,
                config,
                status="backend_capability_blocked",
                started_at=started_at,
                details={
                    "selected_plan_id": plan_id(selected_plan),
                    "analysis_mode": analysis_mode,
                    "blockers": list(blockers),
                    "derived_yaml_emitted": False,
                },
            )
            return WorkflowResult(
                status="backend_capability_blocked",
                selected_plan_id=plan_id(selected_plan),
                output_dir=config.output_dir,
                derived_yaml=None,
                validation=selected_evaluation,
            )
        _progress(
            progress,
            "yaml_compilation_started",
            {"plan_id": plan_id(selected_plan)},
        )
        target_modules = tuple(
            module_name for group in groups for module_name in group.module_names
        )
        module_patterns = {
            module_name: pattern
            for group in groups
            for module_name, pattern in zip(
                group.module_names,
                group.effective_backend_patterns,
                strict=True,
            )
        }
        template_origins: Mapping[PrecisionFormat, str]
        if precision_templates is None:
            source_profile = getattr(runtime, "source_linear_profile", None)
            if source_profile is None:
                source_profile = profile_linear_source(
                    config.original_yaml,
                    target_modules,
                )
            template_resolution = resolve_precision_templates(
                source_profile,
                selected_plan.assignments.values(),
            )
            resolved_templates = template_resolution.templates
            template_origins = template_resolution.origins
        else:
            resolved_templates = precision_templates
            template_origins = {
                precision: "explicit_api_override" for precision in precision_templates
            }
        backend_inventory = tuple(
            getattr(runtime, "backend_module_inventory", full_module_inventory)
        )
        compilation = (compiler or ModelSlimPlanCompiler()).compile(
            config.original_yaml,
            {group.group_id: group.module_names for group in groups},
            selected_plan.assignments,
            resolved_templates,
            target_modules,
            model_module_names=backend_inventory,
            module_patterns=module_patterns,
        )
        derived_path = store.write_text(
            config.output.derived_yaml,
            compilation.derived_yaml,
        )
        source_text = config.original_yaml.read_text(encoding="utf-8")
        diff_text = "".join(
            difflib.unified_diff(
                source_text.splitlines(keepends=True),
                compilation.derived_yaml.splitlines(keepends=True),
                fromfile=str(config.original_yaml),
                tofile=str(derived_path),
            )
        )
        store.write_text(config.output.diff, diff_text)
        _write_provenance(
            store,
            config,
            result.provenance,
            analysis_mode=analysis_mode,
            validation_report_path=validation_report_path,
            selected_plan=selected_plan,
            selected_evaluation=selected_evaluation,
            compilation=compilation,
            template_origins=template_origins,
        )
        _write_manifest(
            manifest_store,
            config,
            status="complete_analysis_only",
            started_at=started_at,
            details={
                "selected_plan_id": plan_id(selected_plan),
                "analysis_mode": analysis_mode,
                "derived_yaml_sha256": compilation.sidecar.derived_sha256,
                "execution_readiness": "not_assessed",
            },
        )
        _progress(
            progress,
            "workflow_completed",
            {"plan_id": plan_id(selected_plan), "derived_yaml": str(derived_path)},
        )
        return WorkflowResult(
            status="complete_analysis_only",
            selected_plan_id=plan_id(selected_plan),
            output_dir=config.output_dir,
            derived_yaml=derived_path,
            validation=selected_evaluation,
        )
    except Exception as exc:
        _write_manifest(
            manifest_store,
            config,
            status="failed",
            started_at=started_at,
            details={
                "analysis_mode": analysis_mode,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise


def _write_manifest(
    store: ArtifactStore,
    config: RunConfig,
    *,
    status: str,
    started_at: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    store.write_json(
        config.output.run_manifest,
        {
            "schema_version": "quantplan.run-manifest/v1",
            "status": status,
            "started_at_utc": started_at,
            "updated_at_utc": _now(),
            "analysis_only": True,
            "execution_readiness": "not_assessed",
            "details": dict(details or {}),
        },
    )


def _write_provenance(
    store: ArtifactStore,
    config: RunConfig,
    runtime_provenance: Any,
    *,
    analysis_mode: str,
    validation_report_path: Path,
    selected_plan: QuantPlan | None = None,
    selected_evaluation: PlanEvaluation | None = None,
    compilation: PlanCompilationResult | None = None,
    template_origins: Mapping[PrecisionFormat, str] | None = None,
) -> None:
    runtime_payload = _jsonable(runtime_provenance)
    normalized_config = _jsonable(config)
    run_config_binding = {
        "sha256": _canonical_sha256(normalized_config),
        "hash_scope": "normalized_run_config",
        "canonicalization": "json_utf8_sort_keys_compact_v1",
        "normalized": normalized_config,
    }
    validation_binding = {
        "path": str(config.output.validation_report),
        "sha256": source_sha256(validation_report_path),
        "size_bytes": validation_report_path.stat().st_size,
        "hash_scope": "exact_artifact_bytes",
    }
    dataset_binding = _dataset_binding(config, runtime_payload)
    evidence_binding: dict[str, Any] = {
        "analysis_mode": analysis_mode,
        "selected_plan_id": plan_id(selected_plan) if selected_plan else None,
        "context_id": selected_plan.context_id if selected_plan else None,
        "validation_report": validation_binding,
        "run_config": run_config_binding,
        "dataset": dataset_binding,
    }
    evidence_binding["binding_sha256"] = _canonical_sha256(evidence_binding)

    payload: dict[str, Any] = {
        "schema_version": "quantplan.provenance/v1",
        "evidence_binding": evidence_binding,
        "runtime": runtime_payload,
    }
    if selected_evaluation is not None:
        payload["selected_evaluation"] = {
            "plan_id": selected_evaluation.plan_id,
            "context_id": selected_evaluation.context_id,
        }
    if compilation is not None:
        payload["modelslim_compilation"] = compilation.sidecar.to_dict()
        payload["derived_yaml_guard"] = _guard_payload(
            compilation.output_guard,
            guard_scope="derived_yaml",
        )
        payload["template_origins"] = {
            precision.value: origin
            for precision, origin in sorted(
                (template_origins or {}).items(),
                key=lambda item: item[0].value,
            )
        }
    store.write_json(config.output.provenance, payload)


def _dataset_binding(
    config: RunConfig,
    runtime_payload: Any,
) -> dict[str, Any]:
    if isinstance(runtime_payload, Mapping):
        runtime_hash = runtime_payload.get("dataset_sha256")
        if isinstance(runtime_hash, str) and runtime_hash:
            return {
                "sha256": runtime_hash,
                "source": "runtime_provenance.dataset_sha256",
                "hash_scope": "runtime_selected_samples",
            }
    if config.data_path.is_file():
        return {
            "sha256": source_sha256(config.data_path),
            "source": "run_config.data_path",
            "hash_scope": "exact_input_file_bytes",
        }
    return {
        "sha256": None,
        "source": "unavailable",
        "hash_scope": "unavailable",
    }


def _measurement_payload(
    item: (
        InterventionMeasurement
        | ApproximateMeasurement
        | StatisticalMeasurement
        | DirectionalMeasurement
    ),
) -> dict[str, Any]:
    payload = _jsonable(item)
    if isinstance(item, InterventionMeasurement):
        payload["delta_nll"] = item.delta_nll
        payload["evidence_role"] = "candidate-specific whole-model intervention"
    elif isinstance(item, StatisticalMeasurement):
        payload["evidence_role"] = "factorized gradient statistics; no block replay"
    elif isinstance(item, DirectionalMeasurement):
        payload["evidence_role"] = _DIRECTIONAL_EVIDENCE_ROLE
    else:
        payload["evidence_role"] = "gradient-weighted local approximation"
    return payload


def _evaluation_payload(item: PlanEvaluation) -> dict[str, Any]:
    payload = _jsonable(item)
    payload["delta_nll"] = item.delta_nll
    return payload


def _plan_payload(plan: QuantPlan) -> dict[str, Any]:
    return {
        "plan_id": plan_id(plan),
        "assignments": {
            key: precision.value for key, precision in sorted(plan.assignments.items())
        },
        "context_id": plan.context_id,
        "estimated_cost": plan.estimated_cost,
        "proposal_score": plan.proposal_score,
        "warnings": list(plan.warnings),
    }


def _guard_payload(result: Any, *, guard_scope: str) -> dict[str, Any]:
    finding_counts: dict[str, int] = {}
    for finding in result.findings:
        status = finding.status.value
        finding_counts[status] = finding_counts.get(status, 0) + 1
    return {
        "schema_version": "quantplan.yaml-audit/v1",
        "guard_scope": guard_scope,
        "source_path": result.source_path,
        "source_sha256": result.source_sha256,
        "source_size_bytes": result.source_size_bytes,
        "manifest_id": result.manifest_id,
        "static_policy_passed": result.static_policy_passed,
        "execution_readiness": result.execution_readiness,
        "execution_ready": result.execution_ready,
        "summary": {
            "finding_count": len(result.findings),
            "finding_counts": finding_counts,
            "issue_count": len(result.issues),
            "blocking_issue_count": len(result.blocking_issues),
            "opaque_field_count": len(result.opaque_fields),
            "blocked_field_count": len(result.blocked_fields),
            "pattern_expansion_count": len(result.pattern_expansions),
        },
        "findings": [
            {
                "path": finding.path,
                "status": finding.status.value,
            }
            for finding in result.findings
        ],
        "issues": [
            {
                "code": issue.code,
                "path": issue.path,
                "message": issue.message,
                "severity": issue.severity.value,
            }
            for issue in result.issues
        ],
        "pattern_expansions": [
            {
                "field_path": item.field_path,
                "pattern": item.pattern,
                "matched_modules": list(item.matched_modules),
            }
            for item in result.pattern_expansions
        ],
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {_json_key(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    raise TypeError(f"cannot serialize {type(value).__name__} as a run artifact")


def _json_key(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(
    callback: Callable[[str, Mapping[str, Any]], None] | None,
    stage: str,
    details: Mapping[str, Any],
) -> None:
    if callback is not None:
        callback(stage, details)


def _measurement_progress(
    callback: Callable[[str, Mapping[str, Any]], None] | None,
) -> Callable[[str, int, int, str, str], None] | None:
    if callback is None:
        return None

    def emit(stage: str, current: int, total: int, group_id: str, precision: str) -> None:
        callback(
            stage,
            {
                "current": current,
                "total": total,
                "group_id": group_id,
                "precision": precision,
            },
        )

    return emit


def _prepare_output(config: RunConfig, *, overwrite: bool) -> None:
    # Resolve every configured artifact through the same confinement boundary
    # used for writes.  Looking only at the lexical ``output_dir / relative``
    # path would let an intermediate symlink redirect ``--overwrite`` outside
    # the run directory before ``ArtifactStore.write_*`` gets a chance to
    # reject it.
    confinement = ArtifactStore(config.output_dir, overwrite=overwrite)
    paths = tuple(
        confinement.path(relative)
        for relative in (
            config.output.model_inventory,
            config.output.risk_evidence,
            config.output.candidate_plan,
            config.output.validation_report,
            config.output.derived_yaml,
            config.output.provenance,
            config.output.diff,
            config.output.yaml_audit,
            config.output.run_manifest,
        )
    )
    existing = tuple(path for path in paths if path.exists() or path.is_symlink())
    if existing and not overwrite:
        listed = ", ".join(str(path) for path in existing)
        raise ArtifactError(
            f"output artifacts already exist: {listed}; use --overwrite to replace them"
        )
    if not overwrite:
        return
    for path in existing:
        if path.is_dir() and not path.is_symlink():
            raise ArtifactError(f"configured artifact path is a directory: {path}")
        path.unlink(missing_ok=True)


def source_sha256(path: str | Path) -> str:
    """Hash one source file for external provenance integrations."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "AnalysisRuntimeLike",
    "WorkflowResult",
    "execute_workflow",
    "source_sha256",
]
