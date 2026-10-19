"""Signed Linear perturbations from one model F+B and shared local projections.

For candidate QX, QW and baseline output gradient G:
  <G, QX QW^T - X W^T> = <G^T QX, QW-W> + <G W, QX-X>.
All W4A8/W8A8 candidates reuse G^T Q8(X). Autograd provides the branch-local
input gradient G W. No candidate affine or Block is evaluated, but one extra
local gradient projection per Linear/batch is required (possibly many GEMMs
when chunked). Candidate affine output rounding and inter-site nonlinear
interactions are omitted. This is not observed delta NLL or a quality bound.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

from quantplan.domain import DirectionalMeasurement, PrecisionFormat
from quantplan.runtime import torch_analysis as ta

TAYLOR_METRIC_ID = "quantplan.shared_linear_taylor_shifted/v1"
_MAX_CHUNK_ELEMENTS = 1_048_576


def _work_dtype(torch: Any, value: Any) -> Any:
    return torch.float64 if value.dtype == torch.float64 else torch.float32


def _project_weight_effects(
    torch: Any,
    value: Any,
    gradient: Any,
    weight: Any,
    bits: Sequence[int],
    *,
    max_chunk_elements: int = _MAX_CHUNK_ELEMENTS,
    token_chunk_size: int = 256,
) -> tuple[dict[int, Any], Any, Any, int]:
    """Share one streamed G^T QX projection across candidates; never retain dW.

    Returns scalar weight effects, QX for the later input-gradient hook,
    a device-side finite flag, and the actual number of chunk GEMMs.
    A single quantized input buffer is retained until this Linear's input
    backward. FP32 projection temporaries are bounded by chunk dimensions.
    """
    if max_chunk_elements < 1 or token_chunk_size < 1:
        raise ValueError("chunk sizes must be positive")
    if len(set(bits)) != len(bits) or any(bit not in (4, 8) for bit in bits):
        raise ValueError("weight bits must be unique supported values (4 or 8)")
    dtype = _work_dtype(torch, value)
    rows = value.detach().reshape(-1, value.shape[-1])
    grads = gradient.detach().reshape(-1, gradient.shape[-1])
    if rows.shape[0] != grads.shape[0] or weight.shape != (grads.shape[1], rows.shape[1]):
        raise ta.TorchAnalysisError("fast-taylor projection shape mismatch")
    token_chunk = max(1, min(token_chunk_size, max_chunk_elements // rows.shape[1]))
    output_chunk = max(1, min(256, max_chunk_elements // rows.shape[1]))
    quantized = torch.empty_like(rows)
    finite = torch.ones((), dtype=torch.bool, device=value.device)
    scores = {bit: torch.zeros((), dtype=dtype, device=value.device) for bit in bits}
    for start in range(0, rows.shape[0], token_chunk):
        target = quantized[start : start + token_chunk]
        target.copy_(ta._fake_quant_activation(
            torch, rows[start : start + token_chunk], 8, check_finite=False,
        ))
        finite.logical_and_(torch.isfinite(target).all())
    gemms = 0
    for start in range(0, weight.shape[0], output_chunk):
        original = weight.detach()[start : start + output_chunk]
        projection = torch.zeros(original.shape, dtype=dtype, device=value.device)
        for token in range(0, rows.shape[0], token_chunk):
            local_gradient = grads[token : token + token_chunk, start : start + output_chunk]
            projection.addmm_(
                local_gradient.to(dtype).transpose(0, 1),
                quantized[token : token + token_chunk].to(dtype),
            )
            gemms += 1
        for bit in bits:
            delta = ta._fake_quant_weight(
                torch, original, bit, check_finite=False,
            ).to(dtype) - original.to(dtype)
            scores[bit].add_((projection * delta).sum())
    for score in scores.values():
        finite.logical_and_(torch.isfinite(score))
    return scores, quantized.reshape(value.shape), finite, gemms


def _activation_effect(
    torch: Any, value: Any, quantized: Any, gradient: Any,
    *, max_chunk_elements: int = _MAX_CHUNK_ELEMENTS,
) -> Any:
    if max_chunk_elements < 1:
        raise ValueError("chunk size must be positive")
    dtype = _work_dtype(torch, value)
    rows = value.detach().reshape(-1, value.shape[-1])
    qrows = quantized.reshape(rows.shape)
    grads = gradient.detach().reshape(rows.shape)
    total = torch.zeros((), dtype=dtype, device=value.device)
    size = max(1, min(256, max_chunk_elements // rows.shape[1]))
    for start in range(0, rows.shape[0], size):
        delta = qrows[start : start + size].to(dtype) - rows[start : start + size].to(dtype)
        total.add_((delta * grads[start : start + size].to(dtype)).sum())
    # No shifted label mask here: context positions can affect later losses.
    return total


@dataclass
class _Site:
    bits: tuple[int, ...]
    weight_sums: dict[int, Any]
    activation_sum: Any
    finite: Any
    forward_batch: int = -1
    output_batch: int = -1
    input_batch: int = -1
    value: Any = None
    quantized: Any = None
    version: int = -1


def run_fast_taylor(
    runtime: ta.TorchAnalysisRuntime,
    precisions: Sequence[PrecisionFormat | str],
    *, progress_callback: ta.ProgressCallback | None = None,
) -> ta.AnalysisRunResult:
    torch = runtime._torch
    normalized = tuple(ta._normalize_precision(value) for value in precisions)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("precisions must be a non-empty sequence without duplicates")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    runtime._assert_context_integrity()
    groups = runtime._runtime_groups
    # The BF16 reference must receive the same solver shift as other candidates.
    pairs = ta._supported_measurement_pairs(
        groups, tuple(dict.fromkeys((*normalized, PrecisionFormat.BF16))),
    )
    runtime._ensure_disjoint_linears(groups)
    modules = dict(runtime.model.named_modules())
    if any(bool(getattr(module, "gradient_checkpointing", False)) for module in modules.values()):
        raise ta.TorchAnalysisError("fast-taylor requires gradient checkpointing disabled")
    if any(group.expert_targets for group in groups):
        raise ta.TorchAnalysisError(
            "fast-taylor has no fused expert statistics adapter; use fast or a dense Linear pilot"
        )
    target_names = tuple(name for group in groups for name in group.linear_module_names)
    if not target_names:
        raise ta.TorchAnalysisError("fast-taylor found no Linear targets")
    sites: dict[str, _Site] = {}
    for group in groups:
        bits = tuple(dict.fromkeys(
            4 if precision is PrecisionFormat.W4A8 else 8
            for candidate_group, precision in pairs
            if candidate_group is group and precision is not PrecisionFormat.BF16
        ))
        for name in group.linear_module_names:
            module = modules[name]
            if type(module) is not torch.nn.Linear:
                raise ta.TorchAnalysisError(f"fast-taylor requires a standard nn.Linear: {name}")
            dtype = _work_dtype(torch, module.weight)
            sites[name] = _Site(
                bits=bits,
                weight_sums={bit: torch.zeros((), device=module.weight.device, dtype=dtype) for bit in bits},
                activation_sum=torch.zeros((), device=module.weight.device, dtype=dtype),
                finite=torch.ones((), device=module.weight.device, dtype=torch.bool),
            )
    handles: list[Any] = []
    tensor_handles: list[Any] = []
    batch_index = -1
    active_shape: tuple[int, ...] | None = None
    nll_sum = 0.0
    token_count = 0
    projections = 0
    gemms = 0
    scans = 0
    started = time.perf_counter()

    def enable(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        args, kwargs, _ = ta._enable_first_tensor_gradient(torch, args, kwargs)
        if not ta._tree_contains_grad_tensor(torch, (args, kwargs)):
            raise ta.TorchAnalysisError("fast-taylor could not enable block input gradients")
        return args, kwargs

    def check_value(site: _Site, name: str) -> None:
        if site.value is None or site.value._version != site.version:
            raise ta.TorchAnalysisError(f"fast-taylor cached input changed in-place: {name}")

    def pre_hook(name: str):
        def capture(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
            value = args[0] if args else kwargs.get("input")
            if (not torch.is_tensor(value) or value.ndim != 3 or active_shape is None
                    or tuple(value.shape[:2]) != active_shape):
                raise ta.TorchAnalysisError(f"fast-taylor requires [batch, sequence, channel]: {name}")
            site = sites[name]
            if site.forward_batch == batch_index:
                raise ta.TorchAnalysisError(f"fast-taylor repeated/shared Linear invocation: {name}")
            site.forward_batch = batch_index
            invocation = batch_index
            # Isolate dL/dX for THIS Linear, not sibling/residual consumers.
            branch = value.clone()
            if not branch.requires_grad:
                raise ta.TorchAnalysisError(f"fast-taylor input has no gradient path: {name}")
            site.value = branch.detach()
            site.version = branch._version

            def contract_input(gradient: Any):
                if site.input_batch == invocation:
                    raise ta.TorchAnalysisError(f"fast-taylor repeated input backward: {name}")
                site.input_batch = invocation
                check_value(site, name)
                if site.bits:
                    if site.quantized is None:
                        raise ta.TorchAnalysisError(f"fast-taylor missing output gradient: {name}")
                    with torch.no_grad():
                        effect = _activation_effect(torch, site.value, site.quantized, gradient)
                        site.activation_sum.add_(effect)
                        site.finite.logical_and_(torch.isfinite(effect))
                site.quantized = None
                site.value = None
                return gradient

            tensor_handles.append(branch.register_hook(contract_input))
            return ((branch, *args[1:]), kwargs) if args else (args, {**kwargs, "input": branch})

        return capture

    def post_hook(name: str):
        def capture(module: Any, _args: Any, output: Any):
            if (not torch.is_tensor(output) or output.ndim != 3 or not output.requires_grad
                    or active_shape is None or tuple(output.shape[:2]) != active_shape):
                raise ta.TorchAnalysisError(f"fast-taylor output has no valid gradient path: {name}")
            site = sites[name]
            invocation = batch_index

            def contract_output(gradient: Any):
                nonlocal projections, gemms, scans
                if site.output_batch == invocation:
                    raise ta.TorchAnalysisError(f"fast-taylor repeated output backward: {name}")
                site.output_batch = invocation
                check_value(site, name)
                if site.bits:
                    with torch.no_grad():
                        effects, site.quantized, finite, chunks = _project_weight_effects(
                            torch, site.value, gradient, module.weight, site.bits,
                        )
                        site.finite.logical_and_(finite)
                        for bit, effect in effects.items():
                            site.weight_sums[bit].add_(effect)
                    projections += 1
                    gemms += chunks
                    scans += len(site.bits)
                return gradient

            tensor_handles.append(output.register_hook(contract_output))

        return capture

    try:
        for root in dict.fromkeys(root for group in groups for root in group.block_module_names):
            handles.append(modules[root].register_forward_pre_hook(enable, with_kwargs=True))
        for name in sites:
            handles.append(modules[name].register_forward_pre_hook(pre_hook(name), with_kwargs=True))
            handles.append(modules[name].register_forward_hook(post_hook(name)))
        total_batches = (len(runtime.samples) + runtime.batch_size - 1) // runtime.batch_size
        with ta._evaluation_mode(runtime.model), ta._temporary_freeze_parameters(runtime.model):
            for batch_index, batch in enumerate(runtime._batches()):
                active_shape = tuple(batch["input_ids"].shape)
                try:
                    with torch.enable_grad():
                        logits = runtime._forward_logits(batch)
                        loss, count = ta._nll_loss_sum_and_count(
                            torch, logits, batch["input_ids"], batch["attention_mask"],
                        )
                        if not count:
                            raise ta.TorchAnalysisError("fast-taylor batch has no valid targets")
                        loss.backward()
                        nll_sum += float(loss.detach().item())
                        token_count += count
                        del logits, loss
                    for name, site in sites.items():
                        if (site.forward_batch != batch_index or site.output_batch != batch_index
                                or site.input_batch != batch_index):
                            raise ta.TorchAnalysisError(f"fast-taylor missing forward/backward coverage: {name}")
                finally:
                    for handle in tensor_handles:
                        handle.remove()
                    tensor_handles.clear()
                    for site in sites.values():
                        site.value = site.quantized = None
                    active_shape = None
                if progress_callback is not None:
                    progress_callback("taylor_batch_complete", batch_index + 1, total_batches, "", "")
    finally:
        for handle in reversed(handles):
            handle.remove()
        for handle in tensor_handles:
            handle.remove()
        for site in sites.values():
            site.value = site.quantized = None
    if token_count <= 0 or not isfinite(nll_sum):
        raise ta.TorchAnalysisError("fast-taylor produced invalid calibration loss")
    # A handful of scalar transfers per module, never per weight/token chunk.
    values = {}
    for name, site in sites.items():
        if not bool(site.finite):
            raise ta.TorchAnalysisError(f"fast-taylor non-finite evidence: {name}")
        values[name] = {
            bit: float((effect + site.activation_sum).item()) / token_count
            for bit, effect in site.weight_sums.items()
        }
    scores = {
        (group.inventory.group_id, precision): (
            0.0 if precision is PrecisionFormat.BF16 else sum(
                values[name][4 if precision is PrecisionFormat.W4A8 else 8]
                for name in group.linear_module_names
            )
        )
        for group, precision in pairs
    }
    offsets = {
        group.inventory.group_id: -min(0.0, *(
            score for (group_id, _precision), score in scores.items()
            if group_id == group.inventory.group_id
        ))
        for group in groups
    }
    elapsed = time.perf_counter() - started
    measurements = []
    for current, (group, precision) in enumerate(pairs, start=1):
        group_id = group.inventory.group_id
        signed = scores[(group_id, precision)]
        offset = offsets[group_id]
        measurements.append(DirectionalMeasurement(
            group_id=group_id, precision=precision, context_id=runtime.context_id,
            module_names=group.inventory.module_names, parameter_count=group.inventory.parameter_count,
            estimated_risk=signed + offset, token_count=token_count, elapsed_seconds=elapsed,
            metric_id=TAYLOR_METRIC_ID, mean_first_order_effect=signed, ranking_offset=offset,
            warnings=(
                ta.PROXY_SEMANTICS,
                "experimental signed first-order proposal; not observed delta NLL or a quality bound",
                "shifted risk is solver-only; compare raw signed effects across groups",
                "negative signed effects and zero shifted scores do not establish quality safety",
                "one shared local gradient projection per Linear/batch; no candidate affine or block replay",
                "candidate affine output rounding and inter-site nonlinear interactions omitted",
                "baseline branch input gradients retain model dtype; this is not FP32 end-to-end differentiation",
                "input boundary caches and standard autograd activations still require memory",
                "compute_kl does not affect ranking; complete plan validation still applies",
                *runtime.loader_warnings, *group.inventory.warnings,
            ),
        ))
        if progress_callback is not None:
            progress_callback("measurement_complete", current, len(pairs), group_id, precision.value)
    runtime._assert_context_integrity()
    runtime._baseline = ta._ForwardAggregate(nll_sum, token_count, elapsed)
    provenance = replace(
        runtime._provenance(
            analysis_mode="fast-taylor", evidence_metric=TAYLOR_METRIC_ID,
            full_model_dataset_passes=1, backward_dataset_passes=1, local_block_replays=0,
        ),
        local_gradient_projections=projections, local_gradient_gemms=gemms,
        candidate_weight_scans=scans, candidate_affine_evaluations=0,
    )
    return ta.AnalysisRunResult(
        groups=runtime.groups, module_inventory=runtime.module_inventory, provenance=provenance,
        baseline_nll=nll_sum / token_count, token_count=token_count, measurements=tuple(measurements),
    )
