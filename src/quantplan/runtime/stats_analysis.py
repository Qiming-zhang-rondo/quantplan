"""One forward/backward sweep for candidate-independent Linear statistics.

For y = Wx, u = mean(x**2), v = mean((dL_sum/dy)**2), the weight
score is sum_ij v_i u_j (Q(W)_ij-W_ij)**2. The activation score uses
mean((Q(x)-x)**2) * mean((dL_sum/dx_branch)**2) per input channel.
Means are over all valid calibration positions, never squared batch means.
This factorizes correlations and discards cross terms; it is a ranking
heuristic, neither the empirical Fisher nor a predicted loss increase.

No parameter gradients, Hessian-vector products, candidate forwards, or
block replays are computed. Standard autograd activations are still required.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from quantplan.domain import PrecisionFormat, StatisticalMeasurement
from quantplan.runtime import torch_analysis as ta

STATS_METRIC_ID = "quantplan.factorized_gradient_noise/v1"
_MAX_CHUNK_ELEMENTS = 1_048_576


@dataclass
class _Moments:
    x2: Any
    gy2: Any
    gx2: Any
    error2: Any
    finite: Any
    forward_batch: int = -1
    backward_input_batch: int = -1
    backward_output_batch: int = -1


def _chunk_rows(width: int) -> int:
    return max(1, min(256, _MAX_CHUNK_ELEMENTS // width))


def _squared_sum(torch: Any, value: Any, valid: Any, destination: Any) -> None:
    rows = value.detach().reshape(-1, value.shape[-1])
    mask = valid.reshape(-1).to(device=value.device)
    size = _chunk_rows(rows.shape[-1])
    for start in range(0, rows.shape[0], size):
        work = rows[start : start + size].float()
        work = work.masked_fill(~mask[start : start + size, None], 0)
        destination.add_(work.square().sum(dim=0))


def _activation_error_sum(
    torch: Any, value: Any, valid: Any, destination: Any, finite: Any,
) -> None:
    rows = value.detach().reshape(-1, value.shape[-1])
    mask = valid.reshape(-1).to(device=value.device)
    size = _chunk_rows(rows.shape[-1])
    for start in range(0, rows.shape[0], size):
        original = rows[start : start + size]
        # Same per-token symmetric A8 QDQ (including cast back) as exact/replay.
        error = ta._fake_quant_activation(
            torch, original, 8, check_finite=False,
        ).float() - original.float()
        # Defer the host sync until the sweep finishes, including invalid padded
        # inputs which the next line masks out of the statistical moments.
        finite.logical_and_(torch.isfinite(error).all())
        error.masked_fill_(~mask[start : start + size, None], 0)
        destination.add_(error.square().sum(dim=0))


def _weight_score(torch: Any, weight: Any, bits: int, u: Any, v: Any) -> float:
    score = torch.zeros((), device=weight.device, dtype=torch.float32)
    size = _chunk_rows(weight.shape[-1])
    for start in range(0, weight.shape[0], size):
        original = weight.detach()[start : start + size]
        # Scan once per candidate, not once per batch; no full quantized clone.
        error = ta._fake_quant_weight(
            torch, original, bits, check_finite=False,
        ).float() - original.float()
        row_energy = (error.square() * u.unsqueeze(0)).sum(dim=-1)
        score.add_((row_energy * v[start : start + size]).sum())
    result = float(score.item())
    if not isfinite(result):
        raise ta.TorchAnalysisError("fast-stats weight score contains non-finite values")
    return result


def run_fast_stats(
    runtime: ta.TorchAnalysisRuntime,
    precisions: Sequence[PrecisionFormat | str],
    *,
    progress_callback: ta.ProgressCallback | None = None,
) -> ta.AnalysisRunResult:
    """Collect factorized moments, then score all candidates without replay.

    V1 covers standard dense Linear boundaries with [batch, sequence, channel]
    tensors. Fused experts, conditional/unobserved targets, repeated modules,
    and checkpoint recomputation fail explicitly, never as zero-risk evidence.
    """

    torch = runtime._torch
    normalized = tuple(ta._normalize_precision(value) for value in precisions)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("precisions must be a non-empty sequence without duplicates")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    runtime._assert_context_integrity()
    groups = runtime._runtime_groups
    pairs = ta._supported_measurement_pairs(groups, normalized)
    modules = dict(runtime.model.named_modules())
    if any(bool(getattr(module, "gradient_checkpointing", False)) for module in modules.values()):
        raise ta.TorchAnalysisError(
            "fast-stats requires gradient checkpointing disabled to guarantee no block replays"
        )
    unsupported = [
        target.module_name for group in groups for target in group.expert_targets
    ]
    if unsupported:
        raise ta.TorchAnalysisError(
            "fast-stats has no internal statistics adapter for fused expert targets: "
            + ", ".join(unsupported)
            + "; use --mode fast (replay) or an explicitly scoped dense Linear pilot"
        )
    target_names = tuple(name for group in groups for name in group.linear_module_names)
    runtime._ensure_disjoint_linears(groups)
    if not target_names:
        raise ta.TorchAnalysisError("fast-stats found no Linear targets")
    for name in target_names:
        if type(modules[name]) is not torch.nn.Linear:
            raise ta.TorchAnalysisError(f"fast-stats requires a standard nn.Linear: {name}")
    moments = {
        name: _Moments(
            x2=torch.zeros(modules[name].in_features, device=modules[name].weight.device, dtype=torch.float32),
            gy2=torch.zeros(modules[name].out_features, device=modules[name].weight.device, dtype=torch.float32),
            gx2=torch.zeros(modules[name].in_features, device=modules[name].weight.device, dtype=torch.float32),
            error2=torch.zeros(modules[name].in_features, device=modules[name].weight.device, dtype=torch.float32),
            finite=torch.ones((), device=modules[name].weight.device, dtype=torch.bool),
        )
        for name in target_names
    }
    active_valid: Any | None = None
    batch_index = -1
    handles: list[Any] = []
    tensor_handles: list[Any] = []
    nll_sum = 0.0
    token_count = 0
    started = time.perf_counter()

    def enable_gradient(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        args, kwargs, _ = ta._enable_first_tensor_gradient(torch, args, kwargs)
        if not ta._tree_contains_grad_tensor(torch, (args, kwargs)):
            raise ta.TorchAnalysisError("fast-stats could not enable block input gradients")
        return args, kwargs

    def input_hook(name: str):
        def collect(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
            value = args[0] if args else kwargs.get("input")
            if (
                not torch.is_tensor(value) or value.ndim != 3 or active_valid is None
                or tuple(value.shape[:2]) != tuple(active_valid.shape)
            ):
                raise ta.TorchAnalysisError(
                    f"fast-stats requires [batch, sequence, channel] Linear input: {name}"
                )
            stat = moments[name]
            if stat.forward_batch == batch_index:
                raise ta.TorchAnalysisError(f"fast-stats repeated/shared Linear invocation: {name}")
            stat.forward_batch = batch_index
            valid = active_valid
            invocation = batch_index
            # An identity clone makes this quantization site a distinct branch.
            # Registering on shared x would include sibling/residual gradients.
            branch = value.clone()
            if not branch.requires_grad:
                raise ta.TorchAnalysisError(f"fast-stats input has no gradient path: {name}")
            with torch.no_grad():
                _squared_sum(torch, branch, valid, stat.x2)
                _activation_error_sum(torch, branch, valid, stat.error2, stat.finite)

            def collect_gradient(gradient: Any):
                if stat.backward_input_batch == invocation:
                    raise ta.TorchAnalysisError(f"fast-stats repeated input backward: {name}")
                stat.backward_input_batch = invocation
                with torch.no_grad():
                    _squared_sum(torch, gradient, valid, stat.gx2)
                return gradient

            tensor_handles.append(branch.register_hook(collect_gradient))
            if args:
                return (branch, *args[1:]), kwargs
            return args, {**kwargs, "input": branch}

        return collect

    def output_hook(name: str):
        def collect(_module: Any, _args: Any, output: Any):
            if (
                not torch.is_tensor(output) or output.ndim != 3
                or not output.requires_grad or active_valid is None
                or tuple(output.shape[:2]) != tuple(active_valid.shape)
            ):
                raise ta.TorchAnalysisError(f"fast-stats output has no valid gradient path: {name}")
            stat = moments[name]
            valid = active_valid
            invocation = batch_index

            def collect_gradient(gradient: Any):
                if stat.backward_output_batch == invocation:
                    raise ta.TorchAnalysisError(f"fast-stats repeated output backward: {name}")
                stat.backward_output_batch = invocation
                with torch.no_grad():
                    _squared_sum(torch, gradient, valid, stat.gy2)
                return gradient

            tensor_handles.append(output.register_hook(collect_gradient))

        return collect

    try:
        roots = dict.fromkeys(root for group in groups for root in group.block_module_names)
        for root in roots:
            handles.append(modules[root].register_forward_pre_hook(enable_gradient, with_kwargs=True))
        for name in target_names:
            handles.append(modules[name].register_forward_pre_hook(input_hook(name), with_kwargs=True))
            handles.append(modules[name].register_forward_hook(output_hook(name)))
        total_batches = (len(runtime.samples) + runtime.batch_size - 1) // runtime.batch_size
        with ta._evaluation_mode(runtime.model), ta._temporary_freeze_parameters(runtime.model):
            for batch_index, batch in enumerate(runtime._batches()):
                active_valid = ta._block_token_mask(torch, batch["attention_mask"], batch["input_ids"].shape)
                try:
                    with torch.enable_grad():
                        logits = runtime._forward_logits(batch)
                        loss, count = ta._nll_loss_sum_and_count(
                            torch, logits, batch["input_ids"], batch["attention_mask"],
                        )
                        if not count:
                            raise ta.TorchAnalysisError("fast-stats batch has no valid next-token targets")
                        loss.backward()
                        nll_sum += float(loss.detach().item())
                        token_count += count
                        del logits, loss
                    for name, stat in moments.items():
                        if (
                            stat.forward_batch != batch_index
                            or stat.backward_input_batch != batch_index
                            or stat.backward_output_batch != batch_index
                        ):
                            raise ta.TorchAnalysisError(
                                f"fast-stats missing forward/backward coverage for {name}; "
                                "unobserved targets cannot be treated as insensitive"
                            )
                finally:
                    for handle in tensor_handles:
                        handle.remove()
                    tensor_handles.clear()
                    active_valid = None
                if progress_callback is not None:
                    progress_callback("stats_batch_complete", batch_index + 1, total_batches, "", "")
    finally:
        for handle in reversed(handles):
            handle.remove()
        for handle in tensor_handles:
            handle.remove()
    if token_count <= 0:
        raise ta.TorchAnalysisError("fast-stats produced no valid next-token targets")
    # Keep only small per-channel vectors; no activation dataset or weight-grad cache.
    with torch.no_grad():
        for stat in moments.values():
            if not bool(stat.finite):
                raise ta.TorchAnalysisError("fast-stats activations contain non-finite values")
            for value in (stat.x2, stat.gy2, stat.gx2, stat.error2):
                value.div_(token_count)
                if not bool(torch.isfinite(value).all()):
                    raise ta.TorchAnalysisError("fast-stats moments contain non-finite values")
    measurements = []
    for current, (group, precision) in enumerate(pairs, start=1):
        weight_risk = 0.0
        activation_risk = 0.0
        if precision is not PrecisionFormat.BF16:
            bits = 4 if precision is PrecisionFormat.W4A8 else 8
            with torch.no_grad():
                for name in group.linear_module_names:
                    stat = moments[name]
                    weight_risk += _weight_score(torch, modules[name].weight, bits, stat.x2, stat.gy2)
                    activation_risk += float((stat.error2 * stat.gx2).sum().item())
        measurements.append(StatisticalMeasurement(
            group_id=group.inventory.group_id,
            precision=precision,
            context_id=runtime.context_id,
            module_names=group.inventory.module_names,
            parameter_count=group.inventory.parameter_count,
            estimated_risk=weight_risk + activation_risk,
            token_count=token_count,
            elapsed_seconds=time.perf_counter() - started,
            metric_id=STATS_METRIC_ID,
            weight_risk=weight_risk,
            activation_risk=activation_risk,
            warnings=(
                ta.PROXY_SEMANTICS,
                "experimental factorized gradient-noise ranking; not delta NLL or exact Fisher",
                "no candidate forwards or block replays; standard autograd activation memory required",
                "channel, token and weight/activation cross terms omitted; routing changes not modeled",
                "global valid-token moments of summed NLL gradients; A8 at each Linear input",
                *(runtime.loader_warnings),
                *(group.inventory.warnings),
            ),
        ))
        if progress_callback is not None:
            progress_callback("measurement_complete", current, len(pairs), group.inventory.group_id, precision.value)
    elapsed = time.perf_counter() - started
    runtime._assert_context_integrity()
    runtime._baseline = ta._ForwardAggregate(nll_sum, token_count, elapsed)
    return ta.AnalysisRunResult(
        groups=runtime.groups,
        module_inventory=runtime.module_inventory,
        provenance=runtime._provenance(
            analysis_mode="fast-stats", evidence_metric=STATS_METRIC_ID,
            full_model_dataset_passes=1, backward_dataset_passes=1, local_block_replays=0,
        ),
        baseline_nll=nll_sum / token_count,
        token_count=token_count,
        measurements=tuple(measurements),
    )
