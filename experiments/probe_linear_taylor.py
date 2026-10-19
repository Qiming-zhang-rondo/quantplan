"""Experimental Linear-local Taylor contraction; this is not a production mode.

One model F+B per calibration batch, plus local candidate Linear matmuls computed
inside backward hooks. Unlike block replay, norm/nonlinearity/attention operations
are not rerun for each candidate. It still does candidate-dependent computation.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import compare_fast_stats as bench
import torch

from quantplan.domain.models import PrecisionFormat
from quantplan.runtime import TorchAnalysisError, TorchAnalysisRuntime
from quantplan.runtime import torch_analysis as ta


@dataclass(frozen=True)
class TaylorMeasurement:
    group_id: str
    precision: PrecisionFormat
    estimated_risk: float
    signed_mean: float
    rms_effect: float
    token_count: int


@dataclass(frozen=True)
class SharedTaylorMeasurement:
    group_id: str
    precision: PrecisionFormat
    estimated_risk: float
    signed_mean: float
    weight_signed: float
    activation_signed: float
    token_count: int


@dataclass(frozen=True)
class ProductionTaylorView:
    group_id: str
    precision: PrecisionFormat
    estimated_risk: float
    signed_mean: float
    token_count: int
    solver_shifted_risk: float
    ranking_offset: float


@dataclass(frozen=True)
class TaylorResult:
    measurements: tuple[TaylorMeasurement | SharedTaylorMeasurement | ProductionTaylorView, ...]
    provenance: object
    counts: dict[str, int]
    baseline_nll: float

    def to_dict(self):
        return {"measurements": [
            {**asdict(row), "precision": row.precision.value} for row in self.measurements
        ]}


def run_linear_taylor(runtime, precisions, *, token_chunk=256, weight_chunk=256):
    """Retain baseline boundary tensors; compute and immediately reduce each delta.

    The peak extra candidate delta is token_chunk × weight_chunk, not all model
    activations. Baseline input/output tensor references stay live until backward.
    Quantized weights are re-created per local call and are not retained globally.
    This prototype only covers resident, standard dense nn.Linear modules.
    """
    runtime._assert_context_integrity()
    groups = runtime._runtime_groups
    runtime._ensure_disjoint_linears(groups)
    if any(group.expert_targets for group in groups):
        raise TorchAnalysisError("Linear Taylor prototype does not support fused experts")
    modules = dict(runtime.model.named_modules())
    pairs = ta._supported_measurement_pairs(groups, precisions)
    group_precisions = {
        group.inventory.group_id: tuple(p for g, p in pairs if g is group) for group in groups
    }
    targets = {name: group.inventory.group_id
               for group in groups for name in group.linear_module_names}
    if any(type(modules[name]) is not torch.nn.Linear for name in targets):
        raise TorchAnalysisError("Linear Taylor prototype requires standard dense nn.Linear")
    if any(getattr(module, "gradient_checkpointing", False) for module in modules.values()):
        raise TorchAnalysisError("Linear Taylor prototype requires checkpointing disabled")
    stats = {(group.inventory.group_id, p): [0.0, 0.0] for group, p in pairs}
    counts = {"local_linear_candidates": 0, "local_linear_chunk_matmuls": 0,
              "local_block_replays": 0}
    handles, tensor_handles = [], []
    effects = {}
    seen_forward, seen_backward = set(), set()
    nll_sum = 0.0
    token_count = 0
    started = time.perf_counter()

    def enable(_module, args, kwargs):
        args, kwargs, _ = ta._enable_first_tensor_gradient(torch, args, kwargs)
        return args, kwargs

    def capture(name):
        group_id = targets[name]

        def hook(module, args, kwargs, output):
            if name in seen_forward:
                raise TorchAnalysisError(f"Repeated Linear invocation: {name}")
            seen_forward.add(name)
            x = args[0] if args else kwargs.get("input")
            if x.ndim != 3 or output.ndim != 3 or not output.requires_grad:
                raise TorchAnalysisError(f"Expected [batch, sequence, channel] gradients: {name}")
            saved_x = x.detach().reshape(-1, x.shape[-1])
            saved_y = output.detach().reshape(-1, output.shape[-1])

            def contract(gradient):
                if name in seen_backward:
                    raise TorchAnalysisError(f"Repeated Linear backward: {name}")
                seen_backward.add(name)
                flat_g = gradient.detach().reshape(-1, gradient.shape[-1])
                with torch.no_grad():
                    for precision in group_precisions[group_id]:
                        if precision is PrecisionFormat.BF16:
                            continue
                        counts["local_linear_candidates"] += 1
                        effect = torch.zeros(saved_x.shape[0], device=x.device, dtype=torch.float32)
                        bits = 4 if precision is PrecisionFormat.W4A8 else 8
                        for out_start in range(0, module.out_features, weight_chunk):
                            out_end = min(out_start + weight_chunk, module.out_features)
                            weight = ta._fake_quant_weight(
                                torch, module.weight[out_start:out_end], bits, check_finite=False,
                            )
                            bias = module.bias[out_start:out_end] if module.bias is not None else None
                            for token_start in range(0, saved_x.shape[0], token_chunk):
                                token_end = min(token_start + token_chunk, saved_x.shape[0])
                                quantized_x = ta._fake_quant_activation(
                                    torch, saved_x[token_start:token_end], 8, check_finite=False,
                                )
                                candidate = torch.nn.functional.linear(quantized_x, weight, bias)
                                delta = candidate.float() - saved_y[
                                    token_start:token_end, out_start:out_end
                                ].float()
                                local_g = flat_g[token_start:token_end, out_start:out_end].float()
                                effect[token_start:token_end].add_((delta * local_g).sum(-1))
                                counts["local_linear_chunk_matmuls"] += 1
                        if not bool(torch.isfinite(effect).all()):
                            raise TorchAnalysisError("Non-finite Linear Taylor effect")
                        # The loss already masks labels. A masked/context position may still
                        # affect later predictions through attention; retain its contraction.
                        key = (group_id, precision)
                        if key not in effects:
                            effects[key] = effect
                        else:
                            effects[key].add_(effect.to(effects[key].device))
                return gradient

            tensor_handles.append(output.register_hook(contract))

        return hook

    try:
        roots = dict.fromkeys(root for group in groups for root in group.block_module_names)
        for root in roots:
            handles.append(modules[root].register_forward_pre_hook(enable, with_kwargs=True))
        for name in targets:
            handles.append(modules[name].register_forward_hook(capture(name), with_kwargs=True))
        with ta._evaluation_mode(runtime.model), ta._temporary_freeze_parameters(runtime.model):
            for batch in runtime._batches():
                seen_forward.clear()
                seen_backward.clear()
                try:
                    with torch.enable_grad():
                        logits = runtime._forward_logits(batch)
                        loss, count = ta._nll_loss_sum_and_count(
                            torch, logits, batch["input_ids"], batch["attention_mask"],
                        )
                        if not count:
                            raise TorchAnalysisError("No valid next-token targets")
                        loss.backward()
                        nll_sum += float(loss.detach())
                        token_count += count
                        del logits, loss
                    if seen_forward != set(targets) or seen_backward != set(targets):
                        raise TorchAnalysisError("Missing forward/backward Linear coverage")
                    for key, value in effects.items():
                        stats[key][0] += float(value.sum())
                        stats[key][1] += float(value.square().sum())
                finally:
                    for handle in tensor_handles:
                        handle.remove()
                    tensor_handles.clear()
                    effects.clear()
    finally:
        for handle in reversed(handles):
            handle.remove()
        for handle in tensor_handles:
            handle.remove()
    if not token_count:
        raise TorchAnalysisError("Empty calibration")
    runtime._assert_context_integrity()
    runtime._baseline = ta._ForwardAggregate(nll_sum, token_count, time.perf_counter() - started)
    rows = []
    for group, precision in pairs:
        signed, squared = stats[(group.inventory.group_id, precision)]
        mean = signed / token_count
        rms = math.sqrt(max(0, squared / token_count))
        rows.append(TaylorMeasurement(group.inventory.group_id, precision, mean, mean,
                                      rms, token_count))
    return TaylorResult(
        tuple(rows), runtime._provenance(
            analysis_mode="probe-linear-taylor", evidence_metric="experimental.linear_taylor/v1",
            full_model_dataset_passes=1, backward_dataset_passes=1, local_block_replays=0,
        ), counts, nll_sum / token_count,
    )


def run_shared_taylor(runtime, precisions, *, token_chunk=256, weight_chunk=256):
    """R=G.T@Q8(X), score(q)=<R,Qq(W)-W>+<grad_x_branch,Q8(X)-X>.

    Each extra GEMM is shared by W4A8/W8A8. R lives only as an output-row chunk;
    parameter .grad buffers stay empty. Mixed delta-X/delta-W is retained, while
    rounding of the hypothetical BF16 Linear output is omitted. The already
    BF16-rounded branch input gradient is reused. No per-token RMS is invented.
    """
    runtime._assert_context_integrity()
    groups = runtime._runtime_groups
    runtime._ensure_disjoint_linears(groups)
    if any(group.expert_targets for group in groups):
        raise TorchAnalysisError("Shared Taylor prototype does not support fused experts")
    modules = dict(runtime.model.named_modules())
    targets = {name: group.inventory.group_id
               for group in groups for name in group.linear_module_names}
    if any(type(modules[name]) is not torch.nn.Linear for name in targets):
        raise TorchAnalysisError("Shared Taylor prototype requires dense nn.Linear")
    if any(getattr(module, "gradient_checkpointing", False) for module in modules.values()):
        raise TorchAnalysisError("Shared Taylor prototype requires checkpointing disabled")
    pairs = ta._supported_measurement_pairs(groups, precisions)
    group_precisions = {
        group.inventory.group_id: tuple(p for g, p in pairs if g is group) for group in groups
    }
    stats = {(group.inventory.group_id, p): [0.0, 0.0] for group, p in pairs}
    batch_weight, batch_activation = {}, {}
    handles, tensor_handles = [], []
    seen_forward, seen_input_backward, seen_output_backward = set(), set(), set()
    counts = {"local_block_replays": 0, "local_linear_candidate_matmuls": 0,
              "shared_gradient_chunk_matmuls": 0, "candidate_weight_chunk_scans": 0}
    started, nll_sum, token_count = time.perf_counter(), 0.0, 0

    def add_scalar(destination, key, value):
        if key not in destination:
            destination[key] = value
        else:
            destination[key].add_(value.to(destination[key].device))

    def enable(_module, args, kwargs):
        args, kwargs, _ = ta._enable_first_tensor_gradient(torch, args, kwargs)
        return args, kwargs

    def input_hook(name):
        group_id = targets[name]

        def capture(_module, args, kwargs):
            if name in seen_forward:
                raise TorchAnalysisError(f"Repeated Linear invocation: {name}")
            seen_forward.add(name)
            original = args[0] if args else kwargs.get("input")
            if original.ndim != 3 or not original.requires_grad:
                raise TorchAnalysisError(f"Expected Linear input gradient: {name}")
            branch = original.clone()
            saved = branch.detach().reshape(-1, branch.shape[-1])

            def activation_contract(gradient):
                if name in seen_input_backward:
                    raise TorchAnalysisError(f"Repeated input backward: {name}")
                seen_input_backward.add(name)
                with torch.no_grad():
                    gx = gradient.detach().reshape_as(saved)
                    value = torch.zeros((), dtype=torch.float32, device=branch.device)
                    for start in range(0, saved.shape[0], token_chunk):
                        original_x = saved[start:start + token_chunk]
                        delta_x = ta._fake_quant_activation(
                            torch, original_x, 8, check_finite=False,
                        ).float() - original_x.float()
                        value.add_((gx[start:start + token_chunk].float() * delta_x).sum())
                    for precision in group_precisions[group_id]:
                        if precision is not PrecisionFormat.BF16:
                            add_scalar(batch_activation, (group_id, precision), value.clone())
                return gradient

            tensor_handles.append(branch.register_hook(activation_contract))
            return ((branch, *args[1:]), kwargs) if args else (args, {**kwargs, "input": branch})

        return capture

    def output_hook(name):
        group_id = targets[name]

        def capture(module, args, kwargs, output):
            x = args[0] if args else kwargs.get("input")
            saved_x = x.detach().reshape(-1, x.shape[-1])

            def weight_contract(gradient):
                if name in seen_output_backward:
                    raise TorchAnalysisError(f"Repeated output backward: {name}")
                seen_output_backward.add(name)
                g = gradient.detach().reshape(-1, module.out_features)
                with torch.no_grad():
                    for out_start in range(0, module.out_features, weight_chunk):
                        out_end = min(out_start + weight_chunk, module.out_features)
                        r = torch.zeros(out_end - out_start, module.in_features,
                                        dtype=torch.float32, device=x.device)
                        for start in range(0, saved_x.shape[0], token_chunk):
                            quantized_x = ta._fake_quant_activation(
                                torch, saved_x[start:start + token_chunk], 8, check_finite=False,
                            )
                            local_g = g[start:start + token_chunk, out_start:out_end].float()
                            r.add_(local_g.T @ quantized_x.float())
                            counts["shared_gradient_chunk_matmuls"] += 1
                        original = module.weight[out_start:out_end]
                        for precision in group_precisions[group_id]:
                            if precision is PrecisionFormat.BF16:
                                continue
                            bits = 4 if precision is PrecisionFormat.W4A8 else 8
                            delta_w = ta._fake_quant_weight(
                                torch, original, bits, check_finite=False,
                            ).float() - original.float()
                            add_scalar(batch_weight, (group_id, precision), (r * delta_w).sum())
                            counts["candidate_weight_chunk_scans"] += 1
                return gradient

            tensor_handles.append(output.register_hook(weight_contract))

        return capture

    try:
        roots = dict.fromkeys(root for group in groups for root in group.block_module_names)
        for root in roots:
            handles.append(modules[root].register_forward_pre_hook(enable, with_kwargs=True))
        for name in targets:
            handles.append(modules[name].register_forward_pre_hook(input_hook(name), with_kwargs=True))
            handles.append(modules[name].register_forward_hook(output_hook(name), with_kwargs=True))
        with ta._evaluation_mode(runtime.model), ta._temporary_freeze_parameters(runtime.model):
            for batch in runtime._batches():
                seen_forward.clear()
                seen_input_backward.clear()
                seen_output_backward.clear()
                try:
                    with torch.enable_grad():
                        logits = runtime._forward_logits(batch)
                        loss, count = ta._nll_loss_sum_and_count(
                            torch, logits, batch["input_ids"], batch["attention_mask"],
                        )
                        if not count:
                            raise TorchAnalysisError("No valid next-token targets")
                        loss.backward()
                        nll_sum += float(loss.detach())
                        token_count += count
                        del logits, loss
                    if any(seen != set(targets) for seen in (
                        seen_forward, seen_input_backward, seen_output_backward,
                    )):
                        raise TorchAnalysisError("Missing Linear gradient coverage")
                    for component, destination in enumerate((batch_weight, batch_activation)):
                        for key, value in destination.items():
                            number = float(value)
                            if not math.isfinite(number):
                                raise TorchAnalysisError("Non-finite shared Taylor effect")
                            stats[key][component] += number
                finally:
                    for handle in tensor_handles:
                        handle.remove()
                    tensor_handles.clear()
                    batch_weight.clear()
                    batch_activation.clear()
    finally:
        for handle in reversed(handles):
            handle.remove()
        for handle in tensor_handles:
            handle.remove()
    if not token_count:
        raise TorchAnalysisError("Empty calibration")
    runtime._assert_context_integrity()
    runtime._baseline = ta._ForwardAggregate(nll_sum, token_count, time.perf_counter() - started)
    rows = []
    for group, precision in pairs:
        weight, activation = stats[(group.inventory.group_id, precision)]
        mean = (weight + activation) / token_count
        rows.append(SharedTaylorMeasurement(
            group.inventory.group_id, precision, mean, mean, weight / token_count,
            activation / token_count, token_count,
        ))
    return TaylorResult(
        tuple(rows), runtime._provenance(
            analysis_mode="probe-shared-taylor", evidence_metric="experimental.shared_taylor/v1",
            full_model_dataset_passes=1, backward_dataset_passes=1, local_block_replays=0,
        ), counts, nll_sum / token_count,
    )


def run_production_shared(runtime, precisions):
    """Benchmark actual runtime API; compare raw signed effects across groups.

    The production solver uses per-group shifts. Those shifted values must never
    be compared as cross-group importance scores. Auto-added BF16 rows are only
    retained when the benchmark requested BF16, keeping paired comparisons equal.
    """
    result = runtime.run_fast_taylor(precisions)
    rows = tuple(ProductionTaylorView(
        row.group_id, row.precision, row.mean_first_order_effect,
        row.mean_first_order_effect, row.token_count, row.estimated_risk, row.ranking_offset,
    ) for row in result.measurements if row.precision in precisions)
    provenance = result.provenance
    counts = {name: getattr(provenance, name) for name in (
        "local_block_replays", "local_gradient_projections", "local_gradient_gemms",
        "candidate_weight_scans", "candidate_affine_evaluations",
    )}
    return TaylorResult(rows, provenance, counts, result.baseline_nll)


def score_variant(result, variant):
    def value(row):
        if variant == "clipped":
            return max(0, row.signed_mean)
        if variant == "abs_rms_diagnostic":
            return abs(row.signed_mean) + row.rms_effect
        return row.signed_mean
    return SimpleNamespace(measurements=tuple(
        SimpleNamespace(group_id=row.group_id, precision=row.precision, estimated_risk=value(row))
        for row in result.measurements
    ))


def make_toy_runtime(model, tokenizer, samples, *, linear_groups):
    """Optional 7 decisions per block while retaining the same block replay root.

    This only changes discovery within an experiment-local context manager;
    production adapters and model architecture are untouched. It isolates the
    repeated-block cost seen with several decision groups inside one block.
    """
    original_discovery = ta._discover_runtime_groups

    def split_discovery(*args, **kwargs):
        groups = original_discovery(*args, **kwargs)
        modules = dict(model.named_modules())
        split = []
        for group in groups:
            for name in group.linear_module_names:
                inventory = replace(
                    group.inventory, group_id=f"{group.inventory.group_id}:{name.rsplit('.', 1)[-1]}",
                    module_names=(name,), parameter_count=modules[name].weight.numel(),
                    backend_patterns=(name,), operator_shapes={name: tuple(modules[name].weight.shape)},
                    kind="experiment.linear_group", warnings=(*group.inventory.warnings,
                                                              "experiment-only Linear subdivision"),
                )
                split.append(replace(group, inventory=inventory, linear_module_names=(name,)))
        return tuple(split)

    if linear_groups:
        with patch.object(ta, "_discover_runtime_groups", split_discovery):
            return TorchAnalysisRuntime(model, tokenizer, samples, batch_size=2)
    return TorchAnalysisRuntime(model, tokenizer, samples, batch_size=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--training-steps", type=int, nargs="+", default=[0, 200])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--without-attention", action="store_true")
    parser.add_argument("--token-chunk", type=int, default=256)
    parser.add_argument("--weight-chunk", type=int, default=256)
    parser.add_argument("--include-shared", action="store_true")
    parser.add_argument("--production-shared", action="store_true")
    parser.add_argument("--linear-groups", action="store_true")
    args = parser.parse_args()
    if min(args.repeats, args.token_chunk, args.weight_chunk, args.layers, args.width) < 1:
        parser.error("repeats/chunks/layers/width must be positive")
    torch.set_num_threads(1)
    precisions = (PrecisionFormat.W4A8, PrecisionFormat.W8A8)
    samples = ["quantization choices", "small models are a smoke test", "residual and norm", "abcde"]
    methods = ("run", "run_fast", "run_fast_stats", "run_linear_taylor")
    shared_method = "run_production_shared" if args.production_shared else "run_shared_taylor"
    if args.include_shared or args.production_shared:
        methods = (*methods, shared_method)
    rows = []
    for training_steps in args.training_steps:
        for seed_index, seed in enumerate(args.seeds):
            torch.manual_seed(seed)
            model = bench.TinyCausalLM(args.width, args.layers, not args.without_attention)
            tokenizer = bench.CharacterTokenizer()
            training = bench.train_tiny(model, tokenizer, samples, training_steps)
            runtime = make_toy_runtime(model.to(torch.bfloat16), tokenizer, samples,
                                       linear_groups=args.linear_groups)
            runtime.run_linear_taylor = lambda ps, active_runtime=runtime: run_linear_taylor(
                active_runtime, ps, token_chunk=args.token_chunk, weight_chunk=args.weight_chunk,
            )
            runtime.run_shared_taylor = lambda ps, active_runtime=runtime: run_shared_taylor(
                active_runtime, ps, token_chunk=args.token_chunk, weight_chunk=args.weight_chunk,
            )
            runtime.run_production_shared = lambda ps, active_runtime=runtime: run_production_shared(
                active_runtime, ps,
            )
            with torch.no_grad():
                for batch in runtime._batches():
                    runtime._forward_logits(batch)
            weights = {name: weight.detach().clone() for name, weight in model.state_dict().items()}
            flags = [parameter.requires_grad for parameter in model.parameters()]
            for repeat in range(args.repeats):
                results, metrics = {}, {}
                offset = (seed_index + repeat) % len(methods)
                for method in (*methods[offset:], *methods[:offset]):
                    results[method], data = bench.measure(runtime, method, precisions)
                    metrics[method] = {key: value for key, value in data.items() if key != "provenance"}
                    if method in {"run_linear_taylor", "run_shared_taylor", "run_production_shared"}:
                        metrics[method]["actual_counts"].update(results[method].counts)
                    assert all(torch.equal(weights[name], value)
                               for name, value in model.state_dict().items())
                    assert all(not m._forward_hooks and not m._forward_pre_hooks
                               for m in model.modules())
                    assert [p.requires_grad for p in model.parameters()] == flags
                    assert all(p.grad is None for p in model.parameters())
                proposals = {
                    "fast": results["run_fast"], "stats": results["run_fast_stats"],
                    **{f"linear_{variant}": score_variant(results["run_linear_taylor"], variant)
                       for variant in ("signed", "clipped", "abs_rms_diagnostic")},
                }
                if args.include_shared or args.production_shared:
                    proposals.update({
                        f"shared_{variant}": score_variant(results[shared_method], variant)
                        for variant in ("signed", "clipped")
                    })
                ranking = {key: bench.compare_ranking(results["run"], result, top_k=2)
                           for key, result in proposals.items()}
                plans = {
                    "exact": bench.validate_fixed_budget_plan(
                        runtime, results["run"], PrecisionFormat.W4A8, 0.5, exact=True,
                    ),
                    **{key: bench.validate_fixed_budget_plan(
                        runtime, result, PrecisionFormat.W4A8, 0.5, exact=False,
                    ) for key, result in proposals.items()},
                }
                row = {"seed": seed, "training_steps": training_steps, "repeat": repeat,
                       "training": training, "methods": metrics, "ranking": ranking,
                       "plans": plans}
                rows.append(row)
                print(json.dumps({
                    "seed": seed, "trained": training_steps, "repeat": repeat,
                    "linear_signed": ranking["linear_signed"]["w4a8"],
                    "plan_delta_nll": {key: value["delta_nll"] for key, value in plans.items()},
                }), flush=True)
    report = {
        "schema": "quantplan.linear-taylor-probe/v1",
        "platform": platform.platform(), "python_version": platform.python_version(),
        "torch_version": torch.__version__, "devices": ["cpu"],
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "notes": [
            "Prototype only, not connected to the production solver.",
            (
                "--production-shared invokes actual runtime.run_fast_taylor; ranking uses its "
                "mean_first_order_effect, never per-group-shifted estimated_risk. BF16 rows "
                "auto-added by the API are filtered to requested benchmark candidates."
            ),
            "One full F+B per batch; local Linear candidate matmuls still scale with candidates.",
            (
                "Optional shared variant uses R=G.T@Q8(X) once per Linear/batch and scans "
                "candidate delta-W; it reuses independent branch input gradients for delta-X."
            ),
            (
                "Shared variant keeps delta-X/delta-W cross terms, omits candidate output "
                "BF16 rounding, and does not compute or fabricate per-token RMS."
            ),
            "Baseline boundary input/output references stay live until their backward hook.",
            (
                "Parameter .grad buffers stay empty; shared R is a temporary gradient-like "
                "chunk. No candidate full-model/backward/block replays are computed."
            ),
            "The first-order signed score can be negative; clipping creates ties.",
            "abs(mean)+RMS is diagnostic, without a fitted mixing coefficient.",
            "3 fixed seeds, optional tiny fitting uses calibration texts; no pretrained LLM claim.",
            "One warmup dataset pass per model and full-plan checks are outside analysis timing.",
            (
                "With --linear-groups, each Linear is an independent decision retaining its "
                "containing Block as replay root; only experiment-local discovery is patched."
            ),
        ],
        "median_seconds": {
            str(steps): {
                method: statistics.median(row["methods"][method]["seconds"]
                                          for row in rows if row["training_steps"] == steps)
                for method in methods
            } for steps in args.training_steps
        },
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
