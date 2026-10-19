"""Compare exact, replay-fast and statistics-only sensitivity on fixed inputs.

No weights are downloaded by the default synthetic experiment. Example:
    .venv/bin/python experiments/compare_fast_stats.py --output /tmp/fast-stats.json

Use --config with an existing QuantPlan config for an in-network representative
subset (analysis.max_groups / modules.include control its expense). This measures
analysis wall time, not deployment latency. Synthetic ranking results are not
evidence of Qwen/GLM/MoE accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from quantplan.domain.models import DirectionalMeasurement, PrecisionFormat
from quantplan.runtime import TorchAnalysisRuntime


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, texts, *, max_length=None, **_kwargs):
        rows = [[1 + ord(char) % 62 for char in text][:max_length] for text in texts]
        ids = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long)
        for index, row in enumerate(rows):
            ids[index, :len(row)] = torch.tensor(row)
        return {"input_ids": ids, "attention_mask": ids.ne(0).long()}


class TinyBlock(torch.nn.Module):
    def __init__(self, width: int, index: int, attention: bool):
        super().__init__()
        self.norm = torch.nn.RMSNorm(width)
        self.gate_proj = torch.nn.Linear(width, 2 * width, bias=False)
        self.up_proj = torch.nn.Linear(width, 2 * width, bias=False)
        self.down_proj = torch.nn.Linear(2 * width, width, bias=False)
        self.residual_scale = (0.15, 0.4, 0.9, 1.4)[index % 4]
        self.attention = attention
        if attention:
            self.q_proj = torch.nn.Linear(width, width, bias=False)
            self.k_proj = torch.nn.Linear(width, width, bias=False)
            self.v_proj = torch.nn.Linear(width, width, bias=False)
            self.o_proj = torch.nn.Linear(width, width, bias=False)

    def forward(self, hidden):
        x = self.norm(hidden)
        if self.attention:
            attended = torch.nn.functional.scaled_dot_product_attention(
                self.q_proj(x).unsqueeze(1), self.k_proj(x).unsqueeze(1),
                self.v_proj(x).unsqueeze(1), is_causal=True,
            ).squeeze(1)
            hidden = hidden + 0.2 * self.o_proj(attended)
        z = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        return hidden + self.residual_scale * self.down_proj(z)


class TinyCausalLM(torch.nn.Module):
    def __init__(self, width: int, layers: int, attention: bool):
        super().__init__()
        self.config = SimpleNamespace(model_type="quantplan_dense_test")
        self.embed = torch.nn.Embedding(64, width)
        self.layers = torch.nn.ModuleList(
            TinyBlock(width, index, attention) for index in range(layers)
        )
        self.final_norm = torch.nn.RMSNorm(width)
        self.lm_head = torch.nn.Linear(width, 64, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, *, input_ids, attention_mask):
        del attention_mask  # Right-padding and causal attention prevent leakage to valid tokens.
        hidden = self.embed(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        return SimpleNamespace(logits=self.lm_head(self.final_norm(hidden)))


def train_tiny(model, tokenizer, samples, steps):
    """Optional local fit: optimization and calibration texts deliberately coincide.

    This diagnoses behavior near a fitted model, not generalization or downstream
    task accuracy. Training work is reported separately from analysis timing.
    """
    if not steps:
        return {"steps": 0, "seconds": 0.0}
    batch = tokenizer(samples)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    valid = batch["attention_mask"][:, 1:].bool()
    started = time.perf_counter()
    initial = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(**batch).logits
        losses = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            batch["input_ids"][:, 1:].reshape(-1), reduction="none",
        ).reshape(valid.shape)
        loss = losses.masked_select(valid).mean()
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "steps": steps, "seconds": time.perf_counter() - started,
        "initial_fp32_nll": initial, "last_training_fp32_nll": float(loss.detach()),
        "uses_calibration_texts_for_toy_fit": True,
    }


def _ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        for index in order[cursor:end]:
            result[index] = (cursor + end - 1) / 2
        cursor = end
    return result


def spearman(left, right):
    if len(left) < 2:
        return None
    x, y = _ranks(left), _ranks(right)
    x = [value - statistics.mean(x) for value in x]
    y = [value - statistics.mean(y) for value in y]
    divisor = math.sqrt(sum(v * v for v in x) * sum(v * v for v in y))
    return sum(a * b for a, b in zip(x, y, strict=True)) / divisor if divisor else None


def _proposal_score(measurement, *, signed_mode=False):
    # Group-specific solver shifts cannot be compared across decision groups.
    if signed_mode or isinstance(measurement, DirectionalMeasurement):
        return measurement.mean_first_order_effect
    return measurement.estimated_risk


def compare_ranking(exact, approximate, *, top_k, signed_mode=False):
    observed = {(m.group_id, m.precision): m.delta_nll for m in exact.measurements}
    predicted = {(m.group_id, m.precision): _proposal_score(m, signed_mode=signed_mode)
                 for m in approximate.measurements}
    result = {}
    for precision in sorted({pair[1].value for pair in observed} - {"bf16"}):
        keys = sorted(key for key in observed if key[1].value == precision and key in predicted)
        positives = sorted(
            (key for key in keys if observed[key] > 0),
            key=lambda key: (-observed[key], key[0]),
        )[:top_k]
        suggested = set(sorted(keys, key=lambda key: (-predicted[key], key[0]))[:top_k])
        result[precision] = {
            "group_count": len(keys),
            "spearman_signed_delta_nll": spearman(
                [observed[key] for key in keys], [predicted[key] for key in keys]
            ),
            "spearman_absolute_delta_nll_diagnostic_only": spearman(
                [abs(observed[key]) for key in keys], [predicted[key] for key in keys]
            ),
            "positive_harm_top_k_recall": (
                len(suggested.intersection(positives)) / len(positives) if positives else None
            ),
            "positive_harm_top_k_count": len(positives),
            "positive_delta_nll_group_count": sum(observed[key] > 0 for key in keys),
            "top_k": top_k,
        }
    return result


def precision_monotonicity(result, *, exact, signed_mode=False):
    scores = {(m.group_id, m.precision): m.delta_nll if exact else
              _proposal_score(m, signed_mode=signed_mode)
              for m in result.measurements}
    groups = sorted({key[0] for key in scores if (key[0], PrecisionFormat.W4A8) in scores
                     and (key[0], PrecisionFormat.W8A8) in scores})
    return {
        "group_count": len(groups),
        "w4a8_ge_w8a8_count": sum(
            scores[(group, PrecisionFormat.W4A8)] >= scores[(group, PrecisionFormat.W8A8)]
            for group in groups
        ),
        "score": "observed_signed_delta_nll" if exact else (
            "mean_first_order_effect" if signed_mode or any(
                isinstance(m, DirectionalMeasurement) for m in result.measurements
            ) else "estimated_risk"
        ),
    }


def _sync(model):
    devices = {parameter.device for parameter in model.parameters()}
    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "npu" and hasattr(torch, "npu"):
            torch.npu.synchronize(device)


def measure(runtime, method, precisions):
    counts = {"full_forwards": 0, "backwards": 0, "block_forwards": 0}

    def forward_hook(_module, _args, output):
        counts["full_forwards"] += 1
        if output.logits.requires_grad:
            def backward_hook(grad):
                counts["backwards"] += 1
                return grad
            output.logits.register_hook(backward_hook)

    def block_hook(*_args):
        counts["block_forwards"] += 1

    modules = dict(runtime.model.named_modules())
    roots = {name for group in runtime._runtime_groups for name in group.block_module_names}
    handles = [runtime.model.register_forward_hook(forward_hook)]
    handles.extend(modules[name].register_forward_hook(block_hook) for name in roots)
    runtime._baseline = None  # Each method pays for its own baseline; do not share cached passes.
    cuda_devices = {p.device for p in runtime.model.parameters() if p.device.type == "cuda"}
    for device in cuda_devices:
        torch.cuda.reset_peak_memory_stats(device)
    _sync(runtime.model)
    started = time.perf_counter()
    try:
        result = getattr(runtime, method)(precisions)
        _sync(runtime.model)
        seconds = time.perf_counter() - started
    finally:
        for handle in reversed(handles):
            handle.remove()
    if method == "run_fast_taylor":
        counts.update({name: getattr(result.provenance, name) for name in (
            "local_block_replays", "local_gradient_projections", "local_gradient_gemms",
            "candidate_weight_scans", "candidate_affine_evaluations",
        )})
    return result, {
        "seconds": seconds,
        "actual_counts": counts,
        "peak_cuda_allocated_bytes": {
            str(device): torch.cuda.max_memory_allocated(device) for device in cuda_devices
        },
        "provenance": result.provenance.to_dict(),
        "measurements": result.to_dict()["measurements"],
    }


def validate_fixed_budget_plan(runtime, result, precision, fraction, *, exact, signed_mode=False):
    rows = [m for m in result.measurements if m.precision == precision]
    if not rows:
        return {"skipped": "requested plan precision unavailable for selected groups"}
    score = (lambda m: m.delta_nll) if exact else (
        lambda m: _proposal_score(m, signed_mode=signed_mode)
    )
    selected = sorted(rows, key=lambda m: (score(m), m.group_id))[
        :max(1, math.ceil(len(rows) * fraction))
    ]
    selected_ids = {m.group_id for m in selected}
    supported = {PrecisionFormat.BF16, PrecisionFormat.W8A8, PrecisionFormat.W4A8}
    assignments = {}
    for group in runtime.groups:
        allowed = set(group.allowed_formats) if group.allowed_formats is not None else supported
        allowed = allowed.intersection(supported)
        if group.group_id in selected_ids:
            if precision not in allowed:
                return {"skipped": f"requested precision is not legal for {group.group_id}"}
            assignments[group.group_id] = precision
        else:
            preferred = (PrecisionFormat.BF16, PrecisionFormat.W8A8,
                         group.source_precision, PrecisionFormat.W4A8)
            reference = next((value for value in preferred if value in allowed), None)
            if reference is None:
                return {"skipped": f"no legal, proxy-supported reference for {group.group_id}"}
            assignments[group.group_id] = reference
    measured = runtime.evaluate_plan(assignments)
    return {
        "assignments": {key: value.value for key, value in assignments.items()},
        "quantized_fraction": sum(value != PrecisionFormat.BF16 for value in assignments.values())
        / len(assignments),
        "baseline_nll": measured.baseline_nll,
        "candidate_nll": measured.candidate_nll,
        "delta_nll": measured.delta_nll,
        "token_count": measured.token_count,
        "note": "Fixed group-count budget, not a latency-optimal solver result.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--precisions", nargs="+", default=["w4a8", "w8a8"])
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--plan-fraction", type=float, default=0.5)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--include-taylor", action="store_true")
    args = parser.parse_args()
    if args.train_steps < 0 or (args.config and args.train_steps):
        parser.error("train-steps must be nonnegative and is only allowed for synthetic models")
    if args.repeats < 1 or args.top_k < 1 or not 0 < args.plan_fraction <= 1:
        parser.error("repeats/top-k must be positive; plan-fraction must be in (0, 1]")
    torch.set_num_threads(args.threads)
    precisions = tuple(PrecisionFormat(value) for value in args.precisions)
    if not any(value != PrecisionFormat.BF16 for value in precisions):
        parser.error("at least one non-BF16 candidate is required")
    configured_runtime = None
    if args.config:
        from quantplan.run_config import load_run_config
        from quantplan.runtime.factory import build_torch_runtime
        config = load_run_config(args.config)
        if args.max_groups is not None:
            config = replace(config, analysis=replace(config.analysis, max_groups=args.max_groups))
        configured_runtime = build_torch_runtime(config)

    rows = []
    methods = ("run", "run_fast", "run_fast_stats")
    if args.include_taylor:
        methods = (*methods, "run_fast_taylor")
    for seed_index, seed in enumerate(args.seeds):
        torch.manual_seed(seed)
        training = None
        if configured_runtime:
            runtime = configured_runtime
        else:
            model = TinyCausalLM(args.width, args.layers, args.attention)
            tokenizer = CharacterTokenizer()
            samples = ["quantization choices", "small models are a smoke test",
                       "residual and norm", "abcde"]
            training = train_tiny(model, tokenizer, samples, args.train_steps)
            runtime = TorchAnalysisRuntime(
                model.to(torch.bfloat16), tokenizer, samples,
                batch_size=args.batch_size, max_groups=args.max_groups,
            )
        # Warm up dtype kernels before timing and counts. This is an extra, reported pass.
        with torch.no_grad():
            for batch in runtime._batches():
                runtime._forward_logits(batch)
        for repeat in range(args.repeats):
            results, timing = {}, {}
            # Rotate ordering so caches/warmup do not systematically favor one method.
            offset = (seed_index + repeat) % len(methods)
            for method in (*methods[offset:], *methods[:offset]):
                results[method], timing[method] = measure(runtime, method, precisions)
            plan_precision = next(p for p in precisions if p != PrecisionFormat.BF16)
            plans = {
                method: validate_fixed_budget_plan(
                    runtime, results[method], plan_precision, args.plan_fraction, exact=method == "run",
                    signed_mode=method == "run_fast_taylor",
                ) for method in methods
            }
            row = {
                "seed": seed, "repeat": repeat, "methods": timing, "training": training,
                "ranking": {
                    method: compare_ranking(results["run"], results[method], top_k=args.top_k,
                                            signed_mode=method == "run_fast_taylor")
                    for method in methods[1:]
                },
                "plans": plans,
                "precision_monotonicity": {
                    method: precision_monotonicity(results[method], exact=method == "run",
                                                   signed_mode=method == "run_fast_taylor")
                    for method in methods
                },
            }
            rows.append(row)
            print(json.dumps({
                "seed": seed, "repeat": repeat,
                "seconds": {method: round(data["seconds"], 5) for method, data in timing.items()},
                "stats_ranking": row["ranking"]["run_fast_stats"],
            }), flush=True)
    report = {
        "schema": "quantplan.fast-stats-comparison/v1",
        "scope": "configured-model" if args.config else (
            "locally-fitted-dense-toy" if args.train_steps else "random-dense-toy"
        ),
        "torch_version": torch.__version__,
        "config": str(args.config) if args.config else None,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "notes": [
            (
                f"Platform: {platform.platform()}; Python {platform.python_version()}; "
                f"analysis devices: {sorted({str(p.device) for p in runtime.model.parameters()})}."
            ),
            "Warmup: one additional forward-only dataset pass per seed, outside measured counts.",
            "Each method gets an independent baseline; validation is outside analysis timing.",
            "Statistics risk is nonnegative proposal evidence, not predicted delta NLL.",
            "Signed delta NLL is the primary ranking target; absolute drift is diagnostic only.",
            "Random toy models do not establish pretrained LLM or MoE ranking quality.",
            "For --config, seeds do not retrain/change loaded weights or calibration samples.",
            (
                "--include-taylor benchmarks actual run_fast_taylor; its cross-group ranking "
                "and fixed-budget plan use raw mean_first_order_effect, never shifted solver risk."
            ),
            "No CPU peak-memory measurement is claimed; CUDA peaks are per-device allocator peaks.",
        ],
        "median_seconds": {
            method: statistics.median(row["methods"][method]["seconds"] for row in rows)
            for method in methods
        },
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
