"""Directional-score contracts, including cross-token and branch semantics."""

from types import SimpleNamespace

import pytest

from quantplan.domain.models import PrecisionFormat
from quantplan.runtime import TorchAnalysisError, TorchAnalysisRuntime
from quantplan.runtime import torch_analysis as ta

torch = pytest.importorskip("torch")

from test_fast_stats import Model, Tokenizer, make_runtime

PRECISIONS = (PrecisionFormat.W4A8, PrecisionFormat.W8A8, PrecisionFormat.BF16)


class SingleSiteModel(Model):
    def __init__(self, *, leak=0.0, temporal=False, mutate_input=False):
        super().__init__()
        self.target = torch.nn.Linear(6, 6, bias=True)
        self.leak = leak
        self.temporal = temporal
        self.mutate_input = mutate_input

    def forward(self, *, input_ids, attention_mask):
        self.calls += 1
        # A non-leaf permits deliberate inplace mutation in the rejection test.
        x = self.embed(input_ids).detach().requires_grad_(True) * 1.0
        hidden = self.target(x)
        if self.mutate_input:
            x.add_(0.25)
        hidden = hidden + self.leak * (x - x.detach())
        if self.temporal:
            hidden = (hidden * attention_mask.unsqueeze(-1)).cumsum(dim=1)
        return SimpleNamespace(logits=self.lm_head(hidden))


def single_runtime(**model_options):
    torch.manual_seed(71)
    model = SingleSiteModel(**model_options).to(torch.bfloat16)
    return make_runtime(model=model, groups={"target": "target"})


def _reference_single_site(runtime, precision, *, source_mask=None):
    """Independent scalar formula using retained branch and output gradients."""
    captured = {}

    def save(_module, args, output):
        captured["x"] = args[0]
        captured["y"] = output
        args[0].retain_grad()
        output.retain_grad()

    handle = runtime.model.target.register_forward_hook(save)
    try:
        with ta._evaluation_mode(runtime.model), ta._temporary_freeze_parameters(runtime.model):
            batch = next(runtime._batches())
            logits = runtime.model(**batch).logits
            loss, count = ta._nll_loss_sum_and_count(
                torch, logits, batch["input_ids"], batch["attention_mask"]
            )
            loss.backward()
        x = captured["x"].detach()
        gy = captured["y"].grad.detach().float()
        gx = captured["x"].grad.detach().float()
        if source_mask is not None:
            mask = source_mask.unsqueeze(-1)
            gy = gy * mask
            gx = gx * mask
        xq = ta._fake_quant_activation(torch, x, 8).float()
        weight = runtime.model.target.weight.detach()
        bits = 4 if precision is PrecisionFormat.W4A8 else 8
        dw = ta._fake_quant_weight(torch, weight, bits).float() - weight.float()
        projected = gy.reshape(-1, gy.shape[-1]).T @ xq.reshape(-1, xq.shape[-1])
        return float(((projected * dw).sum() + (gx * (xq - x.float())).sum()) / count)
    finally:
        handle.remove()


def test_taylor_one_sweep_and_restores_weights_gradients_and_training_flags():
    for precisions in ((PrecisionFormat.W4A8,), PRECISIONS):
        runtime = make_runtime()
        model = runtime.model
        model.train()
        model.layers[0].norm.eval()
        next(iter(model.parameters())).requires_grad_(False)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        flags = [parameter.requires_grad for parameter in model.parameters()]
        training = [module.training for module in model.modules()]
        result = runtime.run_fast_taylor(precisions)
        assert model.calls == 1
        assert [block.calls for block in model.layers] == [1, 1]
        assert result.provenance.analysis_mode == "fast-taylor"
        assert result.provenance.full_model_dataset_passes == 1
        assert result.provenance.backward_dataset_passes == 1
        assert result.provenance.local_block_replays == 0
        assert result.provenance.local_gradient_projections == 4
        assert result.provenance.local_gradient_gemms == 4
        assert result.provenance.candidate_weight_scans == 4 * sum(
            precision is not PrecisionFormat.BF16 for precision in precisions
        )
        assert result.provenance.candidate_affine_evaluations == 0
        assert [p.requires_grad for p in model.parameters()] == flags
        assert [m.training for m in model.modules()] == training
        assert all(parameter.grad is None for parameter in model.parameters())
        assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        for measurement in result.measurements:
            assert measurement.estimated_risk == pytest.approx(
                measurement.mean_first_order_effect + measurement.ranking_offset
            )
            assert measurement.estimated_risk >= 0
            assert not hasattr(measurement, "delta_nll")
            if measurement.precision is PrecisionFormat.BF16:
                assert measurement.mean_first_order_effect == 0


def test_taylor_candidate_offsets_preserve_signed_differences_including_bf16():
    runtime = make_runtime()
    result = runtime.run_fast_taylor(PRECISIONS)
    for group in result.groups:
        items = [item for item in result.measurements if item.group_id == group.group_id]
        offset = -min(0.0, *(item.mean_first_order_effect for item in items))
        for item in items:
            assert item.ranking_offset == pytest.approx(offset)
        for first in items:
            for second in items:
                assert first.estimated_risk - second.estimated_risk == pytest.approx(
                    first.mean_first_order_effect - second.mean_first_order_effect,
                    abs=1e-12,
                )


def test_taylor_matches_independent_single_site_contraction_and_is_branch_local():
    reference = _reference_single_site(single_runtime(), PrecisionFormat.W4A8)
    ordinary = single_runtime().run_fast_taylor(PRECISIONS).measurements
    leaking = single_runtime(leak=100.0).run_fast_taylor(PRECISIONS).measurements
    by_precision = {item.precision: item for item in ordinary}
    assert by_precision[PrecisionFormat.W4A8].mean_first_order_effect == pytest.approx(
        reference, rel=2e-5, abs=2e-8
    )
    for first, second in zip(ordinary, leaking, strict=True):
        assert first.mean_first_order_effect == pytest.approx(
            second.mean_first_order_effect, rel=1e-6, abs=1e-9
        )


def test_taylor_includes_context_positions_without_an_immediate_target():
    class HoleTokenizer(Tokenizer):
        def __call__(self, texts, **_kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4]] * len(texts)),
                "attention_mask": torch.tensor([[1, 0, 1, 1]] * len(texts)),
            }

    def build():
        torch.manual_seed(71)
        return TorchAnalysisRuntime(
            SingleSiteModel(temporal=True).to(torch.bfloat16), HoleTokenizer(), ["abcd"],
            groups={"target": "target"},
        )

    reference = _reference_single_site(build(), PrecisionFormat.W4A8)
    wrongly_masked = _reference_single_site(
        build(), PrecisionFormat.W4A8,
        source_mask=torch.tensor([[False, False, True, False]]),
    )
    assert abs(reference - wrongly_masked) > 1e-6
    result = build().run_fast_taylor((PrecisionFormat.W4A8,))
    assert result.token_count == 1
    measurement = next(item for item in result.measurements if item.precision is PrecisionFormat.W4A8)
    assert measurement.mean_first_order_effect == pytest.approx(reference, rel=2e-5, abs=2e-8)


def test_taylor_sums_signed_contributions_before_group_ranking():
    class CancelBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Linear(6, 6, bias=False)
            self.right = torch.nn.Linear(6, 6, bias=False)
            self.right.load_state_dict(self.left.state_dict())

        def forward(self, x):
            return x + (self.left(x) - self.right(x))

    class CancelModel(Model):
        def __init__(self):
            super().__init__()
            self.block = CancelBlock()

        def forward(self, *, input_ids, attention_mask):
            del attention_mask
            hidden = self.embed(input_ids)
            return SimpleNamespace(logits=self.lm_head(self.block(hidden)))

    def build(groups):
        torch.manual_seed(73)
        return make_runtime(model=CancelModel().to(torch.bfloat16), groups=groups)

    split = build({"left": "block.left", "right": "block.right"}).run_fast_taylor(
        (PrecisionFormat.W4A8,)
    )
    effects = [item.mean_first_order_effect for item in split.measurements
               if item.precision is PrecisionFormat.W4A8]
    assert len(effects) == 2
    assert effects[0] * effects[1] < 0
    assert sum(effects) == pytest.approx(0, abs=1e-8)
    grouped = build({"pair": "block"}).run_fast_taylor((PrecisionFormat.W4A8,))
    effect = next(item for item in grouped.measurements if item.precision is PrecisionFormat.W4A8)
    assert effect.mean_first_order_effect == pytest.approx(0, abs=1e-8)
    assert effect.estimated_risk == pytest.approx(0, abs=1e-8)


def test_taylor_zero_weight_is_a_true_noop_despite_nonzero_activation_error():
    runtime = single_runtime()
    with torch.no_grad():
        runtime.model.target.weight.zero_()
    result = runtime.run_fast_taylor(PRECISIONS)
    assert all(item.mean_first_order_effect == 0 for item in result.measurements)
    assert all(item.estimated_risk == 0 for item in result.measurements)


def test_taylor_duplicate_samples_and_batch_partition_keep_the_signed_scale():
    samples = ["abcde", "abcdefg"]
    runs = [
        make_runtime(samples=samples, batch_size=1),
        make_runtime(samples=samples, batch_size=2),
        make_runtime(samples=samples, batch_size=2, padding=7),
        make_runtime(samples=samples * 2, batch_size=2),
    ]
    results = [runtime.run_fast_taylor(PRECISIONS) for runtime in runs]
    assert [result.token_count for result in results] == [10, 10, 10, 20]
    for result in results[1:]:
        for first, second in zip(results[0].measurements, result.measurements, strict=True):
            assert second.mean_first_order_effect == pytest.approx(
                first.mean_first_order_effect, rel=0.03, abs=2e-6
            )


@pytest.mark.parametrize("failure", ["forward", "input_mutation"])
def test_taylor_failure_cleans_hooks_and_restores_parameter_flags(failure):
    runtime = make_runtime() if failure == "forward" else single_runtime()
    runtime.model.train()
    runtime.model.layers[0].norm.eval()
    mutation_handle = None
    if failure == "forward":
        runtime.model.fail = True
    else:
        # Mutate the cloned branch actually consumed by Linear, not its safely
        # isolated original input. This external hook runs before our post-hook.
        def mutate_branch(_module, args, _output):
            args[0].add_(0.25)

        mutation_handle = runtime.model.target.register_forward_hook(mutate_branch)
    before = {name: value.clone() for name, value in runtime.model.state_dict().items()}
    flags = [parameter.requires_grad for parameter in runtime.model.parameters()]
    training = [module.training for module in runtime.model.modules()]
    error = RuntimeError if failure == "forward" else TorchAnalysisError
    pattern = "intentional model failure" if failure == "forward" else "in.?place|modif|version"
    try:
        with pytest.raises(error, match=pattern):
            runtime.run_fast_taylor(PRECISIONS)
    finally:
        if mutation_handle is not None:
            mutation_handle.remove()
    assert [p.requires_grad for p in runtime.model.parameters()] == flags
    assert [m.training for m in runtime.model.modules()] == training
    assert all(torch.equal(before[name], value) for name, value in runtime.model.state_dict().items())
    assert all(p.grad is None for p in runtime.model.parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in runtime.model.modules())


def test_taylor_cached_branch_survives_original_shared_input_mutation():
    reference = single_runtime().run_fast_taylor(PRECISIONS)
    mutated = single_runtime(mutate_input=True).run_fast_taylor(PRECISIONS)
    for first, second in zip(reference.measurements, mutated.measurements, strict=True):
        assert first.mean_first_order_effect == pytest.approx(
            second.mean_first_order_effect, rel=1e-6, abs=1e-9
        )


def test_directional_helpers_match_fp64_fixed_output_injection_derivative():
    from quantplan.runtime.directional_analysis import (
        _activation_effect,
        _project_weight_effects,
    )

    torch.manual_seed(79)
    value = torch.randn(1, 4, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(7, 5, dtype=torch.float64)
    bias = torch.randn(7, dtype=torch.float64)
    readout = torch.randn(7, 3, dtype=torch.float64)
    baseline = torch.nn.functional.linear(value, weight, bias)

    def objective(output):
        return torch.nn.functional.softplus(torch.tanh(output) @ readout).sum()

    loss = objective(baseline)
    output_gradient, input_gradient = torch.autograd.grad(loss, (baseline, value))
    scores, quantized, finite, _gemms = _project_weight_effects(
        torch, value.detach(), output_gradient, weight, (4, 8),
        max_chunk_elements=16, token_chunk_size=2,
    )
    assert bool(finite)
    activation = _activation_effect(torch, value.detach(), quantized, input_gradient)
    for bits in (4, 8):
        delta_weight = ta._fake_quant_weight(torch, weight, bits) - weight
        delta_input = quantized - value.detach()
        # This fixed output intervention includes the local delta-X delta-W term.
        # Scaling weights and inputs separately would define another epsilon path.
        direction = (
            torch.nn.functional.linear(quantized, delta_weight)
            + torch.nn.functional.linear(delta_input, weight)
        )
        epsilon = 1e-4
        finite_difference = (
            objective(baseline.detach() + epsilon * direction)
            - objective(baseline.detach() - epsilon * direction)
        ) / (2 * epsilon)
        contracted = (output_gradient * direction).sum()
        assert float(contracted) == pytest.approx(float(finite_difference), rel=1e-7, abs=1e-9)
        assert float(scores[bits] + activation) == pytest.approx(
            float(finite_difference), rel=2e-6, abs=2e-8
        )


def test_directional_projection_reuses_gemms_across_weight_bits_and_is_chunk_invariant():
    from quantplan.runtime.directional_analysis import _project_weight_effects

    torch.manual_seed(83)
    value = torch.randn(2, 5, 7, dtype=torch.bfloat16)
    gradient = torch.randn(2, 5, 11, dtype=torch.bfloat16)
    weight = torch.randn(11, 7, dtype=torch.bfloat16)
    first, qfirst, finite_first, first_gemms = _project_weight_effects(
        torch, value, gradient, weight, (4,),
    )
    both, qboth, finite_both, both_gemms = _project_weight_effects(
        torch, value, gradient, weight, (4, 8),
    )
    chunked, qchunked, finite_chunked, chunked_gemms = _project_weight_effects(
        torch, value, gradient, weight, (4, 8), max_chunk_elements=28, token_chunk_size=2,
    )
    assert bool(finite_first) and bool(finite_both) and bool(finite_chunked)
    assert torch.equal(qfirst, qboth)
    assert torch.equal(qfirst, qchunked)
    assert first_gemms == both_gemms > 0
    assert chunked_gemms > both_gemms
    assert float(first[4]) == pytest.approx(float(both[4]), rel=1e-6, abs=1e-7)
    for bits in (4, 8):
        assert float(chunked[bits]) == pytest.approx(float(both[bits]), rel=2e-5, abs=2e-6)


@pytest.mark.parametrize("failure", ["nan_weight", "nan_input", "unused", "checkpoint"])
def test_taylor_rejects_invalid_or_missing_evidence_and_cleans_up(failure):
    torch.manual_seed(41)
    model = Model().to(torch.bfloat16)
    if failure == "unused":
        model.layers[0].unused = torch.nn.Linear(6, 6, bias=False).to(torch.bfloat16)
    elif failure == "checkpoint":
        model.gradient_checkpointing = True
    else:
        with torch.no_grad():
            target = model.embed.weight if failure == "nan_input" else model.layers[0].up_proj.weight
            target.fill_(float("nan"))
    runtime = make_runtime(model=model)
    flags = [parameter.requires_grad for parameter in model.parameters()]
    with pytest.raises(TorchAnalysisError, match="non-finite|invalid|coverage|checkpoint"):
        runtime.run_fast_taylor(PRECISIONS)
    assert [p.requires_grad for p in model.parameters()] == flags
    assert all(p.grad is None for p in model.parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
