"""Contract tests for no-replay, frozen-weight statistical sensitivity."""

from types import SimpleNamespace

import pytest

from quantplan.domain.models import PrecisionFormat
from quantplan.runtime import TorchAnalysisError, TorchAnalysisRuntime

torch = pytest.importorskip("torch")


class Tokenizer:
    pad_token_id = 0

    def __init__(self, padding=0):
        self.padding = padding

    def __call__(self, texts, *, max_length=None, **_kwargs):
        rows = [[1 + ord(c) % 15 for c in text][:max_length] for text in texts]
        ids = torch.zeros(len(rows), max(map(len, rows)) + self.padding, dtype=torch.long)
        for index, row in enumerate(rows):
            ids[index, :len(row)] = torch.tensor(row)
        return {"input_ids": ids, "attention_mask": ids.ne(0).long()}


class Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = torch.nn.RMSNorm(width)
        self.up_proj = torch.nn.Linear(width, width * 2, bias=False)
        self.down_proj = torch.nn.Linear(width * 2, width, bias=False)
        self.calls = 0

    def forward(self, hidden):
        self.calls += 1
        x = self.norm(hidden)
        return hidden + self.down_proj(torch.tanh(self.up_proj(x)))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="dense-test")
        self.embed = torch.nn.Embedding(16, 6)
        self.layers = torch.nn.ModuleList([Block(6), Block(6)])
        self.lm_head = torch.nn.Linear(6, 16, bias=False)
        self.calls = 0
        self.fail = False

    def get_input_embeddings(self):
        return self.embed

    def forward(self, *, input_ids, attention_mask):
        del attention_mask
        self.calls += 1
        hidden = self.embed(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        if self.fail:
            raise RuntimeError("intentional model failure")
        return SimpleNamespace(logits=self.lm_head(hidden))


def make_runtime(*, samples=None, batch_size=2, padding=0, model=None, **kwargs):
    if model is None:
        torch.manual_seed(41)
        model = Model().to(torch.bfloat16)
    return TorchAnalysisRuntime(
        model, Tokenizer(padding), samples or ["abcde", "abcdefg"],
        batch_size=batch_size, **kwargs,
    )


def test_stats_forward_count_does_not_grow_with_candidates_and_state_is_restored():
    for precisions in (
        (PrecisionFormat.W4A8,),
        (PrecisionFormat.W4A8, PrecisionFormat.W8A8, PrecisionFormat.BF16),
    ):
        runtime = make_runtime()
        model = runtime.model
        model.train()
        model.layers[0].norm.eval()
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        next(iter(model.parameters())).requires_grad_(False)
        flags = [parameter.requires_grad for parameter in model.parameters()]
        training = [module.training for module in model.modules()]
        result = runtime.run_fast_stats(precisions)
        assert model.calls == 1
        assert [block.calls for block in model.layers] == [1, 1]
        assert result.provenance.analysis_mode == "fast-stats"
        assert result.provenance.full_model_dataset_passes == 1
        assert result.provenance.backward_dataset_passes == 1
        assert result.provenance.local_block_replays == 0
        assert [p.requires_grad for p in model.parameters()] == flags
        assert [m.training for m in model.modules()] == training
        assert all(p.grad is None for p in model.parameters())
        assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        assert len(result.measurements) == len(runtime.groups) * len(precisions)
        for measurement in result.measurements:
            assert measurement.estimated_risk == pytest.approx(
                measurement.weight_risk + measurement.activation_risk
            )
            assert measurement.estimated_risk >= 0
            assert not hasattr(measurement, "delta_nll")
            if measurement.precision == PrecisionFormat.BF16:
                assert measurement.estimated_risk == 0


def test_stats_mask_duplicate_samples_and_batch_partition_have_stable_normalization():
    samples = ["abcde", "abcdefg"]
    runs = [
        make_runtime(samples=samples, batch_size=1),
        make_runtime(samples=samples, batch_size=2),
        make_runtime(samples=samples, batch_size=2, padding=7),
        make_runtime(samples=samples * 2, batch_size=2),
    ]
    results = [runtime.run_fast_stats((PrecisionFormat.W4A8,)) for runtime in runs]
    assert [result.token_count for result in results] == [10, 10, 10, 20]
    for result in results[1:]:
        for reference, candidate in zip(results[0].measurements, result.measurements, strict=True):
            assert candidate.weight_risk == pytest.approx(reference.weight_risk, rel=0.025)
            assert candidate.activation_risk == pytest.approx(reference.activation_risk, rel=0.025)


def test_stats_cleanup_after_forward_failure():
    runtime = make_runtime()
    runtime.model.fail = True
    before = {name: value.clone() for name, value in runtime.model.state_dict().items()}
    flags = [parameter.requires_grad for parameter in runtime.model.parameters()]
    with pytest.raises(RuntimeError, match="intentional model failure"):
        runtime.run_fast_stats((PrecisionFormat.W4A8,))
    assert [p.requires_grad for p in runtime.model.parameters()] == flags
    assert all(torch.equal(before[name], value)
               for name, value in runtime.model.state_dict().items())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in runtime.model.modules())


@pytest.mark.parametrize("source", ["weight", "activation"])
def test_stats_nonfinite_inputs_are_rejected_and_cleanup_restores_state(source, monkeypatch):
    runtime = make_runtime()
    model = runtime.model
    model.train()
    model.layers[0].norm.eval()
    next(iter(model.parameters())).requires_grad_(False)
    if source == "weight":
        with torch.no_grad():
            model.layers[0].up_proj.weight[0, 0] = float("nan")
    else:
        original = model.embed.forward

        def poisoned_embedding(*args, **kwargs):
            return original(*args, **kwargs) * float("nan")

        monkeypatch.setattr(model.embed, "forward", poisoned_embedding)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    flags = [parameter.requires_grad for parameter in model.parameters()]
    training = [module.training for module in model.modules()]
    with pytest.raises(TorchAnalysisError, match="non-finite"):
        runtime.run_fast_stats((PrecisionFormat.W4A8,))
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert [module.training for module in model.modules()] == training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0, equal_nan=True)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_stats_deferred_weight_and_padded_activation_checks_reject_nan():
    from quantplan.runtime import stats_analysis

    weight = torch.ones(3, 4, dtype=torch.bfloat16)
    weight[0, 0] = float("nan")
    with pytest.raises(TorchAnalysisError, match="weight score.*non-finite"):
        stats_analysis._weight_score(torch, weight, 4, torch.ones(4), torch.ones(3))

    inputs = torch.ones(1, 2, 4, dtype=torch.bfloat16)
    inputs[0, 1, 0] = float("nan")
    valid = torch.tensor([[True, False]])
    destination = torch.zeros(4)
    finite = torch.ones((), dtype=torch.bool)
    stats_analysis._activation_error_sum(torch, inputs, valid, destination, finite)
    assert not bool(finite)  # Invalid padded values must not silently disappear under the mask.
    assert bool(torch.isfinite(destination).all())


def test_stats_rejects_unobserved_target_instead_of_reporting_zero_risk():
    torch.manual_seed(41)
    model = Model()
    model.layers[0].unused = torch.nn.Linear(6, 6, bias=False)
    runtime = make_runtime(model=model.to(torch.bfloat16))
    with pytest.raises(TorchAnalysisError, match="observ|execut|cover|unused|capture"):
        runtime.run_fast_stats((PrecisionFormat.W4A8,))


def test_stats_rejects_fused_expert_parameter_targets():
    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(torch.randn(2, 12, 6))
            self.down_proj = torch.nn.Parameter(torch.randn(2, 6, 6))

        def forward(self, x):
            return x

    model = Model()
    model.config.model_type = "qwen3_5_moe"
    model.layers[0].experts = Experts()
    runtime = make_runtime(model=model.to(torch.bfloat16), groups={"block": "layers.0"})
    with pytest.raises(TorchAnalysisError, match="fused|expert|Linear"):
        runtime.run_fast_stats((PrecisionFormat.W4A8,))
    assert model.calls == 0


def test_stats_activation_gradient_is_local_to_target_branch():
    class BranchedModel(Model):
        def __init__(self, leak):
            super().__init__()
            self.target = torch.nn.Linear(6, 6, bias=False)
            self.leak = leak

        def forward(self, *, input_ids, attention_mask):
            del attention_mask
            x = self.embed(input_ids).detach().requires_grad_(True)
            # Value is zero, but its gradient contaminates hooks placed on the shared x.
            unrelated_branch = self.leak * (x - x.detach())
            hidden = self.target(x) + unrelated_branch
            return SimpleNamespace(logits=self.lm_head(hidden))

    results = []
    for leak in (0.0, 100.0):
        torch.manual_seed(41)
        runtime = make_runtime(
            model=BranchedModel(leak).to(torch.bfloat16), groups={"target": "target"}
        )
        results.append(runtime.run_fast_stats((PrecisionFormat.W4A8,)).measurements[0])
    assert results[0].activation_risk > 0
    assert results[1].activation_risk == pytest.approx(results[0].activation_risk, rel=1e-6)
    assert results[1].weight_risk == pytest.approx(results[0].weight_risk, rel=1e-6)
