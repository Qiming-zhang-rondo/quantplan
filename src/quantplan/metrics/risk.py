"""Prompt-level evidence for quantization-induced residual distortion.

The score implemented here is a *diagonal gradient-weighted relative residual
distortion*.  It is useful as one piece of evidence for ranking quantization
candidates.  It is **not** an estimate or prediction of ``delta NLL``: the
diagonal form drops cross-coordinate terms and a finite quantization
perturbation need not lie inside the local linear region of the BF16 anchor.

For a reference residual stream ``h``, its fake-quantized counterpart ``h_q``,
gradient ``a = dL/dh`` and ``delta = h_q - h``, each prompt receives:

``T1``
    Signed first-order Taylor term, ``sum(a * delta)``.
``N``
    Diagonal gradient-weighted perturbation energy,
    ``sum(a**2 * delta**2)``.
``D``
    Diagonal gradient-weighted reference energy,
    ``sum(a**2 * h**2)``.
``S``
    Relative diagonal distortion, ``N / (D + epsilon)``.
``kappa``
    Cross-term diagnostic, ``T1**2 / (N + epsilon)``.  Values far from one
    show that the diagonal energy differs substantially from the squared
    signed first-order term.
``residual_relative_mse``
    Unweighted residual-stream relative MSE,
    ``sum(delta**2) / (sum(h**2) + epsilon)``.

Shape convention
----------------
``[batch, ..., hidden]`` arrays with three or more dimensions are interpreted
as a batch of prompts.  One- or two-dimensional arrays are interpreted as one
prompt.  A token mask such as ``[batch, sequence]`` or ``[sequence]`` is
broadcast over the hidden dimension.  A full element mask is also accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_EPSILON: Final[float] = 1e-12


@dataclass(frozen=True)
class Quantiles:
    """P50/P95/P99 values for one prompt-level metric."""

    p50: float
    p95: float
    p99: float


@dataclass(frozen=True)
class PromptRiskAggregate:
    """Quantile summary across prompts.

    Prompts with status ``no_valid_elements`` are excluded.  Low-denominator
    prompts remain in the aggregates so that their potentially unstable score
    is not silently hidden; ``low_denominator_count`` makes them explicit.
    """

    t1: Quantiles
    n: Quantiles
    d: Quantiles
    s: Quantiles
    kappa: Quantiles
    residual_relative_mse: Quantiles
    prompt_count: int
    valid_prompt_count: int
    low_denominator_count: int


@dataclass(frozen=True)
class PromptRiskResult:
    """Per-prompt risk evidence and diagnostic status.

    Every numeric field has shape ``[prompt_count]``.  ``status`` is one of
    ``"ok"``, ``"low_denominator"`` or ``"no_valid_elements"``.
    """

    t1: NDArray[np.float64]
    n: NDArray[np.float64]
    d: NDArray[np.float64]
    s: NDArray[np.float64]
    kappa: NDArray[np.float64]
    residual_relative_mse: NDArray[np.float64]
    valid_element_count: NDArray[np.int64]
    low_denominator: NDArray[np.bool_]
    status: NDArray[np.str_]
    epsilon: float
    denominator_threshold: float

    @property
    def prompt_count(self) -> int:
        """Number of prompts represented by this result."""

        return int(self.t1.size)

    @property
    def has_low_denominator(self) -> bool:
        """Whether any non-empty prompt has an unstable reference energy."""

        return bool(np.any(self.low_denominator))

    def aggregate(self) -> PromptRiskAggregate:
        """Return P50/P95/P99 aggregates across valid prompts."""

        return aggregate_prompt_risk(self)


def compute_prompt_risk(
    h_ref: ArrayLike,
    h_q: ArrayLike,
    gradient: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    epsilon: float = DEFAULT_EPSILON,
    denominator_threshold: float | None = None,
) -> PromptRiskResult:
    """Compute prompt-level diagonal gradient-weighted residual distortion.

    Parameters
    ----------
    h_ref, h_q, gradient:
        Real, finite NumPy-compatible arrays with identical, non-empty shape.
        Arrays with ``ndim >= 3`` use axis zero as the prompt/batch axis;
        otherwise the complete array is treated as one prompt.
    mask:
        Optional boolean or binary numeric mask.  Typical token masks omit the
        final hidden dimension and are broadcast across it.  Masked values do
        not contribute to any metric.
    epsilon:
        Strictly positive numerical stabilizer used in ``S``, ``kappa`` and
        relative MSE.
    denominator_threshold:
        Non-negative threshold for flagging a low ``D``.  Defaults to
        ``epsilon``.  The flag is diagnostic: the score is still returned and
        low-denominator prompts are not silently removed from aggregation.

    Returns
    -------
    PromptRiskResult
        Per-prompt evidence.  These values rank or diagnose candidates; they
        must not be reported as predicted changes in NLL.
    """

    ref = _as_valid_float_array("h_ref", h_ref)
    quantized = _as_valid_float_array("h_q", h_q)
    grad = _as_valid_float_array("gradient", gradient)

    if quantized.shape != ref.shape or grad.shape != ref.shape:
        raise ValueError(
            "h_ref, h_q and gradient must have identical shapes; got "
            f"{ref.shape}, {quantized.shape} and {grad.shape}"
        )

    eps = _validate_positive_finite("epsilon", epsilon)
    if denominator_threshold is None:
        threshold = eps
    else:
        threshold = _validate_nonnegative_finite(
            "denominator_threshold", denominator_threshold
        )

    valid = _broadcast_mask(mask, ref.shape)

    # A rank-1/rank-2 input is one prompt.  Higher-rank tensors use the first
    # axis as their prompt axis.
    if ref.ndim <= 2:
        ref = ref[np.newaxis, ...]
        quantized = quantized[np.newaxis, ...]
        grad = grad[np.newaxis, ...]
        valid = valid[np.newaxis, ...]

    reduction_axes = tuple(range(1, ref.ndim))
    delta = quantized - ref

    def masked_sum(value: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sum(np.where(valid, value, 0.0), axis=reduction_axes)

    t1 = masked_sum(grad * delta)
    n = masked_sum(np.square(grad) * np.square(delta))
    d = masked_sum(np.square(grad) * np.square(ref))
    delta_energy = masked_sum(np.square(delta))
    reference_energy = masked_sum(np.square(ref))
    valid_count = np.sum(valid, axis=reduction_axes, dtype=np.int64)

    has_values = valid_count > 0
    s = n / (d + eps)
    kappa = np.square(t1) / (n + eps)
    residual_relative_mse = delta_energy / (reference_energy + eps)

    # An all-masked prompt has no defined evidence.  Returning NaN together
    # with an explicit status keeps batched processing possible without making
    # an empty prompt look safe.
    for metric in (t1, n, d, s, kappa, residual_relative_mse):
        metric[~has_values] = np.nan

    low_denominator = has_values & (d <= threshold)
    status = np.full(t1.shape, "ok", dtype="<U17")
    status[low_denominator] = "low_denominator"
    status[~has_values] = "no_valid_elements"

    return PromptRiskResult(
        t1=t1,
        n=n,
        d=d,
        s=s,
        kappa=kappa,
        residual_relative_mse=residual_relative_mse,
        valid_element_count=valid_count,
        low_denominator=low_denominator,
        status=status,
        epsilon=eps,
        denominator_threshold=threshold,
    )


def aggregate_prompt_risk(result: PromptRiskResult) -> PromptRiskAggregate:
    """Aggregate prompt-level evidence into P50/P95/P99 quantiles.

    This aggregation is descriptive, not a confidence interval and not a
    prediction of ``delta NLL``.  Callers that need uncertainty estimates
    should bootstrap whole prompts/documents rather than individual tokens.
    """

    if not isinstance(result, PromptRiskResult):
        raise TypeError("result must be a PromptRiskResult")

    valid_prompts = result.status != "no_valid_elements"

    def summarize(values: NDArray[np.float64]) -> Quantiles:
        selected = values[valid_prompts]
        if selected.size == 0:
            return Quantiles(np.nan, np.nan, np.nan)
        p50, p95, p99 = np.percentile(selected, (50.0, 95.0, 99.0))
        return Quantiles(float(p50), float(p95), float(p99))

    return PromptRiskAggregate(
        t1=summarize(result.t1),
        n=summarize(result.n),
        d=summarize(result.d),
        s=summarize(result.s),
        kappa=summarize(result.kappa),
        residual_relative_mse=summarize(result.residual_relative_mse),
        prompt_count=result.prompt_count,
        valid_prompt_count=int(np.count_nonzero(valid_prompts)),
        low_denominator_count=int(np.count_nonzero(result.low_denominator)),
    )


def _as_valid_float_array(name: str, value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.ndim == 0 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty array with at least one dimension")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must contain real numeric values")

    converted = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _broadcast_mask(mask: ArrayLike | None, shape: tuple[int, ...]) -> NDArray[np.bool_]:
    if mask is None:
        return np.ones(shape, dtype=np.bool_)

    raw = np.asarray(mask)
    if raw.size == 0:
        raise ValueError("mask must not be empty")
    if np.issubdtype(raw.dtype, np.bool_):
        binary = raw
    elif np.issubdtype(raw.dtype, np.number) and not np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        numeric = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or not np.all(
            (numeric == 0.0) | (numeric == 1.0)
        ):
            raise ValueError("numeric mask must contain only finite 0/1 values")
        binary = numeric.astype(np.bool_)
    else:
        raise TypeError("mask must be boolean or binary numeric")

    # A mask that omits the final hidden dimension is the normal padding-mask
    # form and must be tried before generic NumPy broadcasting.  Otherwise a
    # [sequence] mask can accidentally align with the hidden axis when the two
    # dimensions happen to have equal length.
    candidates: list[NDArray[np.bool_]] = []
    if binary.shape == shape:
        candidates.append(binary)
    if binary.ndim < len(shape):
        candidates.append(binary[..., np.newaxis])
    if binary.ndim == 1 and len(shape) >= 3 and binary.shape[0] == shape[0]:
        candidates.append(binary.reshape((shape[0],) + (1,) * (len(shape) - 1)))
    candidates.append(binary)

    for candidate in candidates:
        try:
            return np.asarray(np.broadcast_to(candidate, shape), dtype=np.bool_)
        except ValueError:
            continue

    raise ValueError(f"mask shape {raw.shape} is not broadcastable to input shape {shape}")


def _validate_positive_finite(name: str, value: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return converted


def _validate_nonnegative_finite(name: str, value: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not np.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted
