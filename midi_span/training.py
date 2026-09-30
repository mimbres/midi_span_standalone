# SPDX-FileCopyrightText: 2026 AnySynth contributors
# SPDX-License-Identifier: Apache-2.0

"""Model-independent losses for continuous MIDI endpoint generation.

Import this module explicitly to use the optional PyTorch training helpers.
The fixed NumPy codec does not import PyTorch. These functions train your
velocity model, not an additional MIDI encoder or decoder.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .flowamt import (
    ENDPOINT_COUNT, FEATURES_PER_ENDPOINT, FRAME_WIDTH, LANES_PER_BANK,
)


_PREFIX_GROUPS = ("on_empty", "on_nonempty", "off_empty", "off_nonempty")
_PREFIX_CLASSES = LANES_PER_BANK + 1


@dataclass(frozen=True, slots=True)
class FieldLoss:
    """Scalar total and five semantic MSEs, suitable for training logs."""

    total: Tensor
    presence: Tensor
    program: Tensor
    pitch: Tensor
    velocity: Tensor
    subsample: Tensor


@dataclass(frozen=True, slots=True)
class PrefixLoss:
    """Normalized CE and its ON/OFF empty/nonempty group diagnostics.

    Missing groups have a zero mean and count, and do not affect ``total``.
    Counts measure frame-bank examples, rather than individual lanes.
    """

    total: Tensor
    group_means: dict[str, Tensor]
    group_counts: dict[str, Tensor]


def _lanes(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point Tensor")
    flat = value.ndim == 3 and value.shape[-1] == FRAME_WIDTH
    expanded = value.ndim == 4 and value.shape[-2:] == (
        ENDPOINT_COUNT, FEATURES_PER_ENDPOINT,
    )
    if not (flat or expanded):
        raise ValueError(f"{name} must have shape [B,T,288] or [B,T,32,9]")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} batch and time dimensions must be nonzero")
    return value.reshape(*value.shape[:2], ENDPOINT_COUNT, FEATURES_PER_ENDPOINT)


def _valid_frames(value: Tensor, valid_mask: Tensor | None) -> Tensor:
    if valid_mask is None:
        return torch.ones(value.shape[:2], device=value.device, dtype=torch.bool)
    if not isinstance(valid_mask, Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be a bool Tensor or None")
    if valid_mask.shape != value.shape[:2] or valid_mask.device != value.device:
        raise ValueError("valid_mask must have shape [B,T] on the tensor device")
    if not bool(valid_mask.any(dim=1).all()):
        raise ValueError("every sample must contain at least one valid frame")
    return valid_mask


def _matching_inputs(clean: Tensor, **values: Tensor) -> None:
    _lanes("clean", clean)
    for name, value in values.items():
        _lanes(name, value)
        if value.shape != clean.shape or value.device != clean.device:
            raise ValueError(f"{name} must match clean's shape and device")


def _coordinate_mean(values: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
    """Mean over selected rows and their coordinates, plus availability."""

    count = rows.sum(dtype=torch.float32)
    numerator = values.masked_fill(~rows[..., None], 0.0).sum()
    mean = numerator / (count * values.shape[-1]).clamp_min(1.0)
    return mean, (count > 0).to(torch.float32)


def _available_mean(terms: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    means, availability = (torch.stack(items) for items in zip(*terms))
    count = availability.sum()
    return (means * availability).sum() / count.clamp_min(1.0), (count > 0).float()


def field_mse(
    prediction: Tensor,
    target: Tensor,
    clean: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> FieldLoss:
    """Compute the original field-balanced endpoint or CFM velocity loss.

    All three tensors share ``[B,T,288]`` or ``[B,T,32,9]``. ``clean`` is the
    fixed codec target: it determines occupied lanes and program groups even
    when ``prediction`` and ``target`` are velocities. For endpoint exposure,
    pass the generated endpoint as prediction and clean as both other inputs.

    Presence balances ON occupied/empty and OFF occupied/empty. Payload MSEs
    score occupied lanes only. Program balances the available P/G/B groups,
    including the single constant piano group in piano-only data. The total
    averages available presence/program/pitch/velocity/subsample fields.
    Padding frames never contribute. Reduction is in FP32, also under AMP.
    """

    _matching_inputs(clean, prediction=prediction, target=target)
    frames = _valid_frames(clean, valid_mask)
    with torch.autocast(device_type=clean.device.type, enabled=False):
        grouped_clean = _lanes("clean", clean).float().reshape(
            *clean.shape[:2], 2, LANES_PER_BANK, FEATURES_PER_ENDPOINT,
        )
        error = (_lanes("prediction", prediction).float()
                 - _lanes("target", target).float()).square().reshape_as(grouped_clean)
        occupied = (grouped_clean[..., 0] > 0.0) & frames[..., None, None]
        empty = (grouped_clean[..., 0] <= 0.0) & frames[..., None, None]

        presence, presence_available = _available_mean([
            _coordinate_mean(error[:, :, bank, :, 0:1], rows[:, :, bank])
            for bank in range(2) for rows in (occupied, empty)
        ])
        codes = grouped_clean.new_tensor([
            [1.0, 0.0], [-0.5, math.sqrt(3.0) / 2.0],
            [-0.5, -math.sqrt(3.0) / 2.0],
        ])
        program_group = grouped_clean[..., 1:3].unsqueeze(-2)
        program_group = (program_group - codes).square().sum(-1).argmin(-1)
        program, program_available = _available_mean([
            _coordinate_mean(error[..., 1:3], occupied & (program_group == group))
            for group in range(3)
        ])
        pitch, payload_available = _coordinate_mean(error[..., 3:6], occupied)
        velocity, _ = _coordinate_mean(error[..., 6:8], occupied)
        subsample, _ = _coordinate_mean(error[..., 8:9], occupied)
        total, _ = _available_mean([
            (presence, presence_available), (program, program_available),
            (pitch, payload_available), (velocity, payload_available),
            (subsample, payload_available),
        ])
    return FieldLoss(total, presence, program, pitch, velocity, subsample)


def prefix_logits(endpoint: Tensor) -> Tensor:
    """Return decoder-aligned count logits ``[B,T,2,17]``.

    Class k scores the first k lanes of an ON/OFF bank. Forward presence is
    hard-clamped to [-1,1], and scores use the decoder's FP64 accumulation,
    scaled by four. The rational-tail surrogate changes only backward:
    gradient 1 inside the range, and 1/(1+d)^2 outside at distance d.
    ``argmax(-1)`` gives exactly the decoder's prefix counts, including ties.
    """

    presence = _lanes("endpoint", endpoint)[..., 0]
    if not bool(torch.isfinite(presence).all()):
        raise ValueError("endpoint presence must be finite")
    with torch.autocast(device_type=endpoint.device.type, enabled=False):
        raw = presence.float()
        hard = raw.clamp(-1.0, 1.0)
        distance = (raw.abs() - 1.0).clamp_min(0.0)
        tail = raw.sign() * (2.0 - (1.0 + distance).reciprocal())
        surrogate = torch.where(raw.abs() <= 1.0, raw, tail)
        projected = hard.detach() + (surrogate - surrogate.detach())
        grouped = projected.to(torch.float64).reshape(
            *endpoint.shape[:2], 2, LANES_PER_BANK,
        )
        zeros = grouped.new_zeros((*grouped.shape[:-1], 1))
        return torch.cat((zeros, grouped.cumsum(dim=-1)), dim=-1) * 4.0


def prefix_cross_entropy(
    endpoint: Tensor,
    clean: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> PrefixLoss:
    """Teach generated endpoints the clean ON/OFF prefix lengths.

    Reuse the same endpoint rollout as ODE exposure, without another model
    call. CE is divided by log(17), then macro-averaged over available
    ON-empty/nonempty and OFF-empty/nonempty groups. It remains positive even
    for a perfect clean endpoint. Padding frames do not contribute.
    """

    _matching_inputs(clean, endpoint=endpoint)
    frames = _valid_frames(clean, valid_mask)
    counts = (_lanes("clean", clean)[..., 0] > 0.0).reshape(
        *clean.shape[:2], 2, LANES_PER_BANK,
    ).sum(-1, dtype=torch.int64)
    with torch.autocast(device_type=clean.device.type, enabled=False):
        logits = prefix_logits(endpoint)
        ce = F.cross_entropy(
            logits.reshape(-1, _PREFIX_CLASSES), counts.reshape(-1), reduction="none",
        ).reshape_as(counts).div(math.log(_PREFIX_CLASSES)).float()
        means, group_counts = [], []
        for bank in range(2):
            for nonempty in (False, True):
                rows = frames & ((counts[..., bank] > 0) == nonempty)
                count = rows.sum(dtype=torch.int64)
                mean = ce[..., bank].masked_fill(~rows, 0.0).sum() / count.float().clamp_min(1.0)
                means.append(mean)
                group_counts.append(count)
        total, _ = _available_mean([
            (mean, (count > 0).float()) for mean, count in zip(means, group_counts)
        ])
    return PrefixLoss(
        total, dict(zip(_PREFIX_GROUPS, means)), dict(zip(_PREFIX_GROUPS, group_counts)),
    )


def differentiable_euler(
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    initial_state: Tensor,
    *,
    steps: int = 4,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Integrate model time 0→1 with gradients through every Euler step.

    ``velocity_fn(state, timestep)`` returns a velocity of the same shape and
    device. Its closure supplies your audio/MIDI condition and model padding
    mask. ``timestep`` is FP32 ``[B]``. State/output remain FP32 and preserve
    the caller's flat or lane layout. Padded frames are zero at every step.

    For exposure use a subset of the CFM noise with ``steps=4``. This adds
    four model calls and retains their computation graphs. For inference,
    choose your sampling step count and call under ``torch.no_grad()``.
    """

    _lanes("initial_state", initial_state)
    if not callable(velocity_fn):
        raise TypeError("velocity_fn must be callable")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    frames = _valid_frames(initial_state, valid_mask)
    elements = frames.reshape(*frames.shape, *([1] * (initial_state.ndim - 2)))
    with torch.autocast(device_type=initial_state.device.type, enabled=False):
        state = initial_state.float().masked_fill(~elements, 0.0)
    for step in range(steps):
        time = torch.full(
            (state.shape[0],), step / steps, device=state.device, dtype=torch.float32,
        )
        velocity = velocity_fn(state, time)
        _matching_inputs(state, velocity=velocity)
        with torch.autocast(device_type=state.device.type, enabled=False):
            state = (state + velocity.float() / steps).masked_fill(~elements, 0.0)
    return state


__all__ = [
    "FieldLoss", "PrefixLoss", "field_mse", "prefix_logits",
    "prefix_cross_entropy", "differentiable_euler",
]
