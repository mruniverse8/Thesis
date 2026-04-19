from __future__ import annotations

from typing import Iterable, Sequence

import torch

from .trajectory import ScalarLike, ScoredStageTrajectory


def _tensor_spec(*values: ScalarLike) -> tuple[torch.device, torch.dtype]:
    for value in values:
        if torch.is_tensor(value):
            return value.device, value.dtype
    return torch.device("cpu"), torch.float32


def _to_tensor(value: ScalarLike, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(float(value), device=device, dtype=dtype)


def _sum_tensors(values: Sequence[ScalarLike], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not values:
        return torch.zeros((), device=device, dtype=dtype)
    return torch.stack([_to_tensor(value, device=device, dtype=dtype) for value in values]).sum()


def trajectory_balance_residual(trajectory: ScoredStageTrajectory) -> torch.Tensor:
    device, dtype = _tensor_spec(
        *trajectory.log_state_flows,
        *trajectory.log_pf_tokens,
        *trajectory.log_stop,
        *trajectory.log_pb_tokens,
    )
    root_log_flow = _to_tensor(trajectory.log_state_flows[0], device=device, dtype=dtype)
    forward_total = _sum_tensors(
        [root_log_flow, *trajectory.log_pf_tokens, trajectory.log_stop[-1]],
        device=device,
        dtype=dtype,
    )
    backward_total = _sum_tensors(
        trajectory.effective_log_pb_tokens(),
        device=device,
        dtype=dtype,
    )
    log_reward = torch.log(torch.tensor(trajectory.terminal_reward, device=device, dtype=dtype))
    return forward_total - backward_total - log_reward


def trajectory_balance_loss(trajectories: Iterable[ScoredStageTrajectory]) -> torch.Tensor:
    residuals = [trajectory_balance_residual(trajectory) for trajectory in trajectories]
    if not residuals:
        return torch.zeros(())
    return torch.stack(residuals).pow(2).mean()


def detailed_balance_residuals(trajectory: ScoredStageTrajectory) -> torch.Tensor:
    device, dtype = _tensor_spec(
        *trajectory.log_state_flows,
        *trajectory.log_pf_tokens,
        *trajectory.log_stop,
        *trajectory.log_pb_tokens,
    )
    residuals: list[torch.Tensor] = []
    backward_terms = trajectory.effective_log_pb_tokens()
    for index, log_pf in enumerate(trajectory.log_pf_tokens):
        lhs = _to_tensor(trajectory.log_state_flows[index], device=device, dtype=dtype) + _to_tensor(
            log_pf,
            device=device,
            dtype=dtype,
        )
        rhs = _to_tensor(
            trajectory.log_state_flows[index + 1],
            device=device,
            dtype=dtype,
        ) + _to_tensor(backward_terms[index], device=device, dtype=dtype)
        residuals.append(lhs - rhs)
    terminal_term = _to_tensor(
        trajectory.log_state_flows[-1],
        device=device,
        dtype=dtype,
    ) + _to_tensor(trajectory.log_stop[-1], device=device, dtype=dtype)
    residuals.append(
        terminal_term - torch.log(torch.tensor(trajectory.terminal_reward, device=device, dtype=dtype))
    )
    return torch.stack(residuals)


def detailed_balance_loss(trajectory: ScoredStageTrajectory) -> torch.Tensor:
    return detailed_balance_residuals(trajectory).pow(2).mean()


def enumerate_subtrajectory_pairs(
    num_prefix_states: int,
    *,
    include_start: bool = True,
) -> list[tuple[int, int]]:
    if num_prefix_states <= 1:
        return []
    start_index = 0 if include_start else 1
    pairs: list[tuple[int, int]] = []
    for i in range(start_index, num_prefix_states - 1):
        for j in range(i + 1, num_prefix_states):
            pairs.append((i, j))
    return pairs


def subtrajectory_balance_residuals(
    trajectory: ScoredStageTrajectory,
    *,
    lambda_decay: float = 1.0,
    include_start: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < lambda_decay <= 1.0:
        raise ValueError("lambda_decay must be in (0, 1].")
    device, dtype = _tensor_spec(*trajectory.log_pf_tokens, *trajectory.log_stop)
    residuals: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for start_index, end_index in enumerate_subtrajectory_pairs(
        len(trajectory.prefix_states),
        include_start=include_start,
    ):
        forward_segment = _sum_tensors(
            trajectory.log_pf_tokens[start_index:end_index],
            device=device,
            dtype=dtype,
        )
        lhs = (
            torch.log(torch.tensor(trajectory.prefix_rewards[start_index], device=device, dtype=dtype))
            + forward_segment
            + _to_tensor(trajectory.log_stop[end_index], device=device, dtype=dtype)
        )
        rhs = (
            torch.log(torch.tensor(trajectory.prefix_rewards[end_index], device=device, dtype=dtype))
            + _to_tensor(trajectory.log_stop[start_index], device=device, dtype=dtype)
        )
        residuals.append(lhs - rhs)
        weights.append(
            torch.tensor(lambda_decay ** (end_index - start_index - 1), device=device, dtype=dtype)
        )
    if not residuals:
        return (
            torch.zeros((0,), device=device, dtype=dtype),
            torch.zeros((0,), device=device, dtype=dtype),
        )
    return torch.stack(residuals), torch.stack(weights)


def subtrajectory_balance_loss(
    trajectories: Iterable[ScoredStageTrajectory],
    *,
    lambda_decay: float = 1.0,
    include_start: bool = True,
) -> torch.Tensor:
    weighted_numerator: list[torch.Tensor] = []
    weighted_denominator: list[torch.Tensor] = []
    fallback_device = torch.device("cpu")
    fallback_dtype = torch.float32
    for trajectory in trajectories:
        residuals, weights = subtrajectory_balance_residuals(
            trajectory,
            lambda_decay=lambda_decay,
            include_start=include_start,
        )
        if residuals.numel() == 0:
            continue
        fallback_device = residuals.device
        fallback_dtype = residuals.dtype
        weighted_numerator.append((weights * residuals.pow(2)).sum())
        weighted_denominator.append(weights.sum())
    if not weighted_numerator:
        return torch.zeros((), device=fallback_device, dtype=fallback_dtype)
    return torch.stack(weighted_numerator).sum() / torch.stack(weighted_denominator).sum().clamp_min(
        1.0e-12
    )
