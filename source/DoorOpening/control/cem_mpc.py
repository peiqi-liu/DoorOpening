"""Receding-horizon CEM/MPC for physics-backed DoorOpening rollouts.

This module deliberately keeps Isaac Lab state ownership outside the optimizer.  An Isaac Lab
adapter supplies a batched ``rollout`` callback that snapshots/restores simulator state and
returns costs for candidate action sequences.  That makes the optimizer usable with a DirectRLEnv
without silently advancing the live environment during candidate evaluation.

The structure follows the sampling-based MPC ideas used by SPIDER/FS-MPC: optimize a short
receding horizon, retain an elite action distribution, bias candidates toward a reference motion,
and expose contact/penetration costs as separate diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import torch


@dataclass(frozen=True)
class CEMMPCConfig:
    horizon: int = 16
    population: int = 256
    elites: int = 32
    iterations: int = 4
    init_std: float = 0.35
    min_std: float = 0.03
    elite_momentum: float = 0.15
    discount: float = 0.99
    reference_weight: float = 0.20
    action_smooth_weight: float = 0.02
    contact_penalty_weight: float = 1.0
    panel_penetration_weight: float = 4.0

    def __post_init__(self) -> None:
        if not (1 <= self.elites <= self.population):
            raise ValueError("elites must be in [1, population]")
        if self.horizon < 1 or self.iterations < 1:
            raise ValueError("horizon and iterations must be positive")
        if not 0.0 <= self.elite_momentum < 1.0:
            raise ValueError("elite_momentum must be in [0, 1)")


@dataclass
class RolloutBatch:
    """Result returned by an Isaac Lab rollout adapter.

    ``cost`` must be one scalar per candidate.  The auxiliary terms are optional and are only
    used for diagnostics unless the adapter has not already folded them into ``cost``.
    """

    cost: torch.Tensor
    terms: Optional[Mapping[str, torch.Tensor]] = None


RolloutFn = Callable[[torch.Tensor], RolloutBatch]


class IsaacLabDoorRolloutAdapter:
    """State-safe adapter for an Isaac Lab ``DirectRLEnv``.

    ``snapshot_fn`` and ``restore_fn`` are intentionally injected because Isaac Lab versions and
    articulation layouts differ.  ``evaluate_fn`` receives ``(env, candidate_chunk)`` and should
    run each candidate for the configured horizon, returning a :class:`RolloutBatch`.  The adapter
    restores the live state both between chunks and in a ``finally`` block, so MPC is safe to use
    alongside the policy rollout.
    """

    def __init__(
        self,
        env,
        snapshot_fn: Callable[[object], object],
        restore_fn: Callable[[object, object], None],
        evaluate_fn: Callable[[object, torch.Tensor], RolloutBatch],
        *,
        chunk_size: int = 32,
    ) -> None:
        self.env = env
        self.snapshot_fn = snapshot_fn
        self.restore_fn = restore_fn
        self.evaluate_fn = evaluate_fn
        self.chunk_size = int(chunk_size)

    def __call__(self, candidates: torch.Tensor) -> RolloutBatch:
        snapshot = self.snapshot_fn(self.env)
        costs, term_chunks = [], []
        try:
            for chunk in candidates.split(self.chunk_size, dim=0):
                self.restore_fn(self.env, snapshot)
                result = self.evaluate_fn(self.env, chunk)
                costs.append(result.cost)
                if result.terms:
                    term_chunks.append(result.terms)
            terms = None
            if term_chunks:
                keys = term_chunks[0].keys()
                terms = {key: torch.cat([part[key] for part in term_chunks], dim=0) for key in keys}
            return RolloutBatch(torch.cat(costs, dim=0), terms)
        finally:
            self.restore_fn(self.env, snapshot)


class CEMMPCPlanner:
    """Torch CEM optimizer with receding-horizon warm starts.

    Args:
        action_dim: Number of policy actions (base, arm, and gripper intent included).
        action_low/high: Per-action bounds, normally ``[-1, 1]`` for DooropeningEnv actions.
        device: Torch device used for candidate sampling and distribution updates.
    """

    def __init__(
        self,
        action_dim: int,
        config: CEMMPCConfig | None = None,
        *,
        action_low: float | torch.Tensor = -1.0,
        action_high: float | torch.Tensor = 1.0,
        device: str | torch.device = "cuda",
    ) -> None:
        self.cfg = config or CEMMPCConfig()
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.low = torch.as_tensor(action_low, device=self.device, dtype=torch.float32).reshape(-1)
        self.high = torch.as_tensor(action_high, device=self.device, dtype=torch.float32).reshape(-1)
        if self.low.numel() == 1:
            self.low = self.low.expand(self.action_dim)
        if self.high.numel() == 1:
            self.high = self.high.expand(self.action_dim)
        if self.low.numel() != self.action_dim or self.high.numel() != self.action_dim:
            raise ValueError("action bounds must be scalar or action_dim-vectors")
        if torch.any(self.low >= self.high):
            raise ValueError("action_low must be strictly smaller than action_high")
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._mean = None
        self._std = None

    def _initial_distribution(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if reference.shape != (self.cfg.horizon, self.action_dim):
            raise ValueError(f"reference must have shape {(self.cfg.horizon, self.action_dim)}")
        if self._mean is None or self._mean.shape != reference.shape:
            mean = reference.clone()
            std = torch.full_like(mean, self.cfg.init_std)
        else:
            mean, std = self._mean, self._std
        return mean, std.clamp_min(self.cfg.min_std)

    def _shift_warm_start(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self._mean = torch.cat((mean[1:], mean[-1:].clone()), dim=0).detach()
        self._std = torch.cat((std[1:], std[-1:].clone()), dim=0).detach()

    def plan(self, reference: torch.Tensor, rollout: RolloutFn) -> tuple[torch.Tensor, dict]:
        """Optimize a candidate sequence and return the first action plus diagnostics.

        The callback is responsible for physics simulation and must not leave the live Isaac Lab
        environment advanced after evaluating candidates.  A typical adapter snapshots robot/door
        articulation state, evaluates candidates in chunks, and restores the snapshot afterward.
        """
        reference = reference.to(device=self.device, dtype=torch.float32)
        mean, std = self._initial_distribution(reference)
        best_cost = torch.tensor(float("inf"), device=self.device)
        best_sequence = mean.clone()
        last_terms: Mapping[str, torch.Tensor] | None = None

        for iteration in range(self.cfg.iterations):
            noise = torch.randn(
                self.cfg.population,
                self.cfg.horizon,
                self.action_dim,
                device=self.device,
            )
            candidates = (mean.unsqueeze(0) + noise * std.unsqueeze(0)).clamp(self.low, self.high)
            result = rollout(candidates)
            cost = result.cost.to(self.device).flatten()
            if cost.numel() != self.cfg.population:
                raise ValueError("rollout cost must have one value per candidate")
            elite_cost, elite_idx = torch.topk(cost, self.cfg.elites, largest=False)
            elite = candidates[elite_idx]
            elite_mean = elite.mean(dim=0)
            elite_std = elite.std(dim=0, unbiased=False).clamp_min(self.cfg.min_std)
            # Momentum prevents a single bad contact rollout from causing a violent distribution
            # jump, while still allowing CEM to move quickly around the reference trajectory.
            beta = self.cfg.elite_momentum
            mean = beta * mean + (1.0 - beta) * elite_mean
            std = beta * std + (1.0 - beta) * elite_std
            if elite_cost[0] < best_cost:
                best_cost = elite_cost[0]
                best_sequence = elite[0].clone()
                last_terms = result.terms

        self._shift_warm_start(mean, std)
        diagnostics = {
            "best_cost": float(best_cost.detach().cpu()),
            "mean_std": float(std.mean().detach().cpu()),
            "iterations": self.cfg.iterations,
            "terms": last_terms,
            "action_sequence": best_sequence.detach(),
            "distribution_mean": mean.detach(),
            "distribution_std": std.detach(),
        }
        return best_sequence[0].detach(), diagnostics
