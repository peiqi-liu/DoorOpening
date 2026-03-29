from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def _sample_choices(
    choices: Sequence[float],
    shape: tuple[int, int],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    choice_tensor = torch.as_tensor(tuple(float(value) for value in choices), device=device, dtype=dtype)
    if choice_tensor.numel() == 0:
        raise ValueError("Randomization choices must contain at least one value.")
    choice_ids = torch.randint(choice_tensor.numel(), shape, device=device)
    return choice_tensor[choice_ids]


def _resolve_joint_ids(asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        start, stop, step = joint_ids.indices(asset.data.joint_stiffness.shape[1])
        return torch.arange(start, stop, step, device=asset.device, dtype=torch.long)
    if isinstance(joint_ids, torch.Tensor):
        return joint_ids.to(device=asset.device, dtype=torch.long)
    return torch.tensor(joint_ids, device=asset.device, dtype=torch.long)


def randomize_actuator_gains_from_choices(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_choices: Sequence[float],
    damping_choices: Sequence[float] | None = None,
):
    """Reset-time discrete sampling for actuator gains.

    This keeps the sampled "normal" door gain separate from the latch override that happens every step.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=asset.device, dtype=torch.long)

    target_joint_ids = _resolve_joint_ids(asset, asset_cfg)
    if target_joint_ids.numel() == 0:
        return

    for actuator in asset.actuators.values():
        if isinstance(actuator.joint_indices, slice):
            actuator_indices = global_indices = target_joint_ids
        else:
            actuator_joint_indices = actuator.joint_indices.to(asset.device)
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, target_joint_ids)).flatten()
            if actuator_indices.numel() == 0:
                continue
            global_indices = actuator_joint_indices[actuator_indices]

        stiffness = actuator.stiffness[env_ids].clone()
        num_selected = stiffness.shape[1] if isinstance(actuator_indices, slice) else actuator_indices.numel()
        stiffness[:, actuator_indices] = _sample_choices(
            stiffness_choices,
            (len(env_ids), num_selected),
            device=asset.device,
            dtype=stiffness.dtype,
        )
        actuator.stiffness[env_ids] = stiffness
        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)

        damping = actuator.damping[env_ids].clone()
        if damping_choices is None:
            damping[:, actuator_indices] = asset.data.default_joint_damping[env_ids][:, global_indices].clone()
        else:
            damping[:, actuator_indices] = _sample_choices(
                damping_choices,
                (len(env_ids), num_selected),
                device=asset.device,
                dtype=damping.dtype,
            )
        actuator.damping[env_ids] = damping
        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)
