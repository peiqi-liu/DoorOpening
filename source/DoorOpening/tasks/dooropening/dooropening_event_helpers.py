from __future__ import annotations

from typing import Literal

import torch

from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg


def _resolve_body_ids(asset_cfg: SceneEntityCfg, asset: RigidObject | Articulation) -> list[int] | None:
    if asset_cfg.body_ids == slice(None):
        return None
    if isinstance(asset_cfg.body_ids, torch.Tensor):
        return asset_cfg.body_ids.tolist()
    if isinstance(asset_cfg.body_ids, slice):
        return list(range(asset.num_bodies))
    return list(asset_cfg.body_ids)


def _get_num_shapes_per_body(asset: Articulation) -> list[int]:
    cache_name = "_dooropening_num_shapes_per_body"
    if hasattr(asset, cache_name):
        return getattr(asset, cache_name)

    num_shapes_per_body: list[int] = []
    for link_path in asset.root_physx_view.link_paths[0]:
        link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore[attr-defined]
        num_shapes_per_body.append(link_physx_view.max_shapes)
    setattr(asset, cache_name, num_shapes_per_body)
    return num_shapes_per_body


def _get_material_buckets(
    asset: RigidObject | Articulation,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
    make_consistent: bool,
) -> torch.Tensor:
    """Cache sampled material buckets per asset/range to avoid creating unbounded unique materials."""
    cache_name = "_dooropening_material_bucket_cache"
    if not hasattr(asset, cache_name):
        setattr(asset, cache_name, {})
    bucket_cache = getattr(asset, cache_name)

    # Round range endpoints so recurring ADR stages map to the same cache key.
    key = (
        round(float(static_friction_range[0]), 8),
        round(float(static_friction_range[1]), 8),
        round(float(dynamic_friction_range[0]), 8),
        round(float(dynamic_friction_range[1]), 8),
        round(float(restitution_range[0]), 8),
        round(float(restitution_range[1]), 8),
        int(num_buckets),
        bool(make_consistent),
    )
    if key in bucket_cache:
        return bucket_cache[key]

    static_friction = torch.empty((num_buckets,), device="cpu").uniform_(*static_friction_range)
    dynamic_friction = torch.empty((num_buckets,), device="cpu").uniform_(*dynamic_friction_range)
    restitution = torch.empty((num_buckets,), device="cpu").uniform_(*restitution_range)
    if make_consistent:
        dynamic_friction = torch.minimum(dynamic_friction, static_friction)

    material_buckets = torch.stack((static_friction, dynamic_friction, restitution), dim=-1)
    bucket_cache[key] = material_buckets
    return material_buckets


def randomize_rigid_body_material_compat(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
    make_consistent: bool = False,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if num_buckets <= 0:
        raise ValueError(f"num_buckets must be > 0, received {num_buckets}.")

    if env_ids is None:
        env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids_cpu = env_ids.cpu()

    material_buckets = _get_material_buckets(
        asset=asset,
        static_friction_range=static_friction_range,
        dynamic_friction_range=dynamic_friction_range,
        restitution_range=restitution_range,
        num_buckets=num_buckets,
        make_consistent=make_consistent,
    )

    total_num_shapes = asset.root_physx_view.max_shapes
    bucket_ids = torch.randint(0, num_buckets, (len(env_ids_cpu), total_num_shapes), device="cpu")
    material_samples = material_buckets[bucket_ids]
    materials = asset.root_physx_view.get_material_properties()

    body_ids = _resolve_body_ids(asset_cfg, asset)
    if body_ids is None:
        materials[env_ids_cpu] = material_samples
    elif isinstance(asset, Articulation):
        num_shapes_per_body = _get_num_shapes_per_body(asset)
        for body_id in body_ids:
            start_idx = sum(num_shapes_per_body[:body_id])
            end_idx = start_idx + num_shapes_per_body[body_id]
            materials[env_ids_cpu, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
    else:
        materials[env_ids_cpu] = material_samples

    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)


def randomize_actuator_gains_compat(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float] | None = None,
    damping_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    def randomize(data: torch.Tensor, params: tuple[float, float], actuator_indices):
        return _randomize_prop_by_op(
            data, params, dim_0_ids=None, dim_1_ids=actuator_indices, operation=operation, distribution=distribution
        )

    for actuator in asset.actuators.values():
        if isinstance(asset_cfg.joint_ids, slice):
            actuator_indices = slice(None)
            if isinstance(actuator.joint_indices, slice):
                global_indices = slice(None)
            elif isinstance(actuator.joint_indices, torch.Tensor):
                global_indices = actuator.joint_indices.to(asset.device)
            else:
                raise TypeError("Actuator joint indices must be a slice or a torch.Tensor.")
        elif isinstance(actuator.joint_indices, slice):
            global_indices = actuator_indices = torch.tensor(asset_cfg.joint_ids, device=asset.device)
        else:
            actuator_joint_indices = actuator.joint_indices
            asset_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
            if len(actuator_indices) == 0:
                continue
            global_indices = actuator_joint_indices[actuator_indices]

        if stiffness_distribution_params is not None:
            stiffness = actuator.stiffness[env_ids].clone()
            stiffness[:, actuator_indices] = asset.data.default_joint_stiffness[env_ids][:, global_indices].clone()
            randomize(stiffness, stiffness_distribution_params, actuator_indices)
            actuator.stiffness[env_ids] = stiffness
            if isinstance(actuator, ImplicitActuator):
                asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)

        if damping_distribution_params is not None:
            damping = actuator.damping[env_ids].clone()
            damping[:, actuator_indices] = asset.data.default_joint_damping[env_ids][:, global_indices].clone()
            randomize(damping, damping_distribution_params, actuator_indices)
            actuator.damping[env_ids] = damping
            if isinstance(actuator, ImplicitActuator):
                asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)


def randomize_joint_parameters_compat(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if friction_distribution_params is None:
        return

    friction_coeff = _randomize_prop_by_op(
        asset.data.default_joint_friction_coeff.clone(),
        friction_distribution_params,
        env_ids,
        joint_ids,
        operation=operation,
        distribution=distribution,
    )
    friction_coeff = torch.clamp(friction_coeff, min=0.0)

    if joint_ids == slice(None):
        static_friction = friction_coeff[env_ids]
    else:
        static_friction = friction_coeff[env_ids[:, None], joint_ids]

    dynamic_friction = None
    viscous_friction = None
    if int(env.sim.get_version()[0]) >= 5:
        dynamic_friction_coeff = _randomize_prop_by_op(
            asset.data.default_joint_dynamic_friction_coeff.clone(),
            friction_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        dynamic_friction_coeff = torch.clamp(dynamic_friction_coeff, min=0.0)
        dynamic_friction_coeff = torch.minimum(dynamic_friction_coeff, friction_coeff)

        viscous_friction_coeff = _randomize_prop_by_op(
            asset.data.default_joint_viscous_friction_coeff.clone(),
            friction_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        viscous_friction_coeff = torch.clamp(viscous_friction_coeff, min=0.0)

        if joint_ids == slice(None):
            dynamic_friction = dynamic_friction_coeff[env_ids]
            viscous_friction = viscous_friction_coeff[env_ids]
        else:
            dynamic_friction = dynamic_friction_coeff[env_ids[:, None], joint_ids]
            viscous_friction = viscous_friction_coeff[env_ids[:, None], joint_ids]

    asset.write_joint_friction_coefficient_to_sim(
        joint_friction_coeff=static_friction,
        joint_dynamic_friction_coeff=dynamic_friction,
        joint_viscous_friction_coeff=viscous_friction,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )
