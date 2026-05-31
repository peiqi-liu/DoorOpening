# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Defines the door configuration for simulation with Isaac Sim.
"""

import os
import glob
import torch
from collections import OrderedDict
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg, Articulation
from DoorOpening.assets.cache_utils import resolve_converter_cache_dir, should_force_asset_usd_conversion
from DoorOpening.constants.env_constants import DOOR_INITIAL_POS, DOOR_INITIAL_ROT
import json
from DoorOpening.utils.urdf_utils import compute_exact_door_keypoints

DOOR_SOLVER_POSITION_ITERS = 8
DOOR_SOLVER_VELOCITY_ITERS = 2
DOOR_CONTACT_OFFSET = 0.015
DOOR_REST_OFFSET = 0.002
DOOR_MAX_DEPENETRATION_VELOCITY = 150.0


def load_meta_data(board_meta_data_paths: str, handle_meta_data_paths: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load the meta data from the json files.
    """
    handle_bboxes = []
    board_bboxes = []

    for handle_path, board_path in zip(handle_meta_data_paths, board_meta_data_paths):

        # ----- Handle -----
        with open(handle_path, "r") as f:
            handle_data = json.load(f)

        handle_min = handle_data["handle_min"]
        handle_max = handle_data["handle_max"]

        # xyzxyz format
        handle_bbox = torch.tensor(handle_min + handle_max,
                                dtype=torch.float32,
                                device=device)

        handle_bboxes.append(handle_bbox)

        # ----- Board -----
        with open(board_path, "r") as f:
            board_data = json.load(f)

        board_min = board_data["min"]
        board_max = board_data["max"]

        board_bbox = torch.tensor(board_min + board_max,
                                dtype=torch.float32,
                                device=device)

        board_bboxes.append(board_bbox)
    
    return handle_bboxes, board_bboxes

def create_initial_state():
    return ArticulationCfg.InitialStateCfg(
        pos=DOOR_INITIAL_POS,
        rot=DOOR_INITIAL_ROT
    )

def create_actuators():
    # These are the base door gains before any reset-time domain randomization scales them.
    board_nominal_stiffness = 30.0
    board_nominal_damping = 10.0
    handle_nominal_stiffness = 50.0
    handle_nominal_damping = 0.6

    return {
        "joint_1": ImplicitActuatorCfg(
            joint_names_expr=["joint_1"],
            stiffness=board_nominal_stiffness,
            damping=board_nominal_damping,
        ),
        "joint_2": ImplicitActuatorCfg(
            joint_names_expr=["joint_2"],
            effort_limit_sim = 100,
            stiffness=handle_nominal_stiffness,
            damping=handle_nominal_damping,
        ),
    }

def create_urdf_door_cfg(
    asset_path: str,
    training_mode: bool = False,
    activate_contact_sensors: bool = True,
):
    cache_variant = "_".join(
        [
            "training" if training_mode else "eval",
            "contacts" if activate_contact_sensors else "no_contacts",
        ]
    )
    usd_dir = resolve_converter_cache_dir(
        asset_path,
        asset_root=asset_base_folder,
        variant=cache_variant,
    )

    return sim_utils.UrdfFileCfg(
            fix_base=True,
            merge_fixed_joints=True,
            make_instanceable=False,
            asset_path=asset_path,
            usd_dir=usd_dir,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=DOOR_MAX_DEPENETRATION_VELOCITY,
                solver_position_iteration_count=DOOR_SOLVER_POSITION_ITERS,
                solver_velocity_iteration_count=DOOR_SOLVER_VELOCITY_ITERS,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=DOOR_SOLVER_POSITION_ITERS,
                solver_velocity_iteration_count=DOOR_SOLVER_VELOCITY_ITERS,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
            force_usd_conversion=should_force_asset_usd_conversion(asset_path, usd_dir),
            # Note: joint_drive is usually not needed for URDF; PD gains can be in actuators
            # scale = (1.0, 1.2, 1.1),
            activate_contact_sensors=activate_contact_sensors,
            collider_type = "convex_hull" if training_mode else "convex_decomposition",
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=DOOR_CONTACT_OFFSET,
                rest_offset=DOOR_REST_OFFSET,
            ),
    )

def create_door_cfg(
    asset_path: str,
    training_mode: bool = False,
    activate_contact_sensors: bool = True,
) -> ArticulationCfg:
    """Helper to create an ArticulationCfg from a URDF path."""
    return ArticulationCfg(
        spawn=create_urdf_door_cfg(
            asset_path,
            training_mode=training_mode,
            activate_contact_sensors=activate_contact_sensors,
        ),
        # spawn=sim_utils.UsdFileCfg(
        #     usd_path=asset_path,
        #     scale = (1.0, 1.2, 0.95),
        #     activate_contact_sensors=True,
        # ),
        init_state=create_initial_state(),
        actuators=create_actuators(),
    )


root_path = os.path.dirname(os.path.dirname(__file__))
asset_base_folder = os.path.join(root_path, "door")
DEFAULT_MULTI_DOOR_FAMILY_NAMES = ["PartNetv5_plusplus", "PartNetv6_plusplus", "PartNetv7_plusplus", "PartNetv8_plusplus"]


def _resolve_multi_door_family_names():
    family_spec = os.environ.get("DOOROPENING_MULTI_DOOR_FAMILIES")
    if family_spec is None or family_spec.strip() == "":
        return list(DEFAULT_MULTI_DOOR_FAMILY_NAMES)

    family_names = [name.strip() for name in family_spec.split(",") if name.strip()]
    if not family_names:
        raise ValueError("DOOROPENING_MULTI_DOOR_FAMILIES did not contain any valid family names.")

    return family_names


DOOR_FAMILY_NAMES = _resolve_multi_door_family_names()
door_family_base_folders = OrderedDict(
    (family_name, os.path.join(asset_base_folder, family_name))
    for family_name in DOOR_FAMILY_NAMES
)


def _collect_multi_family_assets():
    family_asset_paths = OrderedDict()
    for family_name, family_folder in door_family_base_folders.items():
        paths = sorted(glob.glob(os.path.join(family_folder, "**/mobility.urdf"), recursive=True))
        if len(paths) == 0:
            raise FileNotFoundError(f"No mobility.urdf files found under {family_folder}")
        family_asset_paths[family_name] = paths

    asset_counts = {family_name: len(paths) for family_name, paths in family_asset_paths.items()}
    if len(set(asset_counts.values())) != 1:
        raise ValueError(f"Door families must contain the same number of assets. Counts: {asset_counts}")

    asset_paths_out = []
    family_ids_out = []
    family_names_out = []
    num_variants = len(next(iter(family_asset_paths.values())))
    for asset_idx in range(num_variants):
        for family_id, family_name in enumerate(DOOR_FAMILY_NAMES):
            asset_paths_out.append(family_asset_paths[family_name][asset_idx])
            family_ids_out.append(family_id)
            family_names_out.append(family_name)

    return asset_paths_out, family_ids_out, family_names_out


asset_paths, asset_family_ids, asset_family_names = _collect_multi_family_assets()
board_offsets = []
board_bboxes = []
handle_offsets = []

for asset_path in asset_paths:
    keypoints = compute_exact_door_keypoints(asset_path)
    board_offsets.append(keypoints["link_1"])
    board_bboxes.append(keypoints["link_1_bbox_base"])
    handle_offsets.append(keypoints["link_2"])

board_offsets = torch.tensor(board_offsets)
board_bboxes = torch.tensor(board_bboxes)
handle_offsets = torch.tensor(handle_offsets)
asset_family_ids = torch.tensor(asset_family_ids, dtype=torch.long)

motion_traj_paths = [os.path.join(os.path.dirname(asset_path), "traj.pkl") for asset_path in asset_paths]
motion_wrong_traj_paths = [os.path.join(os.path.dirname(asset_path), "traj_wrong.pkl") for asset_path in asset_paths]
motion_family_ids = asset_family_ids.clone()
motion_family_names = list(asset_family_names)


def get_multi_door_rank(rank: int | None = None) -> int:
    """Return the distributed rank used to offset the per-env door asset cycle."""

    if rank is not None:
        return int(rank)
    return int(os.environ.get("RANK", "0"))


def get_multi_door_asset_start_index(num_envs: int, rank: int | None = None) -> int:
    """Return the first door asset index for this rank's local env 0."""

    num_assets = len(asset_paths)
    if num_assets == 0:
        raise ValueError("No multi-door assets are available.")
    return (get_multi_door_rank(rank) * int(num_envs)) % num_assets


def get_multi_door_env_asset_indices(
    num_envs: int,
    rank: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Map local env ids to door asset ids with a rank-aware global env offset."""

    num_assets = len(asset_paths)
    start_index = get_multi_door_asset_start_index(num_envs, rank=rank)
    return (torch.arange(int(num_envs), device=device, dtype=torch.long) + start_index) % num_assets


def _make_ordered_asset_indices(asset_start_index: int = 0) -> list[int]:
    num_assets = len(asset_paths)
    if num_assets == 0:
        raise ValueError("No multi-door assets are available.")
    asset_start_index = int(asset_start_index) % num_assets
    return [(asset_start_index + index) % num_assets for index in range(num_assets)]


def _build_door_urdf_configs(
    training_mode: bool = False,
    activate_contact_sensors: bool = False,
    asset_start_index: int = 0,
):
    door_urdf_configs = []
    for asset_idx in _make_ordered_asset_indices(asset_start_index):
        door_urdf_configs.append(
            create_urdf_door_cfg(
                asset_paths[asset_idx],
                training_mode=training_mode,
                activate_contact_sensors=activate_contact_sensors,
            )
        )
    return door_urdf_configs


def _infer_spawn_options(door_cfg: ArticulationCfg) -> tuple[bool, bool]:
    spawn_cfg = getattr(door_cfg, "spawn", None)
    assets_cfg = getattr(spawn_cfg, "assets_cfg", None)
    if not assets_cfg:
        return False, False

    first_asset_cfg = assets_cfg[0]
    training_mode = getattr(first_asset_cfg, "collider_type", None) == "convex_hull"
    activate_contact_sensors = bool(getattr(first_asset_cfg, "activate_contact_sensors", False))
    return training_mode, activate_contact_sensors


def configure_multi_door_assets_for_rank(
    door_cfg: ArticulationCfg,
    num_envs: int,
    rank: int | None = None,
) -> int:
    """Rotate a MultiAssetSpawner so local env ids continue the global asset cycle across ranks."""

    spawn_cfg = getattr(door_cfg, "spawn", None)
    if spawn_cfg is None or not hasattr(spawn_cfg, "assets_cfg"):
        raise ValueError("Expected door_cfg.spawn to be a MultiAssetSpawnerCfg with assets_cfg.")

    asset_start_index = get_multi_door_asset_start_index(num_envs, rank=rank)
    training_mode, activate_contact_sensors = _infer_spawn_options(door_cfg)
    spawn_cfg.assets_cfg = _build_door_urdf_configs(
        training_mode=training_mode,
        activate_contact_sensors=activate_contact_sensors,
        asset_start_index=asset_start_index,
    )
    return asset_start_index


door_asset_path = asset_paths[0]
board_offset = board_offsets[0]
handle_offset = handle_offsets[0]
print("multi door families: ", ", ".join(DOOR_FAMILY_NAMES))
print("multi door asset count: ", len(asset_paths))
print("door_asset_path: ", door_asset_path)

DOOR_CONFIG = create_door_cfg(door_asset_path, training_mode=False)

DOOR_CONFIGS = []
for asset_path in asset_paths:
    DOOR_CONFIGS.append(create_door_cfg(asset_path, training_mode=False))

def setup_doors(training_mode: bool = False):
    """Load all door cfg"""
    return ArticulationCfg(
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=_build_door_urdf_configs(
                training_mode=training_mode,
                activate_contact_sensors=False,
            ),
            random_choice=False,
            activate_contact_sensors=False,
        ),
        init_state=create_initial_state(),
        actuators=create_actuators(),
    )

ALL_DOOR_CONFIGS = setup_doors()


def edit_door_articulation(
    door: Articulation, 
    nominal_joint_stiffness: torch.Tensor | None = None,
    nominal_joint_damping: torch.Tensor | None = None,
    door_closed_range = 0.01,     # radians
    hinge_range = 0.8,
    locked_stiffness = 1e6,
    locked_damping = 1e5,
    # Optional: disable the latching behavior by setting the hinge range to a negative value
    # hinge_range = -0.1,
):
    joint_idx, joint_names = door.find_joints(["joint_1", "joint_2"])
    j1 = joint_idx[joint_names.index("joint_1")]
    j2 = joint_idx[joint_names.index("joint_2")]

    # joint positions
    q = door.data.joint_pos

    # Only relock when both joints are still very close to the closed pose.
    locked = (q[:, j1].abs() < door_closed_range) & (q[:, j2].abs() < hinge_range)

    default_joint_stiffness = door.data.default_joint_stiffness
    default_joint_damping = door.data.default_joint_damping
    if nominal_joint_stiffness is None:
        nominal_joint_stiffness = default_joint_stiffness
    if nominal_joint_damping is None:
        nominal_joint_damping = default_joint_damping

    joint_stiffness = door.data.joint_stiffness.clone()
    joint_damping = door.data.joint_damping.clone()
    # Restore the normal door-panel gains every step, then temporarily override them only while latched.
    joint_stiffness[:, j1] = nominal_joint_stiffness[:, j1]
    joint_damping[:, j1] = nominal_joint_damping[:, j1]
    joint_stiffness[locked, j1] = locked_stiffness
    joint_damping[locked, j1] = locked_damping
    door.write_joint_stiffness_to_sim(joint_stiffness)
    door.write_joint_damping_to_sim(joint_damping)
