import torch

# HANDLE contact-bonus sensor: the surfaces that should be doing the gripping. For the Franka
# hand that is the two fingers plus the hand body between them. Every finger surface on this hand
# is an inward-facing grasping surface, so there is no knuckle to exclude.
HANDLE_CONTACT_FILTER_PRIM_PATHS = (
    "/World/envs/env_.*/Robot/panda_hand",
    "/World/envs/env_.*/Robot/panda_leftfinger",
    "/World/envs/env_.*/Robot/panda_rightfinger",
)

# PUSH-door hand-only handle-contact reward sensor: ONLY the hand body counts. Fingers are
# excluded entirely -- the push reward is collected by pressing the handle with the hand, not by
# finger contact (which is not rewarded on push).
PALM_ONLY_HANDLE_CONTACT_FILTER_PRIM_PATHS = (
    "/World/envs/env_.*/Robot/panda_hand",
)

# Penalize finger<->panel contact. The hand body (panda_hand) is intentionally EXCLUDED:
# pressing the panel (Door/link_1) with it is NOT penalized.
PANEL_CONTACT_FILTER_PRIM_PATHS = (
    "/World/envs/env_.*/Robot/panda_leftfinger",
    "/World/envs/env_.*/Robot/panda_rightfinger",
)

X5_BODY_NAMES = (
    "x5_base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "x5_camera_link",
)

# Track x5 contact against all articulated door bodies. The URDF contains a fixed
# link_0 frame, but the converter merges fixed joints so the runtime body is `base`.
DOOR_BODY_CONTACT_FILTER_PRIM_PATHS = (
    "/World/envs/env_.*/Door/base",
    "/World/envs/env_.*/Door/link_1",
    "/World/envs/env_.*/Door/link_2",
)
DOOR_FRAME_FILTER_INDEX = 0

# Base<->door contact penalty: the left/right vertical faces of the mobile-base cube plus the
# full chassis/mast avoid the door. front/back panels are intentionally NOT scored here (they
# remain in the self-collision set instead), and top_panel is excluded (a vertical door can't
# hit the top). The franka_control_box is NOT here either -- it has its own graded
# franka-box<->door penalty.
BASE_DOOR_CONTACT_BODY_NAMES = (
    "left_panel",   # +y face
    "right_panel",  # -y face
    # Full chassis + lidar mast (now carries a collision mesh): the base/mast hitting the door
    # is penalized just like the panels.
    "tidybot2_base_link",
)
BASE_DOOR_CONTACT_PRIM_PATH = (
    "/World/envs/env_.*/Robot/(" + "|".join(BASE_DOOR_CONTACT_BODY_NAMES) + ")"
)

# The franka control box houses the arm controller and is critical -- protect it like the x5
# camera arm: contact with any door body feeds the harsh x5<->door penalty AND early
# termination (rollout counted as a failure), rather than the milder base<->door penalty.
FRANKA_BOX_DOOR_CONTACT_PRIM_PATH = "/World/envs/env_.*/Robot/franka_control_box"

# Self-collision penalty (r_contact) sensing. Only the MOVING franka arm needs self-collision
# checks: the x5/arx camera arm and the mobile base are fixed relative to each other, so they can
# never self-collide -- the only self-collisions possible are franka<->x5 and franka<->base. So a
# SINGLE multi-body sensor on the franka arm, filtered against the x5 group + base group + the fixed
# door frame, covers everything. (x5<->frame and base<->frame contacts are already caught by the
# dedicated x5_door / base_door sensors, which filter against DOOR_BODY_CONTACT_FILTER_PRIM_PATHS
# including Door/base.) IsaacLab resolves the regex-alternation prim_path into one multi-body sensor
# whose force_matrix_w is [N, group_bodies, num_filters, 3]; the env counts franka bodies over
# threshold. PhysX already drops directly-jointed (adjacent) pairs.
#
# Excluded on purpose from THIS group: panda_link0/panda_link1 (base-adjacent arm mount), the
# panda_link7 flange, and the whole hand (finger<->finger self-collision drove poses too
# conservative). The fingers ARE separately checked against the panda_link7 flange -- see
# SELF_COLLISION_HAND_* below.
SELF_COLLISION_FRANKA_BODIES = ("panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6")
# The other two groups the franka arm is checked AGAINST (kept fixed relative to each other).
SELF_COLLISION_X5_BODIES = ("link2", "link3", "link4", "link5", "x5_camera_link")
SELF_COLLISION_BASE_BODIES = ("tidybot2_base_link", "franka_control_box")

# The fixed door frame (URDF link_0, merged to `base` at runtime): a franka body striking the
# immovable frame is scored as a self-collision too.
DOOR_FRAME_FILTER_PRIM_PATH = "/World/envs/env_.*/Door/base"


def _self_collision_group_prim_path(body_names) -> str:
    """Multi-body sensor prim_path (regex alternation) covering one self-collision group."""
    return "/World/envs/env_.*/Robot/(" + "|".join(body_names) + ")"


def _self_collision_filter_prims(*body_name_groups) -> tuple[str, ...]:
    """Robot filter prims for the given body groups, plus the fixed door frame."""
    prims = tuple(
        f"/World/envs/env_.*/Robot/{name}" for group in body_name_groups for name in group
    )
    return prims + (DOOR_FRAME_FILTER_PRIM_PATH,)


# Single franka self-collision sensor, filtered against the x5 group + base group + door frame.
SELF_COLLISION_FRANKA_PRIM_PATH = _self_collision_group_prim_path(SELF_COLLISION_FRANKA_BODIES)
SELF_COLLISION_FRANKA_FILTER_PRIM_PATHS = _self_collision_filter_prims(SELF_COLLISION_X5_BODIES, SELF_COLLISION_BASE_BODIES)

# Finger<->flange self-collision: the fingers filtered ONLY against panda_link7. The hand is
# excluded from the franka self-collision group above, but a wrist pose that folds the gripper back
# can still strike the flange behind it, which is a real collision we DO want to avoid. Because the
# filter is JUST the flange, finger<->finger contact never appears in the force matrix and is not
# penalized. panda_hand/palm_center are omitted (fixed-jointed to the flange -> adjacent contact).
FRANKA_FLANGE_BODY = "panda_link7"
HAND_DIGIT_BODIES = ("panda_leftfinger", "panda_rightfinger")
SELF_COLLISION_HAND_PRIM_PATH = _self_collision_group_prim_path(HAND_DIGIT_BODIES)
SELF_COLLISION_HAND_FILTER_PRIM_PATHS = (f"/World/envs/env_.*/Robot/{FRANKA_FLANGE_BODY}",)


def get_filtered_contact_force_w(sensor, expected_num_envs=None, filter_indices: tuple[int, ...] | None = None) -> torch.Tensor:
    force_matrix = sensor.data.force_matrix_w
    if force_matrix is None:
        raise RuntimeError(
            "Expected sensor.data.force_matrix_w but got None. "
            "Filtered contact force requires filter_prim_paths_expr on the contact sensor."
        )
    if force_matrix.ndim != 4 or force_matrix.shape[-1] != 3:
        raise RuntimeError(
            f"Expected force_matrix_w shape [N, B, M, 3], got {tuple(force_matrix.shape)}"
        )

    if filter_indices is not None:
        force_matrix = force_matrix[:, :, filter_indices, :]

    force_w = torch.nan_to_num(force_matrix, nan=0.0).sum(dim=(1, 2))
    if force_w.ndim != 2 or force_w.shape[-1] != 3:
        raise RuntimeError(
            f"Expected filtered force shape [N, 3], got {tuple(force_w.shape)}"
        )
    if expected_num_envs is not None and force_w.shape[0] != expected_num_envs:
        raise RuntimeError(
            f"Expected filtered force batch {expected_num_envs}, got {force_w.shape[0]}"
        )
    return force_w


def get_self_contact_body_force_norm(sensor, expected_num_envs=None) -> torch.Tensor:
    """Per-body self-contact force magnitude for a multi-body contact sensor.

    Returns a ``[num_envs, num_bodies]`` tensor where entry ``[n, b]`` is the magnitude
    of the net self-contact force on end-effector body ``b`` (the contact-force vectors
    are summed over all filtered self shapes before taking the norm). Threshold-count the
    bodies to obtain the self-collision penalty ``r_contact``.
    """
    force_matrix = sensor.data.force_matrix_w
    if force_matrix is None:
        raise RuntimeError(
            "Expected sensor.data.force_matrix_w but got None. "
            "Self-contact force requires filter_prim_paths_expr on the contact sensor."
        )
    if force_matrix.ndim != 4 or force_matrix.shape[-1] != 3:
        raise RuntimeError(
            f"Expected force_matrix_w shape [N, B, M, 3], got {tuple(force_matrix.shape)}"
        )
    # Sum the contact-force vectors over the filtered self shapes -> [N, B, 3].
    force_w = torch.nan_to_num(force_matrix, nan=0.0).sum(dim=2)
    if expected_num_envs is not None and force_w.shape[0] != expected_num_envs:
        raise RuntimeError(
            f"Expected self-contact force batch {expected_num_envs}, got {force_w.shape[0]}"
        )
    return torch.linalg.vector_norm(force_w, dim=-1)
