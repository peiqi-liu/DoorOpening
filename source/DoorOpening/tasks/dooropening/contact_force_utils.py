import torch

# The HANDLE contact-bonus sensor is filtered to the palm and the LOWEST (proximal) finger
# segments only -- palm_center/palm_lower + mcp_joint + pip. The distal segments and tips
# (dip / realtip / fingertip) are intentionally excluded so the bonus rewards wrapping the
# handle with the palm + finger bases, NOT wrenching a fist and pressing down from the top.
HANDLE_CONTACT_FILTER_PRIM_PATHS = (
    "/World/envs/env_.*/Robot/palm_center",
    "/World/envs/env_.*/Robot/palm_lower",
    "/World/envs/env_.*/Robot/mcp_joint_1",
    "/World/envs/env_.*/Robot/pip_1",
    "/World/envs/env_.*/Robot/mcp_joint_2",
    "/World/envs/env_.*/Robot/pip_2",
    "/World/envs/env_.*/Robot/mcp_joint_3",
    "/World/envs/env_.*/Robot/pip_3",
)

# Penalize finger<->panel contact. Unlike the handle bonus, this keeps the FULL hand set
# (palm + every finger link incl. tips): ANY part of the hand on the panel (Door/link_1) is
# undesirable, so the tips must still be tracked here.
PANEL_CONTACT_FILTER_PRIM_PATHS = (
    "/World/envs/env_.*/Robot/palm_center",
    "/World/envs/env_.*/Robot/palm_lower",
    "/World/envs/env_.*/Robot/mcp_joint_1",
    "/World/envs/env_.*/Robot/pip_1",
    "/World/envs/env_.*/Robot/dip_1",
    "/World/envs/env_.*/Robot/realtip_1",
    "/World/envs/env_.*/Robot/fingertip_1",
    "/World/envs/env_.*/Robot/mcp_joint_2",
    "/World/envs/env_.*/Robot/pip_2",
    "/World/envs/env_.*/Robot/dip_2",
    "/World/envs/env_.*/Robot/realtip_2",
    "/World/envs/env_.*/Robot/fingertip_2",
    "/World/envs/env_.*/Robot/mcp_joint_3",
    "/World/envs/env_.*/Robot/pip_3",
    "/World/envs/env_.*/Robot/dip_3",
    "/World/envs/env_.*/Robot/realtip_3",
    "/World/envs/env_.*/Robot/fingertip_3",
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

# Base<->door contact penalty: the mobile base is a cube whose +x face (front_panel) leads
# the approach and is allowed to touch the door. The other three vertical faces should never
# hit the door, plus the franka control box on top (which houses the arm controller and must
# be protected). Only front_panel and top_panel are excluded. A contact sensor watches this
# set against all door bodies.
BASE_NON_FRONT_PANEL_BODY_NAMES = (
    "back_panel",          # -x face
    "left_panel",          # +y face
    "right_panel",         # -y face
    "franka_control_box",  # top/-x controller box -- protect from door contact
)
BASE_DOOR_CONTACT_PRIM_PATH = (
    "/World/envs/env_.*/Robot/(" + "|".join(BASE_NON_FRONT_PANEL_BODY_NAMES) + ")"
)

# Bodies over which the self-collision penalty r_contact is counted. This is the
# fine-grained, per-link self-collision set: every robot link is its own entity, so
# contact is scored link-pair by link-pair (panda_link2 vs panda_link5, an x5 link vs a
# base panel, ...) instead of lumping the arm/x5/base into single blobs. Only links that
# carry collision geometry are listed.
#
# Bodies are excluded on purpose so that a *connection* (structural coincidence) is
# never read as a *collision*:
#   - panda_link0 (static arm mount) and panda_link7 (the flange the hand bolts onto):
#     each is physically coincident with its neighbour through a collision-free connector
#     link (tidybot2_base_link / palm_center), two joints away in the tree, so PhysX's
#     direct-neighbour filter does NOT drop it. The LEAP hand is now excluded entirely
#     (finger self-collision made finger poses too conservative), so panda_link7 and the
#     dedicated finger<->flange sensor below are no longer used.
#   - collision-free connector / sensor links (palm_center, realtip_*, lidar, imu, base
#     chain links, ...): they carry no collision geometry, so they never report contact and
#     cannot be sensor bodies.
# The mobile-base chassis shell (control box + panels) IS included now so an arm folding
# back onto the base is caught -- those panels carry the only base collision geometry.
# PhysX articulation self-collision already drops every directly jointed (parent-child)
# pair among the links below, so joint connections between them are not counted either.
SELF_COLLISION_BODY_NAMES = (
    # Franka arm (skip link0 mount and link7 flange; see note above)
    "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6",
    # x5 / arx camera arm
    "x5_base_link", "link1", "link2", "link3", "link4", "link5", "x5_camera_link",
    # Mobile-base chassis shell (the only base bodies that carry collision geometry).
    # NOTE: these panels are separate sibling bodies fixed to the chassis; if they abut they
    # can register a constant self-contact floor -- watch stats/self_collision_body_count_mean
    # and trim this group if it never drops to ~0.
    "franka_control_box", "front_panel", "back_panel", "left_panel", "right_panel", "top_panel",
)
# LEAP hand bodies are intentionally NOT in the self-collision set: penalizing finger
# self-collision drove the finger poses too conservative.

# One contact sensor covering every self-collision body. IsaacLab treats the leaf path
# as a regex and resolves the alternation into a single multi-body sensor, so
# force_matrix_w has shape [N, num_bodies, num_filter_shapes, 3].
SELF_COLLISION_CONTACT_PRIM_PATH = (
    "/World/envs/env_.*/Robot/(" + "|".join(SELF_COLLISION_BODY_NAMES) + ")"
)

# The "self" side of a self-collision is the same link set, so every non-adjacent link
# pair is reported. PhysX drops the self-pair of a body and the joint-connected
# (adjacent) link pairs.
SELF_COLLISION_FILTER_PRIM_PATHS = tuple(
    f"/World/envs/env_.*/Robot/{name}" for name in SELF_COLLISION_BODY_NAMES
)

# Dedicated flange sensor for "a finger folded back onto the Franka flange". The flange
# (panda_link7) is intentionally left out of the main set above because it is structurally
# coincident with palm_lower (bolted on through the collision-free palm_center connector),
# so a main-set sensor would read that permanent palm_lower<->flange contact as a
# self-collision even at a neutral / zero finger pose. Here panda_link7 is the sensor body
# and the filter is the *distal finger links only* (pip / dip / fingertip, never palm_lower
# or the mcp knuckle): a finger swinging back to the flange is reported, while the
# structural palm_lower<->flange contact is excluded by construction. At a neutral pose the
# fingers point forward, away from the flange, so this reads zero until a finger folds back.
FLANGE_CONTACT_PRIM_PATH = "/World/envs/env_.*/Robot/panda_link7"
FLANGE_SELF_COLLISION_FILTER_BODY_NAMES = (
    "pip_1", "dip_1", "fingertip_1",
    "pip_2", "dip_2", "fingertip_2",
    "pip_3", "dip_3", "fingertip_3",
    "pip_4", "dip_4", "fingertip_4",
)
FLANGE_SELF_COLLISION_FILTER_PRIM_PATHS = tuple(
    f"/World/envs/env_.*/Robot/{name}" for name in FLANGE_SELF_COLLISION_FILTER_BODY_NAMES
)


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
