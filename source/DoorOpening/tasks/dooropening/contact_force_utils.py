import torch

# Keep the handle contact sensor filtered to palm/finger bodies only so force_matrix_w
# reports robot-hand contact on Door/link_2 instead of total handle contact.
HANDLE_CONTACT_FILTER_PRIM_PATHS = (
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

# Source bodies for the x5 collision monitor. We attach the sensor to the robot-side bodies and
# filter against door bodies so PhysX does not have to reconcile a many-door-body source pattern
# against a one-robot-body-per-env filter pattern.
X5_BODY_CONTACT_SENSOR_PRIM_PATH = (
    "/World/envs/env_.*/Robot/(x5_base_link|link1|link2|link3|link4|link5|x5_camera_link)"
)

# Track x5 contact against all articulated door bodies, including the intermediate link_0 body.
DOOR_BODY_CONTACT_FILTER_PRIM_PATHS = (
    "/World/envs/env_.*/Door/link_0",
    "/World/envs/env_.*/Door/link_1",
    "/World/envs/env_.*/Door/link_2",
)


def get_filtered_contact_force_w(sensor, expected_num_envs=None) -> torch.Tensor:
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
