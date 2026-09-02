
import pinocchio as pin
import numpy as np
from DoorOpening.utils.state_machine.pin import PinocchioIKSolver
from DoorOpening.constants.robot_constants import (
    BASE_JOINT_NAMES,
    FRANKA_DEFAULT_JOINT_POS,
    FRANKA_JOINT_NAMES,
    OPEN_FINGER_JOINT_VALUES,
    CLOSE_FINGER_JOINT_VALUES,
)
from DoorOpening.utils.pose_utils import get_base_pos_and_quat, world_to_base_frame, wrap_to_pi
import torch
from DoorOpening.utils.extract_pointcloud_from_articulation import sample_pointcloud, sample_pointcloud_from_link_name
from isaaclab.utils.math import quat_apply, euler_xyz_from_quat

def compute_base_joint(base_pos_w, base_quat_w, abs_pos):
    """
    Convert desired world-frame base pose to robot-frame (x, y, theta).

    abs_pos:
      - torch or np array
      - shape (3,) or (7,)
    """
    if not isinstance(abs_pos, torch.Tensor):
        abs_pos = torch.tensor(abs_pos, dtype=torch.float32)
    
    if not isinstance(base_pos_w, torch.Tensor):
        base_pos_w = torch.tensor(base_pos_w, dtype=torch.float32)
    if not isinstance(base_quat_w, torch.Tensor):
        base_quat_w = torch.tensor(base_quat_w, dtype=torch.float32)

    base_pos_w = base_pos_w.cpu().clone()
    base_quat_w = base_quat_w.cpu().clone()
    abs_pos = abs_pos.cpu().clone()

    # --- desired world position ---
    target_pos_w = abs_pos[:, :3]

    # --- displacement in world frame (planar) ---
    dp_w = target_pos_w[:, :2] - base_pos_w[:, :2]

    # --- robot yaw from quaternion ---
    _, _, base_yaw = euler_xyz_from_quat(base_quat_w)

    # --- rotate into robot frame ---
    c = torch.cos(base_yaw)
    s = torch.sin(base_yaw)

    x_r =  c * dp_w[:, 0] + s * dp_w[:, 1]
    y_r = -s * dp_w[:, 0] + c * dp_w[:, 1]

    # --- theta action ---
    if abs_pos.numel() == 7:
        target_quat_w = abs_pos[:, 3:7]
        _, _, target_yaw = euler_xyz_from_quat(target_quat_w)
        theta_r = wrap_to_pi(target_yaw - base_yaw)
    else:
        theta_r = torch.zeros(abs_pos.shape[0], device=abs_pos.device)

    return torch.cat([x_r, y_r, theta_r], dim=-1)

def solve_ik(robot_urdf_path, q, palm_pose, base_pose, robot_initial_pose, num_attempts=8,
             reference_joint_pos=None, return_debug=False):
    # reference_joint_pos: the null-space ANCHOR the redundant arm DOF resolves toward
    # (PinocchioIKSolver applies it at reference_joint_gain=0.5, so it genuinely shapes the
    # solution branch, not just the seed). Defaults to FRANKA_DEFAULT_JOINT_POS; planners pass
    # their own when that posture puts a link somewhere the task cannot tolerate.
    # num_attempts>1 lets a failed solve retry from randomized arm seeds (robust reposition), at
    # the cost of possibly jumping to a different IK branch. Pass num_attempts=1 inside continuity
    # -critical for-loops so a hard frame returns the best-effort NEAR the previous pose (smooth)
    # instead of jumping to a distant branch.
    # return_debug surfaces what the solver actually did instead of only printing on failure.
    # Defaults to False so every existing caller keeps getting the bare (1, 10) tensor back.
    # A palm_pose of None never runs a solve at all, hence the vacuous-success defaults.
    success, debug_info = True, {}
    robot_initial_pos, robot_initial_quat = robot_initial_pose[..., :3], robot_initial_pose[..., 3:]
    if base_pose is not None:
        base_joint_pos = compute_base_joint(robot_initial_pos, robot_initial_quat, base_pose).squeeze()
        q[:3] = base_joint_pos

    ik_solver = PinocchioIKSolver(
        urdf_path=robot_urdf_path, 
        ee_link_name="panda_hand", 
        controlled_joints=BASE_JOINT_NAMES + FRANKA_JOINT_NAMES,
        reference_joint_pos=reference_joint_pos or FRANKA_DEFAULT_JOINT_POS,
    )
    # if palm_pose is not None:
    #     palm_pose[:, 0] += 0.08
    #     palm_pose[:, 1] -= 0.05
    #     palm_pose[:, 2] += 0.05
    if palm_pose is not None and palm_pose.shape[-1] == 7:
        palm_pos = palm_pose[..., :3]
        palm_quat = palm_pose[..., 3:]
        # print("palm_pos: ", palm_pos)
        # print("palm_quat: ", palm_quat)
        palm_pos, palm_quat = world_to_base_frame(robot_initial_pos, robot_initial_quat, palm_pos, palm_quat)
        palm_pos = palm_pos.squeeze().detach().numpy()
        palm_quat = palm_quat.squeeze().detach().numpy()
        # print("desired palm_quat: ", palm_quat)
        # print("desired palm_pos: ", palm_pos)
        # print("q_init: ", q)
        # num_attempts>1: try q as the first seed, then retry from randomized arm configs if it
        # fails to converge (a reachable target shouldn't be abandoned on one bad seed).
        joint_pos_des, success, debug_info = ik_solver.compute_ik(
            palm_pos, palm_quat, q_init=q, num_attempts=num_attempts
        )
        if not success:
            print(
                "[solve_ik] WARNING: IK did NOT converge -- returning best-effort (contorted) pose. "
                f"palm_pos(base frame)={np.round(palm_pos, 3)} "
                f"best_error_norm={debug_info.get('best_error_norm')}. "
                "A large error means the target is out of reach (base parked too far / bad target); "
                "a small error means the solver stalled near the goal."
            )
        q = joint_pos_des
    elif palm_pose is not None and palm_pose.shape[-1] == 3:
        palm_pos = palm_pose[..., :3]
        palm_pos = palm_pos.squeeze().detach().numpy()
        joint_pos_des, success, debug_info = ik_solver.compute_ik(palm_pos, q_init=q, num_attempts=num_attempts)
        q = joint_pos_des
    # lower_joint_limits = robot.data.soft_joint_pos_limits.clone().detach().cpu()[:, :10, 0]
    # upper_joint_limits = robot.data.soft_joint_pos_limits.clone().detach().cpu()[:, :10, 1]
    q = torch.as_tensor(q).clone().unsqueeze(0)
    # q = q.clamp(lower_joint_limits, upper_joint_limits)
    # q[:, 3:] = wrap_to_pi(q[:, 3:])
    # print("success: ", success)
    # print("debug_info: ", debug_info)
    # print("joint_pos_des: ", joint_pos_des)
    # print("fk: ", ik_solver.compute_fk(q[0, :10]))
    if return_debug:
        return q, success, debug_info
    return q

def get_board_pos(door_urdf_path, door_initial_pose, joint_angles):
    door_initial_pos, door_initial_quat = (
        door_initial_pose[..., :3],
        door_initial_pose[..., 3:]
    )

    # Sample point cloud (link frame)
    pointcloud = sample_pointcloud_from_link_name(
        door_urdf_path, joint_angles, "link_1", device="cpu"
    )

    door_pointcloud = quat_apply(door_initial_quat, pointcloud) + door_initial_pos
    P = door_pointcloud.squeeze()

    return P.mean(dim = 0).unsqueeze(0)


def get_board_edge(
    door_urdf_path,
    door_initial_pose,
    joint_angles,
    edge_inset_ratio: float = 0.12,
    min_inset: float = 0.03,
    max_inset: float = 0.10,
):
    """
    Return a point near the door-board free edge, but inset toward the panel
    center so the contact stays deeper than a true edge touch.

    The estimate is inspired by the door keypoint logic: we recover the board
    width direction from the board cloud, choose the edge farther from the door
    root, and move inward along that width axis by a small amount.
    """
    door_initial_pos, door_initial_quat = (
        door_initial_pose[..., :3],
        door_initial_pose[..., 3:]
    )

    pointcloud = sample_pointcloud_from_link_name(
        door_urdf_path, joint_angles, "link_1", device="cpu"
    )

    door_pointcloud = quat_apply(door_initial_quat, pointcloud) + door_initial_pos
    P = door_pointcloud

    centroid = P.mean(dim=1)                  # (B, 3)
    centered = P - centroid.unsqueeze(1)      # (B, N, 3)

    cov = torch.matmul(centered.transpose(1, 2), centered) / P.shape[1]
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # For a door board: smallest eigenvector ~= thickness, largest ~= height,
    # middle ~= width. We use the width axis to locate the free edge.
    width_dir = eigvecs[:, :, 1]
    width_dir = width_dir / width_dir.norm(dim=1, keepdim=True).clamp_min(1e-6)

    proj = (centered * width_dir.unsqueeze(1)).sum(dim=2)   # (B, N)
    min_proj = proj.min(dim=1).values.unsqueeze(1)
    max_proj = proj.max(dim=1).values.unsqueeze(1)

    edge_min = centroid + min_proj * width_dir
    edge_max = centroid + max_proj * width_dir

    # Use the edge farther from the door root as the free edge.
    dist_min = torch.linalg.norm(edge_min - door_initial_pos, dim=1, keepdim=True)
    dist_max = torch.linalg.norm(edge_max - door_initial_pos, dim=1, keepdim=True)
    use_max = dist_max >= dist_min

    board_width = (max_proj - min_proj).abs()
    inset = (board_width * edge_inset_ratio).clamp(min=min_inset, max=max_inset)
    edge_proj = torch.where(use_max, max_proj - inset, min_proj + inset)

    edge_pos = centroid + edge_proj * width_dir
    return edge_pos.cpu()

def get_hinge_pos(door_urdf_path, door_initial_pose, joint_angles):
    door_initial_pos, door_initial_quat = (
        door_initial_pose[..., :3],
        door_initial_pose[..., 3:]
    )

    # Sample point cloud (link frame)
    pointcloud = sample_pointcloud_from_link_name(
        door_urdf_path, joint_angles, "link_2", device="cpu"
    )

    # Transform to world frame
    door_pointcloud = quat_apply(door_initial_quat, pointcloud) + door_initial_pos
    P = door_pointcloud

    # --------------------------------------------------
    # 1. Estimate rotation axis via PCA (per batch)
    # --------------------------------------------------
    centroid = P.mean(dim=1)               # (B, 3)

    centered = P - centroid.unsqueeze(1)   # (B, N, 3)

    # covariance: (B, 3, 3)
    cov = torch.matmul(centered.transpose(1, 2), centered) / P.shape[1]

    eigvals, eigvecs = torch.linalg.eigh(cov)

    # smallest eigenvalue → axis direction
    axis_dir = eigvecs[:, :, 0]            # (B, 3)
    axis_dir = axis_dir / axis_dir.norm(dim=1, keepdim=True)

    # --------------------------------------------------
    # 2. Estimate axis origin
    # --------------------------------------------------
    # project centroid onto axis
    dot = (centroid * axis_dir).sum(dim=1, keepdim=True)   # (B,1)
    axis_origin = centroid - dot * axis_dir                # (B,3)

    # --------------------------------------------------
    # 3. Compute radial distances
    # --------------------------------------------------
    v = P - axis_origin.unsqueeze(1)       # (B, N, 3)

    proj = (v * axis_dir.unsqueeze(1)).sum(dim=2, keepdim=True) \
           * axis_dir.unsqueeze(1)

    radial = v - proj                      # (B, N, 3)
    radial_dist = radial.norm(dim=2)       # (B, N)

    # --------------------------------------------------
    # 4. Extract lever points
    # --------------------------------------------------
    threshold = 0.6 * radial_dist.max(dim=1, keepdim=True).values  # (B,1)

    mask = radial_dist > threshold         # (B, N)

    hinge_pos = []

    for b in range(P.shape[0]):
        lever_points = P[b][mask[b]]

        if lever_points.shape[0] < 10:
            lever_points = P[b]

        hinge_pos.append(lever_points.mean(dim=0))

    hinge_pos = torch.stack(hinge_pos, dim=0)   # (B, 3)

    return hinge_pos.cpu()

def get_handle_bar_pos(door_urdf_path, door_initial_pose, joint_angles, protrusion_pct=75.0):
    """World position of the LEVER BAR itself -- the part a gripper can actually close around.

    Use this, not get_hinge_pos, for anything that has to touch the handle.

    get_hinge_pos returns a point derived from the whole of link_2, which is escutcheon PLATE +
    lever bar + return hook. It does try to isolate the lever, by taking the points furthest from
    a PCA axis -- but for a plate-dominated mesh the smallest-eigenvalue direction is the PLATE'S
    NORMAL, so "radial distance" ends up measuring spread across the plate face and the outermost
    points are the plate's own corners. Averaging those lands back on the plate. Rendered meshes
    show the consequence plainly: the jaws close on the escutcheon while the lever bar hangs free
    beside them, and the door is never actually grasped. Distance-to-link_2-centroid checks cannot
    see this, because that centroid is exactly the wrong point.

    The split that IS physically meaningful is PROTRUSION from the door face: the plate lies flush
    against the panel, the lever stands off it. So fit the panel plane, orient its normal toward
    the handle, and keep the handle points standing furthest off it.
    """
    door_initial_pos, door_initial_quat = door_initial_pose[..., :3], door_initial_pose[..., 3:]

    panel = sample_pointcloud_from_link_name(door_urdf_path, joint_angles, "link_1", device="cpu")
    handle = sample_pointcloud_from_link_name(door_urdf_path, joint_angles, "link_2", device="cpu")
    panel = quat_apply(door_initial_quat, panel) + door_initial_pos
    handle = quat_apply(door_initial_quat, handle) + door_initial_pos

    out = []
    for b in range(handle.shape[0]):
        pan_b, han_b = panel[b], handle[b]
        centre = pan_b.mean(dim=0)
        # Panel plane normal = smallest singular direction of the panel cloud.
        normal = torch.linalg.svd(pan_b - centre)[2][-1]
        # Orient it toward the handle, so "protruding" is unambiguous at any door angle.
        if torch.dot(han_b.mean(dim=0) - centre, normal) < 0:
            normal = -normal
        protrusion = (han_b - centre) @ normal
        cut = torch.quantile(protrusion, protrusion_pct / 100.0)
        standing = han_b[protrusion >= cut]
        if standing.shape[0] < 10:
            out.append(han_b.mean(dim=0))
            continue
        # The standing-off points are stem-top + lever + return hook. Their centroid is still
        # dragged back toward the stem, so walk out along the lever instead: its long axis is the
        # principal direction of this subset, and the graspable span is the OUTER half of it
        # (the stem sits at the pivot end). Taking the mid-point of that outer half puts the
        # target on the bar itself rather than at its root.
        sub_centre = standing.mean(dim=0)
        axis = torch.linalg.svd(standing - sub_centre)[2][0]
        along = (standing - sub_centre) @ axis
        # Orient the axis to point AWAY from the pivot (the handle-assembly centroid).
        if torch.dot(sub_centre - han_b.mean(dim=0), axis) < 0:
            axis, along = -axis, -along
        outer = standing[along >= torch.quantile(along, 0.35)]
        out.append(outer.mean(dim=0) if outer.shape[0] >= 5 else standing.mean(dim=0))
    return torch.stack(out, dim=0).cpu()


def sample_pointcloud(urdf_path, joint_angles, initial_pose):
    pointcloud = sample_pointcloud(urdf_path, joint_angles.cpu(), device="cpu")
    pointcloud = quat_apply(initial_pose[..., 3:], pointcloud) + initial_pose[..., :3]
    return pointcloud

def open_hand(value):
    joint_names = list(OPEN_FINGER_JOINT_VALUES.keys())
    joint_values = torch.zeros(len(joint_names), dtype=torch.float32)
    for id in range(len(joint_names)):
        joint_values[id] = (OPEN_FINGER_JOINT_VALUES[joint_names[id]] - CLOSE_FINGER_JOINT_VALUES[joint_names[id]]) * value + CLOSE_FINGER_JOINT_VALUES[joint_names[id]]
    return joint_values


def solve_ik_iter(robot, palm_pose, base_pose, iters=15):
    q = robot.data.joint_pos.clone().cpu()
    joint_ids, joint_names = robot.find_joints(BASE_JOINT_NAMES + FRANKA_JOINT_NAMES)
    q = q[:, joint_ids].squeeze().detach().numpy()
    base_joint_pos = compute_base_joint(robot, base_pose).squeeze().detach().numpy()
    q[:3] = base_joint_pos
    initial_joint_pos = {}
    for name in joint_names:
        initial_joint_pos[name] = q[joint_names.index(name)]

    current_palm_pos = robot.data.body_pos_w[:, robot.find_bodies("palm_center")[0]].cpu().clone().detach().squeeze().numpy()
    current_palm_quat = robot.data.body_quat_w[:, robot.find_bodies("palm_center")[0]].cpu().clone().detach().squeeze().numpy()

    ik_solver = PinocchioIKSolver(
        urdf_path=robot.cfg.spawn.asset_path,
        ee_link_name="palm_center",
        controlled_joints=BASE_JOINT_NAMES + FRANKA_JOINT_NAMES,
        reference_joint_pos=FRANKA_DEFAULT_JOINT_POS,
    )
    # ik_solver = PositionIKOptimizer(
    #     ik_solver=PinocchioIKSolver(urdf_path=robot.cfg.spawn.asset_path, ee_link_name="palm_center", controlled_joints=["base_x_joint", "base_y_joint", "base_rotation_joint", "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]),
    #     pos_error_tol=0.01,
    #     ori_error_range=0.01,
    #     pos_weight=1.0,
    #     ori_weight=0.0,
    # )
    if palm_pose is not None and palm_pose.shape[-1] == 7:
        base_pos, base_quat = get_base_pos_and_quat(robot)
        base_pos, base_quat = base_pos.detach().cpu(), base_quat.detach().cpu()
        palm_pos = palm_pose[..., :3]
        palm_quat = palm_pose[..., 3:]
        # print("palm_pos: ", palm_pos)
        # print("palm_quat: ", palm_quat)
        palm_pos, palm_quat = world_to_base_frame(base_pos, base_quat, palm_pos, palm_quat)
        palm_pos = palm_pos.squeeze().detach().numpy()
        palm_quat = palm_quat.squeeze().detach().numpy()
        # print("desired palm_quat: ", palm_quat)
        # print("desired palm_pos: ", palm_pos)
        # print("current_palm_pos: ", current_palm_pos)
        # print("current_palm_quat: ", current_palm_quat)
        joint_pos_des, success, debug_info = ik_solver.compute_ik(palm_pos, palm_quat, q_init=q)
        q = joint_pos_des
    elif palm_pose is not None and palm_pose.shape[-1] == 3:
        palm_pos = palm_pose[..., :3]
        palm_pos = palm_pos.squeeze().detach().numpy()
        joint_pos_des, success, debug_info = ik_solver.compute_ik(palm_pos, q_init=q)
        q = joint_pos_des
    q[:3] = base_joint_pos
    # lower_joint_limits = robot.data.soft_joint_pos_limits.clone().detach().cpu()[:, :10, 0]
    # upper_joint_limits = robot.data.soft_joint_pos_limits.clone().detach().cpu()[:, :10, 1]
    q = torch.as_tensor(q).clone().unsqueeze(0)
    # q = q.clamp(lower_joint_limits, upper_joint_limits)
    q[3:] = wrap_to_pi(q[3:])
    print("success: ", success)
    print("debug_info: ", debug_info)
    print("joint_pos_des: ", q)
    # print("fk: ", ik_solver.compute_fk(q))
    return q.to(robot.data.joint_pos)
