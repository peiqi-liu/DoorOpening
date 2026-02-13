from DoorOpening.utils.state_machine.api import solve_ik, get_hinge_pos, open_hand
import torch
from isaaclab.utils.math import quat_from_euler_xyz, quat_from_matrix, combine_frame_transforms
from DoorOpening.constants.robot_constants import FULL_JOINT_NAMES, CAMERA_JOINT_DEFAULT_VALUES, DEFAULT_JOINT_POS, OPEN_FINGER_JOINT_VALUES, ROBOT_KEY_BODY_NAMES, DM_JOINT_NAMES
import numpy as np
import time
from DoorOpening.constants.env_constants import ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT, DOOR_INITIAL_POS, DOOR_INITIAL_ROT
import pickle as pkl
import os
from scipy.interpolate import CubicSpline

import viser
from viser.extras import ViserUrdf

from yourdfpy import URDF
from DoorOpening.utils.state_machine.pin import PinocchioIKSolver
import glob


def get_robot_constants():
    all_joint_names = FULL_JOINT_NAMES + list(CAMERA_JOINT_DEFAULT_VALUES.keys())
    default_joint_pos_dict = {}
    joint_vector = torch.zeros(len(all_joint_names))
    for i, name in enumerate(all_joint_names):
        if name in DEFAULT_JOINT_POS:
            default_joint_pos_dict[name] = DEFAULT_JOINT_POS[name]
            joint_vector[i] = DEFAULT_JOINT_POS[name]
        elif name in OPEN_FINGER_JOINT_VALUES:
            default_joint_pos_dict[name] = OPEN_FINGER_JOINT_VALUES[name]
            joint_vector[i] = OPEN_FINGER_JOINT_VALUES[name]
        else:
            default_joint_pos_dict[name] = 0.0
            joint_vector[i] = 0.0
    return default_joint_pos_dict, joint_vector
    

def get_rotation_quat(roll, pitch, yaw, device):
    # yaw z
    # roll x
    # pitch y
    return quat_from_euler_xyz(roll = torch.tensor([[roll]]).to(device), pitch = torch.tensor([[pitch]]).to(device), yaw = torch.tensor([[yaw]]).to(device)).squeeze(0)

import torch

def state_machine_offline(
    robot_urdf_path,
    door_urdf_path,
    robot_initial_pose,   # (1, 7) world
    door_initial_pose,    # (1, 7) world
    robot_initial_q,      # (ndof,)
    door_initial_q,       # (2,) [board, hinge]
    device="cpu",
):
    """
    Returns:
        robot_traj: list[Tensor(ndof)]
        door_traj:  list[Tensor(2)]
    """

    robot_traj = []
    door_traj = []
    key_idx_in_key_indices = []

    q_robot = robot_initial_q.clone()
    q_door = door_initial_q.clone()

    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())

    # -------------------------
    # Step 1: Pregrasp
    # -------------------------
    handle_pos = get_hinge_pos(
        door_urdf_path,
        door_initial_pose,
        q_door.unsqueeze(0)
    ).to(device)

    base_target_pos = handle_pos.clone()
    base_target_pos[:, 0] += 0.6
    base_target_pos[:, 1] -= 0.3
    base_target_rot = torch.tensor([[0, 0, 0, 1]], device=device)
    base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)

    palm_target_pos = handle_pos.clone()
    palm_target_pos[:, 0] += 0.25
    palm_target_pos[:, 1] -= 0.1
    palm_target_pos[:, 2] += 0.25
    palm_target_rot = get_rotation_quat(0.0 + torch.pi, 0.0 + torch.pi, torch.pi, device)
    palm_target_pose = torch.cat([palm_target_pos, palm_target_rot], dim=-1)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]

    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())

    key_idx_in_key_indices.append(len(robot_traj) - 1)

    # -------------------------
    # Step 2: Move to grasp
    # -------------------------
    palm_target_pos = handle_pos.clone()
    palm_target_pos[:, 0] += 0.05
    palm_target_pos[:, 1] -= 0.1
    palm_target_pos[:, 2] += 0.05
    palm_target_rot = get_rotation_quat(0.0 + torch.pi, 0.0 + torch.pi, torch.pi, device)
    palm_target_pose = torch.cat([palm_target_pos, palm_target_rot], dim=-1)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]

    q_robot[10:10+16] = open_hand(0.7)

    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())

    key_idx_in_key_indices.append(len(robot_traj) - 1)

    # -------------------------
    # Step 3: Rotate hinge (unlatch)
    # -------------------------
    q_door = torch.tensor([0.0, 1.0], device=device)
    palm_target_pose[:, 2] -= 0.04
    palm_target_pose[:, 1] -= 0.05
    palm_target_pose[:, 3:] = get_rotation_quat(0.0 + torch.pi, 0.0 + torch.pi, torch.pi + 1.0, device)
    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]
    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())

    key_idx_in_key_indices.append(len(robot_traj) - 1)

    # -------------------------
    # Step 4: Pull door open
    # -------------------------
    for theta in torch.arange(0.25, 1.16, 0.1):
        q_door = torch.tensor([theta, 0], device=device)

        new_handle_pos = get_hinge_pos(
            door_urdf_path,
            door_initial_pose,
            q_door.unsqueeze(0)
        ).to(device)

        base_target_pos = new_handle_pos.clone()
        base_target_pos[:, 0] += 0.55
        base_target_pos[:, 1] = theta / 1.45 * 0.25
        base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)

        palm_target_rot = get_rotation_quat(-min(theta.item(), 0.8) + torch.pi, 0.0 + torch.pi, torch.pi, device)

        new_handle_pos[:, 0] += (0.05 * torch.cos(theta) - 0.1 * torch.sin(theta))
        new_handle_pos[:, 1] -= (0.1 * torch.cos(theta) + 0.05 * torch.sin(theta))
        new_handle_pos[:, 2] += 0.05
        palm_target_pose = torch.cat([new_handle_pos, palm_target_rot], dim=-1)

        q_robot[:10] = solve_ik(
            robot_urdf_path,
            q_robot[:10],
            palm_pose=palm_target_pose,
            base_pose=base_target_pose,
            robot_initial_pose=robot_initial_pose,
        )[0]

        robot_traj.append(q_robot.clone())
        door_traj.append(q_door.clone())

    key_idx_in_key_indices.append(len(robot_traj) - 1)

    # -------------------------
    # Step 5 : Base forward
    # -------------------------
    palm_target_pose[:, 0] += 0.45
    palm_target_pose[:, 1] += 0.08
    # palm_target_pose[:, 2] -= 0.1
    palm_target_pose[:, 3:] = get_rotation_quat(0.0 + torch.pi, 0.0 + torch.pi, torch.pi, device)

    q_robot[10:10+16] = open_hand(1.0)
    q_door = torch.tensor([1.5, 0.0], device=device)

    for _ in range(1):
        base_target_pos[:, 0] -= 0.25
        base_target_pos[:, 1] = 0.1
        base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)

        q_robot[:10] = solve_ik(
            robot_urdf_path,
            q_robot[:10],
            palm_pose=palm_target_pose,
            base_pose=base_target_pose,
            robot_initial_pose=robot_initial_pose,
        )[0]

        robot_traj.append(q_robot.clone())
        door_traj.append(q_door.clone())

    key_idx_in_key_indices.append(len(robot_traj) - 1)

    # -------------------------
    # Step 7: Move base completely through the door
    # -------------------------
    delta_palm_pos = palm_target_pos - base_target_pos
    base_target_pos[:, 0] = 0.0
    base_target_pos[:, 1] = 0.0
    base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)

    # palm_pose = base_target_pos.clone()
    # palm_pose[:, :3] += delta_palm_pos[:, :3]
    # palm_pose = torch.cat([palm_pose, base_target_rot], dim=-1)
    prev_q = q_robot[3:10].clone()

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        # palm_pose=palm_pose,
        palm_pose=None,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]
    q_robot[3:10] = prev_q
    q_robot[10:10+16] = open_hand(1.0)
    q_door = torch.tensor([1.5, 0.0], device="cpu")

    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())

    key_idx_in_key_indices.append(len(robot_traj) - 1)

    base_target_pos[:, 0] = -1.0
    base_target_pos[:, 1] = 0.0
    base_target_pose = torch.cat([base_target_pos, base_target_rot], dim=-1)

    # palm_pose = base_target_pos.clone()
    # palm_pose[:, 2] = 0.75
    # palm_pose[:, 0] -= 0.5
    # palm_pose = torch.cat([palm_pose, base_target_rot], dim=-1)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=None,
        # palm_pose=palm_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]

    q_robot[3:10] = robot_initial_q[3:10]
    q_door = torch.tensor([0.0, 0.0], device="cpu")

    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())

    return robot_traj, door_traj, key_idx_in_key_indices

def collocate_and_playback(robot_traj, door_traj, length=1000):
    robot_traj = torch.stack(robot_traj).detach().cpu().numpy()
    door_traj = torch.stack(door_traj).detach().cpu().numpy()
    traj = np.concatenate([robot_traj, door_traj], axis=-1)
    N, D = traj.shape

    # segment lengths -> timing
    dq = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
    dt = np.maximum(dq / 1.0, 0.1)
    seg_ratios = dt / dt.sum()

    traj_out = []
    traj_d_out = []
    key_indices = [0]

    samples_used = 0

    for i in range(N - 1):
        p0 = traj[i]
        p1 = traj[i + 1]

        # allocate samples for this segment
        seg_len = int(np.round(seg_ratios[i] * length))
        if i == N - 2:  # last segment: fill remainder
            seg_len = length - samples_used
        samples_used += seg_len

        key_indices.append(key_indices[-1] + seg_len)

        # local time [0, 1]
        t_local = np.array([0.0, 1.0])
        cs = CubicSpline(t_local, np.stack([p0, p1]), axis=0, bc_type="clamped")

        t_samples = np.linspace(0.0, 1.0, seg_len, endpoint=False if i < N - 2 else True)

        seg_traj = cs(t_samples)
        seg_traj_d = cs(t_samples, 1)

        traj_out.append(seg_traj)
        traj_d_out.append(seg_traj_d)

    traj = torch.tensor(np.concatenate(traj_out, axis=0), dtype=torch.float32)
    traj_d = torch.tensor(np.concatenate(traj_d_out, axis=0), dtype=torch.float32)
    print(key_indices)

    # # automatic timing
    # dq = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
    # dt = np.maximum(dq / 1.0, 0.1)
    # t = np.concatenate([[0], np.cumsum(dt)])
    # t /= t[-1]

    # cs = CubicSpline(t, traj, axis=0, bc_type="clamped")

    # t = np.linspace(0, 1, length)
    # traj = torch.tensor(cs(t))
    # traj_d = torch.tensor(cs(t, 1))

    robot_traj = traj[:, :-2]
    door_traj = traj[:, -2:]
    door_traj[:, 0] = door_traj[:, 0].clamp(min=0.0, max=1.5)
    door_traj[:, 1] = door_traj[:, 1].clamp(min=0.0, max=1.0)
    robot_traj_d = traj_d[:, :-2]
    door_traj_d = traj_d[:, -2:]

    return robot_traj, door_traj, robot_traj_d, door_traj_d, key_indices


def play_trajectories_in_viser(
    robot_urdf,
    door_urdf,
    robot_traj: list[torch.Tensor] | torch.Tensor,   # (T, Ndof)
    door_traj: list[torch.Tensor] | torch.Tensor,    # (T, 2)
    robot_world_pos: torch.Tensor,                  # (T, 3)
    door_world_pos: torch.Tensor,                   # (T, 3)
    robot_world_quat: torch.Tensor,                  # (T, 4)
    door_world_quat: torch.Tensor,                   # (T, 4)
    hz: float = 60.0,
):
    """
    robot_traj: (T, Ndof) torch.Tensor or list of tensors
    door_traj:  (T, 2)    torch.Tensor or list of tensors
    sample_pc_fn(q_door_np) -> (N, 3) numpy array in world frame (optional)
    """

    # Normalize input format
    if isinstance(robot_traj, list):
        robot_traj = torch.stack(robot_traj, dim=0)
    if isinstance(door_traj, list):
        door_traj = torch.stack(door_traj, dim=0)

    T = min(len(robot_traj), len(door_traj))

    # -------------------------
    # Start viser
    # -------------------------
    server = viser.ViserServer()

    server.scene.add_frame(
        "/robot_root",
        position=robot_world_pos,
        wxyz=robot_world_quat,
    )

    server.scene.add_frame(
        "/door_root",
        position=door_world_pos,
        wxyz=door_world_quat,
    )

    # server.scene.add_frame(
    #     "/palm_center",
    #     wxyz=(1.0, 0.0, 0.0, 0.0),
    #     position=robot_world_pos,
    # )

    viser_robot = ViserUrdf(server, urdf_or_path=robot_urdf, root_node_name="/robot_root", load_meshes=True)
    viser_door  = ViserUrdf(server, urdf_or_path=door_urdf, root_node_name="/door_root", load_meshes=True)

    # -------------------------
    # GUI controls
    # -------------------------
    playing = True
    speed = 1.0

    with server.gui.add_folder("Playback"):
        play_btn = server.gui.add_button("Play")
        pause_btn = server.gui.add_button("Pause")
        reset_btn = server.gui.add_button("Reset")
        speed_slider = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)

    @play_btn.on_click
    def _(_):
        nonlocal playing
        playing = True

    @pause_btn.on_click
    def _(_):
        nonlocal playing
        playing = False

    @reset_btn.on_click
    def _(_):
        nonlocal t_idx
        t_idx = 0

    # -------------------------
    # Playback loop
    # -------------------------
    t_idx = 0

    print("Viser running. Open the URL in your browser.")

    # timestamp = time.time()

    # while time.time() - timestamp < 60:
    while True:
        if playing:
            q_robot = robot_traj[t_idx].detach().cpu().numpy()
            q_door  = door_traj[t_idx].detach().cpu().numpy()


            # hinge_pos = get_hinge_pos(
            #     door_urdf_path,
            #     door_initial_pose,
            #     torch.tensor(q_door)
            # ).squeeze().detach().cpu().numpy()
            # fk_pos, fk_quat = ik_solver.compute_fk(q_robot[:10])
            # fk_pos, fk_quat = base_to_world_frame(torch.tensor(robot_world_pos).to(torch.float32), torch.tensor(robot_world_quat).to(torch.float32), torch.tensor(fk_pos).to(torch.float32), torch.tensor(fk_quat).to(torch.float32))
            # server.scene.add_frame(
            #     "/hinge",
            #     wxyz=(1.0, 0.0, 0.0, 0.0),
            #     position=hinge_pos,
            # )
            # server.scene.add_frame(
            #     "/fk",
            #     wxyz=fk_quat[[3, 0, 1, 2]],
            #     position=fk_pos,
            # )

            viser_robot.update_cfg(q_robot)
            viser_door.update_cfg(q_door)

            # t_idx = (t_idx + 1) % T
            t_idx += 1
            if t_idx == T:
                break

        time.sleep(1.0 / (hz * speed_slider.value))
    server.stop()


def play_and_save_traj(robot_urdf_path, door_urdf_path):
    dir_path = os.path.dirname(door_urdf_path)
    robot_initial_pose = torch.tensor([[ROBOT_INITIAL_POS[0], ROBOT_INITIAL_POS[1], ROBOT_INITIAL_POS[2], ROBOT_INITIAL_ROT[0], ROBOT_INITIAL_ROT[1], ROBOT_INITIAL_ROT[2], ROBOT_INITIAL_ROT[3]]], device="cpu")
    door_initial_pose = torch.tensor([[DOOR_INITIAL_POS[0], DOOR_INITIAL_POS[1], DOOR_INITIAL_POS[2], DOOR_INITIAL_ROT[0], DOOR_INITIAL_ROT[1], DOOR_INITIAL_ROT[2], DOOR_INITIAL_ROT[3]]], device="cpu")
    robot_constants, robot_initial_q = get_robot_constants()
    door_initial_q = torch.tensor([0.0, 0.0], device="cpu")
    robot_traj, door_traj, key_idx_in_key_indices = state_machine_offline(robot_urdf_path, door_urdf_path, robot_initial_pose, door_initial_pose, robot_initial_q, door_initial_q, device="cpu")
    torch.set_printoptions(precision=4, sci_mode=False)
    print(torch.stack(robot_traj)[:, :10])
    # new_robot_traj = []
    # new_door_traj = []
    # for i, (robot_point, door_point) in enumerate(zip(robot_traj, door_traj)):
    #     if i in key_idx_in_key_indices:
    #         for _ in range(100):
    #             new_robot_traj.append(robot_point)
    #             new_door_traj.append(door_point)
    # robot_traj = new_robot_traj
    # door_traj = new_door_traj
    robot_traj, door_traj, robot_traj_d, door_traj_d, key_indices = collocate_and_playback(robot_traj, door_traj, length=1000)
    print(robot_traj.shape)
    print(door_traj.shape)
    print(robot_traj_d.shape)
    print(door_traj_d.shape)

    robot_urdf = URDF.load(robot_urdf_path)
    door_urdf  = URDF.load(door_urdf_path)

    robot_world_pos = robot_initial_pose[:, :3].squeeze(0).numpy()
    door_world_pos = door_initial_pose[:, :3].squeeze(0).numpy()
    robot_world_quat = robot_initial_pose[:, 3:].squeeze(0).numpy()
    door_world_quat = door_initial_pose[:, 3:].squeeze(0).numpy()

    play_trajectories_in_viser(
        robot_urdf=robot_urdf,
        door_urdf=door_urdf,
        robot_traj=robot_traj,
        door_traj=door_traj,
        robot_world_pos=robot_world_pos,
        door_world_pos=door_world_pos,
        robot_world_quat=robot_world_quat,
        door_world_quat=door_world_quat,
        hz=60,
    )

    robot_ik_solver = PinocchioIKSolver(
        urdf_path=robot_urdf_path, 
        ee_link_name="palm_center", 
        controlled_joints=DM_JOINT_NAMES
    ) 

    robot_key_bodies = ROBOT_KEY_BODY_NAMES
    robot_body_pos_traj = []
    robot_body_quat_traj = []
    
    for robot_point in robot_traj:
        body_poses = []
        body_quats = []
        for node_a in robot_key_bodies:
            transform = robot_ik_solver.get_frame_pose(config = robot_point[:10], node_b = "base_link", node_a = node_a)
            translation, rotation = torch.tensor(transform.translation).unsqueeze(0).float(), torch.tensor(transform.rotation).unsqueeze(0).float()
            quat = quat_from_matrix(rotation)
            body_world_pos, body_world_quat = combine_frame_transforms(t01 = torch.tensor(robot_world_pos).unsqueeze(0).float(), q01 = torch.tensor(robot_world_quat).unsqueeze(0).float(), t12 = translation, q12 = quat)
            body_poses.append(body_world_pos.squeeze())
            body_quats.append(body_world_quat.squeeze())
        robot_body_pos_traj.append(torch.stack(body_poses, dim=0))
        robot_body_quat_traj.append(torch.stack(body_quats, dim=0))

    robot_body_pos_traj = torch.stack(robot_body_pos_traj, dim=0)
    robot_body_quat_traj = torch.stack(robot_body_quat_traj, dim=0)

    print(robot_body_pos_traj.shape)
    print(robot_body_quat_traj.shape)

    # print(robot_key_bodies)
    # for i in torch.arange(0, robot_body_pos_traj.shape[0], 100):
    #     print(i, robot_body_pos_traj[i], robot_body_quat_traj[i])

    data = {
        "door_traj": door_traj, 
        "robot_body_pos_traj": robot_body_pos_traj,
        "robot_body_quat_traj": robot_body_quat_traj,
        "robot_joint_pos_traj": robot_traj,
        "robot_joint_vel_traj": robot_traj_d,
        "key_indices": torch.tensor(key_indices, dtype=torch.int32)[key_idx_in_key_indices]
        # "key_indices": key_indices
    }
    print(torch.tensor(key_indices, dtype=torch.int32)[key_idx_in_key_indices])
    with open(os.path.join(dir_path, "traj.pkl"), "wb") as f:
        pkl.dump(data, f)
        print("Trajectory saved to " + os.path.join(dir_path, "traj.pkl"))


if __name__ == "__main__":
    robot_urdf_path = "/home/glorbo4/peiqi/DoorOpening/source/DoorOpening/assets/glorbot/glorbot.urdf"
    # door_urdf_path = "/home/glorbo4/peiqi/DoorOpening/source/DoorOpening/assets/door/PartNetv4/99650089960001/mobility.urdf"
    # door_urdf_path = "/home/glorbo4/peiqi/DoorOpening/source/DoorOpening/assets/door/PartNetv4/99655059960012/mobility.urdf"

    root_path = "source/DoorOpening/assets/"
    asset_base_folder = os.path.join(root_path, "door/PartNetv4")
    asset_paths = sorted(glob.glob(os.path.join(asset_base_folder, "**/mobility.urdf"), recursive=True), reverse=False)

    for i, door_urdf_path in enumerate(asset_paths):
        play_and_save_traj(robot_urdf_path, door_urdf_path)
        print("Finished processing ", door_urdf_path, ", index: ", i)
    
