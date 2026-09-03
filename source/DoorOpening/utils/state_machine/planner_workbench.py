"""Viser workbench for co-authoring offline door planners with an LLM.

The product of a session is SOURCE CODE -- a new planner module shaped like
``offline_pull_door.py`` / ``offline_push_door.py``, i.e. a state machine whose every waypoint is
an anchor position (``get_hinge_pos`` / ``get_handle_bar_pos`` / ``get_board_pos`` /
``get_board_edge``) plus tuned offsets, fed through ``solve_ik``. Those anchor functions already
absorb per-door geometry, which is why one such file generalizes across a whole asset set. The
workbench exists to find the offsets by hand on two or three doors, then hand them to code that
computes the rest.

The loop:

1. Pose the scene. Sliders drive the door joints and every robot joint; two draggable gizmos set
   the ``panda_hand`` and base targets.
2. Read the offsets. For whichever anchor is selected, the GUI shows ``gizmo - anchor(q_door)``
   live. Those three numbers are literally what ``pregrasp_palm_x_offset`` and friends are.
3. Ask the LLM. The prompt box ships the whole scene state -- anchors, offsets, joint vectors,
   key-body FK, last run's IK diagnostics -- through ``llm_bridge`` and hot-reloads any planner
   source that comes back.
4. Run and watch. The draft planner executes against the live door and plays back through
   ``collocate_and_playback``, with per-keyframe IK success reported inline.

FRAMES AND CONVENTIONS

Positions are metres in the world frame; the door stands at ``DOOR_INITIAL_POS`` and the robot
starts ~1 m out at +x, facing -x. Door angles are radians, ``q_door = [panel, lever]``. Robot
configs are ``FULL_JOINT_NAMES`` order: base(3) + panda(7) + gripper(1) + x5 camera(6) = 17.

The palm gizmo drives ``panda_hand``, the wrist mount, because that is the frame ``solve_ik``
targets. ``palm_center`` -- where the fingers actually close -- sits 103.4 mm further along the
approach axis and is drawn as a separate non-draggable frame so both are visible.

QUATERNION CONVENTION -- read before touching the rotation code.

``api.solve_ik`` hands ``world_to_base_frame``'s output (IsaacLab **wxyz**) to
``PinocchioIKSolver.compute_ik``, which does ``R.from_quat(...)`` -- scipy **xyzw**. So the
orientation a planner actually commands is its ``get_rotation_quat`` tuple read one slot over.
Every offset in the existing planners is tuned against that behaviour, so it is reproduced here,
not corrected: ``_palm_quat_for_solve_ik`` converts a gizmo's true world orientation into the quat
that makes ``solve_ik`` reach it, and ``_world_quat_from_solve_ik`` inverts that so a planner's
existing ``get_rotation_quat(roll, pitch, yaw)`` can be shown on the gizmo as what it really means.
Verified end-to-end against ``solve_ik`` + FK: round-trip quaternion dot = 0.99999.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import threading
import time
import traceback

import numpy as np
import torch
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
)

from DoorOpening.constants.env_constants import (
    DOOR_INITIAL_POS,
    DOOR_INITIAL_ROT,
    ROBOT_INITIAL_POS,
    ROBOT_INITIAL_ROT,
)
from DoorOpening.constants.robot_constants import (
    DM_JOINT_NAMES,
    FULL_JOINT_NAMES,
    ROBOT_KEY_BODY_NAMES,
)
from DoorOpening.utils.pose_utils import base_to_world_frame
from DoorOpening.utils.state_machine import llm_bridge
from DoorOpening.utils.state_machine import capture_schema
from DoorOpening.utils.state_machine.capture_schema import (
    CaptureSession,
    ChannelSample,
    Keyframe,
    PHASE_IDS,
    VariantClass,
    ARM_CHANNELS,
    CHANNEL_GROUPS,
    default_channel_spec,
    derive_geometry_bucket,
    save_session,
    with_joint_channels,
)
from DoorOpening.utils.state_machine.api import (
    get_board_edge,
    get_board_pos,
    get_handle_bar_pos,
    get_hinge_pos,
    solve_ik,
)
from DoorOpening.utils.state_machine.pin import PinocchioIKSolver

# Anchor functions a planner waypoint can be written against. Every one takes
# (door_urdf_path, door_initial_pose, joint_angles) and returns a (1, 3) world position.
# "world" is not a landmark -- it is the escape hatch for the absolute waypoints the existing
# planners also use (e.g. traverse_far_x = -1.0), where the offset IS the world coordinate.
ANCHOR_FNS = {
    "handle_bar": get_handle_bar_pos,
    "hinge": get_hinge_pos,
    "board_center": get_board_pos,
    "board_edge": get_board_edge,
    "world": None,
}

# The anchor dropdown's neutral position. It has to exist: the dropdown re-points every
# anchor-relative position channel at once, so a concrete initial value would silently stomp the
# per-phase defaults and any --anchor given on the command line before the human touched anything.
ANCHOR_DROPDOWN_DEFAULT = "(phase default)"

# panda_hand -> palm_center, from glorbot.urdf. The fingers close this far past the IK target.
PALM_CENTER_OFFSET_Z = 0.1034

DEFAULT_ROBOT_URDF = "source/DoorOpening/assets/glorbot/glorbot.urdf"


def _wxyz_perm_forward(quat: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) -> (x, y, z, w) held in wxyz slots. See the module docstring."""
    return quat[..., [1, 2, 3, 0]]


def _wxyz_perm_inverse(quat: torch.Tensor) -> torch.Tensor:
    """Undo ``_wxyz_perm_forward``."""
    return quat[..., [3, 0, 1, 2]]


def _palm_quat_for_solve_ik(world_quat: torch.Tensor, robot_base_quat: torch.Tensor) -> torch.Tensor:
    """World orientation the human sees -> the quat to store in a planner's ``palm_pose``."""
    desired_base = quat_mul(quat_conjugate(robot_base_quat), world_quat)
    return quat_mul(robot_base_quat, _wxyz_perm_forward(desired_base))


def _world_quat_from_solve_ik(stored_quat: torch.Tensor, robot_base_quat: torch.Tensor) -> torch.Tensor:
    """A planner's stored ``palm_pose`` quat -> the world orientation it actually commands."""
    stored_base = quat_mul(quat_conjugate(robot_base_quat), stored_quat)
    return quat_mul(robot_base_quat, _wxyz_perm_inverse(stored_base))


def get_rotation_quat(roll, pitch, yaw, device="cpu") -> torch.Tensor:
    """Same helper the planner files define, so captured rpy pastes straight into generated code."""
    return quat_from_euler_xyz(
        roll=torch.tensor([[roll]], device=device),
        pitch=torch.tensor([[pitch]], device=device),
        yaw=torch.tensor([[yaw]], device=device),
    ).squeeze(0)


def _wrap_angle(value: float) -> float:
    """Fold an angle DIFFERENCE into (-pi, pi], so a near-zero delta never reads as +/-2pi."""
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _rpy_from_quat(quat: torch.Tensor) -> tuple[float, float, float]:
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    return float(roll.item()), float(pitch.item()), float(yaw.item())


def build_viser_scene(server, robot_urdf, door_urdf, robot_initial_pose, door_initial_pose):
    """Load robot + door under world-posed root frames.

    Mirrors ``compute_waypoint.play_trajectories_in_viser`` so the workbench and the playback
    script draw the same scene.
    """
    from viser.extras import ViserUrdf

    robot_pos = robot_initial_pose[:, :3].squeeze(0).numpy()
    robot_quat = robot_initial_pose[:, 3:].squeeze(0).numpy()
    door_pos = door_initial_pose[:, :3].squeeze(0).numpy()
    door_quat = door_initial_pose[:, 3:].squeeze(0).numpy()

    # Ground grid at z=0. Without it there is no visual floor, and a door whose origin sits at its
    # own vertical centre (which is every asset set here) looks like it is hanging in space.
    server.scene.add_grid("/ground", width=8.0, height=8.0, cell_size=0.25, position=(0.0, 0.0, 0.0))
    server.scene.add_frame("/robot_root", position=robot_pos, wxyz=robot_quat, show_axes=False)
    server.scene.add_frame("/door_root", position=door_pos, wxyz=door_quat, show_axes=False)

    viser_robot = ViserUrdf(server, urdf_or_path=robot_urdf, root_node_name="/robot_root", load_meshes=True)
    viser_door = ViserUrdf(server, urdf_or_path=door_urdf, root_node_name="/door_root", load_meshes=True)
    return viser_robot, viser_door


class PlannerWorkbench:
    """Viser session: pose the scene, read offsets, talk to the LLM, run the draft planner."""

    def __init__(
        self,
        robot_urdf_path,
        door_urdf_path,
        robot_initial_pose,   # (1, 7) world
        door_initial_pose,    # (1, 7) world
        robot_initial_q,      # (ndof,)
        door_initial_q,       # (2,) [panel, lever]
        *,
        handle_side="right",
        opening_direction="pull",
        session_dir,
        planner_path,
        joint_phases=None,
        port=None,
        device="cpu",
    ):
        self.robot_urdf_path = robot_urdf_path
        self.door_urdf_path = door_urdf_path
        self.robot_initial_pose = robot_initial_pose
        self.door_initial_pose = door_initial_pose
        self.robot_initial_q = robot_initial_q
        self.door_initial_q = door_initial_q
        self.handle_side = handle_side
        self.opening_direction = opening_direction
        self.session_dir = session_dir
        self.planner_path = planner_path
        # {"arm": {phase, ...}, "door": {phase, ...}} -- phases that additionally record a joint
        # posture. Separate sets: a phase can pin the door's angles without pinning the arm's.
        self.joint_phases = {
            "arm": set((joint_phases or {}).get("arm", ())),
            "door": set((joint_phases or {}).get("door", ())),
        }
        self.device = device

        os.makedirs(session_dir, exist_ok=True)

        self.q_robot = robot_initial_q.clone()
        self.q_door = door_initial_q.clone()

        self.robot_base_quat = robot_initial_pose[:, 3:].clone()
        self._door_z_base = float(door_initial_pose[0, 2])

        # get_hinge_pos and friends resample the door point cloud (~0.1 s each). Every slider drag
        # would otherwise re-run four of them per frame, which makes the GUI feel dead.
        self._anchor_cache: dict[tuple[str, float, float], torch.Tensor] = {}

        self._ik_solver = PinocchioIKSolver(
            urdf_path=robot_urdf_path,
            ee_link_name="panda_hand",
            controlled_joints=DM_JOINT_NAMES,
        )

        self.last_run: dict | None = None
        # Captured keyframes, in capture order. This is the session; everything else in the GUI
        # exists to produce entries here.
        self.keyframes: list[Keyframe] = []
        self._next_keyframe_id = 0
        self._sweep_active = False
        # Where the last chat turn ended, so the next one shows only what you captured since.
        self._last_turn_keyframe_id = 0
        self.last_check_report: str = ""
        # The palm/base world positions of the last captured keyframe, which is what a prev_palm
        # or prev_base channel measures from.
        self._prev_palm_world = None
        self._prev_base_world = None
        self._active_ask_turn: int | None = None
        self._stop_polling = threading.Event()
        self._playback_stop = threading.Event()
        self._playback_thread: threading.Thread | None = None
        self._pending_turn: int | None = None

        self._start_server(port)

    # ------------------------------------------------------------------ scene

    def _start_server(self, port):
        import viser
        from yourdfpy import URDF

        self.server = viser.ViserServer(**({"port": port} if port is not None else {}))
        self._robot_urdf = URDF.load(self.robot_urdf_path)
        self.viser_robot, self.viser_door = build_viser_scene(
            self.server,
            self._robot_urdf,
            URDF.load(self.door_urdf_path),
            self.robot_initial_pose,
            self.door_initial_pose,
        )

        palm_pos, palm_quat = self.palm_world_pose()
        # TRANSLATION AND ROTATION ARE TWO CONTROLS, and the translation one is held at IDENTITY
        # orientation on purpose. A transform gizmo draws its arrows along its OWN axes, so a
        # single control carrying the grasp orientation gives three arrows that all point at odd
        # angles -- none of them world "up" -- leaving the plane handles as the only usable grip
        # and the whole thing feeling like a 2D control. Identity here means the arrows always mean
        # world X / Y / Z, so dragging up is dragging up.
        self.palm_gizmo = self.server.scene.add_transform_controls(
            "/palm_target", scale=0.4, line_width=3.5, depth_test=False,
            disable_rotations=True, position=palm_pos,
        )
        # Orientation lives on its own ring-only control, co-located with the translation one.
        self.palm_rot_gizmo = self.server.scene.add_transform_controls(
            "/palm_rotation", scale=0.28, line_width=3.0, depth_test=False,
            disable_axes=True, disable_sliders=True,
            position=palm_pos, wxyz=palm_quat,
        )
        base_pos, base_quat = self.base_world_pose()
        # The base genuinely has three DOF (x, y, yaw): compute_base_joint discards z and
        # roll/pitch. Dragging a vertical arrow that the planner throws away is worse than not
        # offering one, so the z axis is switched off rather than silently ignored.
        self.base_gizmo = self.server.scene.add_transform_controls(
            "/base_target", scale=0.45, line_width=3.5, depth_test=False,
            active_axes=(True, True, False), disable_rotations=True, position=base_pos,
        )
        self.base_rot_gizmo = self.server.scene.add_transform_controls(
            "/base_rotation", scale=0.32, line_width=3.0, depth_test=False,
            disable_axes=True, disable_sliders=True,
            position=base_pos, wxyz=base_quat,
        )

        self._build_gui()
        threading.Thread(target=self._poll_asks, daemon=True).start()
        self._refresh_scene()
        self._refresh_readouts()
        self._refresh_capture_spec()

    # ------------------------------------------------------------- kinematics

    def _anchor_pos(self, name, q_door) -> torch.Tensor:
        """(1, 3) world position of the named anchor at the given door configuration."""
        if name == "world":
            return torch.zeros(1, 3)
        key = (name, round(float(q_door[0]), 4), round(float(q_door[1]), 4))
        if key not in self._anchor_cache:
            self._anchor_cache[key] = ANCHOR_FNS[name](
                self.door_urdf_path, self.door_initial_pose, q_door.unsqueeze(0)
            )
        return self._anchor_cache[key]

    def _fk_world(self, q_robot, link_name):
        """World (pos, wxyz) of one link at the given robot config."""
        trans, rot = self._ik_solver.get_frames_pose_batch(
            config=q_robot[:10].detach().cpu().numpy(),
            node_a_list=[link_name],
            node_b="base_link",
        )
        from isaaclab.utils.math import quat_from_matrix

        pos_b = torch.from_numpy(np.asarray(trans[0], dtype=np.float32)).unsqueeze(0)
        quat_b = quat_from_matrix(torch.from_numpy(np.asarray(rot[0], dtype=np.float32)).unsqueeze(0))
        pos_w, quat_w = base_to_world_frame(
            self.robot_initial_pose[:, :3], self.robot_base_quat, pos_b, quat_b
        )
        return pos_w, quat_w

    def palm_world_pose(self):
        pos, quat = self._fk_world(self.q_robot, "panda_hand")
        return pos.squeeze(0).numpy(), quat.squeeze(0).numpy()

    def base_world_pose(self):
        pos, quat = self._fk_world(self.q_robot, "tidybot2_base_link")
        return pos.squeeze(0).numpy(), quat.squeeze(0).numpy()

    def key_body_poses(self) -> dict:
        out = {}
        for name in ROBOT_KEY_BODY_NAMES:
            pos, quat = self._fk_world(self.q_robot, name)
            out[name] = {
                "pos_w": [round(v, 5) for v in pos.squeeze(0).tolist()],
                "quat_w_wxyz": [round(v, 5) for v in quat.squeeze(0).tolist()],
            }
        return out

    # ------------------------------------------------------------------- state

    def scene_state(self) -> dict:
        """Everything the responder needs to write a planner waypoint against this pose."""
        palm_pos = np.asarray(self.palm_gizmo.position, dtype=np.float32)
        palm_quat = np.asarray(self.palm_rot_gizmo.wxyz, dtype=np.float32)
        base_pos = np.asarray(self.base_gizmo.position, dtype=np.float32)
        base_quat = np.asarray(self.base_rot_gizmo.wxyz, dtype=np.float32)

        anchors, palm_offsets, base_offsets = {}, {}, {}
        for name in ANCHOR_FNS:
            anchor = self._anchor_pos(name, self.q_door).squeeze(0).numpy()
            anchors[name] = [round(float(v), 5) for v in anchor]
            palm_offsets[name] = [round(float(v), 5) for v in (palm_pos - anchor)]
            base_offsets[name] = [round(float(v), 5) for v in (base_pos - anchor)]

        stored_palm_quat = _palm_quat_for_solve_ik(
            torch.from_numpy(palm_quat).unsqueeze(0), self.robot_base_quat
        )
        return {
            "robot_urdf_path": self.robot_urdf_path,
            "door_urdf_path": self.door_urdf_path,
            "handle_side": self.handle_side,
            "opening_direction": self.opening_direction,
            "robot_initial_pose_world": self.robot_initial_pose.squeeze(0).tolist(),
            "door_initial_pose_world": self.door_initial_pose.squeeze(0).tolist(),
            "door_z_lift_m": round(float(self.gui_door_z.value), 5),
            "q_door": {"panel_rad": float(self.q_door[0]), "lever_rad": float(self.q_door[1])},
            "q_robot": {
                name: round(float(v), 5) for name, v in zip(FULL_JOINT_NAMES, self.q_robot.tolist())
            },
            "palm_target_world": {
                "pos_m": [round(float(v), 5) for v in palm_pos],
                "quat_wxyz_true": [round(float(v), 5) for v in palm_quat],
                "quat_wxyz_for_solve_ik": [round(float(v), 5) for v in stored_palm_quat.squeeze(0).tolist()],
                "get_rotation_quat_rpy": [
                    round(v, 5) for v in _rpy_from_quat(stored_palm_quat)
                ],
            },
            "base_target_world": {
                "pos_m": [round(float(v), 5) for v in base_pos],
                "quat_wxyz": [round(float(v), 5) for v in base_quat],
                "yaw_rad": round(_rpy_from_quat(torch.from_numpy(base_quat).unsqueeze(0))[2], 5),
            },
            "anchor_positions_world": anchors,
            "palm_offset_from_anchor": palm_offsets,
            "base_offset_from_anchor": base_offsets,
            "key_body_poses_world": self.key_body_poses(),
            "notes": self.gui_notes.value,
        }

    # --------------------------------------------------------------------- GUI

    def _joint_limits(self, joint_name):
        joint = self._robot_urdf.joint_map.get(joint_name)
        limit = getattr(joint, "limit", None) if joint is not None else None
        if limit is not None and limit.lower is not None and limit.upper is not None:
            if limit.upper > limit.lower:
                return float(limit.lower), float(limit.upper)
        return -math.pi, math.pi

    def _build_gui(self):
        gui = self.server.gui

        gui.add_markdown(
            f"**{os.path.basename(os.path.dirname(self.door_urdf_path))}** &nbsp; "
            f"`{self.handle_side}` / `{self.opening_direction}`"
        )

        with gui.add_folder("Agent", expand_by_default=True):
            self.gui_ask = gui.add_markdown("_no question pending -- the agent is working._")
            self.gui_ask_choice = gui.add_dropdown("answer", ["--"], initial_value="--")
            self.gui_ask_note = gui.add_text(
                "note", "", multiline=True,
                hint="Describe what you see, in words. You never need to write code.",
            )
            self.gui_ask_send = gui.add_button("Answer the agent")
            self.gui_ask_replay = gui.add_button("Replay the draft")
            self.gui_ask_status = gui.add_markdown("")
            self.gui_ask_send.on_click(lambda _: self._answer_ask())
            self.gui_ask_replay.on_click(lambda _: self._replay_for_review())

        with gui.add_folder("How to capture", expand_by_default=False):
            gui.add_markdown(
                "`phase_id` -> anchors -> drag gizmos -> **Solve IK** -> **Add keyframe**\n\n"
                "Palm gizmo is the WRIST; fingers close ~10 cm past it.\n\n"
                "**will record** shows what a capture stores, live.\n\n"
                "Swept phases: **Begin sweep**, then **Capture + advance theta**.\n\n"
                "**Save capture session** at the end."
            )

        with gui.add_folder("Door state"):
            self.gui_panel = gui.add_slider("panel (rad)", -0.2, 1.6, 0.01, float(self.q_door[0]))
            self.gui_lever = gui.add_slider("lever (rad)", -0.2, 1.1, 0.01, float(self.q_door[1]))
            self.gui_panel.on_update(lambda _: self._on_door_change())
            self.gui_lever.on_update(lambda _: self._on_door_change())
            # Every door asset in this repo has its base frame at the panel's VERTICAL CENTRE, so
            # DOOR_INITIAL_POS z = 0.03 renders the door half-buried and puts every handle anchor
            # at negative z. This lifts door_initial_pose for the session -- anchors move with it,
            # so offsets stay consistent -- and is recorded in the export.
            self.gui_door_z = gui.add_number(
                "door z lift (m)", 0.0, min=-0.5, max=1.5, step=0.005,
                hint="0 = DOOR_INITIAL_POS as-is. ~0.9 puts this door's sill on the floor.",
            )
            self.gui_door_z.on_update(lambda _: self._on_door_z_change())

        with gui.add_folder("Targets"):
            self.gui_anchor = gui.add_dropdown("anchor", list(ANCHOR_FNS.keys()), initial_value="handle_bar")
            self.gui_anchor.on_update(lambda _: self._refresh_readouts())
            self.gui_solve = gui.add_button("Solve IK from gizmos")
            self.gui_snap = gui.add_button("Snap gizmos to robot")
            self.gui_offsets = gui.add_markdown("")
            self.gui_ik_status = gui.add_markdown("")
            self.gui_solve.on_click(lambda _: self._solve_from_gizmos())
            self.gui_snap.on_click(lambda _: self._snap_gizmos())
            self.palm_gizmo.on_update(lambda _: self._on_translate("palm"))
            self.base_gizmo.on_update(lambda _: self._on_translate("base"))
            self.palm_rot_gizmo.on_update(lambda _: self._refresh_readouts())
            self.base_rot_gizmo.on_update(lambda _: self._refresh_readouts())

        with gui.add_folder("Type exact offsets"):
            # Dragging is for coarse discovery; the product of this tool is NUMBERS
            # (pregrasp_palm_x_offset = 0.4), so past the first look it is faster to type one than
            # to chase a 0.005 m nudge with the mouse. These boxes always MIRROR the gizmos, so the
            # workflow is: drag roughly -> read -> round the number -> Apply.
            #
            # No on_update handlers here on purpose: _refresh_readouts writes into these boxes, and
            # a handler would turn that into a feedback loop between gizmo and box. They are read
            # only when you press Apply.
            self.gui_entry_palm_x = gui.add_number("palm dx", 0.0, step=0.005)
            self.gui_entry_palm_y = gui.add_number("palm dy", 0.0, step=0.005)
            self.gui_entry_palm_z = gui.add_number("palm dz", 0.0, step=0.005)
            self.gui_entry_roll = gui.add_number("palm roll", 0.0, step=0.01)
            self.gui_entry_pitch = gui.add_number("palm pitch", 0.0, step=0.01)
            self.gui_entry_yaw = gui.add_number("palm yaw", 0.0, step=0.01)
            self.gui_entry_base_x = gui.add_number("base dx", 0.0, step=0.005)
            self.gui_entry_base_y = gui.add_number("base dy", 0.0, step=0.005)
            self.gui_entry_base_yaw = gui.add_number("base yaw (world)", 0.0, step=0.01)
            self.gui_apply_offsets = gui.add_button("Apply to gizmos")
            self.gui_apply_solve = gui.add_button("Apply + solve IK")
            self.gui_entry_hint = gui.add_markdown(
                "_offsets are against the selected anchor; rotations are the "
                "`get_rotation_quat` triple, already in solve_ik's convention_"
            )
            self.gui_apply_offsets.on_click(lambda _: self._apply_typed_offsets())
            self.gui_apply_solve.on_click(
                lambda _: (self._apply_typed_offsets(), self._solve_from_gizmos())
            )

        with gui.add_folder("Robot joints", expand_by_default=False):
            self.gui_joints = []
            for idx, name in enumerate(FULL_JOINT_NAMES):
                lo, hi = self._joint_limits(name)
                slider = gui.add_slider(name, lo, hi, (hi - lo) / 400.0, float(self.q_robot[idx]))
                slider.on_update(lambda _, i=idx: self._on_joint_change(i))
                self.gui_joints.append(slider)
            self.gui_reset_joints = gui.add_button("Reset to initial q")
            self.gui_reset_joints.on_click(lambda _: self._reset_joints())

        with gui.add_folder("Key bodies", expand_by_default=False):
            self.gui_key_bodies = gui.add_markdown("")

        with gui.add_folder("Planner draft"):
            self.gui_planner_path = gui.add_text("path", self.planner_path, disabled=True)
            self.gui_run = gui.add_button("Run planner")
            self.gui_playback = gui.add_button("Playback")
            self.gui_stop = gui.add_button("Stop playback")
            self.gui_run_status = gui.add_markdown("_not run yet_")
            self.gui_run.on_click(lambda _: self._run_planner())
            self.gui_playback.on_click(lambda _: self._start_playback())
            self.gui_stop.on_click(lambda _: self._playback_stop.set())

        with gui.add_folder("Capture"):
            self.gui_phase = gui.add_dropdown("phase_id", list(PHASE_IDS), initial_value="pregrasp")
            self.gui_phase.on_update(lambda _: self._refresh_capture_spec())
            self.gui_block = gui.add_text(
                "continuity_block",
                f"{self.handle_side}_{self.opening_direction}_main",
                hint="Keyframes sharing one solve_ik reference_joint_pos. The existing left-pull "
                     "planner uses ONE block for the whole function; per-step blocks are the "
                     "exception, not the rule.",
            )
            self.gui_mark_keyframe = gui.add_checkbox("mark_keyframe", True)
            self.gui_num_attempts = gui.add_number("num_attempts", 8, min=1, max=16, step=1)
            self.gui_gripper = gui.add_number("gripper width", 0.04, min=0.0, max=0.08, step=0.001)
            # Two dropdowns, because the planner routinely anchors the base and the end effector
            # to different landmarks in the same waypoint (pull:216 vs pull:236).
            _landmarks = [name for name in ANCHOR_FNS if name != "world"]
            self.gui_palm_anchor = gui.add_dropdown(
                "palm anchor (end effector)",
                [ANCHOR_DROPDOWN_DEFAULT] + _landmarks + ["prev_palm", "robot_joints"],
                initial_value=ANCHOR_DROPDOWN_DEFAULT,
                hint="What DEFINES the end effector here. A door landmark or 'prev_palm' records "
                     "a Cartesian offset from it. 'robot_joints' instead pins the seven panda "
                     "angles as the waypoint itself, for a pose that has to be one exact "
                     f"configuration. '{ANCHOR_DROPDOWN_DEFAULT}' keeps the phase default.",
            )
            self.gui_base_anchor = gui.add_dropdown(
                "base anchor",
                [ANCHOR_DROPDOWN_DEFAULT] + _landmarks + ["prev_base"],
                initial_value=ANCHOR_DROPDOWN_DEFAULT,
                hint="Where the BASE offsets are measured from. Independent of the palm.",
            )
            self.gui_palm_anchor.on_update(lambda _: self._refresh_capture_spec())
            self.gui_base_anchor.on_update(lambda _: self._refresh_capture_spec())
            self.gui_channel_overrides = gui.add_text(
                "channel overrides (JSON)", "",
                multiline=True,
                hint='Per-channel escape hatch, e.g. {"base_y": {"mode": "anchor_relative"}}. '
                     'Empty means use the phase defaults shown below.',
            )
            self.gui_channel_overrides.on_update(lambda _: self._refresh_capture_spec())
            self.gui_record_arm = gui.add_checkbox(
                "record arm joints", False,
                hint="Also record the seven panda angles, pinning this waypoint to the posture "
                     "on screen rather than to wherever IK lands.",
            )
            self.gui_record_door = gui.add_checkbox(
                "record door joints", False,
                hint="Also record the door's panel and lever angles as commanded values.",
            )
            self.gui_record_arm.on_update(lambda _: self._refresh_capture_spec())
            self.gui_record_door.on_update(lambda _: self._refresh_capture_spec())
            self.gui_capture_spec = gui.add_markdown("")
            self.gui_add_keyframe = gui.add_button("Add keyframe")
            self.gui_drop_keyframe = gui.add_button("Delete last keyframe")
            self.gui_keyframe_list = gui.add_markdown("_no keyframes captured_")
            self.gui_add_keyframe.on_click(lambda _: self._capture_keyframe())
            self.gui_drop_keyframe.on_click(lambda _: self._drop_keyframe())

        with gui.add_folder("Theta sweep"):
            # Swept phases need many samples across the door's travel, not one pose. This steps the
            # panel slider for you so each sample lands on a known theta and the fits have a clean
            # x-axis to work against.
            self.gui_sweep_start = gui.add_number("theta start", 0.30, min=-0.2, max=1.6, step=0.05)
            self.gui_sweep_stop = gui.add_number("theta stop", 1.25, min=-0.2, max=1.6, step=0.05)
            self.gui_sweep_step = gui.add_number("theta step", 0.10, min=0.01, max=0.5, step=0.01)
            self.gui_sweep_begin = gui.add_button("Begin sweep (go to start)")
            self.gui_sweep_capture = gui.add_button("Capture + advance theta")
            self.gui_sweep_status = gui.add_markdown("_idle_")
            self.gui_sweep_begin.on_click(lambda _: self._sweep_begin())
            self.gui_sweep_capture.on_click(lambda _: self._sweep_capture())

        with gui.add_folder("LLM"):
            self.gui_prompt = gui.add_text("prompt", "", multiline=True)
            self.gui_send = gui.add_button("Send to LLM")
            self.gui_check = gui.add_button("Check for reply")
            self.gui_llm_status = gui.add_markdown("_idle_")
            self.gui_transcript = gui.add_markdown("")
            self.gui_send.on_click(lambda _: self._send_to_llm())
            self.gui_check.on_click(lambda _: self._poll_llm())

        with gui.add_folder("Session"):
            self.gui_notes = gui.add_text(
                "notes", "", multiline=True,
                hint="Attached VERBATIM to the next keyframe you capture, and reproduced verbatim "
                     "in any generated planner. Say why, not what.",
            )
            self.gui_export = gui.add_button("Export session (scene snapshot)")
            self.gui_save_capture = gui.add_button("Save capture session")
            self.gui_export_status = gui.add_markdown("")
            self.gui_export.on_click(lambda _: self._export())
            self.gui_save_capture.on_click(lambda _: self._save_capture())

        with gui.add_folder("Chat loop"):
            # capture a bit -> write a turn -> paste it -> drop the reply into the draft ->
            # check the trajectory -> write the next turn with what you saw.
            self.gui_feedback = gui.add_text(
                "feedback for the next turn", "", multiline=True,
                hint="What you saw when you ran it and what you want changed. Goes at the TOP of "
                     "the turn, above the planner source.",
            )
            self.gui_check_traj = gui.add_button("Check trajectory (run + IK report)")
            self.gui_write_turn = gui.add_button("Write chat turn")
            self.gui_turn_status = gui.add_markdown("_no turn written yet_")
            self.gui_check_traj.on_click(lambda _: self._check_trajectory())
            self.gui_write_turn.on_click(lambda _: self._write_turn())

    # ----------------------------------------------------------------- updates

    def _refresh_scene(self):
        self.viser_robot.update_cfg(self.q_robot.detach().cpu().numpy())
        self.viser_door.update_cfg(self.q_door.detach().cpu().numpy())
        palm_pos, palm_quat = self.palm_world_pose()
        # palm_center is where the fingers actually close, PALM_CENTER_OFFSET_Z past the IK target.
        from isaaclab.utils.math import quat_apply

        offset_w = quat_apply(
            torch.from_numpy(palm_quat).unsqueeze(0),
            torch.tensor([[0.0, 0.0, PALM_CENTER_OFFSET_Z]]),
        ).squeeze(0).numpy()
        self.server.scene.add_frame(
            "/palm_center_actual", position=palm_pos + offset_w, wxyz=palm_quat, axes_length=0.06
        )

    def _on_translate(self, which):
        """Drag the arrows and the rotation ring rides along, so the pair reads as one target."""
        gizmo, ring = ((self.palm_gizmo, self.palm_rot_gizmo) if which == "palm"
                       else (self.base_gizmo, self.base_rot_gizmo))
        ring.position = gizmo.position
        self._refresh_readouts()

    # ------------------------------------------------------------- agent asks

    def _poll_asks(self):
        """Watch the bridge for a question the agent has posted. Runs on its own thread."""
        while not self._stop_polling.is_set():
            try:
                pending = llm_bridge.pending_asks(self.session_dir)
                turn = pending[0] if pending else None
                if turn != self._active_ask_turn:
                    self._active_ask_turn = turn
                    if turn is not None:
                        self._present_ask(llm_bridge.read_ask(self.session_dir, turn))
                    else:
                        self.gui_ask.content = "_no question pending -- the agent is working._"
            except (OSError, ValueError) as exc:
                self.gui_ask_status.content = f"**bridge error:** `{exc}`"
            self._stop_polling.wait(2.0)

    def _present_ask(self, ask):
        """Apply the setup the question is about, then render it."""
        if not ask:
            return
        setup = ask.get("setup") or {}
        # Applying the setup is the point: the human is never asked to reproduce a configuration
        # from prose, so a mis-set dropdown cannot silently answer a different question.
        if setup.get("phase_id") in PHASE_IDS:
            self.gui_phase.value = setup["phase_id"]
        if setup.get("palm_anchor"):
            self.gui_palm_anchor.value = setup["palm_anchor"]
        if setup.get("base_anchor"):
            self.gui_base_anchor.value = setup["base_anchor"]
        if setup.get("gripper") is not None:
            self.gui_gripper.value = float(setup["gripper"])
        if setup.get("theta") is not None:
            self.gui_panel.value = float(setup["theta"])
            self._on_door_change()

        options = list(ask.get("options") or [])
        if ask["kind"] == "confirm":
            options = options or ["yes", "no"]
        elif ask["kind"] == "review":
            options = options or ["looks right", "needs work"]
        elif ask["kind"] == "capture":
            options = options or ["captured", "cannot reach this"]
        self.gui_ask_choice.options = options + ["--"]
        self.gui_ask_choice.value = options[0] if options else "--"

        how = {
            "capture": "Pose the gizmos, press **Add keyframe** (repeat 3x), then **Answer the agent**.",
            "choose": "Pick an answer above, then **Answer the agent**.",
            "review": "Watch the playback, then pick an answer and describe anything wrong.",
            "confirm": "Pick yes or no, then **Answer the agent**.",
        }[ask["kind"]]
        self.gui_ask.content = (
            f"### The agent asks (turn {ask['turn']})\n\n{ask['question']}\n\n_{how}_"
        )
        self._refresh_capture_spec()
        if ask.get("play_first"):
            self._replay_for_review()

    def _replay_for_review(self):
        """Run the draft and play it, so a review question is answered against something seen."""
        self._run_planner()
        if self.last_run and self.last_run.get("ok"):
            self._start_playback()
        else:
            self.gui_ask_status.content = (
                "**the draft did not run** -- answer anyway and paste what the Planner draft "
                "panel says; the agent needs the failure as much as a good run."
            )

    def _answer_ask(self):
        turn = self._active_ask_turn
        if turn is None:
            self.gui_ask_status.content = "**nothing to answer** -- no question is pending."
            return
        capture_path, ids = "", []
        # A capture question is answered with DATA, not prose: save whatever was captured since the
        # question appeared and hand the agent the file, so it reads numbers rather than a summary.
        if self.keyframes and self._active_ask_kind() == "capture":
            capture_path = os.path.join(self.session_dir, f"capture_{int(time.time())}.json")
            save_session(self._capture_session(), capture_path)
            ids = [k.id for k in self.keyframes]
        llm_bridge.write_answer(
            self.session_dir, turn,
            choice=("" if self.gui_ask_choice.value == "--" else self.gui_ask_choice.value),
            note=self.gui_ask_note.value.strip(),
            capture_path=capture_path,
            keyframe_ids=ids,
        )
        self.gui_ask_note.value = ""
        self.gui_ask_status.content = (
            f"answered turn {turn}"
            + (f" with `{os.path.basename(capture_path)}` ({len(ids)} keyframes)" if capture_path else "")
        )
        self.gui_ask.content = "_answer sent -- waiting for the agent._"
        self._active_ask_turn = None

    def _active_ask_kind(self) -> str:
        ask = llm_bridge.read_ask(self.session_dir, self._active_ask_turn or 0)
        return (ask or {}).get("kind", "")

    def _refresh_readouts(self):
        anchor_name = self.gui_anchor.value
        anchor = self._anchor_pos(anchor_name, self.q_door).squeeze(0).numpy()
        palm = np.asarray(self.palm_gizmo.position, dtype=np.float32)
        base = np.asarray(self.base_gizmo.position, dtype=np.float32)
        palm_d, base_d = palm - anchor, base - anchor
        stored = _palm_quat_for_solve_ik(
            torch.from_numpy(np.asarray(self.palm_rot_gizmo.wxyz, dtype=np.float32)).unsqueeze(0),
            self.robot_base_quat,
        )
        roll, pitch, yaw = _rpy_from_quat(stored)
        base_yaw = _rpy_from_quat(
            torch.from_numpy(np.asarray(self.base_rot_gizmo.wxyz, dtype=np.float32)).unsqueeze(0)
        )[2]
        # Mirror the live values into the entry boxes so typing starts from where you dragged.
        for widget, value in (
            (getattr(self, "gui_entry_palm_x", None), palm_d[0]),
            (getattr(self, "gui_entry_palm_y", None), palm_d[1]),
            (getattr(self, "gui_entry_palm_z", None), palm_d[2]),
            (getattr(self, "gui_entry_roll", None), roll),
            (getattr(self, "gui_entry_pitch", None), pitch),
            (getattr(self, "gui_entry_yaw", None), yaw),
            (getattr(self, "gui_entry_base_x", None), base_d[0]),
            (getattr(self, "gui_entry_base_y", None), base_d[1]),
            (getattr(self, "gui_entry_base_yaw", None), base_yaw),
        ):
            if widget is not None:
                widget.value = round(float(value), 4)

        self.gui_offsets.content = (
            f"anchor `{anchor_name}` @ ({anchor[0]:.3f}, {anchor[1]:.3f}, {anchor[2]:.3f})\n\n"
            f"**palm offset** x `{palm_d[0]:+.3f}` y `{palm_d[1]:+.3f}` z `{palm_d[2]:+.3f}`\n\n"
            f"**palm rot** `get_rotation_quat({roll:.4f}, {pitch:.4f}, {yaw:.4f}, device)`\n\n"
            f"**base offset** x `{base_d[0]:+.3f}` y `{base_d[1]:+.3f}` z `{base_d[2]:+.3f}`\n\n"
            f"**base yaw** `{base_yaw:.4f}` rad "
            f"_(compute_base_joint uses yaw only -- base roll/pitch/z are discarded)_"
        )
        self.gui_key_bodies.content = "\n\n".join(
            f"`{name}` ({p['pos_w'][0]:+.3f}, {p['pos_w'][1]:+.3f}, {p['pos_w'][2]:+.3f})"
            for name, p in self.key_body_poses().items()
        )
        # Anything that moved the scene moved the numbers a capture would store, so the preview is
        # refreshed from here rather than from each of the eight callers separately.
        if getattr(self, "gui_capture_spec", None) is not None:
            self._refresh_capture_spec()

    def _on_door_change(self):
        self.q_door = torch.tensor([self.gui_panel.value, self.gui_lever.value], dtype=torch.float32)
        self.viser_door.update_cfg(self.q_door.detach().cpu().numpy())
        self._refresh_readouts()

    def _on_door_z_change(self):
        self.door_initial_pose = self.door_initial_pose.clone()
        self.door_initial_pose[:, 2] = self._door_z_base + float(self.gui_door_z.value)
        self._anchor_cache.clear()
        self.server.scene.add_frame(
            "/door_root",
            position=self.door_initial_pose[:, :3].squeeze(0).numpy(),
            wxyz=self.door_initial_pose[:, 3:].squeeze(0).numpy(),
            show_axes=False,
        )
        self._refresh_readouts()

    def _on_joint_change(self, idx):
        self.q_robot[idx] = float(self.gui_joints[idx].value)
        self._refresh_scene()
        self._refresh_readouts()

    def _reset_joints(self):
        self.q_robot = self.robot_initial_q.clone()
        for idx, slider in enumerate(self.gui_joints):
            slider.value = float(self.q_robot[idx])
        self._refresh_scene()
        self._snap_gizmos()

    def _snap_gizmos(self):
        palm_pos, palm_quat = self.palm_world_pose()
        self.palm_gizmo.position = palm_pos
        self.palm_rot_gizmo.position, self.palm_rot_gizmo.wxyz = palm_pos, palm_quat
        base_pos, base_quat = self.base_world_pose()
        self.base_gizmo.position = base_pos
        self.base_rot_gizmo.position, self.base_rot_gizmo.wxyz = base_pos, base_quat
        self._refresh_readouts()

    def _apply_typed_offsets(self):
        """Drive the gizmos FROM the typed numbers -- the inverse of the offsets readout.

        The rotation boxes hold the get_rotation_quat triple, i.e. what a planner writes and what
        solve_ik effectively commands. The gizmo needs the TRUE world orientation, which is a
        different quaternion (see the module docstring on the wxyz/xyzw slot shift), so it is
        converted back through _world_quat_from_solve_ik rather than used directly.
        """
        anchor = self._anchor_pos(self.gui_anchor.value, self.q_door).squeeze(0).numpy()

        self.palm_gizmo.position = np.array([
            anchor[0] + float(self.gui_entry_palm_x.value),
            anchor[1] + float(self.gui_entry_palm_y.value),
            anchor[2] + float(self.gui_entry_palm_z.value),
        ], dtype=np.float32)
        stored = get_rotation_quat(
            float(self.gui_entry_roll.value),
            float(self.gui_entry_pitch.value),
            float(self.gui_entry_yaw.value),
        )
        world = _world_quat_from_solve_ik(stored, self.robot_base_quat)
        self.palm_rot_gizmo.wxyz = world.squeeze(0).numpy()

        # Base z is not a target -- compute_base_joint keeps x, y and yaw only -- so it is left
        # wherever the base already is instead of being driven off the anchor's height.
        base_pos = np.asarray(self.base_gizmo.position, dtype=np.float32).copy()
        base_pos[0] = anchor[0] + float(self.gui_entry_base_x.value)
        base_pos[1] = anchor[1] + float(self.gui_entry_base_y.value)
        self.base_gizmo.position = base_pos
        self.base_rot_gizmo.wxyz = quat_from_euler_xyz(
            roll=torch.tensor([[0.0]]),
            pitch=torch.tensor([[0.0]]),
            yaw=torch.tensor([[float(self.gui_entry_base_yaw.value)]]),
        ).squeeze(0).numpy()

        self._refresh_readouts()

    def _solve_from_gizmos(self):
        """Drive the arm to the dragged gizmos and report honestly whether the IK got there."""
        palm_quat_world = torch.from_numpy(np.asarray(self.palm_rot_gizmo.wxyz, dtype=np.float32)).unsqueeze(0)
        palm_pose = torch.cat(
            [
                torch.from_numpy(np.asarray(self.palm_gizmo.position, dtype=np.float32)).unsqueeze(0),
                _palm_quat_for_solve_ik(palm_quat_world, self.robot_base_quat),
            ],
            dim=-1,
        )
        base_pose = torch.cat(
            [
                torch.from_numpy(np.asarray(self.base_gizmo.position, dtype=np.float32)).unsqueeze(0),
                torch.from_numpy(np.asarray(self.base_rot_gizmo.wxyz, dtype=np.float32)).unsqueeze(0),
            ],
            dim=-1,
        )
        # solve_ik writes the base joints into its q argument in place, so pass a clone: a failed
        # preview must not corrupt the pose the human is working from.
        q, success, debug = solve_ik(
            self.robot_urdf_path,
            self.q_robot[:10].clone(),
            palm_pose=palm_pose,
            base_pose=base_pose,
            robot_initial_pose=self.robot_initial_pose,
            return_debug=True,
        )
        self.q_robot[:10] = q[0]
        for idx in range(10):
            self.gui_joints[idx].value = float(self.q_robot[idx])
        self._refresh_scene()

        achieved_pos, achieved_quat = self.palm_world_pose()
        pos_err = float(np.linalg.norm(achieved_pos - np.asarray(self.palm_gizmo.position)))
        rot_err = float(
            1.0 - abs((torch.from_numpy(achieved_quat).unsqueeze(0) * palm_quat_world).sum().item())
        )
        self.gui_ik_status.content = self._ik_status_text(success, debug, pos_err, rot_err)
        self._refresh_readouts()

    @staticmethod
    def _ik_status_text(success, debug, pos_err, rot_err):
        # debug_info only carries best_error_norm when the solve FAILED; on success it carries a
        # final_error vector instead, so derive a norm from whichever is present.
        err = debug.get("best_error_norm")
        if err is None and debug.get("final_error") is not None:
            err = float(np.linalg.norm(np.asarray(debug["final_error"])))
        badge = "IK CONVERGED" if success else "**IK FAILED -- target unreachable from this base**"
        return (
            f"{badge}\n\n"
            f"solver err `{err if err is None else round(float(err), 5)}` &nbsp; "
            f"pos err `{pos_err:.4f} m` &nbsp; rot err `{rot_err:.5f}`"
        )

    # ------------------------------------------------------------ planner draft

    def _load_planner(self):
        """Import the draft fresh each Run so edits land without restarting the workbench."""
        spec = importlib.util.spec_from_file_location("_workbench_planner_draft", self.planner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        entry = getattr(module, "state_machine_offline_door", None)
        if entry is None:
            raise AttributeError(
                f"{self.planner_path} must define state_machine_offline_door(...) -- see the "
                "starter template written by write_starter_planner()."
            )
        return entry

    def _run_planner(self):
        if not os.path.exists(self.planner_path):
            self.gui_run_status.content = f"**no draft at** `{self.planner_path}`"
            return
        started = time.time()
        try:
            entry = self._load_planner()
            robot_traj, door_traj, key_idx = entry(
                self.robot_urdf_path,
                self.door_urdf_path,
                self.robot_initial_pose,
                self.door_initial_pose,
                self.robot_initial_q,
                self.door_initial_q,
                handle_side=self.handle_side,
                device=self.device,
            )
        except Exception:
            self.last_run = {"ok": False, "traceback": traceback.format_exc()}
            self.gui_run_status.content = f"**planner raised**\n\n```\n{traceback.format_exc()[-1500:]}\n```"
            return

        self._traj = (robot_traj, door_traj, key_idx)
        # A keyframe pair that collapses to the same config is the signature of an unreachable
        # target the IK best-efforted onto its neighbour; collocate_and_playback silently drops
        # those, so surface the count here rather than letting it vanish into the spline.
        stacked = torch.stack([q.detach().cpu() for q in robot_traj])
        dupes = int((torch.linalg.norm(stacked[1:] - stacked[:-1], dim=-1) < 1e-9).sum())
        self.last_run = {
            "ok": True,
            "waypoints": len(robot_traj),
            "keyframes": len(key_idx),
            "key_idx_in_key_indices": list(map(int, key_idx)),
            "duplicate_adjacent_waypoints": dupes,
            "seconds": round(time.time() - started, 2),
        }
        self.gui_run_status.content = (
            f"ran in `{self.last_run['seconds']}s` -- "
            f"`{len(robot_traj)}` waypoints, `{len(key_idx)}` keyframes, "
            f"`{dupes}` collapsed pairs"
        )

    def _start_playback(self):
        if getattr(self, "_traj", None) is None:
            self.gui_run_status.content = "**run the planner before playing it back**"
            return
        if self._playback_thread is not None and self._playback_thread.is_alive():
            return
        self._playback_stop.clear()
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

    def _playback_loop(self):
        from DoorOpening.utils.state_machine.compute_waypoint import collocate_and_playback

        robot_traj, door_traj, key_idx = self._traj
        try:
            dense_robot, dense_door, _, _, _ = collocate_and_playback(
                robot_traj, door_traj, key_idx, length=600
            )
        except Exception:
            self.gui_run_status.content = f"**interpolation failed**\n\n```\n{traceback.format_exc()[-1200:]}\n```"
            return
        for t in range(len(dense_robot)):
            if self._playback_stop.is_set():
                break
            self.viser_robot.update_cfg(dense_robot[t].numpy())
            self.viser_door.update_cfg(dense_door[t].numpy())
            time.sleep(1.0 / 120.0)

    # ---------------------------------------------------------------- LLM bridge

    def _send_to_llm(self):
        prompt = self.gui_prompt.value.strip()
        if not prompt:
            self.gui_llm_status.content = "_write a prompt first_"
            return
        source = ""
        if os.path.exists(self.planner_path):
            with open(self.planner_path, encoding="utf-8") as handle:
                source = handle.read()
        turn = llm_bridge.next_turn(self.session_dir)
        request = llm_bridge.build_request(
            prompt,
            turn=turn,
            planner_path=self.planner_path,
            planner_source=source,
            scene_state=self.scene_state(),
            last_run=self.last_run,
        )
        path = llm_bridge.write_request(self.session_dir, request)
        llm_bridge.append_transcript(self.session_dir, "human", prompt)
        self._pending_turn = turn
        self.gui_llm_status.content = (
            f"**sent turn {turn}** -- waiting on a reply.\n\n"
            f"Tell your Claude Code session: `check the bridge` (`{path}`)"
        )
        self._append_transcript_view("you", prompt)

    def _poll_llm(self):
        if self._pending_turn is None:
            self.gui_llm_status.content = "_nothing pending_"
            return
        response = llm_bridge.read_response(self.session_dir, self._pending_turn)
        if response is None:
            self.gui_llm_status.content = f"**turn {self._pending_turn}** -- no reply yet"
            return
        reply = response.get("reply", "")
        llm_bridge.append_transcript(self.session_dir, "llm", reply)
        self._append_transcript_view("llm", reply)
        new_source = response.get("planner_source")
        if new_source:
            with open(self.planner_path, "w", encoding="utf-8") as handle:
                handle.write(new_source)
            self.gui_llm_status.content = (
                f"**turn {self._pending_turn}** -- reply received, planner draft updated. "
                "Hit Run planner."
            )
        else:
            self.gui_llm_status.content = f"**turn {self._pending_turn}** -- reply received (no code change)"
        self._pending_turn = None

    def _append_transcript_view(self, role, text):
        prefix = "**you:** " if role == "you" else "**llm:** "
        existing = self.gui_transcript.content
        self.gui_transcript.content = (existing + "\n\n" + prefix + text).strip()[-4000:]

    # ------------------------------------------------------------------- capture

    def _refresh_capture_spec(self):
        """Show the exact numbers a capture would store right now, live as the gizmos move.

        The plan alone (channel, primitive, anchor) was not enough to work with: the question the
        human actually has in front of the scene is "what value am I about to record, and measured
        from what", so both halves belong in one table.
        """
        spec = self._effective_spec()
        # Marking the cells that no longer match the hand-written planners is the whole point of
        # the readout: an anchor is the one field a capture cannot be corrected for afterwards.
        base = default_channel_spec(self.gui_phase.value)
        rows = ["| | channel | value | anchor | |", "|---|---|---|---|---|"]
        for name, s in spec.items():
            try:
                value = f"`{self._channel_value(name, s):+.4f}`"
            except Exception:  # a mid-drag anchor resample must not blank the whole panel
                value = "`--`"
            moved = base.get(name, {}).get("anchor") != s["anchor"]
            anchor = f"**{s['anchor']}**" if moved else s["anchor"]
            rows.append(
                f"| {'&#9679;' if moved else ''} | `{name}` | {value} | {anchor} | "
                f"{'ABS' if s['mode'] == 'world_absolute' else 'rel'} |"
            )
        theta = (f" &nbsp; theta `{float(self.q_door[0]):.2f}`"
                 if self.gui_phase.value in capture_schema.SWEPT_PHASES else "")
        self.gui_capture_spec.content = (
            f"**will record** -- {len(spec)} channels{theta}\n\n" + "\n".join(rows)
        )

    def _effective_spec(self) -> dict:
        """Phase defaults, then the anchor dropdowns, then the JSON box. In that order.

        Widening at each step, so the later and more specific a source is, the more it wins: the
        phase default is what the hand-written planners do, a dropdown is the standing choice for
        every keyframe captured after you set it, the JSON box is a one-channel escape hatch.
        """
        phase_id = self.gui_phase.value
        spec = default_channel_spec(phase_id)
        with_joint_channels(
            spec,
            arm=phase_id in self.joint_phases["arm"] or bool(self.gui_record_arm.value),
            door=phase_id in self.joint_phases["door"] or bool(self.gui_record_door.value),
        )
        for group, dropdown in (("palm", self.gui_palm_anchor), ("base", self.gui_base_anchor)):
            anchor = dropdown.value
            if anchor == ANCHOR_DROPDOWN_DEFAULT:
                continue
            if anchor == "robot_joints":
                # SUBSTITUTIVE, not additive: a configuration already fixes where the hand is and
                # how it is turned, so recording a Cartesian target beside it would store the same
                # waypoint twice and let the two disagree once a fit rounds them apart.
                for name in CHANNEL_GROUPS["palm"] + capture_schema.ROTATION_CHANNELS:
                    if name != "base_yaw":
                        spec.pop(name, None)
                with_joint_channels(spec, arm=True)
                continue
            for name in CHANNEL_GROUPS[group]:
                entry = spec.get(name)
                # Only re-point channels that were already reading from a door anchor; a channel the
                # phase declares world-absolute (a base y that IS a world coordinate) stays that way.
                if entry is not None and entry["mode"] == "anchor_relative":
                    entry["anchor"] = anchor
                    entry["mode"] = capture_schema.implied_mode(anchor)
        raw = (self.gui_channel_overrides.value or "").strip()
        if raw:
            try:
                for name, override in json.loads(raw).items():
                    spec.setdefault(name, {"anchor": "world", "mode": "world_absolute",
                                           "primitive": "constant_offset"}).update(override)
            except (json.JSONDecodeError, AttributeError) as exc:
                self.gui_capture_spec.content = f"**override JSON is not valid:** `{exc}`"
        return spec

    def _channel_value(self, name: str, spec: dict) -> float:
        """The number this channel would contribute, read off the current scene."""
        palm = np.asarray(self.palm_gizmo.position, dtype=np.float32)
        base = np.asarray(self.base_gizmo.position, dtype=np.float32)
        stored = _palm_quat_for_solve_ik(
            torch.from_numpy(np.asarray(self.palm_rot_gizmo.wxyz, dtype=np.float32)).unsqueeze(0),
            self.robot_base_quat,
        )
        roll, pitch, yaw = _rpy_from_quat(stored)
        if name in ("rot_roll", "rot_pitch", "rot_yaw"):
            return {"rot_roll": roll, "rot_pitch": pitch, "rot_yaw": yaw}[name]
        if name == "base_yaw":
            # Stored as a DELTA off the robot's initial yaw, because that is how the planners spell
            # it (robot_initial_yaw.item() + tilt_base_yaw) and what the generator emits.
            base_yaw = _rpy_from_quat(
                torch.from_numpy(np.asarray(self.base_rot_gizmo.wxyz, dtype=np.float32)).unsqueeze(0)
            )[2]
            initial_yaw = _rpy_from_quat(self.robot_base_quat)[2]
            # WRAPPED to (-pi, pi]. euler_xyz_from_quat returns yaw in [0, 2pi), so a base sitting
            # essentially straight can read as -2pi instead of ~0. The planner would not care (both
            # command the same orientation through get_rotation_quat), but the AGGREGATOR would:
            # pooling one demo at -6.27 with another at +0.02 fits a mean near -3.1, which is 180
            # degrees from anything the human demonstrated.
            return _wrap_angle(float(base_yaw - initial_yaw))
        if name in ARM_CHANNELS:
            # q_robot layout is base(3) + panda(7) + ...; arm_j1 is q_robot[3]. Read off the live
            # configuration, so what is recorded is the posture the IK actually landed in.
            return float(self.q_robot[3 + ARM_CHANNELS.index(name)])
        if name == "door_panel":
            return float(self.q_door[0])
        if name == "door_lever":
            return float(self.q_door[1])

        axis = {"base_x": 0, "base_y": 1, "palm_x": 0, "palm_y": 1, "palm_z": 2}[name]
        gizmo = base if name.startswith("base") else palm
        if spec["mode"] == "world_absolute":
            return float(gizmo[axis])
        if spec["anchor"] in ("prev_palm", "prev_base"):
            # A DELTA off the pose carried out of the previous keyframe, because that is what the
            # planner does with it: pull:299 nudges the grasp pose, pull:520 offsets the last palm.
            # Recording the world coordinate here instead would emit `palm_target_pos[:, 2] += 1.02`
            # and throw the hand a metre into the air.
            previous = (self._prev_palm_world if spec["anchor"] == "prev_palm"
                        else self._prev_base_world)
            if previous is None:
                raise capture_schema.CaptureSchemaError(
                    f"channel {name!r} is measured from {spec['anchor']!r}, but nothing has been "
                    "captured yet in this session, so there is no previous pose to measure from. "
                    "Capture the preceding phase first."
                )
            return float(gizmo[axis] - previous[axis])
        if spec["anchor"] == "world":
            # The offset IS the world coordinate -- pull:524's absolute retract height, pull:695's
            # traverse_far_x. Nothing to subtract.
            return float(gizmo[axis])
        eval_theta = spec.get("anchor_eval_theta")
        q_door = self.q_door if eval_theta is None else torch.tensor([eval_theta, 0.0])
        anchor = self._anchor_pos(spec["anchor"], q_door).squeeze(0).numpy()
        return float(gizmo[axis] - anchor[axis])

    def _capture_keyframe(self):
        spec = self._effective_spec()
        phase_id = self.gui_phase.value
        theta = float(self.q_door[0]) if phase_id in capture_schema.SWEPT_PHASES else None
        channels = {}
        try:
            for name, entry in spec.items():
                channels[name] = ChannelSample(
                    value=self._channel_value(name, entry),
                    anchor=entry["anchor"],
                    mode=entry["mode"],
                    primitive=entry["primitive"],
                    anchor_eval_theta=entry.get("anchor_eval_theta"),
                )
        except capture_schema.CaptureSchemaError as exc:
            self.gui_keyframe_list.content = f"**not captured:** {exc}"
            return
        keyframe = Keyframe(
            id=self._next_keyframe_id,
            phase_id=phase_id,
            continuity_block=self.gui_block.value.strip() or None,
            mark_keyframe=bool(self.gui_mark_keyframe.value),
            theta=theta,
            q_door=[float(self.q_door[0]), float(self.q_door[1])],
            arm_joint_snapshot={
                name: round(float(self.q_robot[3 + i]), 6)
                for i, name in enumerate(FULL_JOINT_NAMES[3:10])
            },
            gripper_width=float(self.gui_gripper.value),
            num_attempts=int(self.gui_num_attempts.value),
            channels=channels,
            notes=self.gui_notes.value.strip(),
        )
        try:
            keyframe.validate(f"keyframe {keyframe.id}")
        except capture_schema.CaptureSchemaError as exc:
            self.gui_keyframe_list.content = f"**not captured:** {exc}"
            return
        self.keyframes.append(keyframe)
        self._next_keyframe_id += 1
        # This waypoint is now "the previous pose" for whatever is captured next.
        self._prev_palm_world = np.asarray(self.palm_gizmo.position, dtype=np.float32).copy()
        self._prev_base_world = np.asarray(self.base_gizmo.position, dtype=np.float32).copy()
        # Notes belong to the waypoint they were written about, so clear after attaching rather
        # than silently repeating one note across every later keyframe.
        self.gui_notes.value = ""
        self._refresh_keyframe_list()

    def _drop_keyframe(self):
        if self.keyframes:
            self.keyframes.pop()
            self._refresh_keyframe_list()

    def _refresh_keyframe_list(self):
        if not self.keyframes:
            self.gui_keyframe_list.content = "_no keyframes captured_"
            return
        counts: dict[str, int] = {}
        for keyframe in self.keyframes:
            counts[keyframe.phase_id] = counts.get(keyframe.phase_id, 0) + 1
        summary = ", ".join(f"`{phase}`x{n}" for phase, n in counts.items())
        recent = "\n\n".join(
            f"`{k.id:02d}` {k.phase_id}"
            + (f" theta={k.theta:.2f}" if k.theta is not None else "")
            + ("" if k.mark_keyframe else " (non-key)")
            + (f" -- {k.notes[:48]}" if k.notes else "")
            for k in self.keyframes[-8:]
        )
        self.gui_keyframe_list.content = (
            f"**{len(self.keyframes)} keyframes** -- {summary}\n\n{recent}"
        )

    # --------------------------------------------------------------- theta sweep

    def _sweep_begin(self):
        self._sweep_active = True
        self.gui_panel.value = float(self.gui_sweep_start.value)
        self._on_door_change()
        self.gui_sweep_status.content = (
            f"at theta `{self.gui_panel.value:.2f}` -- drag the gizmos, then "
            "**Capture + advance**."
        )

    def _sweep_capture(self):
        if not self._sweep_active:
            self.gui_sweep_status.content = "**press Begin sweep first**"
            return
        self._capture_keyframe()
        nxt = float(self.gui_panel.value) + float(self.gui_sweep_step.value)
        if nxt > float(self.gui_sweep_stop.value) + 1e-6:
            self._sweep_active = False
            self.gui_sweep_status.content = (
                f"**sweep complete** -- {len(self.keyframes)} keyframes captured in total."
            )
            return
        self.gui_panel.value = nxt
        self._on_door_change()
        self.gui_sweep_status.content = f"at theta `{nxt:.2f}` -- reposition, then capture again."

    # -------------------------------------------------------------------- export

    def _capture_session(self) -> CaptureSession:
        return CaptureSession(
            variant_class=VariantClass(
                handle_side=self.handle_side,
                opening_direction=self.opening_direction,
                door_geometry_bucket=derive_geometry_bucket(self.door_urdf_path),
            ),
            door_urdf_path=self.door_urdf_path,
            robot_urdf_path=self.robot_urdf_path,
            robot_initial_pose_world=self.robot_initial_pose.squeeze(0).tolist(),
            door_initial_pose_world=self.door_initial_pose.squeeze(0).tolist(),
            door_z_lift_m=float(self.gui_door_z.value),
            keyframes=list(self.keyframes),
            exported_at=time.time(),
        )

    def _save_capture(self):
        if not self.keyframes:
            self.gui_export_status.content = "**nothing to save** -- capture some keyframes first"
            return
        path = os.path.join(self.session_dir, f"capture_{int(time.time())}.json")
        save_session(self._capture_session(), path)
        self.gui_export_status.content = (
            f"saved `{path}` (schema v{capture_schema.SCHEMA_VERSION}, "
            f"{len(self.keyframes)} keyframes)"
        )

    def _check_trajectory(self):
        """Run the draft planner and record what the IK managed, for the next turn to carry."""
        from DoorOpening.utils.state_machine.check_planner import format_report, run_and_diagnose

        if not os.path.exists(self.planner_path):
            self.gui_turn_status.content = f"**no draft at** `{self.planner_path}`"
            return
        try:
            report = run_and_diagnose(
                self.planner_path, None, self.door_urdf_path, self.robot_urdf_path,
                device=self.device,
            )
        except Exception:
            self.last_check_report = (
                "TRAJECTORY CHECK\n  the planner raised:\n"
                + traceback.format_exc()[-1200:]
            )
            self.gui_turn_status.content = (
                f"**planner raised** -- captured for the next turn\n\n"
                f"```\n{traceback.format_exc()[-800:]}\n```"
            )
            return
        self.last_check_report = format_report(report)
        self.gui_turn_status.content = (
            f"**{'CLEAN' if report['ok'] else 'NEEDS WORK'}** -- "
            f"{len(report['ik_failures'])} IK failure(s), "
            f"{len(report['collapsed_adjacent_waypoints'])} collapsed pair(s). "
            "Included in the next turn."
        )

    def _write_turn(self):
        """Write one paste-ready turn: feedback, trajectory check, new keyframes, current source."""
        from DoorOpening.utils.state_machine.planner_prompt import build_turn_prompt

        session = self._capture_session()
        planner_source = ""
        if os.path.exists(self.planner_path):
            with open(self.planner_path, encoding="utf-8") as handle:
                planner_source = handle.read()
        try:
            text = build_turn_prompt(
                session.variant_class,
                [session] if self.keyframes else None,
                planner_source=planner_source,
                planner_path=self.planner_path,
                check_report=self.last_check_report,
                feedback=self.gui_feedback.value.strip(),
                since_keyframe=self._last_turn_keyframe_id or None,
            )
        except Exception as exc:
            self.gui_turn_status.content = f"**could not build the turn:** `{exc}`"
            return

        path = os.path.join(self.session_dir, "chat_turn.md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        new_count = self._next_keyframe_id - self._last_turn_keyframe_id
        # Advance the watermark and clear the one-shot fields, so the NEXT turn is genuinely the
        # next delta rather than a repeat of this one.
        self._last_turn_keyframe_id = self._next_keyframe_id
        self.gui_feedback.value = ""
        self.last_check_report = ""
        self.gui_turn_status.content = (
            f"wrote `{path}` -- {len(text.splitlines())} lines, {new_count} new keyframe(s). "
            "Paste it into the chat, then drop the reply into "
            f"`{os.path.basename(self.planner_path)}` and hit **Check trajectory**."
        )

    def _export(self):
        payload = {
            "exported_at": time.time(),
            "scene_state": self.scene_state(),
            "last_run": self.last_run,
            "planner_path": self.planner_path,
        }
        path = os.path.join(self.session_dir, f"session_{int(time.time())}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.gui_llm_status.content = f"exported `{path}`"

    def serve_forever(self):
        print(f"[workbench] session dir: {self.session_dir}")
        print(f"[workbench] planner draft: {self.planner_path}")
        while True:
            time.sleep(1.0)


STARTER_PLANNER = '''"""Draft door planner -- co-authored in the workbench.

Same shape as offline_pull_door.py / offline_push_door.py: every waypoint is an anchor position
plus tuned offsets, run through solve_ik, appended with _append_state. The anchor functions absorb
per-door geometry, so the offsets below are what generalizes across an asset set.

FRAME: palm_pose targets panda_hand (the wrist mount), because that is what solve_ik drives.
palm_center, where the fingers close, is 103.4 mm further along the approach axis.
"""

import math

import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from DoorOpening.constants.robot_constants import (
    DRIVEN_FINGER_JOINT_NAME,
    FRANKA_DEFAULT_JOINT_POS,
    FRANKA_JOINT_NAMES,
    FULL_JOINT_NAMES,
    GRIPPER_OPEN_WIDTH,
)
from DoorOpening.utils.state_machine.api import (
    get_board_pos,
    get_handle_bar_pos,
    get_hinge_pos,
    solve_ik,
)

# The Franka gripper is ONE commanded DOF. q_robot is base(3) + panda(7) + finger(1) + x5(6) = 17.
GRIPPER_Q_IDX = FULL_JOINT_NAMES.index(DRIVEN_FINGER_JOINT_NAME)


def _set_gripper(q_robot, width):
    q_robot[GRIPPER_Q_IDX] = width


def get_rotation_quat(roll, pitch, yaw, device):
    return quat_from_euler_xyz(
        roll=torch.tensor([[roll]], device=device),
        pitch=torch.tensor([[pitch]], device=device),
        yaw=torch.tensor([[yaw]], device=device),
    ).squeeze(0)


def _make_pose(position, quat):
    return torch.cat([position, quat], dim=-1)


def _append_state(robot_traj, door_traj, key_indices, q_robot, q_door, *, mark_keyframe):
    robot_traj.append(q_robot.clone())
    door_traj.append(q_door.clone())
    if mark_keyframe:
        key_indices.append(len(robot_traj) - 1)


def state_machine_offline_door(
    robot_urdf_path,
    door_urdf_path,
    robot_initial_pose,   # (1, 7) world
    door_initial_pose,    # (1, 7) world
    robot_initial_q,      # (ndof,)
    door_initial_q,       # (2,) [panel, lever]
    *,
    handle_side="right",
    device="cpu",
):
    """Entry point the workbench calls. Returns (robot_traj, door_traj, key_idx_in_key_indices)."""
    robot_traj, door_traj, key_idx_in_key_indices = [], [], []
    q_robot = robot_initial_q.clone()
    q_door = door_initial_q.clone()
    _append_state(robot_traj, door_traj, key_idx_in_key_indices, q_robot, q_door, mark_keyframe=False)

    base_target_rot = robot_initial_pose[:, 3:].to(device).clone()
    default_palm_rot = get_rotation_quat(math.pi, math.pi, math.pi, device)

    _append_state(robot_traj, door_traj, key_idx_in_key_indices, q_robot, q_door, mark_keyframe=True)

    # -------------------------
    # Step 1: Pregrasp
    # -------------------------
    handle_pos = get_hinge_pos(door_urdf_path, door_initial_pose, q_door.unsqueeze(0)).to(device)

    pregrasp_base_x_offset = 0.72
    pregrasp_base_y_offset = -0.35
    pregrasp_palm_x_offset = 0.40
    pregrasp_palm_y_offset = -0.15
    pregrasp_palm_z_offset = 0.25

    base_target_pos = handle_pos.clone()
    base_target_pos[:, 0] += pregrasp_base_x_offset
    base_target_pos[:, 1] += pregrasp_base_y_offset
    base_target_pose = _make_pose(base_target_pos, base_target_rot)

    palm_target_pos = handle_pos.clone()
    palm_target_pos[:, 0] += pregrasp_palm_x_offset
    palm_target_pos[:, 1] += pregrasp_palm_y_offset
    palm_target_pos[:, 2] += pregrasp_palm_z_offset
    palm_target_pose = _make_pose(palm_target_pos, default_palm_rot)

    q_robot[:10] = solve_ik(
        robot_urdf_path,
        q_robot[:10],
        palm_pose=palm_target_pose,
        base_pose=base_target_pose,
        robot_initial_pose=robot_initial_pose,
    )[0]

    _append_state(robot_traj, door_traj, key_idx_in_key_indices, q_robot, q_door, mark_keyframe=True)

    # Add the remaining steps here -- grasp, unlatch, the theta sweep, release, traverse.

    return robot_traj, door_traj, key_idx_in_key_indices
'''


def write_starter_planner(path):
    """Seed a draft planner if none exists, so Run works on the first click."""
    if os.path.exists(path):
        return False
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(STARTER_PLANNER)
    return True


def run_workbench(
    robot_urdf_path,
    door_urdf_path,
    *,
    handle_side="auto",
    opening_direction="auto",
    session_dir=None,
    planner_path=None,
    joint_phases=None,
    port=None,
    device="cpu",
):
    """Build the initial scene the same way play_and_save_traj does, then serve the workbench."""
    from DoorOpening.utils.state_machine.compute_waypoint import (
        _apply_initial_state_metadata,
        get_robot_constants,
        resolve_planner_options,
    )

    robot_initial_pose = torch.tensor([[*ROBOT_INITIAL_POS, *ROBOT_INITIAL_ROT]], device=device)
    door_initial_pose = torch.tensor([[*DOOR_INITIAL_POS, *DOOR_INITIAL_ROT]], device=device)
    _, robot_initial_q = get_robot_constants()
    door_initial_q = torch.tensor([0.0, 0.0], device=device)

    (
        robot_initial_pose,
        door_initial_pose,
        robot_initial_q,
        door_initial_q,
        loaded,
    ) = _apply_initial_state_metadata(
        door_urdf_path, robot_initial_pose, door_initial_pose, robot_initial_q, door_initial_q
    )
    if loaded:
        print(f"[workbench] using initial_state from variant_meta.json: {door_urdf_path}")

    handle_side, opening_direction = resolve_planner_options(
        door_urdf_path, handle_side, opening_direction
    )

    variant = os.path.basename(os.path.dirname(door_urdf_path))
    session_dir = session_dir or os.path.join("logs", "workbench", variant)
    planner_path = planner_path or os.path.join(session_dir, "draft_planner.py")
    os.makedirs(session_dir, exist_ok=True)
    if write_starter_planner(planner_path):
        print(f"[workbench] seeded starter planner at {planner_path}")

    workbench = PlannerWorkbench(
        robot_urdf_path,
        door_urdf_path,
        robot_initial_pose,
        door_initial_pose,
        robot_initial_q,
        door_initial_q,
        handle_side=handle_side,
        opening_direction=opening_direction,
        session_dir=session_dir,
        planner_path=planner_path,
        joint_phases=joint_phases,
        port=port,
        device=device,
    )
    workbench.serve_forever()
