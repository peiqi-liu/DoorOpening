import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from DoorOpening.constants.env_constants import DOOR_INITIAL_POS, DOOR_INITIAL_ROT, ROBOT_INITIAL_POS, ROBOT_INITIAL_ROT

"""
Generate simple scratch-built door assets with a fixed 3-link / 2-joint topology.

This script does not depend on any existing door meshes or URDFs. Every variant is
constructed from a few primitive boxes/cylinders:

- `link_0` frame: two side posts plus one top lintel
- `link_1` board: one rectangular box
- `link_2` handle: one stem + one main lever + optional return-lever hook

The resulting assets are meant to stay easy to reason about for point-cloud
training. They preserve the expected IsaacLab naming:

- links: `base`, `link_0`, `link_1`, `link_2`
- joints: `joint_0`, `joint_1`, `joint_2`
"""


DEFAULT_PANEL_WIDTH_RANGE_M = (0.7, 1.0)
DEFAULT_PANEL_HEIGHT_RANGE_M = (1.75, 2.15)
DEFAULT_PANEL_THICKNESS_RANGE_M = (0.028, 0.055)
# Frame BORDER (post width / head height) around the opening. Range STARTS AT 0 so it spans the full
# spectrum: near-0 = effectively NO frame (panel flush in the wall), up to a wide surround. When both
# post width and head height sample near 0 the frame boxes vanish entirely (degenerate -> skipped).
# The panel is centered in the frame (verified: left/right jamb margins are identical on every asset),
# so this value IS the per-side jamb the camera sees -- a 0.20 post reads like a 0.20 m wall flanking
# each side of the door. The upper bound is deliberately WIDE rather than "a normal door frame": the
# jamb is the nearest thing to a wall that is rigidly attached to the door, so spanning frameless ->
# broad casing is a primary randomization axis, not a cosmetic one.
DEFAULT_FRAME_POST_WIDTH_RANGE_M = (0.0, 0.20)
DEFAULT_FRAME_HEAD_HEIGHT_RANGE_M = (0.0, 0.20)
# Frame DEPTH (front-to-back thickness) as a multiple of the panel thickness. From a thin normal jamb
# (~1x) up to a moderately thick wall reveal. Upper bound reduced so the deepest frame is not too large.
DEFAULT_FRAME_DEPTH_SCALE_RANGE = (1.0, 4.0)
DEFAULT_FRAME_CLEARANCE_RANGE_M = (0.003, 0.015)

DEFAULT_HANDLE_HEIGHT_RANGE_M = (0.75, 1.0)
DEFAULT_HANDLE_EDGE_DISTANCE_RANGE_M = (0.03, 0.15)
DEFAULT_HANDLE_RADIUS_RANGE_M = (0.007, 0.013)
# Lever grip bar RADIUS (half cross-section), decoupled from the stem radius -- same convention as
# handle_radius_m, so a value of 0.010 = a 1 cm-radius (2 cm across) bar, matching real lever handles.
# The bar's full cross-section is 2x this. Left at 0 -> falls back to the stem radius.
DEFAULT_HANDLE_LEVER_THICKNESS_RANGE_M = (0.007, 0.013)
# CLEAR underside gap the fingers get between the door PANEL face and the panel-facing (near) surface
# of the lever grip bar. This already ACCOUNTS FOR the lever bar half-thickness -- the stem cylinder is
# extended by that half-thickness so the lever's near surface sits exactly this far above the panel (see
# build_handle_spec). So this range IS the finger clearance under the lever, not the lever-center offset.
DEFAULT_HANDLE_STEM_LENGTH_RANGE_M = (0.043, 0.085)
DEFAULT_HANDLE_LENGTH_RANGE_M = (0.09, 0.14)
DEFAULT_HANDLE_RETURN_LENGTH_RANGE_M = (0.035, 0.08)
DEFAULT_HANDLE_RETURN_TIP_CLEARANCE_RANGE_M = (0.025, 0.050)
DEFAULT_HANDLE_NUM_SEGMENTS = 16
DEFAULT_RETURN_HANDLE_PROB = 0.7

# "Bump" = a raised mount (like a hotel rose/escutcheon) at the handle base that the lever sits on.
# Two shapes are supported:
#   - "box"      : a tall rectangular escutcheon plate (matches real hotel door plates). It is fixed
#                  to the door panel (link_1) so it does NOT rotate when the lever (link_2) turns.
#                  Sized by height (vertical y), width (horizontal x) and protrusion (outward z).
#   - "cylinder" : a round mounting boss co-axial with the stem, on link_2. Rotationally symmetric so
#                  it stays put under the joint_2 sweep. Sized by radius and protrusion (length).
# In both cases the handle stem starts at the OUTER face of the mount, so handle_stem_length is the
# clear gap the fingers get above the mount. On for ~70% of doors by default; when on, the SIZE varies
# down to nearly-flush (smallest plate ≈ no plate), so the no-plate..plate spectrum is covered by both
# the 30% hard-off share AND the size variation of the 70% that are on.
DEFAULT_HANDLE_BUMP_PROB = 0.7
DEFAULT_HANDLE_BUMP_SHAPE = "box"
# Outward protrusion (z) of the mount out of the door face. Range starts near ZERO so a small draw is
# effectively "no bump" (nearly flush), up to a chunky escutcheon (30 mm). This lets a single always-on
# sample cover the whole no-bump..bump spectrum via SIZE instead of a hard prob. The protrusion is
# always clamped so at least MIN_HANDLE_PLATE_GRASP_GAP_M of clear finger space stays above the plate.
DEFAULT_HANDLE_BUMP_LENGTH_RANGE_M = (0.001, 0.030)
# Box plate only: vertical extent (y) and horizontal extent (x) of the escutcheon plate. Start small
# (~2 cm ≈ no visible plate) up to a MODEST escutcheon -- deliberately capped so we never suddenly
# show a huge plate the (not-yet-trained) policy has never seen.
DEFAULT_HANDLE_BUMP_HEIGHT_RANGE_M = (0.02, 0.14)
DEFAULT_HANDLE_BUMP_WIDTH_RANGE_M = (0.02, 0.08)
# Cylinder boss only: radius of the round mount.
DEFAULT_HANDLE_BUMP_RADIUS_RANGE_M = (0.022, 0.035)

DEFAULT_NUM_VARIANTS = 512
DEFAULT_DEBUG_FIRST_N = 0
DEFAULT_START_BASE_SAMPLING = "paired"
DEFAULT_START_BASE_RADIUS_M = 1.0
# Per-door standoff distance is now drawn from this (min, max) range instead of the single fixed
# radius above, so the robot no longer always starts exactly 1.0 m away. Widened to push the robot
# farther back (up to 2.0 m) so the policy must handle longer approaches, not just near-contact starts.
DEFAULT_START_BASE_RADIUS_RANGE_M = (1.0, 1.5)
# Approach-angle spread (was 30 -> 15). This is ALSO the initial base yaw: the robot sits at angle
# alpha on the door-centered ring and must yaw by alpha to face the door, so base_rotation = alpha.
# Tightened to 8 deg so the start heading isn't yawed too far off the door axis.
DEFAULT_START_BASE_ANGLE_RANGE_DEG = 8.0
DEFAULT_START_BASE_PAIR_SEED = 0
DEFAULT_FRANKA_START_JOINT_NOISE_RAD = 0.05

FRANKA_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
FRANKA_DEFAULT_JOINT_POS = [
    0.0,
    -0.7853981633974483,
    0.0,
    -2.356194490192345,
    0.0,
    1.5707963267948966,
    0.0,
]

DOOR_OPEN_LIMIT_RAD = 1.57
HANDLE_OPEN_LIMIT_RAD = 1.57
ROOT_JOINT_RPY = [math.pi / 2.0, 0.0, -math.pi / 2.0]
# With the fixed root rotation above, link_1 local -z points toward the
# default front view used by scripts/test_door.py, while +z points to the back.
FRONT_FACE_SIGN_LINK1 = -1.0
BACK_FACE_SIGN_LINK1 = -FRONT_FACE_SIGN_LINK1
HANDLE_FACE_SIGN_LINK1 = BACK_FACE_SIGN_LINK1

MIN_HANDLE_BOTTOM_CLEARANCE_M = 0.15
MIN_HANDLE_TOP_CLEARANCE_M = 0.15
MIN_HANDLE_EDGE_CLEARANCE_M = 0.02
MIN_RETURN_TIP_CLEARANCE_M = 0.010
# Minimum CLEAR underside gap kept between the escutcheon plate/boss outer face and the lever bar's near
# surface -- i.e. the finger clearance under the lever when a plate is present. The mount protrusion is
# clamped so it can never eat into this, so the plate-referenced underside gap never drops below it (the
# panel-referenced underside gap is the larger DEFAULT_HANDLE_STEM_LENGTH_RANGE_M value).
MIN_HANDLE_PLATE_GRASP_GAP_M = 0.050

REQUIRED_LINK_NAMES = {"base", "link_0", "link_1", "link_2"}
REQUIRED_JOINT_NAMES = {"joint_0", "joint_1", "joint_2"}


def parse_args():
    default_output_dir = REPO_ROOT / "source" / "DoorOpening" / "assets" / "door" / "ScratchDoors"

    parser = argparse.ArgumentParser(
        description="Generate simple scratch-built door assets with procedural frames, boards, and lever handles."
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--num-variants", type=int, default=DEFAULT_NUM_VARIANTS)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First variant index; variant dirs are named <prefix>__rnd_<start-index+i>. Use a nonzero "
        "value (e.g. 1) to avoid the reserved __rnd_00 suffix, which the dagger train/validation split "
        "treats as a held-out validation door (a single __rnd_00 door leaves the training split empty).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--asset-name-prefix",
        type=str,
        default="scratch_door",
        help="Variant directory prefix, matching the `<name>__rnd_XX` pattern used by generate_randomized_doors.py.",
    )
    parser.add_argument(
        "--handle-side",
        choices=("random", "left", "right"),
        default="random",
        help="Which side of the door panel carries the handle. The hinge is always placed on the opposite side.",
    )
    parser.add_argument(
        "--opening-direction",
        choices=("random", "pull", "push"),
        default="random",
        help="Whether the door should open as pull or push, encoded through joint_1 axis sign.",
    )
    parser.add_argument(
        "--panel-width-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_PANEL_WIDTH_RANGE_M,
    )
    parser.add_argument(
        "--panel-height-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_PANEL_HEIGHT_RANGE_M,
    )
    parser.add_argument(
        "--panel-thickness-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_PANEL_THICKNESS_RANGE_M,
    )
    parser.add_argument(
        "--frame-post-width-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_FRAME_POST_WIDTH_RANGE_M,
    )
    parser.add_argument(
        "--frame-head-height-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_FRAME_HEAD_HEIGHT_RANGE_M,
    )
    parser.add_argument(
        "--frame-depth-scale-range",
        type=float,
        nargs=2,
        metavar=("MIN_SCALE", "MAX_SCALE"),
        default=DEFAULT_FRAME_DEPTH_SCALE_RANGE,
    )
    parser.add_argument(
        "--frame-clearance-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_FRAME_CLEARANCE_RANGE_M,
    )
    parser.add_argument(
        "--handle-height-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_HEIGHT_RANGE_M,
    )
    parser.add_argument(
        "--handle-edge-distance-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_EDGE_DISTANCE_RANGE_M,
    )
    parser.add_argument(
        "--handle-radius-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_RADIUS_RANGE_M,
    )
    parser.add_argument(
        "--handle-lever-thickness-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_LEVER_THICKNESS_RANGE_M,
        help="Lever grip bar RADIUS (half cross-section) in meters, decoupled from the stem radius -- "
        "same convention as --handle-radius-range, so 0.010 = a 1 cm-radius (2 cm across) bar. The bar's "
        "full cross-section is 2x this. Set both to 0 to fall back to the stem radius.",
    )
    parser.add_argument(
        "--handle-stem-length-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_STEM_LENGTH_RANGE_M,
        help="CLEAR underside gap (m) between the door PANEL face and the lever bar's near surface -- the "
        "actual finger clearance under the lever (already accounts for the bar half-thickness). With a "
        "plate/bump present the plate-referenced gap is this minus the mount protrusion (>= "
        "MIN_HANDLE_PLATE_GRASP_GAP_M).",
    )
    parser.add_argument(
        "--handle-length-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_LENGTH_RANGE_M,
    )
    parser.add_argument(
        "--handle-return-length-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_RETURN_LENGTH_RANGE_M,
    )
    parser.add_argument(
        "--handle-return-tip-clearance-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_RETURN_TIP_CLEARANCE_RANGE_M,
        help="Minimum remaining gap between the return lever tip and the door face.",
    )
    parser.add_argument("--return-handle-prob", type=float, default=DEFAULT_RETURN_HANDLE_PROB)
    parser.add_argument(
        "--handle-bump-prob",
        type=float,
        default=DEFAULT_HANDLE_BUMP_PROB,
        help="Probability a variant gets a raised mount (bump) at the handle base that the lever "
        "sits on (default 0.7). When on, the plate SIZE still varies down to nearly-flush, so the "
        "smallest plate is ~no plate. 0.0 disables it entirely; 1.0 forces a (possibly tiny) plate on all.",
    )
    parser.add_argument(
        "--handle-bump-shape",
        choices=("box", "cylinder", "random"),
        default=DEFAULT_HANDLE_BUMP_SHAPE,
        help="Mount shape: 'box' = rectangular escutcheon plate on the panel (default); 'cylinder' = "
        "round boss on the handle; 'random' picks per variant.",
    )
    parser.add_argument(
        "--handle-bump-length-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_BUMP_LENGTH_RANGE_M,
        help="How far the mount protrudes out of the door face, meters (both shapes).",
    )
    parser.add_argument(
        "--handle-bump-height-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_BUMP_HEIGHT_RANGE_M,
        help="Box plate only: vertical extent (meters) of the escutcheon plate.",
    )
    parser.add_argument(
        "--handle-bump-width-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_BUMP_WIDTH_RANGE_M,
        help="Box plate only: horizontal extent (meters) of the escutcheon plate.",
    )
    parser.add_argument(
        "--handle-bump-radius-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_BUMP_RADIUS_RANGE_M,
        help="Cylinder boss only: radius (meters); should be wider than the handle stem radius.",
    )
    parser.add_argument("--handle-num-segments", type=int, default=DEFAULT_HANDLE_NUM_SEGMENTS)
    parser.add_argument("--debug-first-n", type=int, default=DEFAULT_DEBUG_FIRST_N)
    parser.add_argument(
        "--start-base-sampling",
        choices=("paired", "random", "none"),
        default=DEFAULT_START_BASE_SAMPLING,
        help="How to sample and store the initial mobile-base state in variant_meta.json.",
    )
    parser.add_argument(
        "--start-base-radius",
        type=float,
        default=DEFAULT_START_BASE_RADIUS_M,
        help="Fixed fallback radius in meters (used only when --start-base-radius-range is disabled).",
    )
    parser.add_argument(
        "--start-base-radius-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=DEFAULT_START_BASE_RADIUS_RANGE_M,
        help="Per-door standoff distance range (min max) in meters; overrides --start-base-radius.",
    )
    parser.add_argument(
        "--start-base-angle-range-deg",
        type=float,
        default=DEFAULT_START_BASE_ANGLE_RANGE_DEG,
        help="Symmetric angle range in degrees for initial mobile-base waypoint sampling.",
    )
    parser.add_argument(
        "--start-base-pair-seed",
        type=int,
        default=DEFAULT_START_BASE_PAIR_SEED,
        help="Seed mixed into the deterministic paired start-base hash.",
    )
    parser.add_argument(
        "--franka-start-joint-noise",
        type=float,
        default=DEFAULT_FRANKA_START_JOINT_NOISE_RAD,
        help="Symmetric uniform noise (rad) added to each Franka default joint at the start pose. "
        "Set 0.0 to start exactly at the default Franka configuration.",
    )
    return parser.parse_args()


def format_vector(values):
    return " ".join(f"{float(value):.6f}" for value in values)


def resolve_range(values, name):
    low = float(values[0])
    high = float(values[1])
    if high < low:
        raise ValueError(f"{name} must satisfy min <= max, got {values}")
    return low, high


def resolve_probability(value, name):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def sample_uniform(rng, value_range):
    return rng.uniform(float(value_range[0]), float(value_range[1]))


def clamp(value, low, high):
    return min(max(value, low), high)


def midpoint(a, b):
    return [(x + y) * 0.5 for x, y in zip(a, b)]


def wrap_to_pi(angle_rad):
    return ((angle_rad + math.pi) % (2.0 * math.pi)) - math.pi


def quat_xyzw_to_yaw(quat):
    x, y, z, w = [float(v) for v in quat]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat_xyzw(yaw_rad):
    half = 0.5 * float(yaw_rad)
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def quat_wxyz_to_yaw(quat):
    # ROBOT_INITIAL_ROT / DOOR_INITIAL_ROT are stored in Isaac's (w, x, y, z) convention, so their
    # yaw must be read as wxyz -- reading (0,0,0,1) as xyzw wrongly gives yaw 0 instead of pi.
    w, x, y, z = [float(v) for v in quat]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat_wxyz(yaw_rad):
    half = 0.5 * float(yaw_rad)
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def stable_unit_interval(key):
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)


def paired_start_angle(pair_key, seed, angle_range_deg):
    angle_limit = math.radians(angle_range_deg)
    unit = stable_unit_interval(f"dooropening-start-base-v1:{int(seed)}:{pair_key}")
    return -angle_limit + 2.0 * angle_limit * unit


def paired_start_radius(pair_key, seed, radius_range):
    lo, hi = float(radius_range[0]), float(radius_range[1])
    unit = stable_unit_interval(f"dooropening-start-radius-v1:{int(seed)}:{pair_key}")
    return lo + (hi - lo) * unit


def compute_base_joint_from_world_pose(base_world_pose, target_world_pose):
    base_pos = base_world_pose["pos"]
    base_yaw = quat_wxyz_to_yaw(base_world_pose["rot"])
    target_pos = target_world_pose["pos"]
    target_yaw = quat_wxyz_to_yaw(target_world_pose["rot"])
    dx = float(target_pos[0]) - float(base_pos[0])
    dy = float(target_pos[1]) - float(base_pos[1])
    x_r = math.cos(base_yaw) * dx + math.sin(base_yaw) * dy
    y_r = -math.sin(base_yaw) * dx + math.cos(base_yaw) * dy
    theta_r = wrap_to_pi(target_yaw - base_yaw)
    return [x_r, y_r, theta_r]


def sample_initial_state(rng, variant_name, args):
    robot_world_pose = {"pos": list(ROBOT_INITIAL_POS), "rot": list(ROBOT_INITIAL_ROT)}
    door_world_pose = {"pos": list(DOOR_INITIAL_POS), "rot": list(DOOR_INITIAL_ROT)}
    door_joint_pos = [0.0, 0.0]
    robot_base_joint_pos = [0.0, 0.0, 0.0]
    franka_start_joint_noise = float(getattr(args, "franka_start_joint_noise", DEFAULT_FRANKA_START_JOINT_NOISE_RAD))
    robot_franka_joint_pos = [
        default_value + rng.uniform(-franka_start_joint_noise, franka_start_joint_noise)
        for default_value in FRANKA_DEFAULT_JOINT_POS
    ]
    sampled_robot_world_pose = None
    sampled_approach_angle_rad = None
    pair_key = None

    sampled_radius = float(args.start_base_radius)
    radius_range = getattr(args, "start_base_radius_range", None)
    if args.start_base_sampling != "none":
        if args.start_base_sampling == "paired":
            pair_key = build_direction_pair_key(variant_name)
            sampled_approach_angle_rad = paired_start_angle(
                pair_key,
                args.start_base_pair_seed,
                args.start_base_angle_range_deg,
            )
            if radius_range is not None:
                sampled_radius = paired_start_radius(pair_key, args.start_base_pair_seed, radius_range)
        else:
            angle_limit = math.radians(args.start_base_angle_range_deg)
            sampled_approach_angle_rad = rng.uniform(-angle_limit, angle_limit)
            if radius_range is not None:
                sampled_radius = rng.uniform(float(radius_range[0]), float(radius_range[1]))

        # Sample a standoff on a circle centered on the DOOR, then aim the base back at the door.
        # The home pose faces the door (ROBOT_INITIAL_ROT is a 180 deg z-rotation in Isaac's wxyz
        # convention -> home_yaw = pi, heading = -x toward the door); from an off-axis standoff the
        # base must yaw by the approach angle to keep facing the door center. compute_base_joint_...
        # then maps the world offset into the base-link frame -- reading the wxyz home yaw correctly
        # is what makes base_x/base_y place the robot ON the door-centered circle (a wrong yaw of 0
        # mirrors it, dropping the robot onto the door for larger radii).
        home_yaw = quat_wxyz_to_yaw(ROBOT_INITIAL_ROT)
        facing_yaw = wrap_to_pi(home_yaw + sampled_approach_angle_rad)
        sampled_robot_world_pose = {
            "pos": [
                float(DOOR_INITIAL_POS[0]) + sampled_radius * math.cos(sampled_approach_angle_rad),
                float(DOOR_INITIAL_POS[1]) + sampled_radius * math.sin(sampled_approach_angle_rad),
                float(ROBOT_INITIAL_POS[2]),
            ],
            "rot": yaw_to_quat_wxyz(facing_yaw),
        }
        robot_base_joint_pos = compute_base_joint_from_world_pose(robot_world_pose, sampled_robot_world_pose)

    return {
        "robot_world_pose": robot_world_pose,
        "door_world_pose": door_world_pose,
        "robot_joint_names": ["base_x_joint", "base_y_joint", "base_rotation_joint"] + FRANKA_JOINT_NAMES,
        "robot_joint_pos": robot_base_joint_pos + robot_franka_joint_pos,
        "robot_base_joint_pos": robot_base_joint_pos,
        "robot_franka_joint_names": FRANKA_JOINT_NAMES,
        "robot_franka_joint_pos": robot_franka_joint_pos,
        "door_joint_pos": door_joint_pos,
        "start_base_sampling": str(args.start_base_sampling),
        "start_base_radius_m": float(sampled_radius),
        "start_base_radius_range_m": None if radius_range is None else [float(radius_range[0]), float(radius_range[1])],
        "start_base_angle_range_deg": float(args.start_base_angle_range_deg),
        "start_base_pair_seed": int(args.start_base_pair_seed),
        "start_base_pair_key": None if pair_key is None else str(pair_key),
        "sampled_robot_world_pose": sampled_robot_world_pose,
        "sampled_approach_angle_rad": (
            None if sampled_approach_angle_rad is None else float(sampled_approach_angle_rad)
        ),
    }


class ObjMeshBuilder:
    def __init__(self):
        self.vertices = []
        self.faces = []

    def add_box_bounds(self, min_corner, max_corner):
        x0, y0, z0 = min_corner
        x1, y1, z1 = max_corner
        if x1 - x0 <= 1e-9 or y1 - y0 <= 1e-9 or z1 - z0 <= 1e-9:
            return
        base = len(self.vertices) + 1
        self.vertices.extend(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ]
        )
        self.faces.extend(
            [
                [base + 0, base + 1, base + 2],
                [base + 0, base + 2, base + 3],
                [base + 4, base + 7, base + 6],
                [base + 4, base + 6, base + 5],
                [base + 0, base + 4, base + 5],
                [base + 0, base + 5, base + 1],
                [base + 1, base + 5, base + 6],
                [base + 1, base + 6, base + 2],
                [base + 2, base + 6, base + 7],
                [base + 2, base + 7, base + 3],
                [base + 3, base + 7, base + 4],
                [base + 3, base + 4, base + 0],
            ]
        )

    def add_cylinder(self, start, end, radius, num_segments):
        axis = [end[i] - start[i] for i in range(3)]
        length = math.sqrt(sum(value * value for value in axis))
        if length <= 1e-9 or radius <= 1e-9:
            return

        axis_dir = [value / length for value in axis]
        reference = [1.0, 0.0, 0.0] if abs(axis_dir[0]) < 0.9 else [0.0, 1.0, 0.0]
        u = normalize(cross(axis_dir, reference))
        v = normalize(cross(axis_dir, u))

        base = len(self.vertices) + 1
        ring_size = max(3, int(num_segments))
        start_ring = []
        end_ring = []
        for segment_idx in range(ring_size):
            theta = 2.0 * math.pi * segment_idx / ring_size
            radial = [
                radius * math.cos(theta) * u[i] + radius * math.sin(theta) * v[i]
                for i in range(3)
            ]
            start_ring.append([start[i] + radial[i] for i in range(3)])
            end_ring.append([end[i] + radial[i] for i in range(3)])
        self.vertices.extend(start_ring)
        self.vertices.extend(end_ring)
        self.vertices.append(list(start))
        self.vertices.append(list(end))

        start_center_idx = base + 2 * ring_size
        end_center_idx = start_center_idx + 1
        for segment_idx in range(ring_size):
            next_idx = (segment_idx + 1) % ring_size
            s0 = base + segment_idx
            s1 = base + next_idx
            e0 = base + ring_size + segment_idx
            e1 = base + ring_size + next_idx
            self.faces.append([s0, s1, e1])
            self.faces.append([s0, e1, e0])
            self.faces.append([start_center_idx, s1, s0])
            self.faces.append([end_center_idx, e0, e1])

    def write(self, path):
        with open(path, "w", encoding="utf-8") as obj_file:
            for vertex in self.vertices:
                obj_file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for face in self.faces:
                obj_file.write(f"f {face[0]} {face[1]} {face[2]}\n")


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def normalize(vector):
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 1e-12:
        raise ValueError(f"Cannot normalize near-zero vector {vector}")
    return [value / length for value in vector]


def prepare_output_dir(output_dir, overwrite):
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"{output_dir} already exists and is not empty. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)


def choose_mode(rng, requested, options):
    if requested != "random":
        return requested
    return rng.choice(tuple(options))


def normalize_x_edge_name(value):
    if value in {"left", "min"}:
        return "min"
    if value in {"right", "max"}:
        return "max"
    raise ValueError(f"Unsupported x-edge name: {value}")


def sample_hinge_and_handle_sides(rng, args):
    handle_side = normalize_x_edge_name(choose_mode(rng, args.handle_side, options=("left", "right")))
    hinge_side = "max" if handle_side == "min" else "min"
    return hinge_side, handle_side


def sample_variant_spec(rng, args):
    panel_width = sample_uniform(rng, args.panel_width_range)
    panel_height = sample_uniform(rng, args.panel_height_range)
    panel_thickness = sample_uniform(rng, args.panel_thickness_range)
    frame_post_width = sample_uniform(rng, args.frame_post_width_range)
    frame_head_height = sample_uniform(rng, args.frame_head_height_range)
    frame_depth_scale = sample_uniform(rng, args.frame_depth_scale_range)
    frame_clearance = sample_uniform(rng, args.frame_clearance_range)
    hinge_side, handle_side = sample_hinge_and_handle_sides(rng, args)
    opening_direction = choose_mode(rng, args.opening_direction, options=("pull", "push"))

    handle_height = sample_uniform(rng, args.handle_height_range)
    handle_edge_distance = sample_uniform(rng, args.handle_edge_distance_range)
    handle_radius = sample_uniform(rng, args.handle_radius_range)
    handle_lever_thickness = sample_uniform(rng, args.handle_lever_thickness_range)
    handle_stem_length = sample_uniform(rng, args.handle_stem_length_range)
    handle_length = sample_uniform(rng, args.handle_length_range)
    handle_return_length = sample_uniform(rng, args.handle_return_length_range)
    handle_return_tip_clearance = sample_uniform(rng, args.handle_return_tip_clearance_range)
    handle_has_return = rng.random() < args.return_handle_prob
    handle_bump_length = sample_uniform(rng, args.handle_bump_length_range)
    handle_bump_height = sample_uniform(rng, args.handle_bump_height_range)
    handle_bump_width = sample_uniform(rng, args.handle_bump_width_range)
    handle_bump_radius = sample_uniform(rng, args.handle_bump_radius_range)
    handle_bump_shape = choose_mode(rng, args.handle_bump_shape, options=("box", "cylinder"))
    handle_has_bump = rng.random() < args.handle_bump_prob

    return {
        "panel_width_m": panel_width,
        "panel_height_m": panel_height,
        "panel_thickness_m": panel_thickness,
        "frame_post_width_m": frame_post_width,
        "frame_head_height_m": frame_head_height,
        "frame_depth_m": panel_thickness * frame_depth_scale,
        "frame_clearance_m": frame_clearance,
        "hinge_side": hinge_side,
        "handle_side": handle_side,
        "opening_direction": opening_direction,
        "handle_height_m": handle_height,
        "handle_edge_distance_m": handle_edge_distance,
        "handle_radius_m": handle_radius,
        "handle_lever_thickness_m": handle_lever_thickness,
        "handle_stem_length_m": handle_stem_length,
        "handle_length_m": handle_length,
        "handle_return_length_m": handle_return_length,
        "handle_return_tip_clearance_m": handle_return_tip_clearance,
        "handle_has_return": handle_has_return,
        "handle_bump_length_m": handle_bump_length,
        "handle_bump_height_m": handle_bump_height,
        "handle_bump_width_m": handle_bump_width,
        "handle_bump_radius_m": handle_bump_radius,
        "handle_bump_shape": handle_bump_shape,
        "handle_has_bump": handle_has_bump,
    }


def build_board_bounds(spec):
    width = spec["panel_width_m"]
    height = spec["panel_height_m"]
    thickness = spec["panel_thickness_m"]
    # Keep link_1 local y as the panel-height axis, but place the panel bottom
    # at y=0 so the asset rests on the ground once joint_0 rotates the door into
    # Isaac's z-up world frame.
    y0 = 0.0
    y1 = height
    z0 = -0.5 * thickness
    z1 = 0.5 * thickness
    if spec["hinge_side"] == "min":
        x0 = 0.0
        x1 = width
    else:
        x0 = -width
        x1 = 0.0
    return [x0, y0, z0], [x1, y1, z1]


def build_frame_boxes(spec):
    width = spec["panel_width_m"]
    height = spec["panel_height_m"]
    post = spec["frame_post_width_m"]
    head = spec["frame_head_height_m"]
    depth = max(spec["frame_depth_m"], spec["panel_thickness_m"] + 0.01)
    clearance = spec["frame_clearance_m"]
    opening_min_x = -0.5 * width
    opening_max_x = 0.5 * width
    z0 = -0.5 * depth
    z1 = 0.5 * depth

    left_post = [
        opening_min_x - clearance - post,
        0.0,
        z0,
    ], [
        opening_min_x - clearance,
        height + clearance + head,
        z1,
    ]
    right_post = [
        opening_max_x + clearance,
        0.0,
        z0,
    ], [
        opening_max_x + clearance + post,
        height + clearance + head,
        z1,
    ]
    top_lintel = [
        opening_min_x - clearance - post,
        height + clearance,
        z0,
    ], [
        opening_max_x + clearance + post,
        height + clearance + head,
        z1,
    ]
    return [left_post, right_post, top_lintel]


def build_joint_1_origin(spec):
    width = spec["panel_width_m"]
    if spec["hinge_side"] == "min":
        return [-0.5 * width, 0.0, 0.0]
    return [0.5 * width, 0.0, 0.0]


def clamp_handle_height(panel_height, requested_height):
    low = MIN_HANDLE_BOTTOM_CLEARANCE_M
    high = max(low, panel_height - MIN_HANDLE_TOP_CLEARANCE_M)
    return clamp(requested_height, low, high)


def clamp_edge_distance(panel_width, requested_edge_distance):
    low = MIN_HANDLE_EDGE_CLEARANCE_M
    high = max(low, panel_width - MIN_HANDLE_EDGE_CLEARANCE_M)
    return clamp(requested_edge_distance, low, high)


def build_handle_spec(spec, board_min, board_max):
    width = board_max[0] - board_min[0]
    height = board_max[1] - board_min[1]
    thickness = board_max[2] - board_min[2]

    handle_height = clamp_handle_height(height, spec["handle_height_m"])
    edge_distance = clamp_edge_distance(width, spec["handle_edge_distance_m"])

    if spec["handle_side"] == "max":
        joint_x = board_max[0] - edge_distance
        lever_direction_x = -1.0
    else:
        joint_x = board_min[0] + edge_distance
        lever_direction_x = 1.0

    # Keep the generated scratch handle on the back face of the door panel.
    # Opening direction is still represented through joint_1, not by moving the
    # single visible handle across the slab faces.
    face_sign = HANDLE_FACE_SIGN_LINK1
    joint_origin_link1 = [
        joint_x,
        board_min[1] + handle_height,
        face_sign * 0.5 * thickness,
    ]

    radius = spec["handle_radius_m"]
    # Lever grip bar RADIUS (half cross-section), decoupled from the stem radius. The sampled
    # handle_lever_thickness_m value IS the radius (like handle_radius_m), so the bar's full cross-section
    # is 2x this -- matching real lever handles (~1 cm radius). Falls back to the stem radius when 0.
    lever_radius = float(spec.get("handle_lever_thickness_m", 0.0))
    lever_half_h = lever_radius if lever_radius > 0.0 else radius
    # handle_stem_length_m is the CLEAR underside gap the fingers get to the PANEL: the distance from the
    # panel face to the panel-facing (near) surface of the lever bar. The stem cylinder runs to the lever
    # CENTER, which is one bar half-thickness further out -- so extend it by lever_half_h. This makes the
    # sampled range the actual finger clearance under the lever (accounting for the bar radius/thickness).
    underside_gap_panel = spec["handle_stem_length_m"]
    stem_length = underside_gap_panel + lever_half_h
    lever_length = spec["handle_length_m"]
    min_tip_clearance = max(MIN_RETURN_TIP_CLEARANCE_M, spec["handle_return_tip_clearance_m"])
    tip_clearance = min(min_tip_clearance, stem_length)
    max_return_length = max(0.0, stem_length - tip_clearance)
    return_length = min(spec["handle_return_length_m"], max_return_length)
    outward_sign = face_sign

    # Raised mount ("bump"): a plate/boss protruding from the door face at the handle base. IMPORTANT:
    # `stem_length` is the gap between the handle and the door PANEL, measured from the PANEL FACE -- NOT
    # from the mount. The mount sits WITHIN that gap, behind the lever, and does NOT push the lever
    # further out. We clamp the mount protrusion so it always stays behind the lever, keeping at least
    # MIN_HANDLE_PLATE_GRASP_GAP_M of clear grasp space between the plate/boss and the lever.
    has_bump = bool(spec.get("handle_has_bump", False))
    # Skip the bump entirely when the underside gap can't fit a protruding plate while keeping the full
    # MIN_HANDLE_PLATE_GRASP_GAP_M of clear finger space -- otherwise a flush (0-protrusion) plate would
    # leave a plate-referenced gap below the minimum. So every bumped door has >= MIN clearance.
    if has_bump and underside_gap_panel < MIN_HANDLE_PLATE_GRASP_GAP_M:
        has_bump = False
    bump_shape = str(spec.get("handle_bump_shape", "box")) if has_bump else "none"
    bump_protrusion = float(spec.get("handle_bump_length_m", 0.0)) if has_bump else 0.0
    # Clamp the mount so at least MIN_HANDLE_PLATE_GRASP_GAP_M of CLEAR finger space remains between the
    # plate outer face and the lever's near surface. Measured from the panel-referenced underside gap, so
    # the resulting plate-referenced underside gap = underside_gap_panel - bump_protrusion >= the minimum.
    bump_protrusion = min(bump_protrusion, max(0.0, underside_gap_panel - MIN_HANDLE_PLATE_GRASP_GAP_M))
    plate_underside_gap = (underside_gap_panel - bump_protrusion) if has_bump else None
    bump_radius = max(float(spec.get("handle_bump_radius_m", 0.0)), radius * 1.5)
    mount_out = outward_sign * bump_protrusion  # signed z of the mount's outer face

    # Stem/lever standoff is measured from the PANEL face (z=0), so the handle<->panel gap == stem_length.
    stem_start = [0.0, 0.0, 0.0]
    stem_end = [0.0, 0.0, outward_sign * stem_length]
    lever_start = list(stem_end)
    lever_tip = [lever_direction_x * lever_length, 0.0, outward_sign * stem_length]
    return_tip = [lever_tip[0], 0.0, outward_sign * (stem_length - return_length)]

    # Round boss (cylinder) lives on link_2 co-axial with the stem: from the door face out to its outer face.
    bump_start = [0.0, 0.0, 0.0]
    bump_end = [0.0, 0.0, mount_out]

    # Box escutcheon plate lives on link_1 (the panel) so it does NOT rotate with the lever. It is a
    # tall, thin rectangular slab centred on the handle joint, protruding out of the handle face. We
    # embed it a few mm into the panel to avoid a coplanar seam, and clamp it to the board bounds.
    plate_min_link1 = None
    plate_max_link1 = None
    if has_bump and bump_shape == "box":
        bump_height = float(spec.get("handle_bump_height_m", 0.0))
        bump_width = float(spec.get("handle_bump_width_m", 0.0))
        cx = joint_x
        cy = joint_origin_link1[1]
        face_z = joint_origin_link1[2]  # outer (handle) face of the panel
        embed = 0.003
        px0 = clamp(cx - 0.5 * bump_width, board_min[0], board_max[0])
        px1 = clamp(cx + 0.5 * bump_width, board_min[0], board_max[0])
        py0 = clamp(cy - 0.5 * bump_height, board_min[1] + 0.01, board_max[1] - 0.01)
        py1 = clamp(cy + 0.5 * bump_height, board_min[1] + 0.01, board_max[1] - 0.01)
        inner_z = face_z - outward_sign * embed
        outer_z = face_z + mount_out
        pz0, pz1 = (inner_z, outer_z) if outer_z >= inner_z else (outer_z, inner_z)
        plate_min_link1 = [px0, py0, pz0]
        plate_max_link1 = [px1, py1, pz1]

    return {
        "type": "return" if spec["handle_has_return"] else "straight",
        "has_bump": has_bump,
        "bump_shape": bump_shape,
        "bump_protrusion_m": bump_protrusion,
        "bump_height_m": float(spec.get("handle_bump_height_m", 0.0)),
        "bump_width_m": float(spec.get("handle_bump_width_m", 0.0)),
        "bump_radius_m": bump_radius,
        "mount_out_m": bump_protrusion,
        "bump_start_link2": bump_start,
        "bump_end_link2": bump_end,
        "plate_min_link1": plate_min_link1,
        "plate_max_link1": plate_max_link1,
        "joint_origin_link1": joint_origin_link1,
        "joint_axis_link1": [0.0, 0.0, -lever_direction_x],
        "lever_direction_link2": [lever_direction_x, 0.0, 0.0],
        "return_direction_link2": [0.0, 0.0, -outward_sign],
        "radius_m": radius,
        "lever_thickness_m": 2.0 * lever_half_h,  # full cross-section (= 2 x lever radius)
        "lever_half_height_m": lever_half_h,
        "stem_length_m": stem_length,
        "panel_underside_gap_m": underside_gap_panel,
        "plate_underside_gap_m": plate_underside_gap,
        "lever_length_m": lever_length,
        "return_length_m": return_length,
        "return_tip_clearance_m": tip_clearance,
        "stem_start_link2": stem_start,
        "stem_end_link2": stem_end,
        "lever_start_link2": lever_start,
        "lever_tip_link2": lever_tip,
        "return_tip_link2": return_tip,
    }


def write_board_mesh(mesh_path, board_min, board_max, plate_box=None):
    builder = ObjMeshBuilder()
    builder.add_box_bounds(board_min, board_max)
    if plate_box is not None:
        # Escutcheon plate is baked into the panel mesh so it is fixed to link_1 (visual + collision).
        plate_min, plate_max = plate_box
        builder.add_box_bounds(plate_min, plate_max)
    builder.write(mesh_path)


def write_frame_mesh(mesh_path, frame_boxes):
    builder = ObjMeshBuilder()
    for box_min, box_max in frame_boxes:
        builder.add_box_bounds(box_min, box_max)
    builder.write(mesh_path)


def write_handle_mesh(mesh_path, handle_spec, num_segments):
    builder = ObjMeshBuilder()
    radius = handle_spec["radius_m"]
    if (
        handle_spec.get("has_bump")
        and handle_spec.get("bump_shape") == "cylinder"
        and handle_spec.get("bump_protrusion_m", 0.0) > 0.0
    ):
        # Round mounting boss the lever sits on. Drawn first so it reads as the base of the handle.
        # (The box plate is instead baked into the panel mesh; see write_board_mesh.)
        builder.add_cylinder(
            handle_spec["bump_start_link2"],
            handle_spec["bump_end_link2"],
            handle_spec["bump_radius_m"],
            num_segments,
        )
    builder.add_cylinder(
        handle_spec["stem_start_link2"],
        handle_spec["stem_end_link2"],
        radius,
        num_segments,
    )

    # The main lever is intentionally just a box. The geometry is simple on
    # purpose: point-cloud policies do not need a photorealistic handle.
    lever_start = handle_spec["lever_start_link2"]
    lever_tip = handle_spec["lever_tip_link2"]
    half_h = handle_spec.get("lever_half_height_m", radius)
    builder.add_box_bounds(
        [min(lever_start[0], lever_tip[0]), -half_h, lever_start[2] - half_h],
        [max(lever_start[0], lever_tip[0]), half_h, lever_start[2] + half_h],
    )

    if handle_spec["type"] == "return":
        builder.add_cylinder(
            handle_spec["lever_tip_link2"],
            handle_spec["return_tip_link2"],
            radius * 0.9,
            num_segments,
        )
    builder.write(mesh_path)


def build_handle_collision_primitives(handle_spec):
    radius = handle_spec["radius_m"]
    collisions = []
    if (
        handle_spec.get("has_bump")
        and handle_spec.get("bump_shape") == "cylinder"
        and handle_spec.get("bump_protrusion_m", 0.0) > 0.0
    ):
        # Only the round boss is a link_2 collision; the box plate is collision on link_1 (panel mesh).
        bump_radius = handle_spec["bump_radius_m"]
        collisions.append(
            {
                "name": "handle_bump",
                "origin_xyz": midpoint(handle_spec["bump_start_link2"], handle_spec["bump_end_link2"]),
                "size": [2.0 * bump_radius, 2.0 * bump_radius, handle_spec["bump_protrusion_m"]],
            }
        )
    collisions += [
        {
            "name": "handle_stem",
            "origin_xyz": midpoint(handle_spec["stem_start_link2"], handle_spec["stem_end_link2"]),
            "size": [2.0 * radius, 2.0 * radius, handle_spec["stem_length_m"]],
        },
        {
            "name": "handle_main_lever",
            "origin_xyz": midpoint(handle_spec["lever_start_link2"], handle_spec["lever_tip_link2"]),
            "size": [
                handle_spec["lever_length_m"],
                2.0 * handle_spec.get("lever_half_height_m", radius),
                2.0 * handle_spec.get("lever_half_height_m", radius),
            ],
        },
    ]
    if handle_spec["type"] == "return":
        collisions.append(
            {
                "name": "handle_return_hook",
                "origin_xyz": midpoint(handle_spec["lever_tip_link2"], handle_spec["return_tip_link2"]),
                "size": [2.0 * radius, 2.0 * radius, handle_spec["return_length_m"]],
            }
        )
    return collisions


def build_door_axis(spec):
    opening_direction = spec["opening_direction"]
    # Push/pull is defined from the visible handle side. If the generated handle
    # lives on the back face instead of the historical front face, flip the
    # joint_1 convention so the label semantics stay unchanged.
    if HANDLE_FACE_SIGN_LINK1 != FRONT_FACE_SIGN_LINK1:
        opening_direction = "push" if opening_direction == "pull" else "pull"

    if spec["hinge_side"] == "max" and opening_direction == "pull":
        return [0.0, -1.0, 0.0]
    if spec["hinge_side"] == "min" and opening_direction == "push":
        return [0.0, -1.0, 0.0]
    return [0.0, 1.0, 0.0]


def add_mesh_body(parent, tag, name, filename):
    body = ET.SubElement(parent, tag)
    if name is not None:
        body.attrib["name"] = name
    origin = ET.SubElement(body, "origin")
    origin.attrib["xyz"] = "0 0 0"
    origin.attrib["rpy"] = "0 0 0"
    geometry = ET.SubElement(body, "geometry")
    mesh = ET.SubElement(geometry, "mesh")
    mesh.attrib["filename"] = filename
    mesh.attrib["scale"] = "1 1 1"
    return body


def add_box_collision(link, name, center, size):
    collision = ET.SubElement(link, "collision")
    collision.attrib["name"] = name
    origin = ET.SubElement(collision, "origin")
    origin.attrib["xyz"] = format_vector(center)
    origin.attrib["rpy"] = "0 0 0"
    geometry = ET.SubElement(collision, "geometry")
    box = ET.SubElement(geometry, "box")
    box.attrib["size"] = format_vector(size)


def build_urdf_tree(handle_mesh_filename, handle_spec, collisions, door_axis, joint_1_origin):
    robot = ET.Element("robot")
    robot.attrib["name"] = "scratch-door"

    ET.SubElement(robot, "link", {"name": "base"})

    link_0 = ET.SubElement(robot, "link", {"name": "link_0"})
    add_mesh_body(link_0, "visual", "frame", "texture_dae/frame.obj")
    add_mesh_body(link_0, "collision", None, "texture_dae/frame.obj")

    joint_0 = ET.SubElement(robot, "joint", {"name": "joint_0", "type": "fixed"})
    # PartNet-style doors use a fixed root rotation because their local door
    # frame is y-up. Keep that convention here so the scratch assets stand
    # upright in Isaac's z-up world.
    ET.SubElement(joint_0, "origin", {"xyz": "0 0 0", "rpy": format_vector(ROOT_JOINT_RPY)})
    ET.SubElement(joint_0, "parent", {"link": "base"})
    ET.SubElement(joint_0, "child", {"link": "link_0"})

    link_1 = ET.SubElement(robot, "link", {"name": "link_1"})
    add_mesh_body(link_1, "visual", "board", "texture_dae/board.obj")
    add_mesh_body(link_1, "collision", None, "texture_dae/board.obj")

    joint_1 = ET.SubElement(robot, "joint", {"name": "joint_1", "type": "revolute"})
    ET.SubElement(joint_1, "origin", {"xyz": format_vector(joint_1_origin), "rpy": "0 0 0"})
    ET.SubElement(joint_1, "axis", {"xyz": format_vector(door_axis)})
    ET.SubElement(joint_1, "parent", {"link": "link_0"})
    ET.SubElement(joint_1, "child", {"link": "link_1"})
    ET.SubElement(joint_1, "limit", {"lower": "0", "upper": f"{DOOR_OPEN_LIMIT_RAD:.12f}"})

    link_2 = ET.SubElement(robot, "link", {"name": "link_2"})
    add_mesh_body(link_2, "visual", "handle", handle_mesh_filename)
    for collision in collisions:
        add_box_collision(link_2, collision["name"], collision["origin_xyz"], collision["size"])

    joint_2 = ET.SubElement(robot, "joint", {"name": "joint_2", "type": "revolute"})
    ET.SubElement(joint_2, "origin", {"xyz": format_vector(handle_spec["joint_origin_link1"]), "rpy": "0 0 0"})
    # Mirror the handle joint axis with the lever direction so positive joint
    # motion rotates the lever downward regardless of hinge / handle side.
    ET.SubElement(joint_2, "axis", {"xyz": format_vector(handle_spec["joint_axis_link1"])})
    ET.SubElement(joint_2, "parent", {"link": "link_1"})
    ET.SubElement(joint_2, "child", {"link": "link_2"})
    ET.SubElement(joint_2, "limit", {"lower": "0", "upper": f"{HANDLE_OPEN_LIMIT_RAD:.12f}"})
    return robot


def validate_variant(variant_dir, root):
    link_names = {link.attrib["name"] for link in root.findall("link")}
    joint_names = {joint.attrib["name"] for joint in root.findall("joint")}
    if not REQUIRED_LINK_NAMES.issubset(link_names):
        raise ValueError(f"Variant missing required links: {sorted(REQUIRED_LINK_NAMES - link_names)}")
    if not REQUIRED_JOINT_NAMES.issubset(joint_names):
        raise ValueError(f"Variant missing required joints: {sorted(REQUIRED_JOINT_NAMES - joint_names)}")

    for mesh in root.findall(".//mesh"):
        mesh_path = variant_dir / mesh.attrib["filename"]
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing mesh referenced by URDF: {mesh_path}")

    if root.find(".//link[@name='link_1']/collision") is None:
        raise ValueError("link_1 collision must exist")
    if root.find(".//link[@name='link_2']/visual") is None:
        raise ValueError("link_2 visual must exist")
    if len(root.findall(".//link[@name='link_2']/collision")) < 2:
        raise ValueError("link_2 should keep multiple simple collision primitives")


def write_variant_meta(meta_path, variant_name, spec, board_min, board_max, handle_spec, door_axis, collisions, initial_state):
    if spec["handle_side"] == "max":
        actual_edge_distance = board_max[0] - handle_spec["joint_origin_link1"][0]
    else:
        actual_edge_distance = handle_spec["joint_origin_link1"][0] - board_min[0]
    frame_height = spec["panel_height_m"] + spec["frame_clearance_m"] + spec["frame_head_height_m"]
    frame_panel_extra_height = frame_height - spec["panel_height_m"]
    actual_props = {
        "panel_width_m": spec["panel_width_m"],
        "panel_height_m": spec["panel_height_m"],
        "frame_height_m": frame_height,
        "frame_panel_extra_height_m": frame_panel_extra_height,
        "handle_height_m": handle_spec["joint_origin_link1"][1] - board_min[1],
        "handle_to_edge_distance_m": actual_edge_distance,
        "handle_side": spec["handle_side"],
        "opening_direction": spec["opening_direction"],
        "handle_attachment_joint": "joint_2",
        "board_min_x": board_min[0],
        "board_max_x": board_max[0],
        "board_min_y": board_min[1],
        "board_max_y": board_max[1],
    }
    target_props = {
        "panel_width_m": spec["panel_width_m"],
        "panel_height_m": spec["panel_height_m"],
        "handle_height_m": spec["handle_height_m"],
        "handle_to_edge_distance_m": spec["handle_edge_distance_m"],
        "flip_hinge_side": False,
        "opening_direction": spec["opening_direction"],
        "handle_side": spec["handle_side"],
    }
    source_props = {
        "panel_width_m": spec["panel_width_m"],
        "panel_height_m": spec["panel_height_m"],
        "panel_thickness_m": spec["panel_thickness_m"],
        "frame_height_m": frame_height,
        "frame_panel_extra_height_m": frame_panel_extra_height,
        "handle_height_m": actual_props["handle_height_m"],
        "handle_to_edge_distance_m": actual_props["handle_to_edge_distance_m"],
        "handle_side": actual_props["handle_side"],
        "opening_direction": actual_props["opening_direction"],
    }
    meta = {
        "generator": "generate_randomized_doors_scratch.py",
        "variant_name": variant_name,
        "source_asset": None,
        "direction_pair_key": build_direction_pair_key(variant_name),
        "target_properties": target_props,
        "actual_properties": actual_props,
        "source_properties": source_props,
        "initial_state": initial_state,
        "topology": {
            "links": ["base", "link_0", "link_1", "link_2"],
            "joints": ["joint_0", "joint_1", "joint_2"],
        },
        "panel": {
            "width_m": spec["panel_width_m"],
            "height_m": spec["panel_height_m"],
            "thickness_m": spec["panel_thickness_m"],
            "bbox_link1": board_min + board_max,
        },
        "frame": {
            "post_width_m": spec["frame_post_width_m"],
            "head_height_m": spec["frame_head_height_m"],
            "depth_m": spec["frame_depth_m"],
            "clearance_m": spec["frame_clearance_m"],
        },
        "door": {
            "hinge_side": spec["hinge_side"],
            "handle_side": spec["handle_side"],
            "opening_direction": spec["opening_direction"],
            "joint_1_origin_link0": build_joint_1_origin(spec),
            "joint_1_axis": door_axis,
        },
        "handle": {
            "type": handle_spec["type"],
            "mesh": f"texture_dae/handle_{handle_spec['type']}.obj",
            "joint_origin_link1": handle_spec["joint_origin_link1"],
            "joint_axis_link1": handle_spec["joint_axis_link1"],
            "lever_direction_link2": handle_spec["lever_direction_link2"],
            "return_direction_link2": handle_spec["return_direction_link2"],
            "radius_m": handle_spec["radius_m"],
            "lever_thickness_m": handle_spec.get("lever_thickness_m", 0.0),
            "stem_length_m": handle_spec["stem_length_m"],
            "panel_underside_gap_m": handle_spec.get("panel_underside_gap_m"),
            "plate_underside_gap_m": handle_spec.get("plate_underside_gap_m"),
            "lever_length_m": handle_spec["lever_length_m"],
            "return_length_m": handle_spec["return_length_m"],
            "return_tip_clearance_m": handle_spec["return_tip_clearance_m"],
            "has_bump": handle_spec.get("has_bump", False),
            "bump_shape": handle_spec.get("bump_shape", "none"),
            "bump_protrusion_m": handle_spec.get("bump_protrusion_m", 0.0),
            "bump_height_m": handle_spec.get("bump_height_m", 0.0),
            "bump_width_m": handle_spec.get("bump_width_m", 0.0),
            "bump_radius_m": handle_spec.get("bump_radius_m", 0.0),
            "plate_min_link1": handle_spec.get("plate_min_link1"),
            "plate_max_link1": handle_spec.get("plate_max_link1"),
            "collision_primitives": collisions,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, indent=2)


def build_variant_name(asset_name_prefix, index):
    return f"{asset_name_prefix}__rnd_{index:02d}"


def build_direction_pair_key(variant_name):
    # Match the shared metadata contract used by the other door generators.
    return str(variant_name)


def generate_variants(args):
    rng = random.Random(args.seed)
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite)

    generated = []
    return_handle_count = 0
    straight_handle_count = 0

    for variant_idx in range(args.num_variants):
        spec = sample_variant_spec(rng, args)
        variant_name = build_variant_name(args.asset_name_prefix, args.start_index + variant_idx)
        variant_dir = output_dir / variant_name
        texture_dir = variant_dir / "texture_dae"
        texture_dir.mkdir(parents=True, exist_ok=False)

        board_min, board_max = build_board_bounds(spec)
        frame_boxes = build_frame_boxes(spec)
        handle_spec = build_handle_spec(spec, board_min, board_max)
        collisions = build_handle_collision_primitives(handle_spec)
        door_axis = build_door_axis(spec)
        joint_1_origin = build_joint_1_origin(spec)
        initial_state = sample_initial_state(rng, variant_name, args)

        plate_box = None
        if handle_spec.get("plate_min_link1") is not None and handle_spec.get("plate_max_link1") is not None:
            plate_box = (handle_spec["plate_min_link1"], handle_spec["plate_max_link1"])
        write_board_mesh(texture_dir / "board.obj", board_min, board_max, plate_box=plate_box)
        write_frame_mesh(texture_dir / "frame.obj", frame_boxes)
        handle_mesh_name = f"handle_{handle_spec['type']}.obj"
        write_handle_mesh(texture_dir / handle_mesh_name, handle_spec, args.handle_num_segments)

        root = build_urdf_tree(
            handle_mesh_filename=f"texture_dae/{handle_mesh_name}",
            handle_spec=handle_spec,
            collisions=collisions,
            door_axis=door_axis,
            joint_1_origin=joint_1_origin,
        )
        ET.ElementTree(root).write(variant_dir / "mobility.urdf", encoding="utf-8", xml_declaration=True)
        validate_variant(variant_dir, root)
        write_variant_meta(
            variant_dir / "variant_meta.json",
            variant_name=variant_name,
            spec=spec,
            board_min=board_min,
            board_max=board_max,
            handle_spec=handle_spec,
            door_axis=door_axis,
            collisions=collisions,
            initial_state=initial_state,
        )

        if handle_spec["type"] == "return":
            return_handle_count += 1
        else:
            straight_handle_count += 1
        if variant_idx < args.debug_first_n:
            print(
                f"[debug] {variant_name}: hinge={spec['hinge_side']} open={spec['opening_direction']} "
                f"handle_side={spec['handle_side']} handle={handle_spec['type']} "
                f"board=({spec['panel_width_m']:.3f}, {spec['panel_height_m']:.3f}, {spec['panel_thickness_m']:.3f})"
            )
        generated.append(variant_dir)

    return {
        "generated": generated,
        "return_handle_count": return_handle_count,
        "straight_handle_count": straight_handle_count,
        "output_dir": output_dir,
    }


def main():
    args = parse_args()
    args.panel_width_range = resolve_range(args.panel_width_range, "panel_width_range")
    args.panel_height_range = resolve_range(args.panel_height_range, "panel_height_range")
    args.panel_thickness_range = resolve_range(args.panel_thickness_range, "panel_thickness_range")
    args.frame_post_width_range = resolve_range(args.frame_post_width_range, "frame_post_width_range")
    args.frame_head_height_range = resolve_range(args.frame_head_height_range, "frame_head_height_range")
    args.frame_depth_scale_range = resolve_range(args.frame_depth_scale_range, "frame_depth_scale_range")
    args.frame_clearance_range = resolve_range(args.frame_clearance_range, "frame_clearance_range")
    args.handle_height_range = resolve_range(args.handle_height_range, "handle_height_range")
    args.handle_edge_distance_range = resolve_range(args.handle_edge_distance_range, "handle_edge_distance_range")
    args.handle_radius_range = resolve_range(args.handle_radius_range, "handle_radius_range")
    args.handle_lever_thickness_range = resolve_range(
        args.handle_lever_thickness_range, "handle_lever_thickness_range"
    )
    args.handle_stem_length_range = resolve_range(args.handle_stem_length_range, "handle_stem_length_range")
    args.handle_length_range = resolve_range(args.handle_length_range, "handle_length_range")
    args.handle_return_length_range = resolve_range(args.handle_return_length_range, "handle_return_length_range")
    args.handle_return_tip_clearance_range = resolve_range(
        args.handle_return_tip_clearance_range, "handle_return_tip_clearance_range"
    )
    args.return_handle_prob = resolve_probability(args.return_handle_prob, "return_handle_prob")
    args.handle_bump_length_range = resolve_range(args.handle_bump_length_range, "handle_bump_length_range")
    args.handle_bump_height_range = resolve_range(args.handle_bump_height_range, "handle_bump_height_range")
    args.handle_bump_width_range = resolve_range(args.handle_bump_width_range, "handle_bump_width_range")
    args.handle_bump_radius_range = resolve_range(args.handle_bump_radius_range, "handle_bump_radius_range")
    args.handle_bump_prob = resolve_probability(args.handle_bump_prob, "handle_bump_prob")
    args.handle_num_segments = max(3, int(args.handle_num_segments))
    args.debug_first_n = max(0, int(args.debug_first_n))
    args.num_variants = max(1, int(args.num_variants))

    summary = generate_variants(args)
    print(f"Generated {len(summary['generated'])} scratch door variants.")
    print(f"Straight-handle variants: {summary['straight_handle_count']}")
    print(f"Return-handle variants: {summary['return_handle_count']}")
    print(f"Output directory: {summary['output_dir']}")
    print("If you overwrite existing assets, clear the IsaacLab URDF->USD cache or force reconversion.")
    if summary["generated"]:
        print("First few variants:")
        for variant_dir in summary["generated"][:5]:
            print(f"  - {variant_dir}")


if __name__ == "__main__":
    main()
