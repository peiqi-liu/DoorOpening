import argparse
import json
import math
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

"""
This script does not build a new door articulation from scratch.

What it currently does:
1. Load an existing simple PartNet-style door URDF.
2. Keep the original articulation topology and joint/link names.
3. Rescale the copied frame/panel to a new width/height.
4. Reposition the existing handle joint on the panel.
5. Optionally replace the original handle mesh with a procedurally generated
   return-lever mesh plus simple primitive collisions.

So the "randomization" here is still source-asset driven. The source mesh and
URDF provide the frame/panel layout, hinge placement, and base articulation.
Only some geometry and transforms are rewritten. If we want a truly from-scratch
generator later, that should probably live in a separate simpler script rather
than extending this clone-and-edit pipeline further.
"""

try:
    import trimesh
except ImportError:
    trimesh = None


DEFAULT_PANEL_WIDTH_RANGE_M = (0.8, 1.1)
# Despite the old name, this range is treated as the target overall door/frame
# height after scaling. That keeps unusual assets with transoms from ending up
# much taller in world space than the rest of the set.
DEFAULT_PANEL_HEIGHT_RANGE_M = (1.80, 2.10)
DEFAULT_HANDLE_HEIGHT_RANGE_M = (0.7, 0.9)
DEFAULT_HANDLE_EDGE_DISTANCE_RANGE_M = (0.05, 0.11)

DEFAULT_RETURN_HANDLE_PROB = 0.0
DEFAULT_HANDLE_LENGTH_RANGE_M = (0.10, 0.16)
DEFAULT_HANDLE_RADIUS_RANGE_M = (0.010, 0.018)
DEFAULT_HANDLE_HOOK_LENGTH_RANGE_M = (0.035, 0.075)
DEFAULT_HANDLE_STEM_LENGTH_RANGE_M = (0.025, 0.060)
DEFAULT_HANDLE_PLATE_PROB = 0.5
DEFAULT_HANDLE_PLATE_WIDTH_RANGE_M = (0.04, 0.08)
DEFAULT_HANDLE_PLATE_HEIGHT_RANGE_M = (0.10, 0.18)
DEFAULT_HANDLE_NUM_SEGMENTS = 16
DEFAULT_HANDLE_COLLISION_MODE = "simple_primitives"

DEFAULT_DEBUG_FIRST_N = 0

MIN_HANDLE_BOTTOM_CLEARANCE_M = 0.15
MIN_HANDLE_TOP_CLEARANCE_M = 0.15
MIN_HANDLE_EDGE_CLEARANCE_M = 0.02
MIN_VISUAL_THICKNESS_M = 0.005
REQUIRED_LINK_NAMES = {"base", "link_0", "link_1", "link_2"}
REQUIRED_JOINT_NAMES = {"joint_0", "joint_1", "joint_2"}


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    default_asset_root = repo_root / "source" / "DoorOpening" / "assets" / "door" / "PartNetv4"
    default_output_dir = default_asset_root / "generated_randomized"

    parser = argparse.ArgumentParser(
        description="Create randomized 3-link / 2-joint door assets by cloning simple PartNetv4 doors."
    )
    parser.add_argument("--asset-root", type=Path, default=default_asset_root)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--variants-per-source", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--flip-hinge-side",
        action="store_true",
        help="Mirror the panel to the opposite hinge edge while keeping the handle on the free edge.",
    )
    parser.add_argument(
        "--opening-direction",
        choices=("pull", "push"),
        default="pull",
        help="Choose whether the generated door opens as a pull door or a push door.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument("--return-handle-prob", type=float, default=DEFAULT_RETURN_HANDLE_PROB)
    parser.add_argument(
        "--handle-length-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_LENGTH_RANGE_M,
    )
    parser.add_argument(
        "--handle-radius-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_RADIUS_RANGE_M,
    )
    parser.add_argument(
        "--handle-hook-length-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_HOOK_LENGTH_RANGE_M,
    )
    parser.add_argument(
        "--handle-stem-length-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_STEM_LENGTH_RANGE_M,
    )
    parser.add_argument("--handle-plate-prob", type=float, default=DEFAULT_HANDLE_PLATE_PROB)
    parser.add_argument(
        "--handle-plate-width-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_PLATE_WIDTH_RANGE_M,
    )
    parser.add_argument(
        "--handle-plate-height-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_HANDLE_PLATE_HEIGHT_RANGE_M,
    )
    parser.add_argument("--handle-num-segments", type=int, default=DEFAULT_HANDLE_NUM_SEGMENTS)
    parser.add_argument(
        "--handle-collision-mode",
        choices=(DEFAULT_HANDLE_COLLISION_MODE,),
        default=DEFAULT_HANDLE_COLLISION_MODE,
    )
    parser.add_argument("--debug-first-n", type=int, default=DEFAULT_DEBUG_FIRST_N)
    return parser.parse_args()


def parse_vector(text, default=(0.0, 0.0, 0.0)):
    if text is None:
        return list(default)
    return [float(value) for value in text.split()]


def format_vector(values):
    return " ".join(f"{value:.6f}" for value in values)


def format_scalar(value):
    return f"{float(value):.6f}"


def resolve_range(args, attr_name, default):
    values = getattr(args, attr_name, default)
    low = float(values[0])
    high = float(values[1])
    if high < low:
        raise ValueError(f"{attr_name} must satisfy min <= max, got {values}")
    return low, high


def resolve_probability(args, attr_name, default):
    value = float(getattr(args, attr_name, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{attr_name} must be in [0, 1], got {value}")
    return value


def sample_uniform(rng, value_range):
    return rng.uniform(float(value_range[0]), float(value_range[1]))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def vector_length(vector):
    return math.sqrt(dot(vector, vector))


def normalize(vector):
    length = vector_length(vector)
    if length <= 1e-12:
        raise ValueError(f"Cannot normalize near-zero vector {vector}")
    return [value / length for value in vector]


def add_vectors(a, b):
    return [x + y for x, y in zip(a, b)]


def subtract_vectors(a, b):
    return [x - y for x, y in zip(a, b)]


def scale_vector(vector, scalar):
    return [scalar * value for value in vector]


def midpoint(a, b):
    return [(x + y) * 0.5 for x, y in zip(a, b)]


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

    def add_box_center_size(self, center, size):
        half = [0.5 * value for value in size]
        self.add_box_bounds(subtract_vectors(center, half), add_vectors(center, half))

    def add_cylinder(self, start, end, radius, num_segments):
        axis = subtract_vectors(end, start)
        length = vector_length(axis)
        if length <= 1e-9 or radius <= 1e-9:
            return

        axis_dir = [value / length for value in axis]
        if abs(axis_dir[0]) < 0.9:
            reference = [1.0, 0.0, 0.0]
        else:
            reference = [0.0, 1.0, 0.0]
        u = normalize(cross(axis_dir, reference))
        v = normalize(cross(axis_dir, u))

        base = len(self.vertices) + 1
        start_ring = []
        end_ring = []
        for segment_idx in range(max(3, num_segments)):
            theta = 2.0 * math.pi * segment_idx / max(3, num_segments)
            radial = add_vectors(
                scale_vector(u, radius * math.cos(theta)),
                scale_vector(v, radius * math.sin(theta)),
            )
            start_ring.append(add_vectors(start, radial))
            end_ring.append(add_vectors(end, radial))
        self.vertices.extend(start_ring)
        self.vertices.extend(end_ring)
        self.vertices.append(list(start))
        self.vertices.append(list(end))

        ring_size = len(start_ring)
        start_center_index = base + 2 * ring_size
        end_center_index = start_center_index + 1

        for segment_idx in range(ring_size):
            next_idx = (segment_idx + 1) % ring_size
            s0 = base + segment_idx
            s1 = base + next_idx
            e0 = base + ring_size + segment_idx
            e1 = base + ring_size + next_idx

            self.faces.append([s0, s1, e1])
            self.faces.append([s0, e1, e0])
            self.faces.append([start_center_index, s1, s0])
            self.faces.append([end_center_index, e0, e1])

    def write(self, output_path):
        with open(output_path, "w", encoding="utf-8") as obj_file:
            for vertex in self.vertices:
                obj_file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for face in self.faces:
                obj_file.write(f"f {face[0]} {face[1]} {face[2]}\n")


def load_obj_bounds(mesh_path, bounds_cache):
    mesh_path = mesh_path.resolve()
    cached = bounds_cache.get(mesh_path)
    if cached is not None:
        return cached

    min_bounds = [float("inf")] * 3
    max_bounds = [float("-inf")] * 3
    vertex_count = 0

    with open(mesh_path, encoding="utf-8", errors="ignore") as mesh_file:
        for line in mesh_file:
            if not line.startswith("v "):
                continue

            _, x, y, z, *_ = line.split()
            values = [float(x), float(y), float(z)]
            for axis, value in enumerate(values):
                min_bounds[axis] = min(min_bounds[axis], value)
                max_bounds[axis] = max(max_bounds[axis], value)
            vertex_count += 1

    if vertex_count == 0:
        raise ValueError(f"No OBJ vertices found in {mesh_path}")

    bounds_cache[mesh_path] = (min_bounds, max_bounds)
    return min_bounds, max_bounds


def get_link(root, link_name):
    link = root.find(f".//link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"Missing link {link_name}")
    return link


def get_link_visual(root, link_name):
    visual = root.find(f".//link[@name='{link_name}']/visual")
    if visual is None:
        raise ValueError(f"Missing visual for {link_name}")
    return visual


def get_joint(root, joint_name):
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"Missing joint {joint_name}")
    return joint


def find_joint(root, joint_name):
    return root.find(f".//joint[@name='{joint_name}']")


def get_mesh_and_origin(body_element):
    origin = body_element.find("origin")
    if origin is None:
        raise ValueError("Expected body element to contain an origin")

    mesh = body_element.find("./geometry/mesh")
    if mesh is None:
        raise ValueError("Expected body element to contain a mesh")
    return mesh, origin


def iter_link_bodies(root, link_name, tags=("visual", "collision")):
    link = root.find(f".//link[@name='{link_name}']")
    if link is None:
        return
    for tag in tags:
        for body in link.findall(tag):
            yield tag, body


def iter_mesh_bodies(root, link_name, tags=("visual", "collision")):
    for tag, body in iter_link_bodies(root, link_name, tags=tags):
        mesh = body.find("./geometry/mesh")
        origin = body.find("origin")
        if mesh is None or origin is None:
            continue
        yield tag, body, mesh, origin


def get_board_geometry(asset_dir, root, bounds_cache):
    # We measure the panel in link_1 local coordinates after the URDF mesh scale
    # and origin are applied. Those transformed bounds are what the simulator sees.
    visual = get_link_visual(root, "link_1")
    mesh, origin = get_mesh_and_origin(visual)
    mesh_scale = parse_vector(mesh.attrib.get("scale"), default=(1.0, 1.0, 1.0))
    mesh_origin = parse_vector(origin.attrib.get("xyz"))
    raw_min, raw_max = load_obj_bounds(asset_dir / mesh.attrib["filename"], bounds_cache)

    board_min = [mesh_origin[i] + raw_min[i] * mesh_scale[i] for i in range(3)]
    board_max = [mesh_origin[i] + raw_max[i] * mesh_scale[i] for i in range(3)]
    return {
        "raw_min": raw_min,
        "raw_max": raw_max,
        "origin": mesh_origin,
        "scale": mesh_scale,
        "min": board_min,
        "max": board_max,
        "width": board_max[0] - board_min[0],
        "height": board_max[1] - board_min[1],
        "thickness": board_max[2] - board_min[2],
    }


def get_frame_geometry(asset_dir, root, bounds_cache):
    visual = get_link_visual(root, "link_0")
    mesh, origin = get_mesh_and_origin(visual)
    mesh_scale = parse_vector(mesh.attrib.get("scale"), default=(1.0, 1.0, 1.0))
    mesh_origin = parse_vector(origin.attrib.get("xyz"))
    raw_min, raw_max = load_obj_bounds(asset_dir / mesh.attrib["filename"], bounds_cache)

    frame_min = [mesh_origin[i] + raw_min[i] * mesh_scale[i] for i in range(3)]
    frame_max = [mesh_origin[i] + raw_max[i] * mesh_scale[i] for i in range(3)]
    return {
        "min": frame_min,
        "max": frame_max,
        "width": frame_max[0] - frame_min[0],
        "height": frame_max[1] - frame_min[1],
    }


def get_handle_attachment_joint_name(root):
    # Some assets attach the handle directly to link_1 through joint_2, while
    # others insert an extra fixed lock link (link_3/joint_3). For randomization
    # we only care about the final handle anchor on the panel.
    return "joint_3" if find_joint(root, "joint_3") is not None else "joint_2"


def get_handle_joint_origin_in_panel_frame(root):
    handle_joint = get_joint(root, get_handle_attachment_joint_name(root))
    origin = handle_joint.find("origin")
    if origin is None:
        raise ValueError("Handle attachment joint is missing an origin")
    return parse_vector(origin.attrib.get("xyz"))


def get_door_properties(asset_dir, root, bounds_cache):
    board = get_board_geometry(asset_dir, root, bounds_cache)
    frame = get_frame_geometry(asset_dir, root, bounds_cache)
    handle_joint_origin = get_handle_joint_origin_in_panel_frame(root)

    distance_to_min = handle_joint_origin[0] - board["min"][0]
    distance_to_max = board["max"][0] - handle_joint_origin[0]
    handle_side = "max" if distance_to_max <= distance_to_min else "min"
    edge_distance = distance_to_max if handle_side == "max" else distance_to_min

    return {
        "panel_width_m": board["width"],
        "panel_height_m": board["height"],
        "frame_height_m": frame["height"],
        "frame_panel_extra_height_m": frame["height"] - board["height"],
        "handle_height_m": handle_joint_origin[1] - board["min"][1],
        "handle_to_edge_distance_m": edge_distance,
        "handle_side": handle_side,
        "handle_attachment_joint": get_handle_attachment_joint_name(root),
        "board_min_x": board["min"][0],
        "board_max_x": board["max"][0],
        "board_min_y": board["min"][1],
        "board_max_y": board["max"][1],
        "board_min_z": board["min"][2],
        "board_max_z": board["max"][2],
    }


def is_supported_source_asset(urdf_path):
    root = ET.parse(urdf_path).getroot()
    link_names = {link.attrib.get("name") for link in root.findall("link")}
    joint_names = {joint.attrib.get("name") for joint in root.findall("joint")}

    if not REQUIRED_LINK_NAMES.issubset(link_names):
        return False
    if not REQUIRED_JOINT_NAMES.issubset(joint_names):
        return False

    joint_2_parent = root.find(".//joint[@name='joint_2']/parent")
    if joint_2_parent is None:
        return False

    joint_2_parent_link = joint_2_parent.attrib.get("link")
    is_supported_topology = joint_2_parent_link == "link_1"

    # Lock-style assets insert link_3/joint_3 between the panel and the handle.
    joint_3 = find_joint(root, "joint_3")
    if joint_2_parent_link == "link_3" and joint_3 is not None:
        joint_3_parent = joint_3.find("parent")
        is_supported_topology = (
            joint_3_parent is not None and joint_3_parent.attrib.get("link") == "link_1"
        )

    return bool(is_supported_topology)


def iter_supported_source_assets(asset_root):
    for urdf_path in sorted(asset_root.glob("*/mobility.urdf")):
        if is_supported_source_asset(urdf_path):
            yield urdf_path.parent


def scale_link_xy(root, link_name, sx, sy):
    for _, _, mesh, origin in iter_mesh_bodies(root, link_name):
        scale = parse_vector(mesh.attrib.get("scale"), default=(1.0, 1.0, 1.0))
        # In these door URDFs x is panel width and y is panel height.
        scale[0] *= sx
        scale[1] *= sy
        mesh.attrib["scale"] = format_vector(scale)

        origin_xyz = parse_vector(origin.attrib.get("xyz"))
        # The panel/frame meshes are already offset from the link frame, so the
        # origin must be scaled together with the mesh to preserve alignment.
        origin_xyz[0] *= sx
        origin_xyz[1] *= sy
        origin.attrib["xyz"] = format_vector(origin_xyz)


def scale_joint_origin_xy(root, joint_name, sx, sy):
    origin = root.find(f".//joint[@name='{joint_name}']/origin")
    if origin is None:
        return

    origin_xyz = parse_vector(origin.attrib.get("xyz"))
    origin_xyz[0] *= sx
    origin_xyz[1] *= sy
    origin.attrib["xyz"] = format_vector(origin_xyz)


def mirror_link_origin_x(root, link_name):
    for _, body in iter_link_bodies(root, link_name):
        origin = body.find("origin")
        if origin is None:
            continue
        origin_xyz = parse_vector(origin.attrib.get("xyz"))
        origin_xyz[0] *= -1.0
        origin.attrib["xyz"] = format_vector(origin_xyz)


def mirror_joint_origin_x(root, joint_name):
    origin = root.find(f".//joint[@name='{joint_name}']/origin")
    if origin is None:
        return

    origin_xyz = parse_vector(origin.attrib.get("xyz"))
    origin_xyz[0] *= -1.0
    origin.attrib["xyz"] = format_vector(origin_xyz)


def negate_joint_axis(root, joint_name):
    axis = root.find(f".//joint[@name='{joint_name}']/axis")
    if axis is None:
        return

    axis_xyz = parse_vector(axis.attrib.get("xyz"))
    axis_xyz = [-value for value in axis_xyz]
    axis.attrib["xyz"] = format_vector(axis_xyz)


def clamp_handle_height(target_height, desired_handle_height):
    low = max(MIN_HANDLE_BOTTOM_CLEARANCE_M, 0.0)
    high = max(low, target_height - MIN_HANDLE_TOP_CLEARANCE_M)
    return min(max(desired_handle_height, low), high)


def clamp_edge_distance(target_width, desired_edge_distance):
    low = MIN_HANDLE_EDGE_CLEARANCE_M
    high = max(low, target_width - MIN_HANDLE_EDGE_CLEARANCE_M)
    return min(max(desired_edge_distance, low), high)


def apply_variant_to_root(asset_dir, root, target_props, bounds_cache, source_props=None):
    if source_props is None:
        source_props = get_door_properties(asset_dir, root, bounds_cache)
    sx = target_props["panel_width_m"] / source_props["panel_width_m"]
    sy = target_props["panel_height_m"] / source_props["panel_height_m"]

    # Scale frame and panel together so the articulation layout stays coherent.
    scale_link_xy(root, "link_0", sx, sy)
    scale_link_xy(root, "link_1", sx, sy)
    scale_joint_origin_xy(root, "joint_1", sx, sy)

    handle_side = source_props["handle_side"]
    if target_props.get("flip_hinge_side"):
        # Mirroring link_1 around its hinge axis moves the panel to the other
        # side of the opening. joint_1 is mirrored too so the hinge sits on the
        # opposite jamb, and the handle side is flipped to stay off the hinge.
        mirror_link_origin_x(root, "link_1")
        mirror_joint_origin_x(root, "joint_1")
        # The mirrored handle mesh would otherwise rotate in the opposite
        # direction, so flip the lever joint axis as well.
        negate_joint_axis(root, "joint_2")
        handle_side = "min" if handle_side == "max" else "max"

    opening_direction = target_props.get("opening_direction", "pull")
    should_flip_open_axis = target_props.get("flip_hinge_side", False) != (opening_direction == "push")
    if should_flip_open_axis:
        # All current PartNetv4 sources are pull doors. Negating joint_1 lets us
        # choose push vs pull independently from the hinge side.
        negate_joint_axis(root, "joint_1")

    board = get_board_geometry(asset_dir, root, bounds_cache)
    clamped_handle_height = clamp_handle_height(board["height"], target_props["handle_height_m"])
    clamped_edge_distance = clamp_edge_distance(board["width"], target_props["handle_to_edge_distance_m"])

    handle_joint_name = source_props["handle_attachment_joint"]
    handle_joint_origin = root.find(f".//joint[@name='{handle_joint_name}']/origin")
    if handle_joint_origin is None:
        raise ValueError(f"Missing {handle_joint_name} origin")
    handle_joint_xyz = parse_vector(handle_joint_origin.attrib.get("xyz"))

    # Reposition the joint that actually anchors the handle assembly to the
    # panel. When a lock link exists, that is joint_3; otherwise it is joint_2.
    if handle_side == "max":
        handle_joint_xyz[0] = board["max"][0] - clamped_edge_distance
    else:
        handle_joint_xyz[0] = board["min"][0] + clamped_edge_distance
    handle_joint_xyz[1] = board["min"][1] + clamped_handle_height
    handle_joint_origin.attrib["xyz"] = format_vector(handle_joint_xyz)

    actual_props = get_door_properties(asset_dir, root, bounds_cache)
    return source_props, actual_props


def mirror_obj_file_x(source_path, mirrored_path):
    with open(source_path, encoding="utf-8", errors="ignore") as src_file:
        lines = src_file.readlines()

    with open(mirrored_path, "w", encoding="utf-8") as dst_file:
        for line in lines:
            if line.startswith("v "):
                tokens = line.rstrip("\n").split()
                tokens[1] = str(-float(tokens[1]))
                dst_file.write(" ".join(tokens) + "\n")
                continue

            if line.startswith("vn "):
                tokens = line.rstrip("\n").split()
                tokens[1] = str(-float(tokens[1]))
                dst_file.write(" ".join(tokens) + "\n")
                continue

            if line.startswith("f "):
                tokens = line.rstrip("\n").split()
                dst_file.write(" ".join([tokens[0], *reversed(tokens[1:])]) + "\n")
                continue

            dst_file.write(line)


def ensure_mirrored_mesh(mesh_path, mirrored_cache):
    mesh_path = mesh_path.resolve()
    cached = mirrored_cache.get(mesh_path)
    if cached is not None:
        return cached

    if mesh_path.suffix.lower() != ".obj":
        raise ValueError(f"Expected OBJ mesh for hinge flip, got {mesh_path}")

    mirrored_path = mesh_path.with_name(f"{mesh_path.stem}__mirrored_x{mesh_path.suffix}")
    if not mirrored_path.exists():
        mirror_obj_file_x(mesh_path, mirrored_path)

    mirrored_cache[mesh_path] = mirrored_path
    return mirrored_path


def mirror_link_meshes_x(root, variant_dir, link_name, mirrored_cache):
    for _, _, mesh, _ in iter_mesh_bodies(root, link_name):
        mesh_path = (variant_dir / mesh.attrib["filename"]).resolve()
        mirrored_path = ensure_mirrored_mesh(mesh_path, mirrored_cache)
        mesh.attrib["filename"] = mirrored_path.relative_to(variant_dir).as_posix()


def apply_flipped_mesh_variants(root, variant_dir, link_names=None):
    mirrored_cache = {}
    # When we flip hinge side, these meshes should look mirrored too rather than
    # only moving the joints. That includes the frame, panel, and any remaining
    # source-mesh handle or lock geometry.
    if link_names is None:
        link_names = ("link_0", "link_1", "link_2", "link_3")
    for link_name in link_names:
        mirror_link_meshes_x(root, variant_dir, link_name, mirrored_cache)


def attach_variant_files(source_dir, variant_dir):
    texture_src = source_dir / "texture_dae"
    texture_dst = variant_dir / "texture_dae"
    texture_dst.mkdir(parents=True, exist_ok=True)
    if texture_src.exists():
        shutil.copytree(texture_src, texture_dst, dirs_exist_ok=True)


def sample_target_panel_height(rng, source_props):
    target_frame_height = rng.uniform(*DEFAULT_PANEL_HEIGHT_RANGE_M)
    source_frame_height = source_props["frame_height_m"]
    if source_frame_height <= 0.0:
        raise ValueError("Source frame height must be positive")

    # Frame and panel are scaled by the same y factor, so to hit a desired
    # overall door height we convert that frame target back into the equivalent
    # panel height for this specific source asset.
    return source_props["panel_height_m"] * (target_frame_height / source_frame_height)


def sample_target_properties(rng, source_props):
    panel_width = rng.uniform(*DEFAULT_PANEL_WIDTH_RANGE_M)
    panel_height = sample_target_panel_height(rng, source_props)
    handle_height = rng.uniform(*DEFAULT_HANDLE_HEIGHT_RANGE_M)
    handle_edge_distance = rng.uniform(*DEFAULT_HANDLE_EDGE_DISTANCE_RANGE_M)
    return {
        "panel_width_m": panel_width,
        "panel_height_m": panel_height,
        "handle_height_m": handle_height,
        "handle_to_edge_distance_m": handle_edge_distance,
        "flip_hinge_side": False,
        "opening_direction": "pull",
    }


def prepare_output_dir(output_dir, overwrite):
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Allow writing variants into an existing asset root such as PartNetv5.
            # Individual variant folders are still created with exist_ok=False, so
            # we won't silently overwrite an existing generated asset.
            if not output_dir.is_dir():
                raise FileExistsError(f"{output_dir} exists and is not a directory.")
            return
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def build_variant_name(source_asset_name, variant_idx):
    return f"{source_asset_name}__rnd_{variant_idx:02d}"


def build_direction_pair_key(variant_name):
    # Pair only the push/pull counterparts of the same variant. Handle-pose
    # changes across different variants should not force shared start poses.
    return str(variant_name)


def build_default_feature_config(args):
    # Collect all handle-generation knobs into one plain dict so the rest of the
    # script can stay independent from argparse and be called programmatically.
    return {
        "return_handle_prob": resolve_probability(args, "return_handle_prob", DEFAULT_RETURN_HANDLE_PROB),
        "handle_length_range": resolve_range(args, "handle_length_range", DEFAULT_HANDLE_LENGTH_RANGE_M),
        "handle_radius_range": resolve_range(args, "handle_radius_range", DEFAULT_HANDLE_RADIUS_RANGE_M),
        "handle_hook_length_range": resolve_range(args, "handle_hook_length_range", DEFAULT_HANDLE_HOOK_LENGTH_RANGE_M),
        "handle_stem_length_range": resolve_range(args, "handle_stem_length_range", DEFAULT_HANDLE_STEM_LENGTH_RANGE_M),
        "handle_plate_prob": resolve_probability(args, "handle_plate_prob", DEFAULT_HANDLE_PLATE_PROB),
        "handle_plate_width_range": resolve_range(args, "handle_plate_width_range", DEFAULT_HANDLE_PLATE_WIDTH_RANGE_M),
        "handle_plate_height_range": resolve_range(args, "handle_plate_height_range", DEFAULT_HANDLE_PLATE_HEIGHT_RANGE_M),
        "handle_num_segments": max(3, int(getattr(args, "handle_num_segments", DEFAULT_HANDLE_NUM_SEGMENTS))),
        "handle_collision_mode": str(getattr(args, "handle_collision_mode", DEFAULT_HANDLE_COLLISION_MODE)),
        "debug_first_n": max(0, int(getattr(args, "debug_first_n", DEFAULT_DEBUG_FIRST_N))),
    }


def ensure_link_visual_mesh(root, link_name, mesh_filename, visual_name=None):
    # For generated handles we replace the link visual outright instead of
    # trying to preserve the source mesh origin/scale convention.
    visual = get_link_visual(root, link_name)
    if visual_name is not None:
        visual.attrib["name"] = visual_name
    origin = visual.find("origin")
    if origin is None:
        origin = ET.SubElement(visual, "origin")
    origin.attrib["xyz"] = "0 0 0"
    origin.attrib["rpy"] = "0 0 0"

    geometry = visual.find("geometry")
    if geometry is None:
        geometry = ET.SubElement(visual, "geometry")
    for child in list(geometry):
        geometry.remove(child)
    mesh = ET.SubElement(geometry, "mesh")
    mesh.attrib["filename"] = mesh_filename
    mesh.attrib["scale"] = "1 1 1"


def clear_link_collisions(root, link_name):
    link = get_link(root, link_name)
    for collision in list(link.findall("collision")):
        link.remove(collision)
    return link


def add_box_collision(link, name, center, size):
    collision = ET.SubElement(link, "collision")
    if name:
        collision.attrib["name"] = name
    origin = ET.SubElement(collision, "origin")
    origin.attrib["xyz"] = format_vector(center)
    origin.attrib["rpy"] = "0 0 0"
    geometry = ET.SubElement(collision, "geometry")
    box = ET.SubElement(geometry, "box")
    box.attrib["size"] = format_vector(size)
    return collision


def sample_handle_shape(rng, feature_cfg, board, handle_joint_origin_link1):
    # This function only decides a procedural handle layout in link_2 local
    # coordinates. It does not touch the URDF directly.
    plate_enabled = rng.random() < feature_cfg["handle_plate_prob"]
    length = sample_uniform(rng, feature_cfg["handle_length_range"])
    radius = sample_uniform(rng, feature_cfg["handle_radius_range"])
    hook_length = sample_uniform(rng, feature_cfg["handle_hook_length_range"])
    stem_length = sample_uniform(rng, feature_cfg["handle_stem_length_range"])
    plate_width = sample_uniform(rng, feature_cfg["handle_plate_width_range"])
    plate_height = sample_uniform(rng, feature_cfg["handle_plate_height_range"])
    plate_thickness = max(MIN_VISUAL_THICKNESS_M, radius * 0.55)

    distance_to_min = abs(handle_joint_origin_link1[0] - board["min"][0])
    distance_to_max = abs(board["max"][0] - handle_joint_origin_link1[0])
    # The lever should always point away from the hinge and toward the free
    # interior of the panel, regardless of which side the handle is on.
    lever_direction = [-1.0, 0.0, 0.0] if distance_to_max <= distance_to_min else [1.0, 0.0, 0.0]

    # The handle anchor lies on one door face in link_1. Use that sign to keep
    # the short hook returning toward the panel instead of further away from it.
    handle_face_sign = 1.0 if handle_joint_origin_link1[2] < 0.0 else -1.0
    hook_direction = [0.0, 0.0, handle_face_sign]
    outward_direction = [0.0, 0.0, -handle_face_sign]

    stem_start = [0.0, 0.0, 0.0]
    stem_end = scale_vector(outward_direction, stem_length)
    lever_start = list(stem_end)
    lever_tip = add_vectors(lever_start, scale_vector(lever_direction, length))
    hook_tip = add_vectors(lever_tip, scale_vector(hook_direction, hook_length))

    return {
        "type": "procedural_return_lever",
        "length": length,
        "radius": radius,
        "hook_length": hook_length,
        "stem_length": stem_length,
        "plate_enabled": plate_enabled,
        "plate_width": plate_width,
        "plate_height": plate_height,
        "plate_thickness": plate_thickness,
        "lever_direction_link2": lever_direction,
        "hook_direction_link2": hook_direction,
        "outward_direction_link2": outward_direction,
        "stem_start_link2": stem_start,
        "stem_end_link2": stem_end,
        "lever_start_link2": lever_start,
        "lever_tip_link2": lever_tip,
        "hook_tip_link2": hook_tip,
    }


def maybe_build_trimesh_handle(shape_cfg, num_segments):
    if trimesh is None:
        return None
    try:
        # The visual handle is just a few analytic pieces stitched together:
        # stem from the joint, straight lever, short return hook, optional plate.
        meshes = []
        radius = shape_cfg["radius"]
        meshes.append(
            trimesh.creation.cylinder(
                radius=radius,
                segment=[shape_cfg["stem_start_link2"], shape_cfg["stem_end_link2"]],
                sections=num_segments,
            )
        )
        meshes.append(
            trimesh.creation.cylinder(
                radius=radius,
                segment=[shape_cfg["lever_start_link2"], shape_cfg["lever_tip_link2"]],
                sections=num_segments,
            )
        )
        meshes.append(
            trimesh.creation.cylinder(
                radius=radius * 0.92,
                segment=[shape_cfg["lever_tip_link2"], shape_cfg["hook_tip_link2"]],
                sections=num_segments,
            )
        )
        if shape_cfg["plate_enabled"]:
            plate = trimesh.creation.box(
                extents=[
                    shape_cfg["plate_width"],
                    shape_cfg["plate_height"],
                    shape_cfg["plate_thickness"],
                ]
            )
            plate.apply_translation(scale_vector(shape_cfg["hook_direction_link2"], 0.5 * shape_cfg["plate_thickness"]))
            meshes.append(plate)
        combined = trimesh.util.concatenate(meshes)
        return combined
    except Exception as exc:
        print(f"Warning: trimesh handle generation failed, falling back to manual OBJ writer: {exc}")
        return None


def write_handle_mesh(texture_dir, variant_serial, shape_cfg, num_segments):
    mesh_name = f"handle_return_{variant_serial:06d}.obj"
    mesh_path = texture_dir / mesh_name

    tri_mesh = maybe_build_trimesh_handle(shape_cfg, num_segments)
    if tri_mesh is not None:
        tri_mesh.export(mesh_path)
        return mesh_name

    # Fallback path when trimesh is unavailable: write the same coarse geometry
    # directly as an OBJ using our tiny local mesh builder.
    builder = ObjMeshBuilder()
    builder.add_cylinder(
        shape_cfg["stem_start_link2"],
        shape_cfg["stem_end_link2"],
        shape_cfg["radius"],
        num_segments,
    )
    builder.add_cylinder(
        shape_cfg["lever_start_link2"],
        shape_cfg["lever_tip_link2"],
        shape_cfg["radius"],
        num_segments,
    )
    builder.add_cylinder(
        shape_cfg["lever_tip_link2"],
        shape_cfg["hook_tip_link2"],
        shape_cfg["radius"] * 0.92,
        num_segments,
    )
    if shape_cfg["plate_enabled"]:
        plate_center = scale_vector(shape_cfg["hook_direction_link2"], 0.5 * shape_cfg["plate_thickness"])
        builder.add_box_center_size(
            plate_center,
            [shape_cfg["plate_width"], shape_cfg["plate_height"], shape_cfg["plate_thickness"]],
        )
    builder.write(mesh_path)
    return mesh_name


def build_handle_collision_primitives(shape_cfg, collision_mode):
    if collision_mode != DEFAULT_HANDLE_COLLISION_MODE:
        raise ValueError(f"Unsupported handle collision mode: {collision_mode}")

    # We intentionally use coarse boxes for collisions. For point-cloud policies
    # the visual shape matters more than exact contact on the handle mesh, and
    # simpler collisions are usually more stable in Isaac/PhysX.
    radius = shape_cfg["radius"]
    stem_center = midpoint(shape_cfg["stem_start_link2"], shape_cfg["stem_end_link2"])
    lever_center = midpoint(shape_cfg["lever_start_link2"], shape_cfg["lever_tip_link2"])
    hook_center = midpoint(shape_cfg["lever_tip_link2"], shape_cfg["hook_tip_link2"])

    primitives = [
        {
            "name": "handle_stem",
            "type": "box",
            "size": [2.0 * radius, 2.0 * radius, shape_cfg["stem_length"]],
            "origin_xyz": stem_center,
        },
        {
            "name": "handle_main_lever",
            "type": "box",
            "size": [shape_cfg["length"], 2.0 * radius, 2.0 * radius],
            "origin_xyz": lever_center,
        },
        {
            "name": "handle_return_hook",
            "type": "box",
            "size": [2.0 * radius, 2.0 * radius, shape_cfg["hook_length"]],
            "origin_xyz": hook_center,
        },
    ]
    if shape_cfg["plate_enabled"]:
        primitives.append(
            {
                "name": "handle_plate",
                "type": "box",
                "size": [
                    shape_cfg["plate_width"],
                    shape_cfg["plate_height"],
                    shape_cfg["plate_thickness"],
                ],
                "origin_xyz": scale_vector(shape_cfg["hook_direction_link2"], 0.5 * shape_cfg["plate_thickness"]),
            }
        )
    return primitives


def apply_handle_variant(root, variant_dir, board, feature_cfg, rng, variant_serial):
    if rng.random() >= feature_cfg["return_handle_prob"]:
        return {"enabled": False}

    # At this point the articulation already exists. We only swap link_2's
    # visual mesh and rewrite its collisions; we do not add links or joints.
    texture_dir = variant_dir / "texture_dae"
    texture_dir.mkdir(parents=True, exist_ok=True)
    handle_joint_origin_link1 = get_handle_joint_origin_in_panel_frame(root)
    shape_cfg = sample_handle_shape(rng, feature_cfg, board, handle_joint_origin_link1)
    mesh_name = write_handle_mesh(texture_dir, variant_serial, shape_cfg, feature_cfg["handle_num_segments"])
    mesh_rel = f"texture_dae/{mesh_name}"
    ensure_link_visual_mesh(root, "link_2", mesh_rel, visual_name="handle")

    collision_primitives = build_handle_collision_primitives(shape_cfg, feature_cfg["handle_collision_mode"])
    link = clear_link_collisions(root, "link_2")
    for primitive in collision_primitives:
        add_box_collision(link, primitive["name"], primitive["origin_xyz"], primitive["size"])

    return {
        "enabled": True,
        "type": "procedural_return_lever",
        "mesh": mesh_rel,
        "length": shape_cfg["length"],
        "radius": shape_cfg["radius"],
        "hook_length": shape_cfg["hook_length"],
        "stem_length": shape_cfg["stem_length"],
        "plate_enabled": shape_cfg["plate_enabled"],
        "plate_width": shape_cfg["plate_width"],
        "plate_height": shape_cfg["plate_height"],
        "lever_direction_link2": shape_cfg["lever_direction_link2"],
        "hook_direction_link2": shape_cfg["hook_direction_link2"],
        "collision_mode": feature_cfg["handle_collision_mode"],
        "collision_primitives": collision_primitives,
    }


def body_geometry_exists(variant_dir, body):
    geometry = body.find("geometry")
    if geometry is None:
        return False

    mesh = geometry.find("mesh")
    if mesh is not None:
        mesh_path = variant_dir / mesh.attrib["filename"]
        return mesh_path.exists()

    for primitive_tag in ("box", "cylinder", "sphere"):
        if geometry.find(primitive_tag) is not None:
            return True
    return False


def validate_variant_structure(variant_dir, root):
    link_names = {link.attrib.get("name") for link in root.findall("link")}
    joint_names = {joint.attrib.get("name") for joint in root.findall("joint")}

    if not REQUIRED_LINK_NAMES.issubset(link_names):
        missing = sorted(REQUIRED_LINK_NAMES - link_names)
        raise ValueError(f"Generated URDF is missing required links: {missing}")
    if not REQUIRED_JOINT_NAMES.issubset(joint_names):
        missing = sorted(REQUIRED_JOINT_NAMES - joint_names)
        raise ValueError(f"Generated URDF is missing required joints: {missing}")

    for mesh in root.findall(".//mesh"):
        mesh_path = variant_dir / mesh.attrib["filename"]
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing mesh referenced by URDF: {mesh_path}")

    handle_visual = root.find(".//link[@name='link_2']/visual")
    if handle_visual is None or not body_geometry_exists(variant_dir, handle_visual):
        raise ValueError("link_2 visual is missing or invalid")

    board_collision = root.find(".//link[@name='link_1']/collision")
    if board_collision is None or not body_geometry_exists(variant_dir, board_collision):
        raise ValueError("link_1 collision is missing or invalid")

    handle_collisions = root.findall(".//link[@name='link_2']/collision")
    if not handle_collisions:
        raise ValueError("link_2 must keep at least one collision element")
    if not all(body_geometry_exists(variant_dir, collision) for collision in handle_collisions):
        raise ValueError("One or more link_2 collision elements are invalid")


def print_debug_variant(variant_dir, metadata):
    handle_meta = metadata.get("handle", {})
    print(
        f"[debug] {variant_dir.name}: "
        f"handle={'on' if handle_meta.get('enabled') else 'off'}"
    )
    if handle_meta.get("enabled"):
        print(
            "        handle mesh="
            f"{handle_meta['mesh']} length={handle_meta['length']:.3f} "
            f"radius={handle_meta['radius']:.3f} hook={handle_meta['hook_length']:.3f} "
            f"lever_dir={handle_meta['lever_direction_link2']} hook_dir={handle_meta['hook_direction_link2']}"
        )


def generate_variants(args):
    rng = random.Random(args.seed)
    asset_root = args.asset_root.resolve()
    output_dir = args.output_dir.resolve()
    feature_cfg = build_default_feature_config(args)
    prepare_output_dir(output_dir, args.overwrite)

    source_assets = list(iter_supported_source_assets(asset_root))

    if not source_assets:
        raise RuntimeError(f"No supported door assets found in {asset_root}")

    bounds_cache = {}
    generated = []
    handle_count = 0
    for source_dir in source_assets:
        source_urdf_path = source_dir / "mobility.urdf"
        source_asset_name = source_dir.name

        for variant_idx in range(args.variants_per_source):
            # Per variant:
            # 1. clone a source URDF into memory,
            # 2. scale/reposition the existing door geometry,
            # 3. copy source meshes into the output folder,
            # 4. optionally replace the handle mesh/collision,
            # 5. write a standalone asset directory.
            variant_serial = len(generated)
            variant_name = build_variant_name(source_asset_name, variant_idx)
            variant_dir = output_dir / variant_name
            variant_dir.mkdir(parents=True, exist_ok=False)
            root = ET.parse(source_urdf_path).getroot()
            source_props = get_door_properties(source_dir, root, bounds_cache)
            target_props = sample_target_properties(rng, source_props)
            target_props["flip_hinge_side"] = args.flip_hinge_side
            target_props["opening_direction"] = args.opening_direction
            source_props, _ = apply_variant_to_root(
                source_dir,
                root,
                target_props,
                bounds_cache,
                source_props=source_props,
            )

            attach_variant_files(
                source_dir=source_dir,
                variant_dir=variant_dir,
            )

            handle_enabled = rng.random() < feature_cfg["return_handle_prob"]
            if target_props["flip_hinge_side"]:
                # If we are still using the source handle mesh, mirror it too.
                # If we will generate a new handle mesh below, only mirror the
                # frame/panel and leave link_2 to the procedural replacement.
                flip_links = ["link_0", "link_1", "link_3"]
                if not handle_enabled:
                    flip_links.append("link_2")
                apply_flipped_mesh_variants(root, variant_dir, link_names=tuple(flip_links))

            board = get_board_geometry(variant_dir, root, bounds_cache)
            handle_meta = {"enabled": False}
            if handle_enabled:
                handle_meta = apply_handle_variant(root, variant_dir, board, feature_cfg, rng, variant_serial)
            actual_props = get_door_properties(variant_dir, root, bounds_cache)
            validate_variant_structure(variant_dir, root)

            # Each generated asset is still a normal standalone URDF directory
            # with the same link/joint structure as the source asset.
            ET.ElementTree(root).write(
                variant_dir / "mobility.urdf",
                encoding="utf-8",
                xml_declaration=True,
            )

            metadata = {
                "source_asset": source_asset_name,
                "variant_name": variant_name,
                "direction_pair_key": build_direction_pair_key(variant_name),
                "target_properties": target_props,
                "actual_properties": actual_props,
                "source_properties": source_props,
                "handle": handle_meta,
            }
            with open(variant_dir / "variant_meta.json", "w", encoding="utf-8") as meta_file:
                json.dump(metadata, meta_file, indent=2)

            if handle_meta.get("enabled"):
                handle_count += 1
            if len(generated) < feature_cfg["debug_first_n"]:
                print_debug_variant(variant_dir, metadata)

            generated.append(variant_dir)

    args._generation_summary = {
        "num_source_assets": len(source_assets),
        "num_variants": len(generated),
        "num_return_handles": handle_count,
        "output_dir": str(output_dir),
    }
    return source_assets, generated


def main():
    args = parse_args()
    source_assets, generated = generate_variants(args)
    summary = getattr(args, "_generation_summary", {})
    print(f"Found {len(source_assets)} supported source assets.")
    print(f"Generated {summary.get('num_variants', len(generated))} randomized variants.")
    print(f"Return-handle variants: {summary.get('num_return_handles', 0)}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print("If you overwrite existing assets, clear the IsaacLab URDF->USD cache or force reconversion.")
    if generated:
        print("First few variants:")
        for variant_dir in generated[:5]:
            print(f"  - {variant_dir}")


if __name__ == "__main__":
    main()
