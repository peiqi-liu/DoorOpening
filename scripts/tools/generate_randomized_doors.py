import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_PANEL_WIDTH_RANGE_M = (0.8, 1.1)
# Despite the old name, this range is treated as the target overall door/frame
# height after scaling. That keeps unusual assets with transoms from ending up
# much taller in world space than the rest of the set.
DEFAULT_PANEL_HEIGHT_RANGE_M = (1.80, 2.10)
DEFAULT_HANDLE_HEIGHT_RANGE_M = (0.7, 1.00)
DEFAULT_HANDLE_EDGE_DISTANCE_RANGE_M = (0.05, 0.11)

MIN_HANDLE_BOTTOM_CLEARANCE_M = 0.15
MIN_HANDLE_TOP_CLEARANCE_M = 0.15
MIN_HANDLE_EDGE_CLEARANCE_M = 0.02


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
    return parser.parse_args()


def parse_vector(text, default=(0.0, 0.0, 0.0)):
    if text is None:
        return list(default)
    return [float(value) for value in text.split()]


def format_vector(values):
    return " ".join(f"{value:.6f}" for value in values)


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


def get_door_properties(asset_dir, root, bounds_cache):
    board = get_board_geometry(asset_dir, root, bounds_cache)
    frame = get_frame_geometry(asset_dir, root, bounds_cache)
    handle_joint = get_joint(root, get_handle_attachment_joint_name(root))
    handle_joint_origin = parse_vector(handle_joint.find("origin").attrib.get("xyz"))

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
    }


def is_supported_source_asset(urdf_path):
    root = ET.parse(urdf_path).getroot()
    link_names = {link.attrib.get("name") for link in root.findall("link")}
    joint_names = {joint.attrib.get("name") for joint in root.findall("joint")}

    required_links = {"base", "link_0", "link_1", "link_2"}
    required_joints = {"joint_0", "joint_1", "joint_2"}
    if not required_links.issubset(link_names):
        return False
    if not required_joints.issubset(joint_names):
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

    if not is_supported_topology:
        return False

    return True


def iter_supported_source_assets(asset_root):
    for urdf_path in sorted(asset_root.glob("*/mobility.urdf")):
        if is_supported_source_asset(urdf_path):
            yield urdf_path.parent


def scale_link_xy(root, link_name, sx, sy):
    for tag in ("visual", "collision"):
        body = root.find(f".//link[@name='{link_name}']/{tag}")
        if body is None:
            continue

        mesh, origin = get_mesh_and_origin(body)
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
    for tag in ("visual", "collision"):
        body = root.find(f".//link[@name='{link_name}']/{tag}")
        if body is None:
            continue

        _, origin = get_mesh_and_origin(body)
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
    for tag in ("visual", "collision"):
        body = root.find(f".//link[@name='{link_name}']/{tag}")
        if body is None:
            continue

        mesh, _ = get_mesh_and_origin(body)
        mesh_path = (variant_dir / mesh.attrib["filename"]).resolve()
        mirrored_path = ensure_mirrored_mesh(mesh_path, mirrored_cache)
        mesh.attrib["filename"] = mirrored_path.relative_to(variant_dir).as_posix()


def apply_flipped_mesh_variants(root, variant_dir):
    mirrored_cache = {}
    # When we flip hinge side, these meshes should look mirrored too rather than
    # only moving the joints. That includes the frame, panel, and handle/lock.
    for link_name in ("link_0", "link_1", "link_2", "link_3"):
        mirror_link_meshes_x(root, variant_dir, link_name, mirrored_cache)


def attach_variant_files(source_dir, variant_dir):
    texture_src = source_dir / "texture_dae"
    texture_dst = variant_dir / "texture_dae"
    if texture_src.exists():
        shutil.copytree(texture_src, texture_dst)


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


def generate_variants(args):
    rng = random.Random(args.seed)
    asset_root = args.asset_root.resolve()
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite)

    source_assets = list(iter_supported_source_assets(asset_root))

    if not source_assets:
        raise RuntimeError(f"No supported door assets found in {asset_root}")

    bounds_cache = {}
    generated = []

    for source_dir in source_assets:
        source_urdf_path = source_dir / "mobility.urdf"
        source_asset_name = source_dir.name

        for variant_idx in range(args.variants_per_source):
            variant_name = build_variant_name(source_asset_name, variant_idx)
            variant_dir = output_dir / variant_name
            variant_dir.mkdir(parents=True, exist_ok=False)

            root = ET.parse(source_urdf_path).getroot()
            source_props = get_door_properties(source_dir, root, bounds_cache)
            target_props = sample_target_properties(rng, source_props)
            target_props["flip_hinge_side"] = args.flip_hinge_side
            target_props["opening_direction"] = args.opening_direction
            source_props, actual_props = apply_variant_to_root(
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
            if target_props["flip_hinge_side"]:
                apply_flipped_mesh_variants(root, variant_dir)

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
                "target_properties": target_props,
                "actual_properties": actual_props,
                "source_properties": source_props,
            }
            with open(variant_dir / "variant_meta.json", "w", encoding="utf-8") as meta_file:
                json.dump(metadata, meta_file, indent=2)

            generated.append(variant_dir)

    return source_assets, generated


def main():
    args = parse_args()
    source_assets, generated = generate_variants(args)
    print(f"Found {len(source_assets)} supported source assets.")
    print(f"Generated {len(generated)} randomized variants in {args.output_dir.resolve()}")
    if generated:
        print("First few variants:")
        for variant_dir in generated[:5]:
            print(f"  - {variant_dir}")


if __name__ == "__main__":
    main()
