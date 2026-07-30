import json
import os
import trimesh
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R


# Which z-plane the link_1 keypoints (fed to the teacher as the door reference) sit on, in the link_1
# thickness axis. "panel_center" = the panel mid-plane (plate-free, matches pre-plate behaviour).
# "plate_top" = the plate's outer face (the surface a depth camera actually sees), so the door reference
# lines up with the visible bump instead of the panel behind it. "mesh_center" = the raw link_1 mesh
# bbox center (the old plate-shifted value; kept only for reference).
LINK1_KEYPOINT_Z_MODE = "mesh_center"


def _load_link1_panel_and_plate(urdf_path):
    """Plate-free PANEL bbox + plate outer-face z (link_1 frame) from variant_meta.json, if present.

    Scratch doors bake the escutcheon PLATE into link_1's mesh, so mesh_1.bounds includes the plate.
    variant_meta.json records the plate-free panel bbox and the plate box, letting us place the link_1
    keypoints deliberately (see LINK1_KEYPOINT_Z_MODE) rather than at the drifted mesh center. Returns
    (panel_min3, panel_max3, plate_top_z_or_None) or None (e.g. PartNet doors with no meta)."""
    meta_path = os.path.join(os.path.dirname(str(urdf_path)), "variant_meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (ValueError, OSError):
        return None
    bbox = meta.get("panel", {}).get("bbox_link1")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 6:
        return None
    handle = meta.get("handle", {})
    plate_max = handle.get("plate_max_link1")
    plate_top_z = None
    if handle.get("has_bump") and isinstance(plate_max, (list, tuple)) and len(plate_max) == 3:
        plate_top_z = float(plate_max[2])  # plate's outer (handle-facing) face
    return np.asarray(bbox[:3], dtype=np.float64), np.asarray(bbox[3:], dtype=np.float64), plate_top_z


def origin_to_transform(origin_element):
    """Convert a URDF <origin> element into a 4x4 transform matrix."""
    xyz = np.zeros(3, dtype=np.float64)
    rpy = np.zeros(3, dtype=np.float64)
    if origin_element is not None:
        if "xyz" in origin_element.attrib:
            xyz = np.array([float(x) for x in origin_element.attrib["xyz"].split()], dtype=np.float64)
        if "rpy" in origin_element.attrib:
            rpy = np.array([float(x) for x in origin_element.attrib["rpy"].split()], dtype=np.float64)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def find_joint_by_child(root, child_name):
    """Return the first joint whose child link matches ``child_name``."""
    for joint in root.findall(".//joint"):
        child = joint.find("child")
        if child is not None and child.attrib.get("link") == child_name:
            return joint
    return None


def transform_points(points, transform):
    """Apply a homogeneous transform to an ``(N, 3)`` point array."""
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    transformed = homog @ transform.T
    return transformed[:, :3]

def get_visual_transform(visual_element):
    """Extracts xyz, rpy, and scale from a URDF visual element."""
    xyz = np.array([0.0, 0.0, 0.0])
    rpy = np.array([0.0, 0.0, 0.0])
    scale = np.array([1.0, 1.0, 1.0])
    
    origin = visual_element.find("origin")
    if origin is not None:
        if 'xyz' in origin.attrib:
            xyz = np.array([float(x) for x in origin.attrib['xyz'].split()])
        if 'rpy' in origin.attrib:
            rpy = np.array([float(x) for x in origin.attrib['rpy'].split()])
            
    mesh_el = visual_element.find(".//mesh")
    if mesh_el is not None and 'scale' in mesh_el.attrib:
        scale = np.array([float(x) for x in mesh_el.attrib['scale'].split()])
        
    return xyz, rpy, scale

def process_link_mesh(urdf_path, link_element):
    """Loads a mesh, applies URDF visual transforms, and returns it in the link's local frame."""
    mesh_el = link_element.find(".//visual/geometry/mesh")
    if mesh_el is None:
        return None
        
    # Dynamically resolve the absolute path to the mesh
    mesh_rel_path = mesh_el.attrib['filename']
    mesh_abs_path = os.path.join(os.path.dirname(urdf_path), mesh_rel_path)
    
    # Extract transforms
    visual_el = link_element.find(".//visual")
    xyz, rpy, scale = get_visual_transform(visual_el)
    
    # Load mesh (force='mesh' ensures we get a Trimesh object, not a Scene)
    mesh = trimesh.load(mesh_abs_path, force='mesh')
    
    # 1. Apply Scale
    mesh.apply_scale(scale)
    
    # 2. Apply Rotation (RPY) and Translation (XYZ)
    rot_matrix = R.from_euler('xyz', rpy).as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rot_matrix
    transform_matrix[:3, 3] = xyz
    
    mesh.apply_transform(transform_matrix)
    
    return mesh

def compute_exact_door_keypoints(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    keypoints = {}

    joint_0 = find_joint_by_child(root, "link_0")
    joint_1 = find_joint_by_child(root, "link_1")
    base_to_frame = origin_to_transform(None if joint_0 is None else joint_0.find("origin"))
    frame_to_board_closed = origin_to_transform(None if joint_1 is None else joint_1.find("origin"))
    base_to_board_closed = base_to_frame @ frame_to_board_closed

    # --- Link 1: Door Board ---
    link_1 = root.find(".//link[@name='link_1']")
    mesh_1 = process_link_mesh(urdf_path, link_1)
    if mesh_1 is not None:
        bounds = mesh_1.bounds # Returns [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        min_b, max_b = bounds[0], bounds[1]

        # link_1 keypoints (fed to the teacher as the door reference) must NOT drift with the escutcheon
        # plate baked into link_1's mesh. Use the plate-free panel bbox for the x/y extents, and place the
        # thickness-axis plane per LINK1_KEYPOINT_Z_MODE. Falls back to the raw mesh bounds for doors
        # without a meta (e.g. PartNet).
        info = _load_link1_panel_and_plate(urdf_path)
        if info is None:
            kp_min, kp_max, plate_top_z = min_b, max_b, None
        else:
            kp_min, kp_max, plate_top_z = info

        center_y = (kp_min[1] + kp_max[1]) / 2.0
        panel_center_z = (kp_min[2] + kp_max[2]) / 2.0
        if LINK1_KEYPOINT_Z_MODE == "plate_top" and plate_top_z is not None:
            center_z = plate_top_z            # door reference on the plate's outer (camera-facing) face
        elif LINK1_KEYPOINT_Z_MODE == "mesh_center":
            center_z = (min_b[2] + max_b[2]) / 2.0  # old plate-shifted value (reference only)
        else:
            center_z = panel_center_z          # "panel_center": panel mid-plane (pre-plate behaviour)

        keypoints["link_1"] = [
            [kp_min[0], center_y, center_z],
            [(kp_min[0] + kp_max[0]) / 2.0, center_y, center_z],
            [kp_max[0], center_y, center_z]
        ]

        board_bbox_corners = np.array(
            [
                [min_b[0], min_b[1], min_b[2]],
                [min_b[0], min_b[1], max_b[2]],
                [min_b[0], max_b[1], min_b[2]],
                [min_b[0], max_b[1], max_b[2]],
                [max_b[0], min_b[1], min_b[2]],
                [max_b[0], min_b[1], max_b[2]],
                [max_b[0], max_b[1], min_b[2]],
                [max_b[0], max_b[1], max_b[2]],
            ],
            dtype=np.float64,
        )
        board_bbox_corners_base = transform_points(board_bbox_corners, base_to_board_closed)
        board_bbox_base = np.stack(
            [
                board_bbox_corners_base.min(axis=0),
                board_bbox_corners_base.max(axis=0),
            ],
            axis=0,
        )
        keypoints["link_1_bbox_base"] = board_bbox_base.tolist()
        keypoints["link_1_bbox_link1"] = [min_b.tolist(), max_b.tolist()]
        # link_1 (panel) pose in the door-base frame at the CLOSED config, as [pos(3), quat_wxyz(4)].
        # Lets callers transform base/world door points into the link_1 LOCAL frame -- the frame the
        # window-hole aug samples in (X=width, Y=height above floor, Z=thickness). Static per asset.
        link1_quat_xyzw = R.from_matrix(base_to_board_closed[:3, :3]).as_quat()
        keypoints["link_1_pose_base"] = [
            *base_to_board_closed[:3, 3].tolist(),
            float(link1_quat_xyzw[3]),
            float(link1_quat_xyzw[0]),
            float(link1_quat_xyzw[1]),
            float(link1_quat_xyzw[2]),
        ]

    # --- Link 2: Handle / Hinge ---
    link_2 = root.find(".//link[@name='link_2']")
    mesh_2 = process_link_mesh(urdf_path, link_2)
    if mesh_2 is not None:
        # To find the true "tip", we look for the vertex furthest from the joint origin (0,0,0)
        vertices = mesh_2.vertices
        distances = np.linalg.norm(vertices, axis=1)
        furthest_vertex = vertices[np.argmax(distances)]

        keypoints["link_2"] = [
            [0.0, 0.0, 0.0],
            furthest_vertex.tolist()
        ]

        # Handle center (mean of the two link_2 keypoints) expressed in the door "base" frame at the
        # CLOSED pose (all joints = 0). Because the door is fix_base, this is a static per-asset
        # constant; composing it with the runtime door-base world pose yields the closed-handle pose
        # without any simulation capture. joint_2 (child=link_2) origin maps link_2 -> link_1.
        joint_2 = find_joint_by_child(root, "link_2")
        board_to_handle_closed = origin_to_transform(None if joint_2 is None else joint_2.find("origin"))
        base_to_handle_closed = base_to_board_closed @ board_to_handle_closed
        handle_center_local = (np.array([0.0, 0.0, 0.0], dtype=np.float64) + furthest_vertex) / 2.0
        handle_center_base = transform_points(handle_center_local[None, :], base_to_handle_closed)[0]
        keypoints["link_2_center_base"] = handle_center_base.tolist()

    # --- Full door outer bbox (frame + board + handle) in the base frame at the closed pose. Used
    # for wall-distractor placement so walls sit outside the WHOLE door, not just the link_1 panel:
    # the frame (link_0) is typically wider than the panel, so referencing only the panel makes the
    # walls overlap the frame (zero/negative gap). ---
    joint_2_el = find_joint_by_child(root, "link_2")
    board_to_handle_closed = origin_to_transform(None if joint_2_el is None else joint_2_el.find("origin"))
    base_to_handle_closed = base_to_board_closed @ board_to_handle_closed
    link_base_transforms = {
        "link_0": base_to_frame,
        "link_1": base_to_board_closed,
        "link_2": base_to_handle_closed,
    }
    full_corners = []
    for link_name, base_T in link_base_transforms.items():
        link_el = root.find(f".//link[@name='{link_name}']")
        if link_el is None:
            continue
        link_mesh = process_link_mesh(urdf_path, link_el)
        if link_mesh is None:
            continue
        mn, mx = link_mesh.bounds
        corners = np.array(
            [[cx, cy, cz] for cx in (mn[0], mx[0]) for cy in (mn[1], mx[1]) for cz in (mn[2], mx[2])],
            dtype=np.float64,
        )
        full_corners.append(transform_points(corners, base_T))
    if full_corners:
        full_pts = np.concatenate(full_corners, axis=0)
        keypoints["door_full_bbox_base"] = np.stack([full_pts.min(axis=0), full_pts.max(axis=0)], axis=0).tolist()

    return keypoints


if __name__ == "__main__":
    keypoints = compute_exact_door_keypoints("/home/glorbo4/peiqi/DoorOpening/source/DoorOpening/assets/door/PartNetv4/99650089960001/mobility.urdf")
    print(keypoints)
