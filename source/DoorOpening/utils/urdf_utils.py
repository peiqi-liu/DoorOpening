import os
import trimesh
import numpy as np
import xml.etree.ElementTree as ET
import warnings
from scipy.spatial.transform import Rotation as R


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

def get_body_transform(body_element):
    """Extract xyz, rpy, and scale from a URDF visual/collision element."""
    xyz = np.array([0.0, 0.0, 0.0])
    rpy = np.array([0.0, 0.0, 0.0])
    scale = np.array([1.0, 1.0, 1.0])
    
    origin = body_element.find("origin")
    if origin is not None:
        if 'xyz' in origin.attrib:
            xyz = np.array([float(x) for x in origin.attrib['xyz'].split()])
        if 'rpy' in origin.attrib:
            rpy = np.array([float(x) for x in origin.attrib['rpy'].split()])
            
    mesh_el = body_element.find(".//mesh")
    if mesh_el is not None and 'scale' in mesh_el.attrib:
        scale = np.array([float(x) for x in mesh_el.attrib['scale'].split()])
        
    return xyz, rpy, scale

def _coerce_trimesh_geometry(loaded_geometry, mesh_abs_path):
    """Normalize trimesh load results into a single non-empty Trimesh."""
    if isinstance(loaded_geometry, trimesh.Scene):
        meshes = [
            geometry.copy()
            for geometry in loaded_geometry.geometry.values()
            if isinstance(geometry, trimesh.Trimesh) and len(geometry.vertices) > 0
        ]
        if not meshes:
            return None
        if len(meshes) == 1:
            return meshes[0]
        return trimesh.util.concatenate(meshes)

    if isinstance(loaded_geometry, trimesh.Trimesh):
        if len(loaded_geometry.vertices) == 0:
            return None
        return loaded_geometry.copy()

    warnings.warn(f"Unsupported mesh type {type(loaded_geometry)!r} while loading {mesh_abs_path}")
    return None

def process_link_mesh(urdf_path, link_element):
    """Load the first non-empty mesh on a link and return it in the link local frame."""
    if link_element is None:
        return None

    load_errors = []
    for body_tag in ("visual", "collision"):
        body_el = link_element.find(f"./{body_tag}")
        if body_el is None:
            continue

        mesh_el = body_el.find("./geometry/mesh")
        if mesh_el is None:
            continue

        mesh_rel_path = mesh_el.attrib["filename"]
        mesh_abs_path = os.path.join(os.path.dirname(urdf_path), mesh_rel_path)
        xyz, rpy, scale = get_body_transform(body_el)

        try:
            loaded_geometry = trimesh.load(mesh_abs_path)
        except Exception as exc:
            load_errors.append(f"{body_tag}:{mesh_abs_path} ({exc})")
            continue

        mesh = _coerce_trimesh_geometry(loaded_geometry, mesh_abs_path)
        if mesh is None:
            load_errors.append(f"{body_tag}:{mesh_abs_path} (no vertices)")
            continue

        mesh.apply_scale(scale)

        rot_matrix = R.from_euler("xyz", rpy).as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_matrix
        transform_matrix[:3, 3] = xyz
        mesh.apply_transform(transform_matrix)
        return mesh

    if load_errors:
        warnings.warn(
            f"Unable to load a non-empty mesh for link '{link_element.attrib.get('name', '<unknown>')}' "
            f"in {urdf_path}: {'; '.join(load_errors)}"
        )
    return None

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
    if mesh_1 is None:
        raise ValueError(f"Door board mesh for link_1 is missing or empty in {urdf_path}")

    bounds = mesh_1.bounds # Returns [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    min_b, max_b = bounds[0], bounds[1]
    
    # In this local frame, the joint origin is exactly (0,0,0).
    # We find the center in Y (height) and Z (thickness)
    center_y = (min_b[1] + max_b[1]) / 2.0
    center_z = (min_b[2] + max_b[2]) / 2.0
    
    keypoints["link_1"] = [
        [min_b[0], center_y, center_z],
        [(min_b[0] + max_b[0]) / 2.0, center_y, center_z],
        [max_b[0], center_y, center_z]
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

    # --- Link 2: Handle / Hinge ---
    link_2 = root.find(".//link[@name='link_2']")
    mesh_2 = process_link_mesh(urdf_path, link_2)
    if mesh_2 is None:
        raise ValueError(f"Door handle mesh for link_2 is missing or empty in {urdf_path}")

    # To find the true "tip", we look for the vertex furthest from the joint origin (0,0,0)
    vertices = mesh_2.vertices
    distances = np.linalg.norm(vertices, axis=1)
    furthest_vertex = vertices[np.argmax(distances)]
    
    keypoints["link_2"] = [
        [0.0, 0.0, 0.0],
        furthest_vertex.tolist()
    ]

    return keypoints


if __name__ == "__main__":
    keypoints = compute_exact_door_keypoints("/home/glorbo4/peiqi/DoorOpening/source/DoorOpening/assets/door/PartNetv4/99650089960001/mobility.urdf")
    print(keypoints)
