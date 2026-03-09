import os
import trimesh
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

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

    # --- Link 1: Door Board ---
    link_1 = root.find(".//link[@name='link_1']")
    mesh_1 = process_link_mesh(urdf_path, link_1)
    if mesh_1 is not None:
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

    return keypoints


if __name__ == "__main__":
    keypoints = compute_exact_door_keypoints("/home/glorbo4/peiqi/DoorOpening/source/DoorOpening/assets/door/PartNetv4/99650089960001/mobility.urdf")
    print(keypoints)