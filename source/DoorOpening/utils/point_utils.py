import open3d as o3d
import torch
import numpy as np

def tensor_to_ply(tensor: torch.Tensor, filename: str):
    """
    Save a (N, 3) torch tensor as a .ply point cloud.
    
    Args:
        tensor: (N, 3) tensor of xyz coordinates (float32/float64)
        filename: output .ply file path (e.g., "points.ply")
    """
    assert tensor.ndim == 2 and tensor.shape[1] == 3, "Input must be (N, 3)"
    
    # Ensure on CPU and convert to numpy float32
    points = tensor.detach().cpu().numpy().astype(np.float32)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Save
    o3d.io.write_point_cloud(filename, pcd, write_ascii=False, compressed=False)
    print(f"Saved {points.shape[0]} points to {filename}")