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

def fit_plane_batch_torch(points, use_svd=True):
    """
    Fit planes to a batch of point clouds using PyTorch (GPU-compatible, differentiable).
    
    Parameters:
        points: Tensor of shape (B, N, 3) — B batches, N points, 3D coordinates
        use_svd: bool — if True (recommended), use SVD for better numerical stability;
                 if False, use eigen-decomposition (like PCA).
    
    Returns:
        normals:    Tensor of shape (B, 3) — unit normals (pointing arbitrarily)
        centroids:  Tensor of shape (B, 3) — centroids
    """
    if points.ndim == 2:
        points = points.unsqueeze(0)
    assert points.ndim == 3 and points.shape[-1] == 3, "points must be (B, N, 3)"
    B, N, _ = points.shape
    assert N >= 3, "At least 3 points per cloud required"

    # 1. Centroid: mean over N (dim=1)
    centroids = points.mean(dim=1)                     # (B, 3)

    # 2. Center points
    centered = points - centroids.unsqueeze(1)         # (B, N, 3)

    if use_svd:
        # ✅ SVD method (more robust, especially with noise or degenerate cases)
        # Compute SVD: centered = U @ S @ V^T  → V contains principal directions
        # Last row of V^T (i.e., last column of V) = direction of least variance = normal
        # torch.svd returns V, not V^T — so normal = V[:, :, -1]
        # Note: use torch.linalg.svd for newer PyTorch (≥1.8), more stable
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        normals = Vt[:, -1, :]   # (B, 3) — last right singular vector

    else:
        # ❗ Eigen method (less stable if points are nearly collinear)
        # Covariance: (1/(N-1)) * X^T X
        covs = torch.bmm(centered.transpose(-1, -2), centered) / (N - 1)  # (B, 3, 3)
        # eigh returns (eigenvalues, eigenvectors); eigenvalues ascending
        eigenvalues, eigenvectors = torch.linalg.eigh(covs)
        normals = eigenvectors[:, :, 0]  # smallest eigenvector → (B, 3)

    # Normalize (important for unit normals, and to handle numerical drift)
    normals = torch.nn.functional.normalize(normals, dim=-1)  # (B, 3)

    return normals, centroids

# ✅ Example usage (CPU or GPU):
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    B, N = 4, 200
    torch.manual_seed(42)

    # Create synthetic batch: each cloud lies near a plane with known normal
    normals_gt = torch.randn(B, 3, device=device)
    normals_gt = torch.nn.functional.normalize(normals_gt, dim=-1)

    # Generate orthonormal basis for each plane
    # v1: arbitrary non-parallel vector
    v1 = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(B, 1)
    # If normal is close to [1,0,0], use [0,1,0] instead
    mask = torch.abs(normals_gt[:, 0]) > 0.9
    v1[mask] = torch.tensor([0.0, 1.0, 0.0], device=device)

    # Gram-Schmidt: u1 = v1 - proj_normal(v1)
    dot = (v1 * normals_gt).sum(dim=1, keepdim=True)
    u1 = v1 - dot * normals_gt
    u1 = torch.nn.functional.normalize(u1, dim=-1)
    u2 = torch.linalg.cross(normals_gt, u1)  # already unit and orthogonal

    # Random coefficients
    a = torch.randn(B, N, 1, device=device)
    b = torch.randn(B, N, 1, device=device)

    # Points = a * u1 + b * u2 + noise
    points = (
        a * u1.unsqueeze(1) +
        b * u2.unsqueeze(1) +
        0.01 * torch.randn(B, N, 3, device=device)
    )  # (B, N, 3)

    # Fit planes
    normals_est, centroids = fit_plane_batch_torch(points)

    print("GT normals:\n", normals_gt)
    print("Est normals:\n", normals_est)
    # Align sign for comparison (normals are undirected)
    sign = torch.sign((normals_gt * normals_est).sum(dim=1, keepdim=True))
    normals_est_aligned = normals_est * sign
    angular_error = torch.acos(torch.clamp((normals_gt * normals_est_aligned).sum(dim=1), -1, 1))
    print("Mean angular error (deg):", torch.rad2deg(angular_error).mean().item())