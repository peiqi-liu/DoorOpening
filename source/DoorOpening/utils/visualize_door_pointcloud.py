from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_partnet_root() -> Path:
    return _repo_root() / "source" / "DoorOpening" / "assets" / "door" / "PartNetv4"


def _list_assets(partnet_root: Path) -> list[Path]:
    return sorted(Path(path) for path in glob.glob(str(partnet_root / "**" / "mobility.urdf"), recursive=True))


def _select_assets(args, assets: list[Path]) -> list[Path]:
    selected: list[Path] = []

    if args.asset_path is not None:
        return [Path(args.asset_path).expanduser().resolve()]

    if args.asset_name is not None:
        matches = [path for path in assets if path.parent.name == args.asset_name]
        if not matches:
            raise ValueError(f"Could not find asset named '{args.asset_name}'.")
        if len(matches) > 1:
            raise ValueError(f"Asset name '{args.asset_name}' is ambiguous across PartNetv4.")
        return [matches[0].resolve()]

    if args.asset_index is not None:
        for index in args.asset_index:
            if index < 0 or index >= len(assets):
                raise IndexError(f"asset_index must be in [0, {len(assets) - 1}], got {index}.")
            selected.append(assets[index].resolve())
        return selected

    return [path.resolve() for path in assets]


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _sanitize_points(points):
    import torch

    points = points.detach().cpu().to(dtype=torch.float32)
    if points.ndim == 3:
        points = points[0]
    finite_mask = torch.isfinite(points).all(dim=-1)
    return points[finite_mask]


def _normalize_door_link_points(board_points, handle_points):
    import torch

    if board_points.numel() == 0 and handle_points.numel() == 0:
        return board_points, handle_points

    combined = torch.cat([tensor for tensor in (board_points, handle_points) if tensor.numel() > 0], dim=0)
    bbox_min = combined.min(dim=0).values
    bbox_max = combined.max(dim=0).values
    center_xy = 0.5 * (bbox_min[:2] + bbox_max[:2])
    shift = torch.tensor([center_xy[0], center_xy[1], bbox_min[2]], dtype=combined.dtype)
    return board_points - shift, handle_points - shift


def _build_colored_cloud(board_points, handle_points, board_color, handle_color):
    import torch

    board_colors = torch.tensor(board_color, dtype=board_points.dtype).unsqueeze(0).repeat(board_points.shape[0], 1)
    handle_colors = torch.tensor(handle_color, dtype=handle_points.dtype).unsqueeze(0).repeat(handle_points.shape[0], 1)
    points = torch.cat([board_points, handle_points], dim=0)
    colors = torch.cat([board_colors, handle_colors], dim=0)
    return points, colors


def _make_o3d_cloud(points, colors, translation=None):
    import numpy as np
    import open3d as o3d

    points_np = points.detach().cpu().numpy()
    colors_np = colors.detach().cpu().numpy()
    if translation is not None:
        points_np = points_np + np.asarray(translation, dtype=np.float32)[None, :]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors_np.astype(np.float64))
    return cloud


def _save_cloud(path: Path, points, colors) -> None:
    import open3d as o3d

    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = _make_o3d_cloud(points, colors)
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)
    print(f"saved {points.shape[0]:5d} pts -> {path}")


def _sample_one_door(asset_path: Path, joint_angles, num_points: int, device: str, board_color, handle_color):
    from DoorOpening.utils.extract_pointcloud_from_articulation import FrankaGripperSampler

    sampler = FrankaGripperSampler(str(asset_path), device=device, num_points=int(num_points))
    board_points = _sanitize_points(sampler.sample_link_set(joint_angles, "link_1")[0])
    handle_points = _sanitize_points(sampler.sample_link_set(joint_angles, "link_2")[0])
    board_points, handle_points = _normalize_door_link_points(board_points, handle_points)
    return _build_colored_cloud(board_points, handle_points, board_color=board_color, handle_color=handle_color)


def _grid_translation(index: int, columns: int, spacing_x: float, spacing_y: float):
    row = index // columns
    col = index % columns
    return [col * spacing_x, -row * spacing_y, 0.0]


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize ground-truth PartNetv4 door point clouds from extract_pointcloud_from_articulation.py."
    )
    parser.add_argument("--partnet-root", type=Path, default=_default_partnet_root())
    parser.add_argument("--list-assets", action="store_true")
    parser.add_argument("--asset-path", type=str, default=None)
    parser.add_argument("--asset-name", type=str, default=None)
    parser.add_argument("--asset-index", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--board-angle", type=float, default=0.0, help="joint_1 angle in radians")
    parser.add_argument("--handle-angle", type=float, default=0.0, help="joint_2 angle in radians")
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--spacing-x", type=float, default=1.8)
    parser.add_argument("--spacing-y", type=float, default=1.8)
    parser.add_argument("--board-color", type=float, nargs=3, default=[0.62, 0.62, 0.62])
    parser.add_argument("--handle-color", type=float, nargs=3, default=[0.90, 0.22, 0.18])
    parser.add_argument("--show", action="store_true", help="open an Open3D viewer")
    parser.add_argument("--save-per-door", action="store_true", help="save one colored ply per door")
    parser.add_argument("--save-scene", action="store_true", help="save the laid-out multi-door scene as one ply")
    parser.add_argument("--output-dir", type=Path, default=Path("debug_door_pointclouds"))
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    assets = _list_assets(Path(args.partnet_root).expanduser().resolve())
    if not assets:
        raise FileNotFoundError(f"No mobility.urdf assets found under {args.partnet_root}.")

    if args.list_assets:
        for idx, asset in enumerate(assets):
            print(f"{idx:03d}: {asset.parent.name} -> {asset}")
        return

    import torch
    import open3d as o3d

    selected_assets = _select_assets(args, assets)
    device = str(args.device)
    joint_angles = torch.tensor(
        [[float(args.board_angle), float(args.handle_angle)]],
        device=device,
        dtype=torch.float32,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_points = []
    scene_colors = []
    scene_geometries = []

    print(f"selected {len(selected_assets)} door assets")
    for idx, asset_path in enumerate(selected_assets):
        points, colors = _sample_one_door(
            asset_path=asset_path,
            joint_angles=joint_angles,
            num_points=int(args.num_points),
            device=device,
            board_color=args.board_color,
            handle_color=args.handle_color,
        )
        translation = _grid_translation(
            index=idx,
            columns=max(1, int(args.columns)),
            spacing_x=float(args.spacing_x),
            spacing_y=float(args.spacing_y),
        )
        scene_geometries.append(_make_o3d_cloud(points, colors, translation=translation))

        translated_points = points.clone()
        translated_points[:, 0] += float(translation[0])
        translated_points[:, 1] += float(translation[1])
        translated_points[:, 2] += float(translation[2])
        scene_points.append(translated_points)
        scene_colors.append(colors)

        print(f"{idx:03d}: {asset_path.parent.name} at grid offset {translation}")
        if args.save_per_door:
            _save_cloud(output_dir / f"{asset_path.parent.name}_gt_colored.ply", points, colors)

    scene_points_tensor = torch.cat(scene_points, dim=0)
    scene_colors_tensor = torch.cat(scene_colors, dim=0)

    if args.save_scene:
        _save_cloud(output_dir / "all_doors_gt_colored_scene.ply", scene_points_tensor, scene_colors_tensor)

    if args.show:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        o3d.visualization.draw_geometries(
            scene_geometries + [frame],
            window_name="PartNetv4 Ground Truth Door Point Clouds",
            width=1600,
            height=1000,
        )


if __name__ == "__main__":
    main()
