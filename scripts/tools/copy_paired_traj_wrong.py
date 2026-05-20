#!/usr/bin/env python3
import argparse
import pickle
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create traj_wrong.pkl for paired pull/push assets by swapping their traj.pkl files. "
            "This preserves the normal positive door joint coordinates; the mismatch comes from "
            "loading that trajectory on the paired asset whose joint_1 axis is flipped."
        )
    )
    parser.add_argument(
        "--a",
        type=Path,
        required=True,
        help="First asset directory, or a root containing asset directories with mobility.urdf.",
    )
    parser.add_argument(
        "--b",
        type=Path,
        required=True,
        help="Second asset directory/root paired with --a.",
    )
    parser.add_argument(
        "--src-traj",
        default="traj.pkl",
        help="Trajectory filename to copy from each asset directory.",
    )
    parser.add_argument(
        "--dst-traj",
        default="traj_wrong.pkl",
        help="Trajectory filename to write into the paired asset directory.",
    )
    parser.add_argument(
        "--pair-by",
        choices=["name", "order"],
        default="name",
        help="Pair multiple assets by directory name or sorted order. Single assets are paired directly.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination trajectory files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    return parser.parse_args()


def iter_asset_dirs(root: Path):
    root = root.resolve()
    if (root / "mobility.urdf").exists():
        return [root]

    asset_dirs = [path.parent for path in root.rglob("mobility.urdf")]
    return sorted(set(asset_dirs))


def make_pairs(a_dirs, b_dirs, pair_by):
    if len(a_dirs) == 1 and len(b_dirs) == 1:
        return [(a_dirs[0], b_dirs[0])]

    if pair_by == "order":
        if len(a_dirs) != len(b_dirs):
            raise ValueError(f"Cannot pair by order: got {len(a_dirs)} dirs for --a and {len(b_dirs)} dirs for --b.")
        return list(zip(sorted(a_dirs), sorted(b_dirs)))

    a_by_name = {path.name: path for path in a_dirs}
    b_by_name = {path.name: path for path in b_dirs}
    duplicate_a = len(a_by_name) != len(a_dirs)
    duplicate_b = len(b_by_name) != len(b_dirs)
    if duplicate_a or duplicate_b:
        raise ValueError("Duplicate asset directory names found. Use --pair-by order or pass more specific roots.")

    missing_in_b = sorted(set(a_by_name) - set(b_by_name))
    missing_in_a = sorted(set(b_by_name) - set(a_by_name))
    if missing_in_b or missing_in_a:
        message = ["Asset directory names do not match."]
        if missing_in_b:
            message.append(f"Only in --a: {missing_in_b[:10]}")
        if missing_in_a:
            message.append(f"Only in --b: {missing_in_a[:10]}")
        raise ValueError(" ".join(message))

    return [(a_by_name[name], b_by_name[name]) for name in sorted(a_by_name)]


def door_traj_range(traj_path: Path):
    try:
        with open(traj_path, "rb") as f:
            data = pickle.load(f)
        door_traj = data.get("door_traj")
        if door_traj is None:
            return None
        board_joint = door_traj[..., 0]
        return float(board_joint.min()), float(board_joint.max())
    except Exception as exc:
        return f"unavailable: {exc}"


def copy_traj(src_dir: Path, dst_dir: Path, src_name: str, dst_name: str, overwrite: bool, dry_run: bool):
    src_path = src_dir / src_name
    dst_path = dst_dir / dst_name

    if not src_path.exists():
        raise FileNotFoundError(src_path)
    dst_exists = dst_path.exists()
    if dst_exists and not overwrite and not dry_run:
        raise FileExistsError(f"{dst_path} already exists. Use --overwrite to replace it.")

    src_range = door_traj_range(src_path)
    range_text = ""
    if isinstance(src_range, tuple):
        range_text = f" door_traj[:, 0]=[{src_range[0]:.4f}, {src_range[1]:.4f}]"
    elif src_range is not None:
        range_text = f" door range {src_range}"

    if dry_run:
        exists_text = " [exists; would require --overwrite]" if dst_exists and not overwrite else ""
        print(f"[dry-run] {src_path} -> {dst_path}{exists_text}{range_text}")
        return

    shutil.copy2(src_path, dst_path)
    print(f"[ok] {src_path} -> {dst_path}{range_text}")


def main():
    args = parse_args()
    a_dirs = iter_asset_dirs(args.a)
    b_dirs = iter_asset_dirs(args.b)
    if not a_dirs:
        raise RuntimeError(f"No mobility.urdf asset directories found under {args.a}")
    if not b_dirs:
        raise RuntimeError(f"No mobility.urdf asset directories found under {args.b}")

    pairs = make_pairs(a_dirs, b_dirs, args.pair_by)
    print(f"Found {len(pairs)} paired asset(s).")

    for a_dir, b_dir in pairs:
        if a_dir.resolve() == b_dir.resolve():
            raise RuntimeError(f"Refusing to copy an asset to itself: {a_dir}")
        copy_traj(a_dir, b_dir, args.src_traj, args.dst_traj, args.overwrite, args.dry_run)
        copy_traj(b_dir, a_dir, args.src_traj, args.dst_traj, args.overwrite, args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
