#!/usr/bin/env python3
import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Hard-copy generated pull door assets into push door assets with identical geometry. "
            "This only negates joint_1 axis in mobility.urdf."
        )
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source pull asset directory, or a root containing many asset dirs with mobility.urdf.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        required=True,
        help="Destination root directory for copied push assets.",
    )
    parser.add_argument(
        "--joint-name",
        default="joint_1",
        help="Door opening joint to negate. Default: joint_1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination asset directories.",
    )
    parser.add_argument(
        "--name-replace",
        nargs=2,
        metavar=("OLD", "NEW"),
        default=None,
        help="Optional string replacement for copied asset folder names, e.g. PartNetv5 PartNetv8.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix added to copied asset folder names if --name-replace is not used.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    return parser.parse_args()


def parse_vector(text):
    if text is None:
        raise ValueError("Missing vector text.")
    return [float(v) for v in text.split()]


def format_vector(values):
    return " ".join(f"{v:.6f}" for v in values)


def iter_asset_dirs(src: Path):
    src = src.resolve()
    if (src / "mobility.urdf").exists():
        yield src
        return

    for child in sorted(src.iterdir()):
        if child.is_dir() and (child / "mobility.urdf").exists():
            yield child


def make_dst_name(src_name: str, args):
    if args.name_replace is not None:
        old, new = args.name_replace
        return src_name.replace(old, new)
    return f"{src_name}{args.suffix}"


def ignore_generated_or_bad_files(_dir, names):
    ignored = []
    for name in names:
        lower = name.lower()
        if name.startswith(".nfs"):
            ignored.append(name)
        elif lower.endswith((".usd", ".usda", ".usdc")):
            ignored.append(name)
        elif name == "__pycache__":
            ignored.append(name)
    return ignored


def negate_joint_axis(urdf_path: Path, joint_name: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    axis = root.find(f".//joint[@name='{joint_name}']/axis")
    if axis is None:
        raise ValueError(f"Could not find axis for joint '{joint_name}' in {urdf_path}")

    old_axis = parse_vector(axis.attrib.get("xyz"))
    new_axis = [-v for v in old_axis]
    axis.attrib["xyz"] = format_vector(new_axis)

    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return old_axis, new_axis


def update_variant_meta(dst_dir: Path, src_dir: Path):
    meta_path = dst_dir / "variant_meta.json"
    if not meta_path.exists():
        return

    src_meta_path = src_dir / "variant_meta.json"
    src_meta = {}
    if src_meta_path.exists():
        with open(src_meta_path, "r", encoding="utf-8") as f:
            src_meta = json.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Negating joint_1 physically turns the pull source into a push door, so every recorded
    # opening_direction must say "push". Updating only target_properties (as before) left
    # actual_properties/source_properties at "pull", and the consumer reads actual_properties
    # first -- which is what mislabeled the copied push families.
    for props_key in ("target_properties", "actual_properties", "source_properties"):
        props = meta.setdefault(props_key, {})
        props["opening_direction"] = "push"

    if isinstance(src_meta, dict) and "initial_state" in src_meta:
        # Keep the exact sampled initial state from the pull asset so the
        # copied push asset can reuse the same base/door start configuration.
        meta["initial_state"] = src_meta["initial_state"]

    meta["copied_from_pull_asset"] = str(src_dir)
    meta["conversion"] = {
        "type": "pull_to_push_same_geometry",
        "changed": ["negated joint_1 axis"],
        "unchanged": [
            "meshes",
            "handle position",
            "hinge side",
            "panel dimensions",
            "joint origins",
        ],
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def copy_one_asset(src_dir: Path, dst_dir: Path, args):
    if dst_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst_dir} already exists. Use --overwrite to replace it.")
        shutil.rmtree(dst_dir)

    if args.dry_run:
        print(f"[dry-run] {src_dir} -> {dst_dir}")
        return

    shutil.copytree(src_dir, dst_dir, ignore=ignore_generated_or_bad_files)

    urdf_path = dst_dir / "mobility.urdf"
    old_axis, new_axis = negate_joint_axis(urdf_path, args.joint_name)
    update_variant_meta(dst_dir, src_dir)

    print(f"[ok] {src_dir.name} -> {dst_dir.name}")
    print(f"     {args.joint_name} axis: {format_vector(old_axis)} -> {format_vector(new_axis)}")


def main():
    args = parse_args()
    src = args.src.resolve()
    dst_root = args.dst_root.resolve()

    if not src.exists():
        raise FileNotFoundError(src)

    asset_dirs = list(iter_asset_dirs(src))
    if not asset_dirs:
        raise RuntimeError(f"No mobility.urdf asset directories found under {src}")

    if not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(asset_dirs)} source pull assets.")
    print(f"Destination root: {dst_root}")

    for src_dir in asset_dirs:
        dst_name = make_dst_name(src_dir.name, args)
        dst_dir = dst_root / dst_name

        if src_dir.resolve() == dst_dir.resolve():
            raise RuntimeError(f"Refusing to overwrite source asset in-place: {src_dir}")

        copy_one_asset(src_dir, dst_dir, args)

    print("Done.")


if __name__ == "__main__":
    main()
