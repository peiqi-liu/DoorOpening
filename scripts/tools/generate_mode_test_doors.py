import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import generate_randomized_doors as randomized_doors


FAMILY_SPECS = [
    {
        "family_name": "PartNetv5",
        "mode_id": 0,
        "flip_hinge_side": True,
        "opening_direction": "pull",
        "description": "pull, mirrored hinge side, handle on min-x side",
    },
    {
        "family_name": "PartNetv6",
        "mode_id": 1,
        "flip_hinge_side": False,
        "opening_direction": "pull",
        "description": "pull, original hinge side, handle on max-x side",
    },
    {
        "family_name": "PartNetv7",
        "mode_id": 2,
        "flip_hinge_side": False,
        "opening_direction": "push",
        "description": "push, original hinge side, handle on max-x side",
    },
    {
        "family_name": "PartNetv8",
        "mode_id": 3,
        "flip_hinge_side": True,
        "opening_direction": "push",
        "description": "push, mirrored hinge side, handle on min-x side",
    },
]


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    default_asset_root = repo_root / "source" / "DoorOpening" / "assets" / "door" / "PartNetv4"
    default_test_base = repo_root / "source" / "DoorOpening" / "assets" / "door" / "test_sets"
    default_reference_root = repo_root / "source" / "DoorOpening" / "assets" / "door"

    parser = argparse.ArgumentParser(
        description=(
            "Generate an isolated four-family door mode-prediction test set. "
            "The output is intentionally placed outside the default PartNetv5-v8 folders."
        )
    )
    parser.add_argument("--asset-root", type=Path, default=default_asset_root)
    parser.add_argument(
        "--test-root",
        type=Path,
        default=None,
        help="Output root containing PartNetv5-v8 subfolders. Defaults to test_sets/mode_prediction_seed_<seed>.",
    )
    parser.add_argument("--test-base", type=Path, default=default_test_base)
    parser.add_argument("--variants-per-source", type=int, default=1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-reference-trajectories",
        action="store_true",
        help=(
            "Copy traj.pkl files from an existing PartNetv5-v8 root into the test set. "
            "This is only a rollout plumbing shortcut, not a newly computed demonstration."
        ),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=default_reference_root,
        help="Root containing the existing PartNetv5-v8 folders used when copying traj.pkl files.",
    )
    return parser.parse_args()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_test_root(args) -> Path:
    if args.test_root is not None:
        return args.test_root.resolve()
    return (args.test_base / f"mode_prediction_seed_{args.seed}").resolve()


def _assert_isolated_test_root(test_root: Path):
    repo_root = Path(__file__).resolve().parents[2]
    current_asset_root = repo_root / "source" / "DoorOpening" / "assets" / "door"
    current_family_dirs = [
        (current_asset_root / spec["family_name"]).resolve()
        for spec in FAMILY_SPECS
    ]

    for family_dir in current_family_dirs:
        if test_root == family_dir or _is_relative_to(test_root, family_dir):
            raise ValueError(
                f"Refusing to write the test set inside the active family folder: {family_dir}"
            )


def _prepare_test_root(test_root: Path, overwrite: bool):
    _assert_isolated_test_root(test_root)
    if test_root.exists():
        if overwrite:
            shutil.rmtree(test_root)
        elif any(test_root.iterdir()):
            raise FileExistsError(
                f"{test_root} already exists and is not empty. Use --overwrite for this isolated test root."
            )
    test_root.mkdir(parents=True, exist_ok=True)


def _copy_reference_trajectories(test_root: Path, generated_by_family: dict[str, list[Path]], reference_root: Path):
    copied = []
    missing = []
    reference_root = reference_root.resolve()

    for spec in FAMILY_SPECS:
        family_name = spec["family_name"]
        for variant_dir in generated_by_family[family_name]:
            source_traj = reference_root / family_name / variant_dir.name / "traj.pkl"
            target_traj = variant_dir / "traj.pkl"
            if not source_traj.exists():
                missing.append(str(source_traj))
                continue
            shutil.copy2(source_traj, target_traj)
            copied.append(str(target_traj))

    if missing:
        preview = "\n  - ".join(missing[:8])
        raise FileNotFoundError(
            "Missing reference trajectories for generated test assets. "
            "The full DooropeningMulti env requires traj.pkl for every asset.\n"
            f"First missing paths:\n  - {preview}"
        )

    return copied


def _write_manifest(
    test_root: Path,
    source_asset_root: Path,
    generated_by_family: dict[str, list[Path]],
    copied_trajectories: list[str],
    args,
):
    manifest = {
        "test_root": str(test_root),
        "source_asset_root": str(source_asset_root.resolve()),
        "variants_per_source": int(args.variants_per_source),
        "seed": int(args.seed),
        "families": {
            spec["family_name"]: {
                "mode_id": spec["mode_id"],
                "flip_hinge_side": spec["flip_hinge_side"],
                "opening_direction": spec["opening_direction"],
                "description": spec["description"],
                "num_assets": len(generated_by_family[spec["family_name"]]),
            }
            for spec in FAMILY_SPECS
        },
        "trajectory_source": str(args.reference_root.resolve()) if args.copy_reference_trajectories else None,
        "copied_reference_trajectory_count": len(copied_trajectories),
        "rollout_env": {
            "DOOR_OPENING_MULTI_DOOR_ROOT": str(test_root),
            "DOOR_OPENING_MULTI_DOOR_FAMILIES": ",".join(spec["family_name"] for spec in FAMILY_SPECS),
        },
        "notes": [
            "This directory is outside the default active PartNetv5-v8 folders, so it does not affect current training assets.",
            "Copied trajectories, when enabled, are only a shortcut to let DooropeningMulti instantiate without running compute_waypoint.py.",
        ],
    }

    manifest_path = test_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    return manifest_path


def main():
    args = parse_args()
    test_root = _resolve_test_root(args)
    _prepare_test_root(test_root, args.overwrite)

    generated_by_family = {}
    source_asset_count = None
    for spec in FAMILY_SPECS:
        family_name = spec["family_name"]
        family_output_dir = test_root / family_name
        generation_args = SimpleNamespace(
            asset_root=args.asset_root,
            output_dir=family_output_dir,
            variants_per_source=args.variants_per_source,
            seed=args.seed,
            flip_hinge_side=spec["flip_hinge_side"],
            opening_direction=spec["opening_direction"],
            overwrite=False,
        )
        source_assets, generated = randomized_doors.generate_variants(generation_args)
        if source_asset_count is None:
            source_asset_count = len(source_assets)
        generated_by_family[family_name] = list(generated)

    copied_trajectories = []
    if args.copy_reference_trajectories:
        copied_trajectories = _copy_reference_trajectories(
            test_root=test_root,
            generated_by_family=generated_by_family,
            reference_root=args.reference_root,
        )

    manifest_path = _write_manifest(
        test_root=test_root,
        source_asset_root=args.asset_root,
        generated_by_family=generated_by_family,
        copied_trajectories=copied_trajectories,
        args=args,
    )

    total_assets = sum(len(paths) for paths in generated_by_family.values())
    print(f"Generated isolated mode test set: {test_root}")
    print(f"Supported source assets: {source_asset_count}")
    print(f"Generated assets: {total_assets}")
    print(f"Manifest: {manifest_path}")
    if args.copy_reference_trajectories:
        print(f"Copied reference traj.pkl files: {len(copied_trajectories)}")
    print("To opt into this test set for DooropeningMulti, set:")
    print(f"  export DOOR_OPENING_MULTI_DOOR_ROOT={test_root}")
    print(
        "  export DOOR_OPENING_MULTI_DOOR_FAMILIES="
        + ",".join(spec["family_name"] for spec in FAMILY_SPECS)
    )


if __name__ == "__main__":
    main()
