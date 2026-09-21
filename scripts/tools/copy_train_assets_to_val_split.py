import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path


DEFAULT_FAMILIES = [
    "PartNetv5_pro",
    "PartNetv6_pro",
    "PartNetv7_pro",
    "PartNetv8_pro",
]
REQUIRED_FILES = ("mobility.urdf", "traj.pkl", "variant_meta.json")


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    default_asset_root = repo_root / "source" / "DoorOpening" / "assets" / "door"

    parser = argparse.ArgumentParser(
        description=(
            "Clone train-split multi-door assets into new validation-split directories. "
            "The validation split is defined by directory names ending with '00'."
        )
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=default_asset_root,
        help="Root containing the PartNet family folders.",
    )
    parser.add_argument(
        "--families",
        type=str,
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated multi-door family list.",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help=(
            "Comma-separated source asset directory names to clone. "
            "Defaults to every train-split asset shared by all families."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit applied after source selection.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="__valcopy00",
        help="Suffix appended to each source directory name. Must end with '00'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination directories if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    return parser.parse_args()


def _parse_families(families_arg):
    families = [name.strip() for name in str(families_arg).split(",") if name.strip()]
    if not families:
        raise ValueError("At least one family must be provided.")
    return families


def _is_validation_name(asset_name):
    return str(asset_name).endswith("00")


def _list_asset_dirs(family_dir):
    return sorted(path for path in family_dir.iterdir() if path.is_dir())


def _validate_family_layout(asset_root, families):
    family_asset_names = OrderedDict()
    for family_name in families:
        family_dir = asset_root / family_name
        if not family_dir.is_dir():
            raise FileNotFoundError(f"Missing family directory: {family_dir}")
        asset_dirs = _list_asset_dirs(family_dir)
        if not asset_dirs:
            raise FileNotFoundError(f"No asset directories found under {family_dir}")
        family_asset_names[family_name] = [path.name for path in asset_dirs]

    reference_family = families[0]
    reference_names = family_asset_names[reference_family]
    for family_name, names in family_asset_names.items():
        if names != reference_names:
            raise ValueError(
                "Family asset layouts must match exactly for multi-door indexing.\n"
                f"Reference family: {reference_family}\n"
                f"Mismatched family: {family_name}"
            )
    return reference_names


def _resolve_source_names(shared_asset_names, sources_arg, limit):
    if sources_arg is None:
        source_names = [name for name in shared_asset_names if not _is_validation_name(name)]
    else:
        requested = [name.strip() for name in str(sources_arg).split(",") if name.strip()]
        if not requested:
            raise ValueError("--sources did not contain any valid asset names.")
        missing = [name for name in requested if name not in shared_asset_names]
        if missing:
            preview = ", ".join(missing[:8])
            raise FileNotFoundError(f"Requested source assets were not found in all families: {preview}")
        source_names = requested

    if limit is not None:
        source_names = source_names[: max(0, int(limit))]
    if not source_names:
        raise ValueError("No source assets selected.")
    return source_names


def _check_required_files(asset_dir):
    missing = [name for name in REQUIRED_FILES if not (asset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Asset directory is missing required files: {asset_dir} -> {missing}")


def _build_copy_plan(asset_root, families, source_names, suffix):
    if not str(suffix).endswith("00"):
        raise ValueError(f"--suffix must end with '00' so the copies land in the validation split: {suffix!r}")

    plan = []
    for source_name in source_names:
        if _is_validation_name(source_name):
            raise ValueError(f"Refusing to clone validation asset as source: {source_name}")
        dest_name = f"{source_name}{suffix}"
        if dest_name == source_name:
            raise ValueError(f"Destination name equals source name for asset: {source_name}")
        if not _is_validation_name(dest_name):
            raise ValueError(f"Destination name must end with '00': {dest_name}")
        for family_name in families:
            source_dir = asset_root / family_name / source_name
            dest_dir = asset_root / family_name / dest_name
            _check_required_files(source_dir)
            plan.append(
                {
                    "family_name": family_name,
                    "source_name": source_name,
                    "dest_name": dest_name,
                    "source_dir": source_dir,
                    "dest_dir": dest_dir,
                }
            )
    return plan


def _execute_copy_plan(plan, overwrite, dry_run):
    copied = []
    for entry in plan:
        dest_dir = entry["dest_dir"]
        if dest_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Destination already exists: {dest_dir}")
            if not dry_run:
                shutil.rmtree(dest_dir)
        if not dry_run:
            shutil.copytree(entry["source_dir"], dest_dir)
        copied.append(entry)
    return copied


def _write_manifest(asset_root, families, copied_entries, dry_run):
    manifest = {
        "asset_root": str(asset_root.resolve()),
        "families": list(families),
        "num_asset_copies_per_family": len({entry["dest_name"] for entry in copied_entries}),
        "num_directory_copies_total": len(copied_entries),
        "copies": [
            {
                "family_name": entry["family_name"],
                "source_name": entry["source_name"],
                "dest_name": entry["dest_name"],
            }
            for entry in copied_entries
        ],
    }
    manifest_path = asset_root / "val_copy_manifest.json"
    if not dry_run:
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
    return manifest_path


def main():
    args = parse_args()
    asset_root = args.asset_root.resolve()
    families = _parse_families(args.families)

    shared_asset_names = _validate_family_layout(asset_root, families)
    source_names = _resolve_source_names(shared_asset_names, args.sources, args.limit)
    plan = _build_copy_plan(asset_root, families, source_names, args.suffix)

    print(f"Asset root: {asset_root}")
    print("Families:", ", ".join(families))
    print(f"Selected train assets: {len(source_names)}")
    print(f"Directory copies to create: {len(plan)}")
    sample_names = [f"{name}{args.suffix}" for name in source_names[:8]]
    if sample_names:
        print("Sample destination names:", ", ".join(sample_names))

    copied_entries = _execute_copy_plan(plan, overwrite=bool(args.overwrite), dry_run=bool(args.dry_run))
    manifest_path = _write_manifest(asset_root, families, copied_entries, dry_run=bool(args.dry_run))

    if args.dry_run:
        print("Dry run complete. No files were written.")
        print(f"Planned manifest path: {manifest_path}")
        return

    print(f"Created validation-split copies: {len({entry['dest_name'] for entry in copied_entries})}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
