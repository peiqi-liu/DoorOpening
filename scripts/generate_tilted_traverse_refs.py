"""Create 20-degree traversal references for the 2026-10-28 back_base batch."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/DoorOpening/assets/door/PartNetv5_plus_hardstop_recess_tipoffset_back_base_20261028"
DEST = ROOT / "source/DoorOpening/assets/door/tilted_traverse"
ROBOT = ROOT / "source/DoorOpening/assets/glorbot/glorbot.urdf"
LOGS = ROOT / "analysis/tilted_traverse/reference_logs_back_base_tilt20"
OFFSET_DEG = 20.0


def prepare_batch() -> list[Path]:
    if not DEST.exists():
        shutil.copytree(SOURCE, DEST)
    urdfs = sorted(DEST.glob("*/mobility.urdf"))
    if len(urdfs) != 1024:
        raise RuntimeError(f"Expected 1024 copied door URDFs in {DEST}; found {len(urdfs)}")
    for urdf in urdfs:
        meta_path = urdf.parent / "variant_meta.json"
        metadata = json.loads(meta_path.read_text())
        metadata["source_asset_family"] = SOURCE.name
        metadata["trajectory_variant"] = {
            "base_traverse_yaw_offset_deg": OFFSET_DEG,
            "description": "Base yaws left by 20 degrees while traversing the doorway.",
            "generated_by": "teleop_ref compute_waypoint.py",
        }
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return urdfs


def generate(urdf_string: str) -> tuple[str, bool, str, str | None]:
    import torch
    from DoorOpening.utils.state_machine.compute_waypoint import play_and_save_traj

    torch.set_num_threads(1)
    urdf = Path(urdf_string)
    traj_path = urdf.parent / "traj.pkl"
    log_path = LOGS / f"{urdf.parent.name}.log"
    try:
        with log_path.open("w") as output, redirect_stdout(output), redirect_stderr(output):
            play_and_save_traj(
                str(ROBOT),
                str(urdf),
                visualize=False,
                traverse_yaw_offset_deg=OFFSET_DEG,
            )
        digest = hashlib.sha256(traj_path.read_bytes()).hexdigest() if traj_path.is_file() else None
        return urdf.parent.name, traj_path.is_file(), "", digest
    except Exception:
        error = traceback.format_exc()
        with log_path.open("a") as output:
            output.write(error)
        return urdf.parent.name, False, error, None


def main() -> None:
    os.chdir(ROOT)
    LOGS.mkdir(parents=True, exist_ok=True)
    urdfs = prepare_batch()
    failures = []
    hashes = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(generate, str(urdf)) for urdf in urdfs]
        for count, future in enumerate(as_completed(futures), start=1):
            door, success, error, digest = future.result()
            if success:
                hashes[door] = digest
            else:
                failures.append(door)
                print(f"FAILED {door}: {error[-500:]}", flush=True)
            if count % 25 == 0 or count == len(urdfs):
                print(f"Generated {count}/{len(urdfs)}; failures={len(failures)}", flush=True)
    (LOGS / "trajectory_sha256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    (LOGS / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(f"Final: success={len(hashes)}, failed={len(failures)}; output={DEST}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
