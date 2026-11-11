#!/usr/bin/env python3
"""Regenerate the 1,024 door references with the 20-degree traverse waypoint."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "source/DoorOpening/assets/door/PartNetv5_plus_hardstop_recess_tipoffset_back_base_teleop_ref_20261107_updated"
PLANNER = ROOT / "source/DoorOpening/utils/state_machine/compute_waypoint.py"
OFFLINE_PULL = PLANNER.with_name("offline_pull_door.py")
OUT = Path(__file__).resolve().parent


def generate(urdf: Path):
    env = os.environ.copy()
    isaaclab_source = Path(sys.executable).resolve().parents[1] / "lib/python3.11/site-packages/isaaclab/source/isaaclab"
    env["PYTHONPATH"] = os.pathsep.join((str(isaaclab_source), str(ROOT / "source"), env.get("PYTHONPATH", "")))
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(PLANNER), "--door-urdf-path", str(urdf)],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return urdf.parent.name, result.returncode == 0 and (urdf.parent / "traj.pkl").is_file(), result.stdout


def main():
    urdfs = sorted(FAMILY.glob("*/mobility.urdf"))
    if len(urdfs) != 1024:
        raise RuntimeError(f"Expected 1024 door assets, found {len(urdfs)}")
    OUT.mkdir(parents=True, exist_ok=True)
    planner_hashes = {
        "compute_waypoint.py": hashlib.sha256(PLANNER.read_bytes()).hexdigest(),
        "offline_pull_door.py": hashlib.sha256(OFFLINE_PULL.read_bytes()).hexdigest(),
    }
    (OUT / "planner_sha256.json").write_text(json.dumps(planner_hashes, indent=2) + "\n")
    started = time.monotonic()
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(generate, urdf) for urdf in urdfs]
        for completed, future in enumerate(as_completed(futures), 1):
            name, ok, output = future.result()
            if not ok:
                failures.append(name)
                (OUT / f"{name}.failed.log").write_text(output)
                print(f"FAILED {name}", flush=True)
            if completed % 16 == 0 or completed == len(futures):
                progress = {"completed": completed, "total": len(futures), "failures": failures,
                            "elapsed_seconds": round(time.monotonic() - started, 1)}
                (OUT / "generation_progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                print(f"{completed}/{len(futures)} generated; failures={len(failures)}", flush=True)
    if failures:
        return 1
    digest = hashlib.sha256()
    for path in sorted(FAMILY.glob("*/traj.pkl")):
        digest.update(path.parent.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    (OUT / "trajectory_manifest_sha256.txt").write_text(digest.hexdigest() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
