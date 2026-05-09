from __future__ import annotations

import contextlib
import hashlib
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_ROOT = REPO_ROOT / "IsaacLab_tmp"
CACHE_ROOT_ENV_VAR = "DOOROPENING_ISAACLAB_CACHE_DIR"
CACHE_FORCE_CONVERSION_ENV_VAR = "DOOROPENING_FORCE_USD_CONVERSION"
DISTRIBUTED_RUN_TAG_ENV_VAR = "DOOROPENING_DISTRIBUTED_RUN_TAG"


def get_converter_cache_root() -> Path:
    """Return the absolute cache root used for generated Isaac Lab USD assets."""
    configured_root = os.environ.get(CACHE_ROOT_ENV_VAR)
    cache_root = Path(configured_root).expanduser() if configured_root else DEFAULT_CACHE_ROOT
    if not cache_root.is_absolute():
        cache_root = (REPO_ROOT / cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def resolve_converter_cache_dir(
    asset_path: str | Path,
    *,
    asset_root: str | Path | None = None,
    namespace: str = "urdf",
    variant: str | None = None,
) -> str:
    """Build a stable absolute cache directory for a converted asset."""
    asset_path = Path(asset_path).expanduser().resolve()

    if asset_root is not None:
        asset_root = Path(asset_root).expanduser().resolve()
        try:
            relative_asset_path = asset_path.relative_to(asset_root)
        except ValueError:
            relative_asset_path = Path(asset_path.name)
    else:
        relative_asset_path = Path(asset_path.name)

    cache_dir = get_converter_cache_root() / namespace
    if variant:
        cache_dir /= variant
    if relative_asset_path.parent != Path("."):
        cache_dir /= relative_asset_path.parent
    cache_dir /= asset_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def should_force_usd_conversion(default: bool = False) -> bool:
    """Return whether URDF converters should ignore the cached USD output."""
    configured = os.environ.get(CACHE_FORCE_CONVERSION_ENV_VAR)
    if configured is None:
        return default
    return configured.strip().lower() not in {"", "0", "false", "no", "off"}


def should_force_asset_usd_conversion(
    asset_path: str | Path,
    usd_dir: str | Path,
    *,
    default: bool = False,
) -> bool:
    """Force reconversion when the source asset tree is newer than the cached USD."""
    if should_force_usd_conversion(default=default):
        return True
    return _is_source_asset_newer_than_cache(str(asset_path), str(usd_dir))


@lru_cache(maxsize=None)
def _is_source_asset_newer_than_cache(asset_path: str, usd_dir: str) -> bool:
    asset_path_obj = Path(asset_path).expanduser().resolve()
    usd_dir_obj = Path(usd_dir).expanduser().resolve()

    if not asset_path_obj.exists() or not usd_dir_obj.exists():
        return False

    cached_usd_files = [path for path in usd_dir_obj.rglob("*.usd") if path.is_file()]
    if not cached_usd_files:
        return False

    newest_cached_usd_mtime = max(path.stat().st_mtime for path in cached_usd_files)
    source_root = asset_path_obj.parent
    relevant_suffixes = {
        ".urdf",
        ".obj",
        ".stl",
        ".dae",
        ".fbx",
        ".gltf",
        ".glb",
        ".mtl",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tga",
    }
    source_files = [
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in relevant_suffixes
    ]
    if not source_files:
        source_files = [asset_path_obj]
    newest_source_mtime = max(path.stat().st_mtime for path in source_files)

    return newest_source_mtime > newest_cached_usd_mtime + 1e-6


def configure_distributed_run_tag() -> None:
    """Populate a stable per-launch tag for distributed asset-prewarm coordination."""
    if os.environ.get(DISTRIBUTED_RUN_TAG_ENV_VAR):
        return

    run_tag = os.environ.get("SLURM_JOB_ID")
    if not run_tag:
        torchelastic_run_id = os.environ.get("TORCHELASTIC_RUN_ID")
        if torchelastic_run_id and torchelastic_run_id.lower() != "none":
            run_tag = torchelastic_run_id
    if not run_tag:
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "0")
        run_tag = f"{master_addr}_{master_port}"

    safe_run_tag = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in run_tag)
    os.environ[DISTRIBUTED_RUN_TAG_ENV_VAR] = safe_run_tag or "default"


def is_distributed_launch() -> bool:
    """Return whether the current process belongs to a multi-rank launch."""
    return max(_read_env_int("WORLD_SIZE", 1), _read_env_int("LOCAL_WORLD_SIZE", 1)) > 1


def preconvert_shared_urdf_assets(
    *,
    timeout_s: float = 1800.0,
    poll_interval_s: float = 1.0,
    door_configs: object | None = None,
    verbose: bool = False,
) -> None:
    """Convert shared DoorOpening URDF assets on rank 0 before other ranks construct the env."""
    from isaaclab.sim.converters import UrdfConverter

    from DoorOpening.assets.glorbot.glorbot_cfg import GLORBOT_CONFIG
    if door_configs is None:
        from DoorOpening.assets.door.door_cfg import ALL_DOOR_CONFIGS as door_configs

    if not is_distributed_launch():
        return

    asset_cfgs = [GLORBOT_CONFIG.spawn]
    asset_cfgs.extend(_iter_multi_asset_urdf_cfgs(getattr(door_configs, "spawn", None)))
    unique_cfgs = _deduplicate_urdf_cfgs(asset_cfgs)
    if not unique_cfgs:
        return

    configure_distributed_run_tag()
    cache_root = get_converter_cache_root()
    sync_root = cache_root / ".asset_prewarm"
    sync_root.mkdir(parents=True, exist_ok=True)

    cfg_key = hashlib.sha1(
        "\n".join(
            f"{cfg.asset_path}|{cfg.usd_dir}|{cfg.usd_file_name}|{int(bool(getattr(cfg, 'force_usd_conversion', False)))}"
            for cfg in unique_cfgs
        ).encode()
    ).hexdigest()[:12]
    run_tag = os.environ[DISTRIBUTED_RUN_TAG_ENV_VAR]
    done_path = sync_root / f"urdf_convert.{cfg_key}.{run_tag}.done"
    error_path = sync_root / f"urdf_convert.{cfg_key}.{run_tag}.failed"
    lock_path = sync_root / f"urdf_convert.{cfg_key}.lock"

    rank = _read_env_int("RANK", _read_env_int("LOCAL_RANK", 0))
    if rank == 0:
        lock_fd = _acquire_lock(lock_path, timeout_s=timeout_s, poll_interval_s=poll_interval_s)
        try:
            with contextlib.suppress(FileNotFoundError):
                error_path.unlink()
            if not done_path.exists():
                if verbose:
                    print(f"[asset-prewarm] Rank 0 converting {len(unique_cfgs)} shared URDF assets.")
                for idx, cfg in enumerate(unique_cfgs, start=1):
                    if verbose and (idx == 1 or idx == len(unique_cfgs) or idx % 10 == 0):
                        asset_label = Path(str(getattr(cfg, "asset_path", ""))).parent.name
                        print(f"[asset-prewarm] Converting {idx}/{len(unique_cfgs)}: {asset_label}")
                    UrdfConverter(cfg)
                done_path.write_text(str(time.time()), encoding="utf-8")
                if verbose:
                    print(f"[asset-prewarm] Rank 0 finished shared URDF conversion: {done_path}")
            elif verbose:
                print(f"[asset-prewarm] Shared URDF conversion already complete: {done_path}")
        except Exception as exc:
            error_path.write_text(repr(exc), encoding="utf-8")
            raise
        finally:
            _release_lock(lock_fd, lock_path)
    else:
        deadline = time.monotonic() + timeout_s
        if verbose:
            print(
                "[asset-prewarm] Rank {} waiting up to {:.0f}s for rank 0 to convert {} shared URDF assets.".format(
                    rank,
                    float(timeout_s),
                    len(unique_cfgs),
                )
            )
        last_log_time = time.monotonic()
        while not done_path.exists():
            if error_path.exists():
                message = error_path.read_text(encoding="utf-8").strip() or "unknown error"
                raise RuntimeError(f"Rank 0 URDF preconversion failed: {message}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for rank 0 to preconvert shared URDF assets "
                    f"after {timeout_s:.0f}s. assets={len(unique_cfgs)} done_path={done_path} lock_path={lock_path}"
                )
            if verbose and time.monotonic() - last_log_time >= 300.0:
                print(f"[asset-prewarm] Rank {rank} is still waiting for rank 0: {done_path}")
                last_log_time = time.monotonic()
            time.sleep(poll_interval_s)
        if verbose:
            print(f"[asset-prewarm] Rank {rank} detected completed shared URDF conversion.")


def _iter_multi_asset_urdf_cfgs(spawn_cfg: object) -> Iterable[object]:
    assets_cfg = getattr(spawn_cfg, "assets_cfg", None)
    if assets_cfg is None:
        return ()
    return tuple(cfg for cfg in assets_cfg if hasattr(cfg, "asset_path") and hasattr(cfg, "usd_dir"))


def _deduplicate_urdf_cfgs(cfgs: Iterable[object]) -> list[object]:
    unique_cfgs: list[object] = []
    seen_keys: set[tuple[str, str | None, str | None, bool]] = set()

    for cfg in cfgs:
        key = (
            str(getattr(cfg, "asset_path", "")),
            getattr(cfg, "usd_dir", None),
            getattr(cfg, "usd_file_name", None),
            bool(getattr(cfg, "force_usd_conversion", False)),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_cfgs.append(cfg)

    return unique_cfgs


def _read_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _acquire_lock(lock_path: Path, *, timeout_s: float, poll_interval_s: float) -> int:
    deadline = time.monotonic() + timeout_s
    stale_after_s = max(timeout_s, 60.0)

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode())
            return fd
        except FileExistsError:
            try:
                lock_age_s = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if lock_age_s > stale_after_s:
                with contextlib.suppress(FileNotFoundError):
                    lock_path.unlink()
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for asset prewarm lock: {lock_path}")
            time.sleep(poll_interval_s)


def _release_lock(lock_fd: int, lock_path: Path) -> None:
    os.close(lock_fd)
    with contextlib.suppress(FileNotFoundError):
        lock_path.unlink()
