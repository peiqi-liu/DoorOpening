from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_ROOT = REPO_ROOT / "IsaacLab_tmp"
CACHE_ROOT_ENV_VAR = "DOOROPENING_ISAACLAB_CACHE_DIR"
CACHE_FORCE_CONVERSION_ENV_VAR = "DOOROPENING_FORCE_USD_CONVERSION"
DISTRIBUTED_RUN_TAG_ENV_VAR = "DOOROPENING_DISTRIBUTED_RUN_TAG"
# Set by preconvert_shared_urdf_assets once the serialized, validated conversion for THIS run has
# finished. While set, spawn-time cfgs LOAD the freshly-converted USDs instead of reconverting them
# (which, under force_usd_conversion=True + distributed, races across ranks on the shared USD files
# and yields a half-written, rigid-body-less door -> a random env dies with "no rigid bodies").
ASSETS_PRECONVERTED_ENV_VAR = "DOOROPENING_ASSETS_PRECONVERTED_TAG"


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
    """Return whether URDF converters should ignore the cached USD output.

    FORCED ON: always reconvert URDF->USD on every run, regardless of the
    ``DOOROPENING_FORCE_USD_CONVERSION`` env var, the ``default`` argument, or the cache state.
    This guarantees the robot AND all doors are rebuilt from the current URDFs, so a stale or
    partial cached USD can never hang initialize_physics. Applies to glorbot (glorbot_cfg) and
    every door (door_cfg / multi_door_cfg) via should_force_asset_usd_conversion.
    To restore cache-aware behavior, remove the ``return True`` line below.
    """
    return True
    # --- Previous cache-aware behavior (disabled by the forced return above) ---
    # configured = os.environ.get(CACHE_FORCE_CONVERSION_ENV_VAR)
    # if configured is None:
    #     return default
    # return configured.strip().lower() not in {"", "0", "false", "no", "off"}


def should_force_asset_usd_conversion(
    asset_path: str | Path,
    usd_dir: str | Path,
    *,
    default: bool = False,
) -> bool:
    """Force reconversion when the source asset tree is newer than the cached USD."""
    # If the serialized+validated preconvert already produced FRESH USDs for this run, LOAD them --
    # do NOT reconvert at spawn. In distributed, each rank rebuilds its door cfgs and would
    # otherwise reconvert the SAME shared door USD files concurrently (force_usd_conversion=True);
    # a rank then reads a half-written USD (geometry, no rigid bodies) and a random env crashes.
    # Only reconvert here if the cached USD is genuinely broken (self-repair fallback).
    if _assets_preconverted_this_run():
        return _is_cached_usd_missing_default_prim(asset_path, usd_dir)
    if should_force_usd_conversion(default=default):
        return True
    if _is_source_asset_newer_than_cache(str(asset_path), str(usd_dir)):
        return True
    return _is_cached_usd_missing_default_prim(asset_path, usd_dir)


def _assets_preconverted_this_run() -> bool:
    """True once this run's serialized preconvert has finished (so spawn should LOAD, not reconvert)."""
    tag = os.environ.get(ASSETS_PRECONVERTED_ENV_VAR)
    return bool(tag) and tag == os.environ.get(DISTRIBUTED_RUN_TAG_ENV_VAR, tag)


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

    # Guard every URDF->USD conversion in this process -- the serialized preconvert below AND
    # any on-demand spawn-time conversion -- with a per-asset lock + private staging + atomic
    # publish, so no two ranks ever write the same shared cache dir concurrently.
    install_rank_safe_urdf_converter()

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
            _gc_stale_prewarm_markers(sync_root)
            with contextlib.suppress(FileNotFoundError):
                error_path.unlink()
            # Only trust an existing .done marker if the USDs it claims actually exist on disk.
            # run_tag can collide across launches (the MASTER_ADDR/PORT fallback is reused), so a
            # stale marker from a previous run -- or a partially cleared cache -- must never let
            # ranks skip straight to spawn and convert on-demand (which races on the shared NFS
            # cache: two ranks writing frame.tmp.usd -> NFS silly-rename -> NULL UsdStage -> abort).
            missing = _doors_missing_usd(unique_cfgs)
            if done_path.exists() and not missing:
                if verbose:
                    print(f"[asset-prewarm] Shared URDF conversion already complete: {done_path}")
            else:
                if done_path.exists():
                    if verbose:
                        print(
                            f"[asset-prewarm] Ignoring stale/incomplete .done marker (missing "
                            f"{len(missing)} USDs, first={missing[0] if missing else '-'}); reconverting."
                        )
                    with contextlib.suppress(FileNotFoundError):
                        done_path.unlink()
                if verbose:
                    print(f"[asset-prewarm] Rank 0 converting {len(unique_cfgs)} shared URDF assets.")
                for idx, cfg in enumerate(unique_cfgs, start=1):
                    if verbose and (idx == 1 or idx == len(unique_cfgs) or idx % 10 == 0):
                        asset_label = Path(str(getattr(cfg, "asset_path", ""))).parent.name
                        print(f"[asset-prewarm] Converting {idx}/{len(unique_cfgs)}: {asset_label}")
                    _convert_and_validate_urdf(UrdfConverter, cfg)
                still_missing = _doors_missing_usd(unique_cfgs)
                if still_missing:
                    raise RuntimeError(
                        "Shared URDF preconversion finished but "
                        f"{len(still_missing)} USDs are still missing; first={still_missing[0]}"
                    )
                # Write .done only after verifying every USD is present, so the marker never lies.
                done_path.write_text(str(time.time()), encoding="utf-8")
                if verbose:
                    print(f"[asset-prewarm] Rank 0 finished shared URDF conversion: {done_path}")
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
        while True:
            # Require BOTH the .done marker AND the actual USDs on disk. This makes a stale marker
            # harmless: keep waiting until rank 0 of THIS launch has really produced every USD,
            # instead of proceeding to a spawn-time on-demand conversion that races across ranks.
            # (done_path.exists() short-circuits the per-door stat sweep until the marker appears.)
            if done_path.exists() and not _doors_missing_usd(unique_cfgs):
                break
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

    # The serialized preconvert above produced FRESH, validated USDs for this run. Signal spawn to
    # LOAD them instead of reconverting (which would race across ranks -> half-written door USDs ->
    # "no rigid bodies" on a random env). Door cfgs are rebuilt after this and consult
    # _assets_preconverted_this_run() via should_force_asset_usd_conversion; the robot + the initial
    # cfgs here (not rebuilt) are switched to load directly.
    os.environ[ASSETS_PRECONVERTED_ENV_VAR] = os.environ.get(DISTRIBUTED_RUN_TAG_ENV_VAR, "1")
    for cfg in unique_cfgs:
        with contextlib.suppress(Exception):
            cfg.force_usd_conversion = False


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


def _convert_and_validate_urdf(converter_cls: type, cfg: object) -> None:
    """Run Isaac Lab's converter and reject stale/corrupt USD cache outputs."""
    _log_asset_event("convert-start", cfg)
    converter_cls(cfg)
    _log_asset_event("convert-finish", cfg)
    if not _is_cached_usd_missing_default_prim(
        getattr(cfg, "asset_path", ""),
        getattr(cfg, "usd_dir", ""),
        getattr(cfg, "usd_file_name", None),
    ):
        return

    # The converter may have skipped a stale cache because the config hash matched.
    # Force one repair attempt before failing the distributed launch.
    if not bool(getattr(cfg, "force_usd_conversion", False)):
        _log_asset_event("convert-repair", cfg)
        setattr(cfg, "force_usd_conversion", True)
        with contextlib.suppress(FileNotFoundError):
            (Path(str(getattr(cfg, "usd_dir", ""))) / ".asset_hash").unlink()
        converter_cls(cfg)
        _log_asset_event("convert-repair-finish", cfg)

    if _is_cached_usd_missing_default_prim(
        getattr(cfg, "asset_path", ""),
        getattr(cfg, "usd_dir", ""),
        getattr(cfg, "usd_file_name", None),
    ):
        with contextlib.suppress(Exception):
            _validate_generated_usd_outputs(cfg)
        usd_path = _expected_usd_path(
            getattr(cfg, "asset_path", ""),
            getattr(cfg, "usd_dir", ""),
            getattr(cfg, "usd_file_name", None),
        )
        with contextlib.suppress(FileNotFoundError):
            (Path(str(getattr(cfg, "usd_dir", ""))) / ".asset_hash").unlink()
        raise RuntimeError(
            "URDF conversion produced an invalid USD cache: "
            f"asset_path={getattr(cfg, 'asset_path', None)} usd_path={usd_path}"
        )
    _validate_generated_usd_outputs(cfg)


def _is_cached_usd_missing_default_prim(
    asset_path: str | Path,
    usd_dir: str | Path,
    usd_file_name: str | None = None,
) -> bool:
    """Return True when the expected cached USD is present but unusable as a referenced asset."""
    usd_path = _expected_usd_path(asset_path, usd_dir, usd_file_name)
    if not usd_path.is_file():
        return False

    try:
        from pxr import Usd
    except Exception:
        return b"defaultPrim" not in usd_path.read_bytes()

    try:
        stage = _safe_open_usd_stage(
            usd_path,
            context=f"cached-usd-validation asset_path={Path(str(asset_path)).expanduser().resolve()}",
        )
    except Exception:
        return True
    default_prim = stage.GetDefaultPrim()
    if not bool(default_prim and default_prim.IsValid()):
        return True
    return _generated_usd_has_tmp_references(usd_dir_obj=Path(str(usd_dir)).expanduser().resolve())


def _expected_usd_path(
    asset_path: str | Path,
    usd_dir: str | Path,
    usd_file_name: str | None = None,
) -> Path:
    usd_file_name = usd_file_name or Path(str(asset_path)).stem
    if not (usd_file_name.endswith(".usd") or usd_file_name.endswith(".usda")):
        usd_file_name += ".usd"
    return Path(str(usd_dir)).expanduser() / usd_file_name


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


_ORIG_URDF_CONVERTER_INIT = None


def install_rank_safe_urdf_converter() -> None:
    """Serialize + isolate URDF->USD conversions so ranks never write a shared cache dir at once.

    Wraps ``UrdfConverter.__init__`` so that whenever a conversion would actually WRITE into a
    shared ``usd_dir`` (the serialized preconvert OR an on-demand spawn), it:
      1. takes a per-asset file lock on ``<usd_dir>.convert.lock``,
      2. converts into a private ``<usd_dir>.staging.<pid>`` directory, then
      3. atomically publishes the finished files into the shared dir (``os.replace`` per file).
    Two processes converting the same door can therefore never collide on the intermediate
    ``configuration/*.tmp.usd`` files (which on NFS silly-renames the open file, fails the write,
    returns a NULL ``UsdStage`` and aborts the process). Generated USDs use relative references,
    so moving the tree into place is safe. Idempotent; a pure cache-hit load takes no lock and
    keeps the stock (fast) behavior.
    """
    global _ORIG_URDF_CONVERTER_INIT
    from isaaclab.sim.converters import UrdfConverter

    if getattr(UrdfConverter, "_dooropening_rank_safe", False):
        return
    _ORIG_URDF_CONVERTER_INIT = UrdfConverter.__init__

    def _rank_safe_init(self, cfg):
        shared_dir = getattr(cfg, "usd_dir", None)
        if shared_dir is None:
            # No shared cache dir -> IsaacLab already uses a random /tmp dir (race-safe).
            _ORIG_URDF_CONVERTER_INIT(self, cfg)
            return
        shared_dir = str(shared_dir)
        force = bool(getattr(cfg, "force_usd_conversion", False))
        if not force and _cached_conversion_is_current(UrdfConverter, cfg, shared_dir):
            # Cache hit: the stock init just loads, no write -> no lock needed.
            _ORIG_URDF_CONVERTER_INIT(self, cfg)
            return
        with _asset_convert_lock(shared_dir):
            # Another process may have published a valid USD while we waited for the lock.
            if _cached_conversion_is_current(UrdfConverter, cfg, shared_dir):
                saved_force = getattr(cfg, "force_usd_conversion", False)
                try:
                    cfg.force_usd_conversion = False
                    _ORIG_URDF_CONVERTER_INIT(self, cfg)
                finally:
                    cfg.force_usd_conversion = saved_force
                return
            staging = Path(f"{shared_dir}.staging.{os.getpid()}")
            _rmtree(staging)
            saved_usd_dir = cfg.usd_dir
            saved_force = getattr(cfg, "force_usd_conversion", False)
            try:
                cfg.usd_dir = str(staging)
                cfg.force_usd_conversion = True  # staging is empty -> must convert
                _ORIG_URDF_CONVERTER_INIT(self, cfg)
                _publish_converted_dir(staging, Path(shared_dir))
            finally:
                cfg.usd_dir = saved_usd_dir
                cfg.force_usd_conversion = saved_force
                _rmtree(staging)
            # Repoint the converter at the published location so ``usd_path`` is correct.
            self._usd_dir = shared_dir

    UrdfConverter.__init__ = _rank_safe_init
    UrdfConverter._dooropening_rank_safe = True


def _cached_conversion_is_current(converter_cls: type, cfg: object, usd_dir: str) -> bool:
    """True when a converted USD for ``cfg`` already exists in ``usd_dir`` with a matching hash."""
    usd_path = _expected_usd_path(
        getattr(cfg, "asset_path", ""), usd_dir, getattr(cfg, "usd_file_name", None)
    )
    if not usd_path.is_file():
        return False
    try:
        saved_hash = (Path(usd_dir) / ".asset_hash").read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return False
    try:
        current_hash = converter_cls._config_to_hash(cfg)
    except Exception:
        return False
    return saved_hash == current_hash


@contextlib.contextmanager
def _asset_convert_lock(usd_dir: str, *, timeout_s: float = 900.0, poll_interval_s: float = 0.5):
    lock_path = Path(f"{usd_dir}.convert.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = _acquire_lock(lock_path, timeout_s=timeout_s, poll_interval_s=poll_interval_s)
    try:
        yield
    finally:
        _release_lock(lock_fd, lock_path)


def _publish_converted_dir(staging_dir: Path, shared_dir: Path) -> None:
    """Atomically move freshly converted files from a private staging dir into the shared cache."""
    shared_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(staging_dir.rglob("*")):
        if not src.is_file():
            continue
        if src.name.endswith(".tmp.usd"):
            # Intermediate importer artifacts; the published USDs reference the flattened files.
            continue
        rel = src.relative_to(staging_dir)
        dst = shared_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(src), str(dst))  # atomic within the same filesystem; overwrites stale files


def _rmtree(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(path, ignore_errors=True)


def _doors_missing_usd(cfgs: Iterable[object]) -> list[str]:
    """Return source asset paths whose expected converted USD is not present on disk."""
    missing: list[str] = []
    for cfg in cfgs:
        asset_path = getattr(cfg, "asset_path", "")
        usd_dir = getattr(cfg, "usd_dir", "")
        if not usd_dir:
            continue
        usd_path = _expected_usd_path(asset_path, usd_dir, getattr(cfg, "usd_file_name", None))
        if not usd_path.is_file():
            missing.append(str(asset_path))
    return missing


def _gc_stale_prewarm_markers(sync_root: Path, max_age_s: float = 3 * 24 * 3600.0) -> None:
    """Prune old .done/.failed/.lock markers so a stale one can never be mistaken for this run."""
    now = time.time()
    for pattern in ("*.done", "*.failed", "*.lock"):
        for marker in sync_root.glob(pattern):
            try:
                if now - marker.stat().st_mtime > max_age_s:
                    marker.unlink()
            except (FileNotFoundError, OSError):
                continue


def _safe_open_usd_stage(path: str | Path, *, context: str) -> object:
    from pxr import Usd

    usd_path = Path(path).expanduser().resolve()
    try:
        stage = Usd.Stage.Open(str(usd_path))
    except Exception as exc:
        raise RuntimeError(f"Usd.Stage.Open failed for path={usd_path} context={context}") from exc
    if stage is None:
        raise RuntimeError(f"Usd.Stage.Open returned None for path={usd_path} context={context}")
    return stage


def _validate_generated_usd_outputs(cfg: object) -> None:
    asset_path = Path(str(getattr(cfg, "asset_path", ""))).expanduser().resolve()
    usd_dir = Path(str(getattr(cfg, "usd_dir", ""))).expanduser().resolve()
    final_usd_path = _expected_usd_path(
        asset_path,
        usd_dir,
        getattr(cfg, "usd_file_name", None),
    )
    if not final_usd_path.is_file():
        raise RuntimeError(
            f"Expected converted USD is missing: asset_path={asset_path} final_usd_path={final_usd_path}"
        )

    stage = _safe_open_usd_stage(
        final_usd_path,
        context=f"post-convert-final-usd asset_path={asset_path}",
    )
    default_prim = stage.GetDefaultPrim()
    if not bool(default_prim and default_prim.IsValid()):
        raise RuntimeError(
            f"Converted USD has no valid default prim: asset_path={asset_path} final_usd_path={final_usd_path}"
        )

    tmp_reference_files = _generated_usd_tmp_reference_files(usd_dir)
    if tmp_reference_files:
        raise RuntimeError(
            "Generated USD cache still references intermediate .tmp.usd files: "
            f"asset_path={asset_path} final_usd_path={final_usd_path} "
            f"referencing_files={[str(path) for path in tmp_reference_files]}"
        )


def _generated_usd_has_tmp_references(*, usd_dir_obj: Path) -> bool:
    return bool(_generated_usd_tmp_reference_files(usd_dir_obj))


def _generated_usd_tmp_reference_files(usd_dir_obj: Path) -> list[Path]:
    tmp_reference_files: list[Path] = []
    if not usd_dir_obj.exists():
        return tmp_reference_files
    for usd_file in sorted(usd_dir_obj.rglob("*.usd")):
        try:
            usd_bytes = usd_file.read_bytes()
        except OSError:
            continue
        if b".tmp.usd" in usd_bytes:
            tmp_reference_files.append(usd_file)
    return tmp_reference_files


def _log_asset_event(event: str, cfg: object) -> None:
    asset_path = Path(str(getattr(cfg, "asset_path", ""))).expanduser().resolve()
    usd_dir = Path(str(getattr(cfg, "usd_dir", ""))).expanduser().resolve()
    final_usd_path = _expected_usd_path(
        asset_path,
        usd_dir,
        getattr(cfg, "usd_file_name", None),
    )
    tmp_usd_paths = _candidate_tmp_usd_paths(asset_path, usd_dir)
    message = (
        f"[asset-prewarm] event={event} {_rank_context_str()} pid={os.getpid()} "
        f"asset_id={asset_path.parent.name} source_path={asset_path} "
        f"tmp_usd_paths={','.join(str(path) for path in tmp_usd_paths) or '-'} "
        f"final_usd_path={final_usd_path}"
    )
    print(message)


def _rank_context_str() -> str:
    return (
        f"rank={_read_env_int('RANK', 0)} "
        f"local_rank={_read_env_int('LOCAL_RANK', 0)} "
        f"world_size={_read_env_int('WORLD_SIZE', 1)}"
    )


def _candidate_tmp_usd_paths(asset_path: Path, usd_dir: Path) -> list[Path]:
    configuration_dir = usd_dir / "configuration"
    return [
        configuration_dir / f"{mesh_path.stem}.tmp.usd"
        for mesh_path in _iter_asset_mesh_paths(asset_path)
    ]


def _iter_asset_mesh_paths(asset_path: Path) -> list[Path]:
    if asset_path.suffix.lower() != ".urdf":
        return []

    try:
        root = ET.parse(asset_path).getroot()
    except ET.ParseError:
        return []

    mesh_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for mesh_element in root.findall(".//mesh"):
        mesh_file = mesh_element.attrib.get("filename")
        if not mesh_file:
            continue
        mesh_path = (asset_path.parent / mesh_file).resolve()
        if mesh_path in seen_paths:
            continue
        seen_paths.add(mesh_path)
        mesh_paths.append(mesh_path)
    return mesh_paths
