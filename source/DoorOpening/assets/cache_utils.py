from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_ROOT = REPO_ROOT / "IsaacLab_tmp"
CACHE_ROOT_ENV_VAR = "DOOROPENING_ISAACLAB_CACHE_DIR"


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
