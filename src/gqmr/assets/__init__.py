"""Trusted external robot asset management."""

from gqmr.assets.catalog import AssetSpec, available_assets, get_asset_spec
from gqmr.assets.manager import (
    AssetStatus,
    default_asset_root,
    install_asset,
    pack_asset,
    status_asset,
    unpack_asset,
)

__all__ = [
    "AssetSpec",
    "AssetStatus",
    "available_assets",
    "default_asset_root",
    "get_asset_spec",
    "install_asset",
    "pack_asset",
    "status_asset",
    "unpack_asset",
]
