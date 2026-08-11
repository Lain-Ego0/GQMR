"""Error types used by the core data layer."""

from __future__ import annotations


class GQMRError(Exception):
    """Base class for user-facing GQMR failures."""


class MotionValidationError(GQMRError, ValueError):
    """Raised when canonical motion data violates its schema."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        self.field = field
        prefix = f"{field}: " if field else ""
        super().__init__(prefix + message)


class UnsafeMotionFileError(GQMRError, ValueError):
    """Raised before or during loading of an unsafe NPZ container."""


class AssetError(GQMRError, ValueError):
    """Raised when an asset cannot be trusted, installed, or verified."""
