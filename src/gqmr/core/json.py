"""Strict JSON parsing shared by user-controlled formats."""

from __future__ import annotations

import json
from typing import Any

from gqmr.core.errors import GQMRError


class StrictJSONError(GQMRError, ValueError):
    """Raised for duplicate keys, non-finite constants, or malformed JSON."""


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def loads_strict_json(text: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                StrictJSONError(f"non-finite JSON constant {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise StrictJSONError(f"malformed JSON: {error}") from error
