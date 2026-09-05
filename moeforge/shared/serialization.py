"""Strict JSON encoding using the shared model serializer."""

import json
from typing import Any

from moeforge.shared.models import to_json_dict


def to_json(value: Any) -> str:
    return json.dumps(to_json_dict(value), allow_nan=False, sort_keys=True)
