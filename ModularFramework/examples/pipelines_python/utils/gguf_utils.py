# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Common functions for reading data from a GGUF file."""
from typing import Any, List, Optional

import numpy as np
from gguf import GGUFValueType


def _to_string(arr: np.ndarray) -> str:
    return arr.tobytes().decode()


def read_string(reader, key) -> Optional[str]:
    field = reader.get_field(key)
    if field is None:
        return None
    assert field.types[0] == GGUFValueType.STRING
    return _to_string(field.parts[field.data[0]])


def read_string_array(reader, key) -> Optional[List[str]]:
    field = reader.get_field(key)
    if field is None:
        return None
    assert field.types[0] == GGUFValueType.ARRAY
    assert field.types[1] == GGUFValueType.STRING
    return [_to_string(field.parts[x]) for x in field.data]


def read_number(reader, key) -> Optional[Any]:
    field = reader.get_field(key)
    if field is None:
        return None
    return field.parts[field.data[0]][0]


def read_array(reader, key) -> Optional[List[Any]]:
    field = reader.get_field(key)
    if field is None:
        return None
    return [field.parts[x][0] for x in field.data]
