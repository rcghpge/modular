# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# Need this import to silence error from circular type dependency between dtype
# and scalar
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class dtype:
    """
    Data type of tensor elements.
    :param id: Integer ID of dtype. Corresponds to IDs in C++ backend.
    :type id: int
    :param: scalar_name: Printable name of dtype.
    :type scalar_name: str
    """

    id: int
    scalar_name: str

    def __str__(self) -> str:
        return f"max.{self.scalar_name}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: dtype) -> bool:
        return self.id == other.id and self.scalar_name == other.scalar_name


# Instantiations of supported dtypes.
bool_ = dtype(0, "bool")
int8 = dtype(4, "int8")
int16 = dtype(5, "int16")
int32 = dtype(6, "int32")
int64 = dtype(7, "int64")
uint8 = dtype(15, "uint8")
uint16 = dtype(16, "uint16")
uint32 = dtype(17, "uint32")
uint64 = dtype(18, "uint64")
float16 = dtype(25, "float16")
float32 = dtype(26, "float32")
float64 = dtype(27, "float64")
