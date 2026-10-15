# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Value-slot codec for the Cascade gRPC transport.

Args, kwargs, and return values travel as :py:class:`ValueSlot` envelopes
so they're portable across language implementations. Composite/structured
values are JSON-encoded; binary leaves (``bytes``, ``np.ndarray``) and
in-runtime :py:class:`Result` references get dedicated slots.

See cascade_runtime_v1.proto for reference.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast, get_args, get_origin

import capnp
import numpy as np
from max.experimental.cascade.core import Result, ResultIter, Runtime
from max.experimental.cascade.core.type_hints import async_elem_type
from max.experimental.cascade.interfaces import gen_ai_capnp
from pydantic import BaseModel, TypeAdapter

from . import cascade_runtime_v1_pb2 as pb

# ---------------------------------------------------------------------------
# ValueSlot encode / decode
# ---------------------------------------------------------------------------


def _target_from_runtime(runtime: Any) -> str:
    """Extract the gRPC target address from a runtime carrying one.

    Accepts both :py:class:`GrpcRuntimeClient` and
    :py:class:`SubprocGrpcRuntimeClient` (the latter exposes ``.target``
    too). Lazy import breaks the circular dependency with ``runtime.py``.
    """
    from .runtime import GrpcRuntimeClient, SubprocGrpcRuntimeClient

    if not isinstance(runtime, (GrpcRuntimeClient, SubprocGrpcRuntimeClient)):
        raise TypeError(
            f"Expected a gRPC runtime for the gRPC codec but got"
            f" {type(runtime).__name__!r}"
        )
    return cast(str, runtime.target)


def encode_value_slot(value: Any) -> pb.ValueSlot:
    """Encode one top-level argument, kwarg, or return value into a ``ValueSlot``.

    :py:class:`Result` and :py:class:`ResultIter` values become wire references
    carrying the server ``target`` and ``result_id``; the backing runtime must
    be a gRPC runtime so the target address is available to encode.
    """
    if isinstance(value, ResultIter):
        return pb.ValueSlot(
            stream_ref=pb.ResultStreamReference(
                target=_target_from_runtime(value.runtime),
                result_id=value.result_id,
            )
        )
    if isinstance(value, Result):
        return pb.ValueSlot(
            result_ref=pb.ResultReference(
                target=_target_from_runtime(value.runtime),
                result_id=value.result_id,
            )
        )
    if isinstance(value, _CAPNP_STRUCTS):
        return pb.ValueSlot(capnp=_encode_structs([value], is_list=False))
    if isinstance(value, _CAPNP_LISTS):
        value = list(value)
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, _CAPNP_STRUCTS) for item in value)
    ):
        return pb.ValueSlot(capnp=_encode_structs(value, is_list=True))
    if isinstance(value, np.ndarray):
        return pb.ValueSlot(ndarray=_encode_ndarray(value))
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return pb.ValueSlot(bytes_value=bytes(value))
    return pb.ValueSlot(
        json_string=json.dumps(
            value, default=_json_default, separators=(",", ":")
        )
    )


def is_remote_ref(slot: pb.ValueSlot) -> bool:
    """Checks if the value slot is a reference to a result on another server."""
    kind = slot.WhichOneof("kind")
    return kind == "result_ref" or kind == "stream_ref"


async def decode_value_slot(
    slot: pb.ValueSlot,
    dial: Callable[[str], Runtime],
    type_hint: object | None = None,
) -> Any:
    """Decode one ValueSlot back into a Python value.

    ``Result`` / ``ResultIter`` slots carry the gRPC ``target`` of the runtime
    that owns the referenced result, which may be a *different* server than the
    one we received the slot from. ``dial`` resolves that target by returning a
    lazy runtime client to that worker.

    ``json_string`` slots are validated against ``type_hint`` when one is
    provided, rebuilding ``pydantic.BaseModel`` payloads, containers like
    ``list[Model]``, etc)

    Without a hint they decode as generic JSON.
    """
    kind = slot.WhichOneof("kind")
    if kind == "json_string":
        if type_hint is None:
            return json.loads(slot.json_string)
        # An empty list carries no structs to tag, so it travels as JSON, and
        # only the receiver's hint says it is a list of structs.
        if _is_struct_list(type_hint) and json.loads(slot.json_string) == []:
            return []
        return TypeAdapter(type_hint).validate_json(slot.json_string)
    if kind == "ndarray":
        return _decode_ndarray(slot.ndarray)
    if kind == "result_ref":
        return Result(
            runtime=dial(slot.result_ref.target),
            result_id=slot.result_ref.result_id,
            type_hint=type_hint,
        )
    if kind == "stream_ref":
        return ResultIter(
            runtime=dial(slot.stream_ref.target),
            result_id=slot.stream_ref.result_id,
            type_hint=async_elem_type(type_hint),
        )
    if kind == "bytes_value":
        return bytes(slot.bytes_value)
    if kind == "capnp":
        return _decode_structs(slot.capnp)
    raise TypeError(f"Unsupported ValueSlot kind {kind!r}")


# ---------------------------------------------------------------------------
# args / kwargs convenience wrappers
# ---------------------------------------------------------------------------


def encode_args(args: Sequence[Any]) -> list[pb.ValueSlot]:
    """Encode positional arguments into a list of ValueSlots."""
    return [encode_value_slot(value) for value in args]


async def decode_args(
    args: Sequence[pb.ValueSlot],
    dial: Callable[[str], Runtime],
    arg_types: dict[str, object | None],
) -> list[Any]:
    """Decode a list of ValueSlots back into positional arguments.

    Positional args always bind to the first ``len(args)`` declared
    parameters in order; the rest arrive via kwargs or defaults, so
    ``arg_types`` (all declared params, declaration-ordered) may be longer.
    """
    if len(args) > len(arg_types):
        raise ValueError(
            f"Got {len(args)} positional args but the method declares only"
            f" {len(arg_types)} parameters"
        )

    return [
        await decode_value_slot(value, dial, type_hint)
        for value, type_hint in zip(
            args, list(arg_types.values())[: len(args)], strict=True
        )
    ]


def encode_kwargs(kwargs: Mapping[str, Any]) -> dict[str, pb.ValueSlot]:
    """Encode keyword arguments into a name -> ValueSlot mapping."""
    return {key: encode_value_slot(value) for key, value in kwargs.items()}


async def decode_kwargs(
    kwargs: Mapping[str, pb.ValueSlot],
    dial: Callable[[str], Runtime],
    kwarg_types: dict[str, object | None],
) -> dict[str, Any]:
    """Decode a name -> ValueSlot mapping back into keyword arguments."""
    for kwarg in kwargs:
        if kwarg not in kwarg_types:
            raise ValueError(
                f"Got unexpected kwarg {kwarg!r} not declared by the method"
            )
    return {
        kwarg: await decode_value_slot(value, dial, kwarg_types[kwarg])
        for kwarg, value in kwargs.items()
    }


# ---------------------------------------------------------------------------
# Error envelope encode / decode
# ---------------------------------------------------------------------------


def encode_error(exc: BaseException) -> pb.ErrorEnvelope:
    """Encode an exception into an :py:class:`ErrorEnvelope`.

    ``message`` and ``type_name`` are always populated and are the
    interoperable surface;
    """
    return pb.ErrorEnvelope(
        message=str(exc),
        type_name=type(exc).__name__,
    )


def decode_error(envelope: pb.ErrorEnvelope) -> BaseException:
    """Report errors as :py:class:`RuntimeError` carrying ``message`` and ``type_name``.

    Errors can come from other language environments, so the original type is not preserved.
    """
    return RuntimeError(f"{envelope.type_name}: {envelope.message}")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_CAPNP_STRUCTS = (capnp._DynamicStructBuilder, capnp._DynamicStructReader)
_CAPNP_LISTS = (capnp._DynamicListBuilder, capnp._DynamicListReader)

# Each gen_ai struct class, keyed by schema id, to read a `CapnpStructs` slot.
_STRUCT_CLASSES = {
    struct.schema.node.id: struct
    for struct in vars(gen_ai_capnp).values()
    if isinstance(struct, capnp._StructModule)
}


def _is_struct_list(type_hint: object) -> bool:
    if get_origin(type_hint) is not list:
        return False
    (item,) = get_args(type_hint)
    return isinstance(item, capnp._StructModule)


def _encode_structs(structs: Sequence[Any], is_list: bool) -> pb.CapnpStructs:
    """Encode Cap'n Proto structs of one type, each as its own message."""
    type_id = structs[0].schema.node.id
    if any(struct.schema.node.id != type_id for struct in structs):
        raise TypeError("a list of Cap'n Proto structs must share one type")
    return pb.CapnpStructs(
        type_id=type_id,
        messages=[_message_bytes(struct) for struct in structs],
        is_list=is_list,
    )


def _message_bytes(
    struct: capnp._DynamicStructBuilder | capnp._DynamicStructReader,
) -> bytes:
    """Serialize ``struct`` as a message of its own.

    Only a message's root builder serializes, so a reader or a struct nested in
    a larger message (a list element, a field) is copied into a new message.
    """
    if isinstance(struct, capnp._DynamicStructReader):
        struct = struct.as_builder()
    elif not struct.is_root:
        struct = struct.copy()
    return struct.to_bytes()


def _decode_structs(slot: pb.CapnpStructs) -> Any:
    """Read a ``CapnpStructs`` slot as readers over its messages."""
    struct = _STRUCT_CLASSES[slot.type_id]
    readers = []
    for message in slot.messages:
        # Readers keep their message alive after the context closes.
        with struct.from_bytes(message) as reader:
            readers.append(reader)
    return readers if slot.is_list else readers[0]


def _encode_ndarray(value: np.ndarray) -> pb.NDArray:
    """Encode a numpy array into an ``NDArray`` protobuf message."""
    return pb.NDArray(
        dtype=value.dtype.str,
        shape=list(value.shape),
        data=value.tobytes(),
    )


def _decode_ndarray(value: pb.NDArray) -> np.ndarray:
    """Decode an ``NDArray`` protobuf message back into a numpy array."""
    dtype = np.dtype(value.dtype)
    return (
        np.frombuffer(value.data, dtype=dtype)
        .copy()
        .reshape(tuple(value.shape))
    )


def _json_default(value: Any) -> Any:
    """``json.dumps`` fallback hook for types stdlib JSON doesn't know."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )
