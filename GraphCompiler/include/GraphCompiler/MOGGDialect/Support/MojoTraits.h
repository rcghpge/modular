//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Verbatim Mojo-source string constants the GraphCompiler emits once per module
// for the advanced-fusion MOGG-ABI lane: the buffer-capture identity plus the
// input-load / output-store / transform fusion functor definitions. Split out
// of MOGGOps.cpp so the emit paths there stay readable; the consumers there
// reference these constants directly.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MOJOTRAITS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MOJOTRAITS_H

#include "llvm/ADT/StringRef.h"

namespace M::MOGG {

// Minimal buffer identity at the MOGG-ABI region→stub boundary (pointer +
// layout only; no trait conformances). Execute-facing dense/fused wrappers are
// assembled inside the stub body from capture + contract metadata.
static constexpr llvm::StringLiteral kTensorBufferCaptureDefs = R"mojo(
from layout import TensorLayout


struct _MoggTensorBufferCapture[
    _dtype: DType,
    _LayoutType: TensorLayout,
](Copyable, Movable):
    comptime dtype = Self._dtype
    comptime LayoutType = Self._LayoutType
    var handle: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var _layout: Self._LayoutType

    @always_inline
    def __init__(
        out self,
        ptr: Pointer[Scalar[Self.dtype], MutUntrackedOrigin],
        layout: Self.LayoutType,
    ):
        self.handle = ptr
        self._layout = layout

    @always_inline
    def layout(self) -> Self.LayoutType:
        return self._layout
)mojo";

// Inline definitions for the trait-lane FUSED input-load argument (Phase 5),
// emitted once per module (like `kDenseTensorArgDefs`). Modeled directly on
// `tensor-trait-mockups/fused_tensor_args.mojo` -- NO `ManagedTensorSlice` and
// none of the legacy fusion machinery (`InputFusion` / `FusionStruct_N` /
// `_bind_to_fused_input` / `simd_load_from_tensor_pointer`):
//   * `InputLoadFusion` is the trait the GC-emitted fusion FUNCTOR conforms to.
//     The functor (emitted per fusion by `emitInputLoadFusionFunctor`) captures
//     its producer's source tensor(s) as a `TileTensor` view and replays the
//     producer body via dense `tile.load(idx)` -- exactly the mockup
//     `AddTwoInputLoadFusion.__call__` shape.
//   * `_MoggInputLoadFusionTensor` is the arg the kernel binds to: it holds the
//     functor + a layout and presents the landed `Input & Fused & LoadAccess`
//     surface. Its callable `Loadable` view delegates `load(idx)` straight to
//     the functor's `__call__` only when `_access_enabled` is true. It is
//     passed directly to `execute` and the kernel infers its generic `T`.
// ONE struct serves both CPU and GPU (mirroring `_MoggDenseTensorArg` and the
// mockup `EmitMojoInputLoadFusionTensor`). `_device_passable` controls whether
// the host proxy also crosses to the device; `_access_enabled` separately
// controls whether the callable `Loadable`/`Storable` surface is available on
// this view of the wrapper. CPU wrappers access directly. GPU host proxies only
// carry graph-visible access markers and project through `device_type` to the
// access-enabled device view. `_to_device_type` still projects the functor
// field-by-field for GPU launches. The functor's captured pointers (bare
// `Pointer`s) translate to device addresses through their own
// `_to_device_type`, exactly as `_MoggDenseTensorArg` forwards its handle. (A
// `TileTensor` field would NOT translate -- the encoder bit-copies it -- which
// is why the functor captures bare pointers and rebuilds the tile in
// `__call__`.)
//
// NOTE: `kInputLoadFusionDefs` and its siblings `kOutputStoreFusionDefs` /
// `kTransformFusionDefs` below are near-identical raw Mojo-source string
// literals (same field layout, `_to_device_type`, `shape()`), differing
// mainly in trait name, the one access method, and the transform variant's
// extra tile field. Being string literals they are hard to single-source
// today; if a fourth fused form appears, generate these from a small builder
// rather than adding a fourth verbatim block.
static constexpr llvm::StringLiteral kInputLoadFusionDefs = R"mojo(
from layout import Coord, TensorLayout, TileTensor
from extensibility import Fused, Input, LoadAccess, Loadable, Tensor
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


trait InputLoadFusion(TrivialRegisterPassable):
    def __call__[
        dtype: DType,
        rank: Int,
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord) -> SIMD[dtype, width]:
        ...


# `LoadAccess` and `Loadable` are UNCONDITIONAL so a closure that captures the
# proxy can type-check a `.load()` call. The access gate is a `comptime assert
# Self._access_enabled` in the `load` body: `_access_enabled` is False on the
# device-passable HOST proxy and True on the projected `device_type` (device_type
# flips it), so a direct host-side load is a compile error while a load on the
# projected device value -- what the closure captures after projection -- is
# allowed. Same context gate as the store proxy.
struct _MoggInputLoadFusionTensor[
    _dtype: DType,
    _LayoutType: TensorLayout,
    _alignment: Int,
    _InFusion: InputLoadFusion,
    _device_passable: Bool,
    _access_enabled: Bool,
](
    Copyable,
    DevicePassable where _device_passable,
    Fused,
    Input,
    LoadAccess,
    Loadable,
    Movable,
    Tensor,
    TrivialRegisterPassable,
):
    comptime dtype = Self._dtype
    comptime LayoutType = Self._LayoutType
    comptime rank = Self.LayoutType.rank
    comptime alignment = Self._alignment
    comptime device_passable = Self._device_passable
    comptime access_enabled = Self._access_enabled
    comptime ShapeType = Coord[*Self.LayoutType._shape_types]
    var in_fusion: Self._InFusion
    var _layout: Self.LayoutType

    comptime device_type: AnyType = _MoggInputLoadFusionTensor[
        Self._dtype,
        Self._LayoutType,
        Self._alignment,
        Self._InFusion,
        False,
        True,
    ]

    @always_inline
    def __init__(out self, in_fusion: Self._InFusion, layout: Self.LayoutType):
        self.in_fusion = in_fusion
        self._layout = layout

    @always_inline
    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ) where Self._device_passable:
        comptime in_fusion_offset = reflect[Self.device_type].field_offset[
            index=0, target=encoder.target()
        ]()
        encoder.encode_fields[Self._InFusion](
            self.in_fusion,
            target.unsafe_bitcast[UInt8]()
            .unsafe_offset(in_fusion_offset)
            .unsafe_bitcast[NoneType](),
        )
        comptime layout_offset = reflect[Self.device_type].field_offset[
            index=1, target=encoder.target()
        ]()
        encoder.encode(
            self._layout,
            target.unsafe_bitcast[UInt8]()
            .unsafe_offset(layout_offset)
            .unsafe_bitcast[NoneType](),
        )

    @staticmethod
    def get_type_name() -> String where Self._device_passable:
        return String("_MoggInputLoadFusionTensor")

    @always_inline
    def shape(self) -> Self.ShapeType:
        return rebind[Self.ShapeType](self._layout.shape_coord())

    @always_inline
    def load[
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord) -> SIMD[Self.dtype, width]:
        # Context gate (see the store proxy): `Loadable`/`LoadAccess` are
        # unconditional so a closure can type-check `.load()`; `_access_enabled`
        # is False on the device-passable host proxy and True on the projected
        # `device_type`, so a host-side load is a compile error while a load on
        # the projected device value is allowed.
        comptime assert Self._access_enabled, (
            "fused load is only valid on an access-enabled tensor (the projected"
            " device value); the device-passable host proxy cannot be loaded"
            " from directly"
        )
        return self.in_fusion.__call__[
            Self.dtype, Self.rank, width, element_alignment
        ](idx)
)mojo";

// Inline definitions for the trait-lane FUSED output-store argument (Phase 5),
// the epilogue counterpart of `kInputLoadFusionDefs`. Modeled on the mockup
// `OutputStoreFusion` / `EmitMojoOutputStoreFusionTensor` -- NO
// `ManagedTensorSlice`. `OutputStoreFusion.__call__(idx, value)` runs the fused
// consumer epilogue and writes; `_MoggOutputStoreFusionTensor` presents the
// landed `Output & Fused & StoreAccess` surface. The callable `Storable`
// conformance is gated by `_access_enabled` for the same reason as fused load:
// GPU host proxies can be enqueued and projected, but cannot directly
// dereference; the projected `device_type` enables the store surface.
static constexpr llvm::StringLiteral kOutputStoreFusionDefs = R"mojo(
from layout import Coord, TensorLayout, TileTensor
from extensibility import Fused, Output, StoreAccess, Storable, Tensor
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


trait OutputStoreFusion(TrivialRegisterPassable):
    def __call__[
        dtype: DType,
        rank: Int,
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord, value: SIMD[dtype, width]):
        ...


# `StoreAccess` and `Storable` are UNCONDITIONAL, so a closure that captures the
# proxy can type-check a `.store()` call. The real access gate is a `comptime
# assert Self._access_enabled` in the `store` body: `_access_enabled` is False on
# the device-passable HOST proxy and True on the projected `device_type` (the
# device_type flips it). So a direct host-side store on the host proxy is a
# compile error, while a store on the projected device value -- which is what the
# closure body captures/runs on after projection -- is allowed. This is the
# context gate that denies host-side dereference of a device-passable proxy
# without any target-triple inspection.
struct _MoggOutputStoreFusionTensor[
    _dtype: DType,
    _LayoutType: TensorLayout,
    _alignment: Int,
    _OutFusion: OutputStoreFusion,
    _device_passable: Bool,
    _access_enabled: Bool,
](
    Copyable,
    DevicePassable where _device_passable,
    Fused,
    Movable,
    Output,
    StoreAccess,
    Storable,
    Tensor,
    TrivialRegisterPassable,
):
    comptime dtype = Self._dtype
    comptime LayoutType = Self._LayoutType
    comptime rank = Self.LayoutType.rank
    comptime alignment = Self._alignment
    comptime device_passable = Self._device_passable
    comptime access_enabled = Self._access_enabled
    comptime ShapeType = Coord[*Self.LayoutType._shape_types]
    var out_fusion: Self._OutFusion
    var _layout: Self.LayoutType

    comptime device_type: AnyType = _MoggOutputStoreFusionTensor[
        Self._dtype,
        Self._LayoutType,
        Self._alignment,
        Self._OutFusion,
        False,
        True,
    ]

    @always_inline
    def __init__(out self, out_fusion: Self._OutFusion, layout: Self.LayoutType):
        self.out_fusion = out_fusion
        self._layout = layout

    @always_inline
    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ) where Self._device_passable:
        comptime out_fusion_offset = reflect[Self.device_type].field_offset[
            index=0, target=encoder.target()
        ]()
        encoder.encode_fields[Self._OutFusion](
            self.out_fusion,
            target.unsafe_bitcast[UInt8]()
            .unsafe_offset(out_fusion_offset)
            .unsafe_bitcast[NoneType](),
        )
        comptime layout_offset = reflect[Self.device_type].field_offset[
            index=1, target=encoder.target()
        ]()
        encoder.encode(
            self._layout,
            target.unsafe_bitcast[UInt8]()
            .unsafe_offset(layout_offset)
            .unsafe_bitcast[NoneType](),
        )

    @staticmethod
    def get_type_name() -> String where Self._device_passable:
        return String("_MoggOutputStoreFusionTensor")

    @always_inline
    def shape(self) -> Self.ShapeType:
        return rebind[Self.ShapeType](self._layout.shape_coord())

    @always_inline
    def store[
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord, value: SIMD[Self.dtype, width]):
        # `Storable`/`StoreAccess` are unconditional so a host-created closure can
        # type-check `.store()`; the real gate is here in the body. `_access_enabled`
        # is False on the device-passable HOST proxy and True on the projected
        # `device_type`, so a direct host-side store is a compile error while a
        # store on the projected device value (the closure body runs there) is
        # allowed.
        comptime assert Self._access_enabled, (
            "fused store is only valid on an access-enabled tensor (the"
            " projected device value); the device-passable host proxy cannot be"
            " stored to directly"
        )
        self.out_fusion.__call__[
            Self.dtype, Self.rank, width, element_alignment
        ](idx, value)
)mojo";

// Inline definitions for the trait-lane FUSED output-TRANSFORM (compute-output)
// argument (Phase 5). Element compute-output is ONE output edge that the kernel
// both transforms and writes, so `_MoggTransformFusionTensor` composes
// `Transformable` (apply the fused transform, returns the value) and
// `TileTensorable` (project the real output buffer so the parent stores the
// result) on a single operand -- NOT two operands. The transform functor is
// pure value->value (it does NOT write); the parent kernel performs the final
// store via `to_tile_tensor(z).store(...)`. The graph-visible marker
// `TransformAccess` is UNCONDITIONAL (so the wrapper satisfies an
// `Output & Fused & TransformAccess` bound on both host and device); the
// callable `Transformable` / `TileTensorable` surfaces are gated by
// `_access_enabled` and the device kernel refines through `downcast[...,
// Transformable & TileTensorable]`. The output buffer is held as a bare
// `Pointer` (NOT a `TileTensor`) so it translates to a device address;
// `to_tile_tensor` rebuilds the tile. Modeled on the mockup
// `EmitMojoTransformFusionTensor`.
static constexpr llvm::StringLiteral kTransformFusionDefs = R"mojo(
from layout import Coord, TensorLayout, TileTensor
from extensibility import DenseTensor, Fused, Output, TileTensorable, TransformAccess, Transformable
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


trait TransformFusion(TrivialRegisterPassable):
    def __call__[
        dtype: DType,
        rank: Int,
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord, value: SIMD[dtype, width]) -> SIMD[dtype, width]:
        ...


# `TransformAccess`, `Transformable`, and `TileTensorable` are UNCONDITIONAL so a
# closure that captures the proxy can type-check `.transform()` / `.to_tile_tensor()`.
# The access gate is a `comptime assert Self._access_enabled` in each callable
# body: `_access_enabled` is False on the device-passable HOST proxy and True on
# the projected `device_type` (device_type flips it), so a direct host-side
# transform/store is a compile error while access on the projected device value --
# what the closure captures after projection -- is allowed. Same context gate as
# the store proxy.
struct _MoggTransformFusionTensor[
    _dtype: DType,
    _LayoutType: TensorLayout,
    _alignment: Int,
    _mut: Bool,
    _TransformFusion: TransformFusion,
    _device_passable: Bool,
    _access_enabled: Bool,
](
    Copyable,
    DenseTensor,
    DevicePassable where _device_passable,
    Fused,
    Movable,
    Output,
    TileTensorable,
    TransformAccess,
    Transformable,
    TrivialRegisterPassable,
):
    comptime dtype = Self._dtype
    comptime LayoutType = Self._LayoutType
    comptime rank = Self.LayoutType.rank
    comptime alignment = Self._alignment
    comptime mut = Self._mut
    comptime device_passable = Self._device_passable
    comptime access_enabled = Self._access_enabled
    comptime ShapeType = Coord[*Self.LayoutType._shape_types]
    var transform_fusion: Self._TransformFusion
    var _layout: Self.LayoutType
    var handle: Pointer[Scalar[Self.dtype], UntrackedOrigin[mut = Self.mut]]

    comptime device_type: AnyType = _MoggTransformFusionTensor[
        Self._dtype,
        Self._LayoutType,
        Self._alignment,
        Self._mut,
        Self._TransformFusion,
        False,
        True,
    ]

    @always_inline
    def __init__(
        out self,
        transform_fusion: Self._TransformFusion,
        layout: Self.LayoutType,
        ptr: Pointer[Scalar[Self.dtype], UntrackedOrigin[mut = Self.mut]],
    ):
        self.transform_fusion = transform_fusion
        self._layout = layout
        self.handle = ptr

    @always_inline
    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ) where Self._device_passable:
        comptime fusion_offset = reflect[Self.device_type].field_offset[
            index=0, target=encoder.target()
        ]()
        encoder.encode_fields[Self._TransformFusion](
            self.transform_fusion,
            target.unsafe_bitcast[UInt8]()
            .unsafe_offset(fusion_offset)
            .unsafe_bitcast[NoneType](),
        )
        comptime layout_offset = reflect[Self.device_type].field_offset[
            index=1, target=encoder.target()
        ]()
        encoder.encode(
            self._layout,
            target.unsafe_bitcast[UInt8]()
            .unsafe_offset(layout_offset)
            .unsafe_bitcast[NoneType](),
        )
        comptime handle_offset = reflect[Self.device_type].field_offset[
            index=2, target=encoder.target()
        ]()
        self.handle._to_device_type(
            encoder,
            target.unsafe_bitcast[UInt8]()
            .unsafe_offset(handle_offset)
            .unsafe_bitcast[NoneType](),
        )

    @staticmethod
    def get_type_name() -> String where Self._device_passable:
        return String("_MoggTransformFusionTensor")

    @always_inline
    def shape(self) -> Self.ShapeType:
        return rebind[Self.ShapeType](self._layout.shape_coord())

    @always_inline
    def layout(self) -> Self.LayoutType:
        return self._layout

    # `TileTensorable`: the parent kernel stores the transformed value through
    # this projection (rebuilt from the bare output-buffer pointer).
    @always_inline
    def to_tile_tensor(
        self,
    ) -> TileTensor[
        dtype = Self.dtype,
        LayoutType = Self.LayoutType,
        origin = UntrackedOrigin[mut = Self.mut],
    ]:
        comptime assert Self._access_enabled, (
            "fused transform output projection is only valid on an access-enabled"
            " tensor (the projected device value); the device-passable host proxy"
            " cannot be projected to a tile directly"
        )
        return TileTensor[
            dtype = Self.dtype,
            LayoutType = Self.LayoutType,
            origin = UntrackedOrigin[mut = Self.mut],
        ](ptr=self.handle, layout=self._layout)

    @always_inline
    def transform[
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord, value: SIMD[Self.dtype, width]) -> SIMD[
        Self.dtype, width
    ]:
        comptime assert Self._access_enabled, (
            "fused transform is only valid on an access-enabled tensor (the"
            " projected device value); the device-passable host proxy cannot be"
            " transformed directly"
        )
        return self.transform_fusion.__call__[
            Self.dtype, Self.rank, width, element_alignment
        ](idx, value)
)mojo";

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MOJOTRAITS_H
