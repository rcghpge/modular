# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from std.builtin.device_passable import DevicePassable
from std.builtin.rebind import downcast
from std.collections import Optional
from std.collections.string import StaticString
from std.math import ceildiv, iota
from std.ffi import external_call
from std.gpu import global_idx
from max.gpu.host.info import is_cpu
from std.sys.info import simd_width_of
from max.algorithm.functional import elementwise
import extensibility
from layout import IntTuple, TensorLayout
from layout import TileTensor, CoordLike, Coord
from layout.tile_layout import Layout as TileLayout
from extensibility import StaticTensorSpec
from max.gpu.host import DeviceContext, DeviceContextArray
from max.gpu.host import DeviceGraphBuilder
from extensibility import (
    InputTensor,
    InputVariadicTensors,
    ManagedTensorSlice,
    OutputTensor,
    OutputVariadicTensors,
    _FusedComputeOutputTensor,
    foreach,
    simd_load_from_managed_tensor_slice,
    simd_store_into_managed_tensor_slice,
)
from builtin_primitives.primitives import (
    foreach_out_func,
    view_copy_impl,
)
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from extensibility import (
    _FusedInputVariadicTensors as FusedInputVariadicTensors,
)
from extensibility import (
    _FusedOutputTensor as FusedOutputTensor,
)
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from extensibility import (
    DenseTensor,
    Fused,
    Input,
    LoadAccess,
    Loadable,
    MutableInput,
    Output,
    StoreAccess,
    Storable,
    Tensor,
    TileTensorable,
    TransformAccess,
    Transformable,
    to_tile_tensor,
)

from std.utils import IndexList, StaticTuple
from std.utils.coord import Coord, coord_to_index_list


# TODO(MOCO-1413): remove this need to keep imported exported funcs alive.
@export
def export() abi("Mojo"):
    comptime _simd_load_from_managed_tensor_slice = simd_load_from_managed_tensor_slice
    comptime _simd_store_into_managed_tensor_slice = simd_store_into_managed_tensor_slice


# ===-----------------------------------------------------------------------===#
# Opaque Reg Types
# ===-----------------------------------------------------------------------===#


struct MyCustomScalarRegTrivialSI32(TrivialRegisterPassable):
    var val: Int32

    @implicit
    def __init__(out self, val: Int32):
        print("MyCustomScalarRegTrivialSI32.__init__", val)
        self.val = val


@extensibility.register("tensor_to_custom_scalar_si32_reg_trivial")
struct OpaqueToCustomScalarSI32RegTrivial:
    @staticmethod
    def execute(
        x: InputTensor[dtype=DType.int32, rank=1, ...]
    ) -> MyCustomScalarRegTrivialSI32:
        return MyCustomScalarRegTrivialSI32(x[0])


struct MyCustomScalarRegSI32(ImplicitlyCopyable, RegisterPassable):
    var val: Int32

    @implicit
    def __init__(out self, val: Int32):
        print("MyCustomScalarRegSI32.__init__", val)
        self.val = val

    def __deinit__(deinit self):
        print("MyCustomScalarRegSI32.__deinit__", self.val)


# It is intentional there are no methods which consume this.
# It is here to support some level of type checking.
struct MyCustomScalarRegF32(ImplicitlyCopyable, RegisterPassable):
    var val: Float32

    @implicit
    def __init__(out self, val: Float32):
        print("MyCustomScalarRegF32.__init__", val)
        self.val = val

    def __deinit__(deinit self):
        print("MyCustomScalarRegF32.__deinit__", self.val)


@extensibility.register("tensor_to_custom_scalar_si32_reg")
struct OpaqueToCustomScalarSI32Reg:
    @staticmethod
    def execute(
        x: InputTensor[dtype=DType.int32, rank=1, ...]
    ) -> MyCustomScalarRegSI32:
        return MyCustomScalarRegSI32(x[0])


# Adds two custom scalar types (one of which is register passable) and writes
# to a tensor
@extensibility.register("opaque_add_to_tensor_si32_reg")
struct OpaqueAddToTensorSI32Reg:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype=DType.int32, rank=1, ...],
        x: MyCustomScalarRegSI32,
        y: MyCustomScalarRegSI32,
    ):
        outp[0] = x.val + y.val


@extensibility.register("opaque_add_to_tensor_f32_reg")
struct OpaqueAddToTensorF32:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype=DType.float32, rank=1, ...],
        x: MyCustomScalarRegF32,
        y: MyCustomScalarRegF32,
    ):
        outp[0] = x.val + y.val


# ===-----------------------------------------------------------------------===#
# Opaque Mem. Types
# ===-----------------------------------------------------------------------===#


struct MyCustomScalarSI32(ImplicitlyCopyable):
    var val: Int32

    @implicit
    def __init__(out self, val: Int32):
        print("MyCustomScalarSI32.__init__", val)
        self.val = val

    def __deinit__(deinit self):
        print("MyCustomScalarSI32.__deinit__", self.val)


@extensibility.register("tensor_to_custom_scalar_si32")
struct OpaqueToCustomScalarSI32:
    @staticmethod
    def execute(
        x: InputTensor[dtype=DType.int32, rank=1, ...]
    ) -> MyCustomScalarSI32:
        return MyCustomScalarSI32(x[0])


# Adds two custom scalar types (one of which is register passable) and writes
# to a tensor
@extensibility.register("opaque_add_to_tensor_si32")
struct OpaqueAddToTensorSI32:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype=DType.int32, rank=1, ...],
        x: MyCustomScalarSI32,
        y: MyCustomScalarSI32,
    ):
        outp[0] = x.val + y.val


@extensibility.register("opaque_add_to_tensor_si32_raises")
struct OpaqueAddToTensorSI32Raises:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype=DType.int32, rank=1, ...],
        x: MyCustomScalarSI32,
        y: MyCustomScalarSI32,
    ) raises:
        outp[0] = x.val + y.val


# ===-----------------------------------------------------------------------===#
# Other Kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("imposter_add")
struct Foo:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: OutputTensor,
        x: InputTensor,
        y: InputTensor,
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](x.load[width](idx)) + rebind[
                SIMD[z.dtype, width]
            ](y.load[width](idx))

        foreach[func, target=target](z, ctx)


@extensibility.register_shape_function("imposter_add")
def imposter_add_shape(x: InputTensor, y: InputTensor) -> IndexList[x.rank]:
    return x.shape()


# Device entry for the GPU negate path: one thread per flat element, negating
# `input` into `output`. The trait arg's device representation is a `TileTensor`
# (see `_MoggDenseTensorArg.device_type`), so the kernel gets Coord/raw-indexable
# tiles rather than bare pointers.
def _gpu_negate_kernel[
    dtype: DType, LayoutType: TensorLayout
](
    output: TileTensor[dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[dtype, LayoutType, ImmutAnyOrigin],
    num_elements_dev: Int32,
):
    var num_elements = Int(num_elements_dev)
    var linear_idx = global_idx.x
    if linear_idx >= num_elements:
        return
    output.raw_store(
        Int(linear_idx),
        rebind[SIMD[dtype, 1]](-input.raw_load(Int(linear_idx))),
    )


# Trait-lane dense unary kernel (Slice 3 / Phase 4 e2e proof). Uses the
# composable tensor-argument traits instead of `ManagedTensorSlice`: the graph
# compiler emits an inline `_MoggDenseTensorArg` for each arg and infers the
# generic `T` from it. Computes `z[i] = -x[i]` over the flat element range.
# A single struct serves both targets: the CPU path runs the host
# `TileTensorable` projection; the GPU path crosses the same value as a
# `DevicePassable` and projects to a device `TileTensor`.
@extensibility.register("trait_negate")
struct TraitNegate:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & DenseTensor],
        x: Some[Input & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        # The host-entry bound is the role + metadata traits; each target arm
        # asserts the capability it needs (mirrors `DenseElementwiseKernel`).
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(z), TileTensorable)
            comptime assert conforms_to(type_of(x), TileTensorable)
            # `to_tile_tensor(z)` picks the Output (mutable) projection; `x` gets
            # the default immutable one. `x` and `z` are independent generic
            # types, so rebind the loaded element to the output dtype.
            var zt = to_tile_tensor(z)
            var xt = to_tile_tensor(x)
            var n = Int(x.shape().product())
            for i in range(n):
                zt.raw_store(
                    i, rebind[SIMD[type_of(zt).dtype, 1]](-xt.raw_load(i))
                )
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime block_size = 64
            var n = Int(x.shape().product())
            ctx.enqueue_function[
                _gpu_negate_kernel[type_of(z).dtype, type_of(z).LayoutType]
            ](
                z,
                x,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


# UNIFIED-closure fused-load variant: the device kernel binds the PROJECTED load
# proxy (`downcast[XHost.device_type, Loadable]`, access-enabled) and reads it
# through a unified closure that captures that projected value. Same context gate
# as store: the closure captures `device_type` (created after projection), so the
# load body's `comptime assert _access_enabled` passes; a host-side load is a
# compile error (see `trait_fused_load_host_negative`).
def _gpu_fused_load_negate_unified_closure_kernel[
    XHost: DevicePassable,
    dtype: DType,
    LayoutType: TensorLayout,
](
    output: TileTensor[dtype, LayoutType, MutAnyOrigin],
    input: downcast[XHost.device_type, Loadable],
    n_dev: Int32,
):
    var n = Int(n_dev)
    var i = Int(global_idx.x)
    if i >= n:
        return

    @always_inline
    def load_fn(idx: Int) {imm input} -> SIMD[dtype, 1]:
        return rebind[SIMD[dtype, 1]](input.load[1](Coord(idx)))

    output.raw_store(i, -load_fn(i))


@extensibility.register("trait_fused_load_negate_unified_closure")
struct TraitFusedLoadNegateUnifiedClosure:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & DenseTensor],
        x: Some[Input & Fused & LoadAccess],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(z), TileTensorable)
            comptime assert conforms_to(type_of(x), Loadable)
            var zt = to_tile_tensor(z)
            var n = Int(z.shape().product())
            for i in range(n):
                var v = x.load[1](Coord(i))
                zt.raw_store(i, rebind[SIMD[type_of(zt).dtype, 1]](-v))
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime block_size = 64
            var n = Int(z.shape().product())
            ctx.enqueue_function[
                _gpu_fused_load_negate_unified_closure_kernel[
                    type_of(x), type_of(z).dtype, type_of(z).LayoutType
                ]
            ](
                z,
                x,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


# NEGATIVE control: loads DIRECTLY from the fused-load proxy at the host entry.
# On a GPU device the emitted proxy is the device-passable HOST proxy
# (`_access_enabled=False`), so the load body's `comptime assert _access_enabled`
# fails at instantiation -- must NOT compile. Exercised (expecting failure) by
# `mogg-kernel-api-e2e-trait-fused-load-host-negative.mlir`.
@extensibility.register("trait_fused_load_host_negative")
struct TraitFusedLoadHostNegative:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & DenseTensor],
        x: Some[Input & Fused & LoadAccess],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(x), DevicePassable)
        comptime assert conforms_to(type_of(x), Loadable)
        # Direct host-side load on the device-passable proxy: the context gate
        # (`comptime assert _access_enabled` in the load body) rejects this.
        _ = x.load[1](Coord(0))


# Trait-lane in-place mutable kernel: `x` is read and written through the same
# buffer (`x[i] += 1`). Proves the emitted `_MoggDenseTensorArg[io=MUTABLE_INPUT]`
# satisfies a `MutableInput & DenseTensor & TileTensorable` stub bound -- i.e.
# the wrapper conforms to `MutableInput` (gated on the io), not just `Input`.
@extensibility.register("trait_inplace_increment")
struct TraitInplaceIncrement:
    @staticmethod
    def execute[
        target: StaticString,
    ](x: Some[MutableInput & DenseTensor], ctx: DeviceContext) raises:
        comptime assert conforms_to(type_of(x), MutableInput)
        comptime assert conforms_to(type_of(x), TileTensorable)
        var xt = to_tile_tensor(x)
        var n = Int(x.shape().product())
        for i in range(n):
            xt.raw_store(
                i, rebind[SIMD[type_of(xt).dtype, 1]](xt.raw_load(i) + 1)
            )


# ===-----------------------------------------------------------------------===#
# Trait-lane FUSED kernels (Phase 5): one per fusion access form.
#
# A fused operand carries the `Fused` marker plus one graph-visible access form:
#   * `Input & Fused & LoadAccess` -- prologue (input-load) fusion.
#     The callable `Loadable` view is used only where `x.load(idx)` is valid.
#   * `Output & Fused & StoreAccess` -- epilogue (output-store) fusion.
#     The callable `Storable` view is used only where `z.store(idx, v)` is valid.
#   * `Output & Fused & Transformable` -- epilogue (transform / compute-output)
#     fusion: `z.transform(idx, v)` runs the consumer and returns the value;
#     the GC output mechanism performs the final dense store.
# For each, the GC emits an inline `_MoggInput*/Output*FusionArg` wrapper around
# the existing fusion functor (`InputFusion`/`OutputFusion`/`ComputeOutputFusion`)
# conforming to the landed access trait; the kernel infers its generic `T`.
# Element access is `Coord`-based (the trait interface); the wrapper bridges to
# the functor's `IndexList`-based method.
# ===-----------------------------------------------------------------------===#


# Input-load (prologue) fusion: z[i] = -x.load(i), rank-1. Mirrors
# `trait_negate` but with a fused input instead of a dense one.
# Device entry for the fused-load negate path: one thread per element negates the
# producer's fused load into the dense output. `input` is the device-projected
# fused-load wrapper, bound by ONLY its `Loadable` capability via
# `downcast[XHost.device_type, Loadable]` (the `XHost: DevicePassable` generic is
# just the carrier that names the device type).
def _gpu_fused_load_negate_kernel[
    XHost: DevicePassable,
    dtype: DType,
    LayoutType: TensorLayout,
](
    output: TileTensor[dtype, LayoutType, MutAnyOrigin],
    input: downcast[XHost.device_type, Loadable],
    n_dev: Int32,
):
    var n = Int(n_dev)
    var i = Int(global_idx.x)
    if i >= n:
        return
    output.raw_store(i, rebind[SIMD[dtype, 1]](-input.load[1](Coord(i))))


@extensibility.register("trait_fused_load_negate")
struct TraitFusedLoadNegate:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & DenseTensor],
        x: Some[Input & Fused & LoadAccess],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(z), TileTensorable)
            comptime assert conforms_to(type_of(x), Loadable)
            var zt = to_tile_tensor(z)
            var n = Int(z.shape().product())
            for i in range(n):
                var v = x.load[1](Coord(i))
                zt.raw_store(i, rebind[SIMD[type_of(zt).dtype, 1]](-v))
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime block_size = 64
            var n = Int(z.shape().product())
            ctx.enqueue_function[
                _gpu_fused_load_negate_kernel[
                    type_of(x), type_of(z).dtype, type_of(z).LayoutType
                ]
            ](
                z,
                x,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


# Output-store (epilogue) fusion: computes `2 * x[i]` and hands it to the fused
# store, which runs the consumer epilogue and writes. `x` is a plain dense input.
# Device entry for the fused-store double path: one thread per element computes
# `2 * x` and hands it to the device-projected fused-store wrapper, bound by ONLY
# its `Storable` capability via `downcast[ZHost.device_type, Storable]`.
def _gpu_fused_store_double_kernel[
    ZHost: DevicePassable,
    dtype: DType,
    LayoutType: TensorLayout,
](
    output: downcast[ZHost.device_type, Storable],
    input: TileTensor[dtype, LayoutType, ImmutAnyOrigin],
    n_dev: Int32,
):
    var n = Int(n_dev)
    var i = Int(global_idx.x)
    if i >= n:
        return
    var v = rebind[SIMD[dtype, 1]](input.raw_load(i)) * 2
    output.store[1](Coord(i), rebind[SIMD[type_of(output).dtype, 1]](v))


@extensibility.register("trait_fused_store_double")
struct TraitFusedStoreDouble:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & StoreAccess],
        x: Some[Input & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(z), Storable)
            comptime assert conforms_to(type_of(x), TileTensorable)
            var xt = to_tile_tensor(x)
            var n = Int(z.shape().product())
            for i in range(n):
                var v = rebind[SIMD[type_of(z).dtype, 1]](xt.raw_load(i)) * 2
                z.store[1](Coord(i), v)
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime block_size = 64
            var n = Int(z.shape().product())
            ctx.enqueue_function[
                _gpu_fused_store_double_kernel[
                    type_of(z), type_of(z).dtype, type_of(x).LayoutType
                ]
            ](
                z,
                x,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


def _gpu_fused_store_parameter_closure_kernel[
    dtype: DType,
    ZDev: Storable,
    LayoutType: TensorLayout,
    store_adapter: def[OutT: Storable, width: Int](
        OutT, Coord, SIMD[OutT.dtype, width]
    ) capturing -> None,
](
    output: ZDev,
    input: TileTensor[dtype, LayoutType, ImmutAnyOrigin],
    n_dev: Int32,
):
    var n = Int(n_dev)
    var i = Int(global_idx.x)
    if i >= n:
        return
    store_adapter[ZDev, 1](
        output,
        Coord(i),
        rebind[SIMD[ZDev.dtype, 1]](input.raw_load(i)),
    )


@always_inline
def _launch_fused_store_parameter_closure[
    dtype: DType,
    ZHost: DevicePassable,
    XHost: DevicePassable,
    LayoutType: TensorLayout,
    store_adapter: def[OutT: Storable, width: Int](
        OutT, Coord, SIMD[OutT.dtype, width]
    ) capturing -> None,
](output: ZHost, input: XHost, n: Int, ctx: DeviceContext,) raises:
    comptime block_size = 64
    ctx.enqueue_function[
        _gpu_fused_store_parameter_closure_kernel[
            dtype,
            downcast[ZHost.device_type, Storable],
            LayoutType,
            store_adapter,
        ]
    ](
        output,
        input,
        Int32(n),
        grid_dim=(ceildiv(n, block_size),),
        block_dim=(block_size,),
    )


@extensibility.register("trait_fused_store_parameter_closure")
struct TraitFusedStoreParameterClosure:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & StoreAccess],
        x: Some[Input & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(z), Storable)
            comptime assert conforms_to(type_of(x), TileTensorable)
            var xt = to_tile_tensor(x)
            var n = Int(z.shape().product())
            for i in range(n):
                z.store[1](
                    Coord(i),
                    rebind[SIMD[type_of(z).dtype, 1]](xt.raw_load(i)),
                )
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime dtype = type_of(x).dtype

            @__parameter
            @always_inline
            def store_adapter[
                OutT: Storable, width: Int
            ](output: OutT, coord: Coord, value: SIMD[OutT.dtype, width]):
                output.store[width](coord, value)

            _launch_fused_store_parameter_closure[
                dtype,
                type_of(z),
                type_of(x),
                type_of(x).LayoutType,
                store_adapter,
            ](z, x, Int(z.shape().product()), ctx)


# Device entry for the UNIFIED-closure fused-store path. The device kernel binds
# the PROJECTED store proxy by its `Storable` capability via
# `downcast[ZHost.device_type, Storable]` -- the projected value has
# `_access_enabled=True`. It then creates a unified closure that captures that
# projected value and performs the store. This is the supported closure shape: the
# closure is created AFTER projection, so it captures `device_type` and the store
# body's `comptime assert _access_enabled` gate passes. A host-created closure
# over the device-passable HOST proxy (or a direct host-side store) hits
# `_access_enabled=False` and fails to compile -- see the negative test kernel.
def _gpu_fused_store_unified_closure_kernel[
    ZHost: DevicePassable,
    dtype: DType,
    LayoutType: TensorLayout,
](
    output: downcast[ZHost.device_type, Storable],
    input: TileTensor[dtype, LayoutType, ImmutAnyOrigin],
    n_dev: Int32,
):
    var n = Int(n_dev)
    var i = Int(global_idx.x)
    if i >= n:
        return
    var v = rebind[SIMD[dtype, 1]](input.raw_load(i)) * 2

    # Unified closure capturing the projected (access-enabled) store proxy.
    @always_inline
    def store_fn(idx: Int, val: SIMD[dtype, 1]) {imm output}:
        output.store[1](Coord(idx), rebind[SIMD[type_of(output).dtype, 1]](val))

    store_fn(i, v)


@extensibility.register("trait_fused_store_unified_closure")
struct TraitFusedStoreUnifiedClosure:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & StoreAccess],
        x: Some[Input & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(z), Storable)
            comptime assert conforms_to(type_of(x), TileTensorable)
            var xt = to_tile_tensor(x)
            var n = Int(z.shape().product())
            for i in range(n):
                z.store[1](
                    Coord(i),
                    rebind[SIMD[type_of(z).dtype, 1]](xt.raw_load(i)) * 2,
                )
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime block_size = 64
            var n = Int(z.shape().product())
            ctx.enqueue_function[
                _gpu_fused_store_unified_closure_kernel[
                    type_of(z), type_of(z).dtype, type_of(x).LayoutType
                ]
            ](
                z,
                x,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


# NEGATIVE control: stores DIRECTLY on the fused-store proxy from the host entry,
# with no projection. On a GPU device the emitted proxy is the device-passable
# HOST proxy (`_access_enabled=False`), so the store body's
# `comptime assert Self._access_enabled` fails at instantiation -- this kernel must
# NOT compile through the gc-pipeline. Exercised (expecting failure) by
# `mogg-kernel-api-e2e-trait-fused-store-host-negative.mlir`.
@extensibility.register("trait_fused_store_host_negative")
struct TraitFusedStoreHostNegative:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & StoreAccess],
        x: Some[Input & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(z), DevicePassable)
        comptime assert conforms_to(type_of(z), Storable)
        # Direct host-side store on the device-passable proxy: `store` resolves
        # (Storable is unconditional) but the store body's context gate
        # (`comptime assert _access_enabled`) rejects it at instantiation, because
        # the emitted host proxy has `access_enabled = False`.
        z.store[1](Coord(0), SIMD[type_of(z).dtype, 1](0))


# Output-transform (compute-output) fusion: the kernel computes `x[i] + 1`,
# applies the fused transform via `z.transform`, then performs the final dense
# store itself (per the design, a transform operand transforms a computed value
# and leaves the parent responsible for the store). `Transformable` is the sole
# access form; `TileTensorable` adds only the dense projection used for that
# final store. `x` is a plain dense input.
# Device entry for the fused-transform path: one thread per element computes
# `x[i] + 1`, applies the fused transform via `z.transform`, and stores through
# the output-buffer projection `z.to_tile_tensor()`. `z` is the device-projected
# transform wrapper, bound by ONLY `Transformable & TileTensorable` via
# `downcast[ZHost.device_type, Transformable & TileTensorable]`.
def _gpu_fused_transform_bias_kernel[
    ZHost: DevicePassable,
    dtype: DType,
    LayoutType: TensorLayout,
](
    z: downcast[ZHost.device_type, Transformable & TileTensorable],
    x: TileTensor[dtype, LayoutType, ImmutAnyOrigin],
    n_dev: Int32,
):
    var n = Int(n_dev)
    var i = Int(global_idx.x)
    if i >= n:
        return
    var coord = Coord(i)
    var raw = rebind[SIMD[type_of(z).dtype, 1]](
        rebind[SIMD[dtype, 1]](x.raw_load(i)) + 1
    )
    var t = z.transform[1](coord, raw)
    var zt = z.to_tile_tensor()
    # An Output edge is always writable; gives `TileTensor.store` its evidence.
    comptime assert type_of(z).mut
    zt.store(coord, rebind[SIMD[type_of(zt).dtype, 1]](t))


@extensibility.register("trait_fused_transform_bias")
struct TraitFusedTransformBias:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        # The output carries ONLY the minimal graph-visible markers `Output &
        # Fused & TransformAccess`. Everything the body uses -- `Transformable`
        # (the transform) and `TileTensorable` (the output projection for the
        # parent store, which refines `DenseTensor` so the layout metadata comes
        # for free) -- is recovered via type refinement below.
        z: Some[Output & Fused & TransformAccess],
        x: Some[Input & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(z), Transformable)
            comptime assert conforms_to(type_of(z), TileTensorable)
            comptime assert conforms_to(type_of(x), TileTensorable)
            var zt = to_tile_tensor(z)
            var xt = to_tile_tensor(x)
            var n = Int(z.shape().product())
            for i in range(n):
                var raw = rebind[SIMD[type_of(zt).dtype, 1]](xt.raw_load(i)) + 1
                var t = z.transform[1](Coord(i), raw)
                zt.raw_store(i, rebind[SIMD[type_of(zt).dtype, 1]](t))
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime block_size = 64
            var n = Int(z.shape().product())
            ctx.enqueue_function[
                _gpu_fused_transform_bias_kernel[
                    type_of(z), type_of(z).dtype, type_of(x).LayoutType
                ]
            ](
                z,
                x,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


# UNIFIED-closure fused-transform variant: the device kernel binds the PROJECTED
# transform proxy (`downcast[ZHost.device_type, Transformable & TileTensorable]`,
# access-enabled) and applies the transform + final store through a unified
# closure that captures that projected value. Same context gate as store/load: the
# closure captures `device_type`, so the `transform`/`to_tile_tensor` bodies'
# `comptime assert _access_enabled` passes; a host-side transform is a compile
# error (see `trait_fused_transform_host_negative`).
def _gpu_fused_transform_bias_unified_closure_kernel[
    ZHost: DevicePassable,
    dtype: DType,
    LayoutType: TensorLayout,
](
    z: downcast[ZHost.device_type, Transformable & TileTensorable],
    x: TileTensor[dtype, LayoutType, ImmutAnyOrigin],
    n_dev: Int32,
):
    var n = Int(n_dev)
    var i = Int(global_idx.x)
    if i >= n:
        return
    var coord = Coord(i)
    var raw = rebind[SIMD[type_of(z).dtype, 1]](
        rebind[SIMD[dtype, 1]](x.raw_load(i)) + 1
    )
    comptime assert type_of(z).mut

    @always_inline
    def transform_and_store(c: Coord, r: SIMD[type_of(z).dtype, 1]) {imm z}:
        var t = z.transform[1](c, r)
        var zt = z.to_tile_tensor()
        zt.store(c, rebind[SIMD[type_of(zt).dtype, 1]](t))

    transform_and_store(coord, raw)


@extensibility.register("trait_fused_transform_bias_unified_closure")
struct TraitFusedTransformBiasUnifiedClosure:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & TransformAccess],
        x: Some[Input & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            comptime assert conforms_to(type_of(z), Transformable)
            comptime assert conforms_to(type_of(z), TileTensorable)
            comptime assert conforms_to(type_of(x), TileTensorable)
            var zt = to_tile_tensor(z)
            var xt = to_tile_tensor(x)
            var n = Int(z.shape().product())
            for i in range(n):
                var raw = rebind[SIMD[type_of(zt).dtype, 1]](xt.raw_load(i)) + 1
                var t = z.transform[1](Coord(i), raw)
                zt.raw_store(i, rebind[SIMD[type_of(zt).dtype, 1]](t))
        else:
            comptime assert conforms_to(type_of(z), DevicePassable)
            comptime assert conforms_to(type_of(x), DevicePassable)
            comptime block_size = 64
            var n = Int(z.shape().product())
            ctx.enqueue_function[
                _gpu_fused_transform_bias_unified_closure_kernel[
                    type_of(z), type_of(z).dtype, type_of(x).LayoutType
                ]
            ](
                z,
                x,
                Int32(n),
                grid_dim=(ceildiv(n, block_size),),
                block_dim=(block_size,),
            )


# NEGATIVE control: transforms DIRECTLY on the fused-transform proxy at the host
# entry. On a GPU device the emitted proxy is the device-passable HOST proxy
# (`_access_enabled=False`), so the transform body's `comptime assert
# _access_enabled` fails at instantiation -- must NOT compile. Exercised
# (expecting failure) by `mogg-kernel-api-e2e-trait-fused-transform-host-negative.mlir`.
@extensibility.register("trait_fused_transform_host_negative")
struct TraitFusedTransformHostNegative:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & Fused & TransformAccess],
        x: Some[Input & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(z), DevicePassable)
        comptime assert conforms_to(type_of(z), Transformable)
        # Direct host-side transform on the device-passable proxy: the context
        # gate (`comptime assert _access_enabled` in the transform body) rejects it.
        _ = z.transform[1](Coord(0), SIMD[type_of(z).dtype, 1](0))


# Kernel with both a struct-level (`flip`) and fn-level (`dtype`) compile-time
# parameter, where an argument type references the fn-level param. Exercises
# two MOToMOGG paths that broke in different ways during the
# MojoLibraryAnalysis migration:
#   1. `validateCustomOpParameters` rejected legitimate calls passing a
#      struct-level param because the validation only walked fn-level params.
#   2. EmitMojo's `inferByInputs` collection mapped depth-0 `ParamIndexRefAttr`s
#      into `fnLevelParams` alone, but Mojo lays out the generator's param
#      list as `structLevelParams + fnLevelParams`, so the lookup was off by
#      `structLevelParams.size()` -- e.g. for a kernel with struct param
#      `flip` and fn params `[dtype, target]`, an arg referencing `dtype` is
#      `*(0, 1)`, which mapped to `target` (`fnLevelParams[1]`) instead of
#      `dtype` (`(struct + fn)[1]`).
@extensibility.register("imposter_add_struct_param")
struct FooWithStructParam[flip: Bool]:
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        z: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[z.dtype, width]:
            var v = rebind[SIMD[z.dtype, width]](x.load[width](idx))
            comptime if Self.flip:
                return -v
            return v

        foreach[func](z, ctx)


@extensibility.register_shape_function("imposter_add_struct_param")
def imposter_add_struct_param_shape(x: InputTensor) -> IndexList[x.rank]:
    return x.shape()


def _matmul[
    elementwise_lambda_fn: Optional[
        def[
            dtype: DType, width: SIMDLength, *, alignment: Int = 1
        ](IndexList[2], SIMD[dtype, width]) capturing -> None
    ] = None,
](
    c: ManagedTensorSlice[mut=True, ...],
    a: ManagedTensorSlice,
    b: ManagedTensorSlice,
) raises:
    var m = a.dim_size(0)
    var n = a.dim_size(1)
    var k = b.dim_size(1)

    for i in range(m):
        for j in range(n):
            var c_val = Scalar[c.dtype](0)
            for l in range(k):
                var a_val = a.load[1](IndexList[2](i, l)).cast[c.dtype]()
                var b_val = b.load[1](IndexList[2](l, j)).cast[c.dtype]()
                c_val += a_val * b_val

            comptime if elementwise_lambda_fn:
                comptime func = elementwise_lambda_fn.value()
                func(IndexList[2](i, j), c_val)
            else:
                c.store[1](IndexList[2](i, j), c_val)


# c = a @ b, should support CPU and GPU
@extensibility.register("imposter_matmul")
struct ImposterMatmul:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        c: OutputTensor,
        a: InputTensor,
        b: InputTensor,
        ctx: DeviceContext,
    ) raises:
        _matmul(c, a, b)


@extensibility.register_shape_function("imposter_matmul")
def imposter_matmul_shape(
    a: InputTensor,
    b: InputTensor,
) -> IndexList[2]:
    var shape = a.shape()
    shape[1] = b.dim_size[1]()
    return rebind[IndexList[2]](shape)


@extensibility.register("print_tensor_spec")
struct PrintTensorSpecOp:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, x: InputTensor, ctx: DeviceContext) raises:
        print("x.shape =", x._static_shape_tuple)
        print("x.strides =", x._static_strides_tuple)
        print("x.alignment =", x.alignment)
        print("x.address_space =", x.address_space)

        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[outp.dtype, width]:
            return rebind[SIMD[outp.dtype, width]](x.load[width](idx))

        foreach[func](outp, ctx)


@extensibility.register_shape_function("print_tensor_spec")
def print_tensor_spec_shape(x: InputTensor) -> IndexList[x.rank]:
    return x.shape()


@extensibility.register("print_tensor")
struct PrintTensor:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, input: InputTensor, ctx: DeviceContext) raises:
        print(input)

        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[outp.dtype, width]:
            return rebind[SIMD[outp.dtype, width]](input.load[width](idx))

        foreach[func](outp, ctx)


@extensibility.register("print_tensor_spec_view")
@extensibility.view_kernel
struct PrintTensorSpecViewOp:
    @staticmethod
    def print_spec(x: InputTensor):
        print("x.shape =", x._static_shape_tuple)
        print("x.strides =", x._static_strides_tuple)
        print("x.alignment =", x.alignment)
        print("x.address_space =", x.address_space)

    @staticmethod
    def update_input_view(input: InputTensor) -> type_of(input):
        Self.print_spec(input)
        return input

    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, x: InputTensor, ctx: DeviceContext) raises:
        Self.print_spec(x)

        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[outp.dtype, width]:
            return rebind[SIMD[outp.dtype, width]](x.load[width](idx))

        foreach[func](outp, ctx)


@extensibility.register_shape_function("print_tensor_spec_view")
def print_tensor_spec_view_shape(x: InputTensor) -> IndexList[x.rank]:
    return x.shape()


@extensibility.register("print_tensor_spec_fused")
struct PrintTensorSpecFusedOp:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        outp: FusedOutputTensor,
        x: FusedInputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        print("x.shape =", x._static_shape_tuple)
        print("x.strides =", x._static_strides_tuple)
        print("x.alignment =", x.alignment)
        print("x.address_space =", x.address_space)

        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[outp.dtype, width]:
            return rebind[SIMD[outp.dtype, width]](x._fused_load[width](idx))

        foreach[func](outp, ctx)


@extensibility.register_shape_function("print_tensor_spec_fused")
def print_tensor_spec_fused_shape(x: InputTensor) -> IndexList[x.rank]:
    return x.shape()


@extensibility.register("print_output_shape_strides_fused")
struct PrintOutputShapeStridesFusedOp:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: FusedOutputTensor,
        x: FusedInputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        print("output._static_shape_tuple = ", output._static_shape_tuple)
        print("output._static_strides_tuple = ", output._static_strides_tuple)
        print("output._runtime_shape = ", output.shape())
        print("output._runtime_strides = ", output.strides())

        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[output.dtype, width]:
            return (
                rebind[SIMD[output.dtype, width]](x._fused_load[width](idx))
                * 2.0
            )

        foreach[func](output, ctx)


@extensibility.register_shape_function("print_output_shape_strides_fused")
def print_output_shape_strides_fused_shape(x: InputTensor) -> IndexList[x.rank]:
    return x.shape()


@extensibility.register("imposter_add_elementwise")
struct AddElementwise:
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs + rhs


@extensibility.register("imposter_add_lhs")
struct AddFuseLHS:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: OutputTensor,
        x: FusedInputTensor,
        y: InputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](
                x._fused_load[width](idx)
            ) + rebind[SIMD[z.dtype, width]](y.load[width](idx))

        foreach[func](z, ctx)


@extensibility.register("imposter_has_prologue_fusion")
struct ImposterHasPrologueFusion:
    @staticmethod
    def execute[
        target: StaticString,
        has_prologue_fusion: Bool,
    ](
        output: OutputTensor,
        x: FusedInputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[output.dtype, width]:
            return rebind[SIMD[output.dtype, width]](x._fused_load[width](idx))

        foreach[func](output, ctx)


@extensibility.register("imposter_add_fuse_inputs")
struct AddFuseInputs:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: OutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](
                x._fused_load[width](idx)
            ) + rebind[SIMD[z.dtype, width]](y._fused_load[width](idx))

        foreach[func](z, ctx)


# c = a @ b, should support CPU and GPU
@extensibility.register("matmul_fuse_out")
struct MatmulFuseOut:
    @staticmethod
    def execute[
        target: StaticString,
        has_epilogue_fusion: Bool,
    ](
        c: FusedOutputTensor,
        a: InputTensor,
        b: InputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        comptime rank = a.rank
        comptime a_dtype = a.dtype
        comptime b_dtype = b.dtype
        comptime c_dtype = c.dtype

        @__parameter
        @always_inline
        def out_func[
            dtype: DType, width: SIMDLength, *, alignment: Int = 1
        ](idx: IndexList[2], val: SIMD[dtype, width]):
            c._fused_store(idx, rebind[SIMD[c.dtype, width]](val))

        print("has_epilogue_fusion =", has_epilogue_fusion)

        _matmul[elementwise_lambda_fn=out_func](c, a, b)


@extensibility.register_shape_function("matmul_fuse_out")
def matmul_fuse_out_shape(
    a: InputTensor,
    b: InputTensor,
) -> IndexList[2]:
    var shape = a.shape()
    shape[1] = b.dim_size[1]()
    return rebind[IndexList[2]](shape)


# Simple, expects variadics to have the same size, and simply copies the first
# number from the associated inputs to outputs, plus a bias
@extensibility.register("variadic_input_to_output")
struct VariadicInputToOutput:
    @staticmethod
    def execute[
        dtype: DType,
        size: Int,
        target: StaticString,
    ](
        output: OutputVariadicTensors[dtype=dtype, rank=1, size=size, ...],
        bias: InputTensor[dtype=dtype, rank=1, ...],
        input: InputVariadicTensors[dtype=dtype, rank=1, size=size, ...],
    ):
        comptime for i in range(size):
            for j in range(input[i].size()):
                output[i][j] = input[i][j]
            output[i][0] += bias[0]


# Simply adds the first number of bias to the first number of boath outputs
# Mainly here to test logic with multiple DPS outputs
@extensibility.register("add_bias_to_two_tensors")
struct AddBiasToDouble:
    @staticmethod
    def execute[
        rank: Int,
        dtype: DType,
    ](
        output1: OutputTensor[dtype=dtype, rank=rank, ...],
        output2: OutputTensor[dtype=dtype, rank=rank, ...],
        input1: InputTensor[dtype=dtype, rank=rank, ...],
        input2: InputTensor[dtype=dtype, rank=rank, ...],
        bias: InputTensor[dtype=dtype, rank=rank, ...],
    ):
        output1[0] = input1[0] + bias[0]
        output2[0] = input2[0] + bias[0]


@extensibility.register("inplace_increment_elem")
struct BasicInplace:
    @staticmethod
    def execute[
        dtype: DType,
    ](input: MutableInputTensor[dtype=dtype, rank=2, ...]):
        var x = input[0, 0]
        x += 1
        input[0, 0] = x


# Have this nearly identical version as having a raise changes the Mojo function's signature
@extensibility.register("inplace_increment_elem_raises")
struct BasicInplaceRaises:
    @staticmethod
    def execute[
        dtype: DType,
    ](input: MutableInputTensor[dtype=dtype, rank=2, ...]) raises:
        var x = input[0, 0]
        x += 1
        input[0, 0] = x


@extensibility.register("variadic_add")
struct VariadicAdd:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        inputs: FusedInputVariadicTensors[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[dtype, width]:
            var acc = SIMD[dtype, width](0)

            comptime for i in range(inputs.size):
                acc += inputs[i]._fused_load[width](idx)

            return acc

        foreach[func](output, ctx)


@extensibility.register("transpose_2d")
@extensibility.view_kernel
struct Transpose2DOp:
    @staticmethod
    def build_view[
        dtype: DType,
    ](x: InputTensor[dtype=dtype, rank=2, ...]) -> StaticTuple[IndexList[2], 2]:
        var new_stride = IndexList[2]()
        var new_shape = IndexList[2]()
        var x_strides = x.strides()
        var x_shape = x.shape()
        new_stride[0] = x_strides[1]
        new_stride[1] = x_strides[0]
        new_shape[0] = x_shape[1]
        new_shape[1] = x_shape[0]

        return StaticTuple[IndexList[2], 2](new_shape, new_stride)

    @staticmethod
    def get_view_strides(
        input_strides: IntTuple,
    ) -> IndexList[2]:
        # transpose the strides of the input
        return IndexList[2](Int(input_strides[1]), Int(input_strides[0]))

    @staticmethod
    def update_input_view[
        dtype: DType, //, output_static_shape: IntTuple
    ](
        input: InputTensor[dtype=dtype, rank=2, ...],
        out result: InputTensor[
            static_spec=input.static_spec.with_int_tuple_layout[
                2,
                output_static_shape,
                Self.get_view_strides(input._static_strides_tuple),
            ]()
        ],
    ):
        var shape_and_strides = Self.build_view(input)

        return type_of(result)(
            input.unsafe_ptr(), shape_and_strides[0], shape_and_strides[1]
        )

    @staticmethod
    def execute[
        target: StaticString,
        dtype: DType,
    ](
        z: OutputTensor[dtype=dtype, rank=2, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        var x_view = Self.update_input_view[z._static_shape_tuple](x)

        view_copy_impl[target=target](z, x_view, ctx)


@extensibility.register("print_shape_fused")
struct ElementwisePrintShape:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))

        print("input.shape =", x.shape())
        print("output.shape =", z.shape())

        foreach[func](z, ctx)


@extensibility.register_shape_function("print_shape_fused")
def print_shape_fused_shape(x: InputTensor) -> IndexList[x.rank]:
    return x.shape()


# Raises if input shape is 10
@extensibility.register("custom_op_that_raises")
struct CustomOpThatRaises:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: OutputTensor, x: InputTensor, ctx: DeviceContext,) raises:
        if x.shape()[0] == 10:
            raise ("input_shape[0] == 10")

        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))

        foreach[func](z, ctx)


@extensibility.register_shape_function("custom_op_that_raises")
def custom_op_that_raises_shape(x: InputTensor) raises -> IndexList[x.rank]:
    print("Hello")
    var out_shape = x.shape()
    if out_shape[0] == 20:
        raise ("data.get_shape()[0] == 20")
    return out_shape


@extensibility.register("mo.test.failing_constraint")
struct OpThatAlwaysFailsConstraint:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank, ...],
        in_tensor: InputTensor[dtype=dtype, rank=rank, ...],
    ):
        comptime assert (
            1 == 2
        ), "Expected constraint failure for error message testing"


@extensibility.register("mo.test.return_error")
struct OpThatAlwaysRaises:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank, ...],
        in_tensor: InputTensor[dtype=dtype, rank=rank, ...],
    ) raises:
        out_tensor[0] = in_tensor[0]
        raise Error("This is an error")


@extensibility.register("monnx.abs_v13")
struct MONNXAbsOverload:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank, ...],
        in_tensor: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[dtype, width]:
            return abs(in_tensor._fused_load[width](idx))

        print("The custom identity op is running!")
        foreach[func](out_tensor, ctx)


@extensibility.register("torch.aten.abs")
struct MTorchAbsOverload:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank, ...],
        in_tensor: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[dtype, width]:
            return abs(in_tensor._fused_load[width](idx))

        print("The custom identity op is running!")
        foreach[func](out_tensor, ctx)


@extensibility.register("op_with_custom_params")
struct OpWithCustomParams:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        custom_int: Int,
        custom_str: StaticString,
        custom_dtype: DType,
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank, ...],
        in_tensor: InputTensor[dtype=dtype, rank=rank, ...],
    ) raises:
        out_tensor[0] = in_tensor[0]
        print("custom_int =", custom_int)
        print("custom_str =", custom_str)
        print("custom_dtype =", custom_dtype)


@extensibility.register("mgprt_test_func")
struct MGPRTTestFunc:
    @staticmethod
    def execute(out_tensor: OutputTensor) raises:
        external_call["MGP_RT_TEST", NoneType]()


@extensibility.register("mutable_test_op")
struct MutableTestOp:
    @staticmethod
    def execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.unsafe_store(0, 0)


# For testing support for Scalar[...] in Mojo
@extensibility.register("supports_scalar_kernel")
struct SupportsScalarKernel:
    @staticmethod
    def execute[
        dtype: DType
    ](
        outp: OutputTensor[dtype=dtype, rank=1, ...],
        x: InputTensor[dtype=dtype, rank=1, ...],
        y: Scalar[dtype],
    ) raises:
        print("datatype is", dtype)
        print("value is", y)


@extensibility.register("kernel_with_no_target")
struct KernelWithNoTarget:
    @staticmethod
    def execute[
        dtype: DType
    ](
        outp: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
    ) raises:
        print("hello from kernel with no target")


@extensibility.register("basic_target")
struct BasicTarget:
    @staticmethod
    def execute[
        dtype: DType, target: StaticString
    ](
        outp: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
    ) raises:
        print("hello from kernel on", target)


struct MyCustomScalarReg[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    var val: Scalar[Self.dtype]

    @implicit
    def __init__(out self, val: Scalar[Self.dtype]):
        print("MyCustomScalarReg.__init__", val)
        self.val = val

    def __deinit__(deinit self):
        print("MyCustomScalarReg.__deinit__", self.val)


@extensibility.register("buff_to_my_custom_scalar_reg")
struct BuffToMyCustomScalarReg:
    @staticmethod
    def execute[
        target: StaticString
    ](x: InputTensor[dtype=DType.int32, rank=1, ...]) -> MyCustomScalarReg[
        DType.int32
    ]:
        return MyCustomScalarReg(x[0])


@extensibility.register("my_custom_scalar_reg_to_buff")
struct CustomScalarRegToBuff:
    @staticmethod
    def execute[
        target: StaticString
    ](
        input: OutputTensor[dtype=DType.int32, rank=1, ...],
        x: MyCustomScalarReg[DType.int32],
    ):
        input[0] = x.val


@extensibility.register("test_custom_op")
struct TestCustomOp:
    @staticmethod
    def execute[
        target: StaticString, dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
    ):
        print("World!")


@extensibility.register_shape_function("test_custom_op")
def test_custom_op_shape[
    dtype: DType, rank: Int
](input: InputTensor[dtype=dtype, rank=rank, ...]) -> IndexList[rank]:
    print("Hello")
    return input.shape()


# A custom op whose shape function is trait-typed: a metadata-only, no-role
# `Some[DenseTensor]` argument that reads only `.shape()`. This exercises the
# trait shape-fn emit path (the operand is constructed as a `_MoggShapeTensorArg`
# instead of a legacy `InputTensor`). The `execute` entry stays on the legacy
# lane -- the shape-fn migration is independent of how `execute` is written --
# and copies input to output so the result is verifiable end-to-end.
@extensibility.register("trait_metadata_shape_op")
struct TraitMetadataShapeOp:
    @staticmethod
    def execute[
        target: StaticString, dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[output.dtype, width]:
            return input.load[width](idx)

        foreach[func](output, ctx)


@extensibility.register_shape_function("trait_metadata_shape_op")
def trait_metadata_shape_op_shape[
    T: DenseTensor
](input: T) -> IndexList[T.rank]:
    return rebind[IndexList[T.rank]](coord_to_index_list(input.shape().tuple()))


# A custom op whose shape function binds the bare `Tensor` metadata trait
# (shape only, no layout/alignment), unlike `trait_metadata_shape_op` above
# which binds `DenseTensor`. This exercises the `_data_visible=False` emit
# path -- the emitted `_MoggShapeTensorArg` conforms only to `Tensor`, so it
# cannot see layout or read elements. `execute` stays on the legacy lane and
# copies input to output so the result is verifiable end-to-end.
@extensibility.register("trait_tensor_shape_op")
struct TraitTensorShapeOp:
    @staticmethod
    def execute[
        target: StaticString, dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[output.dtype, width]:
            return input.load[width](idx)

        foreach[func](output, ctx)


@extensibility.register_shape_function("trait_tensor_shape_op")
def trait_tensor_shape_op_shape[T: Tensor](input: T) -> IndexList[T.rank]:
    return rebind[IndexList[T.rank]](coord_to_index_list(input.shape().tuple()))


# A custom op whose shape function reads element values off a
# `TileTensorable` argument (the `sizes` input) via `to_tile_tensor()`, rather
# than just its metadata -- shape functions may read data since they always
# execute on the host. The `data` argument binds the bare `Tensor` metadata
# trait, mixing tiers within one shape function for extra coverage. `execute`
# stays on the legacy lane and copies `data` into `output` at matching
# coordinates, so the produced values are verifiable end-to-end even though
# the output shape comes from `sizes`, not from `data`'s own shape.
@extensibility.register("trait_reshape_shape_op")
struct TraitReshapeShapeOp:
    @staticmethod
    def execute[
        target: StaticString, dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        data: InputTensor[dtype=dtype, rank=2, ...],
        sizes: InputTensor[dtype=DType.int64, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[output.dtype, width]:
            return data.load[width](idx)

        foreach[func](output, ctx)


@extensibility.register_shape_function("trait_reshape_shape_op")
def trait_reshape_shape_op_shape[
    D: Tensor, S: TileTensorable
](data: D, sizes: S) -> IndexList[2]:
    var new_shape = sizes.to_tile_tensor()
    var shape = IndexList[2]()
    shape[0] = Int(new_shape[0])
    shape[1] = Int(new_shape[1])
    return shape


@extensibility.register("single_device_context")
struct SingleDeviceContext:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
        dev_ctx: DeviceContextArray[1],
    ) raises:
        dev_ctx[0].synchronize()


@extensibility.register("multi_device_context")
struct MultiDeviceContext:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
        dev_ctxs: DeviceContextArray[2],
    ) raises:
        print("dev_ctx0.id() =", dev_ctxs[0].id())
        print("dev_ctx1.id() =", dev_ctxs[1].id())
        dev_ctxs[0].synchronize()
        dev_ctxs[1].synchronize()


@extensibility.register("multi_device_context_dedup")
struct MultiDeviceContextDedup:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
        y: InputTensor[dtype=dtype, ...],
        dev_ctxs: DeviceContextArray[2],
    ) raises:
        dev_ctxs[0].synchronize()
        dev_ctxs[1].synchronize()


@extensibility.register("variadic_device_context")
struct VariadicDeviceContext:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString = "gpu",
    ](
        outputs: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        dev_ctxs: DeviceContextArray,
    ) raises:
        for i in range(len(dev_ctxs)):
            print("dev_ctxs[", i, "].id() =", dev_ctxs[i].id())
            dev_ctxs[i].synchronize()


@extensibility.register("imposter_cast_elementwise")
struct CastElementwise:
    @staticmethod
    def elementwise[
        dtype: DType,
        out_dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width]) -> SIMD[out_dtype, width]:
        return x.cast[out_dtype]()


@extensibility.register("print_vector_size")
struct PrintVectorSize:
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width], idx: IndexList) -> SIMD[dtype, width]:
        if idx[0] == 0:
            # Only print once
            # We can't directly print `width` as it's HW-dependent.
            # Instead compare it with the width of i8 / f32.
            print(
                "width == simd_width_of[DType.int8]():",
                width == simd_width_of[DType.int8](),
            )
            print(
                "width == simd_width_of[DType.float32]():",
                width == simd_width_of[DType.float32](),
            )
        return x


@extensibility.register("tensor_opaque_tensor_kernel")
struct TensorOpaqueTensorKernel:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output1: OutputTensor,
        output2: OutputTensor,
        opaque: MyCustomScalarSI32,
        input: InputTensor,
    ) raises:
        pass


@extensibility.register("test_same_rank_dtype")
struct TestSameRankDtype:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
    ):
        output[0] = input[0]


@extensibility.register("test_fix_rank")
struct TestFixRank:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: InputTensor[dtype=dtype, rank=2, ...],
    ):
        output[0] = input[0]


@extensibility.register("test_simd_input_size_1")
struct TestSIMDInputSize1:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: SIMD[dtype=dtype, length=1],
    ):
        output[0] = input[0]


@extensibility.register("test_simd_input")
struct TestSIMDInput:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: SIMD[dtype=dtype, ...],
    ):
        output[0] = input[0]


@extensibility.register("test_simd_input_size_2")
struct TestSIMDInputSize2:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: SIMD[dtype=dtype, length=2],
    ):
        output[0] = input[0]


@extensibility.register("custom_op_with_custom_parameter_shape_func")
struct CustomOpWithCustomParameterShapeFunc:
    @staticmethod
    def execute[
        test_flag: Int
    ](y: FusedOutputTensor, x: FusedInputTensor) capturing raises:
        raise "This should not run!"


@extensibility.register_shape_function(
    "custom_op_with_custom_parameter_shape_func"
)
def custom_op_with_custom_parameter_shape_func_shape[
    test_flag: Int
](input_tensor: InputTensor) -> IndexList[input_tensor.rank]:
    return input_tensor.shape()


@extensibility.register("custom_op_with_static_string_params")
struct CustomOpWithStaticStringParams:
    @staticmethod
    def execute[target: StaticString, foo: StaticString]():
        print(target)
        print(foo)


comptime _WrapperLayout = TileLayout[
    shape_types=Coord[Int64, Int64].element_types,
    stride_types=Coord[Int64, Int64].element_types,
]


@fieldwise_init
struct TensorWrapper(RegisterPassable):
    var buffer: TileTensor[
        dtype=DType.float32,
        origin=MutUntrackedOrigin,
        LayoutType=_WrapperLayout,
    ]


@extensibility.register("to_tensor_wrapper")
struct ToTensorWrapper:
    @always_inline
    @staticmethod
    def execute[
        spec: StaticTensorSpec[dtype=DType.float32, rank=2, ...], //
    ](tensor: InputTensor[static_spec=spec]) -> TensorWrapper:
        var shape = tensor.shape()
        var strides = tensor.strides()
        var shape_coord = Coord[Int64, Int64]()
        var stride_coord = Coord[Int64, Int64]()
        shape_coord[0] = Int64(shape[0])
        shape_coord[1] = Int64(shape[1])
        stride_coord[0] = Int64(strides[0])
        stride_coord[1] = Int64(strides[1])
        return TensorWrapper(
            TileTensor(
                tensor.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                _WrapperLayout(shape_coord, stride_coord),
            )
        )


@extensibility.register("print_tensor_wrapper")
struct TensorWrapperPrinter:
    @always_inline
    @staticmethod
    def execute(
        tw: TensorWrapper,
    ):
        print(tw.buffer)


@extensibility.register("fused_write_to_tensor_wrapper")
struct MutableStore:
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        tw: TensorWrapper,
        tensor: FusedInputTensor[dtype=DType.float32, rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @__parameter
        @always_inline
        def func[
            width: Int
        ](idx: IndexList[tensor.rank]) -> SIMD[DType.float32, width]:
            return rebind[SIMD[tensor.dtype, width]](
                tensor._fused_load[width](idx)
            )

        @__parameter
        @always_inline
        def out_func[width: Int](index: IndexList[tensor.rank]) capturing:
            var idx = rebind[IndexList[tensor.rank]](index)
            var val = func[width](idx)
            idx[0] += 1
            idx[1] += 1
            tw.buffer.store_linear[width=width](idx, val)

        foreach_out_func[
            func,
            out_func,
            target=target,
            _trace_name=_trace_name,
        ](tensor, ctx)


@extensibility.register("my_square_compute_out_fusion")
struct MySquareComputeOutFusion:
    @staticmethod
    def execute[
        dtype: DType
    ](
        outp: _FusedComputeOutputTensor[dtype=dtype, rank=1, ...],
        input: InputTensor[dtype=dtype, rank=1, ...],
    ) capturing:
        var res = input[0] * input[0]
        comptime has_compute_lambda = type_of(outp)._has_compute_fusion
        comptime has_store_lambda = type_of(outp)._has_output_store_fusion
        print("has_compute_lambda =", has_compute_lambda)
        print("has_store_lambda =", has_store_lambda)

        comptime if has_compute_lambda:
            outp[0] = outp._fused_compute_output_lambda[width=1](
                IndexList[1](0), res
            )
        else:
            # We still need to support normal fusion in case the kernel is fused with a cast.
            outp._fused_store[width=1](IndexList[1](0), res)


@extensibility.register("inplace_no_out_func")
struct InplaceNoOutFunc:
    # This kernel is meant to test that epilogue fusion will
    # be skipped if the out_func overload of foreach is not used.
    @staticmethod
    def execute(
        output: MutableInputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[output.dtype, width]:
            var val = input._fused_load[width](idx)
            return val * SIMD[output.dtype, width](2.0)

        foreach[func](output, ctx)


@fieldwise_init
struct SIMDPair[S0: Int, S1: Int](RegisterPassable):
    var x: SIMD[DType.int32, Self.S0]
    var y: SIMD[DType.int32, Self.S1]


@extensibility.register("make_simd_pair")
struct MakeSimdPair:
    @staticmethod
    def execute[P0: Int, P1: Int]() -> SIMDPair[P0, P1]:
        return SIMDPair[P0, P1](
            iota[DType.int32, P0](), iota[DType.int32, P1](Int32(P0))
        )


@extensibility.register("kernel_with_parameterized_opaque")
struct ParameterizedOpaqueType:
    @staticmethod
    def execute[
        P0: Int
    ](
        output: OutputTensor[dtype=DType.int32, rank=1, ...],
        x: SIMDPair[P0, _],
    ) capturing:
        output.store(IndexList[1](0), x.x)
        output.store(IndexList[1](P0), x.y)


@extensibility.register_shape_function("kernel_with_parameterized_opaque")
def kernel_with_parameterized_opaque_shape[
    P0: Int
](x: SIMDPair[P0, _]) -> IndexList[1]:
    return IndexList[1](x.S0 + x.S1)


@extensibility.register("kernel_with_default_parameter")
struct KernelWithDefaultParameter:
    @staticmethod
    def execute[
        rank: Int,
        rank_defaulted: Int = 128,
        rank_derived_from_other: Int = rank + 1,
        rank_derived_from_derived: Int = rank_derived_from_other + 1,
    ](x: OutputTensor[dtype=DType.int32, rank=1, ...], y: InputTensor):
        print("rank =", rank)
        print("rank_defaulted =", rank_defaulted)
        print("rank_derived_from_other =", rank_derived_from_other)
        print("rank_derived_from_derived =", rank_derived_from_derived)
        x.store(IndexList[1](0), Int32(0))


@extensibility.register("unique-parameters")
struct UniqueParameters:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int,
        target: StaticString,
    ](
        blocks: InputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_lengths: InputTensor[dtype=DType.uint32, rank=2, ...],
    ) -> MyCustomScalarRegTrivialSI32:
        comptime if page_size:
            return MyCustomScalarRegTrivialSI32(0)
        else:
            return MyCustomScalarRegTrivialSI32(1)


@extensibility.register("shape_func_with_mutable")
struct ShapeFuncWithMutable:
    @staticmethod
    def execute(
        output: OutputTensor[rank=2, ...],
        x: MutableInputTensor[dtype=output.dtype, rank=2, ...],
    ):
        x[0, 0] = 2.0
        output[0, 0] = 1.0


@extensibility.register_shape_function("shape_func_with_mutable")
def shape_func_with_mutable_shape(
    x: InputTensor[rank=2, ...],
) -> IndexList[2]:
    return x.shape()


@extensibility.register("elemwise_add_with_param")
struct ElemwiseAddWithParam:
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
        bias: Int,
    ](input: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return input + Scalar[dtype](bias)


@extensibility.register("elemwise_add_with_index")
struct ElemwiseAddWithIndex:
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
        rank: Int,
    ](input: SIMD[dtype, width], indices: IndexList[rank]) -> SIMD[
        dtype, width
    ]:
        var bias = SIMD[dtype, width]()
        comptime for i in range(width):
            bias[i] = Scalar[dtype](indices[0] + i)
        return input + bias


# Kernel that exposes only an `execute` entry point. Device-graph synthesis
# must record it through the builder's recording context, since there is no
# `build_graph` method to call.
@extensibility.register("imposter_execute_only")
struct ExecuteOnly:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        pass


# Kernel that exposes both execute and build_graph entry points.
@extensibility.register("imposter_build_graph_capable")
struct BuildGraphCapable:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        builder: DeviceGraphBuilder,
    ):
        pass


@extensibility.register("device_graph_fill")
struct DeviceGraphFill:
    """Device-graph test kernel that fills its output with the constant `7`.

    Unlike `imposter_build_graph_capable` (a no-op stub used for emit tests),
    `build_graph` here records a *real* compiled fill kernel via
    `DeviceGraphBuilder.add_function`, so a replayed graph produces a
    deterministic output that an end-to-end execution test can check. `execute`
    performs the equivalent fill on the direct (non-graph) path.
    """

    comptime _fill_value = 7

    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def func[width: Int, alignment: Int = 1](idx: Coord) capturing:
            output._fused_store[element_alignment=alignment](
                idx, SIMD[dtype, width](Self._fill_value)
            )

        elementwise[func, 1, target="gpu"](output.shape_coord(), ctx)

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        builder: DeviceGraphBuilder,
    ) raises:
        var ptr = output.unsafe_ptr()
        var length = output.size()

        # Record a real compiled kernel that fills every output element with the
        # constant. The closure captures the output pointer and length by value;
        # `add_function` compiles it against the builder's device context and
        # adds it as a graph node, so replaying the graph re-runs this fill.
        def fill_kernel() {var}:
            var tid = global_idx.x
            if tid >= length:
                return
            ptr[unsafe_offset=tid] = Scalar[dtype](Self._fill_value)

        comptime block_dim = 256
        _ = builder.add_function(
            fill_kernel,
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )


@extensibility.register("device_graph_increment")
struct DeviceGraphIncrement:
    """Device-graph test kernel that adds the constant `1` to every element.

    The end-to-end device-graph test chains this after `device_graph_fill`, so
    replaying the recorded graph computes `fill(7) + 1 == 8` for every element.
    Like `device_graph_fill`, `build_graph` records a *real* compiled kernel via
    `DeviceGraphBuilder.add_function` and `execute` performs the equivalent
    increment on the direct (non-graph) path.
    """

    comptime _increment = 1

    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def func[width: Int, alignment: Int = 1](idx: Coord) capturing:
            var val = rebind[SIMD[dtype, width]](x.load[width](idx)) + (
                Self._increment
            )
            output._fused_store[element_alignment=alignment](idx, val)

        elementwise[func, 1, target="gpu"](output.shape_coord(), ctx)

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        builder: DeviceGraphBuilder,
    ) raises:
        var in_ptr = x.unsafe_ptr()
        var out_ptr = output.unsafe_ptr()
        var length = output.size()

        # Record a real compiled kernel that reads each input element, adds the
        # constant, and writes it to the output. The closure captures the input
        # and output pointers and the length by value; `add_function` compiles it
        # against the builder's device context and adds it as a graph node, so
        # replaying the graph re-runs this increment.
        def increment_kernel() {var}:
            var tid = global_idx.x
            if tid >= length:
                return
            out_ptr[unsafe_offset=tid] = in_ptr[unsafe_offset=tid] + Scalar[
                dtype
            ](Self._increment)

        comptime block_dim = 256
        _ = builder.add_function(
            increment_kernel,
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )


@extensibility.register("add_one", type="cpu")
struct AddOneCPU:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, x: InputTensor, ctx: DeviceContext,) raises:
        print("AddOne CPU kernel")

        @__parameter
        @always_inline
        def func[width: Int, alignment: Int = 1](idx: Coord) capturing:
            var val = rebind[SIMD[outp.dtype, width]](x.load[width](idx)) + 1
            outp._fused_store[element_alignment=alignment](idx, val)

        elementwise[func, 1, target="cpu"](outp.shape_coord(), ctx)


@extensibility.register("gpu_only_kernel", type="gpu")
struct GpuOnlyKernel:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, x: InputTensor, ctx: DeviceContext,) raises:
        pass


@extensibility.register("add_one", type="gpu")
struct AddOneGPU:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, x: InputTensor, ctx: DeviceContext,) raises:
        print("AddOne GPU kernel")

        @__parameter
        @always_inline
        def func[width: Int, alignment: Int = 1](idx: Coord) capturing:
            var val = rebind[SIMD[outp.dtype, width]](x.load[width](idx)) + 1
            outp._fused_store[element_alignment=alignment](idx, val)

        elementwise[func, 1, target="gpu"](outp.shape_coord(), ctx)


# Mixed-lane e2e proof: a trait-bound dense output alongside two legacy
# `InputTensor` operands in the same `execute` signature, computing
# `z[i] = x[i] + y[i]`. Confirms the trait-lane and legacy-lane capture
# assembly compose in the same `CallExecuteOp::emitMojo` call.
@extensibility.register("trait_mixed_legacy_add")
struct TraitMixedLegacyAdd:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & DenseTensor],
        x: InputTensor,
        y: InputTensor,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(z), TileTensorable)
        var zt = to_tile_tensor(z)
        var n = x.size()
        for i in range(n):
            var xv = x[i]
            var yv = rebind[Scalar[type_of(x).dtype]](y[i])
            zt.raw_store(i, rebind[SIMD[type_of(zt).dtype, 1]](xv + yv))


# Mirror of `trait_mixed_legacy_add` with the lanes swapped: a legacy
# `OutputTensor` alongside two trait-bound dense inputs, so the legacy write
# path (`OutputTensor.__setitem__`) and the trait read path
# (`to_tile_tensor(...).raw_load`) both fire in the same emitted kernel.
@extensibility.register("trait_mixed_legacy_out_add")
struct TraitMixedLegacyOutAdd:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: OutputTensor,
        x: Some[Input & DenseTensor],
        y: Some[Input & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime assert conforms_to(type_of(y), TileTensorable)
        var xt = to_tile_tensor(x)
        var yt = to_tile_tensor(y)
        var n = Int(x.shape().product())
        for i in range(n):
            var xv = xt.raw_load(i)
            var yv = rebind[SIMD[type_of(xt).dtype, 1]](yt.raw_load(i))
            z[i] = rebind[Scalar[z.dtype]](xv + yv)


# ===----------------------------------------------------------------------=== #
# Signature-only opaque kernels for exercising MOToMAP lowering coverage.
#
# The bodies are empty: MOToMAP reads only the argument roles (fused vs plain
# output) to decide which results get a store epilogue, so these kernels cover
# one/two outputs across all-fused, none-fused and partially-fused shapes.
# ===----------------------------------------------------------------------=== #


@extensibility.register("map_lower_one_fused_output")
struct MapLowerOneFusedOutput:
    @staticmethod
    def execute[
        target: StaticString,
        has_epilogue_fusion: Bool,
    ](
        res: FusedOutputTensor,
        a: InputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        pass


@extensibility.register("map_lower_one_plain_output")
struct MapLowerOnePlainOutput:
    @staticmethod
    def execute[
        target: StaticString,
    ](res: OutputTensor, a: InputTensor, ctx: DeviceContext,) raises:
        pass


@extensibility.register("map_lower_two_fused_outputs")
struct MapLowerTwoFusedOutputs:
    @staticmethod
    def execute[
        target: StaticString,
        has_epilogue_fusion: Bool,
    ](
        res0: FusedOutputTensor,
        res1: FusedOutputTensor,
        a: InputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        pass


@extensibility.register("map_lower_two_plain_outputs")
struct MapLowerTwoPlainOutputs:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        res0: OutputTensor,
        res1: OutputTensor,
        a: InputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("map_lower_partial_fused_outputs")
struct MapLowerPartialFusedOutputs:
    @staticmethod
    def execute[
        target: StaticString,
        has_epilogue_fusion: Bool,
    ](
        res0: FusedOutputTensor,
        res1: OutputTensor,
        a: InputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        pass


# Two plain inputs feeding a single fused output: exercises the opaque body's
# output block arg (whose arg number is offset by the two leading input block
# args) resolving to the right captured tensor.
@extensibility.register("map_lower_two_inputs_one_fused_output")
struct MapLowerTwoInputsOneFusedOutput:
    @staticmethod
    def execute[
        target: StaticString,
        has_epilogue_fusion: Bool,
    ](
        res: FusedOutputTensor,
        a: InputTensor,
        b: InputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        pass


@extensibility.register("map_lower_scalar")
struct MapLowerScalar:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        res: OutputTensor,
        a: InputTensor,
        s: Scalar[dtype],
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("map_lower_variadic")
struct MapLowerVariadic:
    @staticmethod
    def execute[
        dtype: DType,
        size: Int,
        target: StaticString,
    ](
        res: OutputTensor,
        inputs: InputVariadicTensors[dtype=dtype, rank=1, size=size, ...],
        ctx: DeviceContext,
    ) raises:
        pass


# A real-body counterpart to `map_lower_one_plain_output`: same plain-output
# signature (so it lowers through the identical MAP plain path), but it actually
# writes `res = a + 1` so an end-to-end run can validate a numeric result.
@extensibility.register("map_lower_add_one")
struct MapLowerAddOne:
    @staticmethod
    def execute[
        target: StaticString,
    ](res: OutputTensor, a: InputTensor, ctx: DeviceContext,) raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[res.dtype, width]:
            return rebind[SIMD[res.dtype, width]](a.load[width](idx)) + 1

        foreach[func, target=target](res, ctx)


# Real-body plain-output kernels below broaden the advanced-fusion MAP-ABI
# execution coverage past the single 1-in/1-out shape. Each mirrors
# `map_lower_add_one`: a plain `OutputTensor` (no epilogue fusion) driven by a
# real `foreach` closure, so the MAP plain path both lowers AND computes a
# deterministic result an end-to-end run can validate. Output shape is
# identity-from-first-input for all of them, so none needs a shape function.


# 3 inputs / 1 output: `res = a + b + c`.
@extensibility.register("map_lower_add3")
struct MapLowerAdd3:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        res: OutputTensor,
        a: InputTensor,
        b: InputTensor,
        c: InputTensor,
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[res.dtype, width]:
            return (
                rebind[SIMD[res.dtype, width]](a.load[width](idx))
                + rebind[SIMD[res.dtype, width]](b.load[width](idx))
                + rebind[SIMD[res.dtype, width]](c.load[width](idx))
            )

        foreach[func, target=target](res, ctx)


# 1 input / 2 outputs: `res0 = a + 1`, `res1 = a * 2` (two plain outputs, two
# independent `foreach` loops).
@extensibility.register("map_lower_two_outputs")
struct MapLowerTwoOutputs:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        res0: OutputTensor,
        res1: OutputTensor,
        a: InputTensor,
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def add_one[width: Int](idx: Coord) -> SIMD[res0.dtype, width]:
            return rebind[SIMD[res0.dtype, width]](a.load[width](idx)) + 1

        foreach[add_one](res0, ctx)

        @__parameter
        @always_inline
        def times_two[width: Int](idx: Coord) -> SIMD[res1.dtype, width]:
            return rebind[SIMD[res1.dtype, width]](a.load[width](idx)) * 2

        foreach[times_two](res1, ctx)


# 2 inputs / 2 outputs: `res0 = a + b`, `res1 = a - b`.
@extensibility.register("map_lower_add_sub")
struct MapLowerAddSub:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        res0: OutputTensor,
        res1: OutputTensor,
        a: InputTensor,
        b: InputTensor,
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def add[width: Int](idx: Coord) -> SIMD[res0.dtype, width]:
            return rebind[SIMD[res0.dtype, width]](a.load[width](idx)) + rebind[
                SIMD[res0.dtype, width]
            ](b.load[width](idx))

        foreach[add, target=target](res0, ctx)

        @__parameter
        @always_inline
        def sub[width: Int](idx: Coord) -> SIMD[res1.dtype, width]:
            return rebind[SIMD[res1.dtype, width]](a.load[width](idx)) - rebind[
                SIMD[res1.dtype, width]
            ](b.load[width](idx))

        foreach[sub, target=target](res1, ctx)


# Scalar operand: `res = a * s` where `s: Scalar[dtype]`. Exercises the MAP
# `scalar.create` materialization path with a real computed result.
@extensibility.register("map_lower_scale")
struct MapLowerScale:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        res: OutputTensor[dtype=dtype, ...],
        a: InputTensor[dtype=dtype, ...],
        s: Scalar[dtype],
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[res.dtype, width]:
            return rebind[SIMD[res.dtype, width]](a.load[width](idx)) * rebind[
                Scalar[res.dtype]
            ](s)

        foreach[func](res, ctx)


# Variadic operand: `res = sum(inputs)` over an `InputVariadicTensors`.
# Exercises the MAP `tuple.create` materialization path. Modeled on the
# signature-only `map_lower_variadic` but with a real summing body.
@extensibility.register("map_lower_var_sum")
struct MapLowerVarSum:
    @staticmethod
    def execute[
        dtype: DType,
        size: Int,
        target: StaticString,
    ](
        res: OutputTensor[dtype=dtype, rank=1, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=1, size=size, ...],
        ctx: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[res.dtype, width]:
            var acc = SIMD[res.dtype, width](0)

            comptime for i in range(size):
                acc += rebind[SIMD[res.dtype, width]](
                    inputs[i].load[width](idx)
                )

            return acc

        foreach[func](res, ctx)


@extensibility.register("map_lower_two_fused_inputs")
struct MapLowerTwoFusedInputs:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        res: OutputTensor,
        a: FusedInputTensor,
        b: FusedInputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        pass


@extensibility.register("map_lower_partial_fused_inputs")
struct MapLowerPartialFusedInputs:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        res: OutputTensor,
        a: FusedInputTensor,
        b: InputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        pass
