# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from std.collections import Optional
from std.collections.string import StaticString
from std.math import iota
from std.ffi import external_call
from std.sys.info import simd_width_of
import std.algorithm.functional
import extensibility as compiler
from layout import IntTuple
from layout import TileTensor, CoordLike, Coord
from layout.tile_layout import Layout as TileLayout
from extensibility import StaticTensorSpec
from std.collections import InlineArray
from std.gpu.host import DeviceContext, DeviceContextList, DeviceGraphBuilder
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
    foreach,
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


@compiler.register("tensor_to_custom_scalar_si32_reg_trivial")
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

    def __del__(deinit self):
        print("MyCustomScalarRegSI32.__del__", self.val)


# It is intentional there are no methods which consume this.
# It is here to support some level of type checking.
struct MyCustomScalarRegF32(ImplicitlyCopyable, RegisterPassable):
    var val: Float32

    @implicit
    def __init__(out self, val: Float32):
        print("MyCustomScalarRegF32.__init__", val)
        self.val = val

    def __del__(deinit self):
        print("MyCustomScalarRegF32.__del__", self.val)


@compiler.register("tensor_to_custom_scalar_si32_reg")
struct OpaqueToCustomScalarSI32Reg:
    @staticmethod
    def execute(
        x: InputTensor[dtype=DType.int32, rank=1, ...]
    ) -> MyCustomScalarRegSI32:
        return MyCustomScalarRegSI32(x[0])


# Adds two custom scalar types (one of which is register passable) and writes
# to a tensor
@compiler.register("opaque_add_to_tensor_si32_reg")
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


@compiler.register("opaque_add_to_tensor_f32_reg")
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

    def __del__(deinit self):
        print("MyCustomScalarSI32.__del__", self.val)


@compiler.register("tensor_to_custom_scalar_si32")
struct OpaqueToCustomScalarSI32:
    @staticmethod
    def execute(
        x: InputTensor[dtype=DType.int32, rank=1, ...]
    ) -> MyCustomScalarSI32:
        return MyCustomScalarSI32(x[0])


# Adds two custom scalar types (one of which is register passable) and writes
# to a tensor
@compiler.register("opaque_add_to_tensor_si32")
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


@compiler.register("opaque_add_to_tensor_si32_raises")
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


@compiler.register("imposter_add")
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
        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](x.load[width](idx)) + rebind[
                SIMD[z.dtype, width]
            ](y.load[width](idx))

        foreach[func](z, ctx)


@compiler.register_shape_function("imposter_add")
def imposter_add_shape(x: InputTensor, y: InputTensor) -> IndexList[x.rank]:
    return x.shape()


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
@compiler.register("imposter_add_struct_param")
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
        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[z.dtype, width]:
            var v = rebind[SIMD[z.dtype, width]](x.load[width](idx))
            comptime if Self.flip:
                return -v
            return v

        foreach[func](z, ctx)


@compiler.register_shape_function("imposter_add_struct_param")
def imposter_add_struct_param_shape(x: InputTensor) -> IndexList[x.rank]:
    return x.shape()


def _matmul[
    elementwise_lambda_fn: Optional[
        def[
            dtype: DType, width: SIMDSize, *, alignment: Int = 1
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
@compiler.register("imposter_matmul")
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


@compiler.register_shape_function("imposter_matmul")
def imposter_matmul_shape(
    a: InputTensor,
    b: InputTensor,
) -> IndexList[2]:
    var shape = a.shape()
    shape[1] = b.dim_size[1]()
    return rebind[IndexList[2]](shape)


@compiler.register("print_tensor_spec")
struct PrintTensorSpecOp:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, x: InputTensor, ctx: DeviceContext,) raises:
        print("x.shape =", x._static_shape_tuple)
        print("x.strides =", x._static_strides_tuple)
        print("x.alignment =", x.alignment)
        print("x.address_space =", x.address_space)

        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[outp.dtype, width]:
            return rebind[SIMD[outp.dtype, width]](x.load[width](idx))

        foreach[func](outp, ctx)


@compiler.register_shape_function("print_tensor_spec")
def print_tensor_spec_shape(x: InputTensor) -> IndexList[x.rank]:
    return x.shape()


@compiler.register("print_tensor")
struct PrintTensor:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, input: InputTensor, ctx: DeviceContext,) raises:
        print(input)

        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[outp.dtype, width]:
            return rebind[SIMD[outp.dtype, width]](input.load[width](idx))

        foreach[func](outp, ctx)


@compiler.register("print_tensor_spec_view")
@compiler.view_kernel
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
    ](outp: OutputTensor, x: InputTensor, ctx: DeviceContext,) raises:
        Self.print_spec(x)

        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[outp.dtype, width]:
            return rebind[SIMD[outp.dtype, width]](x.load[width](idx))

        foreach[func](outp, ctx)


@compiler.register_shape_function("print_tensor_spec_view")
def print_tensor_spec_view_shape(x: InputTensor) -> IndexList[x.rank]:
    return x.shape()


@compiler.register("print_tensor_spec_fused")
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

        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[outp.dtype, width]:
            return rebind[SIMD[outp.dtype, width]](x._fused_load[width](idx))

        foreach[func](outp, ctx)


@compiler.register_shape_function("print_tensor_spec_fused")
def print_tensor_spec_fused_shape(x: InputTensor) -> IndexList[x.rank]:
    return x.shape()


@compiler.register("print_output_shape_strides_fused")
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

        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[output.dtype, width]:
            return (
                rebind[SIMD[output.dtype, width]](x._fused_load[width](idx))
                * 2.0
            )

        foreach[func](output, ctx)


@compiler.register_shape_function("print_output_shape_strides_fused")
def print_output_shape_strides_fused_shape(x: InputTensor) -> IndexList[x.rank]:
    return x.shape()


@compiler.register("imposter_add_elementwise")
struct AddElementwise:
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs + rhs


@compiler.register("imposter_add_lhs")
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
        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](
                x._fused_load[width](idx)
            ) + rebind[SIMD[z.dtype, width]](y.load[width](idx))

        foreach[func](z, ctx)


@compiler.register("imposter_has_prologue_fusion")
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
        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[output.dtype, width]:
            return rebind[SIMD[output.dtype, width]](x._fused_load[width](idx))

        foreach[func](output, ctx)


@compiler.register("imposter_add_fuse_inputs")
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
        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](
                x._fused_load[width](idx)
            ) + rebind[SIMD[z.dtype, width]](y._fused_load[width](idx))

        foreach[func](z, ctx)


# c = a @ b, should support CPU and GPU
@compiler.register("matmul_fuse_out")
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

        @parameter
        @always_inline
        def out_func[
            dtype: DType, width: SIMDSize, *, alignment: Int = 1
        ](idx: IndexList[2], val: SIMD[dtype, width]):
            c._fused_store(idx, rebind[SIMD[c.dtype, width]](val))

        print("has_epilogue_fusion =", has_epilogue_fusion)

        _matmul[elementwise_lambda_fn=out_func](c, a, b)


@compiler.register_shape_function("matmul_fuse_out")
def matmul_fuse_out_shape(
    a: InputTensor,
    b: InputTensor,
) -> IndexList[2]:
    var shape = a.shape()
    shape[1] = b.dim_size[1]()
    return rebind[IndexList[2]](shape)


# Simple, expects variadics to have the same size, and simply copies the first
# number from the associated inputs to outputs, plus a bias
@compiler.register("variadic_input_to_output")
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
@compiler.register("add_bias_to_two_tensors")
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


@compiler.register("inplace_increment_elem")
struct BasicInplace:
    @staticmethod
    def execute[
        dtype: DType,
    ](input: MutableInputTensor[dtype=dtype, rank=2, ...]):
        x = input[0, 0]
        x += 1
        input[0, 0] = x


# Have this nearly identical version as having a raise changes the Mojo function's signature
@compiler.register("inplace_increment_elem_raises")
struct BasicInplaceRaises:
    @staticmethod
    def execute[
        dtype: DType,
    ](input: MutableInputTensor[dtype=dtype, rank=2, ...]) raises:
        x = input[0, 0]
        x += 1
        input[0, 0] = x


@compiler.register("variadic_add")
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
        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[dtype, width]:
            var acc = SIMD[dtype, width](0)

            comptime for i in range(inputs.size):
                acc += inputs[i]._fused_load[width](idx)

            return acc

        foreach[func](output, ctx)


@compiler.register("transpose_2d")
@compiler.view_kernel
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


@compiler.register("print_shape_fused")
struct ElementwisePrintShape:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))

        print("input.shape =", x.shape())
        print("output.shape =", z.shape())

        foreach[func](z, ctx)


@compiler.register_shape_function("print_shape_fused")
def print_shape_fused_shape(x: InputTensor) -> IndexList[x.rank]:
    return x.shape()


# Raises if input shape is 10
@compiler.register("custom_op_that_raises")
struct CustomOpThatRaises:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: OutputTensor, x: InputTensor, ctx: DeviceContext,) raises:
        if x.shape()[0] == 10:
            raise ("input_shape[0] == 10")

        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))

        foreach[func](z, ctx)


@compiler.register_shape_function("custom_op_that_raises")
def custom_op_that_raises_shape(x: InputTensor) raises -> IndexList[x.rank]:
    print("Hello")
    var out_shape = x.shape()
    if out_shape[0] == 20:
        raise ("data.get_shape()[0] == 20")
    return out_shape


@compiler.register("mo.test.failing_constraint")
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


@compiler.register("mo.test.return_error")
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


@compiler.register("monnx.abs_v13")
struct MONNXAbsOverload:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank, ...],
        in_tensor: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[dtype, width]:
            return abs(in_tensor._fused_load[width](idx))

        print("The custom identity op is running!")
        foreach[func](out_tensor, ctx)


@compiler.register("torch.aten.abs")
struct MTorchAbsOverload:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank, ...],
        in_tensor: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[dtype, width]:
            return abs(in_tensor._fused_load[width](idx))

        print("The custom identity op is running!")
        foreach[func](out_tensor, ctx)


@compiler.register("op_with_custom_params")
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


@compiler.register("mgprt_test_func")
struct MGPRTTestFunc:
    @staticmethod
    def execute(out_tensor: OutputTensor) raises:
        external_call["MGP_RT_TEST", NoneType]()


@compiler.register("mutable_test_op")
struct MutableTestOp:
    @staticmethod
    def execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.store(0, 0)


# For testing support for Scalar[...] in Mojo
@compiler.register("supports_scalar_kernel")
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


@compiler.register("kernel_with_no_target")
struct KernelWithNoTarget:
    @staticmethod
    def execute[
        dtype: DType
    ](
        outp: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
    ) raises:
        print("hello from kernel with no target")


@compiler.register("basic_target")
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

    def __del__(deinit self):
        print("MyCustomScalarReg.__del__", self.val)


@compiler.register("buff_to_my_custom_scalar_reg")
struct BuffToMyCustomScalarReg:
    @staticmethod
    def execute[
        target: StaticString
    ](x: InputTensor[dtype=DType.int32, rank=1, ...]) -> MyCustomScalarReg[
        DType.int32
    ]:
        return MyCustomScalarReg(x[0])


@compiler.register("my_custom_scalar_reg_to_buff")
struct CustomScalarRegToBuff:
    @staticmethod
    def execute[
        target: StaticString
    ](
        input: OutputTensor[dtype=DType.int32, rank=1, ...],
        x: MyCustomScalarReg[DType.int32],
    ):
        input[0] = x.val


@compiler.register("test_custom_op")
struct TestCustomOp:
    @staticmethod
    def execute[
        target: StaticString, dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
    ):
        print("World!")


@compiler.register_shape_function("test_custom_op")
def test_custom_op_shape[
    dtype: DType, rank: Int
](input: InputTensor[dtype=dtype, rank=rank, ...]) -> IndexList[rank]:
    print("Hello")
    return input.shape()


@compiler.register("single_device_context")
struct SingleDeviceContext:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
        dev_ctx: DeviceContextList[1],
    ) raises:
        dev_ctx[0].synchronize()


@compiler.register("multi_device_context")
struct MultiDeviceContext:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
        dev_ctxs: DeviceContextList[2],
    ) raises:
        print("dev_ctx0.id() =", dev_ctxs[0].id())
        print("dev_ctx1.id() =", dev_ctxs[1].id())
        dev_ctxs[0].synchronize()
        dev_ctxs[1].synchronize()


@compiler.register("multi_device_context_dedup")
struct MultiDeviceContextDedup:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, ...],
        x: InputTensor[dtype=dtype, ...],
        y: InputTensor[dtype=dtype, ...],
        dev_ctxs: DeviceContextList[2],
    ) raises:
        dev_ctxs[0].synchronize()
        dev_ctxs[1].synchronize()


@compiler.register("variadic_device_context")
struct VariadicDeviceContext:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString = "gpu",
    ](
        outputs: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        dev_ctxs: DeviceContextList,
    ) raises:
        for i in range(len(dev_ctxs)):
            print("dev_ctxs[", i, "].id() =", dev_ctxs[i].id())
            dev_ctxs[i].synchronize()


@compiler.register("imposter_cast_elementwise")
struct CastElementwise:
    @staticmethod
    def elementwise[
        dtype: DType,
        out_dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[out_dtype, width]:
        return x.cast[out_dtype]()


@compiler.register("print_vector_size")
struct PrintVectorSize:
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
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


@compiler.register("tensor_opaque_tensor_kernel")
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


@compiler.register("test_same_rank_dtype")
struct TestSameRankDtype:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
    ):
        output[0] = input[0]


@compiler.register("test_fix_rank")
struct TestFixRank:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: InputTensor[dtype=dtype, rank=2, ...],
    ):
        output[0] = input[0]


@compiler.register("test_simd_input_size_1")
struct TestSIMDInputSize1:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: SIMD[dtype=dtype, size=1],
    ):
        output[0] = input[0]


@compiler.register("test_simd_input")
struct TestSIMDInput:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: SIMD[dtype=dtype, ...],
    ):
        output[0] = input[0]


@compiler.register("test_simd_input_size_2")
struct TestSIMDInputSize2:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: SIMD[dtype=dtype, size=2],
    ):
        output[0] = input[0]


@compiler.register("custom_op_with_custom_parameter_shape_func")
struct CustomOpWithCustomParameterShapeFunc:
    @staticmethod
    def execute[
        test_flag: Int
    ](y: FusedOutputTensor, x: FusedInputTensor) capturing raises:
        raise "This should not run!"


@compiler.register_shape_function("custom_op_with_custom_parameter_shape_func")
def custom_op_with_custom_parameter_shape_func_shape[
    test_flag: Int
](input_tensor: InputTensor,) -> IndexList[input_tensor.rank]:
    return input_tensor.shape()


@compiler.register("custom_op_with_static_string_params")
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


@compiler.register("to_tensor_wrapper")
struct ToTensorWrapper:
    @always_inline
    @staticmethod
    def execute[
        spec: StaticTensorSpec[dtype=DType.float32, rank=2, ...], //
    ](tensor: InputTensor[static_spec=spec],) -> TensorWrapper:
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


@compiler.register("print_tensor_wrapper")
struct TensorWrapperPrinter:
    @always_inline
    @staticmethod
    def execute(
        tw: TensorWrapper,
    ):
        print(tw.buffer)


@compiler.register("fused_write_to_tensor_wrapper")
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
        @parameter
        @always_inline
        def func[
            width: Int
        ](idx: IndexList[tensor.rank]) -> SIMD[DType.float32, width]:
            return rebind[SIMD[tensor.dtype, width]](
                tensor._fused_load[width](idx)
            )

        @parameter
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


@compiler.register("my_square_compute_out_fusion")
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


@compiler.register("inplace_no_out_func")
struct InplaceNoOutFunc:
    # This kernel is meant to test that epilogue fusion will
    # be skipped if the out_func overload of foreach is not used.
    @staticmethod
    def execute(
        output: MutableInputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def func[width: Int](idx: Coord) -> SIMD[output.dtype, width]:
            var val = input._fused_load[width](idx)
            return val * SIMD[output.dtype, width](2.0)

        foreach[func](output, ctx)


@fieldwise_init
struct SIMDPair[S0: Int, S1: Int](RegisterPassable):
    var x: SIMD[DType.int32, Self.S0]
    var y: SIMD[DType.int32, Self.S1]


@compiler.register("make_simd_pair")
struct MakeSimdPair:
    @staticmethod
    def execute[P0: Int, P1: Int]() -> SIMDPair[P0, P1]:
        return SIMDPair[P0, P1](
            iota[DType.int32, P0](), iota[DType.int32, P1](Int32(P0))
        )


@compiler.register("kernel_with_parameterized_opaque")
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


@compiler.register_shape_function("kernel_with_parameterized_opaque")
def kernel_with_parameterized_opaque_shape[
    P0: Int
](x: SIMDPair[P0, _]) -> IndexList[1]:
    return IndexList[1](x.S0 + x.S1)


@compiler.register("kernel_with_default_parameter")
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


@compiler.register("unique-parameters")
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


@compiler.register("shape_func_with_mutable")
struct ShapeFuncWithMutable:
    @staticmethod
    def execute(
        output: OutputTensor[rank=2, ...],
        x: MutableInputTensor[dtype=output.dtype, rank=2, ...],
    ):
        x[0, 0] = 2.0
        output[0, 0] = 1.0


@compiler.register_shape_function("shape_func_with_mutable")
def shape_func_with_mutable_shape(
    x: InputTensor[rank=2, ...],
) -> IndexList[2]:
    return x.shape()


@compiler.register("elemwise_add_with_param")
struct ElemwiseAddWithParam:
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
        bias: Int,
    ](input: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return input + Scalar[dtype](bias)


@compiler.register("elemwise_add_with_index")
struct ElemwiseAddWithIndex:
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
        rank: Int,
    ](input: SIMD[dtype, width], indices: IndexList[rank]) -> SIMD[
        dtype, width
    ]:
        var bias = SIMD[dtype, width]()
        comptime for i in range(width):
            bias[i] = Scalar[dtype](indices[0] + i)
        return input + bias


# Kernel that exposes both execute and build_graph entry points.
@compiler.register("imposter_build_graph_capable")
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


@compiler.register("add_one", type="cpu")
struct AddOneCPU:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, x: InputTensor, ctx: DeviceContext,) raises:
        print("AddOne CPU kernel")

        @parameter
        @always_inline
        def func[width: Int, alignment: Int = 1](idx: Coord) capturing:
            var val = rebind[SIMD[outp.dtype, width]](x.load[width](idx)) + 1
            outp._fused_store[element_alignment=alignment](idx, val)

        std.algorithm.functional.elementwise[func, 1, target="cpu"](
            outp.shape_coord(), ctx
        )


@compiler.register("gpu_only_kernel", type="gpu")
struct GpuOnlyKernel:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, x: InputTensor, ctx: DeviceContext,) raises:
        pass


@compiler.register("add_one", type="gpu")
struct AddOneGPU:
    @staticmethod
    def execute[
        target: StaticString,
    ](outp: OutputTensor, x: InputTensor, ctx: DeviceContext,) raises:
        print("AddOne GPU kernel")

        @parameter
        @always_inline
        def func[width: Int, alignment: Int = 1](idx: Coord) capturing:
            var val = rebind[SIMD[outp.dtype, width]](x.load[width](idx)) + 1
            outp._fused_store[element_alignment=alignment](idx, val)

        std.algorithm.functional.elementwise[func, 1, target="gpu"](
            outp.shape_coord(), ctx
        )
