# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from collections import Optional
from collections.string import StaticString
from math import iota
from sys import external_call
from sys.info import simd_width_of

import compiler_internal as compiler
from buffer import NDBuffer
from buffer.dimlist import DimList
from compiler_internal import StaticTensorSpec
from runtime.asyncrt import DeviceContextPtr, DeviceContextPtrList
from tensor import (
    InputTensor,
    InputVariadicTensors,
    ManagedTensorSlice,
    OutputTensor,
    OutputVariadicTensors,
    _FusedComputeOutputTensor,
    _input_fusion_hook_impl,
    _output_fusion_hook_impl,
    foreach,
    simd_load_from_managed_tensor_slice,
    simd_store_into_managed_tensor_slice,
    view_copy_impl,
)
from tensor.managed_tensor_slice import (
    _FusedInputTensor as FusedInputTensor,
)
from tensor.managed_tensor_slice import (
    _FusedInputVariadicTensors as FusedInputVariadicTensors,
)
from tensor.managed_tensor_slice import (
    _FusedOutputTensor as FusedOutputTensor,
)
from tensor.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)

from utils import IndexList, StaticTuple


# TODO(MOCO-1413): remove this need to keep imported exported funcs alive.
@export
fn export():
    comptime _simd_load_from_managed_tensor_slice = simd_load_from_managed_tensor_slice
    comptime _simd_store_into_managed_tensor_slice = simd_store_into_managed_tensor_slice
    comptime __input_fusion_hook_impl = _input_fusion_hook_impl
    comptime __output_fusion_hook_impl = _output_fusion_hook_impl


# ===-----------------------------------------------------------------------===#
# Opaque Reg Types
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct MyCustomScalarRegTrivialSI32(ImplicitlyCopyable):
    var val: Int32

    @implicit
    fn __init__(out self, val: Int32):
        print("MyCustomScalarRegTrivialSI32.__init__", val)
        self.val = val


@compiler.register("tensor_to_custom_scalar_si32_reg_trivial")
struct OpaqueToCustomScalarSI32RegTrivial:
    @staticmethod
    fn execute(
        x: InputTensor[dtype = DType.int32, rank=1]
    ) -> MyCustomScalarRegTrivialSI32:
        return MyCustomScalarRegTrivialSI32(x[0])


@register_passable
struct MyCustomScalarRegSI32(ImplicitlyCopyable):
    var val: Int32

    @implicit
    fn __init__(out self, val: Int32):
        print("MyCustomScalarRegSI32.__init__", val)
        self.val = val

    fn __del__(deinit self):
        print("MyCustomScalarRegSI32.__del__", self.val)


# It is intentional there are no methods which consume this.
# It is here to support some level of type checking.
@register_passable
struct MyCustomScalarRegF32(ImplicitlyCopyable):
    var val: Float32

    @implicit
    fn __init__(out self, val: Float32):
        print("MyCustomScalarRegF32.__init__", val)
        self.val = val

    fn __del__(deinit self):
        print("MyCustomScalarRegF32.__del__", self.val)


@compiler.register("tensor_to_custom_scalar_si32_reg")
struct OpaqueToCustomScalarSI32Reg:
    @staticmethod
    fn execute(
        x: InputTensor[dtype = DType.int32, rank=1]
    ) -> MyCustomScalarRegSI32:
        return MyCustomScalarRegSI32(x[0])


# Adds two custom scalar types (one of which is register passable) and writes
# to a tensor
@compiler.register("opaque_add_to_tensor_si32_reg")
struct OpaqueAddToTensorSI32Reg:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype = DType.int32, rank=1],
        x: MyCustomScalarRegSI32,
        y: MyCustomScalarRegSI32,
    ):
        outp[0] = x.val + y.val


@compiler.register("opaque_add_to_tensor_f32_reg")
struct OpaqueAddToTensorF32:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype = DType.float32, rank=1],
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
    fn __init__(out self, val: Int32):
        print("MyCustomScalarSI32.__init__", val)
        self.val = val

    fn __del__(deinit self):
        print("MyCustomScalarSI32.__del__", self.val)


@compiler.register("tensor_to_custom_scalar_si32")
struct OpaqueToCustomScalarSI32:
    @staticmethod
    fn execute(
        x: InputTensor[dtype = DType.int32, rank=1]
    ) -> MyCustomScalarSI32:
        return MyCustomScalarSI32(x[0])


# Adds two custom scalar types (one of which is register passable) and writes
# to a tensor
@compiler.register("opaque_add_to_tensor_si32")
struct OpaqueAddToTensorSI32:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype = DType.int32, rank=1],
        x: MyCustomScalarSI32,
        y: MyCustomScalarSI32,
    ):
        outp[0] = x.val + y.val


@compiler.register("opaque_add_to_tensor_si32_raises")
struct OpaqueAddToTensorSI32Raises:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype = DType.int32, rank=1],
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
    fn execute[
        target: StaticString,
    ](z: OutputTensor, x: InputTensor, y: InputTensor) raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](x.load[width](idx)) + rebind[
                SIMD[z.dtype, width]
            ](y.load[width](idx))

        foreach[func](z)

    @staticmethod
    fn shape(x: InputTensor, y: InputTensor) -> IndexList[x.rank]:
        return x.shape()


@always_inline
fn toNDBuffer[
    out_dtype: DType, out_rank: Int
](tensor: ManagedTensorSlice) -> NDBuffer[out_dtype, out_rank, MutAnyOrigin]:
    # TODO(GEX-734): forward other static params automatically
    return rebind[NDBuffer[out_dtype, out_rank, MutAnyOrigin]](
        NDBuffer[tensor.dtype, tensor.rank](tensor._ptr, tensor.shape())
    )


fn _matmul[
    elementwise_lambda_fn: Optional[
        fn[
            dtype: DType, width: Int, *, alignment: Int = 1
        ] (IndexList[2], SIMD[dtype, width]) capturing -> None
    ] = None,
](
    c: ManagedTensorSlice[mut=True],
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

            @parameter
            if elementwise_lambda_fn:
                comptime func = elementwise_lambda_fn.value()
                func(IndexList[2](i, j), c_val)
            else:
                c.store[1](IndexList[2](i, j), c_val)


# c = a @ b, should support CPU and GPU
@compiler.register("imposter_matmul")
struct ImposterMatmul:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        c: OutputTensor,
        a: InputTensor,
        b: InputTensor,
        ctx: DeviceContextPtr,
    ) raises:
        _matmul(c, a, b)

    @staticmethod
    fn shape(
        a: InputTensor,
        b: InputTensor,
    ) -> IndexList[2]:
        var shape = a.shape()
        shape[1] = b.dim_size[1]()
        return rebind[IndexList[2]](shape)


@compiler.register("print_tensor_spec")
struct PrintTensorSpecOp:
    @staticmethod
    fn execute[
        target: StaticString,
    ](outp: OutputTensor, x: InputTensor) raises:
        print("x.shape =", x._static_shape)
        print("x.strides =", x._static_strides)
        print("x.alignment =", x.alignment)
        print("x.address_space =", x.address_space)
        print("x.exclusive =", x.exclusive)

        @parameter
        @always_inline
        fn func[
            width: Int
        ](idx: IndexList[outp.rank]) -> SIMD[outp.dtype, width]:
            return rebind[SIMD[outp.dtype, width]](x.load[width](idx))

        foreach[func](outp)

    @staticmethod
    fn shape(x: InputTensor) -> IndexList[x.rank]:
        return x.shape()


@compiler.register("print_tensor")
struct PrintTensor:
    @staticmethod
    fn execute[
        target: StaticString,
    ](outp: OutputTensor, input: InputTensor) raises:
        print(input)

        @parameter
        @always_inline
        fn func[
            width: Int
        ](idx: IndexList[outp.rank]) -> SIMD[outp.dtype, width]:
            return rebind[SIMD[outp.dtype, width]](input.load[width](idx))

        foreach[func](outp)


@compiler.register("print_tensor_spec_view")
@compiler.view_kernel
struct PrintTensorSpecViewOp:
    @staticmethod
    fn print_spec(x: InputTensor):
        print("x.shape =", x._static_shape)
        print("x.strides =", x._static_strides)
        print("x.alignment =", x.alignment)
        print("x.address_space =", x.address_space)
        print("x.exclusive =", x.exclusive)

    @staticmethod
    fn update_input_view(input: InputTensor) -> type_of(input):
        Self.print_spec(input)
        return input

    @staticmethod
    fn execute[
        target: StaticString,
    ](outp: OutputTensor, x: InputTensor) raises:
        Self.print_spec(x)

        @parameter
        @always_inline
        fn func[
            width: Int
        ](idx: IndexList[outp.rank]) -> SIMD[outp.dtype, width]:
            return rebind[SIMD[outp.dtype, width]](x.load[width](idx))

        foreach[func](outp)

    @staticmethod
    fn shape(x: InputTensor) -> IndexList[x.rank]:
        return x.shape()


@compiler.register("print_tensor_spec_fused")
struct PrintTensorSpecFusedOp:
    @staticmethod
    fn execute[
        target: StaticString,
    ](outp: FusedOutputTensor, x: FusedInputTensor) capturing raises:
        print("x.shape =", x._static_shape)
        print("x.strides =", x._static_strides)
        print("x.alignment =", x.alignment)
        print("x.address_space =", x.address_space)
        print("x.exclusive =", x.exclusive)

        @parameter
        @always_inline
        fn func[
            width: Int
        ](idx: IndexList[outp.rank]) -> SIMD[outp.dtype, width]:
            return rebind[SIMD[outp.dtype, width]](x._fused_load[width](idx))

        foreach[func](outp)

    @staticmethod
    fn shape(x: InputTensor) -> IndexList[x.rank]:
        return x.shape()


@compiler.register("print_output_shape_strides_fused")
struct PrintOutputShapeStridesFusedOp:
    @staticmethod
    fn execute[
        target: StaticString,
    ](output: FusedOutputTensor, x: FusedInputTensor) capturing raises:
        print("output._static_shape = ", output._static_shape)
        print("output._static_strides = ", output._static_strides)
        print("output._runtime_shape = ", output._spec.shape)
        print("output._runtime_strides = ", output._runtime_strides)

        @parameter
        @always_inline
        fn func[
            width: Int
        ](idx: IndexList[output.rank]) -> SIMD[output.dtype, width]:
            return (
                rebind[SIMD[output.dtype, width]](x._fused_load[width](idx))
                * 2.0
            )

        foreach[func](output)

    @staticmethod
    fn shape(x: InputTensor) -> IndexList[x.rank]:
        return x.shape()


@compiler.register("imposter_add_elementwise")
struct AddElementwise:
    @staticmethod
    fn elementwise[
        dtype: DType,
        width: Int,
    ](
        lhs: SIMD[dtype, width], rhs: SIMD[dtype, width], _idx: IndexList
    ) -> SIMD[dtype, width]:
        return lhs + rhs


@compiler.register("imposter_add_lhs")
struct AddFuseLHS:
    @staticmethod
    fn execute[
        target: StaticString,
    ](z: OutputTensor, x: FusedInputTensor, y: InputTensor) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](
                x._fused_load[width](idx)
            ) + rebind[SIMD[z.dtype, width]](y.load[width](idx))

        foreach[func](z)


@compiler.register("imposter_add_fuse_inputs")
struct AddFuseInputs:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        z: OutputTensor, x: FusedInputTensor, y: FusedInputTensor
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](
                x._fused_load[width](idx)
            ) + rebind[SIMD[z.dtype, width]](y._fused_load[width](idx))

        foreach[func](z)


# c = a @ b, should support CPU and GPU
@compiler.register("matmul_fuse_out")
struct MatmulFuseOut:
    @staticmethod
    fn execute[
        target: StaticString,
        lambdas_have_fusion: Bool,
    ](
        c: FusedOutputTensor,
        a: InputTensor,
        b: InputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        comptime rank = a.rank
        comptime a_dtype = a.dtype
        comptime b_dtype = b.dtype
        comptime c_dtype = c.dtype

        # Convert everything to NDBuffer
        # var c_buffer = toNDBuffer[c_dtype, 2](c)
        # var a_buffer = toNDBuffer[a_dtype, 2](a)
        # var b_buffer = toNDBuffer[b_dtype, 2](b)

        @parameter
        @always_inline
        fn out_func[
            dtype: DType, width: Int, *, alignment: Int = 1
        ](idx: IndexList[2], val: SIMD[dtype, width]):
            c._fused_store(idx, rebind[SIMD[c.dtype, width]](val))

        print("lambdas_have_fusion =", lambdas_have_fusion)

        _matmul[elementwise_lambda_fn=out_func](c, a, b)

    @staticmethod
    fn shape(
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
    fn execute[
        dtype: DType,
        size: Int,
        target: StaticString,
    ](
        output: OutputVariadicTensors[dtype, rank=1, size=size],
        bias: InputTensor[dtype=dtype, rank=1],
        input: InputVariadicTensors[dtype, rank=1, size=size],
    ):
        @parameter
        for i in range(size):
            for j in range(input[i].size()):
                output[i][j] = input[i][j]
            output[i][0] += bias[0]


# Simply adds the first number of bias to the first number of boath outputs
# Mainly here to test logic with multiple DPS outputs
@compiler.register("add_bias_to_two_tensors")
struct AddBiasToDouble:
    @staticmethod
    fn execute[
        rank: Int,
        dtype: DType,
    ](
        output1: OutputTensor[dtype=dtype, rank=rank],
        output2: OutputTensor[dtype=dtype, rank=rank],
        input1: InputTensor[dtype=dtype, rank=rank],
        input2: InputTensor[dtype=dtype, rank=rank],
        bias: InputTensor[dtype=dtype, rank=rank],
    ):
        output1[0] = input1[0] + bias[0]
        output2[0] = input2[0] + bias[0]


@compiler.register("inplace_increment_elem")
struct BasicInplace:
    @staticmethod
    fn execute[
        dtype: DType,
    ](input: MutableInputTensor[dtype=dtype, rank=2]):
        x = input[0, 0]
        x += 1
        input[0, 0] = x


# Have this nearly identical version as having a raise changes the Mojo function's signature
@compiler.register("inplace_increment_elem_raises")
struct BasicInplaceRaises:
    @staticmethod
    fn execute[
        dtype: DType,
    ](input: MutableInputTensor[dtype=dtype, rank=2]) raises:
        x = input[0, 0]
        x += 1
        input[0, 0] = x


@compiler.register("variadic_add")
struct VariadicAdd:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        inputs: FusedInputVariadicTensors[dtype, rank, *_],
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[rank]) -> SIMD[dtype, width]:
            var acc = SIMD[dtype, width](0)

            @parameter
            for i in range(inputs.size):
                acc += inputs[i]._fused_load[width](idx)

            return acc

        foreach[func](output)


@compiler.register("transpose_2d")
@compiler.view_kernel
struct Transpose2DOp:
    @staticmethod
    fn build_view[
        dtype: DType,
    ](x: InputTensor[dtype=dtype, rank=2]) -> StaticTuple[IndexList[2], 2]:
        var new_stride = IndexList[2]()
        var new_shape = IndexList[2]()
        new_stride[0] = x._runtime_strides[1]
        new_stride[1] = x._runtime_strides[0]
        new_shape[0] = x._spec.shape[1]
        new_shape[1] = x._spec.shape[0]

        return StaticTuple[IndexList[2], 2](new_shape, new_stride)

    @staticmethod
    fn get_view_strides(input_strides: DimList) -> DimList:
        # transpose the strides of the input
        return DimList(input_strides.at[1](), input_strides.at[0]())

    @staticmethod
    fn update_input_view[
        dtype: DType, //, output_static_shape: DimList
    ](
        input: InputTensor[dtype=dtype, rank=2],
        out result: InputTensor[
            static_spec = input.static_spec.with_layout[2](
                output_static_shape,
                Self.get_view_strides(input._static_strides),
            )
        ],
    ):
        var shape_and_strides = Self.build_view(input)

        return type_of(result)(
            input.unsafe_ptr(), shape_and_strides[0], shape_and_strides[1]
        )

    @staticmethod
    fn execute[
        target: StaticString,
        dtype: DType,
    ](
        z: OutputTensor[dtype=dtype, rank=2],
        x: InputTensor[dtype=dtype, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        var x_view = Self.update_input_view[z._static_shape](x)

        view_copy_impl[target=target](z, x_view, ctx)


@compiler.register("print_shape_fused")
struct ElementwisePrintShape:
    @staticmethod
    fn execute[
        target: StaticString,
    ](z: FusedOutputTensor, x: FusedInputTensor) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))

        print("input.shape =", x._spec.shape)
        print("output.shape =", z._spec.shape)

        foreach[func](z)

    @staticmethod
    fn shape(x: InputTensor) -> IndexList[x.rank]:
        return x.shape()


# Raises if input shape is 10
@compiler.register("custom_op_that_raises")
struct CustomOpThatRaises:
    @staticmethod
    fn execute[
        target: StaticString,
    ](z: OutputTensor, x: InputTensor) raises:
        if x.shape()[0] == 10:
            raise ("input_shape[0] == 10")

        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            return rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))

        foreach[func](z)

    @staticmethod
    fn shape(x: InputTensor) raises -> IndexList[x.rank]:
        print("Hello")
        var out_shape = x.shape()
        if out_shape[0] == 20:
            raise ("data.get_shape()[0] == 20")
        return out_shape


@compiler.register("mo.test.failing_constraint")
struct OpThatAlwaysFailsConstraint:
    @staticmethod
    fn execute[
        dtype: DType, rank: Int
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank],
        in_tensor: InputTensor[dtype=dtype, rank=rank],
    ):
        constrained[
            1 == 2,
            "Expected constraint failure for error message testing",
        ]()


@compiler.register("mo.test.return_error")
struct OpThatAlwaysRaises:
    @staticmethod
    fn execute[
        dtype: DType, rank: Int
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank],
        in_tensor: InputTensor[dtype=dtype, rank=rank],
    ) raises:
        out_tensor[0] = in_tensor[0]
        raise Error("This is an error")


@compiler.register("monnx.abs_v13")
struct MONNXAbsOverload:
    @staticmethod
    fn execute[
        dtype: DType, rank: Int
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank],
        in_tensor: InputTensor[dtype=dtype, rank=rank],
    ) raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[rank]) -> SIMD[dtype, width]:
            return abs(in_tensor._fused_load[width](idx))

        print("The custom identity op is running!")
        foreach[func](out_tensor)


@compiler.register("torch.aten.abs")
struct MTorchAbsOverload:
    @staticmethod
    fn execute[
        dtype: DType, rank: Int
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank],
        in_tensor: InputTensor[dtype=dtype, rank=rank],
    ) raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[rank]) -> SIMD[dtype, width]:
            return abs(in_tensor._fused_load[width](idx))

        print("The custom identity op is running!")
        foreach[func](out_tensor)


@compiler.register("op_with_custom_params")
struct OpWithCustomParams:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        custom_int: Int,
        custom_str: StaticString,
        custom_dtype: DType,
    ](
        out_tensor: OutputTensor[dtype=dtype, rank=rank],
        in_tensor: InputTensor[dtype=dtype, rank=rank],
    ) raises:
        out_tensor[0] = in_tensor[0]
        print("custom_int =", custom_int)
        print("custom_str =", custom_str)
        print("custom_dtype =", custom_dtype)


@compiler.register("mgprt_test_func")
struct MGPRTTestFunc:
    @staticmethod
    fn execute(out_tensor: OutputTensor) raises:
        external_call["MGP_RT_TEST", NoneType]()


@compiler.register("mutable_test_op")
struct MutableTestOp:
    @staticmethod
    fn execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.store(0, 0)


# For testing support for Scalar[...] in Mojo
@compiler.register("supports_scalar_kernel")
struct SupportsScalarKernel:
    @staticmethod
    fn execute[
        dtype: DType
    ](
        outp: OutputTensor[dtype=dtype, rank=1],
        x: InputTensor[dtype=dtype, rank=1],
        y: Scalar[dtype],
    ) raises:
        print("datatype is", dtype)
        print("value is", y)


@compiler.register("kernel_with_no_target")
struct KernelWithNoTarget:
    @staticmethod
    fn execute[
        dtype: DType
    ](
        outp: OutputTensor[dtype=dtype, *_],
        x: InputTensor[dtype=dtype, *_],
    ) raises:
        print("hello from kernel with no target")


@compiler.register("basic_target")
struct BasicTarget:
    @staticmethod
    fn execute[
        dtype: DType, target: StaticString
    ](
        outp: OutputTensor[dtype=dtype, *_],
        x: InputTensor[dtype=dtype, *_],
    ) raises:
        print("hello from kernel on", target)


@register_passable
struct MyCustomScalarReg[dtype: DType](ImplicitlyCopyable):
    var val: Scalar[Self.dtype]

    @implicit
    fn __init__(out self, val: Scalar[Self.dtype]):
        print("MyCustomScalarReg.__init__", val)
        self.val = val

    fn __del__(deinit self):
        print("MyCustomScalarReg.__del__", self.val)


@compiler.register("buff_to_my_custom_scalar_reg")
struct BuffToMyCustomScalarReg:
    @staticmethod
    fn execute[
        target: StaticString
    ](x: InputTensor[dtype = DType.int32, rank=1]) -> MyCustomScalarReg[
        DType.int32
    ]:
        return MyCustomScalarReg(x[0])


@compiler.register("my_custom_scalar_reg_to_buff")
struct CustomScalarRegToBuff:
    @staticmethod
    fn execute[
        target: StaticString
    ](
        input: OutputTensor[dtype = DType.int32, rank=1],
        x: MyCustomScalarReg[DType.int32],
    ):
        input[0] = x.val


@compiler.register("test_custom_op")
struct TestCustomOp:
    @staticmethod
    fn execute[
        target: StaticString, dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
    ):
        print("World!")

    @staticmethod
    fn shape[
        dtype: DType, rank: Int
    ](input: InputTensor[dtype=dtype, rank=rank]) -> IndexList[rank]:
        print("Hello")
        return input.shape()


@compiler.register("single_device_context")
struct SingleDeviceContext:
    @staticmethod
    fn execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, *_],
        x: InputTensor[dtype=dtype, *_],
        dev_ctx: DeviceContextPtrList[1],
    ) raises:
        dev_ctx[0].synchronize()


@compiler.register("multi_device_context")
struct MultiDeviceContext:
    @staticmethod
    fn execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, *_],
        x: InputTensor[dtype=dtype, *_],
        dev_ctxs: DeviceContextPtrList[2],
    ) raises:
        print("dev_ctx0.id() =", dev_ctxs[0].id())
        print("dev_ctx1.id() =", dev_ctxs[1].id())
        dev_ctxs[0].synchronize()
        dev_ctxs[1].synchronize()


@compiler.register("multi_device_context_dedup")
struct MultiDeviceContextDedup:
    @staticmethod
    fn execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, *_],
        x: InputTensor[dtype=dtype, *_],
        y: InputTensor[dtype=dtype, *_],
        dev_ctxs: DeviceContextPtrList[2],
    ) raises:
        dev_ctxs[0].synchronize()
        dev_ctxs[1].synchronize()


@compiler.register("variadic_device_context")
struct VariadicDeviceContext:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString = "gpu",
    ](
        outputs: OutputVariadicTensors[dtype, rank, *_],
        inputs: InputVariadicTensors[dtype, rank, *_],
        dev_ctxs: DeviceContextPtrList,
    ) raises:
        for i in range(len(dev_ctxs)):
            print("dev_ctxs[", i, "].id() =", dev_ctxs[i].id())
            dev_ctxs[i].synchronize()


@compiler.register("imposter_cast_elementwise")
struct CastElementwise:
    @staticmethod
    fn elementwise[
        dtype: DType,
        out_dtype: DType,
        width: Int,
    ](x: SIMD[dtype, width], _idx: IndexList) -> SIMD[out_dtype, width]:
        return x.cast[out_dtype]()


@compiler.register("print_vector_size")
struct PrintVectorSize:
    @staticmethod
    fn elementwise[
        dtype: DType,
        width: Int,
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
    fn execute[
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
    fn execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
    ):
        output[0] = input[0]


@compiler.register("test_fix_rank")
struct TestFixRank:
    @staticmethod
    fn execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2],
        input: InputTensor[dtype=dtype, rank=2],
    ):
        output[0] = input[0]


@compiler.register("test_simd_input_size_1")
struct TestSIMDInputSize1:
    @staticmethod
    fn execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2],
        input: SIMD[dtype=dtype, size=1],
    ):
        output[0] = input[0]


@compiler.register("test_simd_input")
struct TestSIMDInput:
    @staticmethod
    fn execute[
        dtype: DType
    ](output: OutputTensor[dtype=dtype, rank=2], input: SIMD[dtype=dtype],):
        output[0] = input[0]


@compiler.register("test_simd_input_size_2")
struct TestSIMDInputSize2:
    @staticmethod
    fn execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=dtype, rank=2],
        input: SIMD[dtype=dtype, size=2],
    ):
        output[0] = input[0]


@compiler.register("custom_op_with_custom_parameter_shape_func")
struct CustomOpWithCustomParameterShapeFunc:
    @staticmethod
    fn execute[
        test_flag: Int
    ](y: FusedOutputTensor, x: FusedInputTensor) capturing raises:
        raise "This should not run!"

    @always_inline
    @staticmethod
    fn shape[
        test_flag: Int
    ](input_tensor: InputTensor,) -> IndexList[input_tensor.rank]:
        return input_tensor.shape()


@compiler.register("custom_op_with_static_string_params")
struct CustomOpWithStaticStringParams:
    @staticmethod
    fn execute[target: StaticString, foo: StaticString]():
        print(target)
        print(foo)


@fieldwise_init
@register_passable
struct TensorWrapper(ImplicitlyCopyable):
    var buffer: NDBuffer[DType.float32, 2, MutAnyOrigin]


@compiler.register("to_tensor_wrapper")
struct ToTensorWrapper:
    @always_inline
    @staticmethod
    fn execute[
        spec: StaticTensorSpec[dtype = DType.float32, rank=2], //
    ](tensor: InputTensor[static_spec=spec],) -> TensorWrapper:
        var ptr = tensor._ptr.address_space_cast[spec.address_space]()
        return TensorWrapper(
            NDBuffer[
                spec.dtype,
                spec.rank,
                _,
                spec.shape,
                spec.strides,
                alignment2 = spec.alignment,
                address_space = spec.address_space,
                exclusive = spec.exclusive,
            ](ptr, tensor.shape(), tensor._runtime_strides)
        )


@compiler.register("print_tensor_wrapper")
struct TensorWrapperPrinter:
    @always_inline
    @staticmethod
    fn execute(
        tw: TensorWrapper,
    ):
        print(tw.buffer)


@compiler.register("fused_write_to_tensor_wrapper")
struct MutableStore:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        tw: TensorWrapper,
        tensor: FusedInputTensor[dtype = DType.float32, rank=2],
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[
            width: Int
        ](idx: IndexList[tensor.rank]) -> SIMD[DType.float32, width]:
            return rebind[SIMD[tensor.dtype, width]](
                tensor._fused_load[width](idx)
            )

        @parameter
        @always_inline
        fn out_func[width: Int](index: IndexList[tensor.rank]) capturing:
            var idx = rebind[IndexList[tensor.rank]](index)
            var val = func[width](idx)
            idx[0] += 1
            idx[1] += 1
            tw.buffer.store[width=width](idx, val)

        foreach[
            func,
            out_func,
            target=target,
            _trace_name=_trace_name,
        ](tensor, ctx)


@compiler.register("my_square_compute_out_fusion")
struct MySquareComputeOutFusion:
    @staticmethod
    fn execute[
        dtype: DType
    ](
        outp: _FusedComputeOutputTensor[dtype=dtype, rank=1],
        input: InputTensor[dtype=dtype, rank=1],
    ) capturing:
        var res = input[0] * input[0]
        comptime has_compute_lambda = outp.static_spec.out_compute_lambda is not None
        comptime has_store_lambda = outp.static_spec.out_lambda is not None
        print("has_compute_lambda =", has_compute_lambda)
        print("has_store_lambda =", has_store_lambda)

        @parameter
        if has_compute_lambda:
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
    fn execute(
        output: MutableInputTensor,
        input: FusedInputTensor[dtype = output.dtype, rank = output.rank],
    ) capturing raises:
        @parameter
        @always_inline
        fn func[
            width: Int
        ](idx: IndexList[output.rank]) -> SIMD[output.dtype, width]:
            var val = input._fused_load[width](idx)
            return val * SIMD[output.dtype, width](2.0)

        foreach[func](output)


@fieldwise_init
@register_passable
struct SIMDPair[S0: Int, S1: Int](ImplicitlyCopyable):
    var x: SIMD[DType.int32, Self.S0]
    var y: SIMD[DType.int32, Self.S1]


@compiler.register("make_simd_pair")
struct MakeSimdPair:
    @staticmethod
    fn execute[P0: Int, P1: Int]() -> SIMDPair[P0, P1]:
        return SIMDPair[P0, P1](
            iota[DType.int32, P0](), iota[DType.int32, P1](P0)
        )


@compiler.register("kernel_with_parameterized_opaque")
struct ParameterizedOpaqueType:
    @staticmethod
    fn execute[
        P0: Int
    ](
        output: OutputTensor[dtype = DType.int32, rank=1], x: SIMDPair[P0, _]
    ) capturing:
        output.store(IndexList[1](0), x.x)
        output.store(IndexList[1](P0), x.y)

    @staticmethod
    fn shape[P0: Int](x: SIMDPair[P0, _]) -> IndexList[1]:
        return IndexList[1](x.S0 + x.S1)


@compiler.register("kernel_with_default_parameter")
struct KernelWithDefaultParameter:
    @staticmethod
    fn execute[
        rank: Int,
        rank_defaulted: Int = 128,
        rank_derived_from_other: Int = rank + 1,
        rank_derived_from_derived: Int = rank_derived_from_other + 1,
    ](x: OutputTensor[dtype = DType.int32, rank=1], y: InputTensor):
        print("rank =", rank)
        print("rank_defaulted =", rank_defaulted)
        print("rank_derived_from_other =", rank_derived_from_other)
        print("rank_derived_from_derived =", rank_derived_from_derived)
        x.store(IndexList[1](0), Int32(0))


@compiler.register("unique-parameters")
struct UniqueParameters:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int,
        target: StaticString,
    ](
        blocks: InputTensor[dtype=dtype, rank=6],
        cache_lengths: InputTensor[dtype = DType.uint32, rank=1],
        lookup_table: InputTensor[dtype = DType.uint32, rank=2],
        max_lengths: InputTensor[dtype = DType.uint32, rank=2],
    ) -> MyCustomScalarRegTrivialSI32:
        @parameter
        if page_size:
            return MyCustomScalarRegTrivialSI32(0)
        else:
            return MyCustomScalarRegTrivialSI32(1)
