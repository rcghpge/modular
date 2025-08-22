# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections import OptionalReg

from buffer.dimlist import Dim, DimList
from gpu.host import DeviceContext
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    random,
)
from linalg.grouped_matmul import grouped_matmul, naive_grouped_matmul
from linalg.utils import elementwise_epilogue_type
from linalg.utils_gpu import MatmulConfig
from testing import assert_almost_equal

from utils import IndexList
from utils.index import Index


@always_inline
fn test_epilogue[
    dtype: DType
](m: Int, n: Int, val: Scalar[dtype]) -> Scalar[dtype]:
    return val + 4 * (Scalar[dtype]((m + n) % 21 - 10))


fn test[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
    has_epilogue: Bool = False,
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
) raises:
    print(
        num_active_experts,
        "active of",
        num_experts,
        "experts of shape",
        expert_shape,
    )
    print("tokens:", end="")
    for i in range(len(num_tokens_by_expert)):
        print(num_tokens_by_expert[i], end=" ")
    print("expert ids:", end="")
    for i in range(len(expert_ids)):
        print(expert_ids[i], end=" ")
    print()

    alias a_type = in_type
    alias b_type = in_type
    alias c_type = out_type

    alias N = expert_shape[0]
    alias K = expert_shape[1]

    # Total and max number of tokens
    total_num_tokens = 0
    max_num_tokens_by_expert = 0
    for i in range(len(num_tokens_by_expert)):
        total_num_tokens += num_tokens_by_expert[i]
        max_num_tokens_by_expert = max(
            max_num_tokens_by_expert, num_tokens_by_expert[i]
        )

    # Create host A C buffers
    alias static_a_shape = DimList(Dim(), K)
    var dynamic_a_shape = DimList(total_num_tokens, K)
    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    alias static_c_shape = DimList(Dim(), N)
    var dynamic_c_shape = DimList(total_num_tokens, N)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var c_ref_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var a_offsets_host = HostNDBuffer[DType.uint32, 1, DimList(Dim())](
        num_active_experts + 1
    )

    # Create host B buffers
    alias static_b_shape = DimList(num_experts, N, K)
    var b_host = HostNDBuffer[b_type, 3, static_b_shape](static_b_shape)
    var expert_ids_host = HostNDBuffer[DType.int32, 1](num_active_experts)

    # Setup  offsets and expert ids
    a_offsets_host.tensor[0] = 0
    for i in range(num_active_experts):
        a_offsets_host.tensor[i + 1] = (
            a_offsets_host.tensor[i] + num_tokens_by_expert[i]
        )
        expert_ids_host.tensor[i] = expert_ids[i]

    # Initialize matmul inputs
    random(a_host.tensor)
    random(b_host.tensor)

    # Create device buffers
    var a_dev = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var c_dev = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_ref_dev = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var b_dev = DeviceNDBuffer[b_type, 3, static_b_shape](
        static_b_shape, ctx=ctx
    )
    var a_offsets_dev = DeviceNDBuffer[DType.uint32, 1](
        num_active_experts + 1, ctx=ctx
    )
    var expert_ids_dev = DeviceNDBuffer[DType.int32, 1](
        num_active_experts, ctx=ctx
    )

    # Move inputs to device
    ctx.enqueue_copy(a_dev.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_dev.buffer, b_host.tensor.data)
    ctx.enqueue_copy(a_offsets_dev.buffer, a_offsets_host.tensor.data)
    ctx.enqueue_copy(expert_ids_dev.buffer, expert_ids_host.tensor.data)

    naive_grouped_matmul(
        c_ref_dev.tensor,
        a_dev.tensor,
        b_dev.tensor,
        a_offsets_dev.tensor,
        expert_ids_dev.tensor,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    var c_dev_ndbuffer = c_dev.tensor

    @always_inline
    @__copy_capture(c_dev_ndbuffer)
    @parameter
    fn epilogue_fn[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        var new_val = val

        @parameter
        for i in range(width):
            new_val[i] = test_epilogue(idx[0], idx[1] + i, val[i])

        c_dev_ndbuffer.store[width=width, alignment=alignment](
            idx, new_val.cast[out_type]()
        )

    grouped_matmul[
        elementwise_lambda_fn = OptionalReg[elementwise_epilogue_type](
            epilogue_fn
        ) if has_epilogue else None,
    ](
        c_dev.tensor,
        a_dev.tensor,
        b_dev.tensor,
        a_offsets_dev.tensor,
        expert_ids_dev.tensor,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    ctx.enqueue_copy(c_ref_host.tensor.data, c_ref_dev.buffer)
    ctx.enqueue_copy(c_host.tensor.data, c_dev.buffer)
    ctx.synchronize()

    rtol = 1e-2
    c_ref_host_buffer = c_ref_host.tensor
    c_host_buffer = c_host.tensor
    for m in range(total_num_tokens):
        for n in range(N):
            var expect: Scalar[out_type]

            @parameter
            if has_epilogue:
                expect = test_epilogue(m, n, c_ref_host_buffer[m, n])
            else:
                expect = c_ref_host_buffer[m, n]

            var actual = c_host_buffer[m, n]
            assert_almost_equal(
                actual, expect, msg=String("m: ", m, " n: ", n), rtol=rtol
            )


fn test_negative_lora_id[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
) raises:
    print(
        "Testing negative lora_id behavior:",
        num_active_experts,
        "active of",
        num_experts,
        "experts of shape",
        expert_shape,
    )
    print("tokens:", end="")
    for i in range(len(num_tokens_by_expert)):
        print(num_tokens_by_expert[i], end=" ")
    print("expert ids:", end="")
    for i in range(len(expert_ids)):
        print(expert_ids[i], end=" ")
    print()

    alias a_type = in_type
    alias b_type = in_type
    alias c_type = out_type

    alias N = expert_shape[0]
    alias K = expert_shape[1]

    # Total and max number of tokens
    total_num_tokens = 0
    max_num_tokens_by_expert = 0
    for i in range(len(num_tokens_by_expert)):
        total_num_tokens += num_tokens_by_expert[i]
        max_num_tokens_by_expert = max(
            max_num_tokens_by_expert, num_tokens_by_expert[i]
        )

    # Create host A C buffers
    alias static_a_shape = DimList(Dim(), K)
    var dynamic_a_shape = DimList(total_num_tokens, K)
    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    alias static_c_shape = DimList(Dim(), N)
    var dynamic_c_shape = DimList(total_num_tokens, N)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var a_offsets_host = HostNDBuffer[DType.uint32, 1, DimList(Dim())](
        num_active_experts + 1
    )

    # Create host B buffers
    alias static_b_shape = DimList(num_experts, N, K)
    var b_host = HostNDBuffer[b_type, 3, static_b_shape](static_b_shape)
    var expert_ids_host = HostNDBuffer[DType.int32, 1](num_active_experts)

    # Setup offsets and expert ids
    a_offsets_host.tensor[0] = 0
    for i in range(num_active_experts):
        a_offsets_host.tensor[i + 1] = (
            a_offsets_host.tensor[i] + num_tokens_by_expert[i]
        )
        expert_ids_host.tensor[i] = expert_ids[i]

    # Initialize matmul inputs
    random(a_host.tensor)
    random(b_host.tensor)

    # Create device buffers
    var a_dev = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var c_dev = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var b_dev = DeviceNDBuffer[b_type, 3, static_b_shape](
        static_b_shape, ctx=ctx
    )
    var a_offsets_dev = DeviceNDBuffer[DType.uint32, 1](
        num_active_experts + 1, ctx=ctx
    )
    var expert_ids_dev = DeviceNDBuffer[DType.int32, 1](
        num_active_experts, ctx=ctx
    )

    # Move inputs to device
    ctx.enqueue_copy(a_dev.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_dev.buffer, b_host.tensor.data)
    ctx.enqueue_copy(a_offsets_dev.buffer, a_offsets_host.tensor.data)
    ctx.enqueue_copy(expert_ids_dev.buffer, expert_ids_host.tensor.data)

    # Run naive grouped matmul
    naive_grouped_matmul(
        c_dev.tensor,
        a_dev.tensor,
        b_dev.tensor,
        a_offsets_dev.tensor,
        expert_ids_dev.tensor,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    # Copy result back to host
    ctx.enqueue_copy(c_host.tensor.data, c_dev.buffer)
    ctx.synchronize()

    # Verify results
    c_host_buffer = c_host.tensor
    var current_offset = 0

    for expert_idx in range(num_active_experts):
        var expert_id = expert_ids[expert_idx]
        var num_tokens = num_tokens_by_expert[expert_idx]
        var has_negative_id = expert_id == -1

        print(
            "Expert",
            expert_idx,
            "has id",
            expert_id,
            "with",
            num_tokens,
            "tokens",
        )

        # Check each token for this expert
        for token_idx in range(num_tokens):
            var global_token_idx = current_offset + token_idx
            var has_non_zero = False

            # Check if any output value is non-zero for this token
            for n in range(N):
                var value = c_host_buffer[global_token_idx, n]
                if value != 0:
                    has_non_zero = True
                    break

            if has_negative_id == has_non_zero:
                print(
                    "ERROR: Found non-zero value for expert_id -1 at token",
                    global_token_idx,
                )
                print("Values:", end="")
                for n in range(min(5, N)):  # Print first 5 values
                    print(c_host_buffer[global_token_idx, n], end=" ")
                print()
                raise Error("Expected zero values for expert_id -1")
            else:
                # For valid expert_id, should have mostly non-zero values
                if not has_non_zero:
                    print(
                        "WARNING: All values are zero for valid expert_id",
                        expert_id,
                        "at token",
                        global_token_idx,
                    )

        current_offset += num_tokens

    print("✓ Negative lora_id test passed - expert_id -1 produces zero outputs")


fn main() raises:
    with DeviceContext() as ctx:
        # Single matmul
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=1,
            expert_shape = Index(256, 256),
        ](1, List[Int](128), List[Int](0), ctx)

        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=1,
            expert_shape = Index(512, 1024),
        ](1, List[Int](256), List[Int](0), ctx)

        # Multiple matmuls selecting part of experts
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(768, 1024),
        ](2, List[Int](128, 256), List[Int](0, 2), ctx)

        # Multiple matmuls selecting part of experts
        # num_tokesn not multiple of tile size
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=6,
            expert_shape = Index(1280, 1024),
        ](4, List[Int](27, 1500, 300, 150), List[Int](0, 3, 2, 4), ctx)

        # Multiple matmuls selecting part of experts
        # num_tokesn not multiple of tile size
        # expert N dimension not multiple of 256
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=6,
            expert_shape = Index(192, 1024),
        ](4, List[Int](27, 1500, 300, 150), List[Int](0, 3, 2, 4), ctx)

        # Multiple matmuls selecting part of experts with epilogue
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=4,
            expert_shape = Index(768, 1024),
            has_epilogue=True,
        ](2, List[Int](128, 256), List[Int](0, 2), ctx)

        # Test that expert id of -1 results in 0s in the output
        test[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=2,
            expert_shape = Index(256, 512),
        ](2, List[Int](64, 128), List[Int](0, -1), ctx)

        # Test negative lora_id behavior with naive matmul
        test_negative_lora_id[
            DType.bfloat16,
            DType.bfloat16,
            num_experts=2,
            expert_shape = Index(256, 512),
        ](2, List[Int](64, 128), List[Int](0, -1), ctx)
