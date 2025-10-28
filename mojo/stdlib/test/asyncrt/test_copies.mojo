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

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu.host import DeviceBuffer, DeviceContext
from testing import TestSuite


fn _run_memcpy(ctx: DeviceContext, length: Int, use_context: Bool) raises:
    print("-")
    print("_run_memcpy(", length, ")")

    var in_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    var out_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    var in_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = i
        out_host[i] = length + i

    # Copy to and from device buffers.
    in_dev.enqueue_copy_from(in_host)
    if use_context:
        ctx.enqueue_copy(out_dev, in_dev)
    else:
        in_dev.enqueue_copy_to(out_dev)
    out_dev.enqueue_copy_to(out_host)

    # Wait for the copies to be completed.
    ctx.synchronize()

    var out_span = out_host.as_span()
    for i in range(len(out_span)):
        if i < 10:
            print("at index", i, "the value is", out_span[i])
        expect_eq(out_span[i], i, "at index ", i, " the value is ", out_span[i])


fn _run_sub_memcpy(ctx: DeviceContext, length: Int) raises:
    print("-")
    print("_run_sub_memcpy(", length, ")")

    var half_length = length // 2

    var in_host = ctx.enqueue_create_host_buffer[DType.int64](length)
    var out_host = ctx.enqueue_create_host_buffer[DType.int64](length)
    var in_dev = ctx.enqueue_create_buffer[DType.int64](length)
    var out_dev = ctx.enqueue_create_buffer[DType.int64](length)
    var first_out_dev = out_dev.create_sub_buffer[DType.int64](0, half_length)
    var second_out_dev = out_dev.create_sub_buffer[DType.int64](
        half_length, half_length
    )

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = i
        out_host[i] = length + i

    # Copy to and from device buffers.
    in_host.enqueue_copy_to(in_dev)
    in_dev.enqueue_copy_to(out_dev)

    # Swap halves on copy back.
    # Using sub buffer
    second_out_dev.enqueue_copy_to(
        out_host.create_sub_buffer[DType.int64](0, half_length)
    )
    # Using host pointer math
    first_out_dev.enqueue_copy_to(out_host.unsafe_ptr().offset(half_length))

    # Wait for the copies to be completed.
    ctx.synchronize()

    for i in range(length):
        var expected: Int
        if i < half_length:
            expected = i + half_length
        else:
            expected = i - half_length
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        expect_eq(
            out_host[i], expected, "at index ", i, " the value is ", out_host[i]
        )


fn _run_fake_memcpy(ctx: DeviceContext, length: Int, use_take_ptr: Bool) raises:
    print("-")
    print("_run_fake_memcpy(", length, ", take_ptr = ", use_take_ptr, ")")

    var half_length = length // 2

    var in_host = ctx.enqueue_create_host_buffer[DType.int64](length)
    var out_host = ctx.enqueue_create_host_buffer[DType.int64](length)
    var in_dev = ctx.enqueue_create_buffer[DType.int64](length)
    var out_dev = ctx.enqueue_create_buffer[DType.int64](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = i
        out_host[i] = length + i

    # Copy to and from device buffers.
    in_host.enqueue_copy_to(in_dev)
    in_dev.enqueue_copy_to(out_dev)

    var out_ptr: UnsafePointer[Int64]
    if use_take_ptr:
        out_ptr = out_dev.take_ptr()
    else:
        out_ptr = out_dev.unsafe_ptr()

    var first_out_dev = DeviceBuffer[DType.int64](
        ctx, out_ptr, half_length, owning=use_take_ptr
    )
    var interior_out_ptr: UnsafePointer[Int64] = out_ptr.offset(half_length)
    var second_out_dev = DeviceBuffer[DType.int64](
        ctx, interior_out_ptr, half_length, owning=False
    )

    # Swap halves on copy back.
    # Using sub buffer
    second_out_dev.enqueue_copy_to(
        out_host.create_sub_buffer[DType.int64](0, half_length)
    )
    # Using host pointer math
    first_out_dev.enqueue_copy_to(out_host.unsafe_ptr().offset(half_length))

    # Wait for the copies to be completed.
    ctx.synchronize()

    for i in range(length):
        var expected: Int
        if i < half_length:
            expected = i + half_length
        else:
            expected = i - half_length
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        expect_eq(
            out_host[i], expected, "at index ", i, " the value is ", out_host[i]
        )


fn _run_cpu_ctx_memcpy_async(
    ctx: DeviceContext, cpu_ctx: DeviceContext, length: Int
) raises:
    print("-")
    print("_run_cpu_ctx_memcpy_async(", length, ")")

    var host_buf = cpu_ctx.enqueue_create_host_buffer[DType.int64](length)
    var dev_buf = ctx.enqueue_create_buffer[DType.int64](length).enqueue_fill(
        13
    )

    for i in range(length):
        host_buf[i] = 2 * i

    ctx.enqueue_copy(dev_buf, host_buf)

    with dev_buf.map_to_host() as dev_buf:
        for i in range(length):
            expect_eq(dev_buf[i], 2 * i)

    host_buf = host_buf.enqueue_fill(12)
    cpu_ctx.enqueue_copy(host_buf, dev_buf)

    for i in range(length):
        expect_eq(host_buf[i], 2 * i)


def test_copies():
    var ctx = create_test_device_context()

    print("-------")
    print("Running test_copies(" + ctx.name() + "):")

    alias one_mb = 1024 * 1024

    _run_memcpy(ctx, 64, True)
    _run_memcpy(ctx, one_mb, True)
    _run_memcpy(ctx, 64, False)
    _run_memcpy(ctx, one_mb, False)

    _run_sub_memcpy(ctx, 64)

    _run_fake_memcpy(ctx, 64, False)
    # _run_fake_memcpy(ctx, 64, True)

    var cpu_ctx = DeviceContext(api="cpu")
    _run_cpu_ctx_memcpy_async(ctx, cpu_ctx, 64)
    _run_cpu_ctx_memcpy_async(ctx, cpu_ctx, one_mb)

    print("Done.")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
