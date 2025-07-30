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

import time
from math import floor
from sys import sizeof

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.comm.allreduce import (
    MAX_GPUS,
    Signal,
    allreduce,
)
from gpu.host import DeviceBuffer, DeviceContext
from testing import assert_almost_equal, assert_true

from utils import IndexList, StaticTuple


fn _pretty_print_float(val: Float64) -> String:
    """This converts the float value to a string, but omits the fractional part
    if not needed (e.g. prints 2 instead of 2.0).
    """
    if Float64(floor(val)) == val:
        return String(Int(val))
    return String(val)


fn _human_memory(size: Int) -> String:
    alias KB = 1024
    alias MB = KB * KB
    alias GB = MB * KB

    if size >= GB:
        return _pretty_print_float(Float64(size) / GB) + "GB"

    if size >= MB:
        return _pretty_print_float(Float64(size) / MB) + "MB"

    if size >= KB:
        return _pretty_print_float(Float64(size) / KB) + "KB"

    return String(size) + "B"


fn allreduce_test[
    dtype: DType, rank: Int, ngpus: Int
](list_of_ctx: List[DeviceContext], length: Int) raises:
    alias num_warmups = 5
    alias num_iters = 100

    constrained[ngpus in (1, 2, 4, 8), "ngpus must be 1, 2, 4, or 8"]()
    constrained[rank == 1, "this test code currently assumes rank 1"]()

    # Create device buffers for all GPUs
    var in_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_buffers = List[UnsafePointer[Scalar[dtype]]](capacity=ngpus)

    # Create signal buffers for synchronization
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal], MAX_GPUS](fill={})

    # Set up temp buffers for GPUs to reduce-scatter into / all-gather from.
    var temp_buffer_num_bytes = ngpus * sizeof[dtype]() * length

    # Initialize buffers for each GPU
    @parameter
    for i in range(ngpus):
        # Create and store device buffers
        in_bufs_list.append(list_of_ctx[i].enqueue_create_buffer[dtype](length))
        out_bufs_list.append(
            list_of_ctx[i].enqueue_create_buffer[dtype](length)
        )

        # Create and initialize host buffers
        var host_buffer = UnsafePointer[Scalar[dtype]].alloc(length)
        host_buffers.append(host_buffer)

        # Initialize host buffer with values (i + 1).0
        var host_nd_buf = NDBuffer[dtype, rank](host_buffer, DimList(length))
        host_nd_buf.fill(Scalar[dtype](i + 1))

        # Create and initialize signal buffers
        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[DType.uint8](
                sizeof[Signal]() + temp_buffer_num_bytes
            )
        )
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
        rank_sigs[i] = signal_buffers[i].unsafe_ptr().bitcast[Signal]()

        # Copy data to device
        list_of_ctx[i].enqueue_copy(in_bufs_list[i], host_buffers[i])

    # Create and initialize input and output buffers.
    var in_bufs = InlineArray[NDBuffer[dtype, rank, MutableAnyOrigin], ngpus](
        fill={}
    )
    var out_bufs = InlineArray[NDBuffer[dtype, rank, MutableAnyOrigin], ngpus](
        fill={}
    )

    @parameter
    for i in range(ngpus):
        in_bufs[i] = NDBuffer[dtype, rank](
            in_bufs_list[i].unsafe_ptr(), DimList(length)
        )
        out_bufs[i] = NDBuffer[dtype, rank](
            out_bufs_list[i].unsafe_ptr(), DimList(length)
        )

    @parameter
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Copy-capture in registers since the lambda will be used on GPU.
    var out_bufs_capture = StaticTuple[
        NDBuffer[dtype, rank, MutableAnyOrigin], ngpus
    ](NDBuffer[dtype, rank, MutableAnyOrigin]())

    @parameter
    for i in range(ngpus):
        out_bufs_capture[i] = NDBuffer[dtype, rank](
            out_bufs_list[i].unsafe_ptr(), DimList(length)
        )

    @always_inline
    @parameter
    @__copy_capture(out_bufs_capture)
    fn outputs_lambda[
        input_index: Int,
        _dtype: DType,
        _rank: Int,
        _width: Int,
        *,
        _alignment: Int,
    ](coords: IndexList[_rank], val: SIMD[_dtype, _width]) -> None:
        out_bufs_capture[input_index].store[width=_width, alignment=_alignment](
            rebind[IndexList[rank]](coords), rebind[SIMD[dtype, _width]](val)
        )

    # Warm up.
    for _ in range(num_warmups):
        allreduce[ngpus=ngpus, outputs_lambda=outputs_lambda](
            in_bufs, out_bufs, rank_sigs, list_of_ctx
        )

    # Synchronize all devices.
    @parameter
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Perform a benchmarked allreduce.
    start_t = time.perf_counter_ns()

    @parameter
    for _ in range(num_iters):
        allreduce[ngpus=ngpus, outputs_lambda=outputs_lambda](
            in_bufs, out_bufs, rank_sigs, list_of_ctx
        )

    # Synchronize all devices.
    @parameter
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    end_t = time.perf_counter_ns()

    # Quick and dirty benchmark since benchmark module doesn't support
    # multi-device contexts
    print("Time taken (ms):", (end_t - start_t) / (1_000_000 * num_iters))

    # Copy results back and verify
    var expected_sum = Scalar[dtype](0)

    @parameter
    for i in range(ngpus):
        expected_sum += i + 1
        list_of_ctx[i].enqueue_copy(host_buffers[i], out_bufs_list[i])

    # Verify results
    @parameter
    for i in range(ngpus):
        for j in range(length):
            try:
                assert_almost_equal(host_buffers[i][j], expected_sum)
            except e:
                print("Verification failed at GPU", i, "index", j)
                print("Value:", host_buffers[i][j])
                print("Expected:", expected_sum)
                raise e

    # Cleanup
    for i in range(ngpus):
        host_buffers[i].free()
    _ = signal_buffers^


fn _get_test_str[dtype: DType](ngpus: Int, length: Int) -> String:
    return String(
        "====allreduce-",
        dtype,
        "-",
        ngpus,
        "-",
        _human_memory(sizeof[dtype]() * length),
    )


def main():
    # Test configurations covering edge cases
    # fmt: off
    alias test_lengths = (
        8 * 1024,           # Small latency bound
        128 * 1024,         # Larger latency bound
        256 * 1024,         # Smallest bandwidth bound
        16 * 1024 * 1024,   # Bandwidth bound
        64 * 1024 * 1024,   # Bandwidth bound: 8192 chunk size at dim = 8192
    )
    # fmt: on

    # Test hyperparameters.
    alias test_dtypes = (DType.bfloat16, DType.float32)
    alias test_gpu_counts = (2, 4, 8)

    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )

    # Run tests for each configuration.
    @parameter
    for gpu_idx in range(len(test_gpu_counts)):
        alias num_gpus = test_gpu_counts[gpu_idx]
        if DeviceContext.number_of_devices() < num_gpus:
            break

        # Create GPU context.
        var ctx = List[DeviceContext]()
        for i in range(num_gpus):
            ctx.append(DeviceContext(device_id=i))

        # Test all cases for this configuration.
        @parameter
        for dtype_idx in range(len(test_dtypes)):
            alias dtype = test_dtypes[dtype_idx]

            @parameter
            for length_idx in range(len(test_lengths)):
                alias length = test_lengths[length_idx]

                print(_get_test_str[dtype](num_gpus, length))
                try:
                    allreduce_test[dtype=dtype, rank=1, ngpus=num_gpus](
                        ctx, length
                    )
                except e:
                    if "OUT_OF_MEMORY" in String(e):
                        print(
                            "Out of memory error occurred for ",
                            _get_test_str[dtype](num_gpus, length),
                        )
                    else:
                        raise e
