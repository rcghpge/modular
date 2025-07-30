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

from collections import InlineArray
from math import floor
from sys import env_get_dtype, env_get_int, sizeof

from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.comm.allreduce import MAX_GPUS, Signal, allreduce, can_enable_p2p
from gpu.host import DeviceBuffer, DeviceContext
from internal_utils import arg_parse
from testing import assert_almost_equal, assert_true

from utils.index import IndexList, StaticTuple


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


# TODO: convert 'ngpus' to runtime variable
fn bench_reduce[
    dtype: DType, rank: Int, ngpus: Int, max_num_blocks: Int
](mut m: Bench, list_of_ctx: List[DeviceContext], num_bytes: Int) raises:
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
    var temp_buffer_num_bytes = ngpus * num_bytes
    var length = num_bytes // sizeof[dtype]()

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

    for i in range(ngpus):
        in_bufs[i] = NDBuffer[dtype, rank](
            in_bufs_list[i].unsafe_ptr(), DimList(length)
        )
        out_bufs[i] = NDBuffer[dtype, rank](
            out_bufs_list[i].unsafe_ptr(), DimList(length)
        )
        # Ensure setup has propagated.
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

    @parameter
    @always_inline
    fn bench_iter(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn call_fn() raises:
            @parameter
            if max_num_blocks:
                allreduce[ngpus=ngpus, outputs_lambda=outputs_lambda](
                    in_bufs, out_bufs, rank_sigs, list_of_ctx, max_num_blocks
                )
            else:
                allreduce[ngpus=ngpus, outputs_lambda=outputs_lambda](
                    in_bufs, out_bufs, rank_sigs, list_of_ctx
                )

        b.iter_custom_multicontext[call_fn](list_of_ctx)

    var name = String(_get_test_str[dtype](ngpus, num_bytes))
    m.bench_function[bench_iter](
        BenchId(name),
        # add data movement to measures
        ThroughputMeasure(BenchMetric.bytes, num_bytes),
    )

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


fn _get_test_str[dtype: DType](ngpus: Int, num_bytes: Int) -> String:
    return String(
        "allreduce-", dtype, "-", ngpus, "-", _human_memory(num_bytes)
    )


def main():
    var num_bytes = arg_parse("num_bytes", 16 * 1024)

    alias dtype = env_get_dtype["dtype", DType.bfloat16]()
    alias num_gpus = env_get_int["num_gpus", 2]()
    alias rank = env_get_int["rank", 1]()
    # Force passing `max_num_blocks` explicitly.
    alias max_num_blocks = env_get_int["TUNE_MAX_NUM_BLOCKS", -1]()

    assert_true(DeviceContext.number_of_devices() >= num_gpus)
    assert_true(num_bytes % sizeof[dtype]() == 0)

    # Create GPU context.
    var ctx = List[DeviceContext]()
    for i in range(num_gpus):
        ctx.append(DeviceContext(device_id=i))

    if not can_enable_p2p(ctx):
        # Don't benchmark the naive allreduce.
        print("P2P not enabled, skipping benchmark.")
        return

    # Generate descriptive test name.
    print(_get_test_str[dtype](num_gpus, num_bytes))

    var m = Bench()

    bench_reduce[
        dtype=dtype, rank=rank, ngpus=num_gpus, max_num_blocks=max_num_blocks
    ](m, ctx, num_bytes)

    m.dump_report()
