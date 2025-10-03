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

from math import rsqrt
from sys import simd_width_of

from gpu.host import DeviceContext, get_gpu_target
from layout import Layout, LayoutTensor, RuntimeLayout
from nn.normalization import *
from testing import assert_almost_equal, assert_true

from utils.index import Index, IndexList


def compute_group_stats[
    t: DType
](vec: LayoutTensor[t, **_], size: Int, eps: Scalar[t]) -> (
    Scalar[t],
    Scalar[t],
):
    constrained[vec.rank == 1, "vec must be rank 1"]()
    var sum_val = Scalar[t]()
    var sum_sq = Scalar[t]()
    for i in range(size):
        sum_val += vec[i][0]
        sum_sq += vec[i][0] * vec[i][0]
    var mean = sum_val / size
    var variance = max((sum_sq / size) - (mean * mean), 0.0)
    return (mean, rsqrt(variance + eps))


fn run_group_norm_gpu[
    dtype: DType, rank: Int
](
    ctx: DeviceContext,
    shape: IndexList[rank],
    num_groups: Int,
    rtol: Float64 = 1e-4,
    atol: Float64 = 1e-5,
) raises:
    print("== run_group_norm_gpu")

    var N = shape[0]
    var C = shape[1]
    var spatial = shape.flattened_length() // (N * C)
    var group_size = C // num_groups * spatial
    var rows = N * num_groups
    var cols = group_size

    var data_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var res = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var gamma_h = UnsafePointer[Scalar[dtype]].alloc(C)
    var beta_h = UnsafePointer[Scalar[dtype]].alloc(C)

    for i in range(rows * cols):
        data_h[i] = Scalar[dtype](i % 256)  # bounded range to avoid overflow

    for i in range(C):
        gamma_h[i] = ((i + C) / C).cast[dtype]()
        beta_h[i] = (i / C).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](C)
    var beta_d = ctx.enqueue_create_buffer[dtype](C)

    var param_shape = Index(C)
    alias layout = Layout.row_major[rank]()
    alias layout_1d = Layout.row_major(UNKNOWN_VALUE)
    var data_buf = LayoutTensor[dtype, layout](
        data_d, RuntimeLayout[layout].row_major(shape)
    )
    var gamma = LayoutTensor[dtype, layout_1d](
        gamma_d, RuntimeLayout[layout_1d].row_major(param_shape)
    )
    var beta = LayoutTensor[dtype, layout_1d](
        beta_d, RuntimeLayout[layout_1d].row_major(param_shape)
    )
    var epsilon = Scalar[dtype](1e-5)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = data_buf.runtime_layout(
            RuntimeTuple[fill_like(data_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )

        return data_buf.ptr.load[width=width](idx)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_scalar_fn[width: Int](coords: IndexList[1]) -> SIMD[dtype, width]:
        var idx = gamma.runtime_layout(
            RuntimeTuple[fill_like(gamma.layout.shape, UNKNOWN_VALUE)](coords)
        )
        return gamma.ptr.load[width=width](idx)

    @__copy_capture(beta)
    @always_inline
    @parameter
    fn beta_scalar_fn[width: Int](coords: IndexList[1]) -> SIMD[dtype, width]:
        var idx = beta.runtime_layout(
            RuntimeTuple[fill_like(beta.layout.shape, UNKNOWN_VALUE)](coords)
        )
        return beta.ptr.load[width=width](idx)

    group_norm[dtype, rank, input_fn, gamma_scalar_fn, beta_scalar_fn, "gpu"](
        shape, epsilon, num_groups, data_buf, ctx=ctx
    )
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = LayoutTensor[dtype, layout_1d](
            data_h + r * cols,
            RuntimeLayout[layout_1d].row_major(IndexList[1](cols)),
        )
        var stats = compute_group_stats(vec, cols, epsilon)
        var mean_ref = stats[0]
        var norm_factor = stats[1]
        for c in range(cols):
            var g = r % num_groups
            var c_base = g * (C // num_groups)
            var offset = c // spatial
            var true_c = c_base + offset
            var idx = r * cols + c
            var val = (
                (vec[c] - mean_ref) * norm_factor * gamma_h[true_c]
            ) + beta_h[true_c]
            assert_almost_equal(val, res[idx], rtol=rtol, atol=atol)

    _ = data_d^
    _ = gamma_d^
    _ = beta_d^
    data_h.free()
    res.free()
    gamma_h.free()
    beta_h.free()


def main():
    with DeviceContext() as ctx:
        alias default_simd = simd_width_of[
            DType.float32, target = get_gpu_target()
        ]()

        # === Warp-Tiling Kernel Dispatch (SIMD-aligned, fits warp strategy) ===

        # Small, SIMD-aligned groups
        run_group_norm_gpu[DType.float32](ctx, Index(2, 8, 2, 2), num_groups=4)
        run_group_norm_gpu[DType.float32](ctx, Index(2, 8, 4), num_groups=4)

        # Larger, but still small enough for warp tiling
        run_group_norm_gpu[DType.float32](ctx, Index(2, 32, 2, 2), num_groups=8)
        run_group_norm_gpu[DType.float32](ctx, Index(2, 32, 4), num_groups=8)

        # SIMD aligned with group boundary, but not aligned with channel boundary
        run_group_norm_gpu[DType.float32](
            ctx, Index(2, 32, 1, 10), num_groups=8
        )

        # === Block Kernel Dispatch (too wide for warp or not divisible by SIMD width) ===

        # Large column count (too wide for warp)
        run_group_norm_gpu[DType.float32](
            ctx, Index(1, 128, 1, 64), num_groups=8
        )
        run_group_norm_gpu[DType.float32](ctx, Index(1, 128, 64), num_groups=8)

        # Aligned, but still too large for warp strategy
        run_group_norm_gpu[DType.float32](
            ctx, Index(1, 64, 1, 64), num_groups=8
        )
        run_group_norm_gpu[DType.float32](ctx, Index(1, 64, 64), num_groups=8)

        # === Invalid Case: cols < simd_width → triggers safety assertion ===

        # Misaligned shape
        try:
            run_group_norm_gpu[DType.float32](
                ctx, Index(1, 33, 1, 1), num_groups=11
            )
        except e:
            assert_true(
                "group_norm_gpu requires num_cols >= simd_width" in String(e)
            )
        try:
            run_group_norm_gpu[DType.float32](
                ctx, Index(1, 33, 1), num_groups=11
            )
        except e:
            assert_true(
                "group_norm_gpu requires num_cols >= simd_width" in String(e)
            )

        # Small group sizes result in too few columns
        try:
            run_group_norm_gpu[DType.float32](
                ctx, Index(1, 12, 1, 1), num_groups=6
            )
        except e:
            assert_true(
                "group_norm_gpu requires num_cols >= simd_width" in String(e)
            )
        try:
            run_group_norm_gpu[DType.float32](
                ctx, Index(1, 12, 1), num_groups=6
            )
        except e:
            assert_true(
                "group_norm_gpu requires num_cols >= simd_width" in String(e)
            )

        # === Edge Cases ===

        # Trivial spatial=1 (all channels collapsed to one dimension)
        run_group_norm_gpu[DType.float32](
            ctx, Index(2, 128, 1, 1), num_groups=1
        )
        run_group_norm_gpu[DType.float32](ctx, Index(2, 128, 1), num_groups=1)

        # Non-multiple of simd_width → scalar fallback block path
        run_group_norm_gpu[DType.float32](ctx, Index(2, 33, 1, 1), num_groups=1)
        run_group_norm_gpu[DType.float32](ctx, Index(2, 33, 1), num_groups=1)

        # One-channel, one-group (channel_per_group=1)
        run_group_norm_gpu[DType.float32](ctx, Index(2, 1, 4, 8), num_groups=1)
        run_group_norm_gpu[DType.float32](ctx, Index(2, 1, 32), num_groups=1)

        # Edge case from group norm layer tests
        run_group_norm_gpu[DType.float32](ctx, Index(2, 2, 4, 4), num_groups=1)
        run_group_norm_gpu[DType.float32](ctx, Index(2, 2, 16), num_groups=1)

        # Mismatched channels/groups → top-level init error
        try:
            run_group_norm_gpu[DType.float32](
                ctx, Index(2, 7, 3, 3), num_groups=3
            )
        except e:
            assert_true("Invalid num_groups" in String(e))
        try:
            run_group_norm_gpu[DType.float32](ctx, Index(2, 7, 9), num_groups=3)
        except e:
            assert_true("Invalid num_groups" in String(e))
