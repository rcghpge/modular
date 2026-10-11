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
"""Checks rounded distribution metadata without accessing virtual edge points."""

from layout import Coord, DefaultEngine, IntTuple, Layout as LegacyLayout
from layout import LayoutTensor, TileTensor, col_major, coord, row_major
from layout.swizzle import Swizzle
from layout.tile_layout import Layout
from std.math import ceildiv
from std.sys import is_gpu, size_of
from std.testing import TestSuite, assert_equal, assert_true


def _check_metadata[
    width: Int,
    //,
    thread_layout: Layout,
    swizzle: Optional[Swizzle] = None,
](
    tensor: TileTensor[
        mut=False, Engine=DefaultEngine[element_width=width], ...
    ],
    allocation: TileTensor[
        mut=False, tensor.dtype, _, address_space=tensor.address_space, ...
    ],
) raises:
    comptime assert tensor.rank == tensor.flat_rank == 2
    comptime assert tensor.all_dims_known
    comptime rows = tensor.static_shape[0]
    comptime cols = tensor.static_shape[1]
    comptime row_stride = tensor.static_stride[0]
    comptime col_stride = tensor.static_stride[1]
    comptime thread_rows = thread_layout.static_shape[0]
    comptime thread_cols = thread_layout.static_shape[1]
    comptime fragment_rows = ceildiv(rows, thread_rows)
    comptime fragment_cols = ceildiv(cols, thread_cols)
    comptime LegacyThreads = LegacyLayout(
        IntTuple(thread_rows, thread_cols),
        IntTuple(
            thread_layout.static_stride[0], thread_layout.static_stride[1]
        ),
    )
    comptime LegacyData = LegacyLayout(
        IntTuple(rows, cols), IntTuple(row_stride, col_stride)
    )
    # Legacy pointer construction requires signed indices, even for the
    # native unsigned-index case checked against this wider reference.
    var legacy = LayoutTensor[
        tensor.dtype,
        LegacyData,
        tensor.origin,
        address_space=tensor.address_space,
        element_layout=LegacyLayout(width),
        linear_idx_type=.int64,
    ](tensor.unsafe_ptr())
    var allocation_handle = allocation.unsafe_ptr().bitcast[
        SIMD[tensor.dtype, width]
    ]()
    var parent_base = tensor.Engine.distance(tensor._storage, allocation_handle)
    var coverage = Array[Int, max(1, rows * cols)](fill=0)
    var inactive = 0
    var virtual_points = 0
    var observed_parent_swizzle = False
    comptime parent_swizzle = Swizzle(1, 3, 3)

    for worker in range(thread_rows * thread_cols + 6):
        if worker >= thread_rows * thread_cols:
            inactive += 1
            continue

        var fragment = tensor.distribute_ceil[thread_layout, swizzle](worker)
        var old_fragment = legacy.distribute[LegacyThreads, swizzle=swizzle](
            worker
        )
        comptime assert fragment.Engine == tensor.Engine
        comptime assert type_of(fragment._storage) == type_of(tensor._storage)
        comptime assert fragment.address_space == tensor.address_space
        comptime assert fragment.linear_idx_type == tensor.linear_idx_type
        comptime assert fragment.element_size == width
        assert_equal(fragment.static_shape[0], fragment_rows)
        assert_equal(fragment.static_shape[1], fragment_cols)
        assert_equal(fragment.static_stride[0], row_stride * thread_rows)
        assert_equal(fragment.static_stride[1], col_stride * thread_cols)
        assert_equal(old_fragment.layout.shape[0].value(), fragment_rows)
        assert_equal(old_fragment.layout.shape[1].value(), fragment_cols)
        assert_equal(
            old_fragment.layout.stride[0].value(),
            0 if fragment_rows == 1 else row_stride * thread_rows,
        )
        assert_equal(
            old_fragment.layout.stride[1].value(),
            0 if fragment_cols == 1 else col_stride * thread_cols,
        )

        var worker_row = (
            worker // thread_layout.static_stride[0]
        ) % thread_rows
        var worker_col = (
            worker // thread_layout.static_stride[1]
        ) % thread_cols
        var expected_base = worker_row * row_stride + worker_col * col_stride
        comptime if swizzle:
            expected_base = swizzle.value()(expected_base // width) * width
        var base = fragment.Engine.distance(fragment._storage, tensor._storage)
        var old_base = (Int(old_fragment.ptr) - Int(legacy.ptr)) // size_of[
            tensor.dtype
        ]()
        assert_equal(base, expected_base)
        assert_equal(old_base, expected_base)
        assert_equal(Int(fragment._distance(tensor)), expected_base)
        assert_equal(
            fragment.Engine.distance(fragment._storage, allocation_handle),
            parent_base + expected_base,
        )

        # Both APIs unravel a scalar logical index with the row axis first.
        for i in range(fragment_rows * fragment_cols):
            var local_row = i % fragment_rows
            var local_col = i // fragment_rows
            var row = worker_row + local_row * thread_rows
            var col = worker_col + local_col * thread_cols
            var valid = row < rows and col < cols
            var local = Int(
                fragment.layout[linear_idx_type=tensor.linear_idx_type](i)
            )
            var old_local = materialize[old_fragment.layout]()(IntTuple(i))
            var expected_local = (
                local_row * row_stride * thread_rows
                + local_col * col_stride * thread_cols
            )
            assert_equal(local, expected_local)
            assert_equal(old_local, expected_local)
            assert_equal(
                Int(fragment.layout(Coord(local_row, local_col))),
                expected_local,
            )
            for lane in range(width):
                assert_equal(
                    parent_base + base + local + lane,
                    parent_base + old_base + old_local + lane,
                )

            # Copier swizzling uses the parent base in scalar units, unlike
            # distribute's optional relative logical-element swizzle.
            var local_base = local % parent_swizzle.size()
            var swizzled = parent_swizzle(parent_base + base + local_base) + (
                local - local_base
            )
            var old_local_base = old_local % parent_swizzle.size()
            var old_swizzled = parent_swizzle(
                parent_base + old_base + old_local_base
            ) + (old_local - old_local_base)
            assert_equal(swizzled, old_swizzled)
            observed_parent_swizzle |= (
                swizzled
                != parent_base
                + base
                + parent_swizzle(local_base)
                + (local - local_base)
            )
            if valid:
                coverage[row * cols + col] += 1
            else:
                virtual_points += 1

    assert_equal(inactive, 6)
    for i in range(rows * cols):
        assert_equal(coverage[i], 1)
    assert_equal(
        virtual_points,
        thread_rows * thread_cols * fragment_rows * fragment_cols - rows * cols,
    )
    if fragment_rows * fragment_cols > 0 and parent_base != 0:
        assert_true(observed_parent_swizzle)


def _check_case[
    rows: Int,
    cols: Int,
    row_stride: Int,
    col_stride: Int,
    thread_layout: Layout,
    width: Int = 1,
    index_dtype: DType = .int32,
    dtype: DType = .int32,
    swizzle: Optional[Swizzle] = None,
]() raises:
    comptime parent_base = 72
    comptime capacity = (
        parent_base
        + (rows + thread_layout.static_shape[0]) * row_stride
        + (cols + thread_layout.static_shape[1]) * col_stride
        + 1024
    )
    var storage = Array[Scalar[dtype], capacity](uninitialized=True)
    var allocation = TileTensor(storage, row_major[capacity]()).as_imm()
    comptime DataLayout = Layout[
        coord[rows, cols].element_types,
        coord[row_stride, col_stride].element_types,
    ]
    var tensor = TileTensor[
        dtype,
        DataLayout,
        allocation.origin,
        Engine=DefaultEngine[element_width=width],
        linear_idx_type=index_dtype,
    ](
        (allocation.unsafe_ptr() + parent_base).bitcast[SIMD[dtype, width]](),
        DataLayout(),
    )
    _check_metadata[thread_layout, swizzle](tensor, allocation)


def test_divisible_and_ragged_metadata() raises:
    _check_case[16, 8, 8, 1, row_major[4, 4]()]()
    _check_case[7, 5, 13, 2, row_major[3, 2](), index_dtype=.int64]()
    _check_case[7, 5, 1, 11, col_major[3, 2](), index_dtype=.uint32]()
    _check_case[2, 1, 4, 1, row_major[5, 3]()]()
    _check_case[7, 5, 0, 2, row_major[3, 2]()]()


def test_zero_extent_metadata() raises:
    _check_case[0, 5, 8, 1, row_major[3, 2]()]()
    _check_case[7, 0, 8, 1, row_major[3, 2]()]()
    _check_case[0, 0, 0, 0, row_major[3, 2]()]()


def test_relative_distribution_swizzle() raises:
    _check_case[
        7,
        5,
        64,
        8,
        row_major[3, 2](),
        width=8,
        dtype=.float16,
        swizzle=Swizzle(1, 0, 3),
    ]()
    _check_case[
        7, 5, 13, 1, row_major[3, 2](), dtype=.uint8, swizzle=Swizzle(1, 0, 2)
    ]()


def test_depth80_vectorized_native_views() raises:
    var storage = Array[Float16, 65536](uninitialized=True)
    var allocation = TileTensor(storage, row_major[512, 128]()).as_imm()
    var source = allocation.tile[128, 128](1, 0).vectorize[1, 8]()
    var destination = (
        TileTensor(storage, row_major[512, 80]())
        .as_imm()
        .tile[128, 80](1, 0)
        .vectorize[1, 8]()
    )
    comptime workers = row_major[25, 10]()
    _check_metadata[workers](source, allocation)
    _check_metadata[workers](destination, allocation)
    var source_fragment = source.distribute_ceil[workers](0)
    var destination_fragment = destination.distribute_ceil[workers](0)
    assert_equal(source_fragment.static_shape[0], 6)
    assert_equal(source_fragment.static_shape[1], 2)
    assert_equal(destination_fragment.static_shape[0], 6)
    assert_equal(destination_fragment.static_shape[1], 1)
    assert_equal(source.distribute[workers](0).static_shape[0], 5)
    assert_equal(source.distribute[workers](0).static_shape[1], 1)
    assert_equal(destination.distribute[workers](0).static_shape[0], 5)
    for i in range(6):
        assert_equal(Int(source_fragment.layout(i)), i * 25 * 128)
        assert_equal(Int(destination_fragment.layout(i)), i * 25 * 80)


def main() raises:
    comptime assert not is_gpu(), "This oracle checks CPU metadata only"
    TestSuite.discover_tests[__functions_in_module()]().run()
