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

# DOC: max/layout/tensors.mdx

from layout import Coord, Idx, TileTensor, row_major, col_major
from layout.tile_layout import Layout, blocked_product
from std.collections import Array
from std.math import ceildiv
from std.memory import unsafe_memset
from std.testing import assert_equal


def accessing_tensor_elements_example() raises:
    comptime rows = 4
    comptime columns = 8
    var layout = row_major[rows, columns]()
    var storage = Array[Float32, rows * columns](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    var tensor = TileTensor(storage, layout)

    var row, col = 0, 1
    # start-access-example-1
    var element = tensor[row, col]  # The element is a scalar value.
    # end-access-example-1

    assert_equal(element, 1)
    row, col = 0, 0

    # start-access-example-2
    var elements = tensor.load[4](Coord(row, col))
    elements = elements * 2
    tensor.store(Coord(row, col), elements)
    # end-access-example-2

    element = tensor[0, 2]
    assert_equal(element, 4)
    assert_equal(tensor.num_elements(), rows * columns)
    comptime assert type_of(tensor).element_size == 1
    for i in range(rows):
        for j in range(columns):
            var offset = i * columns + j
            var expected = offset * 2 if offset < 4 else offset
            assert_equal(tensor[i, j], Float32(expected))


def accessing_nested_tensor_elements_example() raises:
    comptime rows = 4
    comptime columns = 6
    var tiler = row_major[2, 3]()
    var layout = blocked_product(col_major[2, 2](), tiler)
    var storage = Array[Float32, rows * columns](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    var tensor = TileTensor(storage, layout)

    # start-access-nested-tensor-example
    var element = tensor[1, 0, 0, 1]
    # end-access-nested-tensor-example

    assert_equal(element, 5)
    assert_equal(tensor.layout.product(), rows * columns)
    assert_equal(Int(tensor.dim[0]()), rows)
    assert_equal(Int(tensor.dim[1]()), columns)
    for row_inner in range(2):
        for row_outer in range(2):
            for col_inner in range(2):
                for col_outer in range(3):
                    assert_equal(
                        tensor[row_inner, row_outer, col_inner, col_outer],
                        Float32(
                            row_inner
                            + row_outer * 12
                            + col_inner * 2
                            + col_outer * 4
                        ),
                    )


def layout_tensor_on_cpu_example() raises:
    # start-layout-tensor-on-cpu-example
    comptime rows = 8
    comptime columns = 16
    var layout = row_major[rows, columns]()
    var storage = Array[Float32, rows * columns](uninitialized=True)
    var tensor = TileTensor(storage, layout)
    # end-layout-tensor-on-cpu-example
    assert_equal(tensor.num_elements(), rows * columns)
    _ = tensor


def layout_tensor_from_pointer_example() raises:
    # start-layout-tensor-from-pointer-example
    comptime rows = 1024
    comptime columns = 1024
    comptime buf_size = rows * columns
    var layout = row_major[rows, columns]()
    var ptr = alloc[Float32]({count = buf_size}).unsafe_leak()
    unsafe_memset(ptr, 0, buf_size)
    var tensor = TileTensor(ptr, layout)
    # end-layout-tensor-from-pointer-example
    assert_equal(tensor.num_elements(), rows * columns)
    for i in range(rows):
        for j in range(columns):
            assert_equal(tensor[i, j], Float32(0))
    ptr.unsafe_free()


def layout_tensor_tile_example() raises:
    # start-layout-tensor-tile-example
    comptime rows = 2
    comptime columns = 4
    comptime tile_size = 32
    var tiled_layout = Layout(
        Coord(
            Coord(Idx[rows], Idx[tile_size]),
            Coord(Idx[columns], Idx[tile_size]),
        ),
        Coord(
            Coord(Idx[columns * tile_size * tile_size], Idx[tile_size]),
            Coord(Idx[tile_size * tile_size], Idx[1]),
        ),
    )
    var storage = Array[Float32, rows * columns * tile_size * tile_size](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    var tensor = TileTensor(storage, tiled_layout)
    var tile = tensor.tile[32, 32](0, 1)
    # end-layout-tensor-tile-example
    assert_equal(tile[0, 0], Float32(tile_size * tile_size))
    # start-layout-tensor-tile-example-2
    var my_tile = tensor.tile[tile_size, tile_size](0, 0)
    assert_equal(my_tile[0, 0], Float32(0))
    for i in range(rows):
        for j in range(columns):
            my_tile = tensor.tile[tile_size, tile_size](i, j)
            # end-layout-tensor-tile-example-2
            for row in range(tile_size):
                for col in range(tile_size):
                    assert_equal(
                        my_tile[row, col],
                        Float32(
                            (i * columns + j) * tile_size * tile_size
                            + row * tile_size
                            + col
                        ),
                    )


# Visits contiguous 4-by-4 tiles in storage order.
def layout_tensor_iterator_example() raises:
    # start-layout-tensor-iterator-example-1
    comptime buf_size = 128
    comptime tile_size = 16
    var storage = Array[Int16, buf_size](
        fill_with=lambda (i: Int) -> Int16: Int16(i)
    )
    var tensor = TileTensor(storage, row_major[buf_size]())
    for i in range(ceildiv(buf_size, tile_size)):
        var tile = tensor.tile[tile_size](i).reshape[4, 4]()
        # end-layout-tensor-iterator-example-1
        assert_equal(tile[0, 0], Int16(i * tile_size))
        assert_equal(tile.num_elements(), tile_size)
        for row in range(4):
            for col in range(4):
                assert_equal(
                    tile[row, col], Int16(i * tile_size + row * 4 + col)
                )


def layout_tensor_iterator_example2() raises:
    comptime rows = 4
    comptime cols = 8
    comptime size = rows * cols
    comptime tile_size = 2
    var storage = Array[Int32, size](
        fill_with=lambda (i: Int) -> Int32: Int32(i)
    )

    var tensor = TileTensor(storage, row_major[rows, cols]())
    # start-layout-tensor-iterator-example-2
    # given a tensor of size rows x cols
    comptime num_row_tiles = ceildiv(rows, tile_size)
    comptime num_col_tiles = ceildiv(cols, tile_size)

    for i in range(num_row_tiles):
        for j in range(num_col_tiles):
            var tile = tensor.tile[tile_size, tile_size](i, j)
            # end-layout-tensor-iterator-example-2
            assert_equal(tile.num_elements(), tile_size * tile_size)
            assert_equal(Int(tile.layout.stride[0]().value()), cols)
            assert_equal(Int(tile.layout.stride[1]().value()), 1)
            for row in range(tile_size):
                for col in range(tile_size):
                    assert_equal(
                        tile[row, col],
                        Int32(
                            (i * tile_size + row) * cols + j * tile_size + col
                        ),
                    )


def main() raises:
    accessing_tensor_elements_example()
    accessing_nested_tensor_elements_example()
    layout_tensor_on_cpu_example()
    layout_tensor_from_pointer_example()
    layout_tensor_tile_example()
    layout_tensor_iterator_example()
    layout_tensor_iterator_example2()
