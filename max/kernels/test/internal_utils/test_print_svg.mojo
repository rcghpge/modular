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

from pathlib import Path

from layout import IntTuple, Layout, LayoutTensor
from layout._print_svg import print_svg
from layout.swizzle import Swizzle


fn test_svg_nvidia_shape() raises:
    # nvidia tensor core a matrix fragment
    alias layout = Layout.row_major(16, 16)
    var stack = InlineArray[Float32, layout.size()](uninitialized=True)
    var tensor = LayoutTensor[DType.float32, layout](stack)
    var tensor_dist = tensor.vectorize[1, 2]().distribute[
        Layout.row_major(8, 4)
    ](0)

    var tensor_list = List[__type_of(tensor_dist.get_immutable())]()
    for i in range(32):
        tensor_list.append(
            tensor.vectorize[1, 2]()
            .distribute[Layout.row_major(8, 4)](UInt(i))
            .get_immutable()
        )

    fn color_map(t: Int, v: Int) -> String:
        colors = [
            StaticString("red"),
            StaticString("blue"),
            StaticString("green"),
            StaticString("yellow"),
            StaticString("purple"),
            StaticString("orange"),
            StaticString("pink"),
            StaticString("brown"),
            StaticString("gray"),
            StaticString("black"),
            StaticString("white"),
        ]
        return String(colors[t // 4])

    print_svg(
        tensor.get_immutable(),
        tensor_list,
        color_map,
        file_path=Path("./test_svg_nvidia_shape.svg"),
    )


fn test_svg_nvidia_tile() raises:
    # nvidia tensor core a matrix fragment
    alias layout = Layout.row_major(16, 16)
    var stack = InlineArray[Float32, layout.size()](uninitialized=True)
    var tensor = LayoutTensor[DType.float32, layout](stack)
    var tensor_dist = tensor.vectorize[2, 2]().tile[4, 4](0, 1)
    print_svg(
        tensor.get_immutable(),
        [tensor_dist.get_immutable()],
        file_path=Path("./test_svg_nvidia_tile.svg"),
    )


fn test_svg_nvidia_tile_memory_bank() raises:
    # nvidia tensor core a matrix fragment
    alias layout = Layout.row_major(16, 16)
    var stack = InlineArray[Float32, layout.size()](uninitialized=True)
    var tensor = LayoutTensor[DType.float32, layout](stack)
    var tensor_dist = tensor.vectorize[2, 2]().tile[4, 4](0, 1)
    print_svg[memory_bank= (4, 32)](
        tensor.get_immutable(),
        [tensor_dist.get_immutable()],
        file_path=Path("./test_svg_nvidia_tile_memory_bank.svg"),
    )


fn test_svg_amd_shape_a() raises:
    # amd tensor core a matrix fragment
    alias layout = Layout.row_major(16, 16)
    var stack = InlineArray[Float32, layout.size()](uninitialized=True)
    var tensor = LayoutTensor[DType.float32, layout](stack)
    var tensor_dist = tensor.distribute[Layout.col_major(16, 4)](0)
    print_svg(
        tensor.get_immutable(),
        [tensor_dist.get_immutable()],
        file_path=Path("./test_svg_amd_shape_a.svg"),
    )


fn test_svg_amd_shape_b() raises:
    # amd tensor core a matrix fragment
    alias layout = Layout.row_major(16, 16)
    var stack = InlineArray[Float32, layout.size()](uninitialized=True)
    var tensor = LayoutTensor[DType.float32, layout](stack)
    var tensor_dist = tensor.distribute[Layout.row_major(4, 16)](0)
    print_svg(
        tensor.get_immutable(),
        [tensor_dist.get_immutable()],
        file_path=Path("./test_svg_amd_shape_b.svg"),
    )


fn test_svg_amd_shape_d() raises:
    # amd tensor core a matrix fragment
    alias layout = Layout.row_major(16, 16)
    var stack = InlineArray[Float32, layout.size()](uninitialized=True)
    var tensor = LayoutTensor[DType.float32, layout](stack)
    var tensor_dist = tensor.vectorize[4, 1]().distribute[
        Layout.row_major(4, 16)
    ](10)
    var tensor_dist2 = tensor.vectorize[4, 1]().distribute[
        Layout.row_major(4, 16)
    ](11)
    print_svg(
        tensor.get_immutable(),
        [tensor_dist.get_immutable(), tensor_dist2.get_immutable()],
        file_path=Path("./test_svg_amd_shape_d.svg"),
    )


fn test_svg_wgmma_shape() raises:
    # wgmma tensor core a matrix fragment
    alias layout = Layout(
        IntTuple(IntTuple(8, 8), IntTuple(8, 2)),
        IntTuple(IntTuple(8, 64), IntTuple(1, 512)),
    )
    var stack = InlineArray[Float32, layout.size()](uninitialized=True)

    var tensor = LayoutTensor[DType.float32, layout](stack)
    var tensor_dist = tensor.vectorize[1, 1]().distribute[
        Layout.col_major(8, 4)
    ](0)
    var tensor_dist2 = tensor.vectorize[1, 1]().distribute[
        Layout.col_major(8, 4)
    ](3)

    fn color_map(t: Int, v: Int) -> String:
        colors = [
            StaticString("red"),
            StaticString("blue"),
            StaticString("green"),
            StaticString("yellow"),
            StaticString("purple"),
            StaticString("orange"),
            StaticString("pink"),
            StaticString("brown"),
            StaticString("gray"),
            StaticString("black"),
            StaticString("white"),
        ]
        return String(colors[t])

    print_svg(
        tensor.get_immutable(),
        List[__type_of(tensor_dist.get_immutable())](
            tensor_dist.get_immutable(), tensor_dist2.get_immutable()
        ),
        color_map,
        file_path=Path("./test_svg_wgmma_shape.svg"),
    )


fn test_svg_swizzle() raises:
    alias layout = Layout.row_major(8, 8)
    var stack = InlineArray[Float32, layout.size()](uninitialized=True)
    alias swizzle = Swizzle(3, 0, 3)
    var tensor = LayoutTensor[DType.float32, layout](stack)

    # the figure generated here is identical to
    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/async-warpgroup-smem-layout-128B-k.png
    fn color_map(t: Int, v: Int) -> String:
        var colors = [
            StaticString("blue"),
            StaticString("green"),
            StaticString("yellow"),
            StaticString("red"),
            StaticString("lightblue"),
            StaticString("lightgreen"),
            StaticString("lightyellow"),
            StaticString("salmon"),  # lighter variant of red
        ]
        return String(colors[t % len(colors)])

    print_svg[swizzle](
        tensor.get_immutable(),
        List[__type_of(tensor.get_immutable())](),
        color_map=color_map,
        file_path=Path("./test_svg_swizzle.svg"),
    )


def main():
    test_svg_nvidia_shape()
    test_svg_nvidia_tile()
    test_svg_nvidia_tile_memory_bank()
    test_svg_amd_shape_a()
    test_svg_amd_shape_b()
    test_svg_amd_shape_d()
    test_svg_wgmma_shape()
    test_svg_swizzle()
