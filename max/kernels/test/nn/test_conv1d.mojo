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

from math import ceildiv, isclose
from random import rand
from sys.info import simd_width_of

from itertools import product
from layout import Layout, LayoutTensor, RuntimeLayout
from nn.conv import (
    ConvDirectNHWC,
    ConvInfoStatic,
    Naive2dConvolution,
    pack_conv_filter_shape,
    pack_filter,
)
from nn.conv_utils import (
    ConvShape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
)

from utils.index import Index, IndexList

alias simd_size: Int = simd_width_of[DType.float32]()


# CHECK-LABEL: test_conv1d
fn test[
    dtype: DType, filter_packed: Bool
](
    N: Int,
    W: Int,
    C: Int,
    S: Int,
    F: Int,
    stride: Int,
    dilation: Int,
    pad_w: IndexList[2],
    num_groups: Int,
) raises:
    print("== test_conv1d")

    var WO = (W + pad_w[0] + pad_w[1] - dilation * (S - 1) - 1) // stride + 1
    alias HO = 1
    alias H = 1
    alias R = 1

    var conv_shape = ConvShape[1](
        n=N,
        input_dims=Index(W),
        output_dims=Index(WO),
        filter_dims=Index(S),
        c=C,
        f=F,
        stride=stride,
        dilation=dilation,
        pad_d=Index(0, 0),
        pad_h=Index(0, 0),
        pad_w=pad_w,
        num_groups=num_groups,
    )

    var C_per_group = C // num_groups

    var input_ptr = UnsafePointer[Scalar[dtype]].alloc(N * W * C)
    var filter_ptr = UnsafePointer[Scalar[dtype]].alloc(S * C_per_group * F)
    var output_ptr = UnsafePointer[Scalar[dtype]].alloc(N * WO * F)
    var output_ref_ptr = UnsafePointer[Scalar[dtype]].alloc(N * WO * F)

    rand[dtype](input_ptr, N * W * C)
    rand[dtype](filter_ptr, S * C_per_group * F)

    # Find the tile size used in packing.
    alias micro_kernel_height = get_direct_conv_micro_kernel_height()
    alias micro_kernel_width = get_direct_conv_micro_kernel_width()

    var micro_kernel_f_size = get_direct_conv_micro_kernel_width() * simd_size
    var rounded_F = ceildiv(F, micro_kernel_f_size) * micro_kernel_f_size

    # Buffers for direct conv.
    alias layout_3d = Layout.row_major[3]()
    alias layout_4d = Layout.row_major[4]()
    var input = LayoutTensor[dtype, layout_3d](
        input_ptr, RuntimeLayout[layout_3d].row_major(Index(N, W, C))
    )
    var filter = LayoutTensor[dtype, layout_3d](
        filter_ptr, RuntimeLayout[layout_3d].row_major(Index(S, C_per_group, F))
    )
    var packed_filter_shape = pack_conv_filter_shape[False](filter, num_groups)

    var packed_filter_ptr = UnsafePointer[Scalar[dtype]].alloc(
        packed_filter_shape.flattened_length()
    )
    var packed_filter = LayoutTensor[dtype, layout_4d](
        packed_filter_ptr,
        RuntimeLayout[layout_4d].row_major(packed_filter_shape),
    )
    var output = LayoutTensor[dtype, layout_3d](
        output_ptr, RuntimeLayout[layout_3d].row_major(Index(N, WO, F))
    )

    @parameter
    if filter_packed:
        pack_filter(filter, packed_filter, num_groups)

    # Reference: naive conv
    Naive2dConvolution[
        dtype,
        dtype,
        dtype,
    ].run(
        output_ref_ptr,
        input_ptr,
        filter_ptr,
        Index(N, 1, 1, WO, F),  # output shape
        Index(N, 1, 1, W, C),  # input shape
        Index(1, 1, S, C // num_groups, F),  # filter shape
        Index(0, 0),  #  pad_d
        Index(0, 0),  #  pad_h
        pad_w,
        Index(1, 1, stride),
        Index(1, 1, dilation),
        num_groups,
    )

    # Test direct conv
    alias conv_attr = ConvInfoStatic[1]()

    @parameter
    if filter_packed:
        ConvDirectNHWC[
            layout_3d,
            layout_4d,
            layout_3d,
            _,
            _,
            _,
            dtype,
            dtype,
            dtype,
            True,
            conv_attr,
        ].run(output, input, packed_filter, conv_shape)
    else:
        ConvDirectNHWC[
            layout_3d,
            layout_3d,
            layout_3d,
            _,
            _,
            _,
            dtype,
            dtype,
            dtype,
            False,
            conv_attr,
        ].run(output, input, filter, conv_shape)

    input_ptr.free()
    filter_ptr.free()
    packed_filter_ptr.free()

    # Check results, return on the first failed comparison.
    var idx = 0
    for n, wo, f in product(range(N), range(WO), range(F)):
        if not isclose(
            output_ref_ptr[idx],
            output_ptr[idx],
            atol=1e-4,  # absolute error tolerance
            rtol=1e-4,  # relative error tolerance
        ):
            print("Input shape NWC: ", Index(N, W, C))
            print("filter shape SCF: ", Index(S, C, F))
            print("filter packed", filter_packed)
            print("num groups", num_groups)
            print("Test failed at index: ", Index(n, wo, f))
            print("Golden value: ", output_ref_ptr[idx])
            print("Actual value: ", output_ptr[idx])
            output_ptr.free()
            output_ref_ptr.free()
            return
        idx += 1

    output_ptr.free()
    output_ref_ptr.free()

    # CHECK: Succeed
    print("Succeed")


def main():
    alias dtype = DType.float32
    # No packing or padding.
    test[dtype, False](1, 5, 1, 4, 4, 2, 1, Index(0, 0), 1)
    test[dtype, False](1, 12, 12, 3, 64, 1, 1, Index(0, 0), 1)
    test[dtype, False](1, 13, 16, 5, 64, 1, 1, Index(0, 0), 1)
    test[dtype, False](1, 7, 32, 3, 16, 2, 1, Index(0, 0), 1)

    # Pre-packed test w/o padding.
    test[dtype, True](1, 17, 16, 5, 64, 3, 1, Index(0, 0), 1)
    test[dtype, True](5, 12, 8, 3, 64, 2, 1, Index(0, 0), 1)
    test[dtype, True](1, 7, 11, 3, 192, 3, 1, Index(0, 0), 1)
    test[dtype, True](1, 7, 5, 5, 256, 2, 1, Index(0, 0), 1)

    # No packing, w/ padding, and F not multiple of simd_size.
    test[dtype, False](1, 5, 3, 3, 1, 1, 1, Index(1, 1), 1)
    test[dtype, False](2, 11, 5, 3, 2, 1, 1, Index(1, 1), 1)
    test[dtype, False](1, 12, 6, 5, 3, 3, 1, Index(2, 2), 1)
    test[dtype, False](1, 7, 1, 4, 3, 1, 1, Index(2, 1), 1)
    test[dtype, False](1, 5, 2, 3, 6, 2, 1, Index(1, 1), 1)

    # Pre-packed, F not multiple of simd_size
    test[dtype, True](1, 5, 2, 3, 7, 1, 1, Index(0, 0), 1)
    test[dtype, True](1, 7, 2, 3, 42, 2, 1, Index(0, 0), 1)
    test[dtype, True](1, 23, 17, 3, 90, 1, 1, Index(0, 0), 1)
    test[dtype, True](1, 11, 2, 5, 7, 1, 1, Index(2, 2), 1)
    test[dtype, True](1, 9, 2, 3, 42, 2, 1, Index(1, 1), 1)
    test[dtype, True](1, 7, 17, 5, 90, 2, 1, Index(2, 2), 1)

    # Grouped conv tests
    test[dtype, True](1, 1, 2, 1, 2, 1, 1, Index(0, 0), 2)
    test[dtype, True](1, 1, 25, 1, 25, 1, 1, Index(0, 0), 5)
    test[dtype, True](1, 1, 16, 1, 4, 1, 1, Index(0, 0), 2)
    test[dtype, True](1, 1, 32, 1, 20, 1, 1, Index(0, 0), 2)
    test[dtype, True](1, 1, 34, 1, 40, 1, 1, Index(0, 0), 2)
    test[dtype, True](1, 13, 16, 5, 64, 2, 1, Index(0, 0), 4)
    test[dtype, True](1, 1, 2, 1, 2, 1, 1, Index(1, 1), 2)
    test[dtype, True](1, 3, 18, 3, 18, 1, 1, Index(0, 0), 3)
    test[dtype, True](1, 7, 33, 5, 90, 2, 1, Index(2, 2), 3)
    test[dtype, True](3, 17, 36, 5, 93, 2, 1, Index(2, 2), 3)
    test[dtype, True](1, 17, 36, 6, 198, 3, 1, Index(3, 2), 2)

    # Depthwise conv.
    test[dtype, True](1, 7, 33, 5, 66, 2, 1, Index(2, 2), 33)

    # WavLM and Wav2Vec2 shapes.
    test[dtype, True](2, 16000, 1, 10, 512, 5, 1, Index(0, 0), 1)
    test[dtype, True](2, 3199, 512, 3, 512, 2, 1, Index(0, 0), 1)
    test[dtype, True](2, 1599, 512, 3, 512, 2, 1, Index(0, 0), 1)
    test[dtype, True](2, 799, 512, 3, 512, 2, 1, Index(0, 0), 1)
    test[dtype, True](2, 399, 512, 3, 512, 2, 1, Index(0, 0), 1)
    test[dtype, True](2, 199, 512, 2, 512, 2, 1, Index(0, 0), 1)
    test[dtype, True](2, 99, 512, 2, 512, 2, 1, Index(0, 0), 1)
    test[dtype, True](2, 49, 1024, 128, 1024, 1, 1, Index(64, 64), 16)
