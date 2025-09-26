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

from gpu import block_idx, grid_dim
from gpu.host import DeviceBuffer, DeviceContext
from gpu.memory import (
    _GPUAddressSpace,
    async_copy_commit_group,
    async_copy_wait_group,
)
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import Layout, LayoutTensor
from testing import assert_true


def test_copy_dram_to_sram_async(ctx: DeviceContext):
    print("== test_copy_dram_to_sram_async")
    alias tensor_layout = Layout.row_major(4, 16)
    var tensor = ManagedLayoutTensor[DType.float32, tensor_layout](ctx)
    arange(tensor.tensor())

    var check_state = True

    fn copy_to_sram_test_kernel[
        layout: Layout,
    ](
        dram_tensor: LayoutTensor[DType.float32, layout, MutableAnyOrigin],
        flag: UnsafePointer[Scalar[DType.bool]],
    ):
        var dram_tile = dram_tensor.tile[4, 4](0, block_idx.x)
        var sram_tensor = LayoutTensor[
            DType.float32,
            Layout.row_major(4, 4),
            MutableAnyOrigin,
            address_space = _GPUAddressSpace.SHARED,
        ].stack_allocation()
        sram_tensor.copy_from_async(dram_tile)

        async_copy_commit_group()
        async_copy_wait_group(0)

        var col_offset = block_idx.x * 4

        for r in range(4):
            for c in range(4):
                if sram_tensor[r, c] != r * 16 + col_offset + c:
                    flag[] = False

    alias kernel = copy_to_sram_test_kernel[tensor_layout]
    var ptr = UnsafePointer(to=check_state).bitcast[Scalar[DType.bool]]()
    ctx.enqueue_function_checked[kernel, kernel](
        tensor.device_tensor(),
        DeviceBuffer[DType.bool](
            ctx,
            rebind[UnsafePointer[Scalar[DType.bool]]](ptr),
            1,
            owning=False,
        ),
        grid_dim=(4),
        block_dim=(1),
    )
    assert_true(check_state, "Inconsistent values in shared memory")


def main():
    with DeviceContext() as ctx:
        test_copy_dram_to_sram_async(ctx)
