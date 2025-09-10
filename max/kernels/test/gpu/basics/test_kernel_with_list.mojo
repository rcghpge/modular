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

from gpu.host import DeviceContext


fn kernel_with_list(res: UnsafePointer[Float32]):
    var list = List[Float32](10)
    for i in range(4):
        list.append(i + 1)
    res[] = list[0] * list[1] + list[2] * list[3]


fn test_kernel_with_list(ctx: DeviceContext) raises:
    print("== test_kernel_with_list")
    var res_device = ctx.enqueue_create_buffer[DType.float32](1)
    _ = res_device.enqueue_fill(0)
    # CHECK: call.uni
    # CHECK: malloc,
    # CHECK: (
    # CHECK: param0
    # CHECK: );
    # CHECK: call.uni
    # CHECK: free,
    # CHECK: (
    # CHECK: param0
    # CHECK: );
    alias kernel = kernel_with_list
    ctx.enqueue_function_checked[kernel, kernel, dump_asm=True](
        res_device, block_dim=(1), grid_dim=(1)
    )
    with res_device.map_to_host() as res_host:
        # CHECK: 16.0
        print("Res=", res_host[0])


def main():
    with DeviceContext() as ctx:
        test_kernel_with_list(ctx)
