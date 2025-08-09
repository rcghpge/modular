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
# REQUIRES: NVIDIA-GPU

# RUN: %mojo-build %s -o %t
# RUN: %mpirun -n 1 %t

from gpu.host import DeviceContext, DeviceBuffer
from shmem import *
from testing import assert_equal
from pathlib import Path
from os.path import dirname
from sys.param_env import env_get_string


fn simple_shift_kernel(destination: UnsafePointer[Int32]):
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + 1) % npes

    shmem_int_p(destination, mype, peer)


def main():
    var msg = Int32(0)

    shmem_init()
    var mype_node = shmem_team_my_pe(SHMEM_TEAM_MODE)
    var ctx = DeviceContext(device_id=mype_node)

    var destination = shmem_malloc[DType.int32](1)

    # Compile the function and inject shmem into the module
    var func = ctx.compile_function[simple_shift_kernel]()
    shmem_module_init(func)

    ctx.enqueue_function(func, destination, grid_dim=1, block_dim=1)
    shmem_barrier_all_on_stream(ctx)
    DeviceBuffer(ctx, destination, 1, owning=False).enqueue_copy_to(
        UnsafePointer(to=msg)
    )

    ctx.synchronize()
    print("PE:", mype_node, "received message:", msg)

    shmem_free(destination)
    shmem_finalize()
