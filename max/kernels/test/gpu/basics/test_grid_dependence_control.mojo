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

from gpu.grid_controls import _SUPPORT_PDL_LAUNCH as SUPPORT_PDL_LAUNCH
from gpu.grid_controls import (
    PDL,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from gpu.host.compile import _compile_code
from gpu.host import get_gpu_target
from testing import assert_true


fn control_dep_grids_kernel():
    wait_on_dependent_grids()
    launch_dependent_grids()


# CHECK-LABEL: test_grid_control_primitives
# CHECK: griddepcontrol.wait
# CHECK: griddepcontrol.launch_dependents
# CHECK: griddepcontrol.wait
# CHECK: griddepcontrol.launch_dependents
def test_grid_control_primitives():
    print("== test_grid_control_primitives")
    assert_true(SUPPORT_PDL_LAUNCH)
    print(
        _compile_code[
            control_dep_grids_kernel,
            emission_kind="asm",
            target = get_gpu_target["sm_90"](),
        ]()
    )
    print(
        _compile_code[
            control_dep_grids_kernel,
            emission_kind="asm",
            target = get_gpu_target["sm_90a"](),
        ]()
    )


fn control_dep_grids_kernel_context():
    with PDL():
        pass


# CHECK-LABEL: test_grid_control_primitives_context
# CHECK: griddepcontrol.wait
# CHECK: griddepcontrol.launch_dependents
def test_grid_control_primitives_context():
    print("== test_grid_control_primitives_context")
    print(
        _compile_code[
            control_dep_grids_kernel_context,
            emission_kind="asm",
            target = get_gpu_target["sm_90a"](),
        ]()
    )


def main():
    test_grid_control_primitives()
    test_grid_control_primitives_context()
