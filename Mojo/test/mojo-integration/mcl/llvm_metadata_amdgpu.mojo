# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: AMD-GPU
# RUN: mojo %s | FileCheck %s

from max.gpu import thread_idx
from max.gpu.host.compile import _compile_code
from std.utils import IndexList
from std.utils.static_tuple import StaticTuple
from max.gpu.host import get_gpu_target


@__llvm_metadata(`rocdl.waves_per_eu`=SIMDLength(3))
@__llvm_metadata(`rocdl.flat_work_group_size`=SIMDLength(128))
def my_kernel():
    pass


@__llvm_metadata(`rocdl.waves_per_eu`=SIMDLength(3))
@__llvm_metadata(`rocdl.waves_per_eu`=SIMDLength(4))
@__llvm_metadata(`rocdl.waves_per_eu`=SIMDLength(5))
@__llvm_metadata(`rocdl.waves_per_eu`=SIMDLength(6))
def duplicated_metadata():
    pass


# CHECK-DAG: define dso_local amdgpu_kernel void @{{.*}}my_kernel{{.*}}() #[[MY_KERNEL_ATTRS:.+]] {
# CHECK-DAG: attributes #[[MY_KERNEL_ATTRS]] = { {{.*}}"amdgpu-flat-work-group-size"="1, 128" "amdgpu-waves-per-eu"="3" {{.*}} }

# CHECK-DAG: define dso_local amdgpu_kernel void @{{.*}}duplicated{{.*}}() #[[DUPLICATE_METADATAL_ATTRS:.+]] {
# CHECK-DAG: attributes #[[DUPLICATE_METADATAL_ATTRS]] = { {{.*}}"amdgpu-waves-per-eu"="6" {{.*}} }


def main():
    print(
        _compile_code[
            my_kernel,
            emission_kind="llvm",
            target=get_gpu_target["mi355x"](),
        ]()
    )
    print(
        _compile_code[
            duplicated_metadata,
            emission_kind="llvm",
            target=get_gpu_target["mi355x"](),
        ]()
    )
