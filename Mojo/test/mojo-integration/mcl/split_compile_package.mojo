# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: _compile_code default target is get_gpu_target,
# COM: therefore this test requires gpu.
# REQUIRES: NVIDIA-GPU || AMD-GPU

# RUN: mkdir -p %t.split-compile
# RUN: mojo precompile %S/inputs/target_dep_package -o %t.split-compile/target_dep_package.mojoc
# RUN: mojo -I %t.split-compile %s | FileCheck %s

from std.sys.info import _current_target

from max.gpu.host.compile import _compile_code

from target_dep_package.module import target_dep_function


def use_precompiled_gpu_func(a: Pointer[Int8, MutAnyOrigin]):
    target_dep_function(a)


def main():
    # CHECK: {{st.global.b8|v_mov_b32_e32}}{{.*}}41
    print(_compile_code[target_dep_function]())

    var a = Int8(8)
    target_dep_function(Pointer(to=a).as_unsafe_any_origin())
    # CHECK: host 17
    print("host", a)

    # CHECK: {{st.global.b8|v_mov_b32_e32}}{{.*}}41
    print(_compile_code[use_precompiled_gpu_func]())
