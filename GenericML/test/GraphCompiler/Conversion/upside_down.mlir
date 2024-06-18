// RUN: mo-opt -mo-to-mgp="fuse-jit-kernels=false skip-jit-compile-middle=true" %s | FileCheck %s

mo.graph @updside_down(
    %arg0: !mo.tensor<[256], f32>
) -> (!mo.tensor<[256], f32>) {
    %relu = mo.relu(%add) : !mo.tensor<[256], f32>
    %add = mo.add(%mul, %arg0) : !mo.tensor<[256], f32>
    %mul = mo.mul(%arg0, %arg0) : !mo.tensor<[256], f32>
    mo.output %relu : !mo.tensor<[256], f32>
}

// CHECK-LABEL: mgp.model @updside_down : mof init {
// CHECK: "mgp.generic.execute"{{.*}}dSummary = "{{.*}}mo.mul{{.*}}"
// CHECK: "mgp.generic.execute"{{.*}}dSummary = "{{.*}}mo.add{{.*}}"
// CHECK: "mgp.generic.execute"{{.*}}dSummary = "{{.*}}mo.relu{{.*}}"
