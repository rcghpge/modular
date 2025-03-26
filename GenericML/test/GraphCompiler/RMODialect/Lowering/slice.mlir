// RUN: mo-opt --resolve-unknown-parameters --legalize-rmo-operators --shape-inference --verify-diagnostics %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: mo.graph @slice
mo.graph @slice<dim0>(%arg0: !mo.tensor<[dim0], f32>) -> !mo.tensor<[add(dim0, -1)], f32> attributes {argument_names = ["input0"], result_names = ["output0"]} {
  // CHECK: {{[[:space:]]}}mo.slice{{.*}}si64, cpu{{.*}}si64, cpu{{.*}}si64, cpu
  %1 = rmo.slice(%arg0) {starts = #mosh<ape[1]> : !mosh.ape, steps = #mosh<ape[1]> : !mosh.ape, stops = #mosh<ape[dim0]> : !mosh.ape} : (!mo.tensor<[dim0], f32>) -> !mo.tensor<[add(dim0, -1)], f32>
  mo.output %1 : !mo.tensor<[add(dim0, -1)], f32>
}
