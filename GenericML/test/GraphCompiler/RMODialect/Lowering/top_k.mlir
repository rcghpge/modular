// RUN: mo-opt --resolve-unknown-parameters --legalize-rmo-operators --shape-inference --verify-diagnostics %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: @top_k
mo.graph @top_k<dim0, dim1>(%arg0: !mo.tensor<[dim0, dim1], si64, gpu:0>) -> (!mo.tensor<[dim0, 3], si64, gpu:0>, !mo.tensor<[dim0, 3], si64, gpu:0>) {
  // CHECK-DAG: [[K:%.+]] = mo.constant {device = #M.device_ref<"cpu", 0>, value = {{.*}}3{{.*}}si64{{.*}}} : !mo.tensor<[], si64, cpu:0>
  // CHECK-DAG: [[AXIS:%.+]] = mo.constant {device = #M.device_ref<"cpu", 0>, value = {{.*}}1{{.*}}si64{{.*}}} : !mo.tensor<[], si64, cpu:0>
  // CHECK-DAG: [[SORTED:%.+]] = mo.constant {device = #M.device_ref<"cpu", 0>, value = {{.*}}1{{.*}}i8{{.*}}} : !mo.tensor<[], bool, cpu:0>
  // CHECK: mo.top_k(%arg0, [[K]], [[AXIS]], [[SORTED]])
  %values, %indices = rmo.top_k(%arg0) {k = 3 : si64, axis = 1 : si64} : (!mo.tensor<[dim0, dim1], si64, gpu:0>) -> (!mo.tensor<[dim0, 3], si64, gpu:0>, !mo.tensor<[dim0, 3], si64, gpu:0>)
  mo.output %values, %indices : !mo.tensor<[dim0, 3], si64, gpu:0>, !mo.tensor<[dim0, 3], si64, gpu:0>
}
