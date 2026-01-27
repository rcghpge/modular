// RUN: mo-opt --mo-resolve-unknown-parameters --mo-legalize-rmo-operators --mo-shape-inference %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: @top_k
mo.graph @top_k<dim0, dim1>(%arg0: !mo.tensor<[dim0, dim1], si64, gpu:0>) -> (!mo.tensor<[dim0, 3], si64, gpu:0>, !mo.tensor<[dim0, 3], si64, gpu:0>) {
  // CHECK-DAG: [[K:%.+]] = mo.constant {value = {{.*}}3{{.*}}si64{{.*}}} : !mo.tensor<[], si64>
  // CHECK-DAG: [[AXIS:%.+]] = mo.constant {value = {{.*}}1{{.*}}si64{{.*}}} : !mo.tensor<[], si64>
  // CHECK-DAG: [[SORTED:%.+]] = mo.constant {value = {{.*}}1{{.*}}i8{{.*}}} : !mo.tensor<[], bool>
  // CHECK: mo.top_k(%arg0, [[K]], [[AXIS]], [[SORTED]])
  %values, %indices = rmo.top_k(%arg0) {k = 3 : si64, axis = 1 : si64} : (!mo.tensor<[dim0, dim1], si64, gpu:0>) -> (!mo.tensor<[dim0, 3], si64, gpu:0>, !mo.tensor<[dim0, 3], si64, gpu:0>)
  mo.output %values, %indices : !mo.tensor<[dim0, 3], si64, gpu:0>, !mo.tensor<[dim0, 3], si64, gpu:0>
}
