// RUN: mo-opt %s --hoist-param-exprs | FileCheck %s

// CHECK-LABEL: mo.graph @complex_expression
mo.graph @complex_expression<N, M>(%arg0: !mo.tensor<[N], f32>, %arg1: !mo.tensor<[M], f32>) -> (!mo.tensor<[add(N, M)], f32>) {
  %ax = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK: concat
  %1 = mo.concat[%ax:!mo.tensor<[],si64>](%arg0, %arg1):(!mo.tensor<[N], f32>, !mo.tensor<[M], f32>) -> !mo.tensor<[add(N, M)], f32>
  // CHECK: kgen.param.declare [[NAME:[a-zA-Z0-9_]+]] = <add
  // CHECK: rebind
  // CHECK: mo.mul
  // CHECK-NOT: add
  // CHECK: output
  %2 = mo.mul(%1, %1) :!mo.tensor<[add(N, M)], f32>
  mo.output %2 : !mo.tensor<[add(N, M)], f32>
}

// CHECK-LABEL: mo.graph @multiple_complex_expression
mo.graph @multiple_complex_expression<N, M>(%arg0: !mo.tensor<[N], f32>, %arg1: !mo.tensor<[M], f32>) -> (!mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32>) {
  %ax = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = mo.concat[%ax:!mo.tensor<[],si64>](%arg0, %arg1):(!mo.tensor<[N], f32>, !mo.tensor<[M], f32>) -> !mo.tensor<[add(N, M)], f32>
  %2 = mo.static.broadcast_to(%arg1):!mo.tensor<[M], f32> ->!mo.tensor<[N, M], f32>
  // CHECK: reshape
  %3 = mo.static.reshape(%2):!mo.tensor<[N, M], f32> ->!mo.tensor<[mul_nuw(N, M)], f32>
  // CHECK-DAG: kgen.param.declare [[NAME_MUL:[a-zA-Z0-9_]+]] = <mul_nuw
  // CHECK-DAG: kgen.param.declare [[NAME_ADD:[a-zA-Z0-9_]+]] = <add
  // CHECK: mo.concat
  // CHECK-NOT: mul_nuw
  // CHECK-NOT: add
  // CHECK-DAG: [[NAME_MUL]]
  // CHECK-DAG: [[NAME_ADD]]
  // CHECK: {{output|rebind}}
  %4 = mo.concat[%ax:!mo.tensor<[],si64>](%1, %3) :(!mo.tensor<[add(N,M)], f32>, !mo.tensor<[mul_nuw(N,M)], f32>) -> !mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32>
  mo.output %4 : !mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32>
}

// CHECK-LABEL: mo.graph @no_rewrite_decl
mo.graph @no_rewrite_decl<N, M>(%arg0: !mo.tensor<[N], f32>, %arg1: !mo.tensor<[M], f32>) -> (!mo.tensor<[add(N, M)], f32>) {
  %ax = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = mo.concat[%ax:!mo.tensor<[],si64>](%arg0, %arg1):(!mo.tensor<[N], f32>, !mo.tensor<[M], f32>) -> !mo.tensor<[add(N, M)], f32>
  %2 = mo.static.reshape(%1):!mo.tensor<[add(N,M)], f32>->!mo.tensor<[sum], f32>
  // CHECK: reshape
  // CHECK: kgen.param.declare sum = <add
  // CHECK-NOT: kgen.param.declare
  // CHECK: mo.mul
  kgen.param.declare sum = <add(N, M)>
  %3 = mo.mul(%2, %2) :!mo.tensor<[sum], f32>
  %4 = mo.static.reshape(%3):!mo.tensor<[sum], f32>->!mo.tensor<[add(N, M)], f32>
  mo.output %4 : !mo.tensor<[add(N, M)], f32>
}
