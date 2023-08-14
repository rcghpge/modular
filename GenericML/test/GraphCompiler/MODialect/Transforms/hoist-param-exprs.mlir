// RUN: mo-opt %s --hoist-param-exprs | FileCheck %s

// CHECK-LABEL: mo.graph @complex_expression
mo.graph @complex_expression<N, M>(%arg0: !mo.tensor<[N], f32>, %arg1: !mo.tensor<[M], f32>) -> (!mo.tensor<[add(N, M)], f32>) {
  // CHECK-NEXT: mo.constant
  // CHECK-NEXT: mo.concat
  %ax = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = mo.concat[%ax:!mo.tensor<[],si64>](%arg0, %arg1):(!mo.tensor<[N], f32>, !mo.tensor<[M], f32>) -> !mo.tensor<[add(N, M)], f32>
  // CHECK-NEXT: kgen.param.declare [[NAME:[a-zA-Z0-9_]+]] = <add
  // CHECK-NEXT: rebind
  // CHECK-NEXT: mo.mul({{.*}}) : !mo.tensor<[[[NAME]]], f32>
  // CHECK-NEXT: mo.output{{.*}} : !mo.tensor<[add({{.*}})], f32>
  %2 = mo.mul(%1, %1) :!mo.tensor<[add(N, M)], f32>
  mo.output %2 : !mo.tensor<[add(N, M)], f32>
}

// CHECK-LABEL: mo.graph @multiple_complex_expression
mo.graph @multiple_complex_expression<N, M>(%arg0: !mo.tensor<[N], f32>, %arg1: !mo.tensor<[M], f32>) -> (!mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32>) {
  // CHECK-NEXT: mo.constant
  // CHECK-NEXT: mo.concat
  // CHECK-NEXT: mo.static.broadcast_to
  // CHECK-NEXT: mo.static.reshape
  %ax = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = mo.concat[%ax:!mo.tensor<[],si64>](%arg0, %arg1):(!mo.tensor<[N], f32>, !mo.tensor<[M], f32>) -> !mo.tensor<[add(N, M)], f32>
  %2 = mo.static.broadcast_to(%arg1):!mo.tensor<[M], f32> ->!mo.tensor<[N, M], f32>
  %3 = mo.static.reshape(%2):!mo.tensor<[N, M], f32> ->!mo.tensor<[mul_nuw(N, M)], f32>
  // CHECK-DAG: kgen.param.declare [[NAME_ADD:[a-zA-Z0-9_]+]] = <add
  // CHECK-DAG: kgen.param.declare [[NAME_MUL:[a-zA-Z0-9_]+]] = <mul_nuw
  // CHECK-DAG: kgen.param.declare [[NAME_ADD_MUL:[a-zA-Z0-9_]+]] = <add(mul_nuw
  // CHECK-NEXT: mo.rebind
  // CHECK-NEXT: mo.concat{{.*}} : (!mo.tensor<[[[NAME_ADD]]], f32>, !mo.tensor<[[[NAME_MUL]]], f32>) -> !mo.tensor<[[[NAME_ADD_MUL]]], f32>
  // CHECK-NEXT: mo.output{{.*}} : !mo.tensor<[add(mul_nuw{{.*}})], f32>
  %4 = mo.concat[%ax:!mo.tensor<[],si64>](%1, %3) :(!mo.tensor<[add(N,M)], f32>, !mo.tensor<[mul_nuw(N,M)], f32>) -> !mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32>
  mo.output %4 : !mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32>
}

// CHECK-LABEL: mo.graph @no_rewrite_decl
mo.graph @no_rewrite_decl<N, M>(%arg0: !mo.tensor<[N], f32>, %arg1: !mo.tensor<[M], f32>) -> (!mo.tensor<[add(N, M)], f32>) {
  // CHECK-NEXT: mo.constant
  // CHECK-NEXT: mo.concat
  %ax = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = mo.concat[%ax:!mo.tensor<[],si64>](%arg0, %arg1):(!mo.tensor<[N], f32>, !mo.tensor<[M], f32>) -> !mo.tensor<[add(N, M)], f32>
  %2 = mo.static.reshape(%1):!mo.tensor<[add(N, M)], f32>->!mo.tensor<[sum], f32>
  // CHECK-NEXT: kgen.param.declare [[SUM:.*]] = <add
  // CHECK-NEXT: mo.mul({{.*}}) : !mo.tensor<[[[SUM]]], f32>
  kgen.param.declare sum = <add(N, M)>
  %3 = mo.mul(%2, %2) :!mo.tensor<[sum], f32>
  // CHECK-NEXT: mo.rebind
  %4 = mo.static.reshape(%3):!mo.tensor<[sum], f32>->!mo.tensor<[add(N, M)], f32>
  // CHECK-NEXT: mo.output{{.*}} : !mo.tensor<[add({{.*}})], f32>
  mo.output %4 : !mo.tensor<[add(N, M)], f32>
}
