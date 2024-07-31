// RUN: mo-opt %s --hoist-param-exprs | FileCheck %s

// CHECK-LABEL: mo.graph @complex_expression
mo.graph @complex_expression<N, M>(%arg0: !mo.tensor<[N], f32>, %arg1: !mo.tensor<[M], f32>) -> (!mo.tensor<[add(N, M)], f32>) {
  // CHECK-NEXT: mo.constant
  // CHECK-NEXT: mo.concat
  %ax = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = mo.concat[%ax:!mo.tensor<[],si64>](%arg0, %arg1):(!mo.tensor<[N], f32>, !mo.tensor<[M], f32>) -> !mo.tensor<[add(N, M)], f32>
  // CHECK-NEXT: kgen.param.declare [[NAME:[a-zA-Z0-9_]+]] = <add
  // CHECK-NEXT: mo.mul({{.*}}) : !mo.tensor<[[[NAME]]], f32>
  // CHECK-NEXT: rebind
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
  // CHECK-NEXT: mo.concat{{.*}} : (!mo.tensor<[[[NAME_ADD]]], f32>, !mo.tensor<[[[NAME_MUL]]], f32>) -> !mo.tensor<[[[NAME_ADD_MUL]]], f32>
  // CHECK-NEXT: mo.rebind
  // CHECK-NEXT: mo.output{{.*}} : !mo.tensor<[add(mul_nuw{{.*}})], f32>
  %4 = mo.concat[%ax:!mo.tensor<[],si64>](%1, %3) :(!mo.tensor<[add(N,M)], f32>, !mo.tensor<[mul_nuw(N,M)], f32>) -> !mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32>
  mo.output %4 : !mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32>
}

// CHECK-LABEL: mo.graph @scoped_multiple_complex_expression
mo.graph @scoped_multiple_complex_expression<N, M>(%arg0: !mo.tensor<[N], f32>, %arg1: !mo.tensor<[M], f32>, %arg2: !mo.tensor<[1], bool>) -> (!mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32>) {
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
  // CHECK-NEXT: mo.concat{{.*}} : (!mo.tensor<[[[NAME_ADD]]], f32>, !mo.tensor<[[[NAME_MUL]]], f32>) -> !mo.tensor<[[[NAME_ADD_MUL]]], f32>
  // CHECK-NEXT: mo.rebind
  %4 = mo.concat[%ax:!mo.tensor<[],si64>](%1, %3) :(!mo.tensor<[add(N,M)], f32>, !mo.tensor<[mul_nuw(N,M)], f32>) -> !mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32>
  %cond = mo.static.reshape(%arg2) : !mo.tensor<[1], bool> -> !mo.tensor<[], bool>
  %5 = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32> {
    // CHECK: kgen.param.declare [[NAME_ADD_MUL_SCOPED:[a-zA-Z0-9_]+]] = <add(mul_nuw
    // CHECK-NEXT: mo.rebind{{.*}} : !mo.tensor<[[[NAME_ADD_MUL]]], f32> -> !mo.tensor<[[[NAME_ADD_MUL_SCOPED]]], f32>
    %6 = mo.concat[%ax:!mo.tensor<[],si64>](%4) : (!mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32>) -> !mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32>
    mo.yield %6 : !mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32>
  } else {
    mo.yield %4 : !mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32>
  }
  // CHECK: mo.output{{.*}} : !mo.tensor<[add(mul_nuw{{.*}})], f32>
  mo.output %5 : !mo.tensor<[add(add(N, M), mul_nuw(N,M))], f32>
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

// CHECK-LABEL: mo.graph @rebind_output
// CHECK-SAME: [[D21:.*]], [[D22:.*]], [[D23:.*]], [[D24:.*]] -> [[D27:.*]]>([[VARG0:.*]]: !mo.tensor<[[[D21]], 12, [[D22]]], f32>, [[VARG1:.*]]: !mo.tensor<[[[D23]], [[D24]]], f32>) -> !mo.tensor<[[[D21]], 12, [[D27]]], f32> {
// [[V0:.*]] = mo.static.broadcast_to([[VARG0]]) : !mo.tensor<[[[D21]], 12, [[D22]]], f32> -> !mo.tensor<[[[D21]], 12, param_expr_1], f32>
// [[V1:.*]] = mo.static.broadcast_to([[VARG1]]) : !mo.tensor<[[[D23]], [[D24]]], f32> -> !mo.tensor<[[[D21]], 12, param_expr_1], f32>
// kgen.param.declare param_expr_1 = <cond(eq([[D22]], 1), [[D24]], [[D22]])>
// [[V2:.*]] = mo.add([[V0]], [[V1]]) : !mo.tensor<[[[D21]], 12, param_expr_1], f32>
// [[V3:.*]] = mo.rebind([[V2]]) : !mo.tensor<[[[D21]], 12, param_expr_1], f32> -> !mo.tensor<[[[D21]], 12, cond(eq([[D22]], 1), [[D24]], [[D22]])], f32>
// mo.output<cond(eq([[D22]], 1), [[D24]], [[D22]])> [[V3]] : !mo.tensor<[[[D21]], 12, cond(eq([[D22]], 1), [[D24]], [[D22]])], f32>
mo.graph @rebind_output<D21, D22, D23, D24 -> D27>(%arg0: !mo.tensor<[D21, 12, D22], f32>, %arg1: !mo.tensor<[D23, D24], f32>) -> !mo.tensor<[D21, 12, D27], f32> {
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[D21, 12, D22], f32> -> !mo.tensor<[D21, 12, cond(eq(D22, 1), D24, D22)], f32>
  %1 = mo.static.broadcast_to(%arg1) : !mo.tensor<[D23, D24], f32> -> !mo.tensor<[D21, 12, cond(eq(D22, 1), D24, D22)], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[D21, 12, cond(eq(D22, 1), D24, D22)], f32>
  mo.output<cond(eq(D22, 1), D24, D22)> %2 : !mo.tensor<[D21, 12, cond(eq(D22, 1), D24, D22)], f32>
}

// COM: mo.rebind can't used with an opaque type.
// CHECK-LABEL: mo.graph @dont_rebind_opaque
mo.graph @dont_rebind_opaque(%arg0: !mo.tensor<[?], f32>, %arg1: !mo.opaque<"Opaque">) -> (!mo.opaque<"Opaque">) {

  // CHECK-NEXT: mo.rebind
  %0 = mo.rebind<() -> D1>(%arg0) : !mo.tensor<[?], f32> -> !mo.tensor<[D1], f32>

  // CHECK-NEXT: mo.custom
  %1 = mo.custom {symbol = "custom0"}(%0) : (!mo.tensor<[D1], f32>) -> !mo.tensor<[mul_nuw(D1, 2)], f32>

  // CHECK-NEXT: kgen.param.declare [[MUL:.*]] = <mul_nuw

  // CHECK-NEXT: (!mo.tensor<[[[MUL]]], f32>, !mo.opaque<"Opaque">) -> !mo.opaque<"Opaque">
  %2 = mo.custom {symbol = "custom"}(%1, %arg1) : (!mo.tensor<[mul_nuw(D1, 2)], f32>, !mo.opaque<"Opaque">) -> !mo.opaque<"Opaque">

  // CHECK-NEXT: mo.output
  mo.output %2 : !mo.opaque<"Opaque">
}

// CHECK-LABEL: mo.graph @complex_expression_non_mo
mo.graph @complex_expression_non_mo<N, M>(%arg0: !mo.tensor<[N], f32>, %arg1: !mo.tensor<[M], f32>) -> (!mo.tensor<[add(N, M)], f32>) {
  // NOTE: this is a mogg.concat not a mo.concat
  // CHECK-NEXT: mogg.concat
  %1 = mogg.concat(%arg0, %arg1):(!mo.tensor<[N], f32>, !mo.tensor<[M], f32>) -> !mo.tensor<[add(N, M)], f32>
  // CHECK-NEXT: kgen.param.declare [[NAME:[a-zA-Z0-9_]+]] = <add
  // CHECK-NEXT: mo.mul({{.*}}) : !mo.tensor<[[[NAME]]], f32>
  // CHECK-NEXT: rebind
  // CHECK-NEXT: mo.output{{.*}} : !mo.tensor<[add({{.*}})], f32>
  %2 = mo.mul(%1, %1) :!mo.tensor<[add(N, M)], f32>
  mo.output %2 : !mo.tensor<[add(N, M)], f32>
}

// CHECK-LABEL: mo.graph @dont_outline_unknown_dims
mo.graph @dont_outline_unknown_dims<N, M>(%arg0: !mo.tensor<[N, ?], f32>, %arg1: !mo.tensor<[M, ?], f32>) -> (!mo.tensor<[add(N, M), ?], f32>) {
  // No question marks should dissapear (there are 8 in the current IR)
  // CHECK-COUNT-8: ?
  // COUNT-NOT: ?
  %ax = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = mo.concat[%ax:!mo.tensor<[],si64>](%arg0, %arg1):(!mo.tensor<[N, ?], f32>, !mo.tensor<[M, ?], f32>) -> !mo.tensor<[add(N, M), ?], f32>
  %2 = mo.mul(%1, %1) :!mo.tensor<[add(N, M), ?], f32>
  mo.output %2 : !mo.tensor<[add(N, M), ?], f32>
}
