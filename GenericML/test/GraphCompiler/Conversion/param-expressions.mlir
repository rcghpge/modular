// RUN: mo-opt %s --mo-to-primitives | FileCheck %s --enable-var-scope

// CHECK-LABEL: mgp.model @add_expressions : mof
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:[^:]*]]: !mgp.context,
// CHECK: %[[ARG2:[^:]*]]: !mgp.tensor<?x?xf32>)
// CHECK:     %[[SPEC:[^:]*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <?x?xf32>
// CHECK:     %[[DIM0:[^:]*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <?x?xf32>
// CHECK:     %[[DIM1:[^:]*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <?x?xf32>
// CHECK-DAG: %[[VAL1:[^:]*]] = index.add %[[DIM0]], %[[DIM1]]
// CHECK-DAG: %[[VAL2:[^:]*]] = index.constant 2
// CHECK:     %[[VAL3:[^:]*]] = index.add %[[VAL1]], %[[VAL2]]
mo.graph @add_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  kgen.param.declare A0 = <add(X, Y)>
  kgen.param.declare A1 = <add(Y, X)>
  kgen.param.declare A2 = <add(A0, 2)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @sub_expressions : mof
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:[^:]*]]: !mgp.context,
// CHECK: %[[ARG2:[^:]*]]: !mgp.tensor<?x?xf32>)
// CHECK:     %[[SPEC:[^:]*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <?x?xf32>
// CHECK:     %[[DIM0:[^:]*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <?x?xf32>
// CHECK:     %[[DIM1:[^:]*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <?x?xf32>
// CHECK-DAG: %[[VAL1:[^:]*]] = index.constant -1
// CHECK-DAG: %[[VAL2:[^:]*]] = index.mul %[[DIM1]], %[[VAL1]]
// CHECK-DAG: %[[VAL3:[^:]*]] = index.add %[[VAL2]], %[[DIM0]]
// CHECK-DAG: %[[VAL4:[^:]*]] = index.constant -3
// CHECK-DAG: %[[VAL5:[^:]*]] = index.add %[[DIM0]], %[[VAL4]]
mo.graph @sub_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  kgen.param.declare S0 = <sub(X, Y)>
  kgen.param.declare S1 = <sub(X, 3)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @mul_expressions : mof
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:[^:]*]]: !mgp.context,
// CHECK: %[[ARG2:[^:]*]]: !mgp.tensor<?x?xf32>)
// CHECK:     %[[SPEC:[^:]*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <?x?xf32>
// CHECK:     %[[DIM0:[^:]*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <?x?xf32>
// CHECK:     %[[DIM1:[^:]*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <?x?xf32>
// CHECK-DAG: %[[VAL1:[^:]*]] = index.mul %[[DIM0]], %[[DIM1]]
// CHECK-DAG: %[[VAL2:[^:]*]] = index.constant 2
// CHECK:     %[[VAL3:[^:]*]] = index.mul %[[VAL1]], %[[VAL2]]
mo.graph @mul_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  kgen.param.declare M0 = <mul_nuw(X, Y)>
  kgen.param.declare M1 = <mul_nuw(Y, X)>
  kgen.param.declare M2 = <mul_nuw(M0, 2)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @mod_expressions : mof
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:[^:]*]]: !mgp.context,
// CHECK: %[[ARG2:[^:]*]]: !mgp.tensor<?x?xf32>)
// CHECK:     %[[SPEC:[^:]*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <?x?xf32>
// CHECK:     %[[DIM0:[^:]*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <?x?xf32>
// CHECK:     %[[DIM1:[^:]*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <?x?xf32>
// CHECK-DAG: %[[VAL1:[^:]*]] = index.rems %[[DIM0]], %[[DIM1]]
// CHECK-DAG: %[[VAL2:[^:]*]] = index.rems %[[DIM1]], %[[DIM0]]
// CHECK-DAG: %[[VAL3:[^:]*]] = index.constant 2
// CHECK:     %[[VAL4:[^:]*]] = index.rems %[[VAL1]], %[[VAL3]]
mo.graph @mod_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  kgen.param.declare M0 = <mod(X, Y)>
  kgen.param.declare M1 = <mod(Y, X)>
  kgen.param.declare M2 = <mod(M0, 2)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @eq_expressions : mof
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:[^:]*]]: !mgp.context,
// CHECK: %[[ARG2:[^:]*]]: !mgp.tensor<?x?x?xf32>)
// CHECK:     %[[SPEC:[^:]*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <?x?x?xf32>
// CHECK:     %[[DIM0:[^:]*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <?x?x?xf32>
// CHECK:     %[[DIM1:[^:]*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <?x?x?xf32>
// CHECK:     %[[DIM2:[^:]*]] = mgp.tensor_spec.get_dim[2] %[[SPEC]] : <?x?x?xf32>
// CHECK-DAG: %[[VAL1:[^:]*]] = index.cmp eq(%[[DIM0]], %[[DIM1]])
// CHECK-DAG: %[[VAL2:[^:]*]] = index.cmp eq(%[[DIM1]], %[[DIM2]])
// CHECK-DAG: %[[VAL3:[^:]*]] = index.bool.constant true
mo.graph @eq_expressions<X, Y, Z>(%arg0: !mo.tensor<[X, Y, Z], f32>) {
  kgen.param.declare E0: i1 = <eq(X, Y)>
  kgen.param.declare E1: i1 = <eq(Y, X)>
  kgen.param.declare E2: i1 = <eq(Y, Z)>
  kgen.param.declare E3: i1 = <eq(X, X)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @lt_expressions : mof
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:[^:]*]]: !mgp.context,
// CHECK: %[[ARG2:[^:]*]]: !mgp.tensor<?x?xf32>)
// CHECK:     %[[SPEC:[^:]*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <?x?xf32>
// CHECK:     %[[DIM0:[^:]*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <?x?xf32>
// CHECK:     %[[DIM1:[^:]*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <?x?xf32>
// CHECK-DAG: %[[VAL1:[^:]*]] = index.cmp slt(%[[DIM0]], %[[DIM1]])
// CHECK-DAG: %[[VAL2:[^:]*]] = index.cmp slt(%[[DIM1]], %[[DIM0]])
// CHECK-DAG: %[[VAL3:[^:]*]] = index.cmp slt(%[[DIM0]], %[[DIM0]])
mo.graph @lt_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  kgen.param.declare LT0: i1 = <lt(X, Y)>
  kgen.param.declare LT1: i1 = <lt(Y, X)>
  kgen.param.declare LT2: i1 = <lt(X, X)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @le_expressions : mof
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:[^:]*]]: !mgp.context,
// CHECK: %[[ARG2:[^:]*]]: !mgp.tensor<?x?xf32>)
// CHECK:     %[[SPEC:[^:]*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <?x?xf32>
// CHECK:     %[[DIM0:[^:]*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <?x?xf32>
// CHECK:     %[[DIM1:[^:]*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <?x?xf32>
// CHECK-DAG: %[[VAL1:[^:]*]] = index.cmp sle(%[[DIM0]], %[[DIM1]])
// CHECK-DAG: %[[VAL2:[^:]*]] = index.cmp sle(%[[DIM1]], %[[DIM0]])
// CHECK-DAG: %[[VAL3:[^:]*]] = index.cmp sle(%[[DIM0]], %[[DIM0]])
mo.graph @le_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  kgen.param.declare LE0: i1 = <le(X, Y)>
  kgen.param.declare LE1: i1 = <le(Y, X)>
  kgen.param.declare LE2: i1 = <le(X, X)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @cond_expression : mof
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:[^:]*]]: !mgp.context,
// CHECK: %[[ARG2:[^:]*]]: !mgp.tensor<?x?xf32>)
// CHECK:     %[[SPEC:[^:]*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <?x?xf32>
// CHECK:     %[[DIM0:[^:]*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <?x?xf32>
// CHECK:     %[[DIM1:[^:]*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <?x?xf32>
// CHECK-DAG: %[[BOOL:[^:]*]] = index.cmp sle(%[[DIM0]], %[[DIM1]])
// CHECK-DAG: %[[VAL1:[^:]*]] = pop.select %[[BOOL]], %[[DIM0]], %[[DIM1]]
// CHECK-DAG: %[[VAL2:[^:]*]] = pop.select %[[BOOL]], %[[DIM1]], %[[DIM0]]
// CHECK-DAG: %[[VAL3:[^:]*]] = index.constant 2
// CHECK:     %[[VAL4:[^:]*]] = pop.select %[[BOOL]], %[[VAL1]], %[[VAL3]]
mo.graph @cond_expression<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  kgen.param.declare B: i1 = <le(X, Y)>
  kgen.param.declare M0 = <cond(B, X, Y)>
  kgen.param.declare M1 = <cond(B, Y, X)>
  kgen.param.declare M2 = <cond(B, M0, 2)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @bool_expressions : mof
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:[^:]*]]: !mgp.context,
// CHECK: %[[ARG2:[^:]*]]: !mgp.tensor<?x?x?xf32>)
// CHECK:     %[[SPEC:[^:]*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <?x?x?xf32>
// CHECK:     %[[DIM0:[^:]*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <?x?x?xf32>
// CHECK:     %[[DIM1:[^:]*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <?x?x?xf32>
// CHECK:     %[[DIM2:[^:]*]] = mgp.tensor_spec.get_dim[2] %[[SPEC]] : <?x?x?xf32>
// CHECK-DAG: %[[VAL1:[^:]*]] = index.cmp eq(%[[DIM0]], %[[DIM1]])
// CHECK-DAG: %[[VAL2:[^:]*]] = index.cmp eq(%[[DIM1]], %[[DIM2]])
// CHECK-DAG: %[[VAL3:[^:]*]] = pop.or
// CHECK-DAG: %[[VAL4:[^:]*]] = pop.xor
// CHECK-DAG: %[[VAL5:[^:]*]] = pop.and
mo.graph @bool_expressions<X, Y, Z>(%arg0: !mo.tensor<[X, Y, Z], f32>) {
  kgen.param.declare E0: i1 = <eq(X, Y)>
  kgen.param.declare E1: i1 = <eq(Y, Z)>

  kgen.param.declare O0: i1 = <or(E0, E1)>
  kgen.param.declare O1: i1 = <or(E1, E0)>

  kgen.param.declare X0: i1 = <xor(E0, E1)>
  kgen.param.declare X1: i1 = <xor(E1, E0)>

  kgen.param.declare A0: i1 = <and(E0, E1)>
  kgen.param.declare A1: i1 = <and(E1, E0)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @complex_expressions : mof
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:[^:]*]]: !mgp.context,
// CHECK: %[[ARG2:[^:]*]]: !mgp.tensor<?x?x?xf32>)
// CHECK:     %[[SPEC:[^:]*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <?x?x?xf32>
// CHECK:     %[[DIM0:[^:]*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <?x?x?xf32>
// CHECK:     %[[DIM1:[^:]*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <?x?x?xf32>
// CHECK:     %[[DIM2:[^:]*]] = mgp.tensor_spec.get_dim[2] %[[SPEC]] : <?x?x?xf32>
// CHECK-DAG: %[[VAL1:[^:]*]] = index.mul %[[DIM0]], %[[DIM1]]
// CHECK-DAG: %[[VAL2:[^:]*]] = index.mul %[[DIM1]], %[[DIM2]]
// CHECK-DAG: %[[VAL3:[^:]*]] = index.add %[[VAL1]], %[[VAL2]]
mo.graph @complex_expressions<X, Y, Z>(%arg0: !mo.tensor<[X, Y, Z], f32>) {
  kgen.param.declare C0 = <add(mul_nuw(X, Y), mul_nuw(Y, Z))>
  mo.output
}

// CHECK-LABEL: mgp.model @variadic_exprs : mof
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:[^:]*]]: !mgp.context,
// CHECK: %[[ARG2:[^:]*]]: !mgp.tensor<?x?x?xf32>)
// CHECK: %[[SPEC:[^:]*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <?x?x?xf32>
// CHECK: %[[DIM0:[^:]*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <?x?x?xf32>
// CHECK: %[[DIM1:[^:]*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <?x?x?xf32>
// CHECK: %[[DIM2:[^:]*]] = mgp.tensor_spec.get_dim[2] %[[SPEC]] : <?x?x?xf32>
// CHECK: %[[VAL1:[^:]*]] = index.add %[[DIM0]], %[[DIM1]]
// CHECK: %[[VAL2:[^:]*]] = index.add %[[VAL1]], %[[DIM2]]
mo.graph @variadic_exprs<X, Y, Z>(%arg0: !mo.tensor<[X, Y, Z], f32>) {
  kgen.param.declare C0 = <add(X, Y, Z)>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @shape_attr : mof
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:[^:]*]]: !mgp.context,
// CHECK: %[[ARG2:[^:]*]]: !mgp.tensor<?x?x?xf32>)
// CHECK: %[[SPEC:[^:]*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <?x?x?xf32>
// CHECK: %[[DIM0:[^:]*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <?x?x?xf32>
// CHECK: %[[DIM1:[^:]*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <?x?x?xf32>
// CHECK: %[[DIM2:[^:]*]] = mgp.tensor_spec.get_dim[2] %[[SPEC]] : <?x?x?xf32>
// CHECK: %[[VAL1:[^:]*]] = mgp.tensor_spec.create %[[DIM0]], %[[DIM1]], %[[DIM2]] : !mgp.tensor_spec<?x?x?xsi64>
mo.graph @shape_attr<X, Y, Z>(%arg0: !mo.tensor<[X, Y, Z], f32>) {
  kgen.param.declare Sh: !mosh.ape = <[X, Y, Z]>
  mo.output
}

// -----

// CHECK-LABEL: mgp.model @param_expr_in_output_shape : mof
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:[^:]*]]: !mgp.context,
// CHECK: %[[INPUT:[^:]*]]: !mgp.tensor<?x4xf32>) -> (!mef.chain, !mgp.tensor<?x2xf32>) {
// CHECK: %[[SPEC:[^:]*]] = mgp.tensor.extract.tensor_spec %[[INPUT]] : <?x4xf32>
// CHECK: %[[D0:[^:]*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <?x4xf32>
// CHECK: %[[BUF:[^:]*]] = mgp.tensor.extract.buffer %[[INPUT]] : <?x4xf32>
// CHECK: %[[TWO:[^:]*]] = index.constant 2
// CHECK: %[[TWOd0:[^:]*]] = index.mul %[[D0]], %[[TWO]]
// CHECK: %[[OUT_SPEC:[^:]*]] = mgp.tensor_spec.create %[[TWOd0]] : !mgp.tensor_spec<?x2xf32>
// CHECK: %[[OUT_TENSOR:[^:]*]] = mgp.tensor.create %{{.*}}, %[[BUF]], %[[OUT_SPEC]]  : !mgp.tensor<?x2xf32>
// CHECK: mef.output %[[CHAIN]], %[[OUT_TENSOR]] : !mef.chain, !mgp.tensor<?x2xf32>
mo.graph @param_expr_in_output_shape<D0>(
  %arg0: !mo.tensor<[D0, 4], f32>
) -> !mo.tensor<[mul_nuw(D0, 2), 2], f32> {
  %res = mo.static.reshape(%arg0) : !mo.tensor<[D0, 4], f32> -> !mo.tensor<[mul_nuw(D0, 2), 2], f32>
  mo.output %res : !mo.tensor<[mul_nuw(D0, 2), 2], f32>
}

// -----

// CHECK-LABEL: mgp.model @shift_expressions : mof
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:[^:]*]]: !mgp.context,
// CHECK: %[[ARG2:[^:]*]]: !mgp.tensor<?x?x?xf32>)
// CHECK: %[[SPEC:[^:]*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <?x?x?xf32>
// CHECK: %[[DIM0:[^:]*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <?x?x?xf32>
// CHECK: %[[DIM1:[^:]*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <?x?x?xf32>
// CHECK: %[[DIM2:[^:]*]] = mgp.tensor_spec.get_dim[2] %[[SPEC]] : <?x?x?xf32>
// CHECK: %{{.*}} = index.shl %[[DIM0]], %[[DIM1]]
// CHECK: %{{.*}} = index.shrs %[[DIM1]], %[[DIM2]]
mo.graph @shift_expressions<X, Y, Z>(%arg0: !mo.tensor<[X, Y, Z], f32>) {
  kgen.param.declare SHL = <shl(X, Y)>
  kgen.param.declare SHR = <shr(Y, Z)>

  mo.output
}

// -----

// CHECK-LABEL: mgp.model @integer_types : mof
mo.graph @integer_types<X: si64, Y: si64>() {
  // CHECK: %{{.*}} = pop.add %{{.*}}, %{{.*}} : !pop.scalar<si64>
  kgen.param.declare ADD: si64 = <add(:si64 X, Y)>

  // CHECK: %{{.*}} = pop.mul %{{.*}}, %{{.*}} : !pop.scalar<si64>
  kgen.param.declare MUL: si64 = <mul(:si64 Y, X)>

  // CHECK: %{{.*}} = pop.shl %{{.*}}, %{{.*}} : !pop.scalar<si64>
  kgen.param.declare SHL: si64 = <shl(:si64 X, Y)>

  // CHECK: %{{.*}} = pop.shr %{{.*}}, %{{.*}} : !pop.scalar<si64>
  kgen.param.declare SHR: si64 = <shr(:si64 X, Y)>

  // CHECK: %{{.*}} = pop.div %{{.*}}, %{{.*}} : !pop.scalar<si64>
  kgen.param.declare DIV: si64 = <div(:si64 X, Y)>

  // CHECK: %{{.*}} = pop.cmp eq(%{{.*}}, %{{.*}}) : <1, si64>
  kgen.param.declare EQ: i1 = <eq(:si64 X, Y)>

  // CHECK: %{{.*}} = pop.cmp lt(%{{.*}}, %{{.*}}) : <1, si64>
  kgen.param.declare LT: i1 = <lt(:si64 X, Y)>

  // CHECK: %{{.*}} = pop.cmp le(%{{.*}}, %{{.*}}) : <1, si64>
  kgen.param.declare LE: i1 = <le(:si64 X, Y)>
  mo.output
}

