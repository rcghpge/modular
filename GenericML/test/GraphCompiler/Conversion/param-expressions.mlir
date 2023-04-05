// RUN: mo-opt %s --mo-to-mogg --mo-to-primitives --allow-unregistered-dialect | FileCheck %s

// CHECK: mgp.model @add_expressions : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.cpu.tensor<?x?xf32>)
// CHECK:     %[[BUF:.*]], %[[SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x?xf32>
// CHECK:     %[[DIMS:.*]]:2 = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC]] : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG: %[[VAL1:.*]] = mip.add %[[DIMS]]#0, %[[DIMS]]#1
// CHECK-DAG: %[[VAL2:.*]] = mip.constant.index 2
// CHECK:     %[[VAL3:.*]] = mip.add %[[VAL1]], %[[VAL2]]
mo.graph @add_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  kgen.param.declare A0 = <add(X, Y)>
  kgen.param.declare A1 = <add(Y, X)>
  kgen.param.declare A2 = <add(A0, 2)>
  mo.output
}

// CHECK: mgp.model @sub_expressions : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.cpu.tensor<?x?xf32>)
// CHECK:     %[[BUF:.*]], %[[SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x?xf32>
// CHECK:     %[[DIMS:.*]]:2 = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC]] : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG: %[[VAL1:.*]] = mip.constant.index -1
// CHECK-DAG: %[[VAL2:.*]] = mip.mul %[[DIMS]]#1, %1
// CHECK-DAG: %[[VAL3:.*]] = mip.add %[[VAL2]], %[[DIMS]]#0
// CHECK-DAG: %[[VAL4:.*]] = mip.constant.index -3
// CHECK-DAG: %[[VAL5:.*]] = mip.add %[[DIMS]]#0, %[[VAL4]]
mo.graph @sub_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  kgen.param.declare S0 = <sub(X, Y)>
  kgen.param.declare S1 = <sub(X, 3)>
  mo.output
}

// CHECK: mgp.model @mul_expressions : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.cpu.tensor<?x?xf32>)
// CHECK:     %[[BUF:.*]], %[[SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x?xf32>
// CHECK:     %[[DIMS:.*]]:2 = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC]] : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG: %[[VAL1:.*]] = mip.mul %[[DIMS]]#0, %[[DIMS]]#1
// CHECK-DAG: %[[VAL2:.*]] = mip.constant.index 2
// CHECK:     %[[VAL3:.*]] = mip.mul %[[VAL1]], %[[VAL2]]
mo.graph @mul_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  kgen.param.declare M0 = <mul_nuw(X, Y)>
  kgen.param.declare M1 = <mul_nuw(Y, X)>
  kgen.param.declare M2 = <mul_nuw(M0, 2)>
  mo.output
}

// CHECK: mgp.model @eq_expressions : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.cpu.tensor<?x?x?xf32>)
// CHECK:     %[[BUF:.*]], %[[SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x?x?xf32>
// CHECK:     %[[DIMS:.*]]:3 = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC]] : !mgp.cpu.tensor_spec<?x?x?xf32>
// CHECK-DAG: %[[VAL1:.*]] = mip.cmp.eq %[[DIMS]]#0, %[[DIMS]]#1
// CHECK-DAG: %[[VAL2:.*]] = mip.cmp.eq %[[DIMS]]#1, %[[DIMS]]#2
// CHECK-DAG: %[[VAL3:.*]] = mip.constant.bool true
mo.graph @eq_expressions<X, Y, Z>(%arg0: !mo.tensor<[X, Y, Z], f32>) {
  kgen.param.declare E0: i1 = <eq(X, Y)>
  kgen.param.declare E1: i1 = <eq(Y, X)>
  kgen.param.declare E2: i1 = <eq(Y, Z)>
  kgen.param.declare E3: i1 = <eq(X, X)>
  mo.output
}

// CHECK: mgp.model @lt_expressions : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.cpu.tensor<?x?xf32>)
// CHECK:     %[[BUF:.*]], %[[SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x?xf32>
// CHECK:     %[[DIMS:.*]]:2 = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC]] : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG: %[[VAL1:.*]] = mip.cmp.lt %[[DIMS]]#0, %[[DIMS]]#1
// CHECK-DAG: %[[VAL2:.*]] = mip.cmp.lt %[[DIMS]]#1, %[[DIMS]]#0
// CHECK-DAG: %[[VAL3:.*]] = mip.cmp.lt %[[DIMS]]#0, %[[DIMS]]#0
mo.graph @lt_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  kgen.param.declare LT0: i1 = <lt(X, Y)>
  kgen.param.declare LT1: i1 = <lt(Y, X)>
  kgen.param.declare LT2: i1 = <lt(X, X)>
  mo.output
}

// CHECK: mgp.model @le_expressions : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.cpu.tensor<?x?xf32>)
// CHECK:     %[[BUF:.*]], %[[SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x?xf32>
// CHECK:     %[[DIMS:.*]]:2 = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC]] : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG: %[[VAL1:.*]] = mip.cmp.le %[[DIMS]]#0, %[[DIMS]]#1
// CHECK-DAG: %[[VAL2:.*]] = mip.cmp.le %[[DIMS]]#1, %[[DIMS]]#0
// CHECK-DAG: %[[VAL3:.*]] = mip.cmp.le %[[DIMS]]#0, %[[DIMS]]#0
mo.graph @le_expressions<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  kgen.param.declare LE0: i1 = <le(X, Y)>
  kgen.param.declare LE1: i1 = <le(Y, X)>
  kgen.param.declare LE2: i1 = <le(X, X)>
  mo.output
}

// CHECK: mgp.model @bool_expressions : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.cpu.tensor<?x?x?xf32>)
// CHECK:     %[[BUF:.*]], %[[SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x?x?xf32>
// CHECK:     %[[DIMS:.*]]:3 = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC]] : !mgp.cpu.tensor_spec<?x?x?xf32>
// CHECK-DAG: %[[VAL1:.*]] = mip.cmp.eq %[[DIMS]]#0, %[[DIMS]]#1
// CHECK-DAG: %[[VAL2:.*]] = mip.cmp.eq %[[DIMS]]#1, %[[DIMS]]#2
// CHECK-DAG: %[[VAL3:.*]] = mip.bool.or %[[VAL1]], %[[VAL2]]
// CHECK-DAG: %[[VAL4:.*]] = mip.bool.xor %[[VAL1]], %[[VAL2]]
// CHECK-DAG: %[[VAL5:.*]] = mip.bool.and %[[VAL1]], %[[VAL2]]
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

// CHECK: mgp.model @complex_expressions : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.cpu.tensor<?x?x?xf32>)
// CHECK:     %[[BUF:.*]], %[[SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x?x?xf32>
// CHECK:     %[[DIMS:.*]]:3 = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC]] : !mgp.cpu.tensor_spec<?x?x?xf32>
// CHECK-DAG: %[[VAL1:.*]] = mip.mul %[[DIMS]]#0, %[[DIMS]]#1
// CHECK-DAG: %[[VAL2:.*]] = mip.mul %[[DIMS]]#1, %[[DIMS]]#2
// CHECK-DAG: %[[VAL3:.*]] = mip.add %1, %2
mo.graph @complex_expressions<X, Y, Z>(%arg0: !mo.tensor<[X, Y, Z], f32>) {
  kgen.param.declare C0 = <add(mul_nuw(X, Y), mul_nuw(Y, Z))>
  mo.output
}

// CHECK: mgp.model @variadic_exprs : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.cpu.tensor<?x?x?xf32>)
// CHECK: %[[BUF:.*]], %[[SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x?x?xf32>
// CHECK: %[[DIMS:.*]]:3 = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC]] : !mgp.cpu.tensor_spec<?x?x?xf32>
// CHECK: %[[VAL1:.*]] = mip.add %[[DIMS]]#0, %[[DIMS]]#1
// CHECK: %[[VAL2:.*]] = mip.add %[[VAL1]], %[[DIMS]]#2
mo.graph @variadic_exprs<X, Y, Z>(%arg0: !mo.tensor<[X, Y, Z], f32>) {
  kgen.param.declare C0 = <add(X, Y, Z)>
  mo.output
}

// CHECK: mgp.model @shape_attr : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.cpu.tensor<?x?x?xf32>)
// CHECK: %[[BUF:.*]], %[[SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x?x?xf32>
// CHECK: %[[DIMS:.*]]:3 = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC]] : !mgp.cpu.tensor_spec<?x?x?xf32>
// CHECK: %[[VAL1:.*]] = mgp.cpu.tensor_spec.create %[[DIMS]]#0, %[[DIMS]]#1, %[[DIMS]]#2 : !mgp.cpu.tensor_spec<?x?x?xsi64>
mo.graph @shape_attr<X, Y, Z>(%arg0: !mo.tensor<[X, Y, Z], f32>) {
  kgen.param.declare Sh: !mo.shape = <[X, Y, Z]>
  mo.output
}
