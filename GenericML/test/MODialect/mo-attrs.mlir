// RUN: mo-opt --allow-unregistered-dialect %s -o %t
// RUN: mo-opt --allow-unregistered-dialect %t | FileCheck %s

// CHECK: mo.graph @shape_attr_and_type() {
mo.graph @shape_attr_and_type() {
  // CHECK: "foo.static"() {aa = #mo<shape[1, 2, 3]> : !mo.shape} : () -> !mo.shape
  %0 = "foo.static"() {aa = #mo<shape[1, 2, 3]> : !mo.shape} : () -> !mo.shape
  // CHECK: kgen.param.declare [[N:.*]] =
  kgen.param.declare N = <5>
  // CHECK: "foo.parametric"() {bb = #mo<shape[[[N]], ?, 3]> : !mo.shape} : () -> !mo.shape
  %1 = "foo.parametric"() {bb = #mo<shape[N, ?, 3]> : !mo.shape} : () -> !mo.shape
  // CHECK: "foo.empty"() {cc = #mo<shape[]> : !mo.shape} : () -> !mo.shape
  %2 = "foo.empty"() {cc = #mo<shape[]> : !mo.shape} : () -> !mo.shape
  // CHECK: "foo.expression"() {dd = #mo<shape[mul(N, ?), N]> : !mo.shape} : () -> !mo.shape
  %3 = "foo.expression"() {dd = #mo<shape[mul(N, ?), N]> : !mo.shape} : () -> !mo.shape

  // CHECK: kgen.param.declare [[S:.*]]: !mo.shape = <#mo<shape[N, 2, ?]>>
  kgen.param.declare S: !mo.shape = <#mo<shape[N, 2, ?]>>
  mo.output
}

// CHECK: mo.graph @tensor_shape_attr(%[[ARG0:.*]]: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[], f32> {
mo.graph @tensor_shape_attr(%arg0: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[], f32> {
  // CHECK: kgen.param.declare [[N:.*]] = <2>
  kgen.param.declare N = <2>
  // CHECK: kgen.param.declare [[M:.*]] = <5>
  kgen.param.declare M = <5>
  // CHECK: %[[VAL0:.*]] = "m.cast"(%[[ARG0]]) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<[[[N]], [[M]]], f32>
  %0 = "m.cast"(%arg0) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<[N, M], f32>
  // CHECK: kgen.param.declare [[SH:.*]]: !mo.shape = <#mo<shape[[[N]], ?, 3]>>
  kgen.param.declare Sh: !mo.shape = <#mo<shape[N, ?, 3]>>
  // CHECK: %[[VAL1:.*]] = "m.cast"(%[[VAL0]]) : (!mo.tensor<[[[N]], [[M]]], f32>) -> !mo.tensor<[[SH]], f32>
  %1 = "m.cast"(%0) : (!mo.tensor<[N, M], f32>) -> !mo.tensor<Sh, f32>
  // CHECK: %[[VAL2:.*]] = "m.cast"(%[[VAL1]]) : (!mo.tensor<[[SH]], f32>) -> !mo.tensor<[], f32>
  %2 = "m.cast"(%1) : (!mo.tensor<Sh, f32>) -> !mo.tensor<[], f32>
  mo.output %2 : !mo.tensor<[], f32>
}
