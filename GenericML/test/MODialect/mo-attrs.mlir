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

  // CHECK: kgen.param.declare [[S:.*]]: !mo.shape = <[N, 2, ?]>
  kgen.param.declare S: !mo.shape = <[N, 2, ?]>
  mo.output
}
