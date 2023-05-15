// RUN: mosh-opt --allow-unregistered-dialect %s -o %t
// RUN: mosh-opt --allow-unregistered-dialect %t | FileCheck %s

// CHECK: kgen.generator @shape_attr_and_type() {
kgen.generator @shape_attr_and_type() {
  // CHECK: "foo.static"() {aa = #mosh<ape[1, 2, 3]> : !mosh.ape} : () -> !mosh.ape
  %0 = "foo.static"() {aa = #mosh<ape[1, 2, 3]> : !mosh.ape} : () -> !mosh.ape
  // CHECK: kgen.param.declare [[N:.*]] =
  kgen.param.declare N = <5>
  // CHECK: "foo.parametric"() {bb = #mosh<ape[[[N]], ?, 3]> : !mosh.ape} : () -> !mosh.ape
  %1 = "foo.parametric"() {bb = #mosh<ape[N, ?, 3]> : !mosh.ape} : () -> !mosh.ape
  // CHECK: "foo.empty"() {cc = #mosh<ape[]> : !mosh.ape} : () -> !mosh.ape
  %2 = "foo.empty"() {cc = #mosh<ape[]> : !mosh.ape} : () -> !mosh.ape
  // CHECK: "foo.expression"() {dd = #mosh<ape[mul(N, ?), N]> : !mosh.ape} : () -> !mosh.ape
  %3 = "foo.expression"() {dd = #mosh<ape[mul(N, ?), N]> : !mosh.ape} : () -> !mosh.ape

  // CHECK: kgen.param.declare [[P:.*]]: !mosh.ape = <apply(:() -> !mosh.ape "mosh.foobar")>
  kgen.param.declare P: !mosh.ape = <apply(:() -> !mosh.ape "mosh.foobar")>

  // CHECK: kgen.param.declare [[S:.*]]: !mosh.ape = <[N, 2, ?]>
  kgen.param.declare S: !mosh.ape = <[N, 2, ?]>
  kgen.return
}

// CHECK: kgen.func @chain_attr() {
kgen.func @chain_attr() {
  // CHECK-DAG: "foo.chain"() {ch = #mosh.chain : !mosh.chain} : () -> !mosh.chain
  %0 = "foo.chain"() {ch = #mosh.chain : !mosh.chain} : () -> !mosh.chain
  // CHECK-DAG: kgen.param.declare [[CH:.*]]: !mosh.chain = <#mosh.chain>
  kgen.param.declare Ch: !mosh.chain = <#mosh.chain>
  kgen.return
}
