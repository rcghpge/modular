// RUN: mo-opt --canonicalize %s | FileCheck --enable-var-scope %s

// CHECK-LABEL: @chain_create_input_folded_away
// CHECK: [[ARG0:%.*]]:
// CHECK: kgen.param.declare [[COND:.*]]: i1
// CHECK: kgen.param.declare [[CH:.*]]: !mo.chain
// CHECK: [[CH0:%.*]] = mosh.param.to_value{{.*}}<[[CH]]>
// CHECK-NOT: mosh.param.to_value: !mo.chain = <#mo.chain>
// CHECK: mo.chain.create([[CH0]]) : !mo.chain
mo.graph @chain_create_input_folded_away<D1, D2, D3, D4>
(
  %arg0: !mo.tensor<[D1, D2, D3, D4], f32>
) -> (!mo.tensor<[D1, D2, D3, D4], f32>, !mo.tensor<[D1, D2, D3, D4], f32>){

  kgen.param.declare COND: i1 = <eq(D1, D2)>
  kgen.param.declare CH: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "error!" : !kgen.string}, COND)>

  %ch0 = mosh.param.to_value: !mo.chain = <CH>
  %ch1 = mosh.param.to_value: !mo.chain = <#mo.chain>

  %1 = mo.chain.create(%ch0, %ch1) : !mo.chain, !mo.chain

  %r1, %r2 = mo.guard[%1] (%arg0, %arg0) : !mo.tensor<[D1, D2, D3, D4], f32>, !mo.tensor<[D1, D2, D3, D4], f32>
  mo.output %r1, %r2 : !mo.tensor<[D1, D2, D3, D4], f32>, !mo.tensor<[D1, D2, D3, D4], f32>
}

// CHECK-LABEL: @all_fully_folded_away
// CHECK: [[ARG0:%.*]]:
// CHECK-NOT: mosh.param.to_value
// CHECK-NOT: mo.chain.create
// CHECK-NOT: mo.guard
// CHECK: mo.output [[ARG0]], [[ARG0]]
mo.graph @all_fully_folded_away<D1, D2, D3, D4>
(
  %arg0: !mo.tensor<[D1, D2, D3, D4], f32>
) -> (!mo.tensor<[D1, D2, D3, D4], f32>, !mo.tensor<[D1, D2, D3, D4], f32>){

  %0 = mosh.param.to_value: !mo.chain = <#mo.chain>

  %1 = mo.chain.create(%0) : !mo.chain
  %r1, %r2 = mo.guard[%1] (%arg0, %arg0) : !mo.tensor<[D1, D2, D3, D4], f32>, !mo.tensor<[D1, D2, D3, D4], f32>

  mo.output %r1, %r2 : !mo.tensor<[D1, D2, D3, D4], f32>, !mo.tensor<[D1, D2, D3, D4], f32>
}
