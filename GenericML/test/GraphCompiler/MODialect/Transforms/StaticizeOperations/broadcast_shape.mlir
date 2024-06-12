// RUN: mo-opt %s --propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @broadcast_shape
// CHECK-DAG: kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(eq(D0, 1), eq(D0, 50)))>
// CHECK-DAG: kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(eq(add(D0, D1), 1), eq(add(D0, D1), 80)))>
// CHECK: [[V0:%.*]] = mosh.param.to_value: !mosh.ape = <[D0, D1, 4, cond(eq(D0, 1), 50, D0), 70, cond(eq(add(D0, D1), 1), 80, add(D0, D1))]>
// CHECK: [[V1:%.*]] = mo.shape.to_tensor(%0) -> !mo.tensor<[6], si64>
// CHECK: mo.output [[V1]] : !mo.tensor<[6], si64>
mo.graph @broadcast_shape<D0, D1>() -> !mo.tensor<[6], si64> {
  %shape1 = mo.constant {value = #M.dense_array<50, 70, 80>: tensor<3xsi64>} : !mo.tensor<[3], si64>
  %shape2_params = mosh.param.to_value: !mosh.ape = <[D0, D1, 4, D0, 70, add(D0, D1)]>
  %shape2 = mo.shape.to_tensor(%shape2_params) -> !mo.tensor<[6], si64>

  %res = mo.broadcast_shape(%shape1, %shape2): (!mo.tensor<[3], si64>, !mo.tensor<[6], si64>) -> !mo.tensor<[6], si64>
  mo.output %res : !mo.tensor<[6], si64>
}
