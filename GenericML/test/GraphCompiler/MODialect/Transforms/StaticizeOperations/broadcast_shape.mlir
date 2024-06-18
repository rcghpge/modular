// RUN: mo-opt %s --add-fallback-shape-funcs --propagate-shapes --canonicalize | FileCheck %s
// TODO(GRA-627) remove `--add-fallback-shape-funcs`

// CHECK-LABEL: mo.graph @broadcast_shape
// CHECK-DAG: kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(eq(D0, 1), eq(D0, 50)))>
// CHECK-DAG: kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(eq(add(D0, D1), 1), eq(add(D0, D1), 80)))>
// CHECK: [[V0:%.*]] = mo.static.broadcast_to(%arg0) : {{.*}} -> !mo.tensor<[D0, D1, 4, cond(eq(D0, 1), 50, D0), 70, cond(eq(add(D0, D1), 1), 80, add(D0, D1))], si64>
// CHECK: mo.output{{.*}} [[V0]]
mo.graph @broadcast_shape<D0, D1 -> Sh0: !mosh.ape>(%input: !mo.tensor<[], si64>) -> !mo.tensor<Sh0, si64> {
  %shape1 = mo.constant {value = #M.dense_array<50, 70, 80>: tensor<3xsi64>} : !mo.tensor<[3], si64>
  %shape2_params = mosh.param.to_value: !mosh.ape = <[D0, D1, 4, D0, 70, add(D0, D1)]>
  %shape2 = mo.shape.to_tensor(%shape2_params) -> !mo.tensor<[6], si64>
  %bcast_shape = mo.broadcast_shape<() -> D2>(%shape1, %shape2): (!mo.tensor<[3], si64>, !mo.tensor<[6], si64>) -> !mo.tensor<[D2], si64>
  %res = mo.broadcast_to<() -> Sh1: !mosh.ape>(%input, %bcast_shape) : (!mo.tensor<[], si64>, !mo.tensor<[D2], si64>) -> !mo.tensor<Sh1, si64>
  mo.output<:!mosh.ape Sh1> %res : !mo.tensor<Sh1, si64>
}
