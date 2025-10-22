// RUN: EMIT_MOJO=1 mo-opt %s -mo-to-mogg --fuse-affine-views --fuse-prologues-epilogues --fuse-consts  -mogg-to-kgen -mo-to-primitives -canonicalize | FileCheck %s

// CHECK: mogg.kernel.stub @stub_0
// CHECK:       [[LAMBDA:%.*]] = mogg.bind ({{.*}}) output_lambda
// CHECK:       [[RESHAPE:%.*]] = mogg.tensor.reshape({{.*}}, {{.*}}) : (!mo.tensor<[2, 2, 1], f32, {io = 0 : i32}>, si64) -> !mo.tensor<[A, 1], f32>
// CHECK-NEXT:  [[FUSED:%.*]] = mogg.tensor.fused([[LAMBDA]], [[RESHAPE]]) : (!mo.tensor<[A, 1], f32>, !mo.tensor<[A, 1], f32>) -> !mo.tensor<[A, 1], f32>
// CHECK:       mogg.call.execute["mo.reduce.add"] {{.*}} outputs([[FUSED]]

mo.graph @reduce_split_dim_dyn<A, B>(%arg0: !mo.tensor<[A, B], f32>) -> (!mo.tensor<[2, 2, 1], f32>) {
  %axis = mo.constant {value = #M.dense_array<1> : tensor<ui32>} : !mo.tensor<[], ui32>
  %1 = mo.reduce.add(%arg0, %axis) : (!mo.tensor<[A, B], f32>, !mo.tensor<[], ui32>) -> !mo.tensor<[A, 1], f32>
  %2 = mo.split_dim[0](%1): (!mo.tensor<[A, 1], f32>) -> !mo.tensor<[2, 2, 1], f32>
  mo.output %2 : !mo.tensor<[2, 2, 1], f32>
}
