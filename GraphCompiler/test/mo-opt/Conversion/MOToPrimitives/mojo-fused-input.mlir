// RUN: mo-opt %s --mogg-import-packages -mo-to-mogg --mogg-fuse-prologues --mogg-fuse-epilogues --mogg-fuse-consts -mogg-to-mgp -canonicalize | FileCheck %s

// CHECK: mogg.kernel.stub @stub_0
// CHECK:       [[IN:%.*]] = mogg.bind ({{.*}}, {{.*}}) input_lambda
// CHECK:       [[EMPTY_BUFF:%.*]] = mogg.empty_buffer : !mgp.buffer
// CHECK-NEXT:  [[EMPTY_TENSOR:%.*]] = mogg.tensor.__init__([[EMPTY_BUFF]]) {io = 1 : i32, mut = false} : (!mgp.buffer) -> !mo.tensor<[4, 5, 2, 3], f32, {io = 1 : i32, mut = false}>
// CHECK-NEXT:  [[FUSED:%.*]] = mogg.tensor.fused([[IN]], [[EMPTY_TENSOR]]) : (!mo.tensor<[4, 5, 2, 3], f32{{.*}}>, !mo.tensor<[4, 5, 2, 3], f32, {io = 1 : i32, mut = false}>) -> !mo.tensor<[4, 5, 2, 3], f32{{.*}}>
// CHECK:       mogg.call.execute["{{.*}}.ReduceAdd"] inputs([[FUSED]]

mo.graph @reduce_fused_input(%arg0: !mo.tensor<[4, 5, 2, 3], f32>, %arg1: !mo.tensor<[4, 5, 2, 3], f32>, %arg2: !mo.tensor<[4, 5, 1, 3], f32>) -> (!mo.tensor<[4, 5, 1, 3], f32>) {
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<[4, 5, 2, 3], f32>
  %1 = mo.reduce.add(%0) {axis = 2 : index} : (!mo.tensor<[4, 5, 2, 3], f32>) -> !mo.tensor<[4, 5, 1, 3], f32>
  %2 = mo.sub(%1, %arg2) : !mo.tensor<[4, 5, 1, 3], f32>
  mo.output %2 : !mo.tensor<[4, 5, 1, 3], f32>
}
