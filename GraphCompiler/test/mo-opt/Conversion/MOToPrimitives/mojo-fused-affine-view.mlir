// RUN: mo-opt %s --mogg-import-packages -mo-to-mogg --mogg-fuse-epilogues --mogg-fuse-consts -mogg-to-mgp -canonicalize | FileCheck %s

// CHECK: mogg.kernel.stub @Affine_reduce_add_split_dim_mogg
// CHECK:       [[LAMBDA:%.*]] = mogg.bind ({{.*}}) output_lambda
// CHECK:       [[RESHAPE:%.*]] = mogg.tensor.reshape({{.*}}, {{.*}}) : (!mo.tensor<[2, 2, 1], f32, {io = 0 : i32{{.*}}}>, !kgen.scalar<si64>) -> !mo.tensor<[A, 1], f32{{.*}}>
// CHECK-NEXT:  [[FUSED:%.*]] = mogg.tensor.fused([[LAMBDA]], [[RESHAPE]]) : (!mo.tensor<[A, 1], f32{{.*}}>, !mo.tensor<[A, 1], f32{{.*}}>) -> !mo.tensor<[A, 1], f32{{.*}}>
// CHECK:       mogg.call.execute["{{.*}}.ReduceAdd"] {{.*}} outputs([[FUSED]]

mo.graph @reduce_split_dim_dyn<A, B>(%arg0: !mo.tensor<[A, B], f32>) -> (!mo.tensor<[2, 2, 1], f32>) {
  %1 = mo.reduce.add(%arg0) {axis = 1 : index} : (!mo.tensor<[A, B], f32>) -> !mo.tensor<[A, 1], f32>
  %2 = mo.split_dim[0](%1): (!mo.tensor<[A, 1], f32>) -> !mo.tensor<[2, 2, 1], f32>
  mo.output %2 : !mo.tensor<[2, 2, 1], f32>
}
