// RUN: mo-opt %s --mo-to-mogg --mogg-fuse-elementwise --mogg-fuse-prologues-epilogues | FileCheck %s

// CHECK-LABEL: mo.graph @softmax
// CHECK:       mogg.kernel({{.*}}: !mo.tensor<[], si64>, {{.*}}: !mo.tensor<[2, 3], f32>, {{.*}}: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f64> {
// CHECK:       mogg.output_placeholders : !mo.tensor<[2, 3], f64>
// CHECK:       mogg.device_context_placeholder
// CHECK:       mogg.bind ({{.*}}, {{.*}}) input_lambda ({{.*}}: f32, {{.*}}: f32, {{.*}}: !pop.array<2, index>) {
// CHECK-NEXT:  mogg.call.elementwise["mo.add"] ({{.*}}, {{.*}}) : f32, f32 -> f32
// CHECK-NEXT:  mogg.call.elementwise["mo.cast"] ({{.*}}) : f32 -> f64
// CHECK-NEXT:  mogg.output {{.*}}, {{.*}} : f64, !pop.array<2, index>
// CHECK:       }: (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f64>
// CHECK:       mogg.as_scalar
// CHECK:       mogg.call.execute["mo.softmax"] inputs({{.*}}, {{.*}}, {{.*}} : !mo.tensor<[2, 3], f64>, !mo.scalar<si64>, !mogg.context) outputs({{.*}} : !mo.tensor<[2, 3], f64>)
// CHECK:       fusionInfo = Epilogue(Elementwise(mo.add:{{.*}}, mo.cast:{{.*}}), mo.softmax:{{.*}})

mo.graph @softmax(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[], si64>) -> (!mo.tensor<[2, 3], f64>) {
  %add = mo.add(%arg0, %arg0) : !mo.tensor<[2, 3], f32>
  %cast = mo.cast(%add) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f64>
  %0 = mo.softmax(%cast, %arg1) : (!mo.tensor<[2, 3], f64>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f64>
  mo.output %0 : !mo.tensor<[2, 3], f64>
}


// CHECK-LABEL: mo.graph @gather
// CHECK:       mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK:       mogg.kernel({{.*}}: !mo.tensor<[3, 3], f32>, {{.*}}: !mo.tensor<[], si64>, {{.*}}: !mo.tensor<[2, 2], si32>, {{.*}}: !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32> {
// CHECK:       mogg.output_placeholders : !mo.tensor<[2, 2, 3], f32>
// CHECK:       mogg.bind ({{.*}}) input_lambda ({{.*}}: f32, {{.*}}: !pop.array<2, index>) {
// CHECK-NEXT:  mogg.output {{.*}}, {{.*}} : f32, !pop.array<2, index>
// CHECK:       }: (!mo.tensor<[3, 3], f32>) -> !mo.tensor<[3, 3], f32>
// CHECK:       mogg.device_context_placeholder
// CHECK:       mogg.bind ({{.*}}, {{.*}}) input_lambda ({{.*}}: si32, {{.*}}: si32, {{.*}}: !pop.array<2, index>) {
// CHECK-NEXT:  mogg.call.elementwise["mo.add"] ({{.*}}, {{.*}}) : si32, si32 -> si32
// CHECK-NEXT:  mogg.output {{.*}}, {{.*}} : si32, !pop.array<2, index>
// CHECK:       }: (!mo.tensor<[2, 2], si32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2], si32>
// CHECK:       mogg.bind ({{.*}}) output_lambda ({{.*}}: f32, {{.*}}: !pop.array<3, index>) {
// CHECK-NEXT:  mogg.output {{.*}}, {{.*}} : f32, !pop.array<3, index>
// CHECK:       }: (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 2, 3], f32>
// CHECK:       mogg.as_scalar
// CHECK:       mogg.call.execute["mo.gather"] inputs({{.*}}, {{.*}}, {{.*}}, {{.*}} : !mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>, !mo.scalar<si64>, !mogg.context) outputs({{.*}} : !mo.tensor<[2, 2, 3], f32>)
// CHECK:       fusionInfo = Epilogue(mo.add:{{.*}}, mo.gather:{{.*}}, operandIndices=[1])

mo.graph @gather(%arg0: !mo.tensor<[3, 3], f32>, %arg1: !mo.tensor<[2, 2], si32>, %arg2: !mo.tensor<[2, 2], si32>) -> (!mo.tensor<[2, 2, 3], f32>) {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %indices = mo.add(%arg1, %arg2) : !mo.tensor<[2, 2], si32>
  %res = mo.gather(%arg0, %indices, %axis) : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 2, 3], f32>
  mo.output %res : !mo.tensor<[2, 2, 3], f32>
}
