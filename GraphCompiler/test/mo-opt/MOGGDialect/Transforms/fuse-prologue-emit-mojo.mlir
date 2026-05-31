// RUN: mo-opt %s --mogg-import-packages --mo-to-mogg --mogg-fuse-elementwise --mogg-fuse-prologues | FileCheck %s

// CHECK-LABEL: mo.graph @softmax
// CHECK:       mogg.kernel({{.*}}: !mo.tensor<[2, 3], f32>, {{.*}}: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f64> {
// CHECK:       mogg.output_placeholders : !mo.tensor<[2, 3], f64>
// CHECK:       mogg.device_context_placeholder
// CHECK:       mogg.bind ({{.*}}, {{.*}}) input_lambda ({{.*}}: f32, {{.*}}: f32, {{.*}}: !pop.array<2, index>) {
// CHECK-NEXT:  mogg.call.elementwise["{{.*}}.Add"] ({{.*}}, {{.*}})
// CHECK-NEXT:  mogg.call.elementwise["{{.*}}.Cast"] ({{.*}})
// CHECK-NEXT:  mogg.output {{.*}}, {{.*}} : f64, !pop.array<2, index>
// CHECK:       }: (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f64>
// CHECK:       mogg.call.execute["{{.*}}.Softmax"] inputs({{.*}}, {{.*}} : !mo.tensor<[2, 3], f64>, !mogg.context) outputs({{.*}} : !mo.tensor<[2, 3], f64>)
// CHECK:       fusionInfo = Prologue(Elementwise("mo.add":{{.*}}, "mo.cast":{{.*}}), "mo.reduce.softmax":{{.*}})

mo.graph @softmax(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f64>) {
  %add = mo.add(%arg0, %arg0) : !mo.tensor<[2, 3], f32>
  %cast = mo.cast(%add) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f64>
  %0 = mo.reduce.softmax(%cast) {axis = 1 : index} : (!mo.tensor<[2, 3], f64>) -> !mo.tensor<[2, 3], f64>
  mo.output %0 : !mo.tensor<[2, 3], f64>
}


// CHECK-LABEL: mo.graph @gather
// CHECK:       mogg.kernel({{.*}}: !mo.tensor<[3, 3], f32>, {{.*}}: !mo.tensor<[2, 2], si32>, {{.*}}: !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32> {
// CHECK:       mogg.output_placeholders : !mo.tensor<[2, 2, 3], f32>
// CHECK:       mogg.bind ({{.*}}) input_lambda ({{.*}}: f32, {{.*}}: !pop.array<2, index>) {
// CHECK-NEXT:  mogg.output {{.*}}, {{.*}} : f32, !pop.array<2, index>
// CHECK:       }: (!mo.tensor<[3, 3], f32>) -> !mo.tensor<[3, 3], f32>
// CHECK:       mogg.device_context_placeholder
// CHECK:       mogg.bind ({{.*}}, {{.*}}) input_lambda ({{.*}}: si32, {{.*}}: si32, {{.*}}: !pop.array<2, index>) {
// CHECK-NEXT:  mogg.call.elementwise["{{.*}}.Add"] ({{.*}}, {{.*}})
// CHECK-NEXT:  mogg.output {{.*}}, {{.*}} : si32, !pop.array<2, index>
// CHECK:       }: (!mo.tensor<[2, 2], si32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2], si32>
// CHECK:       mogg.bind ({{.*}}) output_lambda ({{.*}}: f32, {{.*}}: !pop.array<3, index>) {
// CHECK-NEXT:  mogg.output {{.*}}, {{.*}} : f32, !pop.array<3, index>
// CHECK:       }: (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 2, 3], f32>
// CHECK:       mogg.call.execute["{{.*}}.Gather"] inputs({{.*}}, {{.*}}, {{.*}} : !mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>, !mogg.context) outputs({{.*}} : !mo.tensor<[2, 2, 3], f32>)
// CHECK:       fusionInfo = Prologue("mo.add":{{.*}}, "mo.gather":{{.*}}, operandIndices=[1])

mo.graph @gather(%arg0: !mo.tensor<[3, 3], f32>, %arg1: !mo.tensor<[2, 2], si32>, %arg2: !mo.tensor<[2, 2], si32>) -> (!mo.tensor<[2, 2, 3], f32>) {
  %indices = mo.add(%arg1, %arg2) : !mo.tensor<[2, 2], si32>
  %res = mo.gather(%arg0, %indices) {axis = 0 : index} : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32>
  mo.output %res : !mo.tensor<[2, 2, 3], f32>
}
