// RUN: mo-opt %s --mo-to-mgp="constant-fold=false" -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// TODO: Support Conv FCSR support on CPU (GEX-2302)
// XFAIL: *

mo.graph @conv(%input : !mo.tensor<[1, 5, 5, 3], f32>) -> (!mo.tensor<[1, 3, 3, 1], f32>) {
  %st = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %ng = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %filter = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<2.0> : tensor<f32>} : () -> !mo.tensor<[1, 3, 3, 3], f32, {layout = #mo.layout<FCRS>}>
  %output = mo.conv(%input, %filter) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
    !mo.tensor<[1, 5, 5, 3], f32>, !mo.tensor<[1, 3, 3, 3], f32, {layout = #mo.layout<FCRS>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[1, 3, 3, 1], f32>
  mo.output %output : !mo.tensor<[1, 3, 3, 1], f32>
}
