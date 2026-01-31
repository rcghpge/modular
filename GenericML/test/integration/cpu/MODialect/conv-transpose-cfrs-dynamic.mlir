// RUN: mo-opt %s --mo-to-mgp="constant-fold=false" -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=1x2x2x16 --input-shapes=16x8x2x2 --result-output-style=full %t.mlir | FileCheck %s

// TODO: Support Conv FCSR support on CPU (GEX-2302)
// XFAIL: *

mo.graph @conv_transpose(%input: !mo.tensor<[?, ?, ?, ?], f32>,
                        %filter: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<CFRS>}>)
                         -> (!mo.tensor<[?, ?, ?, ?], f32>) {
  %st = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %op = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %output = mo.conv_transpose(%input, %filter) {filter_layout = ""}
    [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<CFRS>}>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %output : !mo.tensor<[?, ?, ?, ?], f32>
}
