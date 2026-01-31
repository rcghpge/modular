// RUN: mo-opt %s --mo-shape-inference --mo-to-mgp=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=2x3 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @cumsumDynamic<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D0, D1], f32> {
  %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %output = mo.cumsum(%arg0, %axis) {reverse = 0 : index, exclusive = 0 : index} : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[D0, D1], f32>
  mo.output %output : !mo.tensor<[D0, D1], f32>
}
// CHECK{LITERAL}: 'cumsumDynamic' returned tensor<2x3xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]
