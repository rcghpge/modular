// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths="%kernel_api_test_lib"" -o %t.adv.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=2x2 --input-shapes="" --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths="%kernel_api_test_lib"" -o %t.legacy.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=2x2 --input-shapes="" --result-output-style=full %t.legacy.mlir | FileCheck %s

// Dynamic-shape counterpart of the scalar-operand case in
// adv-fusion-map-execute-scalar-variadic.mlir: same `Scalar` operand class
// (`mogg._scalar.create`) but with a symbolic tensor dim. Same dual-run
// (adv vs legacy) equivalence proof as the sibling files.

// Scalar operand with a dynamic tensor shape: `res = a * s` with
// `--model-inputs=ones --input-shapes=2x2 --input-shapes=""`, a = [1,1,1,1],
// s = 1.0 -> [1,1,1,1].
// CHECK-LABEL: Running 'scale_dynamic':
// CHECK: 'scale_dynamic' returned tensor<2x2xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
mo.graph @scale_dynamic<M, N>(%a: !mo.tensor<[M, N], f32>, %s: !mo.tensor<[], f32>) -> (!mo.tensor<[M, N], f32>) {
  %1 = mo.custom {symbol = "map_lower_scale", device = #M.device_ref<"cpu", 0>}(%a, %s) : (!mo.tensor<[M, N], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[M, N], f32>
  mo.output %1 : !mo.tensor<[M, N], f32>
}
