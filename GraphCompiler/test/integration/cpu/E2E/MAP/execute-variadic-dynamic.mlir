// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths="%kernel_api_test_lib"" -o %t.adv.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=4 --input-shapes=4 --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths="%kernel_api_test_lib"" -o %t.legacy.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=4 --input-shapes=4 --result-output-style=full %t.legacy.mlir | FileCheck %s

// Dynamic-shape counterpart of the variadic-operand case in
// adv-fusion-map-execute-scalar-variadic.mlir: same variadic operand run
// (`mogg._tuple.create`) but with a symbolic tensor dim. Same dual-run
// (adv vs legacy) equivalence proof as the sibling files.

// Variadic operand with dynamic tensor shapes: `res = sum(inputs)` with
// `--model-inputs=ones --input-shapes=4 --input-shapes=4`, in0 = in1 =
// [1,1,1,1] -> [2,2,2,2].
// CHECK-LABEL: Running 'var_sum_dynamic':
// CHECK: 'var_sum_dynamic' returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @var_sum_dynamic<N>(%in0: !mo.tensor<[N], f32>, %in1: !mo.tensor<[N], f32>) -> (!mo.tensor<[N], f32>) {
  %1 = mo.custom {symbol = "map_lower_var_sum", device = #M.device_ref<"cpu", 0>}(%in0, %in1) : (!mo.tensor<[N], f32>, !mo.tensor<[N], f32>) -> !mo.tensor<[N], f32>
  mo.output %1 : !mo.tensor<[N], f32>
}
