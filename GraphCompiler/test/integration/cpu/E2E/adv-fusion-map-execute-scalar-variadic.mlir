// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths="%kernel_api_test_lib"" -o %t.adv.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths="%kernel_api_test_lib"" -o %t.legacy.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.legacy.mlir | FileCheck %s

// End-to-end execution of the advanced-fusion MAP-ABI path for the
// non-tensor-operand classes that the `PackageVariadicsAndScalars` pass
// materializes: a `Scalar` operand (`mogg._scalar.create`) and a variadic
// operand run (`mogg._tuple.create`). Same dual-run equivalence proof as the
// sibling files: the adv (env-gated) and legacy runs are checked against the
// SAME patterns. `--model-inputs=ones` fills the actual-operand graphs below;
// it has no effect on the constant-only graphs.

// Scalar operand: `res = a * s`, a = [1,2,3,4], s = 3.0 -> [3,6,9,12].
// CHECK-LABEL: Running 'scale':
// CHECK: 'scale' returned tensor<2x2xf32> [3.000000e+00, 6.000000e+00, 9.000000e+00, 1.200000e+01]
mo.graph @scale() -> (!mo.tensor<[2, 2], f32>) {
  %a = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %s = mo.constant {value = #M.dense_array<3.0> : tensor<f32>} : !mo.tensor<[], f32>
  %1 = mo.custom {symbol = "map_lower_scale", device = #M.device_ref<"cpu", 0>}(%a, %s) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %1 : !mo.tensor<[2, 2], f32>
}

// Variadic operand: `res = sum(inputs)`, in0 = [1,2,3,4], in1 = [10,20,30,40]
// -> [11,22,33,44].
// CHECK-LABEL: Running 'var_sum':
// CHECK: 'var_sum' returned tensor<4xf32> [1.100000e+01, 2.200000e+01, 3.300000e+01, 4.400000e+01]
mo.graph @var_sum() -> (!mo.tensor<[4], f32>) {
  %in0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %in1 = mo.constant {value = #M.dense_array<10.0, 20.0, 30.0, 40.0> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %1 = mo.custom {symbol = "map_lower_var_sum", device = #M.device_ref<"cpu", 0>}(%in0, %in1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %1 : !mo.tensor<[4], f32>
}

// Scalar operand as an actual (non-constant) operand: `res = a * s` with
// `--model-inputs=ones`, a = [1,1,1,1], s = 1.0 -> [1,1,1,1].
// CHECK-LABEL: Running 'scale_operand':
// CHECK: 'scale_operand' returned tensor<2x2xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
mo.graph @scale_operand(%a: !mo.tensor<[2, 2], f32>, %s: !mo.tensor<[], f32>) -> (!mo.tensor<[2, 2], f32>) {
  %1 = mo.custom {symbol = "map_lower_scale", device = #M.device_ref<"cpu", 0>}(%a, %s) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %1 : !mo.tensor<[2, 2], f32>
}

// Variadic operand as actual (non-constant) operands: `res = sum(inputs)`
// with `--model-inputs=ones`, in0 = in1 = [1,1,1,1] -> [2,2,2,2].
// CHECK-LABEL: Running 'var_sum_operand':
// CHECK: 'var_sum_operand' returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @var_sum_operand(%in0: !mo.tensor<[4], f32>, %in1: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) {
  %1 = mo.custom {symbol = "map_lower_var_sum", device = #M.device_ref<"cpu", 0>}(%in0, %in1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %1 : !mo.tensor<[4], f32>
}
