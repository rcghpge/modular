// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths="%kernel_api_test_lib"" -o %t.adv.mlir
// RUN: mt --execute --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths="%kernel_api_test_lib"" -o %t.legacy.mlir
// RUN: mt --execute --result-output-style=full %t.legacy.mlir | FileCheck %s

// End-to-end execution of the advanced-fusion MAP-ABI path for plain
// single-output opaque kernels across input arities. With
// `MAX_GC_USE_ADV_FUSION=1` the `mo.custom` op lowers through the new MAP-ABI
// pipeline (`mo.custom` -> `map.iter.opaque` -> `mogg.kernel.stub @map_stub_N`
// (MAP ABI) + a `mogg._kernel` container that `MOGGToMGP` turns into a
// `mogg.generic.compiled.region`); without the env var it lowers through the
// legacy MOGG path. Both runs are checked against the SAME FileCheck patterns:
// that identical output IS the equivalence proof (the env gate must be
// correctness-transparent). This file compiles the emitted Mojo AND runs it AND
// checks the numeric result.

// 1 input / 1 output: `res = a + 1`.
// CHECK-LABEL: Running 'add_one':
// CHECK: 'add_one' returned tensor<2x2xf32> [2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]
mo.graph @add_one() -> (!mo.tensor<[2, 2], f32>) {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %1 = mo.custom {symbol = "map_lower_add_one", device = #M.device_ref<"cpu", 0>}(%0) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %1 : !mo.tensor<[2, 2], f32>
}

// 2 inputs / 1 output: `res = a + b`. Reuses the existing `imposter_add`
// kernel (which also carries a shape function), exercising a shape-function
// kernel through the MAP path.
// CHECK-LABEL: Running 'add_2in':
// CHECK: 'add_2in' returned tensor<2x2xf32> [1.100000e+01, 2.200000e+01, 3.300000e+01, 4.400000e+01]
mo.graph @add_2in() -> (!mo.tensor<[2, 2], f32>) {
  %a = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %b = mo.constant {value = #M.dense_array<10.0, 20.0, 30.0, 40.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %1 = mo.custom {symbol = "imposter_add", device = #M.device_ref<"cpu", 0>}(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %1 : !mo.tensor<[2, 2], f32>
}

// 3 inputs / 1 output: `res = a + b + c`.
// CHECK-LABEL: Running 'add3':
// CHECK: 'add3' returned tensor<2x2xf32> [1.110000e+02, 2.220000e+02, 3.330000e+02, 4.440000e+02]
mo.graph @add3() -> (!mo.tensor<[2, 2], f32>) {
  %a = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %b = mo.constant {value = #M.dense_array<10.0, 20.0, 30.0, 40.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %c = mo.constant {value = #M.dense_array<100.0, 200.0, 300.0, 400.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %1 = mo.custom {symbol = "map_lower_add3", device = #M.device_ref<"cpu", 0>}(%a, %b, %c) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %1 : !mo.tensor<[2, 2], f32>
}
