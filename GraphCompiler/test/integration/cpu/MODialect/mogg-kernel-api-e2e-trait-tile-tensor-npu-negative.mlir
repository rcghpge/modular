// RUN: not mo-opt %s --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg default-device-label=npu compiled-framework-label=mof" --allow-unregistered-dialect 2>&1 | FileCheck %s

// NEGATIVE test (expected to FAIL compilation). `trait_tile_tensor_projection_negative`
// calls `to_tile_tensor()` on dense trait args at the host entry. On an NPU device
// the graph compiler emits `_MoggDenseTensorArg` with `_can_project_tile_tensor =
// False`, so `to_tile_tensor()` must not compile.
// CHECK: constraint failed: to_tile_tensor requires a TileTensorable dense argument
mo.graph @trait_tile_tensor_projection_npu_negative(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %z = mo.custom [symbol = "trait_tile_tensor_projection_negative", device = #M.device_ref<"npu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
    mo.output %z : !mo.tensor<[4], f32>
}
