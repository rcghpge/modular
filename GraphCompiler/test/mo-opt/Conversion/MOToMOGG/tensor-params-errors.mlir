// RUN: mo-opt --mo-to-mogg="kernel-library-paths=%kernel_api_test_lib" %s --verify-diagnostics --split-input-file

mo.graph @valid_same_rank_dtype(%x: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[5, 5], f32> {
    %0 = mo.custom {symbol = "test_same_rank_dtype", device = #M.device_ref<"cpu", 0>}(%x) : (!mo.tensor<[4, 4], f32>) -> !mo.tensor<[5, 5], f32>
    mo.output %0 : !mo.tensor<[5, 5], f32>
}

// -----

mo.graph @invalid_rank_differs(%x: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[5], f32> {
    %0 = mo.custom {symbol = "test_same_rank_dtype", device = #M.device_ref<"cpu", 0>}(%x) : (!mo.tensor<[4, 4], f32>) -> !mo.tensor<[5], f32>
    mo.output %0 : !mo.tensor<[5], f32>
}

// -----

mo.graph @valid_fix_rank(%x: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[5, 5], f32> {
    %0 = mo.custom {symbol = "test_fix_rank", device = #M.device_ref<"cpu", 0>}(%x) : (!mo.tensor<[4, 4], f32>) -> !mo.tensor<[5, 5], f32>
    mo.output %0 : !mo.tensor<[5, 5], f32>
}

// -----

mo.graph @invalid_fix_rank_2(%x: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[5], f32> {
    %0 = mo.custom {symbol = "test_fix_rank", device = #M.device_ref<"cpu", 0>}(%x) : (!mo.tensor<[4, 4], f32>) -> !mo.tensor<[5], f32>
    mo.output %0 : !mo.tensor<[5], f32>
}

// -----

mo.graph @valid_simd_scalar_input(%x: !mo.tensor<[], f32>) -> !mo.tensor<[5, 5], f32> {
    %0 = mo.custom {symbol = "test_simd_input_size_1", device = #M.device_ref<"cpu", 0>}(%x) : (!mo.tensor<[], f32>) -> !mo.tensor<[5, 5], f32>
    mo.output %0 : !mo.tensor<[5, 5], f32>
}

// -----

mo.graph @invalid_simd_size_param(%x: !mo.tensor<[], f32>) -> !mo.tensor<[5, 5], f32> {
    %0 = mo.custom {symbol = "test_simd_input", device = #M.device_ref<"cpu", 0>}(%x) : (!mo.tensor<[], f32>) -> !mo.tensor<[5, 5], f32>
    mo.output %0 : !mo.tensor<[5, 5], f32>
}

// -----

mo.graph @invalid_simd_size_not_scalar(%x: !mo.tensor<[], f32>) -> !mo.tensor<[5, 5], f32> {
    %0 = mo.custom {symbol = "test_simd_input_size_2", device = #M.device_ref<"cpu", 0>}(%x) : (!mo.tensor<[], f32>) -> !mo.tensor<[5, 5], f32>
    mo.output %0 : !mo.tensor<[5, 5], f32>
}
