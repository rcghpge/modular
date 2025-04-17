// RUN: mo-opt --mo-to-mogg="kernel-library-paths=%kernel_api_test_lib" %s --verify-diagnostics --split-input-file

mo.graph @valid_same_rank_dtype(%x: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[5, 5], f32> {
    %0 = mo.custom {symbol = "test_same_rank_dtype"}(%x) : (!mo.tensor<[4, 4], f32>) -> !mo.tensor<[5, 5], f32>
    mo.output %0 : !mo.tensor<[5, 5], f32>
}

// -----

mo.graph @invalid_rank_differs(%x: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[5], f32> {
    // expected-error @+1 {{invalid value for parameter `rank` of kernel operand #1: the parameter was resolved to 2 : index from the passed operand, but with a previous usage the parameter was resolved to 1 : index}}
    %0 = mo.custom {symbol = "test_same_rank_dtype"}(%x) : (!mo.tensor<[4, 4], f32>) -> !mo.tensor<[5], f32>
    mo.output %0 : !mo.tensor<[5], f32>
}

// -----

mo.graph @valid_fix_rank(%x: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[5, 5], f32> {
    %0 = mo.custom {symbol = "test_fix_rank"}(%x) : (!mo.tensor<[4, 4], f32>) -> !mo.tensor<[5, 5], f32>
    mo.output %0 : !mo.tensor<[5, 5], f32>
}

// -----

mo.graph @invalid_fix_rank_2(%x: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[5], f32> {
    // expected-error @+1 {{invalid value for parameter `rank` of kernel operand #0: the kernel signature expects the constant value 2 : index but the operand passed to the kernel has the value 1 : index}}
    %0 = mo.custom {symbol = "test_fix_rank"}(%x) : (!mo.tensor<[4, 4], f32>) -> !mo.tensor<[5], f32>
    mo.output %0 : !mo.tensor<[5], f32>
}

// -----

mo.graph @valid_simd_scalar_input(%x: !mo.tensor<[1], f32>) -> !mo.tensor<[5, 5], f32> {
    %0 = mo.custom {symbol = "test_simd_input_size_1"}(%x) : (!mo.tensor<[1], f32>) -> !mo.tensor<[5, 5], f32>
    mo.output %0 : !mo.tensor<[5, 5], f32>
}

// -----

mo.graph @invalid_simd_size_param(%x: !mo.tensor<[1], f32>) -> !mo.tensor<[5, 5], f32> {
    // expected-error @+1 {{Invalid SIMD size parameter for kernel operand #1: must be a constant}}
    %0 = mo.custom {symbol = "test_simd_input"}(%x) : (!mo.tensor<[1], f32>) -> !mo.tensor<[5, 5], f32>
    mo.output %0 : !mo.tensor<[5, 5], f32>
}

// -----

mo.graph @invalid_simd_size_not_scalar(%x: !mo.tensor<[1], f32>) -> !mo.tensor<[5, 5], f32> {
    // expected-error @+1 {{Invalid SIMD size parameter for kernel operand #1: must be 1, but is `2`}}
    %0 = mo.custom {symbol = "test_simd_input_size_2"}(%x) : (!mo.tensor<[1], f32>) -> !mo.tensor<[5, 5], f32>
    mo.output %0 : !mo.tensor<[5, 5], f32>
}
