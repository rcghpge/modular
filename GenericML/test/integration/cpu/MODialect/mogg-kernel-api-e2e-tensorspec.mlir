// RUN: mo-opt %s -o %t_mojo.mlir --mo-to-mgp="kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t_mojo.mlir --result-output-style=full --input-shapes=1x2x3x4 --input-shapes=1x2x3x4 --model-inputs=ones | FileCheck %s

// CHECK-LABEL: Running 'print_static_shape_strides'
// CHECK: x.shape = [1, 2, 3, 4]
// CHECK: x.strides = [24, 12, 4, 1]
// CHECK: x.alignment = {{[0-9][0-9]}}
// CHECK: x.address_space = AddressSpace.GENERIC
// CHECK: x.exclusive = True
// CHECK: 'print_static_shape_strides' returned tensor<1x2x3x4xf32> [1.000000e+00, 1.000000e+00
mo.graph @print_static_shape_strides(%x: !mo.tensor<[1,2,3,4], f32>) -> !mo.tensor<[1,2,3,4], f32> {
    %out = mo.custom {symbol = "print_tensor_spec", device = #M.device_ref<"cpu", 0>} (%x) : (!mo.tensor<[1,2,3,4], f32>) -> !mo.tensor<[1,2,3,4], f32>
    mo.output %out : !mo.tensor<[1,2,3,4], f32>
}

// CHECK-LABEL: Running 'print_dyn_shape_strides'
// CHECK: x.shape = [1, 2, ?, 4]
// CHECK: x.strides = [?, ?, 4, 1]
// CHECK: x.alignment = {{[0-9][0-9]}}
// CHECK: x.address_space = AddressSpace.GENERIC
// CHECK: x.exclusive = True
// CHECK: 'print_dyn_shape_strides' returned tensor<1x2x3x4xf32> [1.000000e+00, 1.000000e+00
mo.graph @print_dyn_shape_strides<D0>(%x: !mo.tensor<[1,2,D0,4], f32>) -> !mo.tensor<[1,2,D0,4], f32> {
    %out = mo.custom {symbol = "print_tensor_spec", device = #M.device_ref<"cpu", 0>} (%x) : (!mo.tensor<[1,2,D0,4], f32>) -> !mo.tensor<[1,2,D0,4], f32>
    mo.output %out : !mo.tensor<[1,2,D0,4], f32>
}

// CHECK-LABEL: Running 'print_exclusive_inference'
// CHECK: x.shape = [1, 2, 3, 4]
// CHECK: x.strides = [24, 12, 4, 1]
// CHECK: x.alignment = {{[0-9][0-9]}}
// CHECK: x.address_space = AddressSpace.GENERIC
// CHECK: x.exclusive = False
mo.graph @print_exclusive_inference(%x: !mo.tensor<[1,2,3,4], f32>) -> !mo.tensor<[1,2,3,4], f32> {
    %out = mo.custom {symbol = "print_tensor_spec_view", device = #M.device_ref<"cpu", 0>} (%x) : (!mo.tensor<[1,2,3,4], f32>) -> !mo.tensor<[1,2,3,4], f32>
    mo.output %out : !mo.tensor<[1,2,3,4], f32>
}
