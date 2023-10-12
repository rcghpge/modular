
// RUN: mo-opt %s --infer-layouts --mo-to-mogg | FileCheck %s

mo.graph @conv_const_params(
  %input: !mo.tensor<[4, 3, 3, 1], f32>,
  %filter: !mo.tensor<[2, 2, 1, 1], f32>
) -> !mo.tensor<[4, 2, 2, 1], f32> {
  %strides = mo.constant {value = #M.dense_array<42, 23> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %dilations = mo.constant {value = #M.dense_array<-24141, 4352> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<1, 2, 3, -16> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %ng = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>

  %res = mo.conv(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %pa, num_groups = %ng] : (
    !mo.tensor<[4, 3, 3, 1], f32>, !mo.tensor<[2, 2, 1, 1], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[4, 2, 2, 1], f32>

  mo.output %res : !mo.tensor<[4, 2, 2, 1], f32>
}

// CHECK-LABEL: mo.graph @conv_const_params
// CHECK: mogg.call "mo.conv" {{.*}} static_dilations = dense<[-24141, 4352]> : vector<2xi64>, static_padding = dense<[1, 2, 3, -16]> : vector<4xi64>, static_strides = dense<[42, 23]> : vector<2xi64>

mo.graph @conv_unknown_params(
  %input: !mo.tensor<[4, 3, 3, 1], f32>,
  %filter: !mo.tensor<[2, 2, 1, 1], f32>,
  %strides: !mo.tensor<[2], si64>,
  %dil: !mo.tensor<[2], si64>,
  %padding: !mo.tensor<[4], si64>
) -> !mo.tensor<[4, 2, 2, 1], f32> {
  %ng = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>

  %res = mo.conv(%input, %filter) [strides = %strides, dilations = %dil, paddings = %padding, num_groups = %ng] : (
    !mo.tensor<[4, 3, 3, 1], f32>, !mo.tensor<[2, 2, 1, 1], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[4, 2, 2, 1], f32>

  mo.output %res : !mo.tensor<[4, 2, 2, 1], f32>
}

// CHECK-LABEL: mo.graph @conv_unknown_params

// CHECK: mogg.call "mo.conv" {{.*}} static_dilations = dense<-9223372036854775808> : vector<2xi64>, static_padding = dense<-9223372036854775808> : vector<4xi64>, static_strides = dense<-9223372036854775808> : vector<2xi64>