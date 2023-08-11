// RUN: mo-opt --symbolic-optimize %s | FileCheck %s

mo.graph @conv_to_matmul_static_dims(
  %input: !mo.tensor<[32, 10, 10, 3], f32>,
  %filter: !mo.tensor<[1, 1, 3, 9], f32>
) -> !mo.tensor<[32, 10, 10, 9], f32> {

  %ones = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>

  %res = mo.conv(%input, %filter) [strides = %ones, dilations = %ones] : (
    !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[1, 1, 3, 9], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[32, 10, 10, 9], f32>

  mo.output %res : !mo.tensor<[32, 10, 10, 9], f32>
}

// CHECK:      mo.graph @conv_to_matmul_static_dims(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[32, 10, 10, 3], f32>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[1, 1, 3, 9], f32>
// CHECK-SAME: ) -> !mo.tensor<[32, 10, 10, 9], f32> no_inline {
// Reshapes for input/filter into matmul input/weight shape
// CHECK-NEXT:   %[[MATMUL_INPUT:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[32, 10, 10, 3], f32> -> !mo.tensor<[3200, 3], f32>
// CHECK-NEXT:   %[[MATMUL_WEIGHT:.*]] = mo.static.reshape(%[[FILTER]]) : !mo.tensor<[1, 1, 3, 9], f32> -> !mo.tensor<[3, 9], f32>
// Matmul
// CHECK-NEXT:   %[[MATMUL:.*]] = mo.matmul(%[[MATMUL_INPUT]], %[[MATMUL_WEIGHT]])
// CHECK-SAME:     : (!mo.tensor<[3200, 3], f32>, !mo.tensor<[3, 9], f32>) -> !mo.tensor<[3200, 9], f32>
// Reshape for matmul result back to conv output shape
// CHECK-NEXT:   %[[RES:.*]] = mo.static.reshape(%[[MATMUL]]) : !mo.tensor<[3200, 9], f32> -> !mo.tensor<[32, 10, 10, 9], f32>
// CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[32, 10, 10, 9], f32>


mo.graph @conv_to_matmul_dynamic_dims<N, H, W, CI, CO>(
  %input: !mo.tensor<[N, H, W, CI], f32>,
  %filter: !mo.tensor<[1, 1, CI, CO], f32>
) -> !mo.tensor<[N, H, W, CO], f32> {

  %ones = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>

  %res = mo.conv(%input, %filter) [strides = %ones, dilations = %ones] : (
    !mo.tensor<[N, H, W, CI], f32>, !mo.tensor<[1, 1, CI, CO], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[N, H, W, CO], f32>

  mo.output %res : !mo.tensor<[N, H, W, CO], f32>
}

// CHECK:      mo.graph @conv_to_matmul_dynamic_dims<[[N:.*]], [[H:.*]], [[W:.*]], [[CI:.*]], [[CO:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[N]], [[H]], [[W]], [[CI]]], f32>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[1, 1, [[CI]], [[CO]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[N]], [[H]], [[W]], [[CO]]], f32> no_inline {
// Reshapes for input/filter into matmul input/weight shape
// CHECK-NEXT:   %[[MATMUL_INPUT:.*]] = mo.static.reshape(%[[INPUT]])
// CHECK-SAME:     : !mo.tensor<[[[N]], [[H]], [[W]], [[CI]]], f32> -> !mo.tensor<[mul_nuw([[H]], [[N]], [[W]]), [[CI]]], f32>
// CHECK-NEXT:   %[[MATMUL_WEIGHT:.*]] = mo.static.reshape(%[[FILTER]])
// CHECK-SAME:     : !mo.tensor<[1, 1, [[CI]], [[CO]]], f32> -> !mo.tensor<[[[CI]], [[CO]]], f32>
// Matmul
// CHECK-NEXT:   %[[MATMUL:.*]] = mo.matmul(%[[MATMUL_INPUT]], %[[MATMUL_WEIGHT]]) : (
// CHECK-SAME:     !mo.tensor<[mul_nuw([[H]], [[N]], [[W]]), [[CI]]], f32>, !mo.tensor<[[[CI]], [[CO]]], f32>
// CHECK-SAME:   ) -> !mo.tensor<[mul_nuw([[H]], [[N]], [[W]]), [[CO]]], f32>
// Reshape for matmul result back to conv output shape
// CHECK-NEXT:   %[[RES:.*]] = mo.static.reshape(%[[MATMUL]])
// CHECK-SAME:     : !mo.tensor<[mul_nuw([[H]], [[N]], [[W]]), [[CO]]], f32> -> !mo.tensor<[[[N]], [[H]], [[W]], [[CO]]], f32>
// CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[[[N]], [[H]], [[W]], [[CO]]], f32>


mo.graph @conv_to_matmul_non_1x1_filter(
  %input: !mo.tensor<[32, 10, 10, 3], f32>,
  %filter: !mo.tensor<[3, 3, 3, 9], f32>
) -> !mo.tensor<[32, 8, 8, 9], f32> {

  %ones = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>

  %res = mo.conv(%input, %filter) [strides = %ones, dilations = %ones] : (
    !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[3, 3, 3, 9], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[32, 8, 8, 9], f32>

  mo.output %res : !mo.tensor<[32, 8, 8, 9], f32>
}

// CHECK:      mo.graph @conv_to_matmul_non_1x1_filter(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[32, 10, 10, 3], f32>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[3, 3, 3, 9], f32>
// CHECK-SAME: ) -> !mo.tensor<[32, 8, 8, 9], f32> no_inline {
//
// CHECK-NEXT:   %[[ONES:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
//
// CHECK-NEXT:   %[[RES:.*]] = mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[ONES]], dilations = %[[ONES]]] : (
// CHECK-SAME:     !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[3, 3, 3, 9], f32>,
// CHECK-SAME:     !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
// CHECK-SAME:   ) -> !mo.tensor<[32, 8, 8, 9], f32>
//
// CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[32, 8, 8, 9], f32>
// CHECK-NEXT: }


mo.graph @conv_to_matmul_non_ones_strides(
  %input: !mo.tensor<[32, 10, 10, 3], f32>,
  %filter: !mo.tensor<[3, 3, 3, 9], f32>
) -> !mo.tensor<[32, 8, 4, 9], f32> {

  %strides = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %dilations = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>

  %res = mo.conv(%input, %filter) [strides = %strides, dilations = %dilations] : (
    !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[3, 3, 3, 9], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[32, 8, 4, 9], f32>

  mo.output %res : !mo.tensor<[32, 8, 4, 9], f32>
}

// CHECK:      mo.graph @conv_to_matmul_non_ones_strides(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[32, 10, 10, 3], f32>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[3, 3, 3, 9], f32>
// CHECK-SAME: ) -> !mo.tensor<[32, 8, 4, 9], f32> no_inline {
//
// CHECK-NEXT:   %[[STRIDES:.*]] = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   %[[DILATIONS:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
//
// CHECK-NEXT:   %[[RES:.*]] = mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]]] : (
// CHECK-SAME:     !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[3, 3, 3, 9], f32>,
// CHECK-SAME:     !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
// CHECK-SAME:   ) -> !mo.tensor<[32, 8, 4, 9], f32>
//
// CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[32, 8, 4, 9], f32>
// CHECK-NEXT: }


mo.graph @conv_to_matmul_non_ones_dilations(
  %input: !mo.tensor<[32, 10, 10, 3], f32>,
  %filter: !mo.tensor<[3, 3, 3, 9], f32>
) -> !mo.tensor<[32, 6, 10, 9], f32> {

  %strides = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %dilations = mo.constant {value = #M.dense_array<2, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>

  %res = mo.conv(%input, %filter) [strides = %strides, dilations = %dilations] : (
    !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[3, 3, 3, 9], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[32, 6, 10, 9], f32>

  mo.output %res : !mo.tensor<[32, 6, 10, 9], f32>
}

// CHECK:      mo.graph @conv_to_matmul_non_ones_dilations(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[32, 10, 10, 3], f32>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[3, 3, 3, 9], f32>
// CHECK-SAME: ) -> !mo.tensor<[32, 6, 10, 9], f32> no_inline {
//
// CHECK-NEXT:   %[[STRIDES:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   %[[DILATIONS:.*]] = mo.constant {value = #M.dense_array<2, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
//
// CHECK-NEXT:   %[[RES:.*]] = mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]]] : (
// CHECK-SAME:     !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[3, 3, 3, 9], f32>,
// CHECK-SAME:     !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
// CHECK-SAME:   ) -> !mo.tensor<[32, 6, 10, 9], f32>
//
// CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[32, 6, 10, 9], f32>
// CHECK-NEXT: }
