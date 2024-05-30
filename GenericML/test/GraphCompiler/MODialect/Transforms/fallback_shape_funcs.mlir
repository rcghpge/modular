// RUN: mo-opt %s --add-fallback-shape-funcs | FileCheck %s

mo.graph @arg_max_fallback_shape_func<D0, D1 -> D2, D3>(
  %input : !mo.tensor<[D0, D1], si64>,
  %axis: !mo.tensor<[], si64>
) -> !mo.tensor<[D2, D3], si64> {

  %res = mo.arg_max<() -> D4, D5>(%input, %axis) :
    (!mo.tensor<[D0, D1], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[D4, D5], si64>

  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], si64>
}
// CHECK-LABEL: mo.graph @arg_max_fallback_shape_func
// CHECK-SAME: <[[D0:.*]], [[D1:.*]] -> [[D2:.*]], [[D3:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], si64>
// CHECK-SAME:   %[[AXIS:.*]]: !mo.tensor<[], si64>
// CHECK-SAME: ) -> !mo.tensor<[[[D2]], [[D3]]], si64> {
//
// CHECK-NEXT: mo.invoke_shape_func["reduce_shape"]<() -> [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[AXIS]])
// CHECK-SAME:   {dataDeptTensors = [1 : i32]} : (!mo.tensor<[[[D0]], [[D1]]], si64>, !mo.tensor<[], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.arg_max(%[[INPUT]], %[[AXIS]]) :
// CHECK-SAME:   (!mo.tensor<[[[D0]], [[D1]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D4]], [[D5]]], si64>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]]], si64>

mo.graph @avg_pool_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter_shape: !mo.tensor<[2], si64>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {

  %res = mo.avg_pool<() -> D4, D5, D6, D7>(%input) [filter_shape = %filter_shape, strides = %strides, dilations = %dilations, paddings = %paddings] {count_boundary = false} :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
    -> !mo.tensor<[D4, D5, D6, D7], f32>

  mo.output<D4, D5, D6, D7> %res : !mo.tensor<[D4, D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph @avg_pool_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT:.*]]: !mo.tensor<[32, 10, 20, 3], f32>, %[[FILTER_SHAPE:arg.]]: !mo.tensor<[2], si64>, %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>, %[[PADDINGS:.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["pool_shape"]<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[FILTER_SHAPE]], %[[STRIDES]], %[[DILATIONS]], %[[PADDINGS]])
// CHECK-SAME:   {dataDeptTensors = [1 : i32, 2 : i32, 3 : i32, 4 : i32]} :
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.avg_pool(%[[INPUT]]) [filter_shape = %[[FILTER_SHAPE]], strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]]]
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>


mo.graph @avg_pool_ceil_mode_true_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter_shape: !mo.tensor<[2], si64>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {

  %res = mo.avg_pool_ceil_mode_true<() -> D4, D5, D6, D7>(%input) [filter_shape = %filter_shape, strides = %strides, dilations = %dilations, paddings = %paddings] {count_boundary = false} :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
    -> !mo.tensor<[D4, D5, D6, D7], f32>

  mo.output<D4, D5, D6, D7> %res : !mo.tensor<[D4, D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph @avg_pool_ceil_mode_true_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT:.*]]: !mo.tensor<[32, 10, 20, 3], f32>, %[[FILTER_SHAPE:arg.]]: !mo.tensor<[2], si64>, %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>, %[[PADDINGS:.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["pool_shape_ceil"]<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[FILTER_SHAPE]], %[[STRIDES]], %[[DILATIONS]], %[[PADDINGS]])
// CHECK-SAME:   {dataDeptTensors = [1 : i32, 2 : i32, 3 : i32, 4 : i32]} :
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.avg_pool_ceil_mode_true(%[[INPUT]]) [filter_shape = %[[FILTER_SHAPE]], strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]]]
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>


mo.graph @batch_matmul<D0, D1, D2, D3, D4, D5 -> D6, D7, D8>(
  %lhs: !mo.tensor<[D0, D1, D2], f32>,
  %rhs: !mo.tensor<[D3, D4, D5], f32>
) -> !mo.tensor<[D6, D7, D8], f32> {

  %res = mo.batch_matmul<() -> D9, D10, D11>(%lhs, %rhs) :
    (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<[D9, D10, D11], f32>

  mo.output<D9, D10, D11> %res : !mo.tensor<[D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph @batch_matmul
// CHECK-SAME: <[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]] -> [[D6:.*]], [[D7:.*]], [[D8:.*]]>(
// CHECK-SAME:   %[[LHS:.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>,
// CHECK-SAME:   %[[RHS:.*]]: !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D6]], [[D7]], [[D8]]], f32> {
//
// CHECK-NEXT:   mo.invoke_shape_func["batch_matmul_shape"]<() -> [[D9:.*]], [[D10:.*]], [[D11:.*]]>(%[[LHS]], %[[RHS]])
// CHECK-SAME:     {dataDeptTensors = []}
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>)
//
// CHECK-NEXT:   %[[RES:.*]] = mo.batch_matmul(%[[LHS]], %[[RHS]]) :
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>) -> !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32>
//
// CHECK-NEXT:   mo.output<[[D9]], [[D10]], [[D11]]> %[[RES]] : !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32>


mo.graph @max_pool_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter_shape: !mo.tensor<[2], si64>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {

  %res = mo.max_pool<() -> D4, D5, D6, D7>(%input) [filter_shape = %filter_shape, strides = %strides, dilations = %dilations, paddings = %paddings] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
    -> !mo.tensor<[D4, D5, D6, D7], f32>

  mo.output<D4, D5, D6, D7> %res : !mo.tensor<[D4, D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph @max_pool_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT:.*]]: !mo.tensor<[32, 10, 20, 3], f32>, %[[FILTER_SHAPE:arg.]]: !mo.tensor<[2], si64>, %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>, %[[PADDINGS:.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["pool_shape"]<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[FILTER_SHAPE]], %[[STRIDES]], %[[DILATIONS]], %[[PADDINGS]])
// CHECK-SAME:   {dataDeptTensors = [1 : i32, 2 : i32, 3 : i32, 4 : i32]} :
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.max_pool(%[[INPUT]]) [filter_shape = %[[FILTER_SHAPE]], strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]]]
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>


mo.graph @max_pool_ceil_mode_true_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter_shape: !mo.tensor<[2], si64>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {

  %res = mo.max_pool_ceil_mode_true<() -> D4, D5, D6, D7>(%input) [filter_shape = %filter_shape, strides = %strides, dilations = %dilations, paddings = %paddings] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
    -> !mo.tensor<[D4, D5, D6, D7], f32>

  mo.output<D4, D5, D6, D7> %res : !mo.tensor<[D4, D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph @max_pool_ceil_mode_true_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT:.*]]: !mo.tensor<[32, 10, 20, 3], f32>, %[[FILTER_SHAPE:arg.]]: !mo.tensor<[2], si64>, %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>, %[[PADDINGS:.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["pool_shape_ceil"]<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[FILTER_SHAPE]], %[[STRIDES]], %[[DILATIONS]], %[[PADDINGS]])
// CHECK-SAME:   {dataDeptTensors = [1 : i32, 2 : i32, 3 : i32, 4 : i32]} :
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.max_pool_ceil_mode_true(%[[INPUT]]) [filter_shape = %[[FILTER_SHAPE]], strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]]]
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>


mo.graph @layer_norm<D0, D1 -> D2, D3>(
  %input: !mo.tensor<[D0, D1], f32>,
  %gamma: !mo.tensor<[], f32>,
  %beta: !mo.tensor<[], f32>,
  %epsilon: !mo.tensor<[], f32>
) -> !mo.tensor<[D2, D3], f32> {

  %res = mo.layer_norm<() -> D4, D5>(%input, %gamma, %beta, %epsilon) :
    (!mo.tensor<[D0, D1], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D4, D5], f32>

  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}
// CHECK-LABEL: mo.graph @layer_norm
// CHECK-SAME: <[[D0:.*]], [[D1:.*]] -> [[D2:.*]], [[D3:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>,
// CHECK-SAME:   %[[GAMMA:arg1]]: !mo.tensor<[], f32>,
// CHECK-SAME:   %[[BETA:arg2]]: !mo.tensor<[], f32>,
// CHECK-SAME:   %[[EPSILON:.*]]: !mo.tensor<[], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D2]], [[D3]]], f32> {
//
// CHECK-NEXT:   mo.invoke_shape_func["layer_norm_shape"]<() -> [[D4:.*]], [[D5:.*]]>(
// CHECK-SAME:     %[[INPUT]], %[[GAMMA]], %[[BETA]], %[[EPSILON]]) {dataDeptTensors = []} :
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>)
//
// CHECK-NEXT:   %[[RES:.*]] = mo.layer_norm(%[[INPUT]], %[[GAMMA]], %[[BETA]], %[[EPSILON]]) :
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>)
// CHECK-SAME:     -> !mo.tensor<[[[D4]], [[D5]]], f32>
//
// CHECK-NEXT:   mo.output<[[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]]], f32>


mo.graph @matrix_solve<D0, D1, D2, D3, D4, D5 -> D6, D7, D8>(
  %lhs: !mo.tensor<[D0, D1, D2], f32>,
  %rhs: !mo.tensor<[D3, D4, D5], f32>
) -> !mo.tensor<[D6, D7, D8], f32> {

  %res = mo.linalg.solve<() -> D9, D10, D11>(%lhs, %rhs) :
    (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<[D9, D10, D11], f32>

  mo.output<D9, D10, D11> %res : !mo.tensor<[D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph @matrix_solve
// CHECK-SAME: <[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]] -> [[D6:.*]], [[D7:.*]], [[D8:.*]]>(
// CHECK-SAME:   %[[LHS:.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>,
// CHECK-SAME:   %[[RHS:.*]]: !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D6]], [[D7]], [[D8]]], f32> {
//
// CHECK-NEXT:   mo.invoke_shape_func["matrix_solve_shape"]<() -> [[D9:.*]], [[D10:.*]], [[D11:.*]]>(%[[LHS]], %[[RHS]])
// CHECK-SAME:     {dataDeptTensors = []}
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>)
//
// CHECK-NEXT:   %[[RES:.*]] = mo.linalg.solve(%[[LHS]], %[[RHS]]) :
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>) -> !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32>
//
// CHECK-NEXT:   mo.output<[[D9]], [[D10]], [[D11]]> %[[RES]] : !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32>


mo.graph @broadcast_to_fallback_shape_func<() -> D0, D1, D2>(
  %input: !mo.tensor<[1, 3], f32>,
  %target_shape: !mo.tensor<[3], si32>
) -> !mo.tensor<[D0, D1, D2], f32> {

  %res = mo.broadcast_to<() -> D3, D4, D5>(%input, %target_shape) :
    (!mo.tensor<[1, 3], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[D3, D4, D5], f32>

  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK-LABEL: mo.graph @broadcast_to_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[1, 3], f32>,
// CHECK-SAME:   %[[TARGET_SHAPE:.*]]: !mo.tensor<[3], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["broadcast_to_shape"]<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[TARGET_SHAPE]])
// CHECK-SAME:   {dataDeptTensors = [1 : i32]} :
// CHECK-SAME:   (!mo.tensor<[1, 3], f32>, !mo.tensor<[3], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.broadcast_to(%[[INPUT]], %[[TARGET_SHAPE]]) :
// CHECK-SAME:   (!mo.tensor<[1, 3], f32>, !mo.tensor<[3], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
//
// CHECK-NEXT: mo.output<[[D3]], [[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>


mo.graph @broadcast_shape_fallback_shape_func<() -> D0>(
  %lhs: !mo.tensor<[1], si32>,
  %rhs: !mo.tensor<[2], si32>
) -> !mo.tensor<[D0], si32> {

  %res = mo.broadcast_shape<() -> D1>(%lhs, %rhs) :
    (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[D1], si32>

  mo.output<D1> %res : !mo.tensor<[D1], si32>
}
// CHECK-LABEL: mo.graph @broadcast_shape_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]]>(
// CHECK-SAME:   %[[LHS:.*]]: !mo.tensor<[1], si32>,
// CHECK-SAME:   %[[RHS:.*]]: !mo.tensor<[2], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]]], si32> {
//
// CHECK-NEXT: mo.invoke_shape_func["broadcast_shape_shape"]<() -> [[D1:.*]]>(%[[LHS]], %[[RHS]])
// CHECK-SAME:   {dataDeptTensors = []} :
// CHECK-SAME:   (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.broadcast_shape(%[[LHS]], %[[RHS]]) :
// CHECK-SAME:   (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D1]]], si32>
//
// CHECK-NEXT: mo.output<[[D1]]> %[[RES]] : !mo.tensor<[[[D1]]], si32>


mo.graph @concat_variadic_input<D0, D1, D2, D3 -> D4, D5>(
  %input1: !mo.tensor<[D0, D1], f32>,
  %input2: !mo.tensor<[D2, D3], f32>,
  %axis: !mo.tensor<[], si32>
) -> !mo.tensor<[D4, D5], f32> {

  %res = mo.concat<() -> D6, D7>[%axis : !mo.tensor<[], si32>] (%input1, %input2) :
    (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D6, D7], f32>

  mo.output<D6, D7> %res : !mo.tensor<[D6, D7], f32>
}
// CHECK-LABEL: mo.graph @concat_variadic_input
// CHECK-SAME: <[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]] -> [[D4:.*]], [[D5:.*]]>(
// CHECK-SAME:   %[[INPUT1:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>,
// CHECK-SAME:   %[[INPUT2:.*]]: !mo.tensor<[[[D2]], [[D3]]], f32>,
// CHECK-SAME:   %[[AXIS:.*]]: !mo.tensor<[], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D4]], [[D5]]], f32> {
//
// CHECK-NEXT:   mo.invoke_shape_func["concat_shape"]<() -> [[D6:.*]], [[D7:.*]]>(%[[AXIS]], %[[INPUT1]], %[[INPUT2]])
// CHECK-SAME:     {dataDeptTensors = [0 : i32], kgenParams = {input_rank = 2 : index, input_type = f32}} :
// CHECK-SAME:     (!mo.tensor<[], si32>, !mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>)
//
// CHECK-NEXT:   %[[RES:.*]] = mo.concat[%[[AXIS]] : !mo.tensor<[], si32>] (%[[INPUT1]], %[[INPUT2]]) :
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D6]], [[D7]]], f32>
//
// CHECK-NEXT:   mo.output<[[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D6]], [[D7]]], f32>


mo.graph @concat_from_list<() -> D0, D1>(
  %input: !mo.list<!mo.tensor<[?, ?], f32>>,
  %axis: !mo.tensor<[], si32>
) -> !mo.tensor<[D0, D1], f32> {

  %res = mo.concat_from_list<() -> D2, D3>[%axis : !mo.tensor<[], si32>](%input) :
    !mo.list<!mo.tensor<[?, ?], f32>> -> !mo.tensor<[D2, D3], f32>

  mo.output<D2, D3> %res : !mo.tensor<[D2, D3], f32>
}
// CHECK-LABEL: mo.graph @concat_from_list
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.list<!mo.tensor<[?, ?], f32>>,
// CHECK-SAME:   %[[AXIS:.*]]: !mo.tensor<[], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]]], f32> {
//
// CHECK-NEXT:   mo.invoke_shape_func["concat_from_list_shape"]<() -> [[D2:.*]], [[D3:.*]]>(%[[INPUT]], %[[AXIS]])
// CHECK-SAME:     {dataDeptTensors = [0 : i32, 1 : i32]} :
// CHECK-SAME:     (!mo.list<!mo.tensor<[?, ?], f32>>, !mo.tensor<[], si32>)
//
// CHECK-NEXT:   %[[RES:.*]] = mo.concat_from_list[%[[AXIS]] : !mo.tensor<[], si32>] (%[[INPUT]]) :
// CHECK-SAME:     !mo.list<!mo.tensor<[?, ?], f32>> -> !mo.tensor<[[[D2]], [[D3]]], f32>
//
// CHECK-NEXT:   mo.output<[[D2]], [[D3]]> %[[RES]] : !mo.tensor<[[[D2]], [[D3]]], f32>


mo.graph @conv_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter: !mo.tensor<[2, 2, 3, 1], f32>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>,
  %num_groups: !mo.tensor<[], si64>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {

  %res = mo.conv<() -> D4, D5, D6, D7>(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %paddings, num_groups = %num_groups] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2, 2, 3, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>)
    -> !mo.tensor<[D4, D5, D6, D7], f32>

  mo.output<D4, D5, D6, D7> %res : !mo.tensor<[D4, D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph @conv_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT:.*]]: !mo.tensor<[32, 10, 20, 3], f32>, %[[FILTER:.*]]: !mo.tensor<[2, 2, 3, 1], f32>, %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>, %[[PADDINGS:.*]]: !mo.tensor<[4], si64>, %[[NUM_GROUPS:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["conv_shape"]<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[FILTER]], %[[STRIDES]], %[[DILATIONS]], %[[PADDINGS]], %[[NUM_GROUPS]])
// CHECK-SAME:   {dataDeptTensors = [2 : i32, 3 : i32, 4 : i32, 5 : i32]} :
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2, 2, 3, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], num_groups = %[[NUM_GROUPS]]]
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2, 2, 3, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>


mo.graph @conv_transpose_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter: !mo.tensor<[2, 2, 3, 1], f32>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>,
  %out_paddings: !mo.tensor<[2], si64>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {

  %res = mo.conv_transpose<() -> D4, D5, D6, D7>(%input, %filter)
    [strides = %strides, dilations = %dilations, paddings = %paddings, output_paddings = %out_paddings] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2, 2, 3, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[D4, D5, D6, D7], f32>

  mo.output<D4, D5, D6, D7> %res : !mo.tensor<[D4, D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph @conv_transpose_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT:.*]]: !mo.tensor<[32, 10, 20, 3], f32>, %[[FILTER:.*]]: !mo.tensor<[2, 2, 3, 1], f32>, %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>, %[[PADDINGS:.*]]: !mo.tensor<[4], si64>, %[[OUTPUT_PADDINGS:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["conv_transpose_shape"]<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[FILTER]], %[[STRIDES]], %[[DILATIONS]], %[[PADDINGS]], %[[OUTPUT_PADDINGS]])
// CHECK-SAME:   {dataDeptTensors = [2 : i32, 3 : i32, 4 : i32, 5 : i32]} :
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2, 2, 3, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.conv_transpose(%[[INPUT]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], output_paddings = %[[OUTPUT_PADDINGS]]]
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2, 2, 3, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>


mo.graph @gather_fallback_shape_func<D0, D1 -> D2, D3, D4>(
  %input : !mo.tensor<[10, D0], f32>,
  %indices : !mo.tensor<[D1, 20], si32>,
  %axis: !mo.tensor<[], si32>
) -> !mo.tensor<[D2, D3, D4], f32> {

  %res = mo.gather<() -> D5, D6, D7>(%input, %indices, %axis) :
    (!mo.tensor<[10, D0], f32>, !mo.tensor<[D1, 20], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[D5, D6, D7], f32>

  mo.output<D5, D6, D7> %res : !mo.tensor<[D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph @gather_fallback_shape_func
// CHECK-SAME: <[[D0:.*]], [[D1:.*]] -> [[D2:.*]], [[D3:.*]], [[D4:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[10, [[D0]]], f32>
// CHECK-SAME:   %[[INDICES:.*]]: !mo.tensor<[[[D1]], 20], si32>
// CHECK-SAME:   %[[AXIS:.*]]: !mo.tensor<[], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["gather_shape"]<() -> [[D5:.*]], [[D6:.*]], [[D7:.*]]>(%[[INPUT]], %[[INDICES]], %[[AXIS]])
// CHECK-SAME:   {dataDeptTensors = [2 : i32]} :
// CHECK-SAME:   (!mo.tensor<[10, [[D0]]], f32>, !mo.tensor<[[[D1]], 20], si32>, !mo.tensor<[], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.gather(%[[INPUT]], %[[INDICES]], %[[AXIS]]) :
// CHECK-SAME:   (!mo.tensor<[10, [[D0]]], f32>, !mo.tensor<[[[D1]], 20], si32>, !mo.tensor<[], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D5]], [[D6]], [[D7]]], f32>


mo.graph @gather_nd_fallback_shape_func<D0, D1 -> D2, D3>(
  %input : !mo.tensor<[D0, D1, 20], f32>,
  %indices : !mo.tensor<[D0, 1], si32>
) -> !mo.tensor<[D2, D3], f32> {

  %res = mo.gather_nd<() -> D4, D5>(%input, %indices)
    { batchDims = 1 } :
    (!mo.tensor<[D0, D1, 20], f32>, !mo.tensor<[D0, 1], si32>) -> !mo.tensor<[D4, D5], f32>

  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}
// CHECK-LABEL: mo.graph @gather_nd_fallback_shape_func
// CHECK-SAME: <[[D0:.*]], [[D1:.*]] -> [[D2:.*]], [[D3:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]], 20], f32>,
// CHECK-SAME:   %[[INDICES:.*]]: !mo.tensor<[[[D0]], 1], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["gather_nd_shape"]<() -> [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[INDICES]])
// CHECK-SAME:   {dataDeptTensors = [], kgenParams = {batch_dims = 1 : i64}} :
// CHECK-SAME:   (!mo.tensor<[[[D0]], [[D1]], 20], f32>, !mo.tensor<[[[D0]], 1], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.gather_nd(%[[INPUT]], %[[INDICES]])
// CHECK-SAME:   {batchDims = 1 : i64} :
// CHECK-SAME:   (!mo.tensor<[[[D0]], [[D1]], 20], f32>, !mo.tensor<[[[D0]], 1], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]]], f32>


mo.graph @pad_constant_fallback_shape_func<() -> D0, D1, D2>(
  %input : !mo.tensor<[1, 2, 3], f32>,
  %paddings : !mo.tensor<[6], si32>,
  %scalar: !mo.tensor<[], f32>
) -> !mo.tensor<[D0, D1, D2], f32> {

  %res = mo.pad.constant<() -> D3, D4, D5>(%input, %paddings, %scalar) :
    (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[D3, D4, D5], f32>

  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK-LABEL: mo.graph @pad_constant_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[1, 2, 3], f32>,
// CHECK-SAME:   %[[PADDINGS:.*]]: !mo.tensor<[6], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["pad_shape"]<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[PADDINGS]])
// CHECK-SAME:   {dataDeptTensors = [1 : i32]} :
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.pad.constant(%[[INPUT]], %[[PADDINGS]], %[[AXIS]]) :
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>, !mo.tensor<[], f32>)
// CHECK-SAME:   -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
//
// CHECK-NEXT: mo.output<[[D3]], [[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>

mo.graph @pad_reflect_fallback_shape_func<() -> D0, D1, D2>(
  %input : !mo.tensor<[1, 2, 3], f32>,
  %paddings : !mo.tensor<[6], si32>
) -> !mo.tensor<[D0, D1, D2], f32> {

  %res = mo.pad.reflect<() -> D3, D4, D5>(%input, %paddings) :
    (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>) -> !mo.tensor<[D3, D4, D5], f32>

  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK-LABEL: mo.graph @pad_reflect_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[1, 2, 3], f32>,
// CHECK-SAME:   %[[PADDINGS:.*]]: !mo.tensor<[6], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["pad_shape"]<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[PADDINGS]])
// CHECK-SAME:   {dataDeptTensors = [1 : i32]} :
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.pad.reflect(%[[INPUT]], %[[PADDINGS]]) :
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
//
// CHECK-NEXT: mo.output<[[D3]], [[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>

mo.graph @pad_repeat_fallback_shape_func<() -> D0, D1, D2>(
  %input : !mo.tensor<[1, 2, 3], f32>,
  %paddings : !mo.tensor<[6], si32>
) -> !mo.tensor<[D0, D1, D2], f32> {

  %res = mo.pad.repeat<() -> D3, D4, D5>(%input, %paddings) :
    (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>) -> !mo.tensor<[D3, D4, D5], f32>

  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK-LABEL: mo.graph @pad_repeat_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[1, 2, 3], f32>,
// CHECK-SAME:   %[[PADDINGS:.*]]: !mo.tensor<[6], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["pad_shape"]<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[PADDINGS]])
// CHECK-SAME:   {dataDeptTensors = [1 : i32]} :
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.pad.repeat(%[[INPUT]], %[[PADDINGS]]) :
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
//
// CHECK-NEXT: mo.output<[[D3]], [[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>


mo.graph @range(%start: !mo.tensor<[], si32>, %stop: !mo.tensor<[], si32>, %step: !mo.tensor<[], si32>) -> !mo.tensor<[D1], si32> {
  %out = mo.range<() -> D1>(%start, %stop, %step) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[D1], si32>
  mo.output %out: !mo.tensor<[D1], si32>
}
// CHECK-LABEL: mo.graph @range
// CHECK:(%[[START:.*]]: !mo.tensor<[], si32>, %[[STOP:.*]]: !mo.tensor<[], si32>, %[[STEP:.*]]: !mo.tensor<[], si32>)
// CHECK-NEXT: mo.invoke_shape_func["range_shape"]<() -> [[RANGE_PARAM:.*]]>(%[[START]], %[[STOP]], %[[STEP]]) {dataDeptTensors = [0 : i32, 1 : i32, 2 : i32]} : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>)
// CHECK-NEXT: mo.range(%[[START]], %[[STOP]], %[[STEP]]) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RANGE_PARAM]]], si32>

mo.graph @reduce_add_fallback_shape_func<D0, D1 -> D2, D3>(
  %input : !mo.tensor<[D0, D1], f32>,
  %axis: !mo.tensor<[], si32>
) -> !mo.tensor<[D2, D3], f32> {

  %res = mo.reduce.add<() -> D4, D5>(%input, %axis) :
    (!mo.tensor<[D0, D1], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[D4, D5], f32>

  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}
// CHECK-LABEL: mo.graph @reduce_add_fallback_shape_func
// CHECK-SAME: <[[D0:.*]], [[D1:.*]] -> [[D2:.*]], [[D3:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>
// CHECK-SAME:   %[[AXIS:.*]]: !mo.tensor<[], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["reduce_shape"]<() -> [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[AXIS]])
// CHECK-SAME:   {dataDeptTensors = [1 : i32]} : (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.reduce.add(%[[INPUT]], %[[AXIS]]) :
// CHECK-SAME:   (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[D4]], [[D5]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]]], f32>


mo.graph @reshape_fallback_shape_func<() -> D0, D1, D2>(
  %input: !mo.tensor<[1, 3], f32>,
  %target_shape: !mo.tensor<[3], si32>
) -> !mo.tensor<[D0, D1, D2], f32> {

  %res = mo.reshape<() -> D3, D4, D5>(%input, %target_shape) :
    (!mo.tensor<[1, 3], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[D3, D4, D5], f32>

  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK-LABEL: mo.graph @reshape_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[1, 3], f32>,
// CHECK-SAME:   %[[TARGET_SHAPE:.*]]: !mo.tensor<[3], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["reshape_shape"]<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[TARGET_SHAPE]])
// CHECK-SAME:   {dataDeptTensors = [1 : i32]} :
// CHECK-SAME:   (!mo.tensor<[1, 3], f32>, !mo.tensor<[3], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.reshape(%[[INPUT]], %[[TARGET_SHAPE]]) :
// CHECK-SAME:   (!mo.tensor<[1, 3], f32>, !mo.tensor<[3], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
//
// CHECK-NEXT: mo.output<[[D3]], [[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>


mo.graph @scatter_fallback_shape_func<D0, D1, D2, D3, D4, D5 -> D6, D7>(
  %input: !mo.tensor<[D0, D1], f32>,
  %updates: !mo.tensor<[D2, D3], f32>,
  %indices: !mo.tensor<[D4, D5], si32>,
  %axis: !mo.tensor<[], si32>
) -> !mo.tensor<[D6, D7], f32> {

  %res = mo.scatter<() -> D8, D9>(%input, %updates, %indices, %axis) : (
    !mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>, !mo.tensor<[D4, D5], si32>, !mo.tensor<[], si32>
  ) -> !mo.tensor<[D8, D9], f32>

  mo.output<D8, D9> %res : !mo.tensor<[D8, D9], f32>
}
// CHECK-LABEL: mo.graph @scatter_fallback_shape_func
// CHECK-SAME:    <[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]] -> [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>,
// CHECK-SAME:    %[[UPDATES:.*]]: !mo.tensor<[[[D2]], [[D3]]], f32>,
// CHECK-SAME:    %[[INDICES:.*]]: !mo.tensor<[[[D4]], [[D5]]], si32>,
// CHECK-SAME:    %[[AXIS:.*]]: !mo.tensor<[], si32>
// CHECK-SAME:  ) -> !mo.tensor<[[[D6]], [[D7]]], f32> {
//
// CHECK-NEXT:   mo.invoke_shape_func["scatter_shape"]<() -> [[D8:.*]], [[D9:.*]]>(%[[INPUT]], %[[UPDATES]], %[[INDICES]], %[[AXIS]])
// CHECK-SAME:     {dataDeptTensors = [3 : i32]} :
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]]], si32>, !mo.tensor<[], si32>)
//
// CHECK-NEXT:   %[[RES:.*]] = mo.scatter(%[[INPUT]], %[[UPDATES]], %[[INDICES]], %[[AXIS]]) : (
// CHECK-SAME:     !mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]]], si32>, !mo.tensor<[], si32>
// CHECK-SAME:   ) -> !mo.tensor<[[[D8]], [[D9]]], f32>
//
// CHECK-NEXT:   mo.output<[[D8]], [[D9]]> %[[RES]] : !mo.tensor<[[[D8]], [[D9]]], f32>
// CHECK-NEXT: }


mo.graph @scatter_nd_fallback_shape_func<D0, D1, D2, D3, D4, D5 -> D6, D7>(
  %input: !mo.tensor<[D0, D1], f32>,
  %updates: !mo.tensor<[D2, D3], f32>,
  %indices: !mo.tensor<[D4, D5], si32>
) -> !mo.tensor<[D6, D7], f32> {

  %res = mo.scatter_nd<() -> D8, D9>(%input, %updates, %indices) : (
    !mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>, !mo.tensor<[D4, D5], si32>
  ) -> !mo.tensor<[D8, D9], f32>

  mo.output<D8, D9> %res : !mo.tensor<[D8, D9], f32>
}
// CHECK-LABEL: mo.graph @scatter_nd_fallback_shape_func
// CHECK-SAME:    <[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]] -> [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>,
// CHECK-SAME:    %[[UPDATES:.*]]: !mo.tensor<[[[D2]], [[D3]]], f32>,
// CHECK-SAME:    %[[INDICES:.*]]: !mo.tensor<[[[D4]], [[D5]]], si32>
// CHECK-SAME:  ) -> !mo.tensor<[[[D6]], [[D7]]], f32> {
//
// CHECK-NEXT:   mo.invoke_shape_func["scatter_nd_shape"]<() -> [[D8:.*]], [[D9:.*]]>(%[[INPUT]], %[[UPDATES]], %[[INDICES]])
// CHECK-SAME:     {dataDeptTensors = []} :
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]]], si32>)
//
// CHECK-NEXT:   %[[RES:.*]] = mo.scatter_nd(%[[INPUT]], %[[UPDATES]], %[[INDICES]]) : (
// CHECK-SAME:     !mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]]], si32>
// CHECK-SAME:   ) -> !mo.tensor<[[[D8]], [[D9]]], f32>
//
// CHECK-NEXT:   mo.output<[[D8]], [[D9]]> %[[RES]] : !mo.tensor<[[[D8]], [[D9]]], f32>
// CHECK-NEXT: }


mo.graph @slice_fallback_shape_func<() -> D0, D1, D2>(
  %input: !mo.tensor<[1, 2, 3], f32>,
  %start: !mo.tensor<[3], si32>,
  %stop: !mo.tensor<[3], si32>,
  %step: !mo.tensor<[3], si32>
) -> !mo.tensor<[D0, D1, D2], f32> {

  %res = mo.slice<() -> D3, D4, D5>(%input, %start, %stop, %step) :
    (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>)
    -> !mo.tensor<[D3, D4, D5], f32>

  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK-LABEL: mo.graph @slice_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]]>(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[1, 2, 3], f32>,
// CHECK-SAME: %[[START:arg1]]: !mo.tensor<[3], si32>,
// CHECK-SAME: %[[STOP:arg2]]: !mo.tensor<[3], si32>,
// CHECK-SAME: %[[STEP:.*]]: !mo.tensor<[3], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["slice_shape"]<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[START]], %[[STOP]], %[[STEP]])
// CHECK-SAME:   {dataDeptTensors = [1 : i32, 2 : i32, 3 : i32]} :
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.slice(%[[INPUT]], %[[START]], %[[STOP]], %[[STEP]])
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
//
// CHECK-NEXT: mo.output<[[D3]], [[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>


mo.graph @split_variadic_output<D0 -> D1>(
  %input: !mo.tensor<[D0, 10], f32>,
  %split_sizes: !mo.tensor<[2], si64>
) -> (!mo.tensor<[D1, 10], f32>, !mo.tensor<[5, 10], f32>) {

  %axis = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>

  %res:2 = mo.split<() -> D2>[%axis : !mo.tensor<[], si32>] (%input, %split_sizes) :
    (!mo.tensor<[D0, 10], f32>, !mo.tensor<[2], si64>) ->
    (!mo.tensor<[D2, 10], f32>, !mo.tensor<[5, 10], f32>)

  mo.output<D2> %res#0, %res#1 : !mo.tensor<[D2, 10], f32>, !mo.tensor<[5, 10], f32>
}
// CHECK-LABEL: mo.graph @split
// CHECK-SAME: <[[D0:.*]] -> [[D1:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]], 10], f32>,
// CHECK-SAME:   %[[SPLIT_SIZES:.*]]: !mo.tensor<[2], si64>
// CHECK-SAME: ) -> (!mo.tensor<[[[D1]], 10], f32>, !mo.tensor<[5, 10], f32>) {
//
// CHECK-NEXT: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
//
// CHECK-NEXT: mo.invoke_shape_func["split_ith_output_shape"]<() -> [[D2:.*]], {{.*}}>(%[[INPUT]], %[[SPLIT_SIZES]], %[[AXIS]])
// CHECK-SAME:  {dataDeptTensors = [1 : i32, 2 : i32], kgenParams = {output_idx = 0 : index}} :
// CHECK-SAME:  (!mo.tensor<[[[D0]], 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si32>)
// CHECK-NEXT: mo.invoke_shape_func["split_ith_output_shape"]<() -> {{.*}}, {{.*}}>(%[[INPUT]], %[[SPLIT_SIZES]], %[[AXIS]])
// CHECK-SAME:  {dataDeptTensors = [1 : i32, 2 : i32], kgenParams = {output_idx = 1 : index}} :
// CHECK-SAME:  (!mo.tensor<[[[D0]], 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si32>)

// CHECK-NEXT: %[[RES:.*]]:2 = mo.split[%[[AXIS]] : !mo.tensor<[], si32>] (%[[INPUT]], %[[SPLIT_SIZES]]) :
// CHECK-SAME:    (!mo.tensor<[[[D0]], 10], f32>, !mo.tensor<[2], si64>) ->
// CHECK-SAME:    (!mo.tensor<[[[D2]], 10], f32>, !mo.tensor<[5, 10], f32>)

// CHECK-NEXT: mo.output<[[D2]]> %[[RES]]#0, %[[RES]]#1 : !mo.tensor<[[[D2]], 10], f32>, !mo.tensor<[5, 10], f32>


mo.graph @squeeze_shape_fallback_shape_func<D0, D1 -> D2>(
  %input:   !mo.tensor<[D0], si64>,
  %indices: !mo.tensor<[D1], si64>
) -> !mo.tensor<[D2], si64> {

  %res = mo.squeeze_shape<() -> D3>(%input, %indices) :
    (!mo.tensor<[D0], si64>, !mo.tensor<[D1], si64>) -> !mo.tensor<[D3], si64>

  mo.output<D3> %res : !mo.tensor<[D3], si64>
}
// CHECK-LABEL: mo.graph @squeeze_shape_fallback_shape_func
// CHECK-SAME: <[[D0:.*]], [[D1:.*]] -> [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]]], si64>,
// CHECK-SAME:   %[[INDICES:.*]]: !mo.tensor<[[[D1]]], si64>
// CHECK-SAME: ) -> !mo.tensor<[[[D2]]], si64> {
//
// CHECK-NEXT: mo.invoke_shape_func["squeeze_shape_shape"]<() -> [[D3:.*]]>(%[[INPUT]], %[[INDICES]])
// CHECK-SAME:   {dataDeptTensors = []} : (!mo.tensor<[[[D0]]], si64>, !mo.tensor<[[[D1]]], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.squeeze_shape(%[[INPUT]], %[[INDICES]]) :
// CHECK-SAME:   (!mo.tensor<[[[D0]]], si64>, !mo.tensor<[[[D1]]], si64>) -> !mo.tensor<[[[D3]]], si64>
//
// CHECK-NEXT: mo.output<[[D3]]> %[[RES]] : !mo.tensor<[[[D3]]], si64>


mo.graph @transpose<D1, D2 -> D3, D4>(%arg0: !mo.tensor<[D1, D2], f32>, %arg1: !mo.tensor<[2], si64>) -> !mo.tensor<[D3, D4], f32> {
  %0 = mo.rebind<() -> D5, D6>(%arg0) : !mo.tensor<[D1, D2], f32> -> !mo.tensor<[D5, D6], f32>
  %1 = mo.transpose<() -> D7, D8>(%0, %arg1) : (!mo.tensor<[D5, D6], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D7, D8], f32>
  %2 = mo.rebind(%1) : !mo.tensor<[D7, D8], f32> -> !mo.tensor<[D3, D4], f32>
  mo.output<D7, D8> %1 : !mo.tensor<[D7, D8], f32>
}
// CHECK-LABEL: mo.graph @transpose
// CHECK: (%[[I0:.*]]: !mo.tensor<[D1, D2], f32>, %[[I1:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[D3, D4], f32>
// CHECK-NEXT: %[[RB:.*]] = mo.rebind<() -> D5, D6>(%arg0) : !mo.tensor<[D1, D2], f32> -> !mo.tensor<[D5, D6], f32>
// CHECK-NEXT: mo.invoke_shape_func["transpose_shape"]<() -> D7, D8>(%[[RB]], %[[I1]]) {dataDeptTensors = [1 : i32]} : (!mo.tensor<[D5, D6], f32>, !mo.tensor<[2], si64>)
// CHECK-NEXT: %[[TRANS:.*]] = mo.transpose(%[[RB]], %[[I1]]) : (!mo.tensor<[D5, D6], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D7, D8], f32>
// CHECK: mo.output<D7, D8> %[[TRANS]] : !mo.tensor<[D7, D8], f32>


mo.graph @arg_nonzero<D0 -> D1>(%arg0: !mo.tensor<[D0, 10], f32>) -> !mo.tensor<[D1, 2], si64> {
  %0 = mo.arg_nonzero<() -> D2>(%arg0) : (!mo.tensor<[D0, 10], f32>) -> !mo.tensor<[D2, 2], si64>
  mo.output<D2> %0 : !mo.tensor<[D2, 2], si64>
}
// CHECK-LABEL: mo.graph @arg_nonzero
// CHECK:(%[[IN:.*]]: !mo.tensor<[[[IN_PARAM:.*]], 10], f32>) -> !mo.tensor<[[[OUT_PARAM:.*]], 2], si64>
// CHECK-NEXT: mo.invoke_shape_func["arg_nonzero_shape"]<() -> [[WHERE_PARAM:.*]], [[UNUSED_PARAM:.*]]>(%[[IN]]) {dataDeptTensors = [0 : i32]}
// CHECK-NEXT: %[[WHERE:.*]] = mo.arg_nonzero(%[[IN]]) : (!mo.tensor<[[[IN_PARAM]], 10], f32>) -> !mo.tensor<[[[WHERE_PARAM]], 2], si64>
// CHECK-NEXT: mo.output<[[WHERE_PARAM]]> %[[WHERE]] : !mo.tensor<[[[WHERE_PARAM]], 2], si64>

mo.graph @top_k_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %K: !mo.tensor<[], si64>,
  %axis: !mo.tensor<[], si64>,
  %sorted : !mo.tensor<[], bool>
) -> (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[D0, D1, D2, D3], si64>) {

  %res:2 = mo.top_k<() -> D8, D9, D10, D11>(%input, %K, %axis, %sorted) :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>)
    -> (!mo.tensor<[D8, D9, D10, D11], f32>, !mo.tensor<[D8, D9, D10, D11], si64>)

  mo.output<D8, D9, D10, D11> %res#0, %res#1 : !mo.tensor<[D8, D9, D10, D11], f32>, !mo.tensor<[D8, D9, D10, D11], si64>
}

// CHECK-LABEL: mo.graph @top_k_fallback_shape_func
// CHECK-NEXT:   mo.invoke_shape_func["top_k_shape"]<() -> [[D8:.*]], [[D9:.*]], [[D10:.*]], [[D11:.*]]>({{.*}}) {dataDeptTensors = [1 : i32, 2 : i32]} : (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>)
// CHECK-NEXT:   %[[VALUES:.*]], %[[INDICES:.*]] = mo.top_k(%[[INPUT:.*]], %[[K:.*]], %[[AXIS:.*]], %[[SORTED:.*]]) : (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>, !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si64>)
// CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> %[[VALUES]], %[[INDICES]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>, !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si64>

mo.graph @bottom_k_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %K: !mo.tensor<[], si64>,
  %axis: !mo.tensor<[], si64>,
  %sorted : !mo.tensor<[], bool>
) -> (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[D0, D1, D2, D3], si64>) {

  %res:2 = mo.bottom_k<() -> D8, D9, D10, D11>(%input, %K, %axis, %sorted) :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>)
    -> (!mo.tensor<[D8, D9, D10, D11], f32>, !mo.tensor<[D8, D9, D10, D11], si64>)

  mo.output<D8, D9, D10, D11> %res#0, %res#1 : !mo.tensor<[D8, D9, D10, D11], f32>, !mo.tensor<[D8, D9, D10, D11], si64>
}

// CHECK-LABEL: mo.graph @bottom_k_fallback_shape_func
// CHECK-NEXT:   mo.invoke_shape_func["top_k_shape"]<() -> [[D8:.*]], [[D9:.*]], [[D10:.*]], [[D11:.*]]>({{.*}}) {dataDeptTensors = [1 : i32, 2 : i32]} : (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>)
// CHECK-NEXT:   %[[VALUES:.*]], %[[INDICES:.*]] = mo.bottom_k(%[[INPUT:.*]], %[[K:.*]], %[[AXIS:.*]], %[[SORTED:.*]]) : (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>, !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si64>)
// CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> %[[VALUES]], %[[INDICES]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>, !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si64>

mo.graph @tile_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[2, 3, 7, 11], f32>,
  %repeats: !mo.tensor<[4], si64>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {

  %res = mo.tile<() -> D4, D5, D6, D7>(%input, %repeats) :
    (!mo.tensor<[2, 3, 7, 11], f32>, !mo.tensor<[4], si64>)
    -> !mo.tensor<[D4, D5, D6, D7], f32>

  mo.output<D4, D5, D6, D7> %res : !mo.tensor<[D4, D5, D6, D7], f32>
}

// CHECK-LABEL: mo.graph @tile_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT:.*]]: !mo.tensor<[2, 3, 7, 11], f32>, %[[REPEATS:.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["tile_shape"]<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[REPEATS]])
// CHECK-SAME:   {dataDeptTensors = [1 : i32]} :
// CHECK-SAME:   (!mo.tensor<[2, 3, 7, 11], f32>, !mo.tensor<[4], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.tile(%[[INPUT]], %[[REPEATS]])
// CHECK-SAME:   (!mo.tensor<[2, 3, 7, 11], f32>, !mo.tensor<[4], si64>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>


mo.graph @roi_align_fallback_shape_func<D0, D1, D2, D3 -> D4, D5, D6, D7>(
  %input: !mo.tensor<[D0, D1, D2, D3], f32>
) -> !mo.tensor<[D4, D5, D6, D7], f32> {

  %output_height = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %spatial_scale = mo.constant {value = #M.dense_array<1.0> : tensor<1xf32>} : !mo.tensor<[], f32>
  %sampling_ratio = mo.constant {value = #M.dense_array<2.0> : tensor<1xf32>} : !mo.tensor<[], f32>
  %rois = mo.constant {value = #M.dense_array<0.0, 0.0, 0.0, 2.0, 2.0> : tensor<5xf32>} : !mo.tensor<[1, 5], f32>

  %output = mo.roi_align<() -> D8, D9, D10, D11>(%input, %rois, %output_height, %output_height, %spatial_scale, %sampling_ratio)
    {aligned = false,  mode = "AVG"}
    : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D8, D9, D10, D11], f32>

  mo.output<D8, D9, D10, D11> %output : !mo.tensor<[D8, D9, D10, D11], f32>
}

// CHECK-LABEL: mo.graph @roi_align_fallback_shape_func
// CHECK-SAME:  <[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]] -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>
// CHECK-SAME:  ) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32> {
//
// CHECK-NEXT:  %[[OUTPUT_HEIGHT:.*]] = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT:  %[[SPATIAL_SCALE:.*]] = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<1xf32>} : !mo.tensor<[], f32>
// CHECK-NEXT:  %[[SAMPLING_RATIO:.*]] = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<1xf32>} : !mo.tensor<[], f32>
// CHECK-NEXT:  %[[ROIS:.*]] = mo.constant {value = #M.dense_array<0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 2.000000e+00> : tensor<5xf32>} : !mo.tensor<[1, 5], f32>
//
// CHECK-NEXT:  mo.invoke_shape_func["roi_align_shape"]<() -> [[D8:.*]], [[D9:.*]], [[D10:.*]], [[D11:.*]]>(
// CHECK-SAME:    %[[INPUT]], %[[ROIS]], %[[OUTPUT_HEIGHT]], %[[OUTPUT_HEIGHT]])
// CHECK-SAME:    {dataDeptTensors = [2 : i32, 3 : i32]}
// CHECK-SAME:    : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>)
//
// CHECK-NEXT:  %[[OUTPUT:.*]] = mo.roi_align(%[[INPUT]], %[[ROIS]], %[[OUTPUT_HEIGHT]], %[[OUTPUT_HEIGHT]], %[[SPATIAL_SCALE]], %[[SAMPLING_RATIO]])
// CHECK-SAME:    {aligned = false,  mode = "AVG"}
// CHECK-SAME:    : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//
// CHECK-NEXT:  mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> %[[OUTPUT]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>


mo.graph @unsqueeze_shape_fallback_shape_func<D0, D1 -> D2>(
  %input:   !mo.tensor<[D0], si64>,
  %indices: !mo.tensor<[D1], si64>
) -> !mo.tensor<[D2], si64> {

  %res = mo.unsqueeze_shape<() -> D3>(%input, %indices) :
    (!mo.tensor<[D0], si64>, !mo.tensor<[D1], si64>) -> !mo.tensor<[D3], si64>

  mo.output<D3> %res : !mo.tensor<[D3], si64>
}
// CHECK-LABEL: mo.graph @unsqueeze_shape_fallback_shape_func
// CHECK-SAME: <[[D0:.*]], [[D1:.*]] -> [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]]], si64>,
// CHECK-SAME:   %[[INDICES:.*]]: !mo.tensor<[[[D1]]], si64>
// CHECK-SAME: ) -> !mo.tensor<[[[D2]]], si64> {
//
// CHECK-NEXT: mo.invoke_shape_func["unsqueeze_shape_shape"]<() -> [[D3:.*]]>(%[[INPUT]], %[[INDICES]])
// CHECK-SAME:   {dataDeptTensors = []} : (!mo.tensor<[[[D0]]], si64>, !mo.tensor<[[[D1]]], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.unsqueeze_shape(%[[INPUT]], %[[INDICES]]) :
// CHECK-SAME:   (!mo.tensor<[[[D0]]], si64>, !mo.tensor<[[[D1]]], si64>) -> !mo.tensor<[[[D3]]], si64>
//
// CHECK-NEXT: mo.output<[[D3]]> %[[RES]] : !mo.tensor<[[[D3]]], si64>


mo.graph @broadcast_to_fallback_shape_func_unraned<D0 -> Sh1: !mosh.ape>(
  %input: !mo.tensor<[1, 3], f32>,
  %target_shape: !mo.tensor<[D0], si32>
) -> !mo.tensor<Sh1, f32> {

  %res = mo.broadcast_to<() -> Sh2: !mosh.ape>(%input, %target_shape) :
    (!mo.tensor<[1, 3], f32>, !mo.tensor<[D0], si32>) -> !mo.tensor<Sh2, f32>

    mo.output<:!mosh.ape Sh2> %res : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph @broadcast_to_fallback_shape_func
// CHECK-SAME: <[[D0:.*]] -> [[SH1:.*]]: !mosh.ape>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[1, 3], f32>,
// CHECK-SAME:   %[[TARGET_SHAPE:.*]]: !mo.tensor<[[[D0]]], si32>
// CHECK-SAME: ) -> !mo.tensor<[[SH1]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["broadcast_to_shape"]<() -> [[SH2:.*]]: !mosh.ape>(%[[INPUT]], %[[TARGET_SHAPE]])
// CHECK-SAME:   {dataDeptTensors = [1 : i32]} :
// CHECK-SAME:   (!mo.tensor<[1, 3], f32>, !mo.tensor<[[[D0]]], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.broadcast_to(%[[INPUT]], %[[TARGET_SHAPE]]) :
// CHECK-SAME:   (!mo.tensor<[1, 3], f32>, !mo.tensor<[[[D0]]], si32>)
// CHECK-SAME:   -> !mo.tensor<[[SH2]], f32>
//
// CHECK-NEXT: mo.output<{{.*}} %[[RES]] : {{.*}}


// Note how the second invocation does not redefine "I1" or "B"
// CHECK-LABEL: mo.graph @broadcast_partially_bound_params
// CHECK: mo.invoke_shape_func["broadcast_to_shape"]<() -> I1, I2, I3>({{.*}}) {dataDeptTensors = [1 : i32]} : (!mo.tensor<[A, B], f32>, !mo.tensor<[3], si32>)
// CHECK: mo.invoke_shape_func["broadcast_to_shape"]<() -> F1, F2, I4>({{.*}}) {dataDeptTensors = [1 : i32]} : (!mo.tensor<[A, B], f32>, !mo.tensor<[3], si32>)
mo.graph @broadcast_partially_bound_params<A, B, C -> O1, O2, O3, O4, O5, O6>(
  %input: !mo.tensor<[A, B], f32>,
  %target_shape: !mo.tensor<[3], si32>
) -> (!mo.tensor<[O1, O2, O3], f32>, !mo.tensor<[O4, O5, O6], f32>) {

  %res1 = mo.broadcast_to<() -> I1, I2, I3>(%input, %target_shape) :
    (!mo.tensor<[A, B], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[I1, I2, I3], f32>

  // NOTE: only declares some of the output parameters, reuses some from res1 and input
  %res2 = mo.broadcast_to<() -> I4>(%input, %target_shape) :
    (!mo.tensor<[A, B], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[I1, B, I4], f32>

  mo.output<I1, I2, I3, I1, B, I4> %res1, %res2 : !mo.tensor<[I1, I2, I3], f32>, !mo.tensor<[I1, B, I4], f32>
}
