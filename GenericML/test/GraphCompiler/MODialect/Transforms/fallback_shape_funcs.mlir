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
// CHECK-SAME:     {dataDeptTensors = [0 : i32], kgenParams = {input_type = f32}} :
// CHECK-SAME:     (!mo.tensor<[], si32>, !mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>)
//
// CHECK-NEXT:   %[[RES:.*]] = mo.concat[%[[AXIS]] : !mo.tensor<[], si32>] (%[[INPUT1]], %[[INPUT2]]) :
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D6]], [[D7]]], f32>
//
// CHECK-NEXT:   mo.output<[[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D6]], [[D7]]], f32>


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


mo.graph @where<D0 -> D1>(%arg0: !mo.tensor<[D0, 10], f32>) -> !mo.tensor<[D1, 2], si64> {
  %0 = mo.where<() -> D2>(%arg0) : (!mo.tensor<[D0, 10], f32>) -> !mo.tensor<[D2, 2], si64>
  mo.output<D2> %0 : !mo.tensor<[D2, 2], si64>
}
// CHECK-LABEL: mo.graph @where
// CHECK:(%[[IN:.*]]: !mo.tensor<[[[IN_PARAM:.*]], 10], f32>) -> !mo.tensor<[[[OUT_PARAM:.*]], 2], si64>
// CHECK-NEXT: mo.invoke_shape_func["where_shape"]<() -> [[WHERE_PARAM:.*]], [[UNUSED_PARAM:.*]]>(%[[IN]]) {dataDeptTensors = [0 : i32]}
// CHECK-NEXT: %[[WHERE:.*]] = mo.where(%[[IN]]) : (!mo.tensor<[[[IN_PARAM]], 10], f32>) -> !mo.tensor<[[[WHERE_PARAM]], 2], si64>
// CHECK-NEXT: mo.output<[[WHERE_PARAM]]> %[[WHERE]] : !mo.tensor<[[[WHERE_PARAM]], 2], si64>
