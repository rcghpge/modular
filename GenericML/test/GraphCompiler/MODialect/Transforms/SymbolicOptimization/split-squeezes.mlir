// RUN: mo-opt %s --symbolic-optimize | FileCheck %s

// CHECK: mo.graph @split_squeezes<[[D0:.*]] -> [[D1:.*]], [[D2:.*]]>(
mo.graph @split_squeezes<D0 -> D1, D2>(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 2, 10], f32>
  %input: !mo.tensor<[D0, 2, 10], f32>
// CHECK-SAME: ) -> (!mo.tensor<[[[D1]], 10], f32>, !mo.tensor<[[[D2]], 10], f32>) no_inline {
) -> (!mo.tensor<[D1, 10], f32>, !mo.tensor<[D2, 10], f32>)  {

  // CHECK-DAG: %[[SPLIT_SIZES:.*]] = mo.constant {value = #M.dense_array<1, 1> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
  %split_sizes = mo.constant {value = #M.dense_array<1, 1> : !M.array<2xsi64>} : !mo.tensor<[2], si64>

  // Notice the change in axis (from -2 to 0)
  // CHECK-DAG: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<0> : !M.array<1xsi32>} : !mo.tensor<[], si32>
  %axis = mo.constant {value = #M.dense_array<-2> : !M.array<1xsi32>} : !mo.tensor<[], si32>

  // Notice the extra transpose on input
  // CHECK-DAG: %[[PERM:.*]] = mo.constant {value = #M.dense_array<1, 0, 2> : !M.array<3xsi64>} : !mo.tensor<[3], si64>
  // CHECK-NEXT: %[[INPUT_T:.*]] = mo.transpose(%[[INPUT]], %[[PERM]])
  // CHECK-SAME: : (!mo.tensor<[[[D0]], 2, 10], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, [[D0]], 10], f32>

  // Notice the change in input and axis
  // CHECK-NEXT: %[[SPLIT_RES:.*]]:2 = mo.split[%[[AXIS]] : !mo.tensor<[], si32>] (%[[INPUT_T]], %[[SPLIT_SIZES]])
  // CHECK-SAME: : (!mo.tensor<[2, [[D0]], 10], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>)
  %split_res:2 = mo.split[%axis : !mo.tensor<[], si32>] (%input, %split_sizes)
  : (!mo.tensor<[D0, 2, 10], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>)

  // Notice the permutation in rebind output shape
  // CHECK-NEXT: %[[RES_REB1:.*]] = mo.rebind<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(%[[SPLIT_RES]]#0)
  // CHECK-SAME: : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[[[D4]], [[D3]], [[D5]]], f32>
  %res_reb1 = mo.rebind<() -> D3, D4, D5>(%split_res#0) : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[D3, D4, D5], f32>
  // CHECK-NEXT: %[[RES_RESHAPE1:.*]] = mo.static.reshape(%[[RES_REB1]])
  // CHECK-SAME: : !mo.tensor<[[[D4]], [[D3]], [[D5]]], f32> -> !mo.tensor<[[[D3]], 10], f32>
  %res_reshape1 = mo.static.reshape(%res_reb1) : !mo.tensor<[D3, D4, D5], f32> -> !mo.tensor<[D3, 10], f32>

  // CHECK-NEXT: %[[RES_REB2:.*]] = mo.rebind<() -> [[D6:.*]], [[D7:.*]], [[D8:.*]]>(%[[SPLIT_RES]]#1)
  // CHECK-SAME: : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[[[D7]], [[D6]], [[D8]]], f32>
  %res_reb2 = mo.rebind<() -> D6, D7, D8>(%split_res#1) : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[D6, D7, D8], f32>
  // CHECK-NEXT: %[[RES_RESHAPE2:.*]] = mo.static.reshape(%[[RES_REB2]])
  // CHECK-SAME: : !mo.tensor<[[[D7]], [[D6]], [[D8]]], f32> -> !mo.tensor<[[[D6]], 10], f32>
  %res_reshape2 = mo.static.reshape(%res_reb2) : !mo.tensor<[D6, D7, D8], f32> -> !mo.tensor<[D6, 10], f32>

  // CHECK-NEXT: mo.output<[[D3]], [[D6]]> %[[RES_RESHAPE1]], %[[RES_RESHAPE2]]
  // CHECK-SAME: !mo.tensor<[[[D3]], 10], f32>, !mo.tensor<[[[D6]], 10], f32>
  mo.output<D3, D6> %res_reshape1, %res_reshape2 : !mo.tensor<[D3, 10], f32>, !mo.tensor<[D6, 10], f32>
}

// No effect
// CHECK: mo.graph @split_squeezes_dyn_axis<[[D0:.*]] -> [[D1:.*]], [[D2:.*]]>(
mo.graph @split_squeezes_dyn_axis<D0 -> D1, D2>(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 2, 10], f32>,
  %input: !mo.tensor<[D0, 2, 10], f32>,
  // CHECK-SAME: %[[AXIS:.*]]: !mo.tensor<[], si32>
  %axis: !mo.tensor<[], si32>
// CHECK-SAME: ) -> (!mo.tensor<[[[D1]], 10], f32>, !mo.tensor<[[[D2]], 10], f32>) no_inline {
) -> (!mo.tensor<[D1, 10], f32>, !mo.tensor<[D2, 10], f32>)  {

  // CHECK-DAG: %[[SPLIT_SIZES:.*]] = mo.constant {value = #M.dense_array<1, 1> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
  %split_sizes = mo.constant {value = #M.dense_array<1, 1> : !M.array<2xsi64>} : !mo.tensor<[2], si64>

  // CHECK-NEXT: %[[SPLIT_RES:.*]]:2 = mo.split[%[[AXIS]] : !mo.tensor<[], si32>] (%[[INPUT]], %[[SPLIT_SIZES]])
  // CHECK-SAME: : (!mo.tensor<[[[D0]], 2, 10], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>)
  %split_res:2 = mo.split[%axis : !mo.tensor<[], si32>] (%input, %split_sizes)
  : (!mo.tensor<[D0, 2, 10], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>)

  // CHECK-NEXT: %[[RES_REB1:.*]] = mo.rebind<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(%[[SPLIT_RES]]#0)
  // CHECK-SAME: : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
  %res_reb1 = mo.rebind<() -> D3, D4, D5>(%split_res#0) : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[D3, D4, D5], f32>
  // CHECK-NEXT: %[[RES_RESHAPE1:.*]] = mo.static.reshape(%[[RES_REB1]])
  // CHECK-SAME: : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32> -> !mo.tensor<[[[D3]], 10], f32>
  %res_reshape1 = mo.static.reshape(%res_reb1) : !mo.tensor<[D3, D4, D5], f32> -> !mo.tensor<[D3, 10], f32>

  // CHECK-NEXT: %[[RES_REB2:.*]] = mo.rebind<() -> [[D6:.*]], [[D7:.*]], [[D8:.*]]>(%[[SPLIT_RES]]#1)
  // CHECK-SAME: : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[[[D6]], [[D7]], [[D8]]], f32>
  %res_reb2 = mo.rebind<() -> D6, D7, D8>(%split_res#1) : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[D6, D7, D8], f32>
  // CHECK-NEXT: %[[RES_RESHAPE2:.*]] = mo.static.reshape(%[[RES_REB2]])
  // CHECK-SAME: : !mo.tensor<[[[D6]], [[D7]], [[D8]]], f32> -> !mo.tensor<[[[D6]], 10], f32>
  %res_reshape2 = mo.static.reshape(%res_reb2) : !mo.tensor<[D6, D7, D8], f32> -> !mo.tensor<[D6, 10], f32>

  // CHECK-NEXT: mo.output<[[D3]], [[D6]]> %[[RES_RESHAPE1]], %[[RES_RESHAPE2]]
  // CHECK-SAME: !mo.tensor<[[[D3]], 10], f32>, !mo.tensor<[[[D6]], 10], f32>
  mo.output<D3, D6> %res_reshape1, %res_reshape2 : !mo.tensor<[D3, 10], f32>, !mo.tensor<[D6, 10], f32>
}

// No effect
// CHECK: mo.graph @split_squeezes_split_sizes_not_all_ones<[[D0:.*]] -> [[D1:.*]], [[D2:.*]]>(
mo.graph @split_squeezes_split_sizes_not_all_ones<D0 -> D1, D2>(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 2, 10], f32>
  %input: !mo.tensor<[D0, 2, 10], f32>
// CHECK-SAME: ) -> (!mo.tensor<[[[D1]], 10], f32>, !mo.tensor<[[[D2]], 10], f32>) no_inline {
) -> (!mo.tensor<[D1, 10], f32>, !mo.tensor<[D2, 10], f32>)  {

  // CHECK-DAG: %[[SPLIT_SIZES:.*]] = mo.constant {value = #M.dense_array<2, 0> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
  %split_sizes = mo.constant {value = #M.dense_array<2, 0> : !M.array<2xsi64>} : !mo.tensor<[2], si64>

  // CHECK-DAG: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<1> : !M.array<1xsi32>} : !mo.tensor<[], si32>
  %axis = mo.constant {value = #M.dense_array<1> : !M.array<1xsi32>} : !mo.tensor<[], si32>

  // CHECK-NEXT: %[[SPLIT_RES:.*]]:2 = mo.split[%[[AXIS]] : !mo.tensor<[], si32>] (%[[INPUT]], %[[SPLIT_SIZES]])
  // CHECK-SAME: : (!mo.tensor<[[[D0]], 2, 10], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>)
  %split_res:2 = mo.split[%axis : !mo.tensor<[], si32>] (%input, %split_sizes)
  : (!mo.tensor<[D0, 2, 10], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>)

  // CHECK-NEXT: %[[RES_REB1:.*]] = mo.rebind<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(%[[SPLIT_RES]]#0)
  // CHECK-SAME: : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
  %res_reb1 = mo.rebind<() -> D3, D4, D5>(%split_res#0) : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[D3, D4, D5], f32>
  // CHECK-NEXT: %[[RES_RESHAPE1:.*]] = mo.static.reshape(%[[RES_REB1]])
  // CHECK-SAME: : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32> -> !mo.tensor<[[[D3]], 10], f32>
  %res_reshape1 = mo.static.reshape(%res_reb1) : !mo.tensor<[D3, D4, D5], f32> -> !mo.tensor<[D3, 10], f32>

  // CHECK-NEXT: %[[RES_REB2:.*]] = mo.rebind<() -> [[D6:.*]], [[D7:.*]], [[D8:.*]]>(%[[SPLIT_RES]]#1)
  // CHECK-SAME: : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[[[D6]], [[D7]], [[D8]]], f32>
  %res_reb2 = mo.rebind<() -> D6, D7, D8>(%split_res#1) : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[D6, D7, D8], f32>
  // CHECK-NEXT: %[[RES_RESHAPE2:.*]] = mo.static.reshape(%[[RES_REB2]])
  // CHECK-SAME: : !mo.tensor<[[[D6]], [[D7]], [[D8]]], f32> -> !mo.tensor<[[[D6]], 10], f32>
  %res_reshape2 = mo.static.reshape(%res_reb2) : !mo.tensor<[D6, D7, D8], f32> -> !mo.tensor<[D6, 10], f32>

  // CHECK-NEXT: mo.output<[[D3]], [[D6]]> %[[RES_RESHAPE1]], %[[RES_RESHAPE2]]
  // CHECK-SAME: !mo.tensor<[[[D3]], 10], f32>, !mo.tensor<[[[D6]], 10], f32>
  mo.output<D3, D6> %res_reshape1, %res_reshape2 : !mo.tensor<[D3, 10], f32>, !mo.tensor<[D6, 10], f32>
}

// No effect; TODO(#13349) consider relaxing this test
// CHECK: mo.graph @split_squeezes_non_squeezing_reshape<[[D0:.*]] -> [[D1:.*]], [[D2:.*]]>(
mo.graph @split_squeezes_non_squeezing_reshape<D0 -> D1, D2>(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 2, 10], f32>
  %input: !mo.tensor<[D0, 2, 10], f32>
// CHECK-SAME: ) -> (!mo.tensor<[[[D1]]], f32>, !mo.tensor<[[[D2]], 10], f32>) no_inline {
) -> (!mo.tensor<[D1], f32>, !mo.tensor<[D2, 10], f32>)  {

  // CHECK-DAG: %[[SPLIT_SIZES:.*]] = mo.constant {value = #M.dense_array<1, 1> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
  %split_sizes = mo.constant {value = #M.dense_array<1, 1> : !M.array<2xsi64>} : !mo.tensor<[2], si64>

  // CHECK-DAG: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<1> : !M.array<1xsi32>} : !mo.tensor<[], si32>
  %axis = mo.constant {value = #M.dense_array<1> : !M.array<1xsi32>} : !mo.tensor<[], si32>

  // CHECK-NEXT: %[[SPLIT_RES:.*]]:2 = mo.split[%[[AXIS]] : !mo.tensor<[], si32>] (%[[INPUT]], %[[SPLIT_SIZES]])
  // CHECK-SAME: : (!mo.tensor<[[[D0]], 2, 10], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>)
  %split_res:2 = mo.split[%axis : !mo.tensor<[], si32>] (%input, %split_sizes)
  : (!mo.tensor<[D0, 2, 10], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>)

  // CHECK-NEXT: %[[RES_REB1:.*]] = mo.rebind<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(%[[SPLIT_RES]]#0)
  // CHECK-SAME: : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
  %res_reb1 = mo.rebind<() -> D3, D4, D5>(%split_res#0) : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[D3, D4, D5], f32>
  // CHECK-NEXT: %[[RES_RESHAPE1:.*]] = mo.static.reshape(%[[RES_REB1]])
  // CHECK-SAME: : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32> -> !mo.tensor<[100], f32>
  %res_reshape1 = mo.static.reshape(%res_reb1) : !mo.tensor<[D3, D4, D5], f32> -> !mo.tensor<[100], f32>

  // CHECK-NEXT: %[[RES_REB2:.*]] = mo.rebind<() -> [[D6:.*]], [[D7:.*]], [[D8:.*]]>(%[[SPLIT_RES]]#1)
  // CHECK-SAME: : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[[[D6]], [[D7]], [[D8]]], f32>
  %res_reb2 = mo.rebind<() -> D6, D7, D8>(%split_res#1) : !mo.tensor<[?, ?, ?], f32> -> !mo.tensor<[D6, D7, D8], f32>
  // CHECK-NEXT: %[[RES_RESHAPE2:.*]] = mo.static.reshape(%[[RES_REB2]])
  // CHECK-SAME: : !mo.tensor<[[[D6]], [[D7]], [[D8]]], f32> -> !mo.tensor<[[[D6]], 10], f32>
  %res_reshape2 = mo.static.reshape(%res_reb2) : !mo.tensor<[D6, D7, D8], f32> -> !mo.tensor<[D6, 10], f32>

  // CHECK-NEXT: mo.output<100, [[D6]]> %[[RES_RESHAPE1]], %[[RES_RESHAPE2]]
  // CHECK-SAME: !mo.tensor<[100], f32>, !mo.tensor<[[[D6]], 10], f32>
  mo.output<100, D6> %res_reshape1, %res_reshape2 : !mo.tensor<[100], f32>, !mo.tensor<[D6, 10], f32>
}
