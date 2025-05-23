// RUN: mo-opt --hoist-constant-subgraphs-into-parent --split-input-file %s | FileCheck %s


// Test case: Multiple constant arguments, multiple constant ops, one boundary value
// CHECK-LABEL: module
module {

// CHECK-LABEL: mo.graph @multi_const_args
// CHECK-SAME: (%[[ARG0:.+]]: !mo.tensor<[4], f32>, %[[HOISTED_ARG:.+]]: !mo.tensor<[4], f32>)
mo.graph @multi_const_args(%arg0: !mo.tensor<[4], f32>, %arg1: !mo.tensor<[4], f32>, %arg2: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) attributes {isSubgraph} {
  // These ops depend only on constant args %arg1, %arg2 and should be hoisted
  %0 = mo.mul(%arg1, %arg2) : !mo.tensor<[4], f32>
  %1 = mo.relu(%0) : !mo.tensor<[4], f32>

  // This op depends on non-constant arg0 and the hoisted boundary value %1
  // CHECK: %[[ADD_OUT:.+]] = mo.add(%[[ARG0]], %[[HOISTED_ARG]])
  %2 = mo.add(%arg0, %1) : !mo.tensor<[4], f32>

  // CHECK: mo.output %[[ADD_OUT]]
  mo.output %2 : !mo.tensor<[4], f32>
}

// CHECK-LABEL: mo.graph @caller_multi_const
// CHECK-SAME: (%[[CALLER_ARG0:.+]]: !mo.tensor<[4], f32>)
mo.graph @caller_multi_const(%arg0: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[CONST1:.+]] = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "cst1"}
  %cst1 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "cst1"} : !mo.tensor<[4], f32>
  // CHECK: %[[CONST2:.+]] = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "cst2"}
  %cst2 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "cst2"} : !mo.tensor<[4], f32>

  // Hoisted ops
  // CHECK: %[[HOISTED_MUL:.+]] = mo.mul(%[[CONST1]], %[[CONST2]])
  // CHECK: %[[HOISTED_RELU:.+]] = mo.relu(%[[HOISTED_MUL]])

  // CHECK: %[[CALL_RESULT:.+]] = mo.call @multi_const_args(%[[CALLER_ARG0]], %[[HOISTED_RELU]])
  %res = mo.call @multi_const_args(%arg0, %cst1, %cst2) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  // CHECK: mo.output %[[CALL_RESULT]]
  mo.output %res : !mo.tensor<[4], f32>
}

} // module

// -----

// Test case: Multiple boundary values
// CHECK-LABEL: module
module {

// CHECK-LABEL: mo.graph @multi_boundary
// CHECK-SAME: (%[[ARG0:.+]]: !mo.tensor<[4], f32>, %[[HOISTED_ARG1:.+]]: !mo.tensor<[4], f32>, %[[HOISTED_ARG2:.+]]: !mo.tensor<[4], f32>)
mo.graph @multi_boundary(%arg0: !mo.tensor<[4], f32>, %arg1: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) attributes {isSubgraph} {
  // Hoistable ops
  %0 = mo.relu(%arg1) : !mo.tensor<[4], f32>
  %1 = mo.relu(%arg1) : !mo.tensor<[4], f32>

  // Non-constant ops using boundary values
  // CHECK: %[[ADD1_OUT:.+]] = mo.add(%[[ARG0]], %[[HOISTED_ARG1]])
  %2 = mo.add(%arg0, %0) : !mo.tensor<[4], f32>
  // CHECK: %[[ADD2_OUT:.+]] = mo.add(%[[ADD1_OUT]], %[[HOISTED_ARG2]])
  %3 = mo.add(%2, %1) : !mo.tensor<[4], f32>

  // CHECK: mo.output %[[ADD2_OUT]]
  mo.output %3 : !mo.tensor<[4], f32>
}

// CHECK-LABEL: mo.graph @caller_multi_boundary
// CHECK-SAME: (%[[CALLER_ARG0:.+]]: !mo.tensor<[4], f32>)
mo.graph @caller_multi_boundary(%arg0: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[CONST1:.+]] = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "cst1"}
  %cst1 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "cst1"} : !mo.tensor<[4], f32>

  // Hoisted ops
  // CHECK: %[[HOISTED_RELU1:.+]] = mo.relu(%[[CONST1]])
  // CHECK: %[[HOISTED_RELU2:.+]] = mo.relu(%[[CONST1]])

  // CHECK: %[[CALL_RESULT:.+]] = mo.call @multi_boundary(%[[CALLER_ARG0]], %[[HOISTED_RELU1]], %[[HOISTED_RELU2]])
  %res = mo.call @multi_boundary(%arg0, %cst1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  // CHECK: mo.output %[[CALL_RESULT]]
  mo.output %res : !mo.tensor<[4], f32>
}

} // module

// -----

// Test case: Unused constant argument gets removed
// CHECK-LABEL: module
module {

// CHECK-LABEL: mo.graph @unused_const_arg
// CHECK-SAME: (%[[ARG0:.+]]: !mo.tensor<[4], f32>, %[[HOISTED_ARG:.+]]: !mo.tensor<[4], f32>)
mo.graph @unused_const_arg(%arg0: !mo.tensor<[4], f32>, %arg1_const_used: !mo.tensor<[4], f32>, %arg2_const_unused: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) attributes {isSubgraph} {
  // Hoistable op using arg1
  %0 = mo.relu(%arg1_const_used) : !mo.tensor<[4], f32>

  // Non-constant op using arg0 and boundary value
  // CHECK: %[[ADD_OUT:.+]] = mo.add(%[[ARG0]], %[[HOISTED_ARG]])
  %1 = mo.add(%arg0, %0) : !mo.tensor<[4], f32>

  // arg2 is never used

  // CHECK: mo.output %[[ADD_OUT]]
  mo.output %1 : !mo.tensor<[4], f32>
}

// CHECK-LABEL: mo.graph @caller_unused_const
// CHECK-SAME: (%[[CALLER_ARG0:.+]]: !mo.tensor<[4], f32>)
mo.graph @caller_unused_const(%arg0: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[CONST1:.+]] = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "cst1"}
  %cst1 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "cst1"} : !mo.tensor<[4], f32>
  // CHECK: %[[CONST2:.+]] = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "cst2"}
  %cst2 = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "cst2"} : !mo.tensor<[4], f32> // Unused const

  // Hoisted op
  // CHECK: %[[HOISTED_RELU:.+]] = mo.relu(%[[CONST1]])

  // Call should only have arg0 and the boundary value. const2 is gone.
  // CHECK: %[[CALL_RESULT:.+]] = mo.call @unused_const_arg(%[[CALLER_ARG0]], %[[HOISTED_RELU]])
  %res = mo.call @unused_const_arg(%arg0, %cst1, %cst2) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  // CHECK: mo.output %[[CALL_RESULT]]
  mo.output %res : !mo.tensor<[4], f32>
}

} // module

// -----

// Test case: Multiple callsites with identical constant arguments
// CHECK-LABEL: module
module {

// CHECK-LABEL: mo.graph @called_twice_const
// CHECK-SAME: (%[[ARG0:.+]]: !mo.tensor<[4], f32>, %[[HOISTED_ARG:.+]]: !mo.tensor<[4], f32>)
mo.graph @called_twice_const(%arg0: !mo.tensor<[4], f32>, %arg1_const: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) attributes {isSubgraph} {
  // Hoistable op
  %0 = mo.relu(%arg1_const) : !mo.tensor<[4], f32>

  // Non-constant op
  // CHECK: %[[ADD_OUT:.+]] = mo.add(%[[ARG0]], %[[HOISTED_ARG]])
  %1 = mo.add(%arg0, %0) : !mo.tensor<[4], f32>
  // CHECK: mo.output %[[ADD_OUT]]
  mo.output %1 : !mo.tensor<[4], f32>
}

// CHECK-LABEL: mo.graph @caller_twice_const
// CHECK-SAME: (%[[CALLER_ARG0:.+]]: !mo.tensor<[4], f32>, %[[CALLER_ARG1:.+]]: !mo.tensor<[4], f32>)
mo.graph @caller_twice_const(%arg0: !mo.tensor<[4], f32>, %arg1: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[CONST:.+]] = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "cst"}
  %cst = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "cst"} : !mo.tensor<[4], f32>

  // Hoisted op for call 1
  // CHECK: %[[HOISTED_RELU1:.+]] = mo.relu(%[[CONST]])
  // CHECK: %[[CALL1_RESULT:.+]] = mo.call @called_twice_const(%[[CALLER_ARG0]], %[[HOISTED_RELU1]])
  %res1 = mo.call @called_twice_const(%arg0, %cst) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>

  // Hoisted op for call 2 (same constant, so same hoist)
  // CHECK: %[[HOISTED_RELU2:.+]] = mo.relu(%[[CONST]])
  // CHECK: %[[CALL2_RESULT:.+]] = mo.call @called_twice_const(%[[CALLER_ARG1]], %[[HOISTED_RELU2]])
  %res2 = mo.call @called_twice_const(%arg1, %cst) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>

  // CHECK: %[[ADD_OUT:.+]] = mo.add(%[[CALL1_RESULT]], %[[CALL2_RESULT]])
  %res_add = mo.add(%res1, %res2) : !mo.tensor<[4], f32>
  // CHECK: mo.output %[[ADD_OUT]]
  mo.output %res_add : !mo.tensor<[4], f32>
}

} // module

// -----

// Test case: Multiple callsites, one with constant, one with variable -> NO hoisting
// CHECK-LABEL: module
module {

// Graph should remain unchanged because arg1 is not constant across all callsites
// CHECK-LABEL: mo.graph @mixed_callsites
// CHECK-SAME: (%[[ARG0:.+]]: !mo.tensor<[4], f32>, %[[ARG1:.+]]: !mo.tensor<[4], f32>)
mo.graph @mixed_callsites(%arg0: !mo.tensor<[4], f32>, %arg1: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) attributes {isSubgraph} {
  // CHECK: %[[RELU_OUT:.+]] = mo.relu(%[[ARG1]])
  %0 = mo.relu(%arg1) : !mo.tensor<[4], f32>
  // CHECK: %[[ADD_OUT:.+]] = mo.add(%[[ARG0]], %[[RELU_OUT]])
  %1 = mo.add(%arg0, %0) : !mo.tensor<[4], f32>
  // CHECK: mo.output %[[ADD_OUT]]
  mo.output %1 : !mo.tensor<[4], f32>
}

// Caller should remain unchanged, no hoisting happens
// CHECK-LABEL: mo.graph @caller_mixed_callsites
// CHECK-SAME: (%[[CALLER_ARG0:.+]]: !mo.tensor<[4], f32>, %[[CALLER_ARG1:.+]]: !mo.tensor<[4], f32>)
mo.graph @caller_mixed_callsites(%arg0: !mo.tensor<[4], f32>, %arg1_variable: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) {
  // CHECK: %[[CONST:.+]] = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "cst"}
  %cst = mo.constant.external {align = 1 : ui64, device = #M.device_ref<"cpu", 0>, name = "cst"} : !mo.tensor<[4], f32>

  // Call 1: Uses constant
  // CHECK: %[[CALL1_RES:.+]] = mo.call @mixed_callsites(%[[CALLER_ARG0]], %[[CONST]])
  %res1 = mo.call @mixed_callsites(%arg0, %cst) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>

  // Call 2: Uses variable
  // CHECK: %[[CALL2_RES:.+]] = mo.call @mixed_callsites(%[[CALLER_ARG0]], %[[CALLER_ARG1]])
  %res2 = mo.call @mixed_callsites(%arg0, %arg1_variable) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>

  // CHECK: %[[ADD_RES:.+]] = mo.add(%[[CALL1_RES]], %[[CALL2_RES]])
  %res_add = mo.add(%res1, %res2) : !mo.tensor<[4], f32>
  // CHECK: mo.output %[[ADD_RES]]
  mo.output %res_add : !mo.tensor<[4], f32>
}

} // module
