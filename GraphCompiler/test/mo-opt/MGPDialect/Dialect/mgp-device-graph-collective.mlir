// RUN: mo-opt --allow-unregistered-dialect %s --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @device_graph_collective_with_stable_args_and_results
func.func @device_graph_collective_with_stable_args_and_results() {
  // CHECK:       %[[VAL:[a-zA-Z0-9_]+]] = "test.op"() : () -> !mgp.tensor<4xf32>
  %val = "test.op"() : () -> !mgp.tensor<4xf32>

  // CHECK:       %{{[a-zA-Z0-9_]+}}:2 = mgp.device_graph.collective builder(
  // CHECK-SAME:    %[[B:[a-zA-Z0-9_]+]]
  // CHECK-SAME:  ) stable(
  // CHECK-SAME:    %[[S0:[a-zA-Z0-9_]+]]: !mgp.stable_addr<<4xf32, <"gpu", 0>>>,
  // CHECK-SAME:    %[[S1:[a-zA-Z0-9_]+]]: !mgp.stable_addr<<4xf32, <"gpu", 1>>>
  // CHECK-SAME:  ) <"gpu", 0> {
  // CHECK-NEXT:    mgp.device_graph.collective.yield %[[VAL]] : !mgp.tensor<4xf32>
  // CHECK-NEXT:  }, <"gpu", 1> {
  // CHECK-NEXT:    mgp.device_graph.collective.yield %[[VAL]] : !mgp.tensor<4xf32>
  // CHECK-NEXT:  } : !mgp.tensor<4xf32>, !mgp.tensor<4xf32>
  %0:2 = mgp.device_graph.collective builder(%b) stable(
    %s0: !mgp.stable_addr<<4xf32, <"gpu", 0>>>,
    %s1: !mgp.stable_addr<<4xf32, <"gpu", 1>>>
  ) <"gpu", 0> {
    mgp.device_graph.collective.yield %val : !mgp.tensor<4xf32>
  }, <"gpu", 1> {
    mgp.device_graph.collective.yield %val : !mgp.tensor<4xf32>
  } : !mgp.tensor<4xf32>, !mgp.tensor<4xf32>

  return
}

// -----

// CHECK-LABEL: func.func @device_graph_collective_uses_stable_addresses
func.func @device_graph_collective_uses_stable_addresses() {
  // CHECK:       %{{[a-zA-Z0-9_]+}}:2 = mgp.device_graph.collective builder(
  // CHECK-SAME:    %[[B:[a-zA-Z0-9_]+]]
  // CHECK-SAME:  ) stable(
  // CHECK-SAME:    %[[S0:[a-zA-Z0-9_]+]]: !mgp.stable_addr<<4xf32, <"gpu", 0>>>,
  // CHECK-SAME:    %[[S1:[a-zA-Z0-9_]+]]: !mgp.stable_addr<<4xf32, <"gpu", 1>>>
  // CHECK-SAME:  ) <"gpu", 0> {
  // CHECK-NEXT:    %[[T0:[a-zA-Z0-9_]+]] = mgp.device_graph.stable.get %[[S0]] : !mgp.stable_addr<<4xf32, <"gpu", 0>>>
  // CHECK-NEXT:    mgp.device_graph.collective.yield %[[T0]] : !mgp.tensor<4xf32, <"gpu", 0>, stable>
  // CHECK-NEXT:  }, <"gpu", 1> {
  // CHECK-NEXT:    %[[T1:[a-zA-Z0-9_]+]] = mgp.device_graph.stable.get %[[S1]] : !mgp.stable_addr<<4xf32, <"gpu", 1>>>
  // CHECK-NEXT:    mgp.device_graph.collective.yield %[[T1]] : !mgp.tensor<4xf32, <"gpu", 1>, stable>
  // CHECK-NEXT:  } : !mgp.tensor<4xf32, <"gpu", 0>, stable>, !mgp.tensor<4xf32, <"gpu", 1>, stable>
  %0:2 = mgp.device_graph.collective builder(%b) stable(
    %s0: !mgp.stable_addr<<4xf32, <"gpu", 0>>>,
    %s1: !mgp.stable_addr<<4xf32, <"gpu", 1>>>
  ) <"gpu", 0> {
    %t0 = mgp.device_graph.stable.get %s0 : !mgp.stable_addr<<4xf32, <"gpu", 0>>>
    mgp.device_graph.collective.yield %t0 : !mgp.tensor<4xf32, <"gpu", 0>, stable>
  }, <"gpu", 1> {
    %t1 = mgp.device_graph.stable.get %s1 : !mgp.stable_addr<<4xf32, <"gpu", 1>>>
    mgp.device_graph.collective.yield %t1 : !mgp.tensor<4xf32, <"gpu", 1>, stable>
  } : !mgp.tensor<4xf32, <"gpu", 0>, stable>, !mgp.tensor<4xf32, <"gpu", 1>, stable>

  return
}

// -----

// CHECK-LABEL: func.func @device_graph_collective_stable_place
func.func @device_graph_collective_stable_place() {
  // CHECK:       %[[VAL:[a-zA-Z0-9_]+]] = "test.op"() : () -> !mgp.tensor<4xf32, <"gpu", 0>>
  %val = "test.op"() : () -> !mgp.tensor<4xf32, <"gpu", 0>>

  // CHECK:       mgp.device_graph.collective builder(
  // CHECK-SAME:    %[[B:[a-zA-Z0-9_]+]]
  // CHECK-SAME:  ) stable(
  // CHECK-SAME:    %[[S0:[a-zA-Z0-9_]+]]: !mgp.stable_addr<<4xf32, <"gpu", 0>>>
  // CHECK-SAME:  ) <"gpu", 0> {
  // CHECK-NEXT:    mgp.device_graph.stable.place %[[S0]], %[[VAL]] : !mgp.tensor<4xf32, <"gpu", 0>>
  // CHECK-NEXT:    mgp.device_graph.collective.yield
  // CHECK-NEXT:  }
  mgp.device_graph.collective builder(%b) stable(
      %s0: !mgp.stable_addr<<4xf32, <"gpu", 0>>>
  ) <"gpu", 0> {
    mgp.device_graph.stable.place %s0, %val : !mgp.tensor<4xf32, <"gpu", 0>>
    mgp.device_graph.collective.yield
  }

  return
}

// -----

// CHECK-LABEL: func.func @device_graph_stable_create
func.func @device_graph_stable_create(%val: !mgp.tensor<4xf32, <"gpu", 0>, mut>) -> !mgp.tensor<4xf32, <"gpu", 0>, mut, stable> {
  // CHECK: %{{[a-zA-Z0-9_]+}} = mgp.device_graph.stable.create %{{[a-zA-Z0-9_]+}} : !mgp.tensor<4xf32, <"gpu", 0>, mut>
  %shared = mgp.device_graph.stable.create %val : !mgp.tensor<4xf32, <"gpu", 0>, mut>
  // CHECK: return %{{[a-zA-Z0-9_]+}} : !mgp.tensor<4xf32, <"gpu", 0>, mut, stable>
  return %shared : !mgp.tensor<4xf32, <"gpu", 0>, mut, stable>
}

// -----

// CHECK-LABEL: func.func @device_graph_stable_create_default_device
func.func @device_graph_stable_create_default_device(%val: !mgp.tensor<4xf32>) -> !mgp.tensor<4xf32, stable> {
  // CHECK: %{{[a-zA-Z0-9_]+}} = mgp.device_graph.stable.create %{{[a-zA-Z0-9_]+}} : !mgp.tensor<4xf32>
  %shared = mgp.device_graph.stable.create %val : !mgp.tensor<4xf32>
  // CHECK: return %{{[a-zA-Z0-9_]+}} : !mgp.tensor<4xf32, stable>
  return %shared : !mgp.tensor<4xf32, stable>
}
