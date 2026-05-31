// RUN: mo-opt -split-input-file --mo-symbolic-optimize %s | FileCheck %s

// CHECK-LABEL:  mo.graph @usual_case
// CHECK-NEXT:    %[[SIZES:.*]] = mo.constant {value = #M.dense_array<2, 2> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:    %[[AXIS:.*]] = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT:    %[[SPLIT:.*]]:2 = mo.split[%[[AXIS]] : !mo.tensor<[], si64>] (%{{.*}}, %[[SIZES]]) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK-NEXT:    mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
// CHECK-NEXT:  }

mo.graph @usual_case(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %start1 = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %start2 = mo.constant {value = #M.dense_array<0, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop1 = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop2 = mo.constant {value = #M.dense_array<2, 4> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %2 = mo.slice(%in, %start1, %stop1, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %3 = mo.slice(%in, %start2, %stop2, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %2, %3 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL:  mo.graph @no_conversion_dynamic_start
// CHECK:          mo.slice
// CHECK:          mo.slice

mo.graph @no_conversion_dynamic_start(%in: !mo.tensor<[2, 4], f32>, %start1: !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %start2 = mo.constant {value = #M.dense_array<0, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop1 = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop2 = mo.constant {value = #M.dense_array<2, 4> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %2 = mo.slice(%in, %start1, %stop1, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %3 = mo.slice(%in, %start2, %stop2, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %2, %3 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL:  mo.graph @no_conversion_multidimensional_slice
// CHECK:          mo.slice
// CHECK:          mo.slice
// CHECK:          mo.slice
// CHECK:          mo.slice

mo.graph @no_conversion_multidimensional_slice(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[1, 2], f32>, !mo.tensor<[1, 2], f32>, !mo.tensor<[1, 2], f32>, !mo.tensor<[1, 2], f32>) {
  %start1 = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %start2 = mo.constant {value = #M.dense_array<0, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %start3 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %start4 = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop1 = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop2 = mo.constant {value = #M.dense_array<1, 4> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop3 = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop4 = mo.constant {value = #M.dense_array<2, 4> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %2 = mo.slice(%in, %start1, %stop1, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[1, 2], f32>
  %3 = mo.slice(%in, %start2, %stop2, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[1, 2], f32>
  %4 = mo.slice(%in, %start3, %stop3, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[1, 2], f32>
  %5 = mo.slice(%in, %start4, %stop4, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[1, 2], f32>
  mo.output %2, %3, %4, %5 : !mo.tensor<[1, 2], f32>, !mo.tensor<[1, 2], f32>, !mo.tensor<[1, 2], f32>, !mo.tensor<[1, 2], f32>
}

// CHECK-LABEL:  mo.graph @no_conversion_overlap
// CHECK:          mo.slice
// CHECK:          mo.slice

mo.graph @no_conversion_overlap(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %start1 = mo.constant {value = #M.dense_array<0, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %start2 = mo.constant {value = #M.dense_array<0, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop1 = mo.constant {value = #M.dense_array<2, 4> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop2 = mo.constant {value = #M.dense_array<2, 4> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %2 = mo.slice(%in, %start1, %stop1, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %3 = mo.slice(%in, %start2, %stop2, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %2, %3 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL:  mo.graph @no_conversion_partial_coverage
// CHECK:          mo.slice
// CHECK:          mo.slice

mo.graph @no_conversion_partial_coverage(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 1], f32>, !mo.tensor<[2, 2], f32>) {
  %start1 = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %start2 = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop1 = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop2 = mo.constant {value = #M.dense_array<2, 4> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %2 = mo.slice(%in, %start1, %stop1, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 1], f32>
  %3 = mo.slice(%in, %start2, %stop2, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %2, %3 : !mo.tensor<[2, 1], f32>, !mo.tensor<[2, 2], f32>
}

// Test that split sizes constant is created on host even with device tensor input
mo.graph @slice_to_split_host_constant<N>(%input: !mo.tensor<[N], si32, dev:0>) -> (!mo.tensor<[1], si32, dev:0>, !mo.tensor<[2], si32, dev:0>, !mo.tensor<[3], si32, dev:0>) {
  %start0 = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[1], si64>
  %stop0 = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[1], si64>
  %start1 = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[1], si64>
  %stop1 = mo.constant {value = #M.dense_array<3> : tensor<si64>} : !mo.tensor<[1], si64>
  %start2 = mo.constant {value = #M.dense_array<3> : tensor<si64>} : !mo.tensor<[1], si64>
  %stop2 = mo.constant {value = #M.dense_array<6> : tensor<si64>} : !mo.tensor<[1], si64>
  %step = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[1], si64>

  %out1 = mo.slice(%input, %start0, %stop0, %step) : (!mo.tensor<[N], si32, dev:0>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si32, dev:0>
  %out2 = mo.slice(%input, %start1, %stop1, %step) : (!mo.tensor<[N], si32, dev:0>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si32, dev:0>
  %out3 = mo.slice(%input, %start2, %stop2, %step) : (!mo.tensor<[N], si32, dev:0>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[3], si32, dev:0>

  %o1 = mo.activation(%out1) {activation = "relu"} : !mo.tensor<[1], si32, dev:0>
  %o2 = mo.activation(%out2) {activation = "relu"} : !mo.tensor<[2], si32, dev:0>
  %o3 = mo.activation(%out3) {activation = "relu"} : !mo.tensor<[3], si32, dev:0>
  mo.output %o1, %o2, %o3 : !mo.tensor<[1], si32, dev:0>, !mo.tensor<[2], si32, dev:0>, !mo.tensor<[3], si32, dev:0>
}

// CHECK-LABEL: mo.graph @slice_to_split_host_constant<N>
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[N], si32, dev:0>
// CHECK-DAG: %[[SPLIT_SIZES:.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[1], si64>
