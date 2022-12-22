// RUN: mo-opt %s | FileCheck %s

// CHECK: mo.graph @matmul(%[[ARG0:.*]]: !mo.tensor<10x20xf32>, %[[ARG1:.*]]: !mo.tensor<20x5xf32>) -> !mo.tensor<10x5xf32> {
mo.graph @matmul(%arg0: !mo.tensor<10x20xf32>, %arg1: !mo.tensor<20x5xf32>) -> (!mo.tensor<10x5xf32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<10x20xf32>, !mo.tensor<20x5xf32>)   -> !mo.tensor<10x5xf32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<10x20xf32>, !mo.tensor<20x5xf32>)   -> !mo.tensor<10x5xf32>
  mo.output %0 : !mo.tensor<10x5xf32>
}

// CHECK: mo.graph @matmul_dynamic(%[[ARG0:.*]]: !mo.tensor<?x?xf32>, %[[ARG1:.*]]: !mo.tensor<?x?xf32>) -> !mo.tensor<?x?xf32> {
mo.graph @matmul_dynamic(%arg0: !mo.tensor<?x?xf32>, %arg1: !mo.tensor<?x?xf32>) -> (!mo.tensor<?x?xf32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<?x?xf32>, !mo.tensor<?x?xf32>)   -> !mo.tensor<?x?xf32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<?x?xf32>, !mo.tensor<?x?xf32>)   -> !mo.tensor<?x?xf32>
  mo.output %0 : !mo.tensor<?x?xf32>
}

// CHECK: mo.graph @matmul_dyndim_1
mo.graph @matmul_dyndim_1(%arg0: !mo.tensor<?x10xf32>, %arg1: !mo.tensor<10x5xf32>) -> (!mo.tensor<?x5xf32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<?x10xf32>, !mo.tensor<10x5xf32>) -> !mo.tensor<?x5xf32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<?x10xf32>, !mo.tensor<10x5xf32>) -> !mo.tensor<?x5xf32>
  mo.output %0 : !mo.tensor<?x5xf32>
}

// CHECK: mo.graph @matmul_dyndim_2
mo.graph @matmul_dyndim_2(%arg0: !mo.tensor<10x10xf32>, %arg1: !mo.tensor<10x?xf32>) -> (!mo.tensor<10x?xf32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<10x10xf32>, !mo.tensor<10x?xf32>) -> !mo.tensor<10x?xf32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<10x10xf32>, !mo.tensor<10x?xf32>) -> !mo.tensor<10x?xf32>
  mo.output %0 : !mo.tensor<10x?xf32>
}

// CHECK: mo.graph @matmul_dyndim_K
mo.graph @matmul_dyndim_K(%arg0: !mo.tensor<10x?xf32>, %arg1: !mo.tensor<?x5xf32>) -> (!mo.tensor<10x5xf32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<10x?xf32>, !mo.tensor<?x5xf32>) -> !mo.tensor<10x5xf32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<10x?xf32>, !mo.tensor<?x5xf32>) -> !mo.tensor<10x5xf32>
  mo.output %0 : !mo.tensor<10x5xf32>
}

// CHECK: mo.graph @batch_matmul(%[[ARG0:.*]]: !mo.tensor<3x10x20xf32>, %[[ARG1:.*]]: !mo.tensor<3x20x5xf32>) -> !mo.tensor<3x10x5xf32> {
mo.graph @batch_matmul(%arg0: !mo.tensor<3x10x20xf32>, %arg1: !mo.tensor<3x20x5xf32>) -> !mo.tensor<3x10x5xf32> {
  // CHECK-NEXT: mo.batch_matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<3x10x20xf32>, !mo.tensor<3x20x5xf32>) -> !mo.tensor<3x10x5xf32>
  %0 = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<3x10x20xf32>, !mo.tensor<3x20x5xf32>) -> !mo.tensor<3x10x5xf32>
  mo.output %0 : !mo.tensor<3x10x5xf32>
}

// CHECK: mo.graph @batch_matmul_4D(%[[ARG0:.*]]: !mo.tensor<7x3x10x20xf32>, %[[ARG1:.*]]: !mo.tensor<7x3x20x5xf32>) -> !mo.tensor<?x?x?x?xf32> {
mo.graph @batch_matmul_4D(%arg0: !mo.tensor<7x3x10x20xf32>, %arg1: !mo.tensor<7x3x20x5xf32>) -> !mo.tensor<?x?x?x?xf32> {
  // CHECK-NEXT: mo.batch_matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<7x3x10x20xf32>, !mo.tensor<7x3x20x5xf32>) -> !mo.tensor<?x?x?x?xf32>
  %0 = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<7x3x10x20xf32>, !mo.tensor<7x3x20x5xf32>) -> !mo.tensor<?x?x?x?xf32>
  mo.output %0 : !mo.tensor<?x?x?x?xf32>
}
