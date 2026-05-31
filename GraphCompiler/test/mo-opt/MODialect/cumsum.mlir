//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: mo.graph @cumsum(%[[ARG0:.*]]: !mo.tensor<[5], f32>) -> !mo.tensor<[5], f32> {
mo.graph @cumsum(%arg0 : !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) {
  // CHECK-NEXT: %[[RES:.*]] = mo.cumsum(%[[ARG0]]) {axis = 0 : index, exclusive = 0 : index, reverse = 0 : index} : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
  %res = mo.cumsum(%arg0) {axis = 0 : index, reverse = 0 : index, exclusive = 0 : index} : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[5], f32>
  mo.output %res : !mo.tensor<[5], f32>
}
