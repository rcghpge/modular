// RUN: mo-opt --fuse-elementwise %s | FileCheck %s

// original:
// mo.graph @basicFusion(%arg0 : !mo.tensor<[8],f32>, %arg1 : !mo.tensor<[8],f32>) -> (!mo.tensor<[8], f32>) {
// %0 = mo.mul(%arg0, %arg1) : !mo.tensor<[8], f32>
//  %1 = mo.add(%0, %arg1) : !mo.tensor<[8], f32>
//  %2 = mo.add(%0, %arg0) : !mo.tensor<[8], f32>
//  %3 = mo.add(%1, %2) : !mo.tensor<[8], f32>
//  mo.output %3 : !mo.tensor<[8], f32>


mo.graph @basicFusion(%arg0: !mo.tensor<[8], f32>, %arg1: !mo.tensor<[8], f32>) -> !mo.tensor<[8], f32> {
  %0 = "mogg.elementwise"(%arg0, %arg1) ({
  ^bb0(%arg2: !mo.tensor<[8], f32>, %arg3: !mo.tensor<[8], f32>):
    %4 = mo.mul(%arg2, %arg3) : !mo.tensor<[8], f32>
    mogg.output %4 : !mo.tensor<[8], f32>
  }) : (!mo.tensor<[8], f32>, !mo.tensor<[8], f32>) -> !mo.tensor<[8], f32>

  %1 = "mogg.elementwise"(%0, %arg1) ({
  ^bb0(%arg2: !mo.tensor<[8], f32>, %arg3: !mo.tensor<[8], f32>):
    %4 = mo.add(%arg2, %arg3) : !mo.tensor<[8], f32>
    mogg.output %4 : !mo.tensor<[8], f32>
  }) : (!mo.tensor<[8], f32>, !mo.tensor<[8], f32>) -> !mo.tensor<[8], f32>
  %2 = "mogg.elementwise"(%0, %arg0) ({
  ^bb0(%arg2: !mo.tensor<[8], f32>, %arg3: !mo.tensor<[8], f32>):
    %4 = mo.add(%arg2, %arg3) : !mo.tensor<[8], f32>
    mogg.output %4 : !mo.tensor<[8], f32>
  }) : (!mo.tensor<[8], f32>, !mo.tensor<[8], f32>) -> !mo.tensor<[8], f32>

  %3 = "mogg.elementwise"(%1, %2) ({
  ^bb0(%arg2: !mo.tensor<[8], f32>, %arg3: !mo.tensor<[8], f32>):
    %4 = mo.add(%arg2, %arg3) : !mo.tensor<[8], f32>
    mogg.output %4 : !mo.tensor<[8], f32>
  }) : (!mo.tensor<[8], f32>, !mo.tensor<[8], f32>) -> !mo.tensor<[8], f32>
  mo.output %3 : !mo.tensor<[8], f32>
}


// CHECK: mo.graph @basicFusion(%[[ARG0:.*]]: !mo.tensor<[8], f32>, %[[ARG1:.*]]: !mo.tensor<[8], f32>)
// CHECK-NEXT: %[[OUT:.*]] = "mogg.elementwise"(%[[ARG0]], %[[ARG1]]) ({
// CHECK-NEXT: ^bb0(%[[BLOCK_A0:.*]]: !mo.tensor<[8], f32>, %[[BLOCK_A1:.*]]: !mo.tensor<[8], f32>):

// First fusion
// CHECK-NEXT: %[[BLOCK_0:.*]] = mo.mul(%[[BLOCK_A0]], %[[BLOCK_A1]])
// CHECK-NEXT: %[[BLOCK_1:.*]] = mo.add(%[[BLOCK_0]], %[[BLOCK_A0]])

// Second fusion (CSE eliminates the extra mul)
// CHECK-NEXT: %[[BLOCK_2:.*]] = mo.mul(%[[BLOCK_A0]], %[[BLOCK_A1]])
// CHECK-NEXT: %[[BLOCK_3:.*]] = mo.add(%[[BLOCK_2]], %[[BLOCK_A1]])

// CHECK-NEXT: %[[BLOCK_4:.*]] = mo.add(%[[BLOCK_3]], %[[BLOCK_1]])
// CHECK-NEXT: mogg.output %[[BLOCK_4]]
// CHECK-NEXT: }
// CHECK-NEXT: mo.output %[[OUT]] : !mo.tensor<[8], f32>




mo.graph @dontFuseDynamic(%arg0: !mo.tensor<[?], f32>, %arg1: !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32> {
  %0 = "mogg.elementwise"(%arg0, %arg1) ({
  ^bb0(%arg2: !mo.tensor<[?], f32>, %arg3: !mo.tensor<[?], f32>):
    %4 = mo.mul(%arg2, %arg3) : !mo.tensor<[?], f32>
    mogg.output %4 : !mo.tensor<[?], f32>
  }) : (!mo.tensor<[?], f32>, !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32>
  %1 = "mogg.elementwise"(%0, %arg1) ({
  ^bb0(%arg2: !mo.tensor<[?], f32>, %arg3: !mo.tensor<[?], f32>):
    %4 = mo.add(%arg2, %arg3) : !mo.tensor<[?], f32>
    mogg.output %4 : !mo.tensor<[?], f32>
  }) : (!mo.tensor<[?], f32>, !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32>
  mo.output %1 : !mo.tensor<[?], f32>
}


// CHECK: mo.graph @dontFuseDynamic(%[[ARG0:.*]]: !mo.tensor<[?], f32>, %[[ARG1:.*]]: !mo.tensor<[?], f32>)
// CHECK-NEXT:  %[[OUT_1:.*]] = "mogg.elementwise"(%[[ARG0]], %[[ARG1]]) ({
// CHECK-NEXT: ^bb0
// CHECK-NEXT:    %[[BLOCK:.*]] = mo.mul
// CHECK-NEXT:    mogg.output %[[BLOCK]] : !mo.tensor<[?], f32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[OUT_2:.*]] = "mogg.elementwise"(%[[OUT_1]], %[[ARG1]]) ({
// CHECK-NEXT:  ^bb0
// CHECK-NEXT:    %[[BLOCK:.*]] = mo.add
// CHECK-NEXT:    mogg.output %[[BLOCK]] : !mo.tensor<[?], f32>
// CHECK-NEXT:  }
// CHECK-NEXT:  mo.output %1 : !mo.tensor<[?], f32>
