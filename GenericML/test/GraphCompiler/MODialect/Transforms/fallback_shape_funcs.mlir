// RUN: mo-opt %s --resolve-unknown-parameters --add-fallback-shape-funcs | FileCheck %s


mo.graph @where(%arg0: !mo.tensor<[?, 10], f32>) -> !mo.tensor<[?, 2], si64> {
  %0 = mo.where(%arg0) : (!mo.tensor<[?, 10], f32>) -> !mo.tensor<[?, 2], si64>
  mo.output %0 : !mo.tensor<[?, 2], si64>
}
// CHECK-LABEL: mo.graph @where
// CHECK:(%[[ARG0:.*]]: !mo.tensor<[[[IN_PARAM:.*]], 10], f32>) -> !mo.tensor<[[[OUT_PARAM:.*]], 2], si64>
// CHECK-NEXT: %[[IN:.*]] = mo.rebind<() -> [[P:.*]]>(%[[ARG0]])
// CHECK-NEXT: mo.invoke_shape_func["where_shape"]<() -> [[WHERE_PARAM:.*]], [[UNUSED_PARAM:.*]]>(%[[IN]]) {dataDeptTensors = [0 : i32]}
// CHECK-NEXT: %[[WHERE:.*]] = mo.where(%[[IN]]) : (!mo.tensor<[[[P]], 10], f32>) -> !mo.tensor<[[[WHERE_PARAM]], 2], si64>
// CHECK-NEXT: mo.rebind(%[[WHERE]]) : !mo.tensor<[[[WHERE_PARAM]], 2], si64> -> !mo.tensor<[[[OUT_PARAM]], 2], si64>
// CHECK-NEXT: mo.output<[[WHERE_PARAM]]> %[[WHERE]] : !mo.tensor<[[[WHERE_PARAM]], 2], si64>
