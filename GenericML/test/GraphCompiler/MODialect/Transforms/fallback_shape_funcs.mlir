// RUN: mo-opt %s --add-fallback-shape-funcs | FileCheck %s


mo.graph @where<D0 -> D1>(%arg0: !mo.tensor<[D0, 10], f32>) -> !mo.tensor<[D1, 2], si64> {
  %0 = mo.where<() -> D2>(%arg0) : (!mo.tensor<[D0, 10], f32>) -> !mo.tensor<[D2, 2], si64>
  mo.output<D2> %0 : !mo.tensor<[D2, 2], si64>
}
// CHECK-LABEL: mo.graph @where
// CHECK:(%[[IN:.*]]: !mo.tensor<[[[IN_PARAM:.*]], 10], f32>) -> !mo.tensor<[[[OUT_PARAM:.*]], 2], si64>
// CHECK-NEXT: mo.invoke_shape_func["where_shape"]<() -> [[WHERE_PARAM:.*]], [[UNUSED_PARAM:.*]]>(%[[IN]]) {dataDeptTensors = [0 : i32]}
// CHECK-NEXT: %[[WHERE:.*]] = mo.where(%[[IN]]) : (!mo.tensor<[[[IN_PARAM]], 10], f32>) -> !mo.tensor<[[[WHERE_PARAM]], 2], si64>
// CHECK-NEXT: mo.output<[[WHERE_PARAM]]> %[[WHERE]] : !mo.tensor<[[[WHERE_PARAM]], 2], si64>
