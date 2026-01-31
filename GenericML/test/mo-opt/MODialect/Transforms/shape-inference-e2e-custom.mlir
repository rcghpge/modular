// RUN: mo-opt --mo-shape-inference --mo-add-fallback-shape-funcs="kernel-library-paths=%kernel_api_test_lib" %s | FileCheck %s


// CHECK-LABEL: mo.graph @custom_op
// CHECK-SAME: <[[D0:.*]], [[D1:.*]] -> [[O0:.*]], [[O1:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[O0]], [[O1]]], f32> {
mo.graph @custom_op<D0, D1 -> O0, O1>(
  %input: !mo.tensor<[D0, D1], f32>
) -> !mo.tensor<[O0, O1], f32> {

  // We can't symbolicize the fallback shape function call because it contains a
  // side-effecting print statement.
  // TODO: consider inlining some commonly known side-effecting ops like print.
  // CHECK-NEXT: mo.invoke_shape_func["test_custom_op"]<() -> [[D2:.*]], [[D3:.*]]>(%[[INPUT]])
  // CHECK-SAME:   {dataDeptTensors = [0 : i32], kgenParams = {output_idx = 0 : index}} : (!mo.tensor<[[[D0]], [[D1]]], f32>)

  // CHECK-NEXT:  %[[RES:.*]] = mo.custom {symbol = "test_custom_op"}(%[[INPUT]]) :
  // CHECK-SAME:    (!mo.tensor<[[[D0]], [[D1]]], f32>) -> !mo.tensor<[[[D2]], [[D3]]], f32>
  %res = mo.custom<() -> D2, D3> {device = #M.device_ref<"cpu", 0>, symbol = "test_custom_op"}(%input) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, D3], f32>

  // CHECK-NEXT:  mo.output<[[D2]], [[D3]]> %[[RES]] : !mo.tensor<[[[D2]], [[D3]]], f32>
  mo.output<D2, D3> %res : !mo.tensor<[D2, D3], f32>
}
