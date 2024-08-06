// RUN: mo-opt --shape-inference="kernel-library-paths=%extra_test_lib" --add-fallback-shape-funcs="kernel-library-paths=%extra_test_lib" %s | FileCheck %s


// CHECK-LABEL: mo.graph @custom_op
// CHECK-SAME: <[[D0:.*]], [[D1:.*]] -> [[O0:.*]], [[O1:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[O0]], [[O1]]], f32> {
mo.graph @custom_op(
  %input: !mo.tensor<[?, ?], f32>
) -> !mo.tensor<[?, ?], f32> {

  // We can't symbolicize the fallback shape function call because it contains a
  // side-effecting print statement.
  // TODO: consider inlining some commonly known side-effecting ops like print.
  // CHECK-NEXT: mo.invoke_shape_func["test_unary_kernel_shape_func"]<() -> [[D2:.*]], [[D3:.*]]>(%[[INPUT]])
  // CHECK-SAME:   {dataDeptTensors = [0 : i32], kgenParams = {output_idx = 0 : index}} : (!mo.tensor<[[[D0]], [[D1]]], f32>)

  // CHECK-NEXT:  %[[RES:.*]] = mo.custom {symbol = "test_custom_op"}(%[[INPUT]]) :
  // CHECK-SAME:    (!mo.tensor<[[[D0]], [[D1]]], f32>) -> !mo.tensor<[[[D2]], [[D3]]], f32>
  %res = mo.custom {symbol = "test_custom_op"}(%input) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32>

  // CHECK-NEXT:  mo.output<[[D2]], [[D3]]> %[[RES]] : !mo.tensor<[[[D2]], [[D3]]], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}
