// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-to-mogg | FileCheck %s


// CHECK-LABEL: mo.graph @no_context
// CHECK: mogg.call.opaque @{{.*}}MapLowerNoContext inputs(%{{.*}} : !mogg.tensor_) outputs(%{{.*}} : !mogg.tensor_)
// The trailing `context(...)` is absent; the call-site spelling has no space
// before its paren, so this cannot match the `_kernel` header's `context (` arg.
// CHECK-NOT: context(
mo.graph @no_context(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_no_context"}(%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: mo.graph @with_context
// CHECK: context (%[[CTX:.*]]: !mogg.context)
// CHECK: mogg.call.opaque @{{.*}}MapLowerOnePlainOutput inputs(%{{.*}} : !mogg.tensor_) outputs(%{{.*}} : !mogg.tensor_) context(%[[CTX]] : !mogg.context)
mo.graph @with_context(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_plain_output"}(%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}
