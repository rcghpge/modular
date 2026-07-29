// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-to-mogg | FileCheck %s

// MAPToMOGG lowers a plain-output `map.iter.opaque` into a `mogg._kernel`:
// inputs stay `!mo.tensor` operands and outputs stay `!mo.tensor` results, but
// inside the body each tensor is reconstructed from a `!mogg.pointer_` block
// argument via `mogg._shape.create` / `mogg._tensor.create` (each recording the
// kernel's `device`), and the kernel call becomes `mogg._call.opaque` wired to
// the body's trailing `!mogg.context`. The grouped `_kernel` header lists the
// pointer block args as `inputs(...) outputs(...) context(...)`. (Store-epilogue
// / fused outputs are deferred and not exercised here.)

// Single plain output.
// CHECK-LABEL: mo.graph @one_plain
// CHECK: mogg._kernel (%[[A:.*]] : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK: inputs (%[[IP:.*]]: !mogg.pointer_)
// CHECK: outputs (%[[OP:.*]]: !mogg.pointer_)
// CHECK: context (%[[CTX:.*]]: !mogg.context) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK:   %[[ISH:.*]] = mogg._shape.create() {shape = #mosh<ape[2, 2]> : !mosh.ape} : () -> !mogg.shape
// CHECK:   %[[IT:.*]] = mogg._tensor.create(%[[IP]], %[[ISH]]) {device = #M.device_ref<"cpu", 0>,{{.*}}mut = false} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK:   %[[OSH:.*]] = mogg._shape.create() {shape = #mosh<ape[2, 2]> : !mosh.ape} : () -> !mogg.shape
// CHECK:   %[[OT:.*]] = mogg._tensor.create(%[[OP]], %[[OSH]]) {device = #M.device_ref<"cpu", 0>,{{.*}}mut = true} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK:   mogg._call.opaque @{{.*}}MapLowerOnePlainOutput inputs(%[[IT]] : !mogg.tensor_) outputs(%[[OT]] : !mogg.tensor_) context(%[[CTX]] : !mogg.context)
mo.graph @one_plain(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_plain_output"}(%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Two plain outputs: one input pointer, two output pointers, one context.
// CHECK-LABEL: mo.graph @two_plain
// CHECK: mogg._kernel (%{{.*}} : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK: inputs (%{{.*}}: !mogg.pointer_)
// CHECK: outputs (%{{.*}}: !mogg.pointer_, %{{.*}}: !mogg.pointer_)
// CHECK: context (%[[CTX:.*]]: !mogg.context) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK:   mogg._call.opaque @{{.*}}MapLowerTwoPlainOutputs inputs(%{{.*}} : !mogg.tensor_) outputs(%{{.*}}, %{{.*}} : !mogg.tensor_, !mogg.tensor_) context(%[[CTX]] : !mogg.context)
mo.graph @two_plain(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c:2 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_two_plain_outputs"}(%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %c#0, %c#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}
