// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-to-mogg | FileCheck %s

// MAPToMOGG lowers a plain-output `map.iter.opaque` into a `mogg.kernel`:
// inputs stay `!mo.tensor` operands and outputs stay `!mo.tensor` results, but
// inside the body each tensor is reconstructed from a `!mogg.pointer` block
// argument via `mogg.shape.create` / `mogg.tensor.create` (each recording the
// kernel's `device`), and the kernel call becomes `mogg.call.opaque` wired to
// the body's trailing `!mogg.context`. The grouped `_kernel` header lists the
// pointer block args as `inputs(...) outputs(...) context(...)`. (Store-epilogue
// / fused outputs are deferred and not exercised here.)

// Single plain output.
// CHECK-LABEL: mo.graph @one_plain
// CHECK: mogg.kernel (%[[A:.*]] : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK: inputs (%[[IP:.*]]: !mogg.pointer)
// CHECK: outputs (%[[OP:.*]]: !mogg.pointer)
// CHECK: context (%[[CTX:.*]]: !mogg.context) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK:   %[[ISH:.*]] = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape> : () -> !mogg.shape
// CHECK:   %[[IT:.*]] = mogg.tensor.create(%[[IP]], %[[ISH]]) <device = "cpu",{{.*}}mut = false{{.*}}> : (!mogg.pointer, !mogg.shape) -> !mogg.tensor
// CHECK:   %[[OSH:.*]] = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape> : () -> !mogg.shape
// CHECK:   %[[OT:.*]] = mogg.tensor.create(%[[OP]], %[[OSH]]) <device = "cpu",{{.*}}mut = true{{.*}}> : (!mogg.pointer, !mogg.shape) -> !mogg.tensor
// CHECK:   mogg.call.opaque @{{.*}}MapLowerOnePlainOutput inputs(%[[IT]] : !mogg.tensor) outputs(%[[OT]] : !mogg.tensor) context(%[[CTX]] : !mogg.context)
// Nothing is fused here, so both flags keep their `false` default, which the
// printer elides -- their absence is the assertion.
// CHECK-NOT: hasEpilogueFusion
// CHECK-NOT: hasPrologueFusion
mo.graph @one_plain(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_plain_output"}(%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Two plain outputs: one input pointer, two output pointers, one context.
// CHECK-LABEL: mo.graph @two_plain
// CHECK: mogg.kernel (%{{.*}} : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK: inputs (%{{.*}}: !mogg.pointer)
// CHECK: outputs (%{{.*}}: !mogg.pointer, %{{.*}}: !mogg.pointer)
// CHECK: context (%[[CTX:.*]]: !mogg.context) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK:   mogg.call.opaque @{{.*}}MapLowerTwoPlainOutputs inputs(%{{.*}} : !mogg.tensor) outputs(%{{.*}}, %{{.*}} : !mogg.tensor, !mogg.tensor) context(%[[CTX]] : !mogg.context)
mo.graph @two_plain(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c:2 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_two_plain_outputs"}(%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %c#0, %c#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}
