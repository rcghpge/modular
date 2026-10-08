// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-allocate-accumulating-buffers --map-to-mogg --mogg-outline-kernels-into-functions | FileCheck %s

// OutlineKernelsIntoFunctions moves each `mogg.kernel` body into a module-level
// `mogg.kernel.stub` labeled with the `MAP` ABI (`abi = 2`), and replaces the
// kernel body with a single `mogg.call.stub` forwarding the pointer / dim /
// context block args. The `mogg.kernel` container is kept for `MOGGToMGP`.
//
// The stub is named after the ops it fuses with a trailing `_mogg` ABI tag (e.g.
// `..._mogg`), not a bare `mogg_stub_<n>` counter; two kernels that fuse the same
// ops share a preferred name, so the symbol table appends a numeric suffix to
// the second (`..._mogg_0`), as `@dyn` and `@one_plain` show below.

// Plain output: the kernel body becomes one `mogg.call.stub`; the shape/tensor
// reconstruction + `mogg.call.opaque` move into the MAP-ABI stub.
// CHECK-LABEL: mo.graph @one_plain
// CHECK: mogg.kernel (%{{.*}} : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK:   inputs (%[[IP:.*]]: !mogg.pointer)
// CHECK:   outputs (%[[OP:.*]]: !mogg.pointer)
// CHECK:   context (%[[C:.*]]: !mogg.context) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK:     mogg.call.stub @[[STUB0:[A-Za-z0-9_]+_mogg[0-9_]*]] %[[IP]], %[[OP]], %[[C]] : !mogg.pointer, !mogg.pointer, !mogg.context
// CHECK-NOT: mogg.call.opaque
// CHECK: mogg.kernel.stub @[[STUB0]] (%{{.*}}: !mogg.pointer, %{{.*}}: !mogg.pointer, %{{.*}}: !mogg.context) fusionInfo = {{.*}} attributes {abi = 2 : i32,
// CHECK:   %[[ISH:.*]] = mogg.shape.create() {{.*}} -> !mogg.shape
// CHECK:   %[[IT:.*]] = mogg.tensor.create(%{{.*}}, %[[ISH]]) {{.*}}mut = false{{.*}} -> !mogg.tensor
// CHECK:   %[[OT:.*]] = mogg.tensor.create(%{{.*}}, %{{.*}}) {{.*}}mut = true{{.*}} -> !mogg.tensor
// CHECK:   mogg.call.opaque @{{.*}}MapLowerOnePlainOutput inputs(%[[IT]] : !mogg.tensor) outputs(%[[OT]] : !mogg.tensor) context(%{{.*}} : !mogg.context)
// CHECK:   mogg.output
mo.graph @one_plain(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_plain_output"}(%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Fused output: the epilogue lambda + `mogg.tensor.create.epilogue` ride inside
// the stub body alongside the call.
// CHECK-LABEL: mo.graph @one_fused
// CHECK:   inputs (%[[IP:.*]]: !mogg.pointer)
// CHECK:   outputs (%[[OP:.*]]: !mogg.pointer)
// CHECK:   context (%[[C:.*]]: !mogg.context) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK:     mogg.call.stub @[[STUB1:[A-Za-z0-9_]+_mogg[0-9_]*]] %[[IP]], %[[OP]], %[[C]] : !mogg.pointer, !mogg.pointer, !mogg.context
// CHECK: mogg.kernel.stub @[[STUB1]] {{.*}} attributes {abi = 2 : i32,
// CHECK:   %{{.*}} = mogg.tensor.create.epilogue %{{.*}} capturing(%{{.*}} : !mogg.tensor) ptr %{{.*}} : !mogg.pointer : !mogg.shape -> !mogg.tensor {{.*}} {
// CHECK:   ^bb0(%{{.*}}: f32, %{{.*}}: !map.index<2>, %[[CAP:.*]]: !mogg.tensor):
// CHECK:     mogg.tensor.store %[[CAP]][%{{.*}}] = %{{.*}}
// CHECK:   }
// CHECK:   mogg.call.opaque @{{.*}}MapLowerOneFusedOutput
// CHECK:   mogg.output
mo.graph @one_fused(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Dynamic dim: the `!mogg.dim` block arg is forwarded through the call.stub into
// the stub signature, where `mogg.shape.create` consumes it.
// CHECK-LABEL: mo.graph @dyn
// CHECK:   inputs (%[[IP:.*]]: !mogg.pointer)
// CHECK:   outputs (%[[OP:.*]]: !mogg.pointer)
// CHECK:   context (%[[C:.*]]: !mogg.context)
// CHECK:   dims (%[[D:.*]]: !mogg.dim<batch>) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK:     mogg.call.stub @[[STUB2:[A-Za-z0-9_]+_mogg[0-9_]*]] %[[IP]], %[[OP]], %[[D]], %[[C]] : !mogg.pointer, !mogg.pointer, !mogg.dim<batch>, !mogg.context
// CHECK: mogg.kernel.stub @[[STUB2]] (%{{.*}}: !mogg.pointer, %{{.*}}: !mogg.pointer, %[[SD:.*]]: !mogg.dim<batch>, %{{.*}}: !mogg.context) fusionInfo = {{.*}} attributes {abi = 2 : i32,
// CHECK:   mogg.shape.create(%[[SD]]) {{.*}} : (!mogg.dim<batch>) -> !mogg.shape
mo.graph @dyn(%a: !mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_plain_output"}(%a) : (!mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32>
  mo.output %c : !mo.tensor<[batch, 4], f32>
}
