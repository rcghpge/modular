// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-to-mogg --mogg-outline-kernels-into-functions | FileCheck %s

// OutlineKernelsIntoFunctions moves each `mogg._kernel` body into a module-level
// `mogg.kernel.stub` labeled with the `MAP` ABI (`abi = 2`), and replaces the
// kernel body with a single `mogg.call.stub` forwarding the pointer / dim /
// context block args. The `mogg._kernel` container is kept for `MOGGToMGP`.
//
// The stub is named after the ops it fuses with a trailing `_map` ABI tag (e.g.
// `..._map`), not a bare `map_stub_<n>` counter; two kernels that fuse the same
// ops share a preferred name, so the symbol table appends a numeric suffix to
// the second (`..._map_0`), as `@dyn` and `@one_plain` show below.

// Plain output: the kernel body becomes one `mogg.call.stub`; the shape/tensor
// reconstruction + `mogg._call.opaque` move into the MAP-ABI stub.
// CHECK-LABEL: mo.graph @one_plain
// CHECK: mogg._kernel (%{{.*}} : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK:   inputs (%[[IP:.*]]: !mogg.pointer_)
// CHECK:   outputs (%[[OP:.*]]: !mogg.pointer_)
// CHECK:   context (%[[C:.*]]: !mogg.context) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK:     mogg.call.stub @[[STUB0:[A-Za-z0-9_]+_map[0-9_]*]] %[[IP]], %[[OP]], %[[C]] : !mogg.pointer_, !mogg.pointer_, !mogg.context
// CHECK-NOT: mogg._call.opaque
// CHECK: mogg.kernel.stub @[[STUB0]] (%{{.*}}: !mogg.pointer_, %{{.*}}: !mogg.pointer_, %{{.*}}: !mogg.context) fusionInfo = {{.*}} attributes {abi = 2 : i32,
// CHECK:   %[[ISH:.*]] = mogg._shape.create() {{.*}} -> !mogg.shape
// CHECK:   %[[IT:.*]] = mogg._tensor.create(%{{.*}}, %[[ISH]]) {{.*}}mut = false{{.*}} -> !mogg.tensor_
// CHECK:   %[[OT:.*]] = mogg._tensor.create(%{{.*}}, %{{.*}}) {{.*}}mut = true{{.*}} -> !mogg.tensor_
// CHECK:   mogg._call.opaque @{{.*}}MapLowerOnePlainOutput inputs(%[[IT]] : !mogg.tensor_) outputs(%[[OT]] : !mogg.tensor_) context(%{{.*}} : !mogg.context)
// CHECK:   mogg.output
mo.graph @one_plain(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_plain_output"}(%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Fused output: the epilogue lambda + `mogg._tensor.create.epilogue` ride inside
// the stub body alongside the call.
// CHECK-LABEL: mo.graph @one_fused
// CHECK:   inputs (%[[IP:.*]]: !mogg.pointer_)
// CHECK:   outputs (%[[OP:.*]]: !mogg.pointer_)
// CHECK:   context (%[[C:.*]]: !mogg.context) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK:     mogg.call.stub @[[STUB1:[A-Za-z0-9_]+_map[0-9_]*]] %[[IP]], %[[OP]], %[[C]] : !mogg.pointer_, !mogg.pointer_, !mogg.context
// CHECK: mogg.kernel.stub @[[STUB1]] {{.*}} attributes {abi = 2 : i32,
// CHECK:   %{{.*}} = mogg._tensor.create.epilogue %{{.*}} capturing(%{{.*}} : !mogg.tensor_) : !mogg.shape -> !mogg.tensor_ attributes {{.*}} {
// CHECK:   ^bb0(%{{.*}}: f32, %{{.*}}: !map.index<2>, %[[CAP:.*]]: !mogg.tensor_):
// CHECK:     mogg._tensor.store %[[CAP]][%{{.*}}] = %{{.*}}
// CHECK:   }
// CHECK:   mogg._call.opaque @{{.*}}MapLowerOneFusedOutput
// CHECK:   mogg.output
mo.graph @one_fused(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Dynamic dim: the `!mogg.dim_` block arg is forwarded through the call.stub into
// the stub signature, where `mogg._shape.create` consumes it.
// CHECK-LABEL: mo.graph @dyn
// CHECK:   inputs (%[[IP:.*]]: !mogg.pointer_)
// CHECK:   outputs (%[[OP:.*]]: !mogg.pointer_)
// CHECK:   context (%[[C:.*]]: !mogg.context)
// CHECK:   dims (%[[D:.*]]: !mogg.dim_<batch>) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK:     mogg.call.stub @[[STUB2:[A-Za-z0-9_]+_map[0-9_]*]] %[[IP]], %[[OP]], %[[D]], %[[C]] : !mogg.pointer_, !mogg.pointer_, !mogg.dim_<batch>, !mogg.context
// CHECK: mogg.kernel.stub @[[STUB2]] (%{{.*}}: !mogg.pointer_, %{{.*}}: !mogg.pointer_, %[[SD:.*]]: !mogg.dim_<batch>, %{{.*}}: !mogg.context) fusionInfo = {{.*}} attributes {abi = 2 : i32,
// CHECK:   mogg._shape.create(%[[SD]]) {{.*}} : (!mogg.dim_<batch>) -> !mogg.shape
mo.graph @dyn(%a: !mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_plain_output"}(%a) : (!mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32>
  mo.output %c : !mo.tensor<[batch, 4], f32>
}
