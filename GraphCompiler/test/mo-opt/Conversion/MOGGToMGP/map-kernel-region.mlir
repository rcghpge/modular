// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-to-mogg --mogg-conform-call-operands --mogg-outline-kernels-into-functions --mogg-to-mgp > %t
// RUN: FileCheck %s < %t
// RUN: FileCheck %s --check-prefix=NONAME < %t

// The region's block arguments print their location after their type, where
// the parser reads it.
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-to-mogg --mogg-conform-call-operands --mogg-outline-kernels-into-functions --mogg-to-mgp --mlir-print-debuginfo | mo-opt --mlir-print-debuginfo | FileCheck %s --check-prefix=LOC
// LOC: mogg.generic.compiled.region (%{{.*}}) : (%{{[^ ]*}}: !mef.chain loc(

// MOGGToMGP lowers a new-path mogg.kernel into a mogg.generic.compiled.region:
// its !mo.tensor inputs become !mgp.buffer operands, outputs are freshly
// allocated, and inside the body each async block arg is unpacked + rebound to
// the handle the already-outlined mogg.call.stub expects. Dim args are rebound
// to an anonymous !mogg.dim (the source param-decl name is dropped, here and on
// the stub signature, so a symbolic dim can't block kernel deduplication).
//
// The guarding chain leads the region's operands -- it is what orders the region
// after the executes that write its inputs -- and has no stub argument of its
// own, so the stub call starts one block arg later.
//
// The region carries the real launch device (from region.setDeviceRef): the
// device attr on mogg.generic.compiled.region is what sequence fusion reads, so
// it is asserted below to keep it from silently regressing to the host default.

// A static-shape kernel: buffer inputs, buffer output, context. No dim args.
// CHECK-LABEL: mgp.model @one_plain
// CHECK: mogg.generic.compiled.region (%{{.*}}) : (%{{.*}}: !mef.chain,
// CHECK: mogg.async.unpack(%{{.*}}) : !mgp.buffer
// CHECK: mogg.rebind %{{.*}} : !mgp.buffer to !mogg.pointer
// CHECK: mogg.rebind %{{.*}} to !mogg.context
// CHECK: mogg.call.stub @[[STUB0:[A-Za-z0-9_]+_mogg[0-9_]*]] %{{.*}} : !mogg.pointer, !mogg.pointer, !mogg.context
// CHECK: mogg.async.ready
// CHECK: device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"iio.">
// Async op: the region keeps `isAsync = true` (was hardcoded `false` before
// KernelOp had an `isAsync` attribute).
// CHECK-SAME: isAsync = true
// CHECK: mogg.kernel.stub @[[STUB0]] (%arg0: !mogg.pointer, %arg1: !mogg.pointer, %arg2: !mogg.context)
// CHECK-SAME: device = "cpu"
// CHECK: mogg.shape.create() {shape = #mosh<ape[2, 2]>{{.*}}} : () -> !mogg.shape
// CHECK: mogg.tensor.create(%arg0, %{{.*}}) {device = {{.*}}} : (!mogg.pointer, !mogg.shape) -> !mogg.tensor
// CHECK: mogg.call.opaque @{{.*}}MapLowerOnePlainOutput inputs(%{{.*}} : !mogg.tensor) outputs(%{{.*}} : !mogg.tensor) context(%arg2 : !mogg.context)
mo.graph @one_plain(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_plain_output"}(%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Dynamic dim: the dim value is materialized as an operand, unpacked, and
// rebound to an anonymous !mogg.dim (NOT !mogg.dim<batch>). The referenced
// stub's dim arg and its in-body shape.create use the anonymous !mogg.dim too.
// CHECK-LABEL: mgp.model @dyn
// CHECK: mogg.generic.compiled.region (%{{.*}}) : (%{{.*}}: !mef.chain,
// CHECK: mogg.rebind %{{.*}} : !kgen.scalar<si64> to !mogg.dim{{$}}
// CHECK: mogg.call.stub @[[STUB1:[A-Za-z0-9_]+_mogg[0-9_]*]] %{{.*}} : !mogg.pointer, !mogg.pointer, !mogg.dim, !mogg.context
// CHECK: mogg.async.ready
// CHECK: device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"iio..">
// CHECK: mogg.kernel.stub @[[STUB1]] (%arg0: !mogg.pointer, %arg1: !mogg.pointer, %arg2: !mogg.dim, %arg3: !mogg.context)
// CHECK-SAME: device = "cpu"
// CHECK: mogg.shape.create(%arg2) {shape = #mosh<ape[dyn, 4]>{{.*}}} : (!mogg.dim) -> !mogg.shape
// CHECK: mogg.call.opaque @{{.*}}MapLowerOnePlainOutput inputs(%{{.*}} : !mogg.tensor) outputs(%{{.*}} : !mogg.tensor) context(%arg3 : !mogg.context)
mo.graph @dyn(%a: !mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_plain_output"}(%a) : (!mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32>
  mo.output %c : !mo.tensor<[batch, 4], f32>
}

// No named !mogg.dim<...> may survive MOGGToMGP anywhere -- a symbolic dim name
// would block kernel deduplication.
// NONAME-NOT: !mogg.dim<
