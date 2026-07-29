// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-to-mogg --mogg-package-variadics-and-scalars --mogg-outline-kernels-into-functions --mogg-to-mgp > %t
// RUN: FileCheck %s < %t
// RUN: FileCheck %s --check-prefix=NONAME < %t

// MOGGToMGP lowers a new-path mogg._kernel into a mogg.generic.compiled.region:
// its !mo.tensor inputs become !mgp.buffer operands, outputs are freshly
// allocated, and inside the body each async block arg is unpacked + rebound to
// the handle the already-outlined mogg.call.stub expects. Dim args are rebound
// to an anonymous !mogg.dim_ (the source param-decl name is dropped, here and on
// the stub signature, so a symbolic dim can't block kernel deduplication).
//
// The region carries the real launch device (from region.setDeviceRef): the
// device attr on mogg.generic.compiled.region is what sequence fusion reads, so
// it is asserted below to keep it from silently regressing to the host default.

// A static-shape kernel: buffer inputs, buffer output, context. No dim args.
// CHECK-LABEL: mgp.model @one_plain
// CHECK: mogg.generic.compiled.region
// CHECK: mogg.async.unpack(%{{.*}}) : !mgp.buffer
// CHECK: mogg._rebind %{{.*}} : !mgp.buffer to !mogg.pointer_
// CHECK: mogg._rebind %{{.*}} to !mogg.context
// CHECK: mogg.call.stub @map_stub_0 %{{.*}} : !mogg.pointer_, !mogg.pointer_, !mogg.context
// CHECK: mogg.async.ready
// CHECK: device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"io.">
// CHECK-LABEL: mogg.kernel.stub @map_stub_0 (%arg0: !mogg.pointer_, %arg1: !mogg.pointer_, %arg2: !mogg.context)
// CHECK-SAME: device = #M.device_ref<"cpu", 0>
// CHECK: mogg._shape.create() {shape = #mosh<ape[2, 2]>{{.*}}} : () -> !mogg.shape
// CHECK: mogg._tensor.create(%arg0, %{{.*}}) {device = {{.*}}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
// CHECK: mogg._call.opaque @{{.*}}MapLowerOnePlainOutput inputs(%{{.*}} : !mogg.tensor_) outputs(%{{.*}} : !mogg.tensor_) context(%arg2 : !mogg.context)
mo.graph @one_plain(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_plain_output"}(%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Dynamic dim: the dim value is materialized as an operand, unpacked, and
// rebound to an anonymous !mogg.dim_ (NOT !mogg.dim_<batch>). The referenced
// stub's dim arg and its in-body shape.create use the anonymous !mogg.dim_ too.
// CHECK-LABEL: mgp.model @dyn
// CHECK: mogg.generic.compiled.region
// CHECK: mogg._rebind %{{.*}} : !kgen.scalar<si64> to !mogg.dim_{{$}}
// CHECK: mogg.call.stub @map_stub_1 %{{.*}} : !mogg.pointer_, !mogg.pointer_, !mogg.dim_, !mogg.context
// CHECK: mogg.async.ready
// CHECK: device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"io..">
// CHECK-LABEL: mogg.kernel.stub @map_stub_1 (%arg0: !mogg.pointer_, %arg1: !mogg.pointer_, %arg2: !mogg.dim_, %arg3: !mogg.context)
// CHECK-SAME: device = #M.device_ref<"cpu", 0>
// CHECK: mogg._shape.create(%arg2) {shape = #mosh<ape[dyn, 4]>{{.*}}} : (!mogg.dim_) -> !mogg.shape
// CHECK: mogg._call.opaque @{{.*}}MapLowerOnePlainOutput inputs(%{{.*}} : !mogg.tensor_) outputs(%{{.*}} : !mogg.tensor_) context(%arg3 : !mogg.context)
mo.graph @dyn(%a: !mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_plain_output"}(%a) : (!mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32>
  mo.output %c : !mo.tensor<[batch, 4], f32>
}

// No named !mogg.dim_<...> may survive MOGGToMGP anywhere -- a symbolic dim name
// would block kernel deduplication.
// NONAME-NOT: !mogg.dim_<
