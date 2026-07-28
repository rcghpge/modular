// RUN: mo-opt %s | FileCheck %s

// C2 scaffolding roundtrip: the new advanced-fusion MOGG codegen ops and types
// (!mogg.pointer_ / !mogg.tensor_ / !mogg.dim_) parse and print. ODS-only (no
// lowering); this exercises each op's assembly variants.

// CHECK-LABEL: func.func @codegen_ops
// A `!mogg.dim_` is optionally named after its source dynamic dim (`%d0` prints
// as `!mogg.dim_<D0>`); a bare `!mogg.dim_` (`%dim`) carries no name. Both
// round-trip (the named form is checked on the `shape.create` below).
func.func @codegen_ops(%ptr: !mogg.pointer_, %dim: !mogg.dim_,
                       %d0: !mogg.dim_<D0>,
                       %ctx: !mogg.context, %ab: !mogg.async<!mgp.buffer>,
                       %ta: !mo.tensor<[2, 2], f32>) {
  // shape.create: static-only and with a dynamic dim.
  // CHECK: mogg._shape.create() {{.*}} : () -> !mogg.shape
  %s0 = mogg._shape.create() {shape = #mosh<ape[2, 2]> : !mosh.ape} : () -> !mogg.shape
  // CHECK: mogg._shape.create(%{{.*}}) {{.*}} : (!mogg.dim_<D0>) -> !mogg.shape
  %s1 = mogg._shape.create(%d0) {shape = #mosh<ape[D0, 2]> : !mosh.ape} : (!mogg.dim_<D0>) -> !mogg.shape

  // tensor.create: immutable input vs mutable output. The `device` records the
  // device kind (cpu vs gpu) locally.
  // CHECK: mogg._tensor.create(%{{.*}}, %{{.*}}) {{.*}} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
  %t0 = mogg._tensor.create(%ptr, %s0) {device = #M.device_ref<"cpu", 0>, elementType = f32, io = 0 : i32, mut = false} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_
  %t1 = mogg._tensor.create(%ptr, %s0) {device = #M.device_ref<"gpu", 0>, elementType = f32, io = 1 : i32, mut = true} : (!mogg.pointer_, !mogg.shape) -> !mogg.tensor_

  // rebind: a no-op handle conversion (async buffer -> pointer).
  // CHECK: mogg._rebind %{{.*}} : !mogg.async<!mgp.buffer> to !mogg.pointer_
  %cp = mogg._rebind %ab : !mogg.async<!mgp.buffer> to !mogg.pointer_

  // epilogue tensor: a `!mogg.shape` operand + extra operands (an index) +
  // captured tensors + device kind, holding its own body (block args + a
  // store), and the call that writes into it (with an explicit device
  // context).
  // CHECK: mogg._tensor.create.epilogue %{{.*}}(%{{.*}} : !mogg.dim_) capturing(%{{.*}} : !mogg.tensor_) : !mogg.shape -> !mogg.tensor_
  %epi = mogg._tensor.create.epilogue %s0(%dim : !mogg.dim_) capturing (%t1 : !mogg.tensor_) : !mogg.shape -> !mogg.tensor_
      attributes {device = #M.device_ref<"gpu", 0>, elementType = f32, io = 1 : i32, mut = true} {
  ^bb0(%val: f32, %i: !mogg.dim_):
    // CHECK: mogg._tensor.store %{{.*}}[%{{.*}}] = %{{.*}} : !mogg.tensor_, !mogg.dim_, f32
    mogg._tensor.store %t1[%i] = %val : !mogg.tensor_, !mogg.dim_, f32
  }
  // CHECK: mogg._call.opaque @kernel inputs(%{{.*}} : !mogg.tensor_) outputs(%{{.*}} : !mogg.tensor_) context(%{{.*}} : !mogg.context)
  mogg._call.opaque @kernel inputs(%t0 : !mogg.tensor_) outputs(%epi : !mogg.tensor_) context(%ctx : !mogg.context)
  // a bare call variant with no tensor operands and no context.
  // CHECK: mogg._call.opaque @bare
  mogg._call.opaque @bare

  // kernel region: !mo.tensor input (operand) + output (result) outside; inside,
  // one !mogg.pointer_ per input then output, a context, then the dynamic dims.
  // The body block args print as labeled groups (context before dims), but the
  // in-memory order stays [inputs, outputs, dims, context]. No terminator: the
  // output pointer block arg is the output. Like `mogg.kernel.stub`, the op
  // carries a `device` attribute (the `MOExplicitDevice` trait) recording its
  // execution device.
  // CHECK: mogg._kernel (%{{.*}} : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  // CHECK: inputs (%{{.*}}: !mogg.pointer_)
  // CHECK: outputs (%{{.*}}: !mogg.pointer_)
  // CHECK: context (%{{.*}}: !mogg.context)
  // CHECK: dims (%{{.*}}: !mogg.dim_)
  // CHECK-SAME: attributes {device = #M.device_ref<"gpu", 0>}
  %kr = mogg._kernel (%ta : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
    inputs(%in_ptr : !mogg.pointer_)
    outputs(%out_ptr : !mogg.pointer_)
    context(%kctx : !mogg.context)
    dims(%kd0 : !mogg.dim_) attributes {device = #M.device_ref<"gpu", 0>} {
    // CHECK: mogg._call.opaque @inner
    mogg._call.opaque @inner
  }

  return
}
