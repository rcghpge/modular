// RUN: mo-opt %s --mogg-anonymize-dim-names | FileCheck %s
// RUN: mo-opt %s --mogg-anonymize-dim-names --mogg-anonymize-dim-names | FileCheck %s

// A symbolic dim name on a stub's boundary would block MergeIdenticalKernelStubs
// from recognizing two structurally-identical stubs that differ only in a dim's
// name as duplicates, so this pass strips it -- from the block arg's type, the
// function-type signature, and any in-body use (changing a block arg's type
// updates its uses automatically). Idempotent: the second RUN line reapplies
// the pass a second time and checks the output is unchanged.

// CHECK-LABEL: mogg.kernel.stub @named_dim
// CHECK-SAME: (%[[ARG:.*]]: !mogg.dim_)
// CHECK: mogg._shape.create(%[[ARG]]) {{.*}} : (!mogg.dim_) -> !mogg.shape
// CHECK-NOT: !mogg.dim_<
mogg.kernel.stub @named_dim(%d: !mogg.dim_<batch>)
  fusionInfo = named_dim
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true} {
  %s = mogg._shape.create(%d) {shape = #mosh<ape[batch, 4]> : !mosh.ape} : (!mogg.dim_<batch>) -> !mogg.shape
  mogg.output
}

// A stub with no dim args at all is left alone.
// CHECK-LABEL: mogg.kernel.stub @no_dims
// CHECK-SAME: ()
mogg.kernel.stub @no_dims()
  fusionInfo = no_dims
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true} {
  mogg.output
}

// An already-anonymous dim arg is left alone (no-op, not just harmless).
// CHECK-LABEL: mogg.kernel.stub @already_anonymous
// CHECK-SAME: (%{{.*}}: !mogg.dim_)
mogg.kernel.stub @already_anonymous(%d: !mogg.dim_)
  fusionInfo = already_anonymous
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true} {
  mogg.output
}
