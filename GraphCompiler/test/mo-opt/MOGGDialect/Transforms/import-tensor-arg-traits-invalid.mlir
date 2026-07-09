// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_invalid_pkg" 2>&1 | FileCheck %s

// Each kernel in the tensor_arg_trait_invalid_stubs package has a trait-bound
// tensor argument that violates exactly one validation rule. Import-time
// validation must reject every one of them and report them all (not abort on
// the first), so all of these diagnostics are present in one run. This is the
// rejection half of the trait-lane distill matrix (the valid half is the
// `IMPORT-DAG` set in lower-tensor-arg-traits.mlir); together they replace the
// deleted `TestTensorArgContract.cpp` gtest.

// === Role errors ========================================================== //

// `Some[DenseTensor]` (kernel `trait_no_role`) -- no role trait. Also covers
// the empty-role boundary: a tensor arg with metadata but no Input/Output/
// MutableInput.
// CHECK-DAG: error: trait-bound tensor argument has no role trait

// `Input & Output & DenseTensor` (kernel `trait_two_roles`) -- two role traits.
// CHECK-DAG: error: trait-bound tensor argument has more than one role trait

// `Input & MutableInput & DenseTensor` (kernel `trait_input_and_mutable_input`)
// -- the Input/MutableInput pair specifically. Exact-leaf matching means
// `MutableInput` does not subsume `Input`, so both roles count (a second
// "more than one role trait" diagnostic in the same run).
// CHECK-DAG: error: trait-bound tensor argument has more than one role trait

// === Access-form errors =================================================== //

// `Output & Fused & DenseTensor` (kernel `trait_fused_no_access`) -- fused with
// no access form.
// CHECK-DAG: error: a Fused tensor argument must declare an access form

// `MutableInput & Fused & Loadable & Storable`
// (kernel `trait_multi_access_forms`) -- two access forms (load + store).
// CHECK-DAG: error: trait-bound tensor argument declares more than one access form

// `Input & LoadAccess & StoreAccess`
// (kernel `trait_multi_access_markers_nonfused`) -- two access MARKERS on a
// non-fused arg; the multi-access check fires independent of `Fused` (a second
// "more than one access form" diagnostic in the same run).
// CHECK-DAG: error: trait-bound tensor argument declares more than one access form

// === Role / access compatibility ========================================== //

// `Output & Fused & Loadable` (kernel `trait_load_on_output`) -- load access on
// an Output.
// CHECK-DAG: error: load access is not valid on an Output argument

// `Input & Fused & Storable` (kernel `trait_store_on_input`) -- store access on
// a read-only Input.
// CHECK-DAG: error: store access is not valid on a read-only Input argument

// `Input & Fused & Transformable` (kernel `trait_transform_on_input`) --
// transform access on a read-only Input.
// CHECK-DAG: error: transform access requires an Output argument

// `MutableInput & Fused & Transformable`
// (kernel `trait_transform_on_mutable_input`) -- transform access on a mutable
// input; transform requires an Output (a second "transform access requires an
// Output argument" diagnostic in the same run).
// CHECK-DAG: error: transform access requires an Output argument

// === Foreign trait ======================================================== //

// `Output & DenseTensor & Copyable` (kernel `trait_foreign`) -- a foreign
// (stdlib) trait in the bound.
// CHECK-DAG: references 'Copyable', which is not a tensor-argument trait

// === Function-kind gating ================================================= //

// A trait-bound arg on a shape function (kernel `trait_shape_fn`): trait args
// are execute-only, and shape functions are rejected at import (a separate decl
// op, so not handled by the KernelDeclOp-based verify check).
// CHECK-DAG: error: trait-bound tensor arguments are not supported on shape functions yet (only `execute`)

module {}
