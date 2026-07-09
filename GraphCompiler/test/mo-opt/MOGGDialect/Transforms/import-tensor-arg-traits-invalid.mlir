// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_invalid_pkg" 2>&1 | FileCheck %s

// Each kernel in the tensor_arg_trait_invalid_stubs package has a trait-bound
// tensor argument that violates exactly one validation rule. Import-time
// validation must reject every one of them and report them all (not abort on
// the first), so all of these diagnostics are present in one run.

// `Some[DenseTensor]` -- no role trait.
// CHECK-DAG: error: trait-bound tensor argument has no role trait

// `Input & Output & DenseTensor` -- two role traits.
// CHECK-DAG: error: trait-bound tensor argument has more than one role trait

// `Output & Fused & DenseTensor` -- fused with no access form.
// CHECK-DAG: error: a Fused tensor argument must declare an access form

// `Output & DenseTensor & Copyable` -- a foreign (stdlib) trait in the bound.
// CHECK-DAG: references 'Copyable', which is not a tensor-argument trait

// `Input & Fused & Storable` -- store access on a read-only Input.
// CHECK-DAG: error: store access is not valid on a read-only Input argument

// A trait-bound arg on a shape function: trait args are execute-only, and shape
// functions are rejected at import (a separate decl op, so not handled by the
// KernelDeclOp-based verify check).
// CHECK-DAG: error: trait-bound tensor arguments are not supported on shape functions yet (only `execute`)

module {}
