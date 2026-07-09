// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg" | FileCheck %s
// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg" --mogg-verify-kernels 2>&1 | FileCheck %s --check-prefix=VERIFY

// Trait-bound kernels (from the dedicated tensor_arg_trait_stubs package, kept
// separate from MOGGKernelAPITest) import via the tensor-argument trait path --
// no ManagedTensorSlice -- and the per-argument contract
// {role, fused, access, has_capability} distilled from the written trait
// composition is baked onto the kernel decl's `execute` KernelFunctionAttr as
// the trailing `tensor_arg_contracts` array. Non-trait args (the DeviceContext,
// and the implicit error/result refs) are `unit`.

// Dense add: a dense output and two dense inputs; no fusion, no access form.
// CHECK-DAG: registration = "trait_add_explicit"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, {access = "none", fused = false, has_capability = false, role = "input"}, {access = "none", fused = false, has_capability = false, role = "input"}, unit, unit, unit]

// The `Some[...]` spelling distills to a byte-identical contract array (D6
// parity with the explicit-generic spelling above).
// CHECK-DAG: registration = "trait_add_some"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, {access = "none", fused = false, has_capability = false, role = "input"}, {access = "none", fused = false, has_capability = false, role = "input"}, unit, unit, unit]

// Fused, loadable input alongside a dense output: the input arg carries
// access = load, fused = true, has_capability = true (the `Loadable` capability
// leaf is present, not just the `LoadAccess` marker).
// CHECK-DAG: registration = "trait_fused_load"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, {access = "load", fused = true, has_capability = true, role = "input"}, unit, unit, unit]

// These kernels all import fine (well-formed compositions) but are rejected by
// mogg-verify-kernels, which owns the per-function-kind and calling-convention
// policy:
//
//   - a trait arg passed `mut` (non-borrowed): mutability lives in the contract
//     (the role), not the calling convention; and
//   - a trait arg on `update_input_view`: trait args are `execute`-only for now.
// VERIFY-DAG: error: Only the borrowed argument (read) convention is supported for trait-bound tensor arguments (argument at position 0 uses 'mut')
// VERIFY-DAG: error: trait-bound tensor arguments are not supported on update_input_view functions yet (only `execute`)

module {}
