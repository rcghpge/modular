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

// Two dense outputs then a dense input: two leading `output` contracts.
// CHECK-DAG: registration = "trait_multi_output"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, {access = "none", fused = false, has_capability = false, role = "output"}, {access = "none", fused = false, has_capability = false, role = "input"}, unit, unit, unit]

// A dense output and a `MutableInput` arg: the second contract is the
// `mutable_input` role.
// CHECK-DAG: registration = "trait_mutable_input"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, {access = "none", fused = false, has_capability = false, role = "mutable_input"}, unit, unit, unit]

// Mixed lane: a trait output (contract) followed by two legacy
// ManagedTensorSlice inputs (no contract -> `unit`).
// CHECK-DAG: registration = "trait_mixed_legacy"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, unit, unit, unit, unit, unit]

// Mixed lane, other direction: a legacy output (`unit`) followed by two trait
// inputs (contracts).
// CHECK-DAG: registration = "trait_mixed_legacy_out"{{.*}}[unit, {access = "none", fused = false, has_capability = false, role = "input"}, {access = "none", fused = false, has_capability = false, role = "input"}, unit, unit, unit]

// D6 parity: explicit-generic spellings distill to byte-identical contract
// arrays as their `Some[...]` twins above, for ALL roles (not just dense add).
// CHECK-DAG: registration = "trait_multi_output_explicit"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, {access = "none", fused = false, has_capability = false, role = "output"}, {access = "none", fused = false, has_capability = false, role = "input"}, unit, unit, unit]
// CHECK-DAG: registration = "trait_mutable_input_explicit"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, {access = "none", fused = false, has_capability = false, role = "mutable_input"}, unit, unit, unit]
// CHECK-DAG: registration = "trait_fused_load_explicit"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, {access = "load", fused = true, has_capability = true, role = "input"}, unit, unit, unit]

// Leading mutable_input: the first contract is the `mutable_input` role, then a
// dense input.
// CHECK-DAG: registration = "trait_leading_mutable"{{.*}}[{access = "none", fused = false, has_capability = false, role = "mutable_input"}, {access = "none", fused = false, has_capability = false, role = "input"}, unit, unit, unit]

// All-outputs: three leading `output` contracts.
// CHECK-DAG: registration = "trait_all_outputs"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, {access = "none", fused = false, has_capability = false, role = "output"}, {access = "none", fused = false, has_capability = false, role = "output"}, unit, unit, unit]

// Non-leading output: input, output, input contracts (the role array records
// the written order; the DPS-split semantics are covered in the role-plumbing
// dump-kernels test).
// CHECK-DAG: registration = "trait_non_leading_output"{{.*}}[{access = "none", fused = false, has_capability = false, role = "input"}, {access = "none", fused = false, has_capability = false, role = "output"}, {access = "none", fused = false, has_capability = false, role = "input"}, unit, unit, unit]

// Mixed lane with a legacy MUTABLE arg: a trait output (contract) then a legacy
// `MutableInputTensor` (no contract -> `unit`).
// CHECK-DAG: registration = "trait_mixed_legacy_mut"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, unit, unit, unit, unit]

// Mixed lane with a fused trait arg: a trait output, a legacy `InputTensor`
// (`unit`), then a fused/loadable trait input.
// CHECK-DAG: registration = "trait_mixed_legacy_fused"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, unit, {access = "load", fused = true, has_capability = true, role = "input"}, unit, unit, unit]

// These kernels all import fine (well-formed compositions) but are rejected by
// mogg-verify-kernels, which owns the per-function-kind and calling-convention
// policy:
//
//   - a trait arg passed `mut` (non-borrowed): mutability lives in the contract
//     (the role), not the calling convention; and
//   - a trait arg on `update_input_view`: trait args are `execute`-only for now.
// VERIFY-DAG: error: Only the borrowed argument (read) convention is supported for trait-bound tensor arguments (argument at position 0 uses 'mut')
// VERIFY-DAG: error: trait-bound tensor arguments are not supported on update_input_view functions yet (only `execute`)

// Off-by-one guard for the trait exclusion in the legacy borrow-check: a
// signature with a borrowed trait input at position 0 AND a legacy
// `ManagedTensorSlice` (`OutputTensor`) at position 1 passed `mut`. The legacy
// (`isTensorType`-keyed) borrow-check must STILL fire for the legacy arg at
// position 1 -- the `!getTensorArgTraitType(...)` exclusion skips ONLY the
// position-0 trait arg. The message is the LEGACY one ("for tensor arguments"),
// not the trait-lane one, proving the legacy lane is still reached.
// VERIFY-DAG: error: Only the borrowed argument (read) convention is supported for tensor arguments (argument at position 1 uses 'mut')

module {}
