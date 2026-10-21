// RUN: not kgen-opt -lower-global-pop-to-llvm -verify-diagnostics %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @external_call(%a: !kgen.simd<1, ui32>) {
  pop.external_call @foo(%a) : (!kgen.simd<1, ui32>) -> ()
  // expected-error @below {{existing function with conflicting signature}}
  // expected-error @below {{failed to legalize}}
  %0 = pop.external_call @foo(%a) : (!kgen.simd<1, ui32>) -> !kgen.simd<4, f64>
  hlcf.return
}

// The negative-count case is rejected by the op verifier; see
// pop-ir/pop-errors.mlir.
kgen.func @variadic_too_many_fixed_args(%a: !kgen.simd<1, ui32>) {
  // expected-error @below {{'numFixedArgs' must not exceed the number of call operands: expected at most 1, found 2}}
  // expected-error @below {{failed to legalize}}
  pop.external_call @bar(%a) attributes {numFixedArgs = 2 : index}
    : (!kgen.simd<1, ui32>) -> ()
  hlcf.return
}


// `pop.compiler.global_load`/`store` are compile-time only and have no
// lowering pattern at all, so nothing rewrites them; the pass lists them as
// necessarily-converted and the leftover must still be reported.
kgen.func @compiler_global_load() -> !kgen.scalar<ui32> {
  // expected-error @below {{failed to legalize operation 'pop.compiler.global_load' that was explicitly marked illegal}}
  %0 = pop.compiler.global_load "foo" : !kgen.scalar<ui32>
  hlcf.return %0 : !kgen.scalar<ui32>
}

kgen.func @compiler_global_store(%v: !kgen.scalar<ui32>) {
  // expected-error @below {{failed to legalize operation 'pop.compiler.global_store' that was explicitly marked illegal}}
  pop.compiler.global_store "foo", %v : !kgen.scalar<ui32>
  hlcf.return
}

}
