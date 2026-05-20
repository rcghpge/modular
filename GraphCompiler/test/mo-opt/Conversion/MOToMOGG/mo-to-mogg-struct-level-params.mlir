// RUN: mo-opt --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-mogg --mogg-outline-kernels --mogg-emit-mojo %s 2>&1 | FileCheck %s

// Regression test for kernels that combine a struct-level compile-time
// parameter with a fn-level parameter that's referenced by an argument type.
// Models `mo.rope_split_store.ragged.paged` (struct `[interleaved: Bool]`,
// fn `[dtype, freq_dtype, //, target]`, args typed `OutputTensor[dtype=dtype,
// ...]`). Two distinct bugs broke this on real models:
//
//   1. `validateCustomOpParameters` rejected `mo.custom`'s `flip = true`
//      because validation only walked fn-level params, missing struct-level
//      params.
//   2. EmitMojo's `inferByInputs` collection mapped depth-0
//      `ParamIndexRefAttr`s into `fnLevelParams` alone, but the IndexRef
//      indexes into `structLevelParams + fnLevelParams`. The off-by-N lookup
//      misreported which fn param was inferable, and `dtype` (which Mojo
//      should deduce from arg types) was rejected as un-inferable.
//
// The test passes the kernel both an explicit struct-level param (`flip`)
// and relies on Mojo's deduction of the fn-level `dtype` from the operand's
// tensor type. Reaching `--mogg-emit-mojo` without errors covers both paths.

// CHECK-LABEL: @struct_level_param_call
mo.graph @struct_level_param_call(
    %x: !mo.tensor<[4], f32>
) -> !mo.tensor<[4], f32> {
  %y = mo.custom {
    symbol = "imposter_add_struct_param",
    device = #M.device_ref<"cpu", 0>,
    parameters = {
      flip = true
    }
  }(%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %y : !mo.tensor<[4], f32>
}

// CHECK: imposter_add_struct_param
