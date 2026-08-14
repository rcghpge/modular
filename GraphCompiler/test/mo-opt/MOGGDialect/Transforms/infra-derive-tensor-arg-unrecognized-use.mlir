// RUN: not mo-opt %s --mogg-to-mgp --allow-unregistered-dialect 2>&1 | FileCheck %s

// A kernel-body use the contract derivation does not recognize is a hard
// diagnostic, never a silent legacy fallback (which would recreate the
// silent-mislabel failure mode of the positional side-table this derivation
// replaced, GEX-4071). The synthetic decl keeps `mogg.call.execute`'s symbol
// verification satisfied so lowering reaches the deriver. (The deriver's
// missing-declaration diagnostic is defense-in-depth only: the execute op's
// own symbol verifier already rejects a module without the decl at parse.)

#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>

mogg.kernel.decl @k registration = "test.k" device_info = <"", "", "", ""> execute = #kf struct_params = [] is_user_defined = false

// CHECK: error: cannot derive a tensor-arg contract through this use of a kernel I/O value ('test.unknown' is not a recognized kernel-body op)
mo.graph @unrecognized_use(%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %0 = mogg.kernel(%a as %arg0: !mo.tensor<[4], f32>, %b as %arg1: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %1 = mogg.output_placeholders : !mo.tensor<[4], f32>
    "test.unknown"(%arg1) : (!mo.tensor<[4], f32>) -> ()
    %2 = mogg.device_context_placeholder {device = #M.device_ref<"cpu", 0>} : !mogg.context
    mogg.call.execute["k"] inputs(%arg0, %2 : !mo.tensor<[4], f32>, !mogg.context) outputs(%1 : !mo.tensor<[4], f32>) {isThrows, kernel_param = {}, moOpName = "test.k"}
  } fusionInfo = "test.k":1 {allocIndices = [], device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false, isAsync = true, outputParamDecls = #kgen<param.decls[]> : !kgen.non_struct_type}
  mo.output %0 : !mo.tensor<[4], f32>
}
