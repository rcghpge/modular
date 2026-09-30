// RUN: mo-opt -split-input-file -verify-diagnostics %s


// mogg.call.build_graph must reference a kernel.decl op.
func.func @build_graph_missing_decl(
    %in: !mo.tensor<[2, 4], f32, gpu:0>,
    %b: !mgp.device_graph.builder) {
  // expected-error@+1 {{'missing.kernel' does not refer to a kernel.decl op}}
  mogg.call.build_graph["missing.kernel"]
      inputs(%in : !mo.tensor<[2, 4], f32, gpu:0>)
      builder(%b) {moOpName = "mo.matmul"}
  return
}
