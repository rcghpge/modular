// RUN: mo-opt -split-input-file -verify-diagnostics %s


// mogg.call.build_graph must reference a kernel.decl op.
func.func @build_graph_missing_decl(
    %in: !mogg.tensor_,
    %b: !mgp.device_graph.builder) {
  // expected-error@+1 {{'missing.kernel' does not refer to a kernel.decl op}}
  mogg.call.build_graph["missing.kernel"]
      inputs(%in : !mogg.tensor_)
      builder(%b) {moOpName = "mo.matmul"}
  return
}
