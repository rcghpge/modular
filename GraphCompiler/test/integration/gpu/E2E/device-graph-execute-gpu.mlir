// RUN: mo-opt --gc-pipeline="compiled-framework-label=mof constant-fold=false default-device-label=gpu synthesize-device-graphs=true kernel-library-paths=%kernel_api_test_lib" %s -o %t
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=full %t | FileCheck %s

// End-to-end device-graph synthesis + replay. The `device_graph_fill` subgraph
// is an `isDeviceGraph` graph whose two custom ops record real compiled kernels
// (via `DeviceGraphBuilder.add_function`): a fill (all 7s) chained into an
// increment (+1). Synthesis lowers the graph to a `mgp.device_graph.create`
// (build the graph once, allocating the intermediate buffer between the two ops
// with `mgp.device_graph.alloc`) + `mgp.device_graph.execute` (replay it). The
// device graph is all-or-nothing for a graph, so the entry graph `main` calls it
// and transfers the GPU result to the host before returning -- a device graph
// cannot itself contain the host transfer. Replaying the recorded fill +
// increment must produce all 8s.

// CHECK-LABEL: --- Running 'main':
// CHECK: 'main' returned tensor<2x3xui8> [8, 8, 8, 8, 8, 8]
mo.graph @main(%x: !mo.tensor<[2, 3], ui8, gpu:0>) -> !mo.tensor<[2, 3], ui8> {
  %ch0 = mo.chain.create()
  %g = mo.call @device_graph_fill(%x)
      : (!mo.tensor<[2, 3], ui8, gpu:0>) -> !mo.tensor<[2, 3], ui8, gpu:0>
  %h, %ch1 = mo.transfer[%ch0] %g
      : !mo.tensor<[2, 3], ui8, gpu:0> to <"cpu", 0>
  mo.output %h : !mo.tensor<[2, 3], ui8>
}

mo.graph @device_graph_fill(%x: !mo.tensor<[2, 3], ui8, gpu:0>) -> !mo.tensor<[2, 3], ui8, gpu:0>
    attributes {isSubgraph, isDeviceGraph} {
  %0 = mo.custom {symbol = "device_graph_fill", device = #M.device_ref<"gpu", 0>}
      (%x) : (!mo.tensor<[2, 3], ui8, gpu:0>) -> !mo.tensor<[2, 3], ui8, gpu:0>
  %1 = mo.custom {symbol = "device_graph_increment", device = #M.device_ref<"gpu", 0>}
      (%0) : (!mo.tensor<[2, 3], ui8, gpu:0>) -> !mo.tensor<[2, 3], ui8, gpu:0>
  mo.output %1 : !mo.tensor<[2, 3], ui8, gpu:0>
}
