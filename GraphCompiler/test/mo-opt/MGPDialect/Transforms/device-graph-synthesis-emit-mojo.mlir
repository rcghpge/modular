// RUN: mo-opt %s \
// RUN:     --gc-pipeline-pre-jit="kernel-library-paths=%kernel_api_test_lib synthesize-device-graphs=true dump-mojo-text=true" \
// RUN:     -o /dev/null 2>&1 | FileCheck %s

// `dump-mojo-text=true` runs the DumpMojo pass, which writes the emitted Mojo to
// stderr; FileCheck matches against it (no temp dir needed).

// End-to-end device-graph synthesis + EmitMojo. An `isDeviceGraph` mo.graph
// with a single custom op lowers (through the synthesis stages) to a
// `mgp.device_graph.create` whose build body records the kernel via
// `DeviceGraphBuilder.region`, invoking the kernel's node-free `build_graph`
// method. `imposter_build_graph_capable` (`BuildGraphCapable` in the test kernel
// library) exposes both `execute` and a node-free `build_graph`.

// The build closure passed to `DeviceGraph.create` is a runtime closure that
// reads its captures, so it carries the `read` capture convention.
// CHECK: def [[BUILD:build[_0-9]+]](mut builder: {{.*}}DeviceGraphBuilder) raises {read}:

// The recording allocates the output buffer inside a `region` work closure. The
// allocated buffer is a body result of the `mgp.device_graph.dependencies` op;
// since `region` returns only the join node, the buffer is surfaced through a
// captured `Optional` (None-initialized) that the `work` closure writes (so it
// captures by `mut`), then unwrapped with `.take()` after `region`.
// CHECK: var [[OUT:var[_0-9]+]] = Optional[{{.*}}](None)
// CHECK: def [[WORK0:work[_0-9]+]](mut builder: {{.*}}DeviceGraphBuilder) raises {mut}:
// The allocation is a builder op: `mgp.device_graph.alloc` lowers to the
// `mgp_device_graph_alloc` primitive taking the builder directly -- no
// `builder.context()` extraction and no `mgp_buffer_alloc` in the work closure.
// CHECK-NOT: builder.context()
// CHECK-NOT: mgp_buffer_alloc
// CHECK: {{.*}}mgp_device_graph_alloc({{.*}}, builder)
// CHECK: [[OUT]] =
// CHECK: var [[NODE0:var[_0-9]+]] = builder.region([[WORK0]], dependencies = [])
// CHECK: {{var[_0-9]+}} = [[OUT]].take()

// The dispatch is recorded in its own `read`-capturing `region`, ordered after
// the alloc's node via the `dependencies=` list, and calls the node-free stub.
// CHECK: def [[WORK1:work[_0-9]+]](mut builder: {{.*}}DeviceGraphBuilder) raises {read}:
// CHECK: stub_0_build_graph({{.*}}, {{.*}}, builder)
// CHECK: builder.region([[WORK1]], dependencies = {{\[}}[[NODE0]]{{\]}})

// The single graph output is registered on the builder: the yielded output
// tensor is packed into a fresh `AsyncValue` via `async_pack` and handed to
// `add_output`.
// CHECK: builder.add_output({{var[_0-9]+}}.async_pack())

// The graph is built via the static `DeviceGraph.create(deviceContext, build)`.
// CHECK: {{.*}}DeviceGraph.create({{.*}}, [[BUILD]])

// The recording stub is node-free (no result) and calls the kernel's
// `build_graph` method with the builder as the launch argument.
// CHECK: def stub_0_build_graph({{.*}}: {{.*}}InputTensor[{{.*}}], {{.*}}: {{.*}}OutputTensor[{{.*}}], {{.*}}: {{.*}}DeviceGraphBuilder) raises:
// CHECK: BuildGraphCapable.build_graph(

mo.graph @dg(%x: !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0>
    attributes {isDeviceGraph} {
  %0 = mo.custom {symbol = "imposter_build_graph_capable", device = #M.device_ref<"gpu", 0>}
      (%x) : (!mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0>
  mo.output %0 : !mo.tensor<[2, 3], f32, gpu:0>
}
