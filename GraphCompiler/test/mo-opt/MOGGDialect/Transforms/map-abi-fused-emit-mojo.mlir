// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t MAX_GC_USE_ADV_FUSION=1 mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg --mogg-package-variadics-and-scalars \
// RUN:   --mogg-outline-kernels-into-functions --mogg-to-mgp --mogg-emit-mojo \
// RUN:   -verify-each=false
// RUN: cat %t/*.mojo | FileCheck %s
// RUN: cat %t/*.mojo | FileCheck %s --check-prefix=NOPARAM

// End-to-end Mojo emission for MAP-ABI fused (store-epilogue) outputs
// (TensorCreateEpilogueOp::emitMojo plus the CallOpaqueOp fused branch). A
// `map.iter.epilogue` store lowers directly to a `mogg._tensor.create.epilogue`
// holding its own epilogue body (no separate lambda op); emission builds an
// explicit-capture `OutputStoreFunctor` conforming to `OutputStoreFusion` by
// recursively lowering the epilogue body's own ops (its `mogg._tensor.store`
// becomes `__call__`'s store statement), wraps it in
// `_MoggOutputStoreFusionTensor(functor, layout)`, and forces the kernel's
// `has_epilogue_fusion` parameter True on the DPS execute call. The whole path
// is `@parameter`-free (NOPARAM below).
//
// The pipeline runs through `mogg-to-mgp` (so the region consumes the outlined
// stub), then emits; `mogg-emit-mojo` still erases the stub, so
// `-verify-each=false` tolerates the dangling `mogg.call.stub`. `mogg-to-mgp`
// names the `ir-output-dir` dump after the graphs (not `graph.mojo`), so the
// checks read `%t/*.mojo`.

// One fused output: the store target (capture 0) is captured as a bare
// pointer + row-major layout, the functor is wrapped, and the wrapped tensor is
// the DPS output of an execute call carrying `has_epilogue_fusion = True`.
// CHECK: var [[F0:.*]] = OutputStoreFunctor{{[0-9_]*}}[{{.*}}]({{.*}}._ptr, {{.*}})
// CHECK: var [[W0:.*]] = _MoggOutputStoreFusionTensor[{{.*}}, type_of([[F0]]), {{.*}}]([[F0]], {{.*}})
// CHECK: {{.*}}.MapLowerOneFusedOutput.execute[target = "cpu", has_epilogue_fusion = True]([[W0]], {{.*}})

mo.graph @one_fused(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Two fused outputs: one functor + wrapper per output, both wrapped tensors
// passed to a single fused execute call.
// CHECK-DAG: {{.*}}.MapLowerTwoFusedOutputs.execute[target = "cpu", has_epilogue_fusion = True]
// CHECK-DAG: {{.*}}.MapLowerPartialFusedOutputs.execute[target = "cpu", has_epilogue_fusion = True]

mo.graph @two_fused(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c:2 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_two_fused_outputs"}(%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %c#0, %c#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// Partial: res0 fused (store epilogue), res1 plain output. The plain output is a
// bare `mogg_tensor_init`; only res0 is wrapped.
mo.graph @partial_fused(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c:2 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_partial_fused_outputs"}(%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %c#0, %c#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// Dynamic dim: the `!mogg.dim_` handle arrives as a trailing `Int` argument
// (mirroring the plain-stub handling in map-abi-stub-emit-mojo.mlir) and
// feeds a runtime `Scalar[DType.int]` Coord component instead of an
// all-compile-time `layout.Idx` Coord; the fused-output functor/wrapper/
// execute shape is otherwise identical to the static case above.
// CHECK: def {{[A-Za-z0-9_]+_map(_[0-9]+)?}}({{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], [[DIM:.*]]: Int, {{.*}}: {{.*}}DeviceContext) raises:
// CHECK:   var {{.*}} = layout.Coord(Scalar[DType.int]([[DIM]]), layout.Idx[2])
// CHECK:   var [[FD:.*]] = OutputStoreFunctor{{[0-9_]*}}[{{.*}}]({{.*}}._ptr, {{.*}})
// CHECK:   var [[WD:.*]] = _MoggOutputStoreFusionTensor[{{.*}}, type_of([[FD]]), {{.*}}]([[FD]], {{.*}})
// CHECK:   {{.*}}.MapLowerOneFusedOutput.execute[target = "cpu", has_epilogue_fusion = True]([[WD]], {{.*}})

mo.graph @one_fused_dynamic(%a: !mo.tensor<[batch, 2], f32>) -> !mo.tensor<[batch, 2], f32> {
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%a) : (!mo.tensor<[batch, 2], f32>) -> !mo.tensor<[batch, 2], f32>
  mo.output %c : !mo.tensor<[batch, 2], f32>
}

// The functor struct is now built via the scoped `structDecl` overload, with
// the `__call__` body produced by recursively lowering the epilogue's own
// body ops (`TensorCreateEpilogueOp::emitMojo` / `TensorStoreOp::emitMojo`)
// instead of substituted into a fixed Mojo template: per-capture (dtype,
// LayoutType, mut) params, bare (ptr, layout) fields, and an `@always_inline
// __call__` that stores through a rebuilt `TileTensor`. NO `@parameter`, NO
// `ManagedTensorSlice` field.
// CHECK:      struct OutputStoreFunctor{{[0-9_]*}}[[[DTYPE:[a-zA-Z0-9_]+]]: DType, [[LAYOUT:[a-zA-Z0-9_]+]]: layout.TensorLayout, [[MUT:[a-zA-Z0-9_]+]]: Bool](OutputStoreFusion, TrivialRegisterPassable):
// CHECK-NEXT:     var [[PTRF:[a-zA-Z0-9_]+]]: UnsafePointer[Scalar[Self.[[DTYPE]]], UntrackedOrigin[mut = Self.[[MUT]]]]
// CHECK-NEXT:     var [[LAYOUTF:[a-zA-Z0-9_]+]]: Self.[[LAYOUT]]
// CHECK:          @always_inline
// CHECK-NEXT:     def __call__[dtype : DType, rank : Int, width : Int, element_alignment : Int = 1](self, idx: layout.Coord, value: SIMD[dtype, width]):
// CHECK-NEXT:         var [[TILE:[a-zA-Z0-9_]+]] = layout.TileTensor[dtype = Self.[[DTYPE]], LayoutType = Self.[[LAYOUT]], origin = UntrackedOrigin[mut = Self.[[MUT]]]](ptr = self.[[PTRF]], layout = self.[[LAYOUTF]])
// CHECK-NEXT:         [[TILE]].store[width = width, alignment = element_alignment](idx, rebind[SIMD[type_of([[TILE]]).dtype, width]](value))

// The advanced-fusion emit path must be entirely `@parameter`-free.
// NOPARAM-NOT: @parameter
