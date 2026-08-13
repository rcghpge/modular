// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels | FileCheck %s

// A `mo.mutable.store` consumer of a fused-output kernel looks like any other
// epilogue candidate -- single-use producer result, one-result consumer -- but
// `lowerMutableStoreToForeach` gives it a `!mo.chain` result and a body with no
// output block argument, the buffer input being the store target. MOGG's
// `FuseEpilogues` fuses exactly this, retargeting the store to that buffer;
// `EpilogueFuser` declines it, because `map.iter.opaque` carries neither an out
// chain nor memory-effect fields to fuse it soundly.
//
// TODO(GEX-3964): reach MOGG parity, at which point this becomes a positive
// test -- the store folds into the kernel's epilogue and writes the buffer
// directly, and the `map.iter.foreach` below disappears.

// CHECK-LABEL: mo.graph @mutable_store_consumer_not_fused
// The kernel keeps its own tensor output and its plain store epilogue.
// CHECK:       map.iter.opaque
// CHECK:         outputs (%{{.*}} : !mo.tensor<[4], f32>)
// CHECK:       map.call.opaque @{{.*}}MapLowerOneFusedOutput
// The store stays a separate kernel, writing the buffer itself.
// CHECK:       map.iter.foreach
// CHECK:       map.call.elementwise @{{.*}}MutableStore
mo.graph @mutable_store_consumer_not_fused(
  %buf: !mo.buffer<[4], f32>,
  %a: !mo.tensor<[4], f32>
) {
  %chain = mo.chain.create()
  %c = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "map_lower_one_fused_output"}(%a) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %out = mo.mutable.store(%buf, %c, %chain) : (!mo.buffer<[4], f32>, !mo.tensor<[4], f32>, !mo.chain) -> !mo.chain
  mo.output
}
