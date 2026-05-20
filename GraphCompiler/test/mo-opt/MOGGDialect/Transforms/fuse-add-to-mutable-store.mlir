// RUN: mo-opt --mogg-import-packages --mo-to-mogg --mogg-fuse-elementwise %s | FileCheck %s

// CHECK-LABEL: mo.graph @fuse_store_to_add
// CHECK: [[CHAIN:%.*]] = mo.chain.create()
// CHECK: mogg.kernel
// CHECK: fusionInfo = Elementwise("mo.add":{{[0-9]+}}, "mo.mutable.store":{{[0-9]+}})
mo.graph @fuse_store_to_add(%buf: !mo.buffer<[2, 2], f32>, %a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>) {

  %chain = mo.chain.create()

  %res = mo.add(%a, %b) : !mo.tensor<[2, 2], f32>
  %out_chain = mo.mutable.store(%buf, %res, %chain) : (!mo.buffer<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.chain) -> !mo.chain

  mo.output
}
