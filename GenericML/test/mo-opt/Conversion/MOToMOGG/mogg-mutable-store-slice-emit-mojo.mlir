// RUN: mo-opt --mo-to-mogg %s | FileCheck --check-prefix="CHECK-KERNEL-API" %s

// COM: Check the lowering under the kernel API is correct
// CHECK-KERNEL-API-LABEL: @mutable_load
// CHECK-KERNEL-API-SAME: ([[ch:%.*]]: [[ch_type:!mo.chain]],
// CHECK-KERNEL-API-SAME: [[buf:%.*]]: [[buf_type:!mo.buffer<\[10, 20\], f32>]],
// CHECK-KERNEL-API-SAME: [[slice:%.*]]: [[slice_type:!mo.tensor<\[3, 3\], f32>]],
// CHECK-KERNEL-API-SAME: [[start:%.*]]: [[index_type:!mo.tensor<\[2\], si64>]], [[stop:%.*]]: [[index_type]], [[step:%.*]]: [[index_type]]) {
// CHECK-KERNEL-API: mogg.kernel([[buf]]: [[buf_type]], [[slice]]: [[slice_type]], [[start]]: [[index_type]], [[stop]]: [[index_type]], [[step]]: [[index_type]]) chain([[ch]]) -> [[ch_type]] {
// CHECK-KERNEL-API: mogg.call.execute
// CHECK-KERNEL-API-SAME: mo.mutable.store.slice
mo.graph @mutable_load(%ch: !mo.chain,
                       %buf: !mo.buffer<[10, 20], f32>,
                       %slice: !mo.tensor<[3, 3], f32>,
                       %start: !mo.tensor<[2], si64>,
                       %stop: !mo.tensor<[2], si64>,
                       %step: !mo.tensor<[2], si64>)
{
  %outchain = mo.mutable.store.slice(%buf, %slice, %start, %stop, %step, %ch) :
    (!mo.buffer<[10, 20], f32>, !mo.tensor<[3, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.chain) -> !mo.chain

  mo.output
}
