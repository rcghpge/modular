// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-outline-structured-kernel-bodies --map-to-mogg --mogg-legalize-rowwise-kernels | FileCheck %s

// A fused reduction, from MO to a rowwise kernel ready to print as Mojo.

// CHECK-LABEL: mo.graph @sumOfSquaresPlusBias
// CHECK:       mogg.call.rowwise {{.*}} attributes {accum = f32, dtype = bf16, is_cached = false, primary = 0 : i64} {
// CHECK-NEXT:    %[[ACC:.*]] = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
// CHECK-NEXT:      %[[P:.*]] = mogg.rowwise.primary : !mogg.simd<bf16>
// CHECK-NEXT:      %[[A:.*]] = mogg.call.elementwise @builtin_kernels.elementwise.Cast(%[[P]])
// CHECK-NEXT:      %[[SQ:.*]] = mogg.call.elementwise @builtin_kernels.elementwise.Mul(%[[A]], %[[A]])
// CHECK-NEXT:      mogg.yield %[[SQ]] : !mogg.simd<f32>
// CHECK-NEXT:    } {fields = ["acc"], params = {dtype = f32}}
// CHECK-NEXT:    %[[B:.*]] = mogg.tensor.load %cap1[%row]
// CHECK-NEXT:    %[[Y:.*]] = mogg.call.elementwise @builtin_kernels.elementwise.Add(%[[ACC]], %[[B]])
// CHECK-NEXT:    mogg.rowwise.emit {
// CHECK-NEXT:      mogg.tensor.store %cap2[%row] = %[[Y]]
// CHECK-NEXT:    }
// CHECK-NEXT:  }
mo.graph @sumOfSquaresPlusBias(%x: !mo.tensor<[8, 128], bf16>, %b: !mo.tensor<[8, 1], f32>) -> !mo.tensor<[8, 1], f32> {
  %a = mo.cast(%x) : (!mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], f32>
  %sq = mo.mul(%a, %a) : !mo.tensor<[8, 128], f32>
  %r = mo.custom [symbol = "mo.reduce.add.tmp", device = #M.device_ref<"cpu", 0>](%sq) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %y = mo.add(%r, %b) : !mo.tensor<[8, 1], f32>
  mo.output %y : !mo.tensor<[8, 1], f32>
}
