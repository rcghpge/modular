// RUN: mo-opt --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-mogg %s | FileCheck %s

!tt = !mo.tensor<[2, 2], f32>
!bt = !mo.buffer<[2, 2], f32>
!ct = !mo.chain

// CHECK-LABEL: mo.graph @supports_scalars
// CHECK-NEXT: mo.chain.create
// CHECK-NEXT: mogg.kernel
// CHECK-NEXT: mogg.call.execute["{{.*}}.MutableTestOp"]
// CHECK: mo.output
mo.graph @supports_scalars(%b: !bt) -> !bt {
  %c = mo.chain.create()

  %outch = mo.custom {symbol = "mutable_test_op", device = #M.device_ref<"cpu", 0>}(%b, %c) : (!bt, !ct) -> (!ct)

  mo.output %b : !bt
}
