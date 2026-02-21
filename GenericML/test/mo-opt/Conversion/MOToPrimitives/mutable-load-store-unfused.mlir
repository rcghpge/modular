// RUN: mo-opt --mo-to-mogg --mgp-mo-to-primitives %s | FileCheck %s

!tt = !mo.tensor<[2, 2], f32>
!bt = !mo.buffer<[2, 2], f32>
!ct = !mo.chain

// CHECK-LABEL: mgp.model @unfused_mutable_load_store
// CHECK: execute
// CHECK-SAME: !mgp.opaque{{.*}}, [[buf:%.*]]: !mgp.tensor<2x2xf32>, [[rhs:%.*]]: !mgp.tensor<2x2xf32>)
// CHECK: grt.chain.create ([[buf]]
// CHECK: [[ebuf:%.*]] = mgp.tensor.extract.buffer{{.*}} [[buf]]
// The check below corresponds to the (lowered) mo.mutable.load
// CHECK: mgp.buffer.device_to_device{{.*}}[[ebuf]]
// CHECK: mogg.generic.compiled.region
// CHECK: mogg.generic.compiled.region
// CHECK: mogg.generic.compiled.region
mo.graph @unfused_mutable_load_store(%buf: !bt, %rhs: !tt) -> !bt
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = mo.mutable.load(%buf, %ch0) : (!bt, !ct) -> (!tt, !ct)
  %res1 = mo.add(%vbuf, %rhs) : !tt
  %res2 = mo.negative(%res1) : !tt
  %ch4 = mo.mutable.store(%buf, %res2, %ch1) : (!bt, !tt, !ct) -> (!ct)

  mo.output %buf : !bt
}
