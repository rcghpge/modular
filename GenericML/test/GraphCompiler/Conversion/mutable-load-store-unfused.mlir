// RUN: mo-opt --mo-to-mogg --mo-to-primitives %s | FileCheck %s

!tt = !mo.tensor<[2, 2], f32>
!bt = !mo.buffer<[2, 2], f32>
!ct = !mo.chain

// CHECK-LABEL: mgp.model @unfused_mutable_load_store
// CHECK: execute
// CHECK-SAME: !mgp.opaque, [[buf:%.*]]: !mgp.tensor<2x2xf32>, [[rhs:%.*]]: !mgp.tensor<2x2xf32>)
// CHECK: grt.chain.create ([[buf]]
// CHECK: [[ebuf:%.*]] = mgp.tensor.extract.buffer{{.*}} [[buf]] : !mgp.tensor<2x2xf32>
// The check below corresponds to the (lowered) mo.mutable.load
// CHECK: mgp.buffer.device_to_device{{.*}}[[ebuf]] 
// CHECK: mogg.generic.compiled.region
// CHECK: summary = "mo.add
// CHECK: mogg.generic.compiled.region
// CHECK: summary = "mo.negative
// CHECK: [[ebuf2:%.*]] = mgp.tensor.extract.buffer{{.*}} [[buf]] : !mgp.tensor<2x2xf32>
// CHECK: mgp.buffer.device_to_device{{.*}}[[ebuf2]]
mo.graph @unfused_mutable_load_store(%buf: !bt, %rhs: !tt) -> !bt
{
  %ch0 = mo.chain.create(%buf) : !bt

  %ch1, %vbuf = mo.mutable.load(%ch0, %buf) : (!ct, !bt) -> (!ct, !tt)
  %res1 = mo.add(%vbuf, %rhs) : !tt
  %res2 = mo.negative(%res1) : !tt
  %ch4 = mo.mutable.store(%ch1, %buf, %res2) : (!ct, !bt, !tt) -> (!ct)

  %cf = mo.chain.create(%ch4) : !ct
  %gb = mo.guard[%cf](%buf) : !bt

  mo.output %gb : !bt
}
