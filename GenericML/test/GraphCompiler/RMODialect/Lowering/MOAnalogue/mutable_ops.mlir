// RUN: mo-opt --split-input-file --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

!tt  = !mo.tensor<[d2, d3], f32>
!bt  = !mo.buffer<[d0, d1], f32>
!c   = !mo.chain

// CHECK-LABEL: mo.graph
// CHECK-SAME: @mutable_load_ok<d0, d1, d2, d3>([[VARG0:[^ ]*]]: !mo.buffer<[d0, d1], f32>) -> !mo.tensor<[d2, d3], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.chain.create([[VARG0]]) : !mo.buffer<[d0, d1], f32>
// CHECK-NEXT: [[VOUTCHAIN:[^ ]*]], [[VOUTTENSOR:[^ ]*]] = mo.mutable.load([[V0]], [[VARG0]]) : (!mo.chain, !mo.buffer<[d0, d1], f32>) -> (!mo.chain, !mo.tensor<[d0, d1], f32>)
// CHECK-NEXT: rmo.rebind_tensor_shape([[VOUTTENSOR]])
// CHECK-SAME: (!mo.tensor<[d0, d1], f32>) -> !mo.tensor<[d2, d3], f32>
mo.graph @mutable_load_ok<d0, d1, d2, d3>(%buf: !bt) -> !tt
{
  %ch0 = mo.chain.create(%buf) : !bt

  %ch1, %vbuf = rmo.mo.mutable.load(%ch0, %buf) : (!c, !bt) -> (!c, !tt)

  mo.output %vbuf : !tt
}
 
// -----

!tt  = !mo.tensor<[d2, d3], f32>
!bt  = !mo.buffer<[d0, d1], f32>
!c   = !mo.chain

// CHECK-LABEL: mo.graph
// CHECK-SAME: @mutable_store_ok<d0, d1, d2, d3>([[VARG0:[^ ]*]]: !mo.buffer<[d0, d1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[d2, d3], f32>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.chain.create([[VARG0]]) : !mo.buffer<[d0, d1], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG1]])
// CHECK-SAME: (!mo.tensor<[d2, d3], f32>) -> !mo.tensor<[d0, d1], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.mutable.store([[V0]], [[VARG0]], [[V1]]) : (!mo.chain, !mo.buffer<[d0, d1], f32>, !mo.tensor<[d0, d1], f32>) -> !mo.chain
// CHECK-NEXT: mo.output
// CHECK-NEXT: }
mo.graph @mutable_store_ok<d0, d1, d2, d3>(%buf: !bt, %tensor: !tt)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %ch1 = rmo.mo.mutable.store(%ch0, %buf, %tensor) : (!c, !bt, !tt) -> (!c)

  mo.output
}
