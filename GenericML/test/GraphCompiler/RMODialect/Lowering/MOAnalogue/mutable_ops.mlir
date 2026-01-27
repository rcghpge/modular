// RUN: mo-opt --split-input-file --mo-resolve-unknown-parameters --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

!tt  = !mo.tensor<[d2, d3], f32>
!bt  = !mo.buffer<[d0, d1], f32>
!c   = !mo.chain

// CHECK-LABEL: mo.graph
// CHECK-SAME: @mutable_load_ok<d0, d1, d2, d3>([[VARG0:[^ ]*]]: !mo.buffer<[d0, d1], f32>) -> (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d2, d3], f32>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.chain.create([[VARG0]]) : !mo.buffer<[d0, d1], f32>
// CHECK-NEXT: [[VOUTTENSOR:[^ ]*]], [[VOUTCHAIN:[^ ]*]] = mo.mutable.load([[VARG0]], [[V0]]) : (!mo.buffer<[d0, d1], f32>, !mo.chain) -> (!mo.tensor<[d0, d1], f32>, !mo.chain)
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind([[VOUTTENSOR]]) : !mo.tensor<[d0, d1], f32> -> !mo.tensor<[d2, d3], f32>
// CHECK-NEXT: mo.output [[VARG0]], [[V1]] : !mo.buffer<[d0, d1], f32>, !mo.tensor<[d2, d3], f32>
mo.graph @mutable_load_ok<d0, d1, d2, d3>(%buf: !bt) -> (!bt, !tt)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = rmo.mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)

  mo.output %buf, %vbuf : !bt, !tt
}

// -----

!tt  = !mo.tensor<[d2, d3], f32>
!bt  = !mo.buffer<[d0, d1], f32>
!c   = !mo.chain

// CHECK-LABEL: mo.graph
// CHECK-SAME: @mutable_store_ok<d0, d1, d2, d3>([[VARG0:[^ ]*]]: !mo.buffer<[d0, d1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[d2, d3], f32>) -> (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d2, d3], f32>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.chain.create([[VARG0]]) : !mo.buffer<[d0, d1], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG1]]) {message ={{.*}}} : (!mo.tensor<[d2, d3], f32>) -> !mo.tensor<[d0, d1], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.mutable.store([[VARG0]], [[V1]], [[V0]]) : (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d0, d1], f32>, !mo.chain) -> !mo.chain
// CHECK-NEXT: mo.output [[VARG0]], [[VARG1]] : !mo.buffer<[d0, d1], f32>, !mo.tensor<[d2, d3], f32>
mo.graph @mutable_store_ok<d0, d1, d2, d3>(%buf: !bt, %tensor: !tt) -> (!bt, !tt)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %ch1 = rmo.mo.mutable.store(%buf, %tensor, %ch0) : (!bt, !tt, !c) -> (!c)

  mo.output %buf, %tensor : !bt, !tt
}

// -----

!tt  = !mo.tensor<[?, ?], f32>
!bt  = !mo.buffer<[?, ?], f32>
!ct   = !mo.chain

// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_all_unk<[[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]], [[D12:[^ ]*]] -> [[D13:[^ ]*]], [[D14:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.buffer<[[[D9]], [[D10]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D11]], [[D12]]], f32>) -> (!mo.buffer<[[[D9]], [[D10]]], f32>, !mo.tensor<[[[D13]], [[D14]]], f32>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.chain.create([[VARG0]]) : !mo.buffer<[[[D9]], [[D10]]], f32>
// CHECK-NEXT: [[VOUTTENSOR:[^ ]*]], [[VOUTCHAIN:[^ ]*]] = mo.mutable.load([[VARG0]], [[V0]]) : (!mo.buffer<[[[D9]], [[D10]]], f32>, !mo.chain) -> (!mo.tensor<[[[D9]], [[D10]]], f32>, !mo.chain)
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D3:[^ ]*]], [[D4:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D11]], [[D12]]], f32> -> !mo.tensor<[[[D3]], [[D4]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]], [[D6:[^ ]*]]>([[VOUTTENSOR]]) : !mo.tensor<[[[D9]], [[D10]]], f32> -> !mo.tensor<[[[D5]], [[D6]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape([[V2]]) {message ={{.*}}} : (!mo.tensor<[[[D5]], [[D6]]], f32>) -> !mo.tensor<[[[D3]], [[D4]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.add([[V1]], [[V3]]) : !mo.tensor<[[[D3]], [[D4]]], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D7:[^ ]*]], [[D8:[^ ]*]]>([[V4]]) {message ={{.*}}} : (!mo.tensor<[[[D3]], [[D4]]], f32>) -> !mo.tensor<[[[D7]], [[D8]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]]>([[V5]]) : !mo.tensor<[[[D7]], [[D8]]], f32> -> !mo.tensor<[[[D1]], [[D2]]], f32>
// CHECK-NEXT: [[V7:[^ ]*]] = rmo.rebind_tensor_shape([[V6]]) {message ={{.*}}} : (!mo.tensor<[[[D1]], [[D2]]], f32>) -> !mo.tensor<[[[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V8:[^ ]*]] = mo.mutable.store([[VARG0]], [[V7]], [[VOUTCHAIN]]) : (!mo.buffer<[[[D9]], [[D10]]], f32>, !mo.tensor<[[[D9]], [[D10]]], f32>, !mo.chain) -> !mo.chain
// CHECK-NEXT: mo.output<[[D7]], [[D8]]> [[VARG0]], [[V5]] : !mo.buffer<[[[D9]], [[D10]]], f32>, !mo.tensor<[[[D7]], [[D8]]], f32>
mo.graph @add_all_unk(%buf: !bt, %lhs: !tt) -> (!bt, !tt)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = rmo.mo.mutable.load(%buf, %ch0) : (!bt, !ct) -> (!tt, !ct)
  %res1 = rmo.mo.add(%lhs, %vbuf) : (!tt, !tt) -> !tt
  %ch2 = rmo.mo.mutable.store(%buf, %res1, %ch1) : (!bt, !tt, !ct) -> (!ct)

  mo.output %buf, %res1 : !bt, !tt
}

// -----

!tt  = !mo.tensor<[d0, d1], f32>
!bt  = !mo.buffer<[?, ?], f32>
!ct   = !mo.chain

// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_buf_unk<d0, d1, [[D1:[^ ]*]], [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.buffer<[[[D1]], [[D2]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[d0, d1], f32>) -> (!mo.buffer<[[[D1]], [[D2]]], f32>, !mo.tensor<[d0, d1], f32>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.chain.create([[VARG0]]) : !mo.buffer<[[[D1]], [[D2]]], f32>
// CHECK-NEXT: [[VOUTTENSOR:[^ ]*]], [[VOUTCHAIN:[^ ]*]] = mo.mutable.load([[VARG0]], [[V0]]) : (!mo.buffer<[[[D1]], [[D2]]], f32>, !mo.chain) -> (!mo.tensor<[[[D1]], [[D2]]], f32>, !mo.chain)
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind([[VOUTTENSOR]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[d0, d1], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.add([[VARG1]], [[V1]]) : !mo.tensor<[d0, d1], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape([[V2]]) {message ={{.*}}} : (!mo.tensor<[d0, d1], f32>) -> !mo.tensor<[[[D1]], [[D2]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.mutable.store([[VARG0]], [[V3]], [[VOUTCHAIN]]) : (!mo.buffer<[[[D1]], [[D2]]], f32>, !mo.tensor<[[[D1]], [[D2]]], f32>, !mo.chain) -> !mo.chain
// CHECK-NEXT: mo.output [[VARG0]], [[V2]] : !mo.buffer<[[[D1]], [[D2]]], f32>, !mo.tensor<[d0, d1], f32>
mo.graph @add_buf_unk<d0, d1>(%buf: !bt, %lhs: !tt) -> (!bt, !tt)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = rmo.mo.mutable.load(%buf, %ch0) : (!bt, !ct) -> (!tt, !ct)
  %res1 = rmo.mo.add(%lhs, %vbuf) : (!tt, !tt) -> !tt
  %ch2 = rmo.mo.mutable.store(%buf, %res1, %ch1) : (!bt, !tt, !ct) -> (!ct)

  mo.output %buf, %res1 : !bt, !tt
}

// -----

!tt  = !mo.tensor<[?, ?], f32>
!bt  = !mo.buffer<[d0, d1], f32>
!ct   = !mo.chain

// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_tensor_unk<d0, d1, [[D9:[^ ]*]], [[D10:[^ ]*]] -> [[D11:[^ ]*]], [[D12:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.buffer<[d0, d1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D9]], [[D10]]], f32>) -> (!mo.buffer<[d0, d1], f32>, !mo.tensor<[[[D11]], [[D12]]], f32>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.chain.create([[VARG0]]) : !mo.buffer<[d0, d1], f32>
// CHECK-NEXT: [[VOUTTENSOR:[^ ]*]], [[VOUTCHAIN:[^ ]*]] = mo.mutable.load([[VARG0]], [[V0]]) : (!mo.buffer<[d0, d1], f32>, !mo.chain) -> (!mo.tensor<[d0, d1], f32>, !mo.chain)
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D3:[^ ]*]], [[D4:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D9]], [[D10]]], f32> -> !mo.tensor<[[[D3]], [[D4]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]], [[D6:[^ ]*]]>([[VOUTTENSOR]]) : !mo.tensor<[d0, d1], f32> -> !mo.tensor<[[[D5]], [[D6]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape([[V2]]) {message ={{.*}}} : (!mo.tensor<[[[D5]], [[D6]]], f32>) -> !mo.tensor<[[[D3]], [[D4]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.add([[V1]], [[V3]]) : !mo.tensor<[[[D3]], [[D4]]], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D7:[^ ]*]], [[D8:[^ ]*]]>([[V4]]) {message ={{.*}}} : (!mo.tensor<[[[D3]], [[D4]]], f32>) -> !mo.tensor<[[[D7]], [[D8]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]]>([[V5]]) : !mo.tensor<[[[D7]], [[D8]]], f32> -> !mo.tensor<[[[D1]], [[D2]]], f32>
// CHECK-NEXT: [[V7:[^ ]*]] = rmo.rebind_tensor_shape([[V6]]) {message ={{.*}}} : (!mo.tensor<[[[D1]], [[D2]]], f32>) -> !mo.tensor<[d0, d1], f32>
// CHECK-NEXT: [[V8:[^ ]*]] = mo.mutable.store([[VARG0]], [[V7]], [[VOUTCHAIN]]) : (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d0, d1], f32>, !mo.chain) -> !mo.chain
// CHECK-NEXT: mo.output<[[D7]], [[D8]]> [[VARG0]], [[V5]] : !mo.buffer<[d0, d1], f32>, !mo.tensor<[[[D7]], [[D8]]], f32>
mo.graph @add_tensor_unk<d0, d1>(%buf: !bt, %lhs: !tt) -> (!bt, !tt)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = rmo.mo.mutable.load(%buf, %ch0) : (!bt, !ct) -> (!tt, !ct)
  %res1 = rmo.mo.add(%lhs, %vbuf) : (!tt, !tt) -> !tt
  %ch2 = rmo.mo.mutable.store(%buf, %res1, %ch1) : (!bt, !tt, !ct) -> (!ct)

  mo.output %buf, %res1 : !bt, !tt
}
