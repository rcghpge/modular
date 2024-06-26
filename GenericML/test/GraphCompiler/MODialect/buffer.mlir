// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck --enable-var-scope %s

// CHECK-LABEL: mo.graph @buffer_type
// CHECK-SAME:   {{.*}}: !mo.buffer<[10, 20], f32>,
// CHECK-SAME:   {{.*}}: !mo.buffer<[D0, D1], f32>,
// CHECK-SAME:   {{.*}}: !mo.buffer<Sh0, f32>)
mo.graph @buffer_type<D0, D1, Sh0: !mosh.ape>(
  %b0: !mo.buffer<[10, 20], f32>,
  %b3: !mo.buffer<[D0, D1], f32>,
  %b4: !mo.buffer<Sh0, f32>
) {
  mo.output
}

// CHECK-LABEL: mo.graph @mutable_load
// CHECK-SAME: ([[VARG0:[^ ]*]]: !mo.chain, [[VARG1:[^ ]*]]: !mo.buffer<[10, 20], f32>) {
// CHECK-NEXT: [[VOUTCHAIN:[^ ]*]], [[VOUTTENSOR:[^ ]*]] = mo.mutable.load([[VARG0]], [[VARG1]]) : (!mo.chain, !mo.buffer<[10, 20], f32>) -> (!mo.chain, !mo.tensor<[10, 20], f32>)
// CHECK-NEXT: mo.output
mo.graph @mutable_load(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>) {
  %outChain, %tensor = mo.mutable.load(%ch, %buf) :
    (!mo.chain, !mo.buffer<[10, 20], f32>) ->
    (!mo.chain, !mo.tensor<[10, 20], f32>)
  mo.output
}


// CHECK-LABEL: mo.graph @mutable_store
// CHECK-SAME: ([[VARG0:[^ ]*]]: !mo.chain, [[VARG1:[^ ]*]]: !mo.buffer<[10, 20], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[10, 20], f32>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.mutable.store([[VARG0]], [[VARG1]], [[VARG2]]) : (!mo.chain, !mo.buffer<[10, 20], f32>, !mo.tensor<[10, 20], f32>) -> !mo.chain
// CHECK-NEXT: mo.output
mo.graph @mutable_store(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>, %tensor: !mo.tensor<[10, 20], f32>) {
  %outChain = mo.mutable.store(%ch, %buf, %tensor) :
    (!mo.chain, !mo.buffer<[10, 20], f32>, !mo.tensor<[10, 20], f32>) -> (!mo.chain)
  mo.output
}

// equivalent to torch.add(blhs, rhs, out=blhs)
// CHECK-LABEL: mo.graph @inplace_add
// CHECK-SAME: ([[VARG0:[^ ]*]]: !mo.chain, [[VARG1:[^ ]*]]: !mo.buffer<[10, 20], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[10, 20], f32>) {
// CHECK-NEXT: [[VOUTCHAIN:[^ ]*]], [[VOUTTENSOR:[^ ]*]] = mo.mutable.load([[VARG0]], [[VARG1]]) : (!mo.chain, !mo.buffer<[10, 20], f32>) -> (!mo.chain, !mo.tensor<[10, 20], f32>)
// CHECK-NEXT: [[V0:[^ ]*]] = mo.add([[VOUTTENSOR]], [[VARG2]]) : !mo.tensor<[10, 20], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.mutable.store([[VOUTCHAIN]], [[VARG1]], [[V0]]) : (!mo.chain, !mo.buffer<[10, 20], f32>, !mo.tensor<[10, 20], f32>) -> !mo.chain
// CHECK-NEXT: mo.output
mo.graph @inplace_add(%ch0: !mo.chain, %blhs: !mo.buffer<[10, 20], f32>, %rhs: !mo.tensor<[10, 20], f32>) {

  %ch1, %vblhs = mo.mutable.load(%ch0, %blhs) :
    (!mo.chain, !mo.buffer<[10, 20], f32>) ->
    (!mo.chain, !mo.tensor<[10, 20], f32>)

  %res = mo.add(%vblhs, %rhs) : !mo.tensor<[10, 20], f32>

  %ch2 = mo.mutable.store(%ch1, %blhs, %res) :
    (!mo.chain, !mo.buffer<[10, 20], f32>, !mo.tensor<[10, 20], f32>) -> (!mo.chain)

  mo.output
}
