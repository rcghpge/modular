// RUN: mo-opt --canonicalize-chain-sequences %s | FileCheck %s

!tt = !mo.tensor<[3, 3], f32>
!bt = !mo.buffer<[3, 3], f32>
!ct = !mo.chain
!ot = !mo.opaque<"T">

// CHECK-LABEL: mo.graph @one_long_sequence
// CHECK-SAME: [[buf0:%.*]]:{{.*}}[[buf1:%.*]]:
// CHECK: [[ch0:%.*]] = mo.chain.create
// CHECK: {{%.*}}, [[ch1:%.*]] = mo.mutable.load({{.*}}, [[ch0]]
// CHECK: mo.mutable.store({{.*}}, [[ch1]]
// CHECK: {{%.*}}, [[ch2:%.*]] = mo.mutable.load({{.*}}, [[ch0]]
// CHECK: mo.mutable.store({{.*}}, [[ch2]]
mo.graph @one_long_sequence(%buf0: !bt, %buf1: !bt, %rhs: !tt) -> (!bt, !bt)
{
  %ch0 = mo.chain.create()

  %vbuf, %ch1 = mo.mutable.load(%buf0, %ch0) : (!bt, !ct) -> (!tt, !ct)
  %res1 = mo.add(%rhs, %vbuf) : !tt
  %ch2 = mo.mutable.store(%buf0, %res1, %ch1) : (!bt, !tt, !ct) -> (!ct)

  %vbuf2, %ch3 = mo.mutable.load(%buf1, %ch2) : (!bt, !ct) -> (!tt, !ct)
  %res2 = mo.negative(%vbuf2) : !tt
  %ch4 = mo.mutable.store(%buf1, %res2, %ch3) : (!bt, !tt, !ct) -> (!ct)

  mo.output %buf0, %buf1 : !bt, !bt
}

// CHECK-LABEL: mo.graph @interleaved_sequence
// CHECK-SAME: [[buf0:%.*]]:{{.*}}[[buf1:%.*]]:
// CHECK-DAG: [[ch0:%.*]] = mo.chain.create
// CHECK-DAG: {{%.*}}, [[ch1:%.*]] = mo.mutable.load({{%.*}}, [[ch0]]
// CHECK-DAG: mo.mutable.store({{.*}}, [[ch1]]
// CHECK-DAG: {{%.*}}, [[ch2:%.*]] = mo.mutable.load({{%.*}}, [[ch0]]
// CHECK-DAG: mo.mutable.store({{.*}}, [[ch2]]
mo.graph @interleaved_sequence(%buf0: !bt, %buf1: !bt, %lhs: !tt) -> (!bt, !bt)
{
  %ch0 = mo.chain.create()

  %vbuf, %ch1 = mo.mutable.load(%buf0, %ch0) : (!bt, !ct) -> (!tt, !ct)
  %vbuf2, %ch2 = mo.mutable.load(%buf1, %ch1) : (!bt, !ct) -> (!tt, !ct)

  %res1 = mo.add(%lhs, %vbuf) : !tt
  %res2 = mo.negative(%vbuf2) : !tt

  %ch3 = mo.mutable.store(%buf0, %res1, %ch2) : (!bt, !tt, !ct) -> (!ct)
  %ch4 = mo.mutable.store(%buf1, %res2, %ch3) : (!bt, !tt, !ct) -> (!ct)

  mo.output %buf0, %buf1 : !bt, !bt
}

// CHECK-LABEL: mo.graph @custom_ops
// CHECK-SAME: [[buf0:%.*]]:{{.*}}[[buf1:%.*]]:
// CHECK-DAG: [[ch0:%.*]] = mo.chain.create
// CHECK-DAG: [[ch1:%.*]] = mo.custom {symbol = "foo"}([[buf0]], [[ch0]])
// CHECK-DAG: [[ch2:%.*]] = mo.custom {symbol = "foo"}([[buf1]], [[ch0]])
mo.graph @custom_ops(%buf0: !bt, %buf1: !bt)
{
  %ch0 = mo.chain.create()

  %ch1 = mo.custom {symbol = "foo"}(%buf0, %ch0) : (!bt, !ct) -> (!ct)
  %ch2 = mo.custom {symbol = "foo"}(%buf1, %ch1) : (!bt, !ct) -> (!ct)

  mo.output
}

// CHECK-LABEL: mo.graph @custom_ops_opaque
// CHECK-SAME: [[BUF:%.*]]:{{.*}}[[OPAQUE:%.*]]:
mo.graph @custom_ops_opaque(%buf: !bt, %opaque: !ot)
{
  // CHECK: [[CH0:%.*]] = mo.chain.create()
  %ch0 = mo.chain.create()

  // Pair of custom ops whose linear chains should get decomposed.
  // CHECK-DAG: [[CH1:%.*]] = mo.custom{{.*}}foo{{.*}}([[BUF]], [[CH0]])
  // CHECK-DAG: [[CH2:%.*]] = mo.custom{{.*}}bar{{.*}}([[OPAQUE]], [[CH0]])
  %ch1 = mo.custom {symbol = "foo"}(%buf, %ch0) : (!bt, !ct) -> (!ct)
  %ch2 = mo.custom {symbol = "bar"} (%opaque, %ch1) : (!ot, !ct) -> (!ct)

  // Check that this true linear dependency on %opaque remains.
  // CHECK-DAG: [[CH3:%.*]] = mo.custom{{.*}}baz{{.*}}([[OPAQUE]], [[CH2]])
  %ch3 = mo.custom {symbol = "baz"} (%opaque, %ch2) : (!ot, !ct) -> (!ct)

  mo.output
}

// CHECK-LABEL: mo.graph @different_opaque_vals_same_type
// CHECK-SAME: [[OPAQUE1:%.*]]: !mo.opaque<"T">, [[OPAQUE2:%.*]]: !mo.opaque<"T">
mo.graph @different_opaque_vals_same_type(%opaque1: !ot, %opaque2: !ot)
{
  // CHECK: [[CH0:%.*]] = mo.chain.create()
  %ch0 = mo.chain.create()

  // Check that we don't move baz past bar even though since they operate on an
  // opaque value of the same type (and we do have the potential of aliasing).

  // CHECK-DAG: [[CH1:%.*]] = mo.custom{{.*}}bar{{.*}}([[OPAQUE1]], [[CH0]])
  // CHECK-DAG: [[CH2:%.*]] = mo.custom{{.*}}baz{{.*}}([[OPAQUE2]], [[CH1]])
  %ch1 = mo.custom {symbol = "bar"} (%opaque1, %ch0) : (!ot, !ct) -> (!ct)
  %ch2 = mo.custom {symbol = "baz"} (%opaque2, %ch1) : (!ot, !ct) -> (!ct)

  mo.output
}

// CHECK-LABEL: @everything
// CHECK-SAME: ([[arg0:%.*]]: !mo.buffer<[3, 3], f32>, [[arg1:%.*]]: !mo.buffer<[3, 3], f32>, [[arg2:%.*]]: !mo.tensor<[3, 3], f32>) {
// CHECK-DAG:  [[a0:%.*]] = mo.chain.create()
// CHECK-DAG:  [[a1:%.*]] = mo.custom {symbol = "foo"}([[arg0]], [[a0]]) : (!mo.buffer<[3, 3], f32>, !mo.chain) -> !mo.chain
// CHECK-DAG:  [[outTensor_1:%.*]], [[outChain_0:%.*]] = mo.mutable.load([[arg0]], [[a1]]) : (!mo.buffer<[3, 3], f32>, !mo.chain) -> (!mo.tensor<[3, 3], f32>, !mo.chain)
// CHECK-DAG:  [[a3:%.*]] = mo.add([[outTensor_1]], [[arg2]]) : !mo.tensor<[3, 3], f32>
// CHECK-DAG:  [[a5:%.*]] = mo.mutable.store([[arg0]], [[a3]], [[outChain_0]]) : (!mo.buffer<[3, 3], f32>, !mo.tensor<[3, 3], f32>, !mo.chain) -> !mo.chain
// CHECK-DAG:  [[outTensor:%.*]], [[outChain:%.*]] = mo.mutable.load([[arg1]], [[a0]]) : (!mo.buffer<[3, 3], f32>, !mo.chain) -> (!mo.tensor<[3, 3], f32>, !mo.chain)
// CHECK-DAG:  [[a2:%.*]] = mo.add([[outTensor]], [[arg2]]) : !mo.tensor<[3, 3], f32>
// CHECK-DAG:  [[a4:%.*]] = mo.mutable.store([[arg1]], [[a2]], [[outChain]]) : (!mo.buffer<[3, 3], f32>, !mo.tensor<[3, 3], f32>, !mo.chain) -> !mo.chain
// CHECK-DAG:  [[a6:%.*]] = mo.custom {symbol = "foo"}([[arg1]], [[a4]]) : (!mo.buffer<[3, 3], f32>, !mo.chain) -> !mo.chain
mo.graph @everything(%buf0: !bt, %buf1: !bt, %tensor: !tt)
{
  %ch0 = mo.chain.create()

  %ch1 = mo.custom {symbol = "foo"}(%buf0, %ch0) : (!bt, !ct) -> (!ct)

  %vbuf, %ch2 = mo.mutable.load(%buf1, %ch1) : (!bt, !ct) -> (!tt, !ct)
  %vbuf2, %ch3 = mo.mutable.load(%buf0, %ch2) : (!bt, !ct) -> (!tt, !ct)

  %res1 = mo.add(%vbuf, %tensor) : !tt
  %res2 = mo.add(%vbuf2, %tensor) : !tt

  %ch4 = mo.mutable.store(%buf1, %res1, %ch3) : (!bt, !tt, !ct) -> (!ct)
  %ch5 = mo.mutable.store(%buf0, %res2, %ch4) : (!bt, !tt, !ct) -> (!ct)

  %ch6 = mo.custom {symbol = "foo"}(%buf1, %ch5) : (!bt, !ct) -> (!ct)

  mo.output
}
