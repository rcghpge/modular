// RUN: mo-opt --canonicalize-chain-sequences %s | FileCheck %s

!tt = !mo.tensor<[3, 3], f32>
!bt = !mo.buffer<[3, 3], f32>
!ct = !mo.chain
!ot = !mo.opaque<"T">

// CHECK-LABEL: mo.graph @one_long_sequence
// CHECK-SAME: [[buf0:%.*]]:{{.*}}[[buf1:%.*]]:
// CHECK: [[ch0:%.*]] = mo.chain.create
// CHECK: [[ch1:%.*]], {{%.*}} = mo.mutable.load([[ch0]],
// CHECK: mo.mutable.store([[ch1]],
// CHECK: [[ch2:%.*]], {{%.*}} = mo.mutable.load([[ch0]],
// CHECK: mo.mutable.store([[ch2]],
mo.graph @one_long_sequence(%buf0: !bt, %buf1: !bt, %rhs: !tt) -> (!bt, !bt)
{
  %ch0 = mo.chain.create()

  %ch1, %vbuf = mo.mutable.load(%ch0, %buf0) : (!ct, !bt) -> (!ct, !tt)
  %res1 = mo.add(%rhs, %vbuf) : !tt
  %ch2 = mo.mutable.store(%ch1, %buf0, %res1) : (!ct, !bt, !tt) -> (!ct)

  %ch3, %vbuf2 = mo.mutable.load(%ch2, %buf1) : (!ct, !bt) -> (!ct, !tt)
  %res2 = mo.negative(%vbuf2) : !tt
  %ch4 = mo.mutable.store(%ch3, %buf1, %res2) : (!ct, !bt, !tt) -> (!ct)

  mo.output %buf0, %buf1 : !bt, !bt
}

// CHECK-LABEL: mo.graph @interleaved_sequence
// CHECK-SAME: [[buf0:%.*]]:{{.*}}[[buf1:%.*]]:
// CHECK-DAG: [[ch0:%.*]] = mo.chain.create
// CHECK-DAG: [[ch1:%.*]], {{%.*}} = mo.mutable.load([[ch0]],
// CHECK-DAG: mo.mutable.store([[ch1]],
// CHECK-DAG: [[ch2:%.*]], {{%.*}} = mo.mutable.load([[ch0]],
// CHECK-DAG: mo.mutable.store([[ch2]],
mo.graph @interleaved_sequence(%buf0: !bt, %buf1: !bt, %lhs: !tt) -> (!bt, !bt)
{
  %ch0 = mo.chain.create()

  %ch1, %vbuf = mo.mutable.load(%ch0, %buf0) : (!ct, !bt) -> (!ct, !tt)
  %ch2, %vbuf2 = mo.mutable.load(%ch1, %buf1) : (!ct, !bt) -> (!ct, !tt)

  %res1 = mo.add(%lhs, %vbuf) : !tt
  %res2 = mo.negative(%vbuf2) : !tt

  %ch3 = mo.mutable.store(%ch2, %buf0, %res1) : (!ct, !bt, !tt) -> (!ct)
  %ch4 = mo.mutable.store(%ch3, %buf1, %res2) : (!ct, !bt, !tt) -> (!ct)

  mo.output %buf0, %buf1 : !bt, !bt
}

// CHECK-LABEL: mo.graph @custom_ops
// CHECK-SAME: [[buf0:%.*]]:{{.*}}[[buf1:%.*]]:
// CHECK-DAG: [[ch0:%.*]] = mo.chain.create
// CHECK-DAG: [[ch1:%.*]] = mo.custom {symbol = "foo"}([[ch0]], [[buf0]])
// CHECK-DAG: [[ch2:%.*]] = mo.custom {symbol = "foo"}([[ch0]], [[buf1]])
mo.graph @custom_ops(%buf0: !bt, %buf1: !bt)
{
  %ch0 = mo.chain.create()

  %ch1 = mo.custom {symbol = "foo"}(%ch0, %buf0) : (!ct, !bt) -> (!ct)
  %ch2 = mo.custom {symbol = "foo"}(%ch1, %buf1) : (!ct, !bt) -> (!ct)

  mo.output
}

// CHECK-LABEL: mo.graph @custom_ops_opaque
// CHECK-SAME: [[BUF:%.*]]:{{.*}}[[OPAQUE:%.*]]:
mo.graph @custom_ops_opaque(%buf: !bt, %opaque: !ot)
{
  // CHECK: [[CH0:%.*]] = mo.chain.create()
  %ch0 = mo.chain.create()

  // Pair of custom ops whose linear chains should get decomposed.
  // CHECK-DAG: [[CH1:%.*]] = mo.custom{{.*}}foo{{.*}}([[CH0]], [[BUF]])
  // CHECK-DAG: [[CH2:%.*]] = mo.custom{{.*}}bar{{.*}}([[CH0]], [[OPAQUE]])
  %ch1 = mo.custom {symbol = "foo"} (%ch0, %buf) : (!ct, !bt) -> (!ct)
  %ch2 = mo.custom {symbol = "bar"} (%ch1, %opaque) : (!ct, !ot) -> (!ct)

  // Check that this true linear dependency on %opaque remains.
  // CHECK-DAG: [[CH3:%.*]] = mo.custom{{.*}}baz{{.*}}([[CH2]], [[OPAQUE]])
  %ch3 = mo.custom {symbol = "baz"} (%ch2, %opaque) : (!ct, !ot) -> (!ct)

  mo.output
}

// Checks that chain re-writing occurs when there's a shared buffer value
// between values with no write effect from either op.
//
// CHECK-LABEL: mo.graph @shared_buffer_no_shared_write_effect
// CHECK-SAME: [[ARG0:%.*]]: !mo.buffer<[2, 2], f32>, [[ARG1:%.*]]: !mo.buffer<[2, 2], f32>, [[ARG2:%.*]]: !mo.buffer<[2, 2], f32>
mo.graph @shared_buffer_no_shared_write_effect(%arg0: !mo.buffer<[2, 2], f32>,
                                               %arg1: !mo.buffer<[2, 2], f32>,
                                               %arg2: !mo.buffer<[2, 2], f32>)
{
  // CHECK: [[CH0:%.*]] = mo.chain.create()
  %ch0 = mo.chain.create()

  // CHECK-DAG: [[CH1:%.*]] = mo.custom{{.*}}([[CH0]], [[ARG0]], [[ARG1]])
  %ch1 = mo.custom  {symbol = "foo", mutSpec = #mo<muts[0:R_, 1:_W]>} (%ch0, %arg0, %arg1) :
   (!ct, !mo.buffer<[2, 2], f32>, !mo.buffer<[2, 2], f32>) -> !ct

  // CHECK-DAG: [[CH2:%.*]] = mo.custom{{.*}}([[CH0]], [[ARG2]], [[ARG0]])
   %ch2 = mo.custom {symbol = "bar", mutSpec = #mo<muts[0:_W, 1:R_]>} (%ch1, %arg2, %arg0) :
   (!ct, !mo.buffer<[2, 2], f32>, !mo.buffer<[2, 2], f32>) -> !ct

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

  // CHECK-DAG: [[CH1:%.*]] = mo.custom{{.*}}bar{{.*}}([[CH0]], [[OPAQUE1]])
  // CHECK-DAG: [[CH2:%.*]] = mo.custom{{.*}}baz{{.*}}([[CH1]], [[OPAQUE2]])
  %ch1 = mo.custom {symbol = "bar"} (%ch0, %opaque1) : (!ct, !ot) -> (!ct)
  %ch2 = mo.custom {symbol = "baz"} (%ch1, %opaque2) : (!ct, !ot) -> (!ct)

  mo.output
}

// CHECK-LABEL: @everything
// CHECK-SAME: ([[arg0:%.*]]: !mo.buffer<[3, 3], f32>, [[arg1:%.*]]: !mo.buffer<[3, 3], f32>, [[arg2:%.*]]: !mo.tensor<[3, 3], f32>) {
// CHECK-DAG:  [[a0:%.*]] = mo.chain.create()
// CHECK-DAG:  [[a1:%.*]] = mo.custom {symbol = "foo"}([[a0]], [[arg0]]) : (!mo.chain, !mo.buffer<[3, 3], f32>) -> !mo.chain
// CHECK-DAG:  [[outChain_0:%.*]], [[outTensor_1:%.*]] = mo.mutable.load([[a1]], [[arg0]]) : (!mo.chain, !mo.buffer<[3, 3], f32>) -> (!mo.chain, !mo.tensor<[3, 3], f32>)
// CHECK-DAG:  [[a3:%.*]] = mo.add([[outTensor_1]], [[arg2]]) : !mo.tensor<[3, 3], f32>
// CHECK-DAG:  [[a5:%.*]] = mo.mutable.store(%outChain_0, [[arg0]], [[a3]]) : (!mo.chain, !mo.buffer<[3, 3], f32>, !mo.tensor<[3, 3], f32>) -> !mo.chain
// CHECK-DAG:  [[outChain:%.*]], [[outTensor:%.*]] = mo.mutable.load([[a0]], [[arg1]]) : (!mo.chain, !mo.buffer<[3, 3], f32>) -> (!mo.chain, !mo.tensor<[3, 3], f32>)
// CHECK-DAG:  [[a2:%.*]] = mo.add([[outTensor]], [[arg2]]) : !mo.tensor<[3, 3], f32>
// CHECK-DAG:  [[a4:%.*]] = mo.mutable.store([[outChain]], [[arg1]], [[a2]]) : (!mo.chain, !mo.buffer<[3, 3], f32>, !mo.tensor<[3, 3], f32>) -> !mo.chain
// CHECK-DAG:  [[a6:%.*]] = mo.custom {symbol = "foo"}([[a4]], [[arg1]]) : (!mo.chain, !mo.buffer<[3, 3], f32>) -> !mo.chain
mo.graph @everything(%buf0: !bt, %buf1: !bt, %tensor: !tt)
{
  %ch0 = mo.chain.create()

  %ch1 = mo.custom {symbol = "foo"}(%ch0, %buf0) : (!ct, !bt) -> (!ct)

  %ch2, %vbuf = mo.mutable.load(%ch1, %buf1) : (!ct, !bt) -> (!ct, !tt)
  %ch3, %vbuf2 = mo.mutable.load(%ch2, %buf0) : (!ct, !bt) -> (!ct, !tt)

  %res1 = mo.add(%vbuf, %tensor) : !tt
  %res2 = mo.add(%vbuf2, %tensor) : !tt

  %ch4 = mo.mutable.store(%ch3, %buf1, %res1) : (!ct, !bt, !tt) -> (!ct)
  %ch5 = mo.mutable.store(%ch4, %buf0, %res2) : (!ct, !bt, !tt) -> (!ct)

  %ch6 = mo.custom {symbol = "foo"}(%ch5, %buf1) : (!ct, !bt) -> (!ct)

  mo.output
}
