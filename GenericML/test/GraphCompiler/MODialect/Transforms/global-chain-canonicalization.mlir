// RUN: mo-opt --canonicalize %s | FileCheck %s

!tt = !mo.tensor<[3, 3], f32>
!bt = !mo.buffer<[3, 3], f32>
!ct = !mo.chain

// CHECK-LABEL: mo.graph @one_long_sequence
// CHECK-SAME: [[buf0:%.*]]:{{.*}}[[buf1:%.*]]: 
// CHECK: [[ch0:%.*]] = mo.chain.create
// CHECK: [[ch1:%.*]], {{%.*}} = mo.mutable.load([[ch0]],
// CHECK: mo.mutable.store([[ch1]],
// CHECK: [[ch2:%.*]], {{%.*}} = mo.mutable.load([[ch0]],
// CHECK: mo.mutable.store([[ch2]],
mo.graph @one_long_sequence(%buf0: !bt, %buf1: !bt, %rhs: !tt) -> (!bt, !bt)
{
  %ch0 = mo.chain.create(%buf0, %buf1) : !bt, !bt

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
  %ch0 = mo.chain.create(%buf0, %buf1) : !bt, !bt

  %ch1, %vbuf = mo.mutable.load(%ch0, %buf0) : (!ct, !bt) -> (!ct, !tt)
  %ch2, %vbuf2 = mo.mutable.load(%ch1, %buf1) : (!ct, !bt) -> (!ct, !tt)

  %res1 = mo.add(%lhs, %vbuf) : !tt
  %res2 = mo.negative(%vbuf2) : !tt

  %ch3 = mo.mutable.store(%ch2, %buf0, %res1) : (!ct, !bt, !tt) -> (!ct)
  %ch4 = mo.mutable.store(%ch3, %buf1, %res2) : (!ct, !bt, !tt) -> (!ct)

  mo.output %buf0, %buf1 : !bt, !bt
}
