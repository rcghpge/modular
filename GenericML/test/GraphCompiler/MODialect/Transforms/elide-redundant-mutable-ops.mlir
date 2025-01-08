// RUN: mo-opt --elide-redundant-mutable-ops %s | FileCheck --enable-var-scope %s

!tt = !mo.tensor<[2, 2], f32>
!bt = !mo.buffer<[2, 2], f32>
!ct = !mo.chain

// CHECK-LABEL: mo.graph @sequenced_adds(
// CHECK-SAME: [[buf:%.*]]: !mo.buffer<[2, 2], f32>
// CHECK-SAME: [[rhs:%.*]]: !mo.tensor<[2, 2], f32>
// CHECK: [[ch1:%.*]], [[vbuf:%.*]] = mo.mutable.load
// CHECK: [[res1:%.*]] = mo.add([[vbuf]], [[rhs]])
// CHECK-NOT: mo.mutable.store
// CHECK-NOT: mo.mutable.load
// CHECK: [[res2:%.*]] = mo.add([[res1]], [[rhs]])
// CHECK: mo.mutable.store([[ch1]], [[buf]], [[res2]])
mo.graph @sequenced_adds(%buf: !bt, %rhs: !tt) -> (!bt)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %ch1, %vbuf = mo.mutable.load(%ch0, %buf) : (!ct, !bt) -> (!ct, !tt)
  %res1 = mo.add(%vbuf, %rhs) : !tt
  %ch2 = mo.mutable.store(%ch1, %buf, %res1) : (!ct, !bt, !tt) -> (!ct)

  %ch3, %vbuf2 = mo.mutable.load(%ch2, %buf) : (!ct, !bt) -> (!ct, !tt)
  %res2 = mo.add(%vbuf2, %rhs) : !tt
  %ch4 = mo.mutable.store(%ch3, %buf, %res2) : (!ct, !bt, !tt) -> (!ct)

  %cf = mo.chain.create(%ch4) : !ct

  mo.output %buf : !bt
}

// CHECK-LABEL: mo.graph @add_relu(
// CHECK-SAME: [[buf:%.*]]: !mo.buffer<[2, 2], f32>
// CHECK-SAME: [[rhs:%.*]]: !mo.tensor<[2, 2], f32>
// CHECK: [[ch1:%.*]], [[vbuf:%.*]] = mo.mutable.load
// CHECK: [[res1:%.*]] = mo.add([[vbuf]], [[rhs]])
// CHECK-NOT: mo.mutable.store
// CHECK-NOT: mo.mutable.load
// CHECK: [[res2:%.*]] = mo.relu([[res1]])
// CHECK: mo.mutable.store([[ch1]], [[buf]], [[res2]])
mo.graph @add_relu(%buf: !bt, %rhs: !tt) -> !bt
{
  %ch0 = mo.chain.create(%buf) : !bt

  %ch1, %vbuf = mo.mutable.load(%ch0, %buf) : (!ct, !bt) -> (!ct, !tt)
  %res1 = mo.add(%vbuf, %rhs) : !tt

  %res2 = mo.relu(%res1) : !tt
  %ch4 = mo.mutable.store(%ch1, %buf, %res2) : (!ct, !bt, !tt) -> (!ct)

  mo.output %buf : !bt
}

// CHECK-LABEL: mo.graph @multiple_chain_users_load(
// CHECK-SAME: [[buf:%.*]]: !mo.buffer<[2, 2], f32>
// CHECK-SAME: [[rhs:%.*]]: !mo.tensor<[2, 2], f32>
// CHECK: [[ch1:%.*]], [[vbuf:%.*]] = mo.mutable.load
// CHECK: [[res1:%.*]] = mo.add([[vbuf]], [[rhs]])
// CHECK: [[ch2:%.*]] =  mo.mutable.store([[ch1]]
mo.graph @multiple_chain_users_load(%buf: !bt, %rhs: !tt) -> (!bt, !tt, !ct)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %ch1, %vbuf = mo.mutable.load(%ch0, %buf) : (!ct, !bt) -> (!ct, !tt)
  %res1 = mo.add(%vbuf, %rhs) : !tt
  %ch2 = mo.mutable.store(%ch1, %buf, %res1) : (!ct, !bt, !tt) -> (!ct)

  %ch5 = mo.chain.create(%ch2) : !mo.chain
  %ch3, %vbuf2 = mo.mutable.load(%ch2, %buf) : (!ct, !bt) -> (!ct, !tt)
  %ch6 = mo.chain.create(%ch3) : !mo.chain

  %res2 = mo.add(%vbuf2, %rhs) : !tt

  %cf = mo.chain.create(%ch3, %ch5, %ch6) : !ct, !ct, !ct

  mo.output %buf, %res2, %cf : !bt, !tt, !ct
}

// CHECK-LABEL: mo.graph @multiple_chain_users_store(
// CHECK-SAME: [[buf:%.*]]: !mo.buffer<[2, 2], f32>
// CHECK-SAME: [[rhs:%.*]]: !mo.tensor<[2, 2], f32>
// CHECK: [[ch1:%.*]], [[vbuf:%.*]] = mo.mutable.load
// CHECK: [[res1:%.*]] = mo.add([[vbuf]], [[rhs]])
// CHECK: [[ch2:%.*]] = mo.mutable.store([[ch1]], [[buf]], [[res1]])
// CHECK: [[res2:%.*]] = mo.add([[res1]], [[rhs]])
// CHECK: mo.mutable.store([[ch2]], [[buf]], [[res2]])
mo.graph @multiple_chain_users_store(%buf: !bt, %rhs: !tt) -> (!bt, !ct)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %ch1, %vbuf = mo.mutable.load(%ch0, %buf) : (!ct, !bt) -> (!ct, !tt)
  %res1 = mo.add(%vbuf, %rhs) : !tt

  %ch5 = mo.chain.create(%ch1) : !ct
  %ch2 = mo.mutable.store(%ch1, %buf, %res1) : (!ct, !bt, !tt) -> (!ct)
  %ch6 = mo.chain.create(%ch2) : !ct

  %res2 = mo.add(%res1, %rhs) : !tt
  %ch4 = mo.mutable.store(%ch2, %buf, %res2) : (!ct, !bt, !tt) -> (!ct)

  %cf = mo.chain.create(%ch4, %ch5, %ch6) : !ct, !ct, !ct

  mo.output %buf, %cf : !bt, !ct
}
