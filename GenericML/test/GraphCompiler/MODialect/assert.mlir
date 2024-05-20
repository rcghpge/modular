// RUN: mo-opt --canonicalize %s | FileCheck --enable-var-scope %s

// CHECK: kgen.func @assert(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> !mo.chain {
kgen.func @assert(%arg0: index, %arg1: index) -> !mo.chain {
  // CHECK: %[[VAL0:.*]] = index.cmp eq(%[[ARG0]], %[[ARG1]])
  %0 = index.cmp eq(%arg0, %arg1)
  // CHECK: mo.assert %[[VAL0]], "args are not equal"
  %ch0 = mo.assert %0, "args are not equal"
  kgen.return %ch0 : !mo.chain
}
