// RUN: mo-opt --constant-fold --mlir-print-ir-before=cse %s 2>&1 | FileCheck %s

// CHECK: constant_fold_subgraphs
// CHECK-NEXT: mo.graph @large_const_subgraph{{.*}}(%arg0: !mo.tensor<[10], ui32>)
// CHECK-NEXT: mo.add(%arg0, %arg0)
// CHECK: mo.constant {value = #M.dense_array<2, 2, 2, 2, 2, 2, 2, 2, 2, 2> : !M.array<10xui32>}
mo.graph @large_const() -> !mo.tensor<[10], ui32> {
  %1 = mo.constant {value = dense_resource<large_const> : !M.array<10xui32>} : !mo.tensor<[10], ui32>
  %2 = mo.add(%1, %1) : !mo.tensor<[10], ui32>
  mo.output %2 : !mo.tensor<[10], ui32>
}

{-#
  dialect_resources: {
    builtin:{
      large_const: "0x0400000001000000010000000100000001000000010000000100000001000000010000000100000001000000"
    }
  }
#-}
