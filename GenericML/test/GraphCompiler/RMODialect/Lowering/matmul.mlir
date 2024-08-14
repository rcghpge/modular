// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators --shape-inference --verify-diagnostics %s | FileCheck %s --enable-var-scope

mo.graph @matmul_static(%arg0: !mo.tensor<[1, 4], si64>, %arg1: !mo.tensor<[4, 1], si64>) -> !mo.tensor<[1, 1], si64> {
    %1 = rmo.matmul(%arg0, %arg1) : (!mo.tensor<[1, 4], si64>, !mo.tensor<[4, 1], si64>) -> !mo.tensor<[1, 1], si64>
    mo.output %1: !mo.tensor<[1, 1], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @matmul_static([[VARG0:[^ ]*]]: !mo.tensor<[1, 4], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[4, 1], si64>) -> !mo.tensor<[1, 1], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.matmul([[VARG0]], [[VARG1]]) : (!mo.tensor<[1, 4], si64>, !mo.tensor<[4, 1], si64>) -> !mo.tensor<[1, 1], si64>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[1, 1], si64>

// -----

mo.graph @matmul_static_broadcast(%arg0: !mo.tensor<[10, 4, 1], si64>, %arg1: !mo.tensor<[1, 8], si64>) -> !mo.tensor<[10, 4, 8], si64> {
    %1 = rmo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 4, 1], si64>, !mo.tensor<[1, 8], si64>) -> !mo.tensor<[10, 4, 8], si64>
    mo.output %1: !mo.tensor<[10, 4, 8], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @matmul_static_broadcast([[VARG0:[^ ]*]]: !mo.tensor<[10, 4, 1], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 8], si64>) -> !mo.tensor<[10, 4, 8], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.static.reshape([[VARG1]]) : !mo.tensor<[1, 8], si64> -> !mo.tensor<[1, 1, 8], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[1, 1, 8], si64> -> !mo.tensor<[10, 1, 8], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.batch_matmul([[VARG0]], [[V1]]) : (!mo.tensor<[10, 4, 1], si64>, !mo.tensor<[10, 1, 8], si64>) -> !mo.tensor<[10, 4, 8], si64>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[10, 4, 8], si64>

// -----

mo.graph @matmul_complex<D1, D2>(%arg0: !mo.tensor<[10, D1, 7, D2], si64>, %arg1: !mo.tensor<[D1, D2, 100], si64>) -> !mo.tensor<[10, D1, 7, 100], si64> {
    %1 = rmo.matmul(%arg0, %arg1) : (!mo.tensor<[10, D1, 7, D2], si64>, !mo.tensor<[D1, D2, 100], si64>) -> !mo.tensor<[10, D1, 7, 100], si64>
    mo.output %1: !mo.tensor<[10, D1, 7, 100], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @matmul_complex<[[D1:[^ ]*]], [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[10, [[D1]], 7, [[D2]]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], 100], si64>) -> !mo.tensor<[10, [[D1]], 7, 100], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.static.reshape([[VARG1]]) : !mo.tensor<[[[D1]], [[D2]], 100], si64> -> !mo.tensor<[1, [[D1]], [[D2]], 100], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[1, [[D1]], [[D2]], 100], si64> -> !mo.tensor<[10, [[D1]], [[D2]], 100], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.batch_matmul([[VARG0]], [[V1]]) : (!mo.tensor<[10, [[D1]], 7, [[D2]]], si64>, !mo.tensor<[10, [[D1]], [[D2]], 100], si64>) -> !mo.tensor<[10, [[D1]], 7, 100], si64>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[10, [[D1]], 7, 100], si64>

// -----

mo.graph @matmul_dtype_cast_int_float(%arg0: !mo.tensor<[1, 4], si8>, %arg1: !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[1, 1], bf16> {
    %1 = rmo.matmul(%arg0, %arg1) : (!mo.tensor<[1, 4], si8>, !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[1, 1], bf16>
    mo.output %1: !mo.tensor<[1, 1], bf16>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @matmul_dtype_cast_int_float([[VARG0:[^ ]*]]: !mo.tensor<[1, 4], si8>, [[VARG1:[^ ]*]]: !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[1, 1], bf16> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.cast([[VARG0]]) : (!mo.tensor<[1, 4], si8>) -> !mo.tensor<[1, 4], bf16>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.matmul([[V0]], [[VARG1]]) : (!mo.tensor<[1, 4], bf16>, !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[1, 1], bf16>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[1, 1], bf16>


// -----

mo.graph @matmul_dtype_cast_unsigned_signed(%arg0: !mo.tensor<[1, 4], si16>, %arg1: !mo.tensor<[4, 1], ui8>) -> !mo.tensor<[1, 1], si16> {
    %1 = rmo.matmul(%arg0, %arg1) : (!mo.tensor<[1, 4], si16>, !mo.tensor<[4, 1], ui8>) -> !mo.tensor<[1, 1], si16>
    mo.output %1: !mo.tensor<[1, 1], si16>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @matmul_dtype_cast_unsigned_signed([[VARG0:[^ ]*]]: !mo.tensor<[1, 4], si16>, [[VARG1:[^ ]*]]: !mo.tensor<[4, 1], ui8>) -> !mo.tensor<[1, 1], si16> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.cast([[VARG1]]) : (!mo.tensor<[4, 1], ui8>) -> !mo.tensor<[4, 1], si16>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.matmul([[VARG0]], [[V0]]) : (!mo.tensor<[1, 4], si16>, !mo.tensor<[4, 1], si16>) -> !mo.tensor<[1, 1], si16>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[1, 1], si16>

// -----

mo.graph @matmul_dtype_cast_bool_other(%arg0: !mo.tensor<[1, 4], bool>, %arg1: !mo.tensor<[4, 1], ui8>) -> !mo.tensor<[1, 1], ui8> {
    %1 = rmo.matmul(%arg0, %arg1) : (!mo.tensor<[1, 4], bool>, !mo.tensor<[4, 1], ui8>) -> !mo.tensor<[1, 1], ui8>
    mo.output %1: !mo.tensor<[1, 1], ui8>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @matmul_dtype_cast_bool_other([[VARG0:[^ ]*]]: !mo.tensor<[1, 4], bool>, [[VARG1:[^ ]*]]: !mo.tensor<[4, 1], ui8>) -> !mo.tensor<[1, 1], ui8> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.cast([[VARG0]]) : (!mo.tensor<[1, 4], bool>) -> !mo.tensor<[1, 4], ui8>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.matmul([[V0]], [[VARG1]]) : (!mo.tensor<[1, 4], ui8>, !mo.tensor<[4, 1], ui8>) -> !mo.tensor<[1, 1], ui8>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[1, 1], ui8>

// -----

mo.graph @matmul_dtype_cast_float_float(%arg0: !mo.tensor<[1, 4], f64>, %arg1: !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[1, 1], f64> {
    %1 = rmo.matmul(%arg0, %arg1) : (!mo.tensor<[1, 4], f64>, !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[1, 1], f64>
    mo.output %1: !mo.tensor<[1, 1], f64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @matmul_dtype_cast_float_float([[VARG0:[^ ]*]]: !mo.tensor<[1, 4], f64>, [[VARG1:[^ ]*]]: !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[1, 1], f64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.cast([[VARG1]]) : (!mo.tensor<[4, 1], bf16>) -> !mo.tensor<[4, 1], f64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.matmul([[VARG0]], [[V0]]) : (!mo.tensor<[1, 4], f64>, !mo.tensor<[4, 1], f64>) -> !mo.tensor<[1, 1], f64>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[1, 1], f64>

// -----

mo.graph @matmul_dtype_cast_with_reshape(%arg0: !mo.tensor<[7, 1, 4], f64>, %arg1: !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[7, 1, 1], f64> {
    %1 = rmo.matmul(%arg0, %arg1) : (!mo.tensor<[7, 1, 4], f64>, !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[7, 1, 1], f64>
    mo.output %1: !mo.tensor<[7, 1, 1], f64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @matmul_dtype_cast_with_reshape([[VARG0:[^ ]*]]: !mo.tensor<[7, 1, 4], f64>, [[VARG1:[^ ]*]]: !mo.tensor<[4, 1], bf16>) -> !mo.tensor<[7, 1, 1], f64> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.cast([[VARG1]]) : (!mo.tensor<[4, 1], bf16>) -> !mo.tensor<[4, 1], f64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.reshape([[V0]]) : !mo.tensor<[4, 1], f64> -> !mo.tensor<[1, 4, 1], f64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.static.broadcast_to([[V1]]) : !mo.tensor<[1, 4, 1], f64> -> !mo.tensor<[7, 4, 1], f64>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.batch_matmul([[VARG0]], [[V2]]) : (!mo.tensor<[7, 1, 4], f64>, !mo.tensor<[7, 4, 1], f64>) -> !mo.tensor<[7, 1, 1], f64>
// CHECK-NEXT: mo.output [[V3]] : !mo.tensor<[7, 1, 1], f64>
