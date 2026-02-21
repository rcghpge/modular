// RUN: mo-opt %s --mo-to-mgp="constant-fold=false" -o /dev/null --mlir-print-ir-after=mogg-fuse-prologues-epilogues 2>&1 | FileCheck %s

// The kernel summary string should reflect an accurate fusion record.
// DAG because the order things are printed in the middle of the pipeline is
// not predictable.
// CHECK-LABEL: mo.graph @simple_merge_dim_rank2_merge_0
// CHECK-DAG: fusionInfo = Prologue(Affine(mo.matmul:{{[0-9]+}}, mo.merge_dim:{{[0-9]+}}), mo.add:{{[0-9]+}}, operandIndices=[0])
mo.graph @simple_merge_dim_rank2_merge_0(%arg0: !mo.tensor<[4, 10], f32>, %arg1: !mo.tensor<[10, 8], f32>) -> !mo.tensor<[32], f32> {
    %bias = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
       14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.,
       27., 28., 29., 30., 31., 32.> : tensor<32xf32>} : () -> !mo.tensor<[32], f32>
    %res = mo.matmul(%arg0, %arg1): (!mo.tensor<[4, 10], f32>, !mo.tensor<[10, 8], f32>) -> !mo.tensor<[4, 8], f32>
    %out = mo.merge_dim[0](%res): (!mo.tensor<[4, 8], f32>) -> !mo.tensor<[32], f32>
    %final = mo.add(%out, %bias): !mo.tensor<[32], f32>
    mo.output %final: !mo.tensor<[32], f32>
}
