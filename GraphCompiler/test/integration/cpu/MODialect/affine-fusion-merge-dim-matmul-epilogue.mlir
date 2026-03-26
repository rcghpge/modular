// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full --model-inputs=ones %t.mlir | FileCheck %s

// Tests that a chain of 4 affine views interleaved with elementwise epilogues
// is fully fused into a single kernel by FuseEpilogues. FuseEpilogues
// interleaves epilogue fusion with affine view fusion in a loop until
// convergence, handling arbitrary-length chains:
//   matmul -> add -> merge_dim -> add -> split_dim -> add -> merge_dim -> add -> split_dim -> add
//
// Graph: matmul[4,10]x[10,8] -> (add -> affine_view) x 4 -> add -> [4,8]
// Each add uses an all-ones bias. With model-inputs=ones:
//   matmul output: 10.0 (sum of 10 ones)
//   + 5 adds of 1.0 each = 15.0 for every output element

// CHECK-LABEL: 'matmul_add_4hop' returned tensor<4x8xf32>
// CHECK-SAME: 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01
// CHECK-SAME: 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01
// CHECK-SAME: 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01
// CHECK-SAME: 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01
mo.graph @matmul_add_4hop(%arg0: !mo.tensor<[4, 10], f32>, %arg1: !mo.tensor<[10, 8], f32>) -> !mo.tensor<[4, 8], f32> {
    %bias1 = "mo.constant"() {value = #M.dense_array<
        1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.> : tensor<4x8xf32>} : () -> !mo.tensor<[4, 8], f32>
    %bias2 = "mo.constant"() {value = #M.dense_array<
        1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.> : tensor<32xf32>} : () -> !mo.tensor<[32], f32>
    %bias3 = "mo.constant"() {value = #M.dense_array<
        1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.> : tensor<4x8xf32>} : () -> !mo.tensor<[4, 8], f32>
    %bias4 = "mo.constant"() {value = #M.dense_array<
        1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.> : tensor<32xf32>} : () -> !mo.tensor<[32], f32>
    %bias5 = "mo.constant"() {value = #M.dense_array<
        1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.> : tensor<4x8xf32>} : () -> !mo.tensor<[4, 8], f32>
    %res = mo.matmul(%arg0, %arg1): (!mo.tensor<[4, 10], f32>, !mo.tensor<[10, 8], f32>) -> !mo.tensor<[4, 8], f32>
    %add1 = mo.add(%res, %bias1): !mo.tensor<[4, 8], f32>
    %merge1 = mo.merge_dim[0](%add1): (!mo.tensor<[4, 8], f32>) -> !mo.tensor<[32], f32>
    %add2 = mo.add(%merge1, %bias2): !mo.tensor<[32], f32>
    %split1 = mo.split_dim[0](%add2): (!mo.tensor<[32], f32>) -> !mo.tensor<[4, 8], f32>
    %add3 = mo.add(%split1, %bias3): !mo.tensor<[4, 8], f32>
    %merge2 = mo.merge_dim[0](%add3): (!mo.tensor<[4, 8], f32>) -> !mo.tensor<[32], f32>
    %add4 = mo.add(%merge2, %bias4): !mo.tensor<[32], f32>
    %split2 = mo.split_dim[0](%add4): (!mo.tensor<[32], f32>) -> !mo.tensor<[4, 8], f32>
    %add5 = mo.add(%split2, %bias5): !mo.tensor<[4, 8], f32>
    mo.output %add5: !mo.tensor<[4, 8], f32>
}
