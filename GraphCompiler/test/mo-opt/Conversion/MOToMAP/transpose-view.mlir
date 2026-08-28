// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages --mo-to-map | FileCheck %s

// Verify that mo.transpose (with a compile-time-constant perm) lowers to a
// standalone (unfused) map.iter.foreach over the output shape, with a
// map.index.transpose op computing the input index -- not map.iter.opaque,
// and not the "view kernels are not yet supported" error.

// 2D swap: out.dim[0] = in.dim[1], out.dim[1] = in.dim[0].

// CHECK-LABEL: mo.graph @transposeSwap
// CHECK-NOT:     map.iter.opaque
// CHECK:         [[OUT:%.*]] = map.iter.foreach<[3, 2]>
// CHECK:         inputs  ([[INPUT:%.*]]: !mo.tensor<[2, 3], f32>{{.*}})
// CHECK:         outputs ([[OUTARG:%.*]]: !mo.tensor<[3, 2], f32>)
// CHECK:         index   ([[IDX:%.*]]: !map.index<2>) -> !mo.tensor<[3, 2], f32> {
// CHECK:         [[IDX_:%.*]] = map.index.transpose<#map.index.transpose.spec<[2, 3] -> [3, 2], perm = [1, 0]>> ([[IDX]])
// CHECK:         [[VAL:%.*]] = map.load [[INPUT]]{{\[}}[[IDX_]]] : !mo.tensor<[2, 3], f32>
// CHECK:         map.store [[OUTARG]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[3, 2], f32>
mo.graph @transposeSwap(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32> {
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// 3D permutation: out.dim[i] = in.dim[perm[i]] for perm = [2, 0, 1].

// CHECK-LABEL: mo.graph @transpose3D
// CHECK-NOT:     map.iter.opaque
// CHECK:         [[OUT2:%.*]] = map.iter.foreach<[4, 2, 3]>
// CHECK:         inputs  ([[INPUT2:%.*]]: !mo.tensor<[2, 3, 4], f32>{{.*}})
// CHECK:         outputs ([[OUTARG2:%.*]]: !mo.tensor<[4, 2, 3], f32>)
// CHECK:         index   ([[IDX2:%.*]]: !map.index<3>) -> !mo.tensor<[4, 2, 3], f32> {
// CHECK:         [[IDX2_:%.*]] = map.index.transpose<#map.index.transpose.spec<[2, 3, 4] -> [4, 2, 3], perm = [2, 0, 1]>> ([[IDX2]])
// CHECK:         [[VAL2:%.*]] = map.load [[INPUT2]]{{\[}}[[IDX2_]]] : !mo.tensor<[2, 3, 4], f32>
// CHECK:         map.store [[OUTARG2]]{{\[}}[[IDX2]]] = [[VAL2]] : !mo.tensor<[4, 2, 3], f32>
mo.graph @transpose3D(%input: !mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[4, 2, 3], f32> {
  %perm = mo.constant {value = #M.dense_array<2, 0, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[2, 3, 4], f32>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<[4, 2, 3], f32>
  mo.output %res : !mo.tensor<[4, 2, 3], f32>
}

// Symbolic dim (N): proving the lowering works when the static shape isn't
// fully concrete either.

// CHECK-LABEL: mo.graph @transposeSymbolicDim
// CHECK-NOT:     map.iter.opaque
// CHECK:         map.iter.foreach<[64, N]>
// CHECK:         map.index.transpose<#map.index.transpose.spec<[N, 64] -> [64, N], perm = [1, 0]>>
mo.graph @transposeSymbolicDim<N>(%input: !mo.tensor<[N, 64], f32>) -> !mo.tensor<[64, N], f32> {
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[N, 64], f32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[64, N], f32>
  mo.output %res : !mo.tensor<[64, N], f32>
}
