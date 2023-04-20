// RUN: mo-opt --mo-to-mogg=use-jit=true --fuse-elementwise --canonicalize --cse --fuse-bcasts --canonicalize --cse --fuse-consts %s | FileCheck %s

mo.graph @basic_constant_fuse(%arg0: !mo.tensor<[2, 2, 4], f32>) -> (!mo.tensor<[2, 2, 4], f32>) {
  %cst1 = mo.constant {value = #M.dense_array<1.500000e-01, 1.500000e-01, 2.500000e-01, 3.500000e-01, 4.500000e-01, 5.500000e-01, 6.500000e-01, 7.500000e-01, 1.500000e-01, 1.500000e-01, 2.500000e-01, 3.500000e-01, 4.500000e-01, 5.500000e-01, 6.500000e-01, 7.500000e-01> : tensor<16xf32>} : !mo.tensor<[2, 2, 4], f32>
  %cst2 = mo.constant {value = #M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00> : tensor<16xf32>} : !mo.tensor<[2, 2, 4], f32>
  %cst3 = mo.constant {value = #M.dense_array<15.000000e+00, 14.000000e+00, 13.000000e+00, 12.000000e+00, 11.000000e+00, 10.000000e+00, 9.000000e+00, 8.000000e+00, 7.000000e+00, 6.000000e+00, 5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00, 0.000000e+00> : tensor<16xf32>} : !mo.tensor<[2, 2, 4], f32>

  %1 = mo.add(%arg0, %cst1) : !mo.tensor<[2, 2, 4], f32>
  %2 = mo.sub(%1, %cst2) : !mo.tensor<[2, 2, 4], f32>
  %3 = mo.mul(%2, %cst3) : !mo.tensor<[2, 2, 4], f32>
  mo.output %3 : !mo.tensor<[2, 2, 4], f32>
}

// CHECK-LABEL: mo.graph @basic_constant_fuse
// CHECK: (%[[ARG:.*]]: !mo.tensor<[2, 2, 4], f32>)

// Check kernel is next (so consts have been dce'd) & that it has one arg.
// CHECK-NEXT: mogg.kernel (%[[ARG]])
// CHECK-NEXT: mogg.output_placeholders

// Check the constants are inside the kernel now.
// CHECK-NEXT: mo.constant {value = #M.dense_array<1.500000e+01,
// CHECK-NEXT: mo.constant {value = #M.dense_array<1.000000e+00,
// CHECK-NEXT: mo.constant {value = #M.dense_array<1.500000e-01


// Check broadcasted constants get fused
mo.graph @bcasted_const_fusion<D14, D15>(%arg0: !mo.tensor<[D14, D15], f32>) -> !mo.tensor<[D14, D15], f32> {
  %0 = mo.constant {value = #M.dense_array<1.250000e-01> : tensor<f32>} : !mo.tensor<[], f32>
  %1 = mo.constant {value = #M.dense_array<6.000000e+02> : tensor<f32>} : !mo.tensor<[], f32>
  %2 = mo.static.broadcast_to(%0) : !mo.tensor<[], f32> -> !mo.tensor<[D14, D15], f32>
  %3 = mo.static.broadcast_to(%1) : !mo.tensor<[], f32> -> !mo.tensor<[D14, D15], f32>
  %4 = mo.add(%2, %arg0) : !mo.tensor<[D14, D15], f32>
  %5 = mo.div(%4, %3) : !mo.tensor<[D14, D15], f32>
  mo.output %5 : !mo.tensor<[D14, D15], f32>
}


// CHECK-LABEL: mo.graph @bcasted_const_fusion
// CHECK-NEXT: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders

// Check constants are in kernel and used in broadcasts.
// CHECK-NEXT: %[[CST1:.*]] = mo.constant {value = #M.dense_array<1.250000e-01>
// CHECK-NEXT: %[[CST2:.*]] = mo.constant {value = #M.dense_array<6.000000e+02>
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST2]]
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST1]]

// Check broadcasted si32 constants get fused
mo.graph @bcasted_si32_const_fusion<D14, D15>(%arg0: !mo.tensor<[D14, D15], si32>) -> !mo.tensor<[D14, D15], si32> {
  %0 = mo.constant {value = #M.dense_array<1> : tensor<si32>} : !mo.tensor<[], si32>
  %1 = mo.constant {value = #M.dense_array<6> : tensor<si32>} : !mo.tensor<[], si32>
  %2 = mo.static.broadcast_to(%0) : !mo.tensor<[], si32> -> !mo.tensor<[D14, D15], si32>
  %3 = mo.static.broadcast_to(%1) : !mo.tensor<[], si32> -> !mo.tensor<[D14, D15], si32>
  %4 = mo.add(%2, %arg0) : !mo.tensor<[D14, D15], si32>
  %5 = mo.div(%4, %3) : !mo.tensor<[D14, D15], si32>
  mo.output %5 : !mo.tensor<[D14, D15], si32>
}

// CHECK-LABEL: mo.graph @bcasted_si32_const_fusion
// CHECK-NEXT: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders

// Check constants are in kernel and used in broadcasts.
// CHECK-NEXT: %[[CST1:.*]] = mo.constant {value = #M.dense_array<1>
// CHECK-NEXT: %[[CST2:.*]] = mo.constant {value = #M.dense_array<6>
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST2]]
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST1]]


// Check broadcasted si64 constants get fused
mo.graph @bcasted_si64_const_fusion<D14, D15>(%arg0: !mo.tensor<[D14, D15], si64>) -> !mo.tensor<[D14, D15], si64> {
  %0 = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %1 = mo.constant {value = #M.dense_array<6> : tensor<si64>} : !mo.tensor<[], si64>
  %2 = mo.static.broadcast_to(%0) : !mo.tensor<[], si64> -> !mo.tensor<[D14, D15], si64>
  %3 = mo.static.broadcast_to(%1) : !mo.tensor<[], si64> -> !mo.tensor<[D14, D15], si64>
  %4 = mo.add(%2, %arg0) : !mo.tensor<[D14, D15], si64>
  %5 = mo.div(%4, %3) : !mo.tensor<[D14, D15], si64>
  mo.output %5 : !mo.tensor<[D14, D15], si64>
}

// CHECK-LABEL: mo.graph @bcasted_si64_const_fusion
// CHECK-NEXT: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders

// Check constants are in kernel and used in broadcasts.
// CHECK-NEXT: %[[CST1:.*]] = mo.constant {value = #M.dense_array<1>
// CHECK-NEXT: %[[CST2:.*]] = mo.constant {value = #M.dense_array<6>
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST2]]
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST1]]

mo.graph @transpose_fold_perm<D14, D15, D16>(%arg0: !mo.tensor<[D14, D15, D16], f32>) -> !mo.tensor<[D16, D15, D14], f32> {
  %perm = mo.constant {value = #M.dense_array<2, 1, 0> : !M.array<3xsi64>} : !mo.tensor<[3], si64>
  %trans = mo.transpose(%arg0, %perm) : (!mo.tensor<[D14, D15, D16], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D16, D15, D14], f32>
  mo.output %trans : !mo.tensor<[D16, D15, D14], f32>
}

// CHECK-LABEL: mo.graph @transpose_fold_perm
// CHECK-NEXT: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders
// CHECK-NEXT: %[[PERM:.*]] = mo.constant {value = #M.dense_array<2, 1, 0> 
// CHECK-NEXT: mogg.call "mo.transpose"(%[[NULL:.*]], %[[PERM]]
