// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false" -o %t.adv.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.legacy.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.legacy.mlir | FileCheck %s

// `mo.static.reshape` of a literal `mo.constant` folds to a plain constant
// tensor at the MO level (`Static::ReshapeOp::fold`), which would bypass
// `mogg._tensor.create.reshape` entirely and prove nothing about it. Instead
// -- mirroring cumsum.mlir's own idiom -- multiply a same-shaped `ones`
// graph argument by the literal values: the result is numerically identical
// to the literal, but opaque to the compiler (its defining op is a runtime
// argument, not a constant), so the reshape survives into MAPToMOGG and
// EmitMojo and actually exercises the primitive.

// Rank-reducing case (flatten): [4, 6] -> [24]. Distinct input values catch
// a bug that reads the wrong buffer, drops/duplicates elements, or otherwise
// disturbs the flat element order -- not just a wrong output shape.

// CHECK-LABEL: Running 'reshape_flatten':
// CHECK: 'reshape_flatten' returned tensor<24xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]
mo.graph @reshape_flatten(%ones: !mo.tensor<[4, 6], f32>) -> (!mo.tensor<[24], f32>) {
  %values = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0> : tensor<24xf32>} : !mo.tensor<[4, 6], f32>
  %input = mo.mul(%ones, %values) : !mo.tensor<[4, 6], f32>
  %res = mo.static.reshape(%input) : !mo.tensor<[4, 6], f32> -> !mo.tensor<[24], f32>
  mo.output %res : !mo.tensor<[24], f32>
}

// Rank-expanding case (unflatten): [24] -> [4, 6], same underlying values,
// same reasoning as above in the other direction.

// CHECK-LABEL: Running 'reshape_expand':
// CHECK: 'reshape_expand' returned tensor<4x6xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]
mo.graph @reshape_expand(%ones: !mo.tensor<[24], f32>) -> (!mo.tensor<[4, 6], f32>) {
  %values = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0> : tensor<24xf32>} : !mo.tensor<[24], f32>
  %input = mo.mul(%ones, %values) : !mo.tensor<[24], f32>
  %res = mo.static.reshape(%input) : !mo.tensor<[24], f32> -> !mo.tensor<[4, 6], f32>
  mo.output %res : !mo.tensor<[4, 6], f32>
}
