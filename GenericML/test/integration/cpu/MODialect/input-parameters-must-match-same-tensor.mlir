// RUN: mo-opt --mo-to-mgp="constant-fold=false" %s -o %t.mlir

// RUN: not mt --execute  --model-inputs=ones --input-shapes=7x10 --result-output-style=full %t.mlir | FileCheck --enable-var-scope %s

// CHECK-LABEL: --- Running 'dims_must_match_same_tensor':
// CHECK-NEXT: 'dims_must_match_same_tensor' returned <error: symbolic dimension 'D0' for input 0 does not match prior uses of that dimension>
mo.graph @dims_must_match_same_tensor<D0>(%X: !mo.tensor<[D0, D0], f32>) -> !mo.tensor<[D0, D0], f32>
{
  mo.output %X: !mo.tensor<[D0, D0], f32>
}
