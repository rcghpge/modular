// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt

// Roundtrip tests for the MAP dialect ops and the `!map.index` type. These
// exercise custom parse/print (map.iter.opaque, map.iter.epilogue) and
// declarative formats (map.call, map.iter.store), so re-running through mo-opt
// (the second RUN line) also checks idempotence.

// map.call verifies its callee resolves to a mogg.kernel.decl, so the decls the
// kernels below reference are declared here (the import pass cannot be scoped to
// a single kernel, and importing a whole package is far too much for a parse
// test).
mogg.kernel.decl @kernels.matmul registration = "mo.matmul" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false

// The `!map.index<N>` type prints and parses with its static rank.

// CHECK-LABEL: func.func @index_type_roundtrip
// CHECK-SAME: (%{{.*}}: !map.index<3>) -> !map.index<3>
func.func @index_type_roundtrip(%i: !map.index<3>) -> !map.index<3> {
  return %i : !map.index<3>
}

// A matmul-shaped opaque kernel: two inputs, one rank-2 output, a store
// epilogue that captures the output, and a call carrying a params dictionary.

// CHECK-LABEL: func.func @opaque_matmul
// CHECK: %[[MM:.*]] = map.iter.opaque inputs (%[[A:.*]] : !mo.tensor<[128, 256], f32>, %[[B:.*]] : !mo.tensor<[256, 512], f32>) outputs (%[[OUT:.*]] : !mo.tensor<[128, 512], f32>) {
// CHECK:   %[[EPI:.*]] = map.iter.epilogue (%[[VAL:.*]] : f32, %[[IDX:.*]] : !map.index<2>) capturing (%[[CAP:.*]] = %[[OUT]] : !mo.tensor<[128, 512], f32>) {
// CHECK:     map.iter.store %[[CAP]][%[[IDX]]] = %[[VAL]] : !mo.tensor<[128, 512], f32>, !map.index<2>, f32
// CHECK:   } : !mo.tensor<[128, 512], f32>
// CHECK:   map.call @kernels.matmul params = {packed_b = false, transpose_b = false} inputs(%[[A]], %[[B]]) outputs(%[[EPI]]) : (!mo.tensor<[128, 256], f32>, !mo.tensor<[256, 512], f32>) -> !mo.tensor<[128, 512], f32>
// CHECK: }
// CHECK: return %[[MM]]
func.func @opaque_matmul(%A: !mo.tensor<[128, 256], f32>, %B: !mo.tensor<[256, 512], f32>) -> !mo.tensor<[128, 512], f32> {
  %mm = map.iter.opaque inputs (%A : !mo.tensor<[128, 256], f32>, %B : !mo.tensor<[256, 512], f32>) outputs (%out : !mo.tensor<[128, 512], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>) capturing (%cap = %out : !mo.tensor<[128, 512], f32>) {
      map.iter.store %cap[%idx] = %val : !mo.tensor<[128, 512], f32>, !map.index<2>, f32
    } : !mo.tensor<[128, 512], f32>
    map.call @kernels.matmul params = {packed_b = false, transpose_b = false} inputs (%A, %B) outputs (%epi) : (!mo.tensor<[128, 256], f32>, !mo.tensor<[256, 512], f32>) -> (!mo.tensor<[128, 512], f32>)
  }
  return %mm : !mo.tensor<[128, 512], f32>
}

// A unary rank-3 opaque kernel: one input, one output, an empty params dict.
// Exercises `!map.index<3>` and the single-input / empty-params paths.

// CHECK-LABEL: func.func @opaque_unary
// CHECK: map.iter.opaque inputs (%[[X:.*]] : !mo.tensor<[2, 3, 4], f32>) outputs (%[[O:.*]] : !mo.tensor<[2, 3, 4], f32>) {
// CHECK:   map.iter.epilogue (%[[V:.*]] : f32, %[[I:.*]] : !map.index<3>) capturing (%[[C:.*]] = %[[O]] : !mo.tensor<[2, 3, 4], f32>) {
// CHECK:     map.iter.store %[[C]][%[[I]]] = %[[V]] : !mo.tensor<[2, 3, 4], f32>, !map.index<3>, f32
// CHECK:   } : !mo.tensor<[2, 3, 4], f32>
// CHECK:   map.call @kernels.relu params = {} inputs(%[[X]]) outputs(%{{.*}}) : (!mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[2, 3, 4], f32>
func.func @opaque_unary(%X: !mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[2, 3, 4], f32> {
  %y = map.iter.opaque inputs (%X : !mo.tensor<[2, 3, 4], f32>) outputs (%out : !mo.tensor<[2, 3, 4], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<3>) capturing (%cap = %out : !mo.tensor<[2, 3, 4], f32>) {
      map.iter.store %cap[%idx] = %val : !mo.tensor<[2, 3, 4], f32>, !map.index<3>, f32
    } : !mo.tensor<[2, 3, 4], f32>
    map.call @kernels.relu params = {} inputs (%X) outputs (%epi) : (!mo.tensor<[2, 3, 4], f32>) -> (!mo.tensor<[2, 3, 4], f32>)
  }
  return %y : !mo.tensor<[2, 3, 4], f32>
}
