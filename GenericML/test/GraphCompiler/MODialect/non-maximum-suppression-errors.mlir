//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @non_maximum_suppression(%arg0 : !mo.tensor<[6, 4], f32>, %arg1 : !mo.tensor<[1, 1, 6], f32>, %arg2 : !mo.tensor<[], si64>, %arg3 : !mo.tensor<[], f32>, %arg4 : !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64> {
    // expected-error@+1 {{'mo.non_maximum_suppression' op operand #0 must be rank-3 tensor, but got '!mo.tensor<[6, 4], f32>'}}
    %output = mo.non_maximum_suppression(%arg0, %arg1, %arg2, %arg3, %arg4) : (!mo.tensor<[6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64>
    mo.output %output : !mo.tensor<[?, ?], si64>
}

// -----

mo.graph @non_maximum_suppression(%arg0 : !mo.tensor<[1, 6, 4], f32>, %arg1 : !mo.tensor<[1, 6], f32>, %arg2 : !mo.tensor<[], si64>, %arg3 : !mo.tensor<[], f32>, %arg4 : !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64> {
    // expected-error@+1 {{'mo.non_maximum_suppression' op operand #1 must be rank-3 tensor, but got '!mo.tensor<[1, 6], f32>'}}
    %output = mo.non_maximum_suppression(%arg0, %arg1, %arg2, %arg3, %arg4) : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64>
    mo.output %output : !mo.tensor<[?, ?], si64>
}

// -----

mo.graph @non_maximum_suppression(%arg0 : !mo.tensor<[1, 6, 4], f32>, %arg1 : !mo.tensor<[1, 1, 6], f32>, %arg2 : !mo.tensor<[], f32>, %arg3 : !mo.tensor<[], f32>, %arg4 : !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64> {
    // expected-error@+1 {{'mo.non_maximum_suppression' op operand #2 must be ranked scalar index tensor, but got '!mo.tensor<[], f32>'}}
    %output = mo.non_maximum_suppression(%arg0, %arg1, %arg2, %arg3, %arg4) : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64>
    mo.output %output : !mo.tensor<[?, ?], si64>
}

// -----

mo.graph @non_maximum_suppression(%arg0 : !mo.tensor<[1, 6, 4], f32>, %arg1 : !mo.tensor<[1, 1, 6], f32>, %arg2 : !mo.tensor<[], si64>, %arg3 : !mo.tensor<[], si64>, %arg4 : !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64> {
    // expected-error@+1 {{'mo.non_maximum_suppression' op operand #3 must be ranked scalar floating point tensor, but got '!mo.tensor<[], si64>'}}
    %output = mo.non_maximum_suppression(%arg0, %arg1, %arg2, %arg3, %arg4) : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64>
    mo.output %output : !mo.tensor<[?, ?], si64>
}

// -----

mo.graph @non_maximum_suppression(%arg0 : !mo.tensor<[1, 6, 4], f32>, %arg1 : !mo.tensor<[1, 1, 6], f32>, %arg2 : !mo.tensor<[], si64>, %arg3 : !mo.tensor<[], f32>, %arg4 : !mo.tensor<[], si64>) -> !mo.tensor<[?, ?], si64> {
    // expected-error@+1 {{'mo.non_maximum_suppression' op operand #4 must be ranked scalar floating point tensor, but got '!mo.tensor<[], si64>'}}
    %output = mo.non_maximum_suppression(%arg0, %arg1, %arg2, %arg3, %arg4) : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?], si64>
    mo.output %output : !mo.tensor<[?, ?], si64>
}
