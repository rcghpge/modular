// RUN: mo-opt %s --mo-to-mogg=use-jit=true | FileCheck %s

// A collection of the support floating point dtypes support by mogg for lowering
// Everything should be jitted to a mo.kernel
// CHECK-NOT: mo.mul

// CHECK: mo.graph @f16Kernel
// CHECK: %[[VAL:.*]] = mogg.kernel
mo.graph @f16Kernel(%arg0: !mo.tensor<[3, 3, 3], f16>, %arg1: !mo.tensor<[3, 3, 3], f16>) -> (!mo.tensor<[3, 3, 3], f16>) {
  %1 = mo.mul(%arg0, %arg1) : !mo.tensor<[3, 3, 3], f16>
  mo.output %1 : !mo.tensor<[3, 3, 3], f16>
}

// CHECK: mo.graph @bf16Kernel
// CHECK: %[[VAL:.*]] = mogg.kernel
mo.graph @bf16Kernel(%arg0: !mo.tensor<[3, 3, 3], bf16>, %arg1: !mo.tensor<[3, 3, 3], bf16>) -> (!mo.tensor<[3, 3, 3], bf16>) {
  %1 = mo.mul(%arg0, %arg1) : !mo.tensor<[3, 3, 3], bf16>
  mo.output %1 : !mo.tensor<[3, 3, 3], bf16>
}

// CHECK: mo.graph @f32Kernel
// CHECK: %[[VAL:.*]] = mogg.kernel
mo.graph @f32Kernel(%arg0: !mo.tensor<[3, 3, 3], f32>, %arg1: !mo.tensor<[3, 3, 3], f32>) -> (!mo.tensor<[3, 3, 3], f32>) {
  %1 = mo.mul(%arg0, %arg1) : !mo.tensor<[3, 3, 3], f32>
  mo.output %1 : !mo.tensor<[3, 3, 3], f32>
}

// CHECK: mo.graph @f64Kernel
// CHECK: %[[VAL:.*]] = mogg.kernel
mo.graph @f64Kernel(%arg0: !mo.tensor<[3, 3, 3], f64>, %arg1: !mo.tensor<[3, 3, 3], f64>) -> (!mo.tensor<[3, 3, 3], f64>) {
  %1 = mo.mul(%arg0, %arg1) : !mo.tensor<[3, 3, 3], f64>
  mo.output %1 : !mo.tensor<[3, 3, 3], f64>
}
