// RUN: mo-opt --mo-to-primitives %s | FileCheck --enable-var-scope %s

// CHECK-LABEL: mgp.model @no_params
// CHECK: execute ([[CHAIN:%.*]]: !mef.chain{{.*}}[[GPU:%.*]]: !mgp.device_context<<"gpu", 0>>
// CHECK: [[SPEC:%.*]] = mgp.tensor_spec.create  : !mgp.tensor_spec<2x2xf32>
// CHECK: [[IDX4:%.*]] = index.constant 4
// CHECK: [[DIM0:%.*]] = mgp.tensor_spec.get_dim[0] [[SPEC]]
// CHECK: [[MUL0:%.*]] = index.mul [[IDX4]], [[DIM0]]
// CHECK: [[DIM1:%.*]] = mgp.tensor_spec.get_dim[1] [[SPEC]]
// CHECK: [[MUL1:%.*]] = index.mul [[MUL0]], [[DIM1]]
// CHECK: [[BUF:%.*]] = mgp.buffer.alloc [[CHAIN]], [[MUL1]], [[GPU]]
// CHECK: [[TENSOR:%.*]] = mgp.tensor.create [[CHAIN]], [[BUF]], [[SPEC]]
// CHECK: mef.output [[CHAIN]]
mo.graph @no_params() {
  %darg0 = mo.buffer.create : !mo.buffer<[2, 2], f32, gpu:0>
  mo.output
}

// CHECK-LABEL: mgp.model @params
// CHECK: execute ([[CHAIN:%.*]]: !mef.chain{{.*}}[[GPU:%.*]]: !mgp.device_context<<"gpu", 0>>
// CHECK: [[D0:%.*]] = mosh.param.to_value = <d0>
// CHECK: [[D0_IDX:%.*]] = index.casts [[D0]]
// CHECK: [[D1:%.*]] = mosh.param.to_value = <d1>
// CHECK: [[D1_IDX:%.*]] = index.casts [[D1]]
// CHECK: [[SPEC:%.*]] = mgp.tensor_spec.create [[D0_IDX]], [[D1_IDX]]
// CHECK: [[IDX4:%.*]] = index.constant 4
// CHECK: [[DIM0:%.*]] = mgp.tensor_spec.get_dim[0] [[SPEC]]
// CHECK: [[MUL0:%.*]] = index.mul [[IDX4]], [[DIM0]]
// CHECK: [[DIM1:%.*]] = mgp.tensor_spec.get_dim[1] [[SPEC]]
// CHECK: [[MUL1:%.*]] = index.mul [[MUL0]], [[DIM1]]
// CHECK: [[BUF:%.*]] = mgp.buffer.alloc [[CHAIN]], [[MUL1]], [[GPU]]
// CHECK: [[TENSOR:%.*]] = mgp.tensor.create [[CHAIN]], [[BUF]], [[SPEC]]
// CHECK: mef.output [[CHAIN]]
mo.graph @params<d0, d1>() {
  %darg0 = mo.buffer.create : !mo.buffer<[d0, d1], f32, gpu:0>
  mo.output
}
