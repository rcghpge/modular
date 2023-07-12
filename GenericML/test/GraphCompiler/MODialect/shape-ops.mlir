// RUN: mo-opt %s --allow-unregistered-dialect | FileCheck %s

// CHECK: mo.graph @broadcast_to(%[[FROM:.*]]: !mo.tensor<[3], f32>, %[[TO:.*]]: !mosh.ape) -> !mo.tensor<[2, 3], f32> {
mo.graph @broadcast_to(%from: !mo.tensor<[3], f32>, %to: !mosh.ape) -> !mo.tensor<[2, 3], f32> {
  // CHECK: [[RES:.*]] = mo.shape.broadcast_to(%[[FROM]], %[[TO]]) : (!mo.tensor<[3], f32>, !mosh.ape) -> !mo.tensor<[2, 3], f32>
  %0 = mo.shape.broadcast_to(%from, %to) : (!mo.tensor<[3], f32>, !mosh.ape) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @reshape(%[[FROM:.*]]: !mo.tensor<[3], f32>, %[[TO:.*]]: !mosh.ape) -> !mo.tensor<[2, 3], f32> {
mo.graph @reshape(%from: !mo.tensor<[3], f32>, %to: !mosh.ape) -> !mo.tensor<[2, 3], f32> {
  // CHECK: [[RES:.*]] = mo.shape.reshape(%[[FROM]], %[[TO]]) : (!mo.tensor<[3], f32>, !mosh.ape) -> !mo.tensor<[2, 3], f32>
  %0 = mo.shape.reshape(%from, %to) : (!mo.tensor<[3], f32>, !mosh.ape) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}


// CHECK-LABEL: mo.graph @invoke_shape_func_reduce
mo.graph @invoke_shape_func_reduce<D1, D2 -> D3, D4>(
  %input: !mo.tensor<[D1, D2], f32>,
  %axis: !mo.tensor<[], si32>
) -> !mo.tensor<[D3, D4], f32> {

  %input_shape = mo.shape_of(%input) : (!mo.tensor<[D1, D2], f32>) -> !mo.tensor<[2], si32>

  // CHECK: mo.invoke_shape_func["reduce_mul_fw_shape_func"]<() -> D5, D6
  mo.invoke_shape_func["reduce_mul_fw_shape_func"]<() -> D5, D6>(%input_shape, %axis)
  : (!mo.tensor<[2], si32>, !mo.tensor<[], si32>)

  // CHECK: mo.reduce.mul(%{{.*}}, %{{.*}}) : (!mo.tensor<[D1, D2], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[D5, D6], f32>
  %out = mo.reduce.mul(%input, %axis) : (
    !mo.tensor<[D1, D2], f32>, !mo.tensor<[], si32>
  ) -> !mo.tensor<[D5, D6], f32>

  mo.output<D5, D6> %out : !mo.tensor<[D5, D6], f32>
}

// CHECK-LABEL: mo.graph @invoke_shape_func_slice
mo.graph @invoke_shape_func_slice<D1, D2 -> D3, D4>(
  %input: !mo.tensor<[D1, D2], f32>,
  %starts: !mo.tensor<[2], si32>,
  %stops: !mo.tensor<[2], si32>,
  %steps: !mo.tensor<[2], si32>
) -> !mo.tensor<[D3, D4], f32> {

  %input_shape = mo.shape_of(%input) : (!mo.tensor<[D1, D2], f32>) -> !mo.tensor<[2], si32>

  // CHECK: mo.invoke_shape_func["slice_fw_shape_func"]<() -> D5, D6
  mo.invoke_shape_func["slice_fw_shape_func"]<() -> D5, D6>
  (%input_shape, %starts, %stops, %steps)
  : (!mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>)

  // CHECK: mo.slice({{.*}}) : (!mo.tensor<[D1, D2], f32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[D5, D6], f32>
  %out = mo.slice(%input, %starts, %stops, %steps) : (
    !mo.tensor<[D1, D2], f32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[D5, D6], f32>

  mo.output<D5, D6> %out : !mo.tensor<[D5, D6], f32>
}

// CHECK-LABEL: mo.graph @invoke_shape_func_matmul_packing
mo.graph @invoke_shape_func_matmul_packing<D1, D2 -> D3, D4, D5>(
  %input: !mo.tensor<[D1, D2], f32>
) -> !mo.tensor<[D3, D4, D5], f32> {

  %input_shape = mo.shape_of(%input) : (!mo.tensor<[D1, D2], f32>) -> !mo.tensor<[2], si32>

  // CHECK: mo.invoke_shape_func["pack_matmul_b_shape_func"]<() -> D6, D7, D8>
  mo.invoke_shape_func["pack_matmul_b_shape_func"]<() -> D6, D7, D8>(%input_shape)
  {transpose = true} : (!mo.tensor<[2], si32>)

  // CHECK: "mop.kernels.pack_matmul_b_transp"{{.*}}(!mo.tensor<[D1, D2], f32>) -> !mo.tensor<[D6, D7, D8], f32>
  %out = "mop.kernels.pack_matmul_b_transp"(%input)
  : (!mo.tensor<[D1, D2], f32>) -> !mo.tensor<[D6, D7, D8], f32>

  mo.output<D6, D7, D8> %out : !mo.tensor<[D6, D7, D8], f32>
}
