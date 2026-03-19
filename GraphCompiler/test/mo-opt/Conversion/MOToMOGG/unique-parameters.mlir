// RUN: mo-opt --mo-to-mogg="kernel-library-paths=%kernel_api_test_lib" %s | FileCheck %s

// Expecting `head_dim` not to collide with mo.custom's `head_dim` attribute.
mo.graph @create_collection<num_blocks, num_layers, page_size, num_kv_heads, head_dim, batch_size, steps_remaining>(%arg0: !mo.tensor<[num_blocks, 2, num_layers, page_size, num_kv_heads, head_dim], f32>, %arg1: !mo.tensor<[batch_size], ui32>, %arg2: !mo.tensor<[batch_size], ui32>, %arg3: !mo.tensor<[steps_remaining, 2], ui32>) -> !mo.opaque<"MyCustomScalarRegTrivialSI32"> attributes {argument_names = ["input0", "input1", "input2", "input3"], result_names = ["output0"]} {
  %0 = mo.custom {
    symbol = "unique-parameters",
    device = #M.device_ref<"cpu", 0>,
    parameters = {
      head_dim = 128 : index,
      num_heads = 8 : index
    }
  }(%arg0, %arg1, %arg2, %arg3) : (!mo.tensor<[num_blocks, 2, num_layers, page_size, num_kv_heads, head_dim], f32>, !mo.tensor<[batch_size], ui32>, !mo.tensor<[batch_size], ui32>, !mo.tensor<[steps_remaining, 2], ui32>) -> !mo.opaque<"MyCustomScalarRegTrivialSI32">
  mo.output %0 : !mo.opaque<"MyCustomScalarRegTrivialSI32">
}

// CHECK-LABEL: @create_collection
// CHECK: mogg.{{.*}}kernel{{.*}}head_dim
