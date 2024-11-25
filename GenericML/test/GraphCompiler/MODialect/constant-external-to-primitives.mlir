// RUN: mo-opt %s --mo-to-primitives | FileCheck %s

// Test host weights with device output.
// CHECK-LABEL: @external_weights_cpu_to_gpu
// CHECK-DAG: [[HOST_BUFFER:%.*]] = mgp.buffer.constant.external
// CHECK-DAG: [[CUDA_BUFFER:%.*]] = mgp.buffer.alloc{{.*}}gpu
// CHECK: mgp.buffer.host_to_device{{.*}}[[HOST_BUFFER]], [[CUDA_BUFFER]]
mo.graph @external_weights_cpu_to_gpu() -> !mo.tensor<[4096], f32, gpu:0> attributes {argument_names = [], device = #M.device_ref<"gpu", 0>, result_names = ["output0"], target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_86", index_bit_width = 0>} {
  %0 = mo.constant.external {align = 4 : ui64, device = #M.device_ref<"cpu", 0>, name = "foo"} : !mo.tensor<[4096], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4096], f32, gpu:0>
}

// Test external weights that are already on device.
// CHECK-LABEL: @external_weights_gpu_resident
// CHECK: [[WEIGHTS_BUFFER:%.*]] = mgp.buffer.constant.external{{.*}}gpu
// CHECK-NOT: mgp.buffer.host_to_device
mo.graph @external_weights_gpu_resident() -> !mo.tensor<[4096], f32, gpu:0> attributes {argument_names = [], device = #M.device_ref<"gpu", 0>, result_names = ["output0"], target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_86", index_bit_width = 0>} {
  %0 = mo.constant.external {align = 4 : ui64, device = #M.device_ref<"gpu", 0>, name = "foo"} : !mo.tensor<[4096], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4096], f32, gpu:0>
}
