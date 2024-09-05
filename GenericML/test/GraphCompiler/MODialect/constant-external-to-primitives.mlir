// RUN: mo-opt %s --mo-to-primitives | FileCheck %s

// CHECK-LABEL: @external_weights
// CHECK-DAG: [[HOST_BUFFER:%.*]] = mgp.buffer.constant.external
// CHECK-DAG: [[CUDA_BUFFER:%.*]] = mgp.buffer.alloc{{.*}}cuda
// CHECK: mgp.buffer.host_to_device{{.*}}[[HOST_BUFFER]], [[CUDA_BUFFER]]
mo.graph @external_weights() -> !mo.tensor<[4096], f32, cuda:0> attributes {argument_names = [], device = #M.device_ref<"cuda", 0>, result_names = ["output0"], target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_86", index_bit_width = 0>} {
  %0 = mo.constant.external {align = 4 : ui64, device = #M.device_ref<"cuda", 0>, name = "foo"} : !mo.tensor<[4096], f32, cuda:0>
  mo.output %0 : !mo.tensor<[4096], f32, cuda:0>
}
