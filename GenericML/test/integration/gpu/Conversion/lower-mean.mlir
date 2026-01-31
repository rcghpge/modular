// RUN: mo-opt --mo-to-mgp="compiled-framework-label=mof constant-fold=false default-device-label=gpu" %s | FileCheck --check-prefix=CHECK-IR %s
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=compact %s | FileCheck --check-prefix=CHECK-MT %s

// CHECK-IR-LABEL: mef.callable @graph$exec
// CHECK-IR: %{{.*}} = "mgp.generic.execute"{{.*}}mean

// CHECK-MT-LABEL: --- Running 'graph':
// CHECK-MT: 'graph' returned tensor({{\[}}[1.0000],
// CHECK-MT-NEXT: [1.0000],
// CHECK-MT-NEXT: [1.0000],
// CHECK-MT-NEXT: [1.0000]], dtype=f32, shape=[4,1])

mo.graph @graph(%a : !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 1], f32>
{
  %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %ch0 = mo.chain.create()
  %a_gpu, %ch1 = mo.transfer[%ch0] %a : !mo.tensor<[4, 4], f32> to <"gpu", 0>
  %b_gpu = mo.mean (%a_gpu, %axis) { device = #M.device_ref<"gpu", 0> } : (!mo.tensor<[4, 4], f32, gpu:0>, !mo.tensor<[], si64>) -> !mo.tensor<[4, 1], f32, gpu:0>
  %b, %ch2 = mo.transfer[%ch1] %b_gpu : !mo.tensor<[4, 1], f32, gpu:0> to <"cpu", 0>
  mo.output %b : !mo.tensor<[4, 1], f32>
}
