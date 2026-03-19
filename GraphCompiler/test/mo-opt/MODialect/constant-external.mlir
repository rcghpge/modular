// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck %s

// CHECK-LABEL: @constant_external
mo.graph @constant_external() -> (!mo.tensor<[128, 128], bf16, gpu:0>, !mo.tensor<[256, 256], bf16, gpu:0>, !mo.tensor<[128, 128], bf16, gpu:0>) {
  // CHECK: %{{.*}} = mo.constant.external {{.*}} name = "foo"
  %weight = mo.constant.external {
    align = 16 : ui64, device = #M.device_ref<"gpu", 0>, name = "foo"
  } : !mo.tensor<[128, 128], bf16, gpu:0>

  // CHECK: %{{.*}} = mo.constant.external {{.*}} hasAlias = true, name = "bar"
  %weight_mutable = mo.constant.external {
    align = 16 : ui64, device = #M.device_ref<"gpu", 0>, name = "bar", hasAlias = true
  } : !mo.tensor<[256, 256], bf16, gpu:0>

  // CHECK: %{{.*}} = mo.constant.external {{.*}} isPlaceholder = true, name = "placeholder"
  %weight_placeholder = mo.constant.external {
    align = 16 : ui64, device = #M.device_ref<"gpu", 0>, name = "placeholder", isPlaceholder = true
  } : !mo.tensor<[128, 128], bf16, gpu:0>

  mo.output %weight, %weight_mutable, %weight_placeholder : !mo.tensor<[128, 128], bf16, gpu:0>, !mo.tensor<[256, 256], bf16, gpu:0>, !mo.tensor<[128, 128], bf16, gpu:0>
}
