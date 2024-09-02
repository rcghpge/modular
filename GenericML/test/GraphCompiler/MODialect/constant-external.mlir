// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck %s

// CHECK: @constant_external
mo.graph @constant_external() -> !mo.tensor<[4096, 4096], bf16> {
    %weight = mo.constant.external {
      align = 16 : ui64, device = #M.device_ref<"cuda", 0>, name = "foo"
    } : !mo.tensor<[4096, 4096], bf16>
    mo.output %weight : !mo.tensor<[4096, 4096], bf16>
}
