// RUN: mo-opt %s --mo-to-mogg="kernel-library-paths=%kernel_api_test_lib" | FileCheck %s

// Check mo-to-mogg correctly lowers an elementwise KernelAPI

// CHECK: mogg.kernel
// CHECK: kgen.param.declare.region {{.+}}_output_{{.+}}_fn =
// CHECK: kgen.param.declare.region {{.+}}_input_{{.+}}_fn =
// CHECK: kgen.param.declare.region {{.+}}_input_{{.+}}_fn =
// CHECK: kgen.param.declare.region {{.+}}func
// CHECK: fusionBodyLambda = "{{.+}}func
// CHECK-SAME: inputFusionInterface = ["MOGG{{.+}}", "MOGG{{.+}}"]{{.+}} outputFusionInterface = ["MOGG{{.+}}"]

mo.graph @add_elemwise() -> !mo.tensor<[2, 3], f32> {
  %x = mo.constant {value = #M.dense_array<1.0, 1.5, 2.0, 2.5, 3.0, 3.5> : tensor<6xf32>} : !mo.tensor<[2, 3], f32>
  %y = mo.constant {value = #M.dense_array<3.0, 3.5, 4.0, 4.5, 5.0, 5.5> : tensor<6xf32>} : !mo.tensor<[2, 3], f32>
  %out = mo.custom {symbol = "imposter_add_elementwise"} (%x, %y) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %out : !mo.tensor<[2, 3], f32>
}
