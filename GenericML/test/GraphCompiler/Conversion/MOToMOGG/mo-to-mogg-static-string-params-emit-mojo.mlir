// RUN: mo-opt --mo-to-mogg="kernel-library-paths=%kernel_api_test_pkg" %s | FileCheck %s

// CHECK-LABEL: mo.graph @static_string_params
// CHECK-NEXT: mogg.kernel
// CHECK-NEXT: mogg.call.execute["custom_op_with_static_string_params"] {kernel_param = {foo = "bar"}}
// CHECK-NEXT: summary = "custom__custom_op_with_static_string_params
// CHECK-NEXT: mo.output
mo.graph @static_string_params() {
    mo.custom {
      symbol = "custom_op_with_static_string_params",
      device = #M.device_ref<"cpu", 0>,
      parameters = { foo = "bar" }
    } () : () -> ()
    mo.output
}
