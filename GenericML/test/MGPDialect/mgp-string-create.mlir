// RUN: mo-opt --allow-unregistered-dialect %s | FileCheck %s

//
// mgp.string.create
//

// CHECK-LABEL: @string_create_basic
func.func @string_create_basic() {
  // CHECK: %{{.*}} = mgp.string.create "hello world" : !mgp.opaque<"mojo_string">
  %str = mgp.string.create "hello world" : !mgp.opaque<"mojo_string">
  mef.output %str : !mgp.opaque<"mojo_string">
}

// CHECK-LABEL: @string_create_empty
func.func @string_create_empty() {
  // CHECK: %{{.*}} = mgp.string.create "" : !mgp.opaque<"mojo_string">
  %str = mgp.string.create "" : !mgp.opaque<"mojo_string">
  mef.output %str : !mgp.opaque<"mojo_string">
}

// CHECK-LABEL: @string_create_special_chars
func.func @string_create_special_chars() {
  // CHECK: %{{.*}} = mgp.string.create "path/to/file.txt" : !mgp.opaque<"mojo_string">
  %str = mgp.string.create "path/to/file.txt" : !mgp.opaque<"mojo_string">
  mef.output %str : !mgp.opaque<"mojo_string">
}

// CHECK-LABEL: @string_create_in_model
mgp.model @string_create_in_model : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %chain = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %chain, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%chain: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.opaque<"mojo_string">) {
  // CHECK: %{{.*}} = mgp.string.create "test_string" : !mgp.opaque<"mojo_string">
  %str = mgp.string.create "test_string" : !mgp.opaque<"mojo_string">
  mef.output %chain, %str : !mef.chain, !mgp.opaque<"mojo_string">
}
