// RUN: mo-opt --allow-unregistered-dialect --canonicalize --cse %s | mo-opt --allow-unregistered-dialect | FileCheck %s

//
// mgp.runtime.create
//

// CHECK-LABEL: @runtime_create
func.func @runtime_create(%ctx : !mgp.context, %ch0 : !mef.chain) {
  // CHECK: %{{.*}} = mgp.runtime.create %{{.*}}, %{{.*}}[1] device <"cuda", 3>
  %ch1 = mgp.runtime.create %ch0, %ctx[1] device <"cuda", 3>
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: @runtime_create_host
func.func @runtime_create_host(%ctx : !mgp.context, %ch0 : !mef.chain) {
  // CHECK: %{{.*}} = mgp.runtime.create %{{.*}}, %{{.*}}[1]
  %ch1 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch1 : !mef.chain
}

//
// mgp.module.load
//

// CHECK-LABEL: @module_load_dense_attribute
func.func @module_load_dense_attribute(%ctx : !mgp.context, %ch0 : !mef.chain) {
  // CHECK: %{{.*}} = mgp.module.load %{{.*}}, %{{.*}}[1], dense_resource<resource> : !M.aligned_bytes<4, align 64>, ["f1", "f2"],  ["g1", "g2"] device <"cuda", 3>
  %ch1 = mgp.module.load %ch0, %ctx[1], dense_resource<resource> : !M.aligned_bytes<4, align 64>, ["f1", "f2"], ["g1", "g2"] device <"cuda", 3>
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: @module_load_multiline
func.func @module_load_multiline(%ctx : !mgp.context, %ch0 : !mef.chain) {
  // CHECK: %{{.*}} = mgp.module.load %{{.*}}, %{{.*}}[1], #M<multiline["a", "b", "c"]>, ["f"], [] device <"cuda", 3>
  %ch1 = mgp.module.load %ch0, %ctx[1], #M<multiline["a", "b", "c"]>, ["f"], [] device <"cuda", 3>
  mef.output %ch1 : !mef.chain
}

//
// mgp.chain.create
//

// CHECK-LABEL: @chain_create
func.func @chain_create(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %ch1 : !mgp.chain<<"cuda", 3>>, %ch2 : !mgp.chain, %ch3 : !mef.chain, %dev_buf : !mgp.buffer<, <"cuda", 3>>, %host_buf : !mgp.buffer<>) {
  // CHECK: %{{.*}} = mgp.chain.create %{{.*}}[1], other <"cuda", 3>, (%{{.*}}, %{{.*}}, %{{.*}}: !mgp.chain, %{{.*}}: !mef.chain, %{{.*}}: !mgp.buffer<, <"cuda", 3>>, %{{.*}}: !mgp.buffer<>)
  %ch4 = mgp.chain.create %ctx[1], other <"cuda", 3>, (%ch0, %ch1, %ch2: !mgp.chain, %ch3: !mef.chain, %dev_buf: !mgp.buffer<, <"cuda", 3>>, %host_buf: !mgp.buffer<>)
  mef.output %ch4 : !mgp.chain<<"cuda", 3>>
}

// CHECK-LABEL: @chain_create_canonicalization
// CHECK-SAME: (%{{.*}}: !mgp.context, %[[CH0:.*]]: !mgp.chain<<"cuda", 3>>, %[[CH1:.*]]: !mgp.chain<<"cuda", 3>>) {
func.func @chain_create_canonicalization(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %ch1 : !mgp.chain<<"cuda", 3>>) {
  // CHECK: %[[CH2:.*]] = mgp.chain.create %{{.*}}[1], other <"cuda", 3>, (%[[CH0]], %[[CH1]])
  %ch2 = mgp.chain.create %ctx[1], other <"cuda", 3>, (%ch0, %ch1, %ch0, %ch1)
  // CHECK: mef.output %[[CH2]]
  %ch3 = mgp.chain.create %ctx[1], other <"cuda", 3>, (%ch2)
  mef.output %ch3 : !mgp.chain<<"cuda", 3>>
}

//
// mgp.chain.device_to_host
//

// CHECK-LABEL: @chain_device_to_host
func.func @chain_device_to_host(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.chain.device_to_host %{{.*}}, %{{.*}}[1] device <"cuda", 3>
  %ch1 = mgp.chain.device_to_host %ch0, %ctx[1] device <"cuda", 3>
  mef.output %ch1 : !mef.chain
}

//
// mgp.chain.host_to_device
//

// CHECK-LABEL: @chain_host_to_device
func.func @chain_host_to_device(%ctx : !mgp.context, %ch0 : !mef.chain) {
  // CHECK: %{{.*}} = mgp.chain.host_to_device %{{.*}}, %{{.*}}[1] device <"cuda", 3>
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1] device <"cuda", 3>
  mef.output %ch1 : !mgp.chain<<"cuda", 3>>
}

// CHECK-LABEL:@chain_host_device_canonicalization_1
func.func @chain_host_device_canonicalization_1(%ctx : !mgp.context, %ch0 : !mef.chain) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1] device <"cuda", 3>
  %ch2 = mgp.chain.device_to_host %ch1, %ctx[1] device <"cuda", 3>
  // CHECK-NEXT: mef.output
  mef.output %ch2 : !mef.chain
}

// CHECK-LABEL:@chain_host_device_canonicalization_2
func.func @chain_host_device_canonicalization_2(%ctx : !mgp.context, %ch0 : !mef.chain) {
  // CHECK-NEXT: %[[CH1:.*]] = mgp.chain.host_to_device %{{.*}}, %{{.*}}[1] device <"cuda", 3>
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1] device <"cuda", 3>
  %ch2 = mgp.chain.device_to_host %ch1, %ctx[1] device <"cuda", 3>
  %ch3 = mgp.chain.host_to_device %ch2, %ctx[1] device <"cuda", 3>
  // CHECK-NEXT: "some.op"(%[[CH1]], %[[CH1]])
  "some.op"(%ch1, %ch3) : (!mgp.chain<<"cuda", 3>>, !mgp.chain<<"cuda", 3>>) -> ()
  mef.output %ch0 : !mef.chain
}

//
// mgp.buffer.constant
//

// CHECK-LABEL: @buffer_constant
func.func @buffer_constant(%ch: !mef.chain, %ctx : !mgp.context) {
  // CHECK: %{{.*}} = mgp.buffer.constant %{{.*}}, %{{.*}}[1], dense_resource<resource> : !M.aligned_bytes<4, align 64> device <"cuda", 3>
  %buf = mgp.buffer.constant %ch, %ctx[1], dense_resource<resource> : !M.aligned_bytes<4, align 64> device <"cuda", 3>
  mef.output %buf : !mgp.buffer<4, <"cuda", 3>>
}

//
// mgp.buffer.constant.load
//

// CHECK-LABEL: @buffer_constant_load
func.func @buffer_constant_load(%ctx : !mgp.context) {
  // CHECK: %{{.*}} = mgp.buffer.constant.load %{{.*}}[1], (3, 7) : !mgp.buffer<124, <"cuda", 3>>
  %buf = mgp.buffer.constant.load %ctx[1], (3, 7) : !mgp.buffer<124, <"cuda", 3>>
  mef.output %buf : !mgp.buffer<124, <"cuda", 3>>
}

//
// mgp.buffer.alloc.static
//

// CHECK-LABEL: @buffer_alloc_static
func.func @buffer_alloc_static(%ctx : !mgp.context, %ch : !mgp.chain<<"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.buffer.alloc.static %{{.*}}, %{{.*}}[1] : !mgp.buffer<124, <"cuda", 3>>
  %buf = mgp.buffer.alloc.static %ch, %ctx[1] : !mgp.buffer<124, <"cuda", 3>>
  mef.output %buf : !mgp.buffer<124, <"cuda", 3>>
}

//
// mgp.buffer.alloc.dynamic
//

// CHECK-LABEL: @buffer_alloc_dynamic
func.func @buffer_alloc_dynamic(%ctx : !mgp.context, %ch : !mgp.chain<<"cuda", 3>>, %size : index) {
  // CHECK: %{{.*}} = mgp.buffer.alloc.dynamic %{{.*}}, %{{.*}}[1], %{{.*}} : !mgp.buffer<, <"cuda", 3>>
  %buf = mgp.buffer.alloc.dynamic %ch, %ctx[1], %size : !mgp.buffer<, <"cuda", 3>>
  mef.output %buf : !mgp.buffer<, <"cuda", 3>>
}


//
// mgp.buffer.cache
//

// CHECK-LABEL: @buffer_cache
func.func @buffer_cache(%ctx : !mgp.context, %ch0 : !mef.chain, %buf : !mgp.buffer<256, <"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.buffer.cache %{{.*}}, %{{.*}}[1, 5], %{{.*}} : !mgp.buffer<256, <"cuda", 3>>
  %ch1 = mgp.buffer.cache %ch0, %ctx[1, 5], %buf : !mgp.buffer<256, <"cuda", 3>>
  mef.output %ch1 : !mef.chain
}

//
// mgp.buffer.get_cached
//

// CHECK-LABEL: @buffer_get_cached
func.func @buffer_get_cached(%ctx : !mgp.context, %ch0 : !mef.chain) {
  // CHECK: %{{.*}} = mgp.buffer.get_cached %{{.*}}, %{{.*}}[1, 5] : !mgp.buffer<256, <"cuda", 3>>
  %buf = mgp.buffer.get_cached %ch0, %ctx[1, 5] : !mgp.buffer<256, <"cuda", 3>>
  mef.output %buf : !mgp.buffer<256, <"cuda", 3>>
}

//
// mgp.buffer.host_to_device
//

// CHECK-LABEL: @buffer_host_to_device_static
func.func @buffer_host_to_device_static(%ctx : !mgp.context, %ch0 : !mgp.chain, %cpu_buf : !mgp.buffer<48>, %cuda_buf : !mgp.buffer<48, <"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.buffer.host_to_device %{{.*}}, %{{.*}}[0, 1], %{{.*}}, %{{.*}} : !mgp.buffer<48> -> !mgp.buffer<48, <"cuda", 3>>
  %ch1 = mgp.buffer.host_to_device %ch0, %ctx[0, 1], %cpu_buf, %cuda_buf : !mgp.buffer<48> -> !mgp.buffer<48, <"cuda", 3>>
  mef.output %ch1 : !mgp.chain<<"cuda", 3>>
}

// CHECK-LABEL: @buffer_host_to_device_dynamic
func.func @buffer_host_to_device_dynamic(%ctx : !mgp.context, %ch0 : !mgp.chain, %cpu_buf : !mgp.buffer<>, %cuda_buf : !mgp.buffer<, <"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.buffer.host_to_device %{{.*}}, %{{.*}}[0, 1], %{{.*}}, %{{.*}} : !mgp.buffer<> -> !mgp.buffer<, <"cuda", 3>>
  %ch1 = mgp.buffer.host_to_device %ch0, %ctx[0, 1], %cpu_buf, %cuda_buf : !mgp.buffer<> -> !mgp.buffer<, <"cuda", 3>>
  mef.output %ch1 : !mgp.chain<<"cuda", 3>>
}

//
// mgp.buffer.device_to_host
//

// CHECK-LABEL: @buffer_device_to_host_static
func.func @buffer_device_to_host_static(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %cuda_buf : !mgp.buffer<48, <"cuda", 3>>, %cpu_buf : !mgp.buffer<48>) {
  // CHECK: %{{.*}} = mgp.buffer.device_to_host %{{.*}}, %{{.*}}[1, 0], %{{.*}}, %{{.*}} : !mgp.buffer<48, <"cuda", 3>> -> !mgp.buffer<48>
  %ch1 = mgp.buffer.device_to_host %ch0, %ctx[1, 0], %cuda_buf, %cpu_buf : !mgp.buffer<48, <"cuda", 3>> -> !mgp.buffer<48>
  mef.output %ch1 : !mgp.chain
}

// CHECK-LABEL: @buffer_device_to_host_dynamic
func.func @buffer_device_to_host_dynamic(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %cuda_buf : !mgp.buffer<, <"cuda", 3>>, %cpu_buf : !mgp.buffer<>) {
  // CHECK: %{{.*}} = mgp.buffer.device_to_host %{{.*}}, %{{.*}}[1, 0], %{{.*}}, %{{.*}} : !mgp.buffer<, <"cuda", 3>> -> !mgp.buffer<>
  %ch1 = mgp.buffer.device_to_host %ch0, %ctx[1, 0], %cuda_buf, %cpu_buf : !mgp.buffer<, <"cuda", 3>> -> !mgp.buffer<>
  mef.output %ch1 : !mgp.chain
}

//
// mgp.buffer.device_to_device
//

// CHECK-LABEL: @buffer_device_to_device_static
func.func @buffer_device_to_device_static(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %buf0 : !mgp.buffer<48, <"cuda", 3>>, %buf1 : !mgp.buffer<48, <"cuda", 1>>) {
  // CHECK: %{{.*}} = mgp.buffer.device_to_device %{{.*}}, %{{.*}}[1, 2], %{{.*}}, %{{.*}} : !mgp.buffer<48, <"cuda", 3>> -> !mgp.buffer<48, <"cuda", 1>>
  %ch1 = mgp.buffer.device_to_device %ch0, %ctx[1, 2], %buf0, %buf1 : !mgp.buffer<48, <"cuda", 3>> -> !mgp.buffer<48, <"cuda", 1>>
  mef.output %ch1 : !mgp.chain<<"cuda", 1>>
}

// CHECK-LABEL: @buffer_device_to_device_dynamic
func.func @buffer_device_to_device_dynamic(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %buf0 : !mgp.buffer<, <"cuda", 3>>, %buf1 : !mgp.buffer<, <"cuda", 1>>) {
  // CHECK: %{{.*}} = mgp.buffer.device_to_device %{{.*}}, %{{.*}}[1, 2], %{{.*}}, %{{.*}} : !mgp.buffer<, <"cuda", 3>> -> !mgp.buffer<, <"cuda", 1>>
  %ch1 = mgp.buffer.device_to_device %ch0, %ctx[1, 2], %buf0, %buf1 : !mgp.buffer<, <"cuda", 3>> -> !mgp.buffer<, <"cuda", 1>>
  mef.output %ch1 : !mgp.chain<<"cuda", 1>>
}

//
// mgp.buffer.unsize
//

// CHECK-LABEL: @buffer_unsize
func.func @buffer_unsize(%in_buf : !mgp.buffer<64, <"cuda", 3>>) {
  // CHECK:  %{{.*}} = mgp.buffer.unsize %{{.*}} : !mgp.buffer<64, <"cuda", 3>>
  %out_buf = mgp.buffer.unsize %in_buf : !mgp.buffer<64, <"cuda", 3>>
  mef.output %out_buf : !mgp.buffer<, <"cuda", 3>>
}

//
// mgp.buffer.resize
//

// CHECK-LABEL: @buffer_resize
func.func @buffer_resize(%in_buf : !mgp.buffer<, <"cuda", 3>>) {
  // CHECK:  %{{.*}} = mgp.buffer.resize %{{.*}} : !mgp.buffer<64, <"cuda", 3>>
  %out_buf = mgp.buffer.resize %in_buf : !mgp.buffer<64, <"cuda", 3>>
  mef.output %out_buf : !mgp.buffer<64, <"cuda", 3>>
}

//
// mgp.buffer.concat
//

// CHECK-LABEL: @buffer_concat_static
func.func @buffer_concat_static(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %out_buf : !mgp.buffer<192, <"cuda", 3>>, %buf0 : !mgp.buffer<64, <"cuda", 3>>, %buf1 : !mgp.buffer<128, <"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.buffer.concat %{{.*}}, %{{.*}}[1], %{{.*}}, %{{.*}}, %{{.*}} : (!mgp.buffer<64, <"cuda", 3>>, !mgp.buffer<128, <"cuda", 3>>) -> !mgp.buffer<192, <"cuda", 3>>
  %ch1 = mgp.buffer.concat %ch0, %ctx[1], %out_buf, %buf0, %buf1 : (!mgp.buffer<64, <"cuda", 3>>, !mgp.buffer<128, <"cuda", 3>>) -> !mgp.buffer<192, <"cuda", 3>>
  mef.output %ch1 : !mgp.chain<<"cuda", 3>>
}

// CHECK-LABEL: @buffer_concat_dynamic
func.func @buffer_concat_dynamic(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %out_buf : !mgp.buffer<, <"cuda", 3>>, %buf0 : !mgp.buffer<64, <"cuda", 3>>, %buf1 : !mgp.buffer<, <"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.buffer.concat %{{.*}}, %{{.*}}[1], %{{.*}}, %{{.*}}, %{{.*}} : (!mgp.buffer<64, <"cuda", 3>>, !mgp.buffer<, <"cuda", 3>>) -> !mgp.buffer<, <"cuda", 3>>
  %ch1 = mgp.buffer.concat %ch0, %ctx[1], %out_buf, %buf0, %buf1 : (!mgp.buffer<64, <"cuda", 3>>, !mgp.buffer<, <"cuda", 3>>) -> !mgp.buffer<, <"cuda", 3>>
  mef.output %ch1 : !mgp.chain<<"cuda", 3>>
}

//
// mgp.buffer.split.borrow
//

// CHECK-LABEL: @buffer_split_borrow_static
func.func @buffer_split_borrow_static(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %in_buf : !mgp.buffer<192, <"cuda", 3>>) {
  // CHECK: %{{.*}}:2 = mgp.buffer.split.borrow %{{.*}}, %{{.*}}[1], %{{.*}} : !mgp.buffer<192, <"cuda", 3>> -> (!mgp.buffer<64, <"cuda", 3>>, !mgp.buffer<128, <"cuda", 3>>)
  %buf0, %buf1 = mgp.buffer.split.borrow %ch0, %ctx[1], %in_buf : !mgp.buffer<192, <"cuda", 3>> -> (!mgp.buffer<64, <"cuda", 3>>, !mgp.buffer<128, <"cuda", 3>>)
  mef.output %buf0, %buf1 : !mgp.buffer<64, <"cuda", 3>>, !mgp.buffer<128, <"cuda", 3>>
}

// CHECK-LABEL: @buffer_split_borrow_dynamic
func.func @buffer_split_borrow_dynamic(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %in_buf : !mgp.buffer<, <"cuda", 3>>, %size : index) {
  // CHECK: %{{.*}}:2 = mgp.buffer.split.borrow %{{.*}}, %{{.*}}[1], %{{.*}}, %{{.*}} : !mgp.buffer<, <"cuda", 3>> -> (!mgp.buffer<64, <"cuda", 3>>, !mgp.buffer<, <"cuda", 3>>)
  %buf0, %buf1 = mgp.buffer.split.borrow %ch0, %ctx[1], %in_buf, %size : !mgp.buffer<, <"cuda", 3>> -> (!mgp.buffer<64, <"cuda", 3>>, !mgp.buffer<, <"cuda", 3>>)
  mef.output %buf0, %buf1 : !mgp.buffer<64, <"cuda", 3>>, !mgp.buffer<, <"cuda", 3>>
}

//
// mgp.buffer.section.borrow.static
//

// CHECK-LABEL: @buffer_section_borrow_static
func.func @buffer_section_borrow_static(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %in_buf : !mgp.buffer<192, <"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.buffer.section.borrow.static %{{.*}}, %{{.*}}[1], %{{.*}}, offset 16, size 128 : !mgp.buffer<192, <"cuda", 3>>
  %out_buf = mgp.buffer.section.borrow.static %ch0, %ctx[1], %in_buf, offset 16, size 128 : !mgp.buffer<192, <"cuda", 3>>
  mef.output %out_buf : !mgp.buffer<128, <"cuda", 3>>
}

//
// mgp.buffer.section.borrow.dynamic
//

// CHECK-LABEL: @buffer_section_borrow_dynamic
func.func @buffer_section_borrow_dynamic(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %in_buf : !mgp.buffer<, <"cuda", 3>>, %scale : index) {
  // CHECK: %{{.*}} = mgp.buffer.section.borrow.dynamic %{{.*}}, %{{.*}}[1], %{{.*}}, %{{.*}}, offset 16, size 128 : !mgp.buffer<, <"cuda", 3>>
  %out_buf = mgp.buffer.section.borrow.dynamic %ch0, %ctx[1], %in_buf, %scale, offset 16, size 128 : !mgp.buffer<, <"cuda", 3>>
  mef.output %out_buf : !mgp.buffer<, <"cuda", 3>>
}

//
// mgp.buffer.to_bool
//

// CHECK-LABEL: @buffer_to_bool
func.func @buffer_to_bool(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %buf : !mgp.buffer<1, <"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.buffer.to_bool %{{.*}}, %{{.*}}[1], %{{.*}} : !mgp.buffer<1, <"cuda", 3>>
  %b = mgp.buffer.to_bool %ch0, %ctx[1], %buf : !mgp.buffer<1, <"cuda", 3>>
  mef.output %b : i1
}

//
// mgp.buffer.set_with_index
//

// CHECK-LABEL: @buffer_set_with_index
func.func @buffer_set_with_index(%ctx : !mgp.context, %buf: !mgp.buffer<4, <"cuda", 3>>, %idx: index) {
  // CHECK: %{{.*}} = mgp.buffer.set_with_index %{{.*}}[1], %{{.*}}, %{{.*}} : !mgp.buffer<4, <"cuda", 3>>
  %ch = mgp.buffer.set_with_index %ctx[1], %buf, %idx : !mgp.buffer<4, <"cuda", 3>>
  mef.output %ch : !mgp.chain<<"cuda", 3>>
}

// CHECK-LABEL: @buffer_set_with_index_variadic
func.func @buffer_set_with_index_variadic(%ctx :  !mgp.context, %buf: !mgp.buffer<16, <"cuda", 3>>, %idx0: index, %idx1: index) {
  // CHECK: %{{.*}} = mgp.buffer.set_with_index %{{.*}}[1], %{{.*}}, %{{.*}}, %{{.*}} : !mgp.buffer<16, <"cuda", 3>>
  %ch = mgp.buffer.set_with_index %ctx[1], %buf, %idx0, %idx1 : !mgp.buffer<16, <"cuda", 3>>
  mef.output %ch : !mgp.chain<<"cuda", 3>>
}

//
// mgp.tensor.create
//

// CHECK-LABEL: @tensor_create
func.func @tensor_create(%ctx : !mgp.context, %ch : !mgp.chain<<"cuda", 3>>, %buf : !mgp.buffer<60, <"cuda", 3>>, %spec : !mgp.tensor_spec<3x5xf32>) {
  // CHECK: %{{.*}} = mgp.tensor.create %{{.*}}, %{{.*}}[1], %{{.*}}, %{{.*}} : !mgp.tensor<3x5xf32, <"cuda", 3>>
  %0 = mgp.tensor.create %ch, %ctx[1], %buf, %spec : !mgp.tensor<3x5xf32, <"cuda", 3>>
  mef.output %0 : !mgp.tensor<3x5xf32, <"cuda", 3>>
}

//
// mgp.tensor.extract.buffer
//

// CHECK-LABEL: @tensor_extract_buffer
func.func @tensor_extract_buffer(%ctx : !mgp.context, %tensor : !mgp.tensor<3x5xf32, <"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.tensor.extract.buffer %{{.*}}[1], %{{.*}} : !mgp.tensor<3x5xf32, <"cuda", 3>>
  %buf = mgp.tensor.extract.buffer %ctx[1], %tensor : !mgp.tensor<3x5xf32, <"cuda", 3>>
  mef.output %buf : !mgp.buffer<60, <"cuda", 3>>
}

//
// cuda.tensor.extract.tensor_spec
//

// CHECK-LABEL: @tensor_extract_tensor_spec
func.func @tensor_extract_tensor_spec(%ctx : !mgp.context, %tensor : !mgp.tensor<3x?xf32, <"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.tensor.extract.tensor_spec %{{.*}}[1], %{{.*}} : !mgp.tensor<3x?xf32, <"cuda", 3>>
  %shape = mgp.tensor.extract.tensor_spec %ctx[1], %tensor : !mgp.tensor<3x?xf32, <"cuda", 3>>
  mef.output %shape : !mgp.tensor_spec<3x?xf32>
}

// CHECK-LABEL: @tensor_extract_shape
func.func @tensor_extract_shape(%ctx : !mgp.context, %tensor : !mgp.tensor<3x?xf32, <"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.tensor.extract.tensor_spec %{{.*}}[1], %{{.*}} shape_only true : !mgp.tensor<3x?xf32, <"cuda", 3>>
  %shape = mgp.tensor.extract.tensor_spec %ctx[1], %tensor shape_only true : !mgp.tensor<3x?xf32, <"cuda", 3>>
  mef.output %shape : !mgp.tensor_spec<3x?xunk>
}

//
// mgp.debug.tensor.print
//

// CHECK-LABEL: @debug_tensor_print
func.func @debug_tensor_print(%ctx : !mgp.context, %ch0 : !mef.chain, %buf : !mgp.buffer<60, <"cuda", 3>>, %spec : !mgp.tensor_spec<3x5xf32>) {
  // CHECK: %{{.*}} = mgp.debug.tensor.print %{{.*}}, %{{.*}}[1], %{{.*}}, %{{.*}} : !mgp.tensor<3x5xf32, <"cuda", 3>>
  %ch1 = mgp.debug.tensor.print %ch0, %ctx[1], %buf, %spec : !mgp.tensor<3x5xf32, <"cuda", 3>>
  mef.output %ch1 : !mef.chain
}

//
// mgp.kernel.execute
//

// CHECK-LABEL: @kernel_execute
func.func @kernel_execute(
  %ctx : !mgp.context,
  %ch0 : !mgp.chain<<"cuda", 3>>,
  %buf0 : !mgp.buffer<, <"cuda", 3>>,
  %buf1 : !mgp.buffer<8>,
  %buf2 : !mgp.buffer<48, <"cuda", 3>>,
  %dim0 : index,
  %dim1 : index
) {
  // CHECK: %{{.*}} = mgp.kernel.execute %{{.*}}, %{{.*}}[0, 1], (7, 9), (in %{{.*}} : !mgp.buffer<, <"cuda", 3>>, in %{{.*}} : !mgp.buffer<8>, out %{{.*}} : !mgp.buffer<48, <"cuda", 3>>), (%{{.*}}, %{{.*}}), other <"cuda", 3> summary "kernel2" is_async true
  %ch1 = mgp.kernel.execute %ch0, %ctx[0, 1], (7, 9),
            (in %buf0 : !mgp.buffer<, <"cuda", 3>>,
             in %buf1 : !mgp.buffer<8>,
             out %buf2 : !mgp.buffer<48, <"cuda", 3>>),
            (%dim0, %dim1),
            other <"cuda", 3> summary "kernel2" is_async true
  mef.output %ch1 : !mgp.chain<<"cuda", 3>>
}

// CHECK-LABEL: @kernel_execute_out_dims
func.func @kernel_execute_out_dims(
  %ctx : !mgp.context,
  %ch0 : !mgp.chain,
  %buf0 : !mgp.buffer<8>,
  %buf1 : !mgp.buffer<48>,
  %dim0 : index
) {
  // CHECK: %{{.*}}, %{{.*}}:2 = mgp.kernel.execute %{{.*}}, %{{.*}}[0, 0], (7, 9), (in %{{.*}} : !mgp.buffer<8>, out %{{.*}} : !mgp.buffer<48>), (%{{.*}}), summary "kernel2" out_dims 2
  %ch1, %dim1, %dim2 = mgp.kernel.execute %ch0, %ctx[0, 0], (7, 9),
                         (in %buf0 : !mgp.buffer<8>,
                          out %buf1 : !mgp.buffer<48>),
                         (%dim0),
                         summary "kernel2" out_dims 2
  mef.output %ch1, %dim1, %dim2 : !mgp.chain, index, index
}

{-#
  dialect_resources: {
    builtin: {
      resource:
        // First 4 bytes are alignment of 64 in little-endian form, then 4 bytes.
        "0x4000000000010203"
    }
  }
#-}
