// RUN: mgp-opt --allow-unregistered-dialect --canonicalize %s | mgp-opt --allow-unregistered-dialect | FileCheck %s

//
// mgp.runtime.create
//

// CHECK-LABEL: @runtime_create
func.func @runtime_create(%ctx : !mgp.context, %ch0 : !mef.chain) {
  // CHECK: %{{.*}} = mgp.runtime.create %{{.*}}, %{{.*}}[1, 2], <"cuda", 3>
  %ch1 = mgp.runtime.create %ch0, %ctx[1, 2], <"cuda", 3>
  mef.output %ch1 : !mef.chain
}

//
// mgp.module.load
//

// CHECK-LABEL: @module_load_dense_attribute
func.func @module_load_dense_attribute(%ctx : !mgp.context, %ch0 : !mef.chain) {
  // CHECK: %{{.*}} = mgp.module.load %{{.*}}, %{{.*}}[1], dense_resource<resource> : !M.aligned_bytes<4, align 64>, ["f1", "f2"],  ["g1", "g2"], <"cuda", 3>
  %ch1 = mgp.module.load %ch0, %ctx[1], dense_resource<resource> : !M.aligned_bytes<4, align 64>, ["f1", "f2"], ["g1", "g2"], <"cuda", 3>
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: @module_load_multiline
func.func @module_load_multiline(%ctx : !mgp.context, %ch0 : !mef.chain) {
  // CHECK: %{{.*}} = mgp.module.load %{{.*}}, %{{.*}}[1], #M<multiline["a", "b", "c"]>, ["f"], [], <"cuda", 3>
  %ch1 = mgp.module.load %ch0, %ctx[1], #M<multiline["a", "b", "c"]>, ["f"], [], <"cuda", 3>
  mef.output %ch1 : !mef.chain
}

//
// mgp.chain.create
//

// CHECK-LABEL: @chain_create
func.func @chain_create(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %ch1 : !mgp.chain<<"cuda", 3>>, %ch2 : !mgp.chain<<"cpu", 0>>, %ch3 : !mef.chain, %dev_buf : !mgp.buffer<<"cuda", 3>>, %host_buf : !mgp.buffer<<"cpu", 0>>) {
  // CHECK: %{{.*}} = mgp.chain.create %{{.*}}[1], host <"cpu", 0>, other <"cuda", 3>, (%{{.*}}, %{{.*}}, %{{.*}}: !mgp.chain<<"cpu", 0>>, %{{.*}}: !mef.chain, %{{.*}}: !mgp.buffer<<"cuda", 3>>, %{{.*}}: !mgp.buffer<<"cpu", 0>>)
  %ch4 = mgp.chain.create %ctx[1], host <"cpu", 0>, other <"cuda", 3>, (%ch0, %ch1, %ch2: !mgp.chain<<"cpu", 0>>, %ch3: !mef.chain, %dev_buf: !mgp.buffer<<"cuda", 3>>, %host_buf: !mgp.buffer<<"cpu", 0>>)
  mef.output %ch4 : !mgp.chain<<"cuda", 3>>
}

//
// mgp.chain.device_to_host
//

// CHECK-LABEL: @chain_device_to_host
func.func @chain_device_to_host(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.chain.device_to_host %{{.*}}, %{{.*}}[1], <"cuda", 3>
  %ch1 = mgp.chain.device_to_host %ch0, %ctx[1], <"cuda", 3>
  mef.output %ch1 : !mef.chain
}

//
// mgp.chain.host_to_device
//

// CHECK-LABEL: @chain_host_to_device
func.func @chain_host_to_device(%ctx : !mgp.context, %ch0 : !mef.chain) {
  // CHECK: %{{.*}} = mgp.chain.host_to_device %{{.*}}, %{{.*}}[1], <"cuda", 3>
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1], <"cuda", 3>
  mef.output %ch1 : !mgp.chain<<"cuda", 3>>
}

// CHECK-LABEL:@chain_canonicalization_1
func.func @chain_canonicalization_1(%ctx : !mgp.context, %ch0 : !mef.chain) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1], <"cuda", 3>
  %ch2 = mgp.chain.device_to_host %ch1, %ctx[1], <"cuda", 3>
  // CHECK-NEXT: mef.output
  mef.output %ch2 : !mef.chain
}

// CHECK-LABEL:@chain_canonicalization_2
func.func @chain_canonicalization_2(%ctx : !mgp.context, %ch0 : !mef.chain) {
  // CHECK-NEXT: %[[CH1:.*]] = mgp.chain.host_to_device %{{.*}}, %{{.*}}[1], <"cuda", 3>
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1], <"cuda", 3>
  %ch2 = mgp.chain.device_to_host %ch1, %ctx[1], <"cuda", 3>
  %ch3 = mgp.chain.host_to_device %ch2, %ctx[1], <"cuda", 3>
  // CHECK-NEXT: "some.op"(%[[CH1]], %[[CH1]])
  "some.op"(%ch1, %ch3) : (!mgp.chain<<"cuda", 3>>, !mgp.chain<<"cuda", 3>>) -> ()
  mef.output %ch0 : !mef.chain
}

//
// mgp.buffer.constant
//

// CHECK-LABEL: @buffer_constant
func.func @buffer_constant(%ctx : !mgp.context) {
  // CHECK: %{{.*}} = mgp.buffer.constant %{{.*}}[1], dense_resource<resource> : !M.aligned_bytes<4, align 64>, <"cuda", 3>
  %buf = mgp.buffer.constant %ctx[1], dense_resource<resource> : !M.aligned_bytes<4, align 64>, <"cuda", 3>
  mef.output %buf : !mgp.buffer<<"cuda", 3>, 4>
}

//
// mgp.buffer.constant.load
//

// CHECK-LABEL: @buffer_constant_load
func.func @buffer_constant_load(%ctx : !mgp.context) {
  // CHECK: %{{.*}} = mgp.buffer.constant.load %{{.*}}[1], (3, 7) : !mgp.buffer<<"cuda", 3>, 124>
  %buf = mgp.buffer.constant.load %ctx[1], (3, 7) : !mgp.buffer<<"cuda", 3>, 124>
  mef.output %buf : !mgp.buffer<<"cuda", 3>, 124>
}

//
// mgp.buffer.alloc.static
//

// CHECK-LABEL: @buffer_alloc_static
func.func @buffer_alloc_static(%ctx : !mgp.context, %ch : !mgp.chain<<"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.buffer.alloc.static %{{.*}}, %{{.*}}[1] : !mgp.buffer<<"cuda", 3>, 124>
  %buf = mgp.buffer.alloc.static %ch, %ctx[1] : !mgp.buffer<<"cuda", 3>, 124>
  mef.output %buf : !mgp.buffer<<"cuda", 3>, 124>
}

//
// mgp.buffer.alloc.dynamic
//

// CHECK-LABEL: @buffer_alloc_dynamic
func.func @buffer_alloc_dynamic(%ctx : !mgp.context, %ch : !mgp.chain<<"cuda", 3>>, %size : index) {
  // CHECK: %{{.*}} = mgp.buffer.alloc.dynamic %{{.*}}, %{{.*}}[1], %{{.*}} : !mgp.buffer<<"cuda", 3>>
  %buf = mgp.buffer.alloc.dynamic %ch, %ctx[1], %size : !mgp.buffer<<"cuda", 3>>
  mef.output %buf : !mgp.buffer<<"cuda", 3>>
}

//
// mgp.buffer.realloc.static
//

// CHECK-LABEL: @buffer_realloc_static
func.func @buffer_realloc_static_size(%ctx : !mgp.context, %ch : !mgp.chain<<"cuda", 3>>, %size : index) {
  // CHECK:  %{{.*}} = mgp.buffer.realloc.static %{{.*}}, %{{.*}}[1, 3, 0] : !mgp.buffer<<"cuda", 3>, 256>
  %0 = mgp.buffer.realloc.static %ch, %ctx[1, 3, 0] : !mgp.buffer<<"cuda", 3>, 256>
  mef.output %0 : !mgp.buffer<<"cuda", 3>, 256>
}

//
// mgp.buffer.realloc.dynamic
//

// CHECK-LABEL: @buffer_realloc_dynamic
func.func @buffer_realloc_dynamic_size(%ctx : !mgp.context, %ch : !mgp.chain<<"cuda", 3>>, %size : index) {
  // CHECK:  %{{.*}} = mgp.buffer.realloc.dynamic %{{.*}}, %{{.*}}[1, 3, 0], %{{.*}} : !mgp.buffer<<"cuda", 3>>
  %0 = mgp.buffer.realloc.dynamic %ch, %ctx[1, 3, 0], %size : !mgp.buffer<<"cuda", 3>>
  mef.output %0 : !mgp.buffer<<"cuda", 3>>
}

//
// mgp.buffer.cache
//

// CHECK-LABEL: @buffer_cache
func.func @buffer_cache(%ctx : !mgp.context, %ch0 : !mef.chain, %buf : !mgp.buffer<<"cuda", 3>, 256>) {
  // CHECK: %{{.*}} = mgp.buffer.cache %{{.*}}, %{{.*}}[1, 5], %{{.*}} : !mgp.buffer<<"cuda", 3>, 256>
  %ch1 = mgp.buffer.cache %ch0, %ctx[1, 5], %buf : !mgp.buffer<<"cuda", 3>, 256>
  mef.output %ch1 : !mef.chain
}

//
// mgp.buffer.get_cached
//

// CHECK-LABEL: @buffer_get_cached
func.func @buffer_get_cached(%ctx : !mgp.context, %ch0 : !mef.chain) {
  // CHECK: %{{.*}} = mgp.buffer.get_cached %{{.*}}, %{{.*}}[1, 5] : !mgp.buffer<<"cuda", 3>, 256>
  %buf = mgp.buffer.get_cached %ch0, %ctx[1, 5] : !mgp.buffer<<"cuda", 3>, 256>
  mef.output %buf : !mgp.buffer<<"cuda", 3>, 256>
}

//
// mgp.buffer.host_to_device
//

// CHECK-LABEL: @buffer_host_to_device_static
func.func @buffer_host_to_device_static(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cpu", 0>>, %cpu_buf : !mgp.buffer<<"cpu", 0>, 48>, %cuda_buf : !mgp.buffer<<"cuda", 3>, 48>) {
  // CHECK: %{{.*}} = mgp.buffer.host_to_device %{{.*}}, %{{.*}}[0, 1], %{{.*}}, %{{.*}} : !mgp.buffer<<"cpu", 0>, 48> -> !mgp.buffer<<"cuda", 3>, 48>
  %ch1 = mgp.buffer.host_to_device %ch0, %ctx[0, 1], %cpu_buf, %cuda_buf : !mgp.buffer<<"cpu", 0>, 48> -> !mgp.buffer<<"cuda", 3>, 48>
  mef.output %ch1 : !mgp.chain<<"cuda", 3>>
}

// CHECK-LABEL: @buffer_host_to_device_dynamic
func.func @buffer_host_to_device_dynamic(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cpu", 0>>, %cpu_buf : !mgp.buffer<<"cpu", 0>>, %cuda_buf : !mgp.buffer<<"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.buffer.host_to_device %{{.*}}, %{{.*}}[0, 1], %{{.*}}, %{{.*}} : !mgp.buffer<<"cpu", 0>> -> !mgp.buffer<<"cuda", 3>>
  %ch1 = mgp.buffer.host_to_device %ch0, %ctx[0, 1], %cpu_buf, %cuda_buf : !mgp.buffer<<"cpu", 0>> -> !mgp.buffer<<"cuda", 3>>
  mef.output %ch1 : !mgp.chain<<"cuda", 3>>
}

//
// mgp.buffer.device_to_host
//

// CHECK-LABEL: @buffer_device_to_host_static
func.func @buffer_device_to_host_static(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %cuda_buf : !mgp.buffer<<"cuda", 3>, 48>, %cpu_buf : !mgp.buffer<<"cpu", 0>, 48>) {
  // CHECK: %{{.*}} = mgp.buffer.device_to_host %{{.*}}, %{{.*}}[1, 0], %{{.*}}, %{{.*}} : !mgp.buffer<<"cuda", 3>, 48> -> !mgp.buffer<<"cpu", 0>, 48>
  %ch1 = mgp.buffer.device_to_host %ch0, %ctx[1, 0], %cuda_buf, %cpu_buf : !mgp.buffer<<"cuda", 3>, 48> -> !mgp.buffer<<"cpu", 0>, 48>
  mef.output %ch1 : !mgp.chain<<"cpu", 0>>
}

// CHECK-LABEL: @buffer_device_to_host_dynamic
func.func @buffer_device_to_host_dynamic(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %cuda_buf : !mgp.buffer<<"cuda", 3>>, %cpu_buf : !mgp.buffer<<"cpu", 0>>) {
  // CHECK: %{{.*}} = mgp.buffer.device_to_host %{{.*}}, %{{.*}}[1, 0], %{{.*}}, %{{.*}} : !mgp.buffer<<"cuda", 3>> -> !mgp.buffer<<"cpu", 0>>
  %ch1 = mgp.buffer.device_to_host %ch0, %ctx[1, 0], %cuda_buf, %cpu_buf : !mgp.buffer<<"cuda", 3>> -> !mgp.buffer<<"cpu", 0>>
  mef.output %ch1 : !mgp.chain<<"cpu", 0>>
}

//
// mgp.buffer.device_to_device
//

// CHECK-LABEL: @buffer_device_to_device_static
func.func @buffer_device_to_device_static(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %buf0 : !mgp.buffer<<"cuda", 3>, 48>, %buf1 : !mgp.buffer<<"cuda", 3>, 48>) {
  // CHECK: %{{.*}} = mgp.buffer.device_to_device %{{.*}}, %{{.*}}[1], %{{.*}}, %{{.*}} : !mgp.buffer<<"cuda", 3>, 48> -> !mgp.buffer<<"cuda", 3>, 48>
  %ch1 = mgp.buffer.device_to_device %ch0, %ctx[1], %buf0, %buf1 : !mgp.buffer<<"cuda", 3>, 48> -> !mgp.buffer<<"cuda", 3>, 48>
  mef.output %ch1 : !mgp.chain<<"cuda", 3>>
}

// CHECK-LABEL: @buffer_device_to_device_dynamic
func.func @buffer_device_to_device_dynamic(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %buf0 : !mgp.buffer<<"cuda", 3>>, %buf1 : !mgp.buffer<<"cuda", 3>>) {
  // CHECK: %{{.*}} = mgp.buffer.device_to_device %{{.*}}, %{{.*}}[1], %{{.*}}, %{{.*}} : !mgp.buffer<<"cuda", 3>> -> !mgp.buffer<<"cuda", 3>>
  %ch1 = mgp.buffer.device_to_device %ch0, %ctx[1], %buf0, %buf1 : !mgp.buffer<<"cuda", 3>> -> !mgp.buffer<<"cuda", 3>>
  mef.output %ch1 : !mgp.chain<<"cuda", 3>>
}

//
// mgp.buffer.unsize
//

// CHECK-LABEL: @buffer_unsize
func.func @buffer_unsize(%in_buf : !mgp.buffer<<"cuda", 3>, 64>) {
  // CHECK:  %{{.*}} = mgp.buffer.unsize %{{.*}} : !mgp.buffer<<"cuda", 3>, 64>
  %out_buf = mgp.buffer.unsize %in_buf : !mgp.buffer<<"cuda", 3>, 64>
  mef.output %out_buf : !mgp.buffer<<"cuda", 3>>
}

//
// mgp.buffer.split.borrow
//

// CHECK-LABEL: @buffer_split_borrow_static
func.func @buffer_split_borrow_static(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %in_buf : !mgp.buffer<<"cuda", 3>, 192>) {
  // CHECK: %{{.*}}:2 = mgp.buffer.split.borrow %{{.*}}, %{{.*}}[1], %{{.*}} : !mgp.buffer<<"cuda", 3>, 192> -> (!mgp.buffer<<"cuda", 3>, 64>, !mgp.buffer<<"cuda", 3>, 128>)
  %buf0, %buf1 = mgp.buffer.split.borrow %ch0, %ctx[1], %in_buf : !mgp.buffer<<"cuda", 3>, 192> -> (!mgp.buffer<<"cuda", 3>, 64>, !mgp.buffer<<"cuda", 3>, 128>)
  mef.output %buf0, %buf1 : !mgp.buffer<<"cuda", 3>, 64>, !mgp.buffer<<"cuda", 3>, 128>
}

// CHECK-LABEL: @buffer_split_borrow_dynamic
func.func @buffer_split_borrow_dynamic(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %in_buf : !mgp.buffer<<"cuda", 3>>, %size : index) {
  // CHECK: %{{.*}}:2 = mgp.buffer.split.borrow %{{.*}}, %{{.*}}[1], %{{.*}}, %{{.*}} : !mgp.buffer<<"cuda", 3>> -> (!mgp.buffer<<"cuda", 3>, 64>, !mgp.buffer<<"cuda", 3>>)
  %buf0, %buf1 = mgp.buffer.split.borrow %ch0, %ctx[1], %in_buf, %size : !mgp.buffer<<"cuda", 3>> -> (!mgp.buffer<<"cuda", 3>, 64>, !mgp.buffer<<"cuda", 3>>)
  mef.output %buf0, %buf1 : !mgp.buffer<<"cuda", 3>, 64>, !mgp.buffer<<"cuda", 3>>
}

//
// mgp.buffer.section.borrow.static
//

// CHECK-LABEL: @buffer_section_borrow_static
func.func @buffer_section_borrow_static(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %in_buf : !mgp.buffer<<"cuda", 3>, 192>) {
  // CHECK: %{{.*}} = mgp.buffer.section.borrow.static %{{.*}}, %{{.*}}[1], %{{.*}}, offset 16, size 128 : !mgp.buffer<<"cuda", 3>, 192>
  %out_buf = mgp.buffer.section.borrow.static %ch0, %ctx[1], %in_buf, offset 16, size 128 : !mgp.buffer<<"cuda", 3>, 192>
  mef.output %out_buf : !mgp.buffer<<"cuda", 3>, 128>
}

//
// mgp.buffer.section.borrow.dynamic
//

// CHECK-LABEL: @buffer_section_borrow_dynamic
func.func @buffer_section_borrow_dynamic(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %in_buf : !mgp.buffer<<"cuda", 3>>, %scale : index) {
  // CHECK: %{{.*}} = mgp.buffer.section.borrow.dynamic %{{.*}}, %{{.*}}[1], %{{.*}}, %{{.*}}, offset 16, size 128 : !mgp.buffer<<"cuda", 3>>
  %out_buf = mgp.buffer.section.borrow.dynamic %ch0, %ctx[1], %in_buf, %scale, offset 16, size 128 : !mgp.buffer<<"cuda", 3>>
  mef.output %out_buf : !mgp.buffer<<"cuda", 3>>
}

//
// mgp.tensor.create
//

// CHECK-LABEL: @tensor_create
func.func @tensor_create(%ctx : !mgp.context, %ch : !mgp.chain<<"cuda", 3>>, %buf : !mgp.buffer<<"cuda", 3>, 60>, %spec : !mgp.tensor_spec<3x5xf32>) {
  // CHECK: %{{.*}} = mgp.tensor.create %{{.*}}, %{{.*}}[1], %{{.*}}, %{{.*}} : !mgp.tensor<<"cuda", 3>, 3x5xf32>
  %0 = mgp.tensor.create %ch, %ctx[1], %buf, %spec : !mgp.tensor<<"cuda", 3>, 3x5xf32>
  mef.output %0 : !mgp.tensor<<"cuda", 3>, 3x5xf32>
}

//
// mgp.tensor.extract.buffer
//

// CHECK-LABEL: @tensor_extract_buffer
func.func @tensor_extract_buffer(%ctx : !mgp.context, %tensor : !mgp.tensor<<"cuda", 3>, 3x5xf32>) {
  // CHECK: %{{.*}} = mgp.tensor.extract.buffer %{{.*}}[1], %{{.*}} : !mgp.tensor<<"cuda", 3>, 3x5xf32>
  %buf = mgp.tensor.extract.buffer %ctx[1], %tensor : !mgp.tensor<<"cuda", 3>, 3x5xf32>
  mef.output %buf : !mgp.buffer<<"cuda", 3>, 60>
}

//
// cuda.tensor.extract.tensor_spec
//

// CHECK-LABEL: @tensor_extract_tensor_spec
func.func @tensor_extract_tensor_spec(%ctx : !mgp.context, %tensor : !mgp.tensor<<"cuda", 3>, 3x?xf32>) {
  // CHECK: %{{.*}} = mgp.tensor.extract.tensor_spec %{{.*}}[1], %{{.*}} : !mgp.tensor<<"cuda", 3>, 3x?xf32>
  %shape = mgp.tensor.extract.tensor_spec %ctx[1], %tensor : !mgp.tensor<<"cuda", 3>, 3x?xf32>
  mef.output %shape : !mgp.tensor_spec<3x?xf32>
}

// CHECK-LABEL: @tensor_extract_shape
func.func @tensor_extract_shape(%ctx : !mgp.context, %tensor : !mgp.tensor<<"cuda", 3>, 3x?xf32>) {
  // CHECK: %{{.*}} = mgp.tensor.extract.tensor_spec %{{.*}}[1], %{{.*}}, true : !mgp.tensor<<"cuda", 3>, 3x?xf32>
  %shape = mgp.tensor.extract.tensor_spec %ctx[1], %tensor, true : !mgp.tensor<<"cuda", 3>, 3x?xf32>
  mef.output %shape : !mgp.tensor_spec<3x?xunk>
}

//
// mgp.debug.tensor.print
//

// CHECK-LABEL: @debug_tensor_print
func.func @debug_tensor_print(%ctx : !mgp.context, %ch0 : !mef.chain, %buf : !mgp.buffer<<"cuda", 3>, 60>, %spec : !mgp.tensor_spec<3x5xf32>) {
  // CHECK: %{{.*}} = mgp.debug.tensor.print %{{.*}}, %{{.*}}[1], %{{.*}}, %{{.*}} : !mgp.tensor<<"cuda", 3>, 3x5xf32>
  %ch1 = mgp.debug.tensor.print %ch0, %ctx[1], %buf, %spec : !mgp.tensor<<"cuda", 3>, 3x5xf32>
  mef.output %ch1 : !mef.chain
}

//
// mgp.kernel.execute.generic
//

// CHECK-LABEL: @kernel_execute_generic
func.func @kernel_execute_generic(
  %ctx : !mgp.context,
  %ch0 : !mgp.chain<<"cuda", 3>>,
  %buf0 : !mgp.buffer<<"cuda", 3>>,
  %buf1 : !mgp.buffer<<"cpu", 0>, 8>,
  %buf2 : !mgp.buffer<<"cuda", 3>, 48>,
  %dim0 : index,
  %dim1 : index
) {
  // CHECK: %{{.*}} = mgp.kernel.execute.generic %{{.*}}, %{{.*}}[0, 1], (7, 9), (in %{{.*}} : !mgp.buffer<<"cuda", 3>>, in %{{.*}} : !mgp.buffer<<"cpu", 0>, 8>, out %{{.*}} : !mgp.buffer<<"cuda", 3>, 48>), (%{{.*}}, %{{.*}}), host <"cpu", 0> other <"cuda", 3> summary "kernel2" is_async true
  %ch1 = mgp.kernel.execute.generic %ch0, %ctx[0, 1], (7, 9),
            (in %buf0 : !mgp.buffer<<"cuda", 3>>,
             in %buf1 : !mgp.buffer<<"cpu", 0>, 8>,
             out %buf2 : !mgp.buffer<<"cuda", 3>, 48>),
            (%dim0, %dim1),
            host <"cpu", 0> other <"cuda", 3> summary "kernel2" is_async true
  mef.output %ch1 : !mgp.chain<<"cuda", 3>>
}

// CHECK-LABEL: @kernel_execute_generic_out_dims
func.func @kernel_execute_generic_out_dims(
  %ctx : !mgp.context,
  %ch0 : !mgp.chain<<"cpu", 0>>,
  %buf0 : !mgp.buffer<<"cpu", 0>, 8>,
  %buf1 : !mgp.buffer<<"cpu", 0>, 48>,
  %dim0 : index
) {
  // CHECK: %{{.*}}, %{{.*}}:2 = mgp.kernel.execute.generic %{{.*}}, %{{.*}}[0, 0], (7, 9), (in %{{.*}} : !mgp.buffer<<"cpu", 0>, 8>, out %{{.*}} : !mgp.buffer<<"cpu", 0>, 48>), (%{{.*}}), host <"cpu", 0> other <"cpu", 0> summary "kernel2" out_dims 2
  %ch1, %dim1, %dim2 = mgp.kernel.execute.generic %ch0, %ctx[0, 0], (7, 9),
                         (in %buf0 : !mgp.buffer<<"cpu", 0>, 8>,
                          out %buf1 : !mgp.buffer<<"cpu", 0>, 48>),
                         (%dim0),
                         host <"cpu", 0> other <"cpu", 0> summary "kernel2" out_dims 2
  mef.output %ch1, %dim1, %dim2 : !mgp.chain<<"cpu", 0>>, index, index
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
