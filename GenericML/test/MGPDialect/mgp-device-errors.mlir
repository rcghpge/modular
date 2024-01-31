// RUN: mgp-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect

//
// mgp.chain.device_to_host
//

func.func @chain_device_to_host_mismatched_devices(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cpu", 0>>) {
  // expected-error@+1 {{'mgp.chain.device_to_host' op expected all operands, results, and attributes to use the device(s) {#M.device_ref<"cuda", 3>}, but they instead use the devices(s) {#M.device_ref<"cuda", 3>, #M.device_ref<"cpu", 0>}.}}
  %ch1 = "mgp.chain.device_to_host" (%ch0, %ctx) { aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cuda", 3> }
           : (!mgp.chain<<"cpu", 0>>, !mgp.context) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// -----

//
// mgp.buffer.constant
//

func.func @buffer_constant_mismatched_size(%ctx : !mgp.context) {
  // expected-error@+1 {{'mgp.buffer.constant' op expected result buffer type does not match actual result buffer type ('!mgp.buffer<<"cuda", 3>, 3>' vs '!mgp.buffer<<"cuda", 3>, 4>').}}
  %0 = "mgp.buffer.constant" (%ctx) { aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x000102", align 32>, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cuda", 3> }
         : (!mgp.context) -> !mgp.buffer<<"cuda", 3>, 4>
  mef.output %0 : !mgp.buffer<<"cuda", 3>, 4>
}

// -----

func.func @buffer_constant_invalid_elem_type(%ctx : !mgp.context) {
  // expected-error@+1 {{'mgp.buffer.constant' op only supports attributes of uint8_t elements (have 'i16').}}
  %0 = "mgp.buffer.constant" (%ctx) { aRuntimeSlot = 1 : ui64, bValue = #M.dense_array<1, 2, 3, 4> : !M.array<4xi16>, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cuda", 3> }
         : (!mgp.context) -> !mgp.buffer<<"cuda", 3>, 4>
  mef.output %0 : !mgp.buffer<<"cuda", 3>, 4>
}

// -----

//
// mgp.buffer.alloc.static
//

func.func @buffer_alloc_static_mismatched_size(%ctx : !mgp.context, %ch : !mgp.chain<<"cuda", 3>>) {
  // expected-error@+1 {{'mgp.buffer.alloc.static' op expected result buffer type does not match actual result buffer type ('!mgp.buffer<<"cuda", 3>, 125>' vs '!mgp.buffer<<"cuda", 3>, 124>').}}
  %buf = "mgp.buffer.alloc.static" (%ch, %ctx) { aRuntimeSlot = 1 : ui64, bSize = 125 : ui64, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cuda", 3> }
           : (!mgp.chain<<"cuda", 3>>, !mgp.context) -> !mgp.buffer<<"cuda", 3>, 124>
  mef.output %buf : !mgp.buffer<<"cuda", 3>, 124>
}

// -----

//
// mgp.buffer.host_to_device
//

func.func @buffer_host_to_device_mismatched_size(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cpu", 0>>, %cpu_buf : !mgp.buffer<<"cpu", 0>, 48>, %cuda_buf : !mgp.buffer<<"cuda", 3>, 64>) {
  // expected-error@+1 {{'mgp.buffer.host_to_device' op host and other buffer sizes do not match (48 vs 64).}}
  %ch1 = "mgp.buffer.host_to_device" (%ch0, %ctx, %cpu_buf, %cuda_buf) { aHostRuntimeSlot = 0 : ui64, bOtherRuntimeSlot = 1 : ui64, cHostDevice = #M.device_ref<"cpu", 0>, dOtherDevice = #M.device_ref<"cuda", 3> }
           : (!mgp.chain<<"cpu", 0>>, !mgp.context, !mgp.buffer<<"cpu", 0>, 48>, !mgp.buffer<<"cuda", 3>, 64>) -> !mgp.chain<<"cuda", 3>>
  mef.output %ch1 : !mgp.chain<<"cuda", 3>>
}

// -----

//
// mgp.buffer.unsize
//

func.func @buffer_unsize(%in_buf : !mgp.buffer<<"cuda", 3>, 64>) {
  // expected-error@+1 {{'mgp.buffer.unsize' op expected output buffer to be dynamically sized.}}
  %out_buf = "mgp.buffer.unsize" (%in_buf) : (!mgp.buffer<<"cuda", 3>, 64>) -> !mgp.buffer<<"cuda", 3>, 32>
  mef.output %out_buf : !mgp.buffer<<"cuda", 3>, 32>
}

// -----

//
// mgp.buffer.split.borrow
//

func.func @buffer_split_mismatched_sizes(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %in_buf : !mgp.buffer<<"cuda", 3>, 192>) {
  // expected-error@+1 {{'mgp.buffer.split.borrow' op expected input buffer type does not match actual input buffer type ('!mgp.buffer<<"cuda", 3>, 320>' vs '!mgp.buffer<<"cuda", 3>, 192>').}}
  %buf0, %buf1 = "mgp.buffer.split.borrow" (%ch0, %ctx, %in_buf) { aRuntimeSlot = 1 : ui64, bOutBufferRawSizes = #M.dense_array<64, 256> : !M.array<2xui64>, cDevice = #M.device_ref<"cuda", 3> }
                   : (!mgp.chain<<"cuda", 3>>, !mgp.context, !mgp.buffer<<"cuda", 3>, 192>) -> (!mgp.buffer<<"cuda", 3>, 64>, !mgp.buffer<<"cuda", 3>, 256>)
  mef.output %buf0, %buf1 : !mgp.buffer<<"cuda", 3>, 64>, !mgp.buffer<<"cuda", 3>, 256>
}

// -----

//
// mgp.buffer.section.borrow.static
//

func.func @buffer_section_borrow_static_beyond_end(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %in_buf : !mgp.buffer<<"cuda", 3>, 192>) {
  // expected-error@+1 {{'mgp.buffer.section.borrow.static' op requesting section end 256 which is beyond end of input buffer of size 192.}}
  %buf = "mgp.buffer.section.borrow.static" (%ch0, %ctx, %in_buf) { aRuntimeSlot = 1 : ui64, bOffset = 128 : ui64, cSize = 128 : ui64, dDevice = #M.device_ref<"cuda", 3> }
           : (!mgp.chain<<"cuda", 3>>, !mgp.context, !mgp.buffer<<"cuda", 3>, 192>) -> !mgp.buffer<<"cuda", 3>, 128>
  mef.output %buf : !mgp.buffer<<"cuda", 3>, 128>
}

// -----

//
// mgp.cpu.buffer.section.borrow.dynamic
//

func.func @buffer_section_borrow_dynamic_sized(%ctx : !mgp.context, %ch0 : !mgp.chain<<"cuda", 3>>, %in_buf : !mgp.buffer<<"cuda", 3>, 192>, %scale : index) {
  // expected-error@+1 {{'mgp.buffer.section.borrow.dynamic' op expected input buffer to be dynamically sized.}}
  %buf = "mgp.buffer.section.borrow.dynamic" (%ch0, %ctx, %in_buf, %scale) { aRuntimeSlot = 1 : ui64, bUnscaledOffset = 128 : ui64, cUnscaledSize = 128 : ui64, dDevice = #M.device_ref<"cuda", 3> }
           : (!mgp.chain<<"cuda", 3>>, !mgp.context, !mgp.buffer<<"cuda", 3>, 192>, index) -> !mgp.buffer<<"cuda", 3>>
  mef.output %buf : !mgp.buffer<<"cuda", 3>>
}

// -----

//
// mgp.tensor.create
//

func.func @tensor_create_mismatched_size(%ctx : !mgp.context, %ch : !mgp.chain<<"cuda", 3>>, %buf : !mgp.buffer<<"cuda", 3>, 61>, %spec : !mgp.tensor_spec<3x5xf32>) {
  // expected-error@+1 {{'mgp.tensor.create' op expected buffer type to match type implied by tensor spec ('!mgp.buffer<<"cuda", 3>, 61>' vs '!mgp.buffer<<"cuda", 3>, 60>').}}
  %tensor = "mgp.tensor.create" (%ch, %ctx, %buf, %spec) { aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cuda", 3> }
              : (!mgp.chain<<"cuda", 3>>, !mgp.context, !mgp.buffer<<"cuda", 3>, 61>, !mgp.tensor_spec<3x5xf32>) -> !mgp.tensor<<"cuda", 3>, 3x5xf32>
  mef.output %tensor : !mgp.tensor<<"cuda", 3>, 3x5xf32>
}

// -----

//
// mgp.tensor.extract.buffer
//

func.func @tensor_extract_buffer_mismatched_size(%ctx : !mgp.context, %tensor : !mgp.tensor<<"cuda", 3>, 3x5xf32>) {
  // expected-error@+1 {{'mgp.tensor.extract.buffer' op expected the result buffer to match the type implied by the tensor type ('!mgp.buffer<<"cuda", 3>, 61>' vs '!mgp.buffer<<"cuda", 3>, 60>').}}
  %buf = "mgp.tensor.extract.buffer" (%ctx, %tensor) { aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cuda", 3> }
           : (!mgp.context, !mgp.tensor<<"cuda", 3>, 3x5xf32>) -> !mgp.buffer<<"cuda", 3>, 61>
  mef.output %buf : !mgp.buffer<<"cuda", 3>, 61>
}

// -----

//
// mgp.kernel.execute.generic
//

func.func @kernel_execute_generic_mismatched_arity(
  %ctx : !mgp.context,
  %ch0 : !mgp.chain<<"cuda", 3>>,
  %buf0 : !mgp.buffer<<"cuda", 3>>,
  %buf1 : !mgp.buffer<<"cpu", 0>, 8>,
  %buf2 : !mgp.buffer<<"cuda", 3>, 48>,
  %dim0 : index,
  %dim1 : index
) {
  // expected-error@+1 {{'mgp.kernel.execute.generic' op kernel has 3 arguments, but the in/out signature attribute has arity 4.}}
  %ch1 = "mgp.kernel.execute.generic" (%ch0, %ctx, %buf0, %buf1, %buf2, %dim0, %dim1)
         { aHostRuntimeSlot = 0 : ui64, bOtherRuntimeSlot = 1 : ui64,
           cHostModuleIndex = 7 : ui64, dHostKernelIndex = 9 : ui64,
           eInOutSignature = #M.inout_sig<"iioo">,
           fHostDevice = #M.device_ref<"cpu", 0>,
           gOtherDevice = #M.device_ref<"cuda", 3>,
           hSummary = "kernel2",
           iNumOutDimensions = 0 : ui64,
           jIsAsync = true,
           operandSegmentSizes = array<i32: 1, 1, 3, 2>}
         : (!mgp.chain<<"cuda", 3>>, !mgp.context, !mgp.buffer<<"cuda", 3>>,
            !mgp.buffer<<"cpu", 0>, 8>, !mgp.buffer<<"cuda", 3>, 48>,
            index, index) -> !mgp.chain<<"cuda", 3>>
  mef.output %ch1 : !mgp.chain<<"cuda", 3>>
}
