// RUN: mo-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect

//
// mgp.buffer.constant
//

func.func @buffer_constant_mismatched_size(%ch: !mef.chain, %ctx : !mgp.context) {
  // expected-error@+1 {{'mgp.buffer.constant' op expected result buffer type does not match actual result buffer type ('!mgp.buffer<3, <"cuda", 3>>' vs '!mgp.buffer<4, <"cuda", 3>>').}}
  %0 = "mgp.buffer.constant" (%ch, %ctx) { aRuntimeSlot = 1 : ui64, bValue = #M.aligned_bytes<"0x000102", align 32>, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cuda", 3> }
         : (!mef.chain, !mgp.context) -> !mgp.buffer<4, <"cuda", 3>>
  mef.output %0 : !mgp.buffer<4, <"cuda", 3>>
}

// -----

func.func @buffer_constant_invalid_elem_type(%ch: !mef.chain, %ctx : !mgp.context) {
  // expected-error@+1 {{'mgp.buffer.constant' op only supports attributes of uint8_t elements (have 'i16').}}
  %0 = "mgp.buffer.constant" (%ch, %ctx) { aRuntimeSlot = 1 : ui64, bValue = #M.dense_array<1, 2, 3, 4> : !M.array<4xi16>, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cuda", 3> }
         : (!mef.chain, !mgp.context) -> !mgp.buffer<4, <"cuda", 3>>
  mef.output %0 : !mgp.buffer<4, <"cuda", 3>>
}

// -----

//
// mgp.buffer.alloc.static
//

func.func @buffer_alloc_static_mismatched_size(%ctx : !mgp.context, %ch : !mef.chain) {
  // expected-error@+1 {{'mgp.buffer.alloc.static' op expected result buffer type does not match actual result buffer type ('!mgp.buffer<125, <"cuda", 3>>' vs '!mgp.buffer<124, <"cuda", 3>>').}}
  %buf = "mgp.buffer.alloc.static" (%ch, %ctx) { aRuntimeSlot = 1 : ui64, bSize = 125 : ui64, cRawAlign = 32 : ui64, dDevice = #M.device_ref<"cuda", 3> }
           : (!mef.chain, !mgp.context) -> !mgp.buffer<124, <"cuda", 3>>
  mef.output %buf : !mgp.buffer<124, <"cuda", 3>>
}

// -----

//
// mgp.buffer.host_to_device
//

func.func @buffer_host_to_device_mismatched_size(%ctx : !mgp.context, %ch0 : !mef.chain, %cpu_buf : !mgp.buffer<48>, %cuda_buf : !mgp.buffer<64, <"cuda", 3>>) {
  // expected-error@+1 {{'mgp.buffer.host_to_device' op host and other buffer sizes do not match (48 vs 64).}}
  %ch1 = "mgp.buffer.host_to_device" (%ch0, %ctx, %cpu_buf, %cuda_buf) { aHostRuntimeSlot = 0 : ui64, bOtherRuntimeSlot = 1 : ui64, cHostDevice = #M.device_ref<"cpu", 0>, dOtherDevice = #M.device_ref<"cuda", 3> }
           : (!mef.chain, !mgp.context, !mgp.buffer<48>, !mgp.buffer<64, <"cuda", 3>>) -> !mef.chain
  mef.output
}

// -----

//
// mgp.buffer.unsize
//

func.func @buffer_unsize(%in_buf : !mgp.buffer<64, <"cuda", 3>>) {
  // expected-error@+1 {{'mgp.buffer.unsize' op expected output buffer to be dynamically sized.}}
  %out_buf = "mgp.buffer.unsize" (%in_buf) : (!mgp.buffer<64, <"cuda", 3>>) -> !mgp.buffer<32, <"cuda", 3>>
  mef.output %out_buf : !mgp.buffer<32, <"cuda", 3>>
}

// -----

//
// mgp.buffer.split.borrow
//

func.func @buffer_split_mismatched_sizes(%ctx : !mgp.context, %ch0 : !mef.chain, %in_buf : !mgp.buffer<192, <"cuda", 3>>) {
  // expected-error@+1 {{'mgp.buffer.split.borrow' op expected input buffer type does not match actual input buffer type ('!mgp.buffer<320, <"cuda", 3>>' vs '!mgp.buffer<192, <"cuda", 3>>').}}
  %buf0, %buf1 = "mgp.buffer.split.borrow" (%ch0, %ctx, %in_buf) { aRuntimeSlot = 1 : ui64, bOutBufferRawSizes = #M.dense_array<64, 256> : !M.array<2xui64>, cDevice = #M.device_ref<"cuda", 3> }
                   : (!mef.chain, !mgp.context, !mgp.buffer<192, <"cuda", 3>>) -> (!mgp.buffer<64, <"cuda", 3>>, !mgp.buffer<256, <"cuda", 3>>)
  mef.output %buf0, %buf1 : !mgp.buffer<64, <"cuda", 3>>, !mgp.buffer<256, <"cuda", 3>>
}

// -----

//
// mgp.buffer.section.borrow.static
//

func.func @buffer_section_borrow_static_beyond_end(%ctx : !mgp.context, %ch0 : !mef.chain, %in_buf : !mgp.buffer<192, <"cuda", 3>>) {
  // expected-error@+1 {{'mgp.buffer.section.borrow.static' op requesting section end 256 which is beyond end of input buffer of size 192.}}
  %buf = "mgp.buffer.section.borrow.static" (%ch0, %ctx, %in_buf) { aRuntimeSlot = 1 : ui64, bOffset = 128 : ui64, cSize = 128 : ui64, dDevice = #M.device_ref<"cuda", 3> }
           : (!mef.chain, !mgp.context, !mgp.buffer<192, <"cuda", 3>>) -> !mgp.buffer<128, <"cuda", 3>>
  mef.output %buf : !mgp.buffer<128, <"cuda", 3>>
}

// -----

//
// mgp.buffer.section.borrow.dynamic
//

func.func @buffer_section_borrow_dynamic_sized(%ctx : !mgp.context, %ch0 : !mef.chain, %in_buf : !mgp.buffer<192, <"cuda", 3>>, %scale : index) {
  // expected-error@+1 {{'mgp.buffer.section.borrow.dynamic' op expected input buffer to be dynamically sized.}}
  %buf = "mgp.buffer.section.borrow.dynamic" (%ch0, %ctx, %in_buf, %scale) { aRuntimeSlot = 1 : ui64, bUnscaledOffset = 128 : ui64, cUnscaledSize = 128 : ui64, dDevice = #M.device_ref<"cuda", 3> }
           : (!mef.chain, !mgp.context, !mgp.buffer<192, <"cuda", 3>>, index) -> !mgp.buffer<, <"cuda", 3>>
  mef.output %buf : !mgp.buffer<, <"cuda", 3>>
}

// -----

//
// mgp.tensor.create
//

func.func @tensor_create_mismatched_size(%ctx : !mgp.context, %ch : !mef.chain, %buf : !mgp.buffer<61, <"cuda", 3>>, %spec : !mgp.tensor_spec<3x5xf32>) {
  // expected-error@+1 {{'mgp.tensor.create' op expected buffer type to match type implied by tensor spec ('!mgp.buffer<61, <"cuda", 3>>' vs '!mgp.buffer<60, <"cuda", 3>>').}}
  %tensor = "mgp.tensor.create" (%ch, %ctx, %buf, %spec) { aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cuda", 3> }
              : (!mef.chain, !mgp.context, !mgp.buffer<61, <"cuda", 3>>, !mgp.tensor_spec<3x5xf32>) -> !mgp.tensor<3x5xf32, <"cuda", 3>>
  mef.output %tensor : !mgp.tensor<3x5xf32, <"cuda", 3>>
}

// -----

//
// mgp.tensor.extract.buffer
//

func.func @tensor_extract_buffer_mismatched_size(%ctx : !mgp.context, %tensor : !mgp.tensor<3x5xf32, <"cuda", 3>>) {
  // expected-error@+1 {{'mgp.tensor.extract.buffer' op expected the result buffer to match the type implied by the tensor type ('!mgp.buffer<61, <"cuda", 3>>' vs '!mgp.buffer<60, <"cuda", 3>>').}}
  %buf = "mgp.tensor.extract.buffer" (%ctx, %tensor) { aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cuda", 3> }
           : (!mgp.context, !mgp.tensor<3x5xf32, <"cuda", 3>>) -> !mgp.buffer<61, <"cuda", 3>>
  mef.output %buf : !mgp.buffer<61, <"cuda", 3>>
}

// -----

//
// mgp.kernel.execut
//

func.func @kernel_execute_mismatched_arity(
  %ctx : !mgp.context,
  %ch0 : !mef.chain,
  %buf0 : !mgp.buffer<, <"cuda", 3>>,
  %buf1 : !mgp.buffer<8>,
  %buf2 : !mgp.buffer<48, <"cuda", 3>>,
  %dim0 : index,
  %dim1 : index
) {
  // expected-error@+1 {{'mgp.kernel.execute' op kernel has 3 arguments, but the in/out signature attribute has arity 4.}}
  %ch1 = "mgp.kernel.execute" (%ch0, %ctx, %buf0, %buf1, %buf2, %dim0, %dim1)
         { aHostRuntimeSlot = 0 : ui64, bOtherRuntimeSlot = 1 : ui64,
           cHostModuleIndex = 7 : ui64, dHostFunctionIndex = 9 : ui64,
           eInOutSignature = #M.inout_sig<"iioo">,
           fHostDevice = #M.device_ref<"cpu", 0>,
           gOtherDevice = #M.device_ref<"cuda", 3>,
           hSummary = "kernel2",
           iNumOutDimensions = 0 : ui64,
           jIsAsync = true,
           operandSegmentSizes = array<i32: 1, 1, 3, 2>}
         : (!mef.chain, !mgp.context, !mgp.buffer<, <"cuda", 3>>,
            !mgp.buffer<8>, !mgp.buffer<48, <"cuda", 3>>,
            index, index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}
