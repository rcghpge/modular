// RUN: mo-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect

//
// mgp.buffer.constant
//

func.func @buffer_constant_mismatched_size(%ch: !mef.chain, %ctx : !mgp.context) {
  // expected-error@+1 {{'mgp.buffer.constant' op expected result buffer type does not match actual result buffer type ('!mgp.buffer<3>' vs '!mgp.buffer<5>').}}
  %0 = "mgp.buffer.constant" () { aValue = #M.aligned_bytes<"0x000102", align 32>, bRawAlign = 32 : ui64 }
         : () -> !mgp.buffer<5>
  mef.output %0 : !mgp.buffer<5>
}

// -----

func.func @buffer_constant_invalid_elem_type(%ch: !mef.chain, %ctx : !mgp.context) {
  // expected-error@+1 {{'mgp.buffer.constant' op only supports attributes of uint8_t elements (have 'i16').}}
  %0 = "mgp.buffer.constant" () { aValue = #M.dense_array<1, 2, 3, 4> : !M.array<4xi16>, bRawAlign = 32 : ui64 }
         : () -> !mgp.buffer<4>
  mef.output %0 : !mgp.buffer<4>
}

// -----

//
// mgp.buffer.host_to_device
//

func.func @buffer_host_to_device_mismatched_size(%ctx : !mgp.context, %ch0 : !mef.chain, %cpu_buf : !mgp.buffer<48>, %cuda_buf : !mgp.buffer<64, <"cuda", 3>>, %gpu : !mgp.opaque<"cuda0_context">) {
  // expected-error@+1 {{'mgp.buffer.host_to_device' op host and other buffer sizes do not match (48 vs 64).}}
  %ch1 = "mgp.buffer.host_to_device" (%ch0, %ctx, %cpu_buf, %cuda_buf, %gpu) { aHostRuntimeSlot = 0 : ui64, bOtherRuntimeSlot = 1 : ui64, cHostDevice = #M.device_ref<"cpu", 0>, dOtherDevice = #M.device_ref<"cuda", 3> }
           : (!mef.chain, !mgp.context, !mgp.buffer<48>, !mgp.buffer<64, <"cuda", 3>>, !mgp.opaque<"cuda0_context">) -> !mef.chain
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
