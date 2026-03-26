// RUN: mo-opt %s -o %t.mlir --gc-pipeline --allow-unregistered-dialect --mlir-print-ir-after="mogg-fuse-epilogues" --mlir-disable-threading 2>&1 | FileCheck %s --check-prefix=CHECK-IR
// XFAIL: *

// CHECK-IR-LABEL: @rope_kv_cache_epilogue_fusion
// CHECK-IR: summary = "custom__mo.rope.ragged:{{[0-9]+}}"
// CHECK-IR: summary = "custom__mo.rope.ragged:{{[0-9]+}}+custom__mo.kv_cache.store.paged.ragged:{{[0-9]+}}"

mo.graph @rope_kv_cache_epilogue_fusion<total_seq_len, input_row_offsets_len, total_num_pages, replica_0_batch_size, replica_0_max_num_pages, replica_0_steps_remaining>(
  %query: !mo.tensor<[total_seq_len, 32, 64], f32, gpu:0>,
  %key: !mo.tensor<[total_seq_len, 8, 64], f32, gpu:0>,
  %value: !mo.tensor<[total_seq_len, 8, 64], f32, gpu:0>,
  %row_offsets: !mo.tensor<[input_row_offsets_len], ui32>,
  %kv_cache: !mo.buffer<[total_num_pages, 2, 1, 128, 8, 64], f32, gpu:0>,
  %batch_sizes: !mo.tensor<[replica_0_batch_size], ui32, gpu:0>,
  %page_table: !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32, gpu:0>,
  %steps_remaining: !mo.tensor<[replica_0_steps_remaining, 2], ui32>,
  %rope_freqs: !mo.tensor<[4096, 2], f32, gpu:0>
) -> !mo.tensor<[total_seq_len, 32, 64], f32, gpu:0> {
  %ch0 = mo.chain.create()
  %zero = mo.constant {value = #M.dense_array<0> : tensor<ui32>} : !mo.tensor<[], ui32>

  %q_rope = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {interleaved = true}, symbol = "mo.rope.ragged"}(%query, %row_offsets, %batch_sizes, %rope_freqs) : (!mo.tensor<[total_seq_len, 32, 64], f32, gpu:0>, !mo.tensor<[input_row_offsets_len], ui32>, !mo.tensor<[replica_0_batch_size], ui32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>) -> !mo.tensor<[total_seq_len, 32, 64], f32, gpu:0>
  %k_rope = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {interleaved = true}, symbol = "mo.rope.ragged"}(%key, %row_offsets, %batch_sizes, %rope_freqs) : (!mo.tensor<[total_seq_len, 8, 64], f32, gpu:0>, !mo.tensor<[input_row_offsets_len], ui32>, !mo.tensor<[replica_0_batch_size], ui32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>) -> !mo.tensor<[total_seq_len, 8, 64], f32, gpu:0>
  %k_cast = mo.cast(%k_rope) : (!mo.tensor<[total_seq_len, 8, 64], f32, gpu:0>) -> !mo.tensor<[total_seq_len, 8, 64], f32, gpu:0>
  %ch1 = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {key_or_value = 0 : index}, symbol = "mo.kv_cache.store.paged.ragged"}(%k_cast, %kv_cache, %batch_sizes, %page_table, %row_offsets, %steps_remaining, %zero, %ch0) : (!mo.tensor<[total_seq_len, 8, 64], f32, gpu:0>, !mo.buffer<[total_num_pages, 2, 1, 128, 8, 64], f32, gpu:0>, !mo.tensor<[replica_0_batch_size], ui32, gpu:0>, !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32, gpu:0>, !mo.tensor<[input_row_offsets_len], ui32>, !mo.tensor<[replica_0_steps_remaining, 2], ui32>, !mo.tensor<[], ui32>, !mo.chain) -> !mo.chain

  %v_cast = mo.cast(%value) : (!mo.tensor<[total_seq_len, 8, 64], f32, gpu:0>) -> !mo.tensor<[total_seq_len, 8, 64], f32, gpu:0>
  %ch2 = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {key_or_value = 1 : index}, symbol = "mo.kv_cache.store.paged.ragged"}(%v_cast, %kv_cache, %batch_sizes, %page_table, %row_offsets, %steps_remaining, %zero, %ch1) : (!mo.tensor<[total_seq_len, 8, 64], f32, gpu:0>, !mo.buffer<[total_num_pages, 2, 1, 128, 8, 64], f32, gpu:0>, !mo.tensor<[replica_0_batch_size], ui32, gpu:0>, !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32, gpu:0>, !mo.tensor<[input_row_offsets_len], ui32>, !mo.tensor<[replica_0_steps_remaining, 2], ui32>, !mo.tensor<[], ui32>, !mo.chain) -> !mo.chain

  mo.output %q_rope : !mo.tensor<[total_seq_len, 32, 64], f32, gpu:0>
}
