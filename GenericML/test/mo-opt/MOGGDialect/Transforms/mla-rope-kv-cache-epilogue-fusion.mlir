// RUN: mo-opt %s -o %t.mlir --mo-to-mgp --allow-unregistered-dialect --mlir-print-ir-after="mogg-fuse-prologues-epilogues" --mlir-disable-threading 2>&1 | FileCheck %s --check-prefix=CHECK-IR
// XFAIL: *

// CHECK-IR-LABEL: @mla_rope_kv_cache_epilogue_fusion
// Expect rope -> concat -> kv_cache.store to fuse into one kernel summary.
// This test should fail if fusion does not happen.
// CHECK-IR: summary = "custom__mo.rope.ragged:{{[0-9]+}}+mo.concat:{{[0-9]+}}+custom__mo.kv_cache.store.paged.ragged:{{[0-9]+}}"

mo.graph @mla_rope_kv_cache_epilogue_fusion<total_seq_len, input_row_offsets_len, total_num_pages, replica_0_batch_size, replica_0_max_num_pages, replica_0_steps_remaining>(
  %k_nope: !mo.tensor<[total_seq_len, 1, 512], f32, gpu:0>,
  %k_rope_input: !mo.tensor<[total_seq_len, 1, 64], f32, gpu:0>,
  %row_offsets: !mo.tensor<[input_row_offsets_len], ui32>,
  %kv_cache: !mo.buffer<[total_num_pages, 1, 1, 128, 1, 576], f32, gpu:0>,
  %batch_sizes: !mo.tensor<[replica_0_batch_size], ui32, gpu:0>,
  %page_table: !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32, gpu:0>,
  %steps_remaining: !mo.tensor<[replica_0_steps_remaining, 2], ui32>,
  %rope_freqs: !mo.tensor<[4096, 2], f32, gpu:0>
) -> !mo.tensor<[total_seq_len, 1, 64], f32, gpu:0> {
  %ch0 = mo.chain.create()
  %zero = mo.constant {value = #M.dense_array<0> : tensor<ui32>} : !mo.tensor<[], ui32>

  %k_rope = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {interleaved = true}, symbol = "mo.rope.ragged"}(%k_rope_input, %row_offsets, %batch_sizes, %rope_freqs) : (!mo.tensor<[total_seq_len, 1, 64], f32, gpu:0>, !mo.tensor<[input_row_offsets_len], ui32>, !mo.tensor<[replica_0_batch_size], ui32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>) -> !mo.tensor<[total_seq_len, 1, 64], f32, gpu:0>
  %k_full = rmo.concat(%k_nope, %k_rope) {axis = 2 : index} : (!mo.tensor<[total_seq_len, 1, 512], f32, gpu:0>, !mo.tensor<[total_seq_len, 1, 64], f32, gpu:0>) -> !mo.tensor<[total_seq_len, 1, 576], f32, gpu:0>
  %k_cast = mo.cast(%k_full) : (!mo.tensor<[total_seq_len, 1, 576], f32, gpu:0>) -> !mo.tensor<[total_seq_len, 1, 576], f32, gpu:0>
  %ch1 = mo.custom {device = #M.device_ref<"gpu", 0>, parameters = {key_or_value = 0 : index}, symbol = "mo.kv_cache.store.paged.ragged"}(%k_cast, %kv_cache, %batch_sizes, %page_table, %row_offsets, %steps_remaining, %zero, %ch0) : (!mo.tensor<[total_seq_len, 1, 576], f32, gpu:0>, !mo.buffer<[total_num_pages, 1, 1, 128, 1, 576], f32, gpu:0>, !mo.tensor<[replica_0_batch_size], ui32, gpu:0>, !mo.tensor<[replica_0_batch_size, replica_0_max_num_pages], ui32, gpu:0>, !mo.tensor<[input_row_offsets_len], ui32>, !mo.tensor<[replica_0_steps_remaining, 2], ui32>, !mo.tensor<[], ui32>, !mo.chain) -> !mo.chain

  mo.output %k_rope : !mo.tensor<[total_seq_len, 1, 64], f32, gpu:0>
}
