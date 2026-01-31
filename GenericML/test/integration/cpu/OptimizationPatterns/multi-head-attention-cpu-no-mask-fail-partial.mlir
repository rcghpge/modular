// Check pattern works for partial matches which should not infinitely loop
// and fail gracefully
// RUN: mo-opt %s --mogg-pattern-fusion | FileCheck %s

// CHECK-NOT: mo.custom {symbol = "no_mask_flash_attention_cpu"}
mo.graph @cpu_flash_attn_pattern<BH, B, H, S, D>(
    %attn_mat: !mo.tensor<[BH, S, S], f32>,
    %value_pre_transposed: !mo.tensor<[B, S, H, D], f32>,
    %norm: !mo.tensor<[BH, S, S], f32>
) -> !mo.tensor<[B, H, S, D], f32> {
    %perm1 = mo.constant {value = #M.dense_array<0, 2, 1, 3> : tensor<4xsi64>} : !mo.tensor<[4], si64>
    %axis = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %value = mo.transpose(%value_pre_transposed, %perm1) : (!mo.tensor<[B, S, H, D], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[B, H, S, D], f32>
    %value_reshaped = mo.static.reshape(%value) : !mo.tensor<[B, H, S, D], f32> -> !mo.tensor<[BH, S, D], f32>

    %attn_mat_norm = mo.mul(%attn_mat, %norm) : !mo.tensor<[BH, S, S], f32>
    %attn_mat_softmax = mo.softmax(%attn_mat_norm, %axis) : (!mo.tensor<[BH, S, S], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[BH, S, S], f32>
    %reproj = mo.batch_matmul(%attn_mat_softmax, %value_reshaped) : (!mo.tensor<[BH, S, S], f32>, !mo.tensor<[BH, S, D], f32>) -> !mo.tensor<[BH, S, D], f32>

    %final_answer = mo.static.reshape(%reproj): !mo.tensor<[BH, S, D], f32> -> !mo.tensor<[B, H, S, D], f32>
    mo.output %final_answer: !mo.tensor<[B, H, S, D], f32>
}
