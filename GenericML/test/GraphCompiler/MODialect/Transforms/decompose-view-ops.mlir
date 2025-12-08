// RUN: mo-opt --decompose-views --cse %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: mo.graph @slice
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[3, 11, 5], si32>, %[[START:.*]]: !mo.tensor<[3], si32>, %[[END:.*]]: !mo.tensor<[3], si32>, %[[STEP:.*]]: !mo.tensor<[3], si32>

// CHECK-NEXT: %[[ONE:.*]] = mo.constant {value = #M.dense_array<1>
// CHECK-NEXT: %[[ZERO:.*]] = mo.constant {value = #M.dense_array<0>

// Slice the start / stop / step.
// CHECK-NEXT: %[[S1:.*]] = mo.slice_dim[0] (%[[START]], %[[ZERO]], %[[ONE]], %[[ONE]]
// CHECK-NEXT: %[[E1:.*]] = mo.slice_dim[0] (%[[END]], %[[ZERO]], %[[ONE]], %[[ONE]]
// CHECK-NEXT: %[[STEP1:.*]] = mo.slice_dim[0] (%[[STEP]], %[[ZERO]], %[[ONE]], %[[ONE]]

// CHECK-NEXT: %[[O1:.*]] = mo.slice_dim[0] (%[[INPUT]], %[[S1]], %[[E1]], %[[STEP1]]) : (!mo.tensor<[3, 11, 5], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[2, 11, 5], si32>

// CHECK-NEXT: %[[TWO:.*]] = mo.constant {value = #M.dense_array<2>

// CHECK-NEXT: %[[S2:.*]] = mo.slice_dim[0] (%[[START]], %[[ONE]], %[[TWO]], %[[ONE]]
// CHECK-NEXT: %[[E2:.*]] = mo.slice_dim[0] (%[[END]], %[[ONE]], %[[TWO]], %[[ONE]]
// CHECK-NEXT: %[[STEP2:.*]] = mo.slice_dim[0] (%[[STEP]], %[[ONE]], %[[TWO]], %[[ONE]]

// CHECK-NEXT: %[[O2:.*]] = mo.slice_dim[1] (%[[O1]], %[[S2]], %[[E2]], %[[STEP2]]) : (!mo.tensor<[2, 11, 5], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[2, 7, 5], si32>

// CHECK-NEXT: %[[THREE:.*]] = mo.constant {value = #M.dense_array<3>

// CHECK-NEXT: %[[S3:.*]] = mo.slice_dim[0] (%[[START]], %[[TWO]], %[[THREE]], %[[ONE]]
// CHECK-NEXT: %[[E3:.*]] = mo.slice_dim[0] (%[[END]], %[[TWO]], %[[THREE]], %[[ONE]]
// CHECK-NEXT: %[[STEP3:.*]] = mo.slice_dim[0] (%[[STEP]], %[[TWO]], %[[THREE]], %[[ONE]]

// CHECK-NEXT: %[[O3:.*]] = mo.slice_dim[2] (%[[O2]], %[[S3]], %[[E3]], %[[STEP3]]) : (!mo.tensor<[2, 7, 5], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[2, 7, 3], si32>

// CHECK-NEXT: mo.output %[[O3]]

mo.graph @slice(%input: !mo.tensor<[3, 11, 5], si32>, %start: !mo.tensor<[3], si32>, %stop: !mo.tensor<[3], si32>, %step: !mo.tensor<[3], si32>) -> (!mo.tensor<[2, 7, 3], si32>) {
	%res = mo.slice(%input, %start, %stop, %step) : (!mo.tensor<[3, 11, 5], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[2, 7, 3], si32>
	mo.output %res : !mo.tensor<[2, 7, 3], si32>
}

// CHECK-LABEL: mo.graph @singleton_dims_scalar
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[], si32>)
// CHECK-NEXT: %[[V0:.*]] = mo.add_singleton_dim[0] (%[[ARG0]]) : (!mo.tensor<[], si32>) -> !mo.tensor<[1], si32>
// CHECK-NEXT: %[[V1:.*]] = mo.add_singleton_dim[1] (%[[V0]]) : (!mo.tensor<[1], si32>) -> !mo.tensor<[1, 1], si32>
// CHECK-NEXT: %[[V2:.*]] = mo.add_singleton_dim[2] (%[[V1]]) : (!mo.tensor<[1, 1], si32>) -> !mo.tensor<[1, 1, 1], si32>
// CHECK-NEXT: mo.output %[[V2]]

mo.graph @singleton_dims_scalar(%input: !mo.tensor<[], si32>) -> (!mo.tensor<[1, 1, 1], si32>) {
    %res = mo.static.reshape(%input) : !mo.tensor<[], si32> -> !mo.tensor<[1, 1, 1], si32>
	mo.output %res : !mo.tensor<[1, 1, 1], si32>
}

// CHECK-LABEL: mo.graph @singleton_dims_2<N, K, L>
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[N, K, L], si32>)
// CHECK-NEXT: %[[V0:.*]] = mo.add_singleton_dim[0] (%[[ARG0]]) : (!mo.tensor<[N, K, L], si32>) -> !mo.tensor<[1, N, K, L], si32>
// CHECK-NEXT: %[[V1:.*]] = mo.add_singleton_dim[3] (%[[V0]]) : (!mo.tensor<[1, N, K, L], si32>) -> !mo.tensor<[1, N, K, 1, L], si32>
// CHECK-NEXT: %[[V2:.*]] = mo.add_singleton_dim[5] (%[[V1]]) : (!mo.tensor<[1, N, K, 1, L], si32>) -> !mo.tensor<[1, N, K, 1, L, 1], si32>
// CHECK-NEXT: mo.output %[[V2]]

mo.graph @singleton_dims_2<N, K, L>(%input: !mo.tensor<[N, K, L], si32>) -> (!mo.tensor<[1, N, K, 1, L, 1], si32>) {
    %res = mo.static.reshape(%input) : !mo.tensor<[N, K, L], si32> -> !mo.tensor<[1, N, K, 1, L, 1], si32>
	mo.output %res : !mo.tensor<[1, N, K, 1, L, 1], si32>
}

// CHECK-LABEL: mo.graph @split_dim<N, K>
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[N, K], si32>)
// CHECK-NEXT: %[[V0:.*]] = mo.split_dim[0] (%[[ARG0]]) : (!mo.tensor<[N, K], si32>) -> !mo.tensor<[4, 8, K], si32>
// CHECK-NEXT: mo.output %[[V0]]

mo.graph @split_dim<N, K>(%input: !mo.tensor<[N, K], si32>) -> (!mo.tensor<[4, 8, K], si32>) {
    %res = mo.static.reshape(%input) : !mo.tensor<[N, K], si32> -> !mo.tensor<[4, 8, K], si32>
	mo.output %res : !mo.tensor<[4, 8, K], si32>
}

// CHECK-LABEL: mo.graph @flatten_dim
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[1, 2, 3, 4, 5], si32>)
// CHECK-NEXT: %[[V0:.*]] = mo.merge_dim[3] (%[[ARG0]]) : (!mo.tensor<[1, 2, 3, 4, 5], si32>) -> !mo.tensor<[1, 2, 3, 20], si32>
// CHECK-NEXT: %[[V1:.*]] = mo.merge_dim[2] (%[[V0]]) : (!mo.tensor<[1, 2, 3, 20], si32>) -> !mo.tensor<[1, 2, 60], si32>
// CHECK-NEXT: %[[V2:.*]] = mo.merge_dim[1] (%[[V1]]) : (!mo.tensor<[1, 2, 60], si32>) -> !mo.tensor<[1, 120], si32>
// CHECK-NEXT: %[[V3:.*]] = mo.merge_dim[0] (%[[V2]]) : (!mo.tensor<[1, 120], si32>) -> !mo.tensor<[120], si32>
// CHECK-NEXT: mo.output %[[V3]]
mo.graph @flatten_dim(%input : !mo.tensor<[1,2,3,4,5], si32>) -> (!mo.tensor<[120], si32>) {
    %res = mo.static.reshape(%input) : !mo.tensor<[1,2,3,4,5], si32> -> !mo.tensor<[120], si32>
	mo.output %res : !mo.tensor<[120], si32>
}

// CHECK-LABEL: mo.graph @unflatten_dim
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[120], si32>)
// CHECK-NEXT: %[[V0:.*]] = mo.split_dim[0] (%[[ARG0]]) : (!mo.tensor<[120], si32>) -> !mo.tensor<[1, 120], si32>
// CHECK-NEXT: %[[V1:.*]] = mo.split_dim[1] (%[[V0]]) : (!mo.tensor<[1, 120], si32>) -> !mo.tensor<[1, 2, 60], si32>
// CHECK-NEXT: %[[V2:.*]] = mo.split_dim[2] (%[[V1]]) : (!mo.tensor<[1, 2, 60], si32>) -> !mo.tensor<[1, 2, 3, 20], si32>
// CHECK-NEXT: %[[V3:.*]] = mo.split_dim[3] (%[[V2]]) : (!mo.tensor<[1, 2, 3, 20], si32>) -> !mo.tensor<[1, 2, 3, 4, 5], si32>
// CHECK-NEXT: mo.output %[[V3]]
mo.graph @unflatten_dim(%input : !mo.tensor<[120], si32>) -> (!mo.tensor<[1,2,3,4,5], si32>) {
    %res = mo.static.reshape(%input) : !mo.tensor<[120], si32> -> !mo.tensor<[1,2,3,4,5], si32>
	mo.output %res : !mo.tensor<[1,2,3,4,5], si32>
}

// CHECK-LABEL: mo.graph @dont_unflatten_reshape_with_zero_elts
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[0], si32>)
// CHECK-NEXT: %[[V0:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[0], si32> -> !mo.tensor<[2, 3, 4, 0], si32>
// CHECK-NEXT: mo.output %[[V0]]
mo.graph @dont_unflatten_reshape_with_zero_elts(%input : !mo.tensor<[0], si32>) -> (!mo.tensor<[2,3,4,0], si32>) {
    %res = mo.static.reshape(%input) : !mo.tensor<[0], si32> -> !mo.tensor<[2,3,4,0], si32>
	mo.output %res : !mo.tensor<[2,3,4,0], si32>
}

// CHECK-LABEL: mo.graph @dont_flatten_reshape_with_zero_elts
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[2, 3, 4, 0], si32>)
// CHECK-NEXT: %[[V0:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[2, 3, 4, 0], si32> -> !mo.tensor<[0], si32>
// CHECK-NEXT: mo.output %[[V0]]
mo.graph @dont_flatten_reshape_with_zero_elts(%input : !mo.tensor<[2,3,4,0], si32>) -> (!mo.tensor<[0], si32>) {
    %res = mo.static.reshape(%input) : !mo.tensor<[2,3,4,0], si32> -> !mo.tensor<[0], si32>
	mo.output %res : !mo.tensor<[0], si32>
}

// CHECK-LABEL: mo.graph @flatten_dim_with_layout
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[1, 2, 3, 4], si32, {layout = #mo.layout<NHWC>}>)
// CHECK-NEXT: %[[V0:.*]] = mo.merge_dim[2] (%[[ARG0]]) : (!mo.tensor<[1, 2, 3, 4], si32, {layout = #mo.layout<NHWC>}>) -> !mo.tensor<[1, 2, 12], si32>
// CHECK-NEXT: %[[V1:.*]] = mo.merge_dim[1] (%[[V0]]) : (!mo.tensor<[1, 2, 12], si32>) -> !mo.tensor<[1, 24], si32>
// CHECK-NEXT: %[[V2:.*]] = mo.merge_dim[0] (%[[V1]]) : (!mo.tensor<[1, 24], si32>) -> !mo.tensor<[24], si32>
// CHECK-NEXT: mo.output %[[V2]]
mo.graph @flatten_dim_with_layout(%input : !mo.tensor<[1,2,3,4], si32, {layout = #mo.layout<NHWC>}>) -> (!mo.tensor<[24], si32>) {
  %res = mo.static.reshape(%input) : !mo.tensor<[1,2,3,4], si32, {layout = #mo.layout<NHWC>}> -> !mo.tensor<[24], si32>
	mo.output %res : !mo.tensor<[24], si32>
}

// CHECK-LABEL: mo.graph @flatten_dim_with_device
// CHECK: {{.*}} = mo.merge_dim[0] ({{.*}}) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
mo.graph @flatten_dim_with_device() -> (!mo.tensor<[4], f32, gpu:0>) {
    %0 = mo.constant {value = #M.dense_array<4.0, 5.0, 6.0, 7.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32, gpu:0>
	%1 = mo.static.reshape(%0) : !mo.tensor<[2, 2], f32, gpu:0> -> !mo.tensor<[4], f32, gpu:0>
	mo.output %1 : !mo.tensor<[4], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @unflatten_dim_with_layout
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[24], si32>)
// CHECK-NEXT: %[[V0:.*]] = mo.split_dim[0] (%[[ARG0]]) : (!mo.tensor<[24], si32>) -> !mo.tensor<[1, 24], si32>
// CHECK-NEXT: %[[V1:.*]] = mo.split_dim[1] (%[[V0]]) : (!mo.tensor<[1, 24], si32>) -> !mo.tensor<[1, 2, 12], si32>
// CHECK-NEXT: %[[V2:.*]] = mo.split_dim[2] (%[[V1]]) : (!mo.tensor<[1, 2, 12], si32>) -> !mo.tensor<[1, 2, 3, 4], si32, {layout = #mo.layout<NHWC>}>
// CHECK-NEXT: mo.output %[[V2]]
mo.graph @unflatten_dim_with_layout(%input : !mo.tensor<[24], si32>) -> (!mo.tensor<[1,2,3,4], si32, {layout = #mo.layout<NHWC>}>) {
    %res = mo.static.reshape(%input) : !mo.tensor<[24], si32> -> !mo.tensor<[1,2,3,4], si32, {layout = #mo.layout<NHWC>}>
	mo.output %res : !mo.tensor<[1,2,3,4], si32, {layout = #mo.layout<NHWC>}>
}

// CHECK-LABEL: mo.graph @flatten_dim_symbolic
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[D1, D2, D3, D4], f32>)
// CHECK-NEXT: %[[V0:.*]] = mo.merge_dim[2] (%arg0) : (!mo.tensor<[D1, D2, D3, D4], f32>) -> !mo.tensor<[D1, D2, mul_no_wrap(D3, D4)], f32>
// CHECK-NEXT: %[[V1:.*]] = mo.merge_dim[1] (%[[V0]]) : (!mo.tensor<[D1, D2, mul_no_wrap(D3, D4)], f32>) -> !mo.tensor<[D1, mul_no_wrap(D2, D3, D4)], f32>
// CHECK-NEXT: %[[V2:.*]] = mo.merge_dim[0] (%[[V1]]) : (!mo.tensor<[D1, mul_no_wrap(D2, D3, D4)], f32>) -> !mo.tensor<[mul_no_wrap(D1, D2, D3, D4)], f32>
// CHECK-NEXT: mo.output %[[V2]]
mo.graph @flatten_dim_symbolic<D1,D2,D3,D4>(%input : !mo.tensor<[D1,D2,D3,D4], f32>) -> (!mo.tensor<[mul_no_wrap(D1,D2,D3,D4)], f32>) {
    %res = mo.static.reshape(%input) : !mo.tensor<[D1,D2,D3,D4], f32> -> !mo.tensor<[mul_no_wrap(D1,D2,D3,D4)], f32>
	mo.output %res : !mo.tensor<[mul_no_wrap(D1,D2,D3,D4)], f32>
}

// CHECK-LABEL: @flatten_dim_symbolic_with_rebind<D1, D2, D12>
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[D1, D2], f32>)
// CHECK-NEXT: %[[V0:.*]] = mo.merge_dim[0] (%[[ARG0]]) : (!mo.tensor<[D1, D2], f32>) -> !mo.tensor<[mul_no_wrap(D1, D2)], f32>
// CHECK-NEXT: %[[V1:.*]] = mo.rebind(%[[V0]]) : !mo.tensor<[mul_no_wrap(D1, D2)], f32> -> !mo.tensor<[D12], f32>
// CHECK-NEXT: mo.output %[[V1]] : !mo.tensor<[D12], f32>
mo.graph @flatten_dim_symbolic_with_rebind<D1,D2,D12>(%input : !mo.tensor<[D1,D2], f32>) -> (!mo.tensor<[D12], f32>) {
    %res = mo.static.reshape(%input) : !mo.tensor<[D1,D2], f32> -> !mo.tensor<[D12], f32>
	mo.output %res : !mo.tensor<[D12], f32>
}

// CHECK-LABEL: mo.graph @unflatten_dim_symbolic
// CHECK: %[[V1:.*]] = mo.split_dim[0] ({{.*}}) : (!mo.tensor<[mul_no_wrap(D1, D2, D3, D4)], f32>) -> !mo.tensor<[D1, mul_no_wrap(D2, D3, D4)], f32>
// CHECK-NEXT: %[[V2:.*]] = mo.split_dim[1] (%[[V1]]) : (!mo.tensor<[D1, mul_no_wrap(D2, D3, D4)], f32>) -> !mo.tensor<[D1, D2, mul_no_wrap(D3, D4)], f32>
// CHECK-NEXT: %[[V3:.*]] = mo.split_dim[2] (%[[V2]]) : (!mo.tensor<[D1, D2, mul_no_wrap(D3, D4)], f32>) -> !mo.tensor<[D1, D2, D3, D4], f32>
// CHECK-NEXT: %[[V4:.*]] = mo.relu(%[[V3]]) : !mo.tensor<[D1, D2, D3, D4], f32>
// CHECK-NEXT: mo.output %[[V4]] : !mo.tensor<[D1, D2, D3, D4], f32>
mo.graph @unflatten_dim_symbolic<D1,D2,D3,D4>() -> (!mo.tensor<[D1,D2,D3,D4], f32>) {
	%0 = mo.constant {value = #M.dense_array<1.0> : tensor<f32>} : !mo.tensor<[], f32>
	%1 = mo.static.reshape(%0) : !mo.tensor<[], f32> -> !mo.tensor<[mul_no_wrap(D1,D2,D3,D4)], f32>
	%2 = mo.static.reshape(%1) : !mo.tensor<[mul_no_wrap(D1,D2,D3,D4)], f32> -> !mo.tensor<[D1,D2,D3,D4], f32>
	%3 = mo.relu(%2) : !mo.tensor<[D1,D2,D3,D4], f32>
	mo.output %3 : !mo.tensor<[D1,D2,D3,D4], f32>
}

// CHECK-LABEL: mo.graph @flatten_dim_partially_symbolic
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[D1, D2, 8, D3, D4], f32>)
// CHECK-NEXT: %[[V0:.*]] = mo.merge_dim[3] (%[[ARG0]]) : (!mo.tensor<[D1, D2, 8, D3, D4], f32>) -> !mo.tensor<[D1, D2, 8, mul_no_wrap(D3, D4)], f32>
// CHECK-NEXT: %[[V1:.*]] = mo.merge_dim[2] (%[[V0]]) : (!mo.tensor<[D1, D2, 8, mul_no_wrap(D3, D4)], f32>) -> !mo.tensor<[D1, D2, mul_no_wrap(D3, D4, 8)], f32>
// CHECK-NEXT: %[[V2:.*]] = mo.merge_dim[1] (%[[V1]]) : (!mo.tensor<[D1, D2, mul_no_wrap(D3, D4, 8)], f32>) -> !mo.tensor<[D1, mul_no_wrap(D2, D3, D4, 8)], f32>
// CHECK-NEXT: %[[V3:.*]] = mo.merge_dim[0] (%[[V2]]) : (!mo.tensor<[D1, mul_no_wrap(D2, D3, D4, 8)], f32>) -> !mo.tensor<[mul_no_wrap(D1, D2, D3, D4, 8)], f32>
// CHECK-NEXT: mo.output %[[V3]]
mo.graph @flatten_dim_partially_symbolic<D1,D2,D3,D4>(%input : !mo.tensor<[D1,D2,8,D3,D4], f32>) -> (!mo.tensor<[mul_no_wrap(D1,D2,8,D3,D4)], f32>) {
    %res = mo.static.reshape(%input) : !mo.tensor<[D1,D2,8,D3,D4], f32> -> !mo.tensor<[mul_no_wrap(D1,D2,8,D3,D4)], f32>
	mo.output %res : !mo.tensor<[mul_no_wrap(D1,D2,8,D3,D4)], f32>
}

// CHECK-LABEL: mo.graph @unflatten_dim_symbolic_with_rebind<D1, D2, D12>() -> !mo.tensor<[D1, D2, 3], f32> {
// CHECK: %[[V0:.*]] = mo.split_dim[0] ({{.*}}) : (!mo.tensor<[D12], f32>) -> !mo.tensor<[D1, cond(eq(:si64 D1, 0), 0, div(D12, cond(eq(:si64 D1, 0), 1, D1)))], f32>
// CHECK-NEXT: %[[V1:.*]] = mo.split_dim[1] (%[[V0]]) : (!mo.tensor<[D1, cond(eq(:si64 D1, 0), 0, div(D12, cond(eq(:si64 D1, 0), 1, D1)))], f32>) -> !mo.tensor<[D1, D2, cond(eq(:si64 D2, 0), 0, div(cond(eq(:si64 D1, 0), 0, div(D12, cond(eq(:si64 D1, 0), 1, D1))), cond(eq(:si64 D2, 0), 1, D2)))], f32>
// CHECK-NEXT: %[[V2:.*]] = mo.rebind(%[[V1]]) : !mo.tensor<[D1, D2, cond(eq(:si64 D2, 0), 0, div(cond(eq(:si64 D1, 0), 0, div(D12, cond(eq(:si64 D1, 0), 1, D1))), cond(eq(:si64 D2, 0), 1, D2)))], f32> -> !mo.tensor<[D1, D2, 3], f32>
// CHECK-NEXT: %[[V3:.*]] = mo.relu(%[[V2]]) : !mo.tensor<[D1, D2, 3], f32>
mo.graph @unflatten_dim_symbolic_with_rebind<D1,D2,D12>() -> (!mo.tensor<[D1,D2,3], f32>) {
	%0 = mo.constant {value = #M.dense_array<1.0> : tensor<f32>} : !mo.tensor<[], f32>
	%1 = mo.static.reshape(%0) : !mo.tensor<[], f32> -> !mo.tensor<[D12], f32>
	%2 = mo.static.reshape(%1) : !mo.tensor<[D12], f32> -> !mo.tensor<[D1,D2,3], f32>
	%3 = mo.relu(%2) : !mo.tensor<[D1,D2,3], f32>
	mo.output %3 : !mo.tensor<[D1,D2,3], f32>
}

// CHECK-LABEL: mo.graph @unflatten_with_div<D1>
// CHECK: %[[V1:.*]] = mo.static.reshape(%{{.*}}) : !mo.tensor<[], f32> -> !mo.tensor<[div(D1, 2)], f32>
// CHECK: %[[V2:.*]] = mo.split_dim[0] (%[[V1]]) : (!mo.tensor<[div(D1, 2)], f32>) -> !mo.tensor<[1, div(D1, 2)], f32>
// CHECK: mo.output %[[V2]]
mo.graph @unflatten_with_div<D1>() -> (!mo.tensor<[1, div(D1, 2)], f32>) {
    %0 = mo.constant {value = #M.dense_array<1.0> : tensor<f32>} : !mo.tensor<[], f32>
    %1 = mo.static.reshape(%0) : !mo.tensor<[], f32> -> !mo.tensor<[div(D1, 2)], f32>
    %2 = mo.static.reshape(%1) : !mo.tensor<[div(D1, 2)], f32> -> !mo.tensor<[1, div(D1, 2)], f32>
    mo.output %2 : !mo.tensor<[1, div(D1, 2)], f32>
}

// CHECK-LABEL: @do_not_decompose_contig_slice
// CHECK: mo.slice
// CHECK-NOT: mo.slice_dim
mo.graph @do_not_decompose_contig_slice(%input: !mo.tensor<[10, 2], si32>) -> (!mo.tensor<[2, 2], si32>) {
	%start = mo.constant {value = #M.dense_array<8, 0> : tensor<2xsi32>} : !mo.tensor<[2], si32>
	%stop = mo.constant {value = #M.dense_array<10, 2> : tensor<2xsi32>} : !mo.tensor<[2], si32>
	%step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi32>} : !mo.tensor<[2], si32>
	%res = mo.slice(%input, %start, %stop, %step) : (!mo.tensor<[10, 2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 2], si32>
  %mm = mo.matmul(%res, %res) : (!mo.tensor<[2, 2], si32>, !mo.tensor<[2, 2], si32>)  -> !mo.tensor<[2, 2], si32>
	mo.output %mm : !mo.tensor<[2, 2], si32>
}
