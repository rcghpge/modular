// RUN: mo-opt --decompose-views --cse %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: mo.graph @slice
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[3, 11, 5], si32>, %[[START:.*]]: !mo.tensor<[3], si32>, %[[END:.*]]: !mo.tensor<[3], si32>, %[[STEP:.*]]: !mo.tensor<[3], si32>

// CHECK-NEXT: %[[ONE:.*]] = mo.constant {value = #M.dense_array<1>
// CHECK-NEXT: %[[ZERO:.*]] = mo.constant {value = #M.dense_array<0>

// Slice the start / stop / step.
// CHECK-NEXT: %[[S1:.*]] = mo.slice_dim[0 : index] (%[[START]], %[[ZERO]], %[[ONE]], %[[ONE]]
// CHECK-NEXT: %[[E1:.*]] = mo.slice_dim[0 : index] (%[[END]], %[[ZERO]], %[[ONE]], %[[ONE]]
// CHECK-NEXT: %[[STEP1:.*]] = mo.slice_dim[0 : index] (%[[STEP]], %[[ZERO]], %[[ONE]], %[[ONE]]

// CHECK-NEXT: %[[O1:.*]] = mo.slice_dim[0 : index] (%[[INPUT]], %[[S1]], %[[E1]], %[[STEP1]]) : (!mo.tensor<[3, 11, 5], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[2, 11, 5], si32>

// CHECK-NEXT: %[[TWO:.*]] = mo.constant {value = #M.dense_array<2>

// CHECK-NEXT: %[[S2:.*]] = mo.slice_dim[0 : index] (%[[START]], %[[ONE]], %[[TWO]], %[[ONE]]
// CHECK-NEXT: %[[E2:.*]] = mo.slice_dim[0 : index] (%[[END]], %[[ONE]], %[[TWO]], %[[ONE]]
// CHECK-NEXT: %[[STEP2:.*]] = mo.slice_dim[0 : index] (%[[STEP]], %[[ONE]], %[[TWO]], %[[ONE]]

// CHECK-NEXT: %[[O2:.*]] = mo.slice_dim[1 : index] (%[[O1]], %[[S2]], %[[E2]], %[[STEP2]]) : (!mo.tensor<[2, 11, 5], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[2, 7, 5], si32>

// CHECK-NEXT: %[[THREE:.*]] = mo.constant {value = #M.dense_array<3>

// CHECK-NEXT: %[[S3:.*]] = mo.slice_dim[0 : index] (%[[START]], %[[TWO]], %[[THREE]], %[[ONE]]
// CHECK-NEXT: %[[E3:.*]] = mo.slice_dim[0 : index] (%[[END]], %[[TWO]], %[[THREE]], %[[ONE]]
// CHECK-NEXT: %[[STEP3:.*]] = mo.slice_dim[0 : index] (%[[STEP]], %[[TWO]], %[[THREE]], %[[ONE]]

// CHECK-NEXT: %[[O3:.*]] = mo.slice_dim[2 : index] (%[[O2]], %[[S3]], %[[E3]], %[[STEP3]]) : (!mo.tensor<[2, 7, 5], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[2, 7, 3], si32>

// CHECK-NEXT: mo.output %[[O3]]

mo.graph @slice(%input: !mo.tensor<[3, 11, 5], si32>, %start: !mo.tensor<[3], si32>, %stop: !mo.tensor<[3], si32>, %step: !mo.tensor<[3], si32>) -> (!mo.tensor<[2, 7, 3], si32>) {
	%res = mo.slice(%input, %start, %stop, %step) : (!mo.tensor<[3, 11, 5], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[2, 7, 3], si32>
	mo.output %res : !mo.tensor<[2, 7, 3], si32>
}

// CHECK-LABEL: mo.graph @slice_offset_only_inner_dim
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[3, 11, 5], si32>
// CHECK-NEXT: %[[START:.*]] = mo.constant {value = #M.dense_array<1, 2, 1> : tensor<3xsi32>} : !mo.tensor<[3], si32>
// CHECK-NEXT: %[[END:.*]] = mo.constant {value = #M.dense_array<3, 8, 4> : tensor<3xsi32>} : !mo.tensor<[3], si32>
// CHECK-NEXT: %[[STEP:.*]] = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi32>} : !mo.tensor<[3], si32>
// CHECK-NEXT: %[[ONE:.*]] = mo.constant {value = #M.dense_array<1>
// CHECK-NEXT: %[[ZERO:.*]] = mo.constant {value = #M.dense_array<0>

// Slice the start / stop / step.
// CHECK-NEXT: %[[S1:.*]] = mo.slice_dim[0 : index] (%[[START]], %[[ZERO]], %[[ONE]], %[[ONE]]
// CHECK-NEXT: %[[E1:.*]] = mo.slice_dim[0 : index] (%[[END]], %[[ZERO]], %[[ONE]], %[[ONE]]

// CHECK-NEXT: %[[O1:.*]] = mo.slice_dim_outer(%[[INPUT]], %[[S1]], %[[E1]]) : (!mo.tensor<[3, 11, 5], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[2, 11, 5], si32>

// CHECK-NEXT: %[[TWO:.*]] = mo.constant {value = #M.dense_array<2>

// CHECK-NEXT: %[[S2:.*]] = mo.slice_dim[0 : index] (%[[START]], %[[ONE]], %[[TWO]], %[[ONE]]
// CHECK-NEXT: %[[E2:.*]] = mo.slice_dim[0 : index] (%[[END]], %[[ONE]], %[[TWO]], %[[ONE]]
// CHECK-NEXT: %[[STEP2:.*]] = mo.slice_dim[0 : index] (%[[STEP]], %[[ONE]], %[[TWO]], %[[ONE]]

// CHECK-NEXT: %[[O2:.*]] = mo.slice_dim[1 : index] (%[[O1]], %[[S2]], %[[E2]], %[[STEP2]]) : (!mo.tensor<[2, 11, 5], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[2, 7, 5], si32>

// CHECK-NEXT: %[[THREE:.*]] = mo.constant {value = #M.dense_array<3>

// CHECK-NEXT: %[[S3:.*]] = mo.slice_dim[0 : index] (%[[START]], %[[TWO]], %[[THREE]], %[[ONE]]
// CHECK-NEXT: %[[E3:.*]] = mo.slice_dim[0 : index] (%[[END]], %[[TWO]], %[[THREE]], %[[ONE]]
// CHECK-NEXT: %[[STEP3:.*]] = mo.slice_dim[0 : index] (%[[STEP]], %[[TWO]], %[[THREE]], %[[ONE]]

// CHECK-NEXT: %[[O3:.*]] = mo.slice_dim[2 : index] (%[[O2]], %[[S3]], %[[E3]], %[[STEP3]]) : (!mo.tensor<[2, 7, 5], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[2, 7, 3], si32>

// CHECK-NEXT: mo.output %[[O3]]

mo.graph @slice_offset_only_inner_dim(%input: !mo.tensor<[3, 11, 5], si32>) -> (!mo.tensor<[2, 7, 3], si32>) {
	%start = mo.constant {value = #M.dense_array<1, 2, 1> : tensor<3xsi32>} : !mo.tensor<[3], si32>
	%stop = mo.constant {value = #M.dense_array<3, 8, 4> : tensor<3xsi32>} : !mo.tensor<[3], si32>
	%step = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi32>} : !mo.tensor<[3], si32>
	%res = mo.slice(%input, %start, %stop, %step) : (!mo.tensor<[3, 11, 5], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[2, 7, 3], si32>
	mo.output %res : !mo.tensor<[2, 7, 3], si32>
}

// CHECK-LABEL: mo.graph @slice_offset_only_outer_dim
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[1, 11, 5], si32>
// CHECK-NEXT: %[[START:.*]] = mo.constant {value = #M.dense_array<0, 2, 1> : tensor<3xsi32>} : !mo.tensor<[3], si32>
// CHECK-NEXT: %[[END:.*]] = mo.constant {value = #M.dense_array<1, 8, 4> : tensor<3xsi32>} : !mo.tensor<[3], si32>
// CHECK-NEXT: %[[STEP:.*]] = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi32>} : !mo.tensor<[3], si32>
// CHECK-NEXT: %[[ONE:.*]] = mo.constant {value = #M.dense_array<1>
// CHECK-NEXT: %[[ZERO:.*]] = mo.constant {value = #M.dense_array<0>

// Slice the start / stop / step.
// CHECK-NEXT: %[[S1:.*]] = mo.slice_dim[0 : index] (%[[START]], %[[ZERO]], %[[ONE]], %[[ONE]]
// CHECK-NEXT: %[[E1:.*]] = mo.slice_dim[0 : index] (%[[END]], %[[ZERO]], %[[ONE]], %[[ONE]]

// CHECK-NEXT: %[[O1:.*]] = mo.slice_dim_outer(%[[INPUT]], %[[S1]], %[[E1]]) : (!mo.tensor<[1, 11, 5], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[1, 11, 5], si32>

// CHECK-NEXT: %[[TWO:.*]] = mo.constant {value = #M.dense_array<2>

// CHECK-NEXT: %[[S2:.*]] = mo.slice_dim[0 : index] (%[[START]], %[[ONE]], %[[TWO]], %[[ONE]]
// CHECK-NEXT: %[[E2:.*]] = mo.slice_dim[0 : index] (%[[END]], %[[ONE]], %[[TWO]], %[[ONE]]

// CHECK-NEXT: %[[SQUEEZED:.*]] = mo.static.reshape(%[[O1]]) : !mo.tensor<[1, 11, 5], si32> -> !mo.tensor<[11, 5], si32>
// CHECK-NEXT: %[[O2_SQUEEZED:.*]] = mo.slice_dim_outer(%[[SQUEEZED]], %[[S2]], %[[E2]]) : (!mo.tensor<[11, 5], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[7, 5], si32>
// CHECK-NEXT: %[[O2:.*]] = mo.static.reshape(%[[O2_SQUEEZED]]) : !mo.tensor<[7, 5], si32> -> !mo.tensor<[1, 7, 5], si32>

// CHECK-NEXT: %[[THREE:.*]] = mo.constant {value = #M.dense_array<3>

// CHECK-NEXT: %[[S3:.*]] = mo.slice_dim[0 : index] (%[[START]], %[[TWO]], %[[THREE]], %[[ONE]]
// CHECK-NEXT: %[[E3:.*]] = mo.slice_dim[0 : index] (%[[END]], %[[TWO]], %[[THREE]], %[[ONE]]
// CHECK-NEXT: %[[STEP3:.*]] = mo.slice_dim[0 : index] (%[[STEP]], %[[TWO]], %[[THREE]], %[[ONE]]

// CHECK-NEXT: %[[O3:.*]] = mo.slice_dim[2 : index] (%[[O2]], %[[S3]], %[[E3]], %[[STEP3]]) : (!mo.tensor<[1, 7, 5], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[1, 7, 3], si32>

// CHECK-NEXT: mo.output %[[O3]]

mo.graph @slice_offset_only_outer_dim(%input: !mo.tensor<[1, 11, 5], si32>) -> (!mo.tensor<[1, 7, 3], si32>) {
	%start = mo.constant {value = #M.dense_array<0, 2, 1> : tensor<3xsi32>} : !mo.tensor<[3], si32>
	%stop = mo.constant {value = #M.dense_array<1, 8, 4> : tensor<3xsi32>} : !mo.tensor<[3], si32>
	%step = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi32>} : !mo.tensor<[3], si32>
	%res = mo.slice(%input, %start, %stop, %step) : (!mo.tensor<[1, 11, 5], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[1, 7, 3], si32>
	mo.output %res : !mo.tensor<[1, 7, 3], si32>
}

// CHECK-LABEL: mo.graph @singleton_dims_scalar
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[], si32>)
// CHECK-NEXT: %[[V0:.*]] = mo.add_singleton_dim[0 : index] (%[[ARG0]]) : (!mo.tensor<[], si32>) -> !mo.tensor<[1], si32>
// CHECK-NEXT: %[[V1:.*]] = mo.add_singleton_dim[1 : index] (%[[V0]]) : (!mo.tensor<[1], si32>) -> !mo.tensor<[1, 1], si32>
// CHECK-NEXT: %[[V2:.*]] = mo.add_singleton_dim[2 : index] (%[[V1]]) : (!mo.tensor<[1, 1], si32>) -> !mo.tensor<[1, 1, 1], si32>
// CHECK-NEXT: mo.output %[[V2]]

mo.graph @singleton_dims_scalar(%input: !mo.tensor<[], si32>) -> (!mo.tensor<[1, 1, 1], si32>) {
    %res = mo.static.reshape(%input) : !mo.tensor<[], si32> -> !mo.tensor<[1, 1, 1], si32>
	mo.output %res : !mo.tensor<[1, 1, 1], si32>
}

// CHECK-LABEL: mo.graph @singleton_dims_2<N, K, L>
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[N, K, L], si32>)
// CHECK-NEXT: %[[V0:.*]] = mo.add_singleton_dim[0 : index] (%[[ARG0]]) : (!mo.tensor<[N, K, L], si32>) -> !mo.tensor<[1, N, K, L], si32>
// CHECK-NEXT: %[[V1:.*]] = mo.add_singleton_dim[3 : index] (%[[V0]]) : (!mo.tensor<[1, N, K, L], si32>) -> !mo.tensor<[1, N, K, 1, L], si32>
// CHECK-NEXT: %[[V2:.*]] = mo.add_singleton_dim[5 : index] (%[[V1]]) : (!mo.tensor<[1, N, K, 1, L], si32>) -> !mo.tensor<[1, N, K, 1, L, 1], si32>
// CHECK-NEXT: mo.output %[[V2]]

mo.graph @singleton_dims_2<N, K, L>(%input: !mo.tensor<[N, K, L], si32>) -> (!mo.tensor<[1, N, K, 1, L, 1], si32>) {
    %res = mo.static.reshape(%input) : !mo.tensor<[N, K, L], si32> -> !mo.tensor<[1, N, K, 1, L, 1], si32>
	mo.output %res : !mo.tensor<[1, N, K, 1, L, 1], si32>
}
