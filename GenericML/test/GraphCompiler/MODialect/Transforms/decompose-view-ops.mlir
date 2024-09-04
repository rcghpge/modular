// RUN: mo-opt --decompose-views --cse %s | FileCheck %s

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
