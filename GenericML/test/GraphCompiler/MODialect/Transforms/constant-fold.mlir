// RUN: mo-opt --split-input-file --constant-fold %s | FileCheck %s

// COM: Examine dumped MO constant subgraph before execution
// RUN: mo-opt --split-input-file --constant-fold --mlir-print-ir-after=mo-to-mogg 2>&1 %s | FileCheck %s -check-prefix=CHECK-CONST-MO-SUBGRAPH

// No kernel for large_broadcast because it is not constant folded.
// CHECK-CONST-MO-SUBGRAPH-NOT: large_broadcast
mo.graph @large_broadcast() -> (!mo.tensor<[1024], si8>) {
  %shape = mo.constant {value = #M.dense_array<1024> : tensor<1xsi32>} :
			!mo.tensor<[1], si32>
  %data = mo.constant {value = #M.dense_array<1> : tensor<1xsi8>} :
			!mo.tensor<[1], si8>
  %broadcast = mo.broadcast_to(%data, %shape) :
     (!mo.tensor<[1], si8>, !mo.tensor<[1], si32>) -> !mo.tensor<[1024], si8>
  mo.output %broadcast : !mo.tensor<[1024], si8>
}

// -----

// CHECK-CONST-MO-SUBGRAPH-LABEL: IR Dump After MOToMOGG
// CHECK-CONST-MO-SUBGRAPH: module
// CHECK-CONST-MO-SUBGRAPH: all_const
// CHECK-CONST-MO-SUBGRAPH-DAG: mo.mul
// CHECK-CONST-MO-SUBGRAPH-DAG: mo.sub
// CHECK-CONST-MO-SUBGRAPH-DAG: mo.add

// CHECK-LABEL: @all_const
// CHECK-NEXT: %[[S1:.*]] = mo.constant {value = #M.dense_array<0, 5>
// CHECK-NEXT: mo.output %[[S1]]
mo.graph @all_const() -> (!mo.tensor<[2], si32>) {
  %s1 = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>
  %s2 = mo.constant {value = #M.dense_array<6, 3> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>

  %s3 = mo.mul(%s1, %s2) : !mo.tensor<[2], si32>
  %s4 = mo.sub(%s3, %s1) : !mo.tensor<[2], si32>
  %s5 = mo.add(%s4, %s3) : !mo.tensor<[2], si32>
  mo.output %s5 : !mo.tensor<[2], si32>
}

// -----

// CHECK-CONST-MO-SUBGRAPH-LABEL: IR Dump After MOToMOGG
// CHECK-CONST-MO-SUBGRAPH: multiple_subgraphs
// CHECK-CONST-MO-SUBGRAPH-DAG: mo.mul
// CHECK-CONST-MO-SUBGRAPH-DAG: mo.sub

// CHECK: multiple_subgraphs(%[[R1:.*]]:
mo.graph @multiple_subgraphs(%r1 : !mo.tensor<[2], si32>) -> (!mo.tensor<[2], si32>) {
  %r2 = mo.constant {value = #M.dense_array<1, -1> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>
  %r3 = mo.constant {value = #M.dense_array<6, 3> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>

  // CHECK-DAG: %[[R4:.*]] = mo.constant {value = #M.dense_array<1, 1>
  %r4 = mo.mul(%r2, %r2) : !mo.tensor<[2], si32>
  // CHECK-DAG: %[[R5:.*]] = mo.constant {value = #M.dense_array<5, 4>
  %r5 = mo.sub(%r3, %r2) : !mo.tensor<[2], si32>
  // CHECK-DAG = mo.sub(%[[R1]], %[[R5]])
  %r6 = mo.sub(%r1, %r5) : !mo.tensor<[2], si32>
  // CHECK-DAG = mo.sub(%[[R1]], %[[R4]])
  %r7 = mo.sub(%r1, %r4) : !mo.tensor<[2], si32>
  %r8 = mo.sub(%r6, %r7) : !mo.tensor<[2], si32>
  mo.output %r8 : !mo.tensor<[2], si32>
}

// -----

// TODO: if should be handled by applying connonicalization

// CHECK-CONST-MO-SUBGRAPH-LABEL: IR Dump After MOToMOGG
// CHECK-CONST-MO-SUBGRAPH: if_not_handled
// CHECK-CONST-MO-SUBGRAPH: mo.mul

// CHECK: if_not_handled
mo.graph @if_not_handled(%e1 : !mo.tensor<[2], si32>) -> (!mo.tensor<[2], si32>) {
  // CHECK-DAG: %[[E2:.*]] = mo.constant {value = #M.dense_array<0, 9>
  %e2 = mo.constant {value = #M.dense_array<0, 3> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>
  // CHECK-DAG: %{{.*}} = mo.constant {value = #M.dense_array<0, 6>
  // CHECK-DAG: %{{.*}} = mo.constant {value = #M.dense_array<1>
  %e3 = mo.constant {value = #M.dense_array<1> : tensor<1xi8>} :
				!mo.tensor<[], bool>

  // CHECK-NEXT: mo.if
  // CHECK-NEXT: mo.yield %[[E2]]
  %e4 = mo.if(%e3 : !mo.tensor<[], bool>) -> !mo.tensor<[2], si32> {
    %e5 = mo.mul(%e2, %e2) : !mo.tensor<[2], si32>
    mo.yield %e5 : !mo.tensor<[2], si32>
  } else {
    %e6 = mo.add(%e2, %e2) : !mo.tensor<[2], si32>
    mo.yield %e6 : !mo.tensor<[2], si32>
  }
  mo.output %e4 : !mo.tensor<[2], si32>
}

// -----

// CHECK-CONST-MO-SUBGRAPH-LABEL: @nested_region_constant_subgraphs
// CHECK-CONST-MO-SUBGRAPH: mo.mul

// CHECK: nested_region
mo.graph @nested_region(%g1 : !mo.tensor<[], bool>) -> (!mo.tensor<[2], si32>) {
  // CHECK-DAG: %[[G0:.*]] = mo.constant {value = #M.dense_array<0, 1>
  // CHECK-DAG: %[[G1:.*]] = mo.constant {value = #M.dense_array<6, 1>
  // CHECK-NEXT: mo.if
  %g2 = mo.if(%g1 : !mo.tensor<[], bool>) -> !mo.tensor<[2], si32> {
    // CHECK-NEXT: mo.yield %[[G0]]
    %g3 = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>
    %g4 = mo.mul(%g3, %g3) : !mo.tensor<[2], si32>
    mo.yield %g4 : !mo.tensor<[2], si32>
  // CHECK-NEXT: else
  } else {
    // CHECK-NEXT: mo.yield %[[G1]]
    %g3 = mo.constant {value = #M.dense_array<6, 1> : tensor<2xsi32>} :
				!mo.tensor<[2], si32>
    mo.yield %g3 : !mo.tensor<[2], si32>
  }
  mo.output %g2 : !mo.tensor<[2], si32>
}

// -----

// CHECK: @repeated_application
mo.graph @repeated_application(%e0 : !mo.tensor<[], si32>) -> !mo.tensor<[], si32> {
  // CHECK-NEXT: %{{.*}} = mo.constant {value = #M.dense_array<4>
  // CHECK-NEXT: mo.output
  %e1 = mo.constant {value = #M.dense_array<3> : tensor<1xsi32>} :
				!mo.tensor<[], si32>
  %e2 = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} :
				!mo.tensor<[], si32>
  %e3 = mo.add(%e1, %e1) : !mo.tensor<[], si32>
  %e4 = mo.greater_equal(%e3, %e2) :(!mo.tensor<[], si32>, !mo.tensor<[], si32>)
    -> !mo.tensor<[], bool>
  %e5 = mo.select(%e4, %e3, %e0) :
    (!mo.tensor<[], bool>, !mo.tensor<[], si32>, !mo.tensor<[], si32>)
    -> !mo.tensor<[], si32>
  %e6 = mo.sub(%e5, %e2) : !mo.tensor<[], si32>
  mo.output %e6 : !mo.tensor<[], si32>
}

// -----

// COM: Should not constant fold due to presence of parameters.
// COM: Must run shape inference first to get @shape_func graph which will fold
// CHECK-LABEL: @incomplete_shape_func
// CHECK: mo.invoke_shape_func["pack_matmul_b_shape_func"]
mo.graph @incomplete_shape_func() -> (!mo.tensor<[?, ?], f32>) {
  %1 = mo.constant {value = #M.dense_array<1.> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %2 = mo.static.broadcast_to(%1):!mo.tensor<[1], f32> -> !mo.tensor<[768, 768], f32, {layout = #mo.layout<KN>}>
  mo.invoke_shape_func["pack_matmul_b_shape_func"]<() -> MMPackSh0, MMPackSh1>(%2) {kgenParams = {a_shape = dense<[-9223372036854775808, 768]> : vector<2xi64>, a_type = f32, b_shape = dense<[768, 768]> : vector<2xi64>, b_type = f32, c_shape = dense<[-9223372036854775808, 768]> : vector<2xi64>, c_type = f32, transpose_in_0 = false}} : (!mo.tensor<[768, 768], f32, {layout = #mo.layout<KN>}>)
  %3 = mo.layout.transform(%2) {kgenParams = {a_shape = dense<[-9223372036854775808, 768]> : vector<2xi64>, a_type = f32, b_shape = dense<[768, 768]> : vector<2xi64>, b_type = f32, c_shape = dense<[-9223372036854775808, 768]> : vector<2xi64>, c_type = f32}} : !mo.tensor<[768, 768], f32, {layout = #mo.layout<KN>}> -> !mo.tensor<[MMPackSh0, MMPackSh1], f32, {layout = #mo.layout<KNkni>}>
  %4 = mo.rebind(%3): !mo.tensor<[MMPackSh0, MMPackSh1], f32, {layout = #mo.layout<KNkni>}> -> !mo.tensor<[?, ?], f32>
  mo.output %4: !mo.tensor<[?, ?], f32>
}

// -----

// CHECK-LABEL: mo.graph @multiple_internal_ops
// CHECK-SAME: (%[[ARG0:.*]]:
// CHECK-NEXT: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<{{.*}}> : !M.array<6xf32>} : !mo.tensor<[3, 2], f32>
// CHECK-NEXT: %[[VAL1:.*]] = mo.matmul(%[[ARG0]], %[[VAL0]])
// CHECK-NEXT: mo.output %[[VAL1]]
mo.graph @multiple_internal_ops(%arg0: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[1, 2], f32> {
  %c1 = mo.constant {value = #M.dense_array<-1.0> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %1 = mo.static.broadcast_to(%c1) : !mo.tensor<[1], f32> -> !mo.tensor<[1, 3], f32>
  %3 = mo.constant {value = #M.dense_array<1, 0> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
  %4 = mo.static.broadcast_to(%c1) : !mo.tensor<[1], f32> -> !mo.tensor<[2, 3], f32>
  %16 = mo.transpose(%4, %3) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>
  %17 = mo.matmul(%arg0, %16) : (!mo.tensor<[1, 3], f32>, !mo.tensor<[3, 2], f32>) -> !mo.tensor<[1, 2], f32>
  mo.output %17 : !mo.tensor<[1, 2], f32>
}

// -----

// Check that various operations on lists are constant folded.
// CHECK-LABEL: mo.graph @list
// CHECK-NOT: mo.concat_from_list
// CHECK-NOT: mo.list.insert
// CHECK: mo.list.create
// CHECK: mo.output
mo.graph @list() -> (!mo.tensor<[2], f32>, !mo.list<!mo.tensor<[1], f32>>) {
  %0 = mo.constant {value = #M.dense_array<1.000000e+00> : !M.array<1xf32>} : !mo.tensor<[1], f32>
  %zero = mo.constant {value = #M.dense_array<0.000000e+00> : !M.array<1xf32>} : !mo.tensor<[1], f32>
  %axis = mo.constant {value = #M.dense_array<0> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %idx = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %list = mo.list.create(%0, %0) :(!mo.tensor<[1], f32>, !mo.tensor<[1], f32>) -> !mo.list<!mo.tensor<[1], f32>>
  %1 = mo.concat_from_list[%axis: !mo.tensor<[], si64>](%list) : !mo.list<!mo.tensor<[1], f32>> -> !mo.tensor<[2], f32>
  %2 = mo.list.insert(%list, %zero, %idx) : (!mo.list<!mo.tensor<[1], f32>>, !mo.tensor<[1], f32>, !mo.tensor<[], si64>) -> !mo.list<!mo.tensor<[1], f32>>
  mo.output %1, %2 :!mo.tensor<[2], f32>, !mo.list<!mo.tensor<[1], f32>>
}

// -----

// Check that transitive dependencies are removed in the policy.
// CHECK-LABEL: @transitive_remove
// CHECK: mo.output
mo.graph @transitive_remove(%arg0: !mo.tensor<[11, 4, 4, 6, 5], f32>) -> !mo.tensor<[11, 2, 2, 2, 10], f32> {
  %0 = mo.constant {value = #M.dense_array<0, 0, 0, 0, 0, 0> : tensor<6xsi64>} : !mo.tensor<[6], si64>
  %1 = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %2 = mo.constant {value = #M.dense_array<2, 2, 2> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %3 = mo.constant {value = #M.dense_array<3> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %4 = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %5 = mo.constant {value = #M.dense_array<4> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %6 = mo.constant {value = #M.dense_array<1.000000e+00> : !M.array<1xf32>} : !mo.tensor<[], f32>
  %7 = mo.static.broadcast_to(%6) : !mo.tensor<[], f32> -> !mo.tensor<[2, 2, 3, 1, 10], f32>
  %8 = mo.shape_of(%arg0) : (!mo.tensor<[11, 4, 4, 6, 5], f32>) -> !mo.tensor<[5], si64>
  %9 = mo.shape_of(%7) : (!mo.tensor<[2, 2, 3, 1, 10], f32>) -> !mo.tensor<[5], si64>
  %10 = mo.gather(%8, %5, %4) : (!mo.tensor<[5], si64>, !mo.tensor<[1], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[1], si64>
  %11 = mo.gather(%9, %3, %4) : (!mo.tensor<[5], si64>, !mo.tensor<[1], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[1], si64>
  %12 = mo.div(%10, %11) : !mo.tensor<[1], si64>
  %13 = mo.static.reshape(%12) : !mo.tensor<[1], si64> -> !mo.tensor<[], si64>
  %14 = mo.conv(%arg0, %7) [strides = %2, dilations = %1, paddings = %0, num_groups = %13] : (!mo.tensor<[11, 4, 4, 6, 5], f32>, !mo.tensor<[2, 2, 3, 1, 10], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[11, 2, 2, 2, 10], f32>
  mo.output %14 : !mo.tensor<[11, 2, 2, 2, 10], f32>
}

// -----

// Check that mogg ops are folded
// CHECK-LABEL: @fold_mogg_ops
// CHECK-NOT: mogg.concat
// CHECK: mo.output
mo.graph @fold_mogg_ops() -> !mo.tensor<[2], si64> {
  %1 = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %2 = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %3 = mogg.concat(%1, %2) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  mo.output %3 : !mo.tensor<[2], si64>
}

// -----

// CHECK-LABEL: mo.graph @large_op
// CHECK-DAG: %{{.*}} = mo.constant {value = #M.dense_array<3, 1> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-DAG: %{{.*}} = mo.constant {value = #M.dense_array<3> : !M.array<1xsi64>} : !mo.tensor<[1], si64>
// CHECK-DAG: %{{.*}} = mo.constant {value = #M.dense_array<0> : !M.array<1xsi64>} : !mo.tensor<[1], si64>
// CHECK-DAG: %{{.*}} = mo.constant {value = #M.dense_array<0> : !M.array<1xsi32>} : !mo.tensor<[], si32>
// CHECK-DAG: %{{.*}} = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[1], si64>
// CHECK-DAG: %{{.*}} = mo.constant {value = #M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00> : tensor<3xf32>} : !mo.tensor<[3], f32>
mo.graph @large_op(%arg0: !mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 2], f32> {
  %0 = mo.constant {value = #M.dense_array<3> : !M.array<1xsi64>} : !mo.tensor<[1], si64>
  %1 = mo.constant {value = #M.dense_array<0> : !M.array<1xsi64>} : !mo.tensor<[1], si64>
  %2 = mo.constant {value = #M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00> : tensor<3xf32>} : !mo.tensor<[3], f32>
  %3 = mo.constant {value = #M.dense_array<0> : !M.array<1xsi32>} : !mo.tensor<[], si32>
  %4 = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[1], si64>
  %5 = mo.shape_of(%2) : (!mo.tensor<[3], f32>) -> !mo.tensor<[1], si64>
  %6 = mo.concat[%3 : !mo.tensor<[], si32>] (%5, %4) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %7 = mo.reshape<() -> D15>(%2, %6) : (!mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D15, 1], f32>
  %8 = mo.shape_of(%arg0) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[3], si64>
  %9 = mo.rebind<() -> D14>(%7) : !mo.tensor<[D15, 1], f32> -> !mo.tensor<[D14, 1], f32>
  %10 = mo.shape_of(%9) : (!mo.tensor<[D14, 1], f32>) -> !mo.tensor<[2], si64>
  %11 = mo.slice(%8, %1, %4, %4) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
  %12 = mo.slice(%8, %4, %0, %4) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %13 = mo.slice(%10, %1, %1, %4) : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[0], si64>
  %14 = mo.broadcast_shape(%11, %13) : (!mo.tensor<[1], si64>, !mo.tensor<[0], si64>) -> !mo.tensor<[1], si64>
  %15 = mo.concat[%3 : !mo.tensor<[], si32>] (%14, %12) : (!mo.tensor<[1], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
  %16 = mo.concat[%3 : !mo.tensor<[], si32>] (%14, %10) : (!mo.tensor<[1], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
  %17 = mo.broadcast_to<() -> D11, D12, D13>(%arg0, %15) : (!mo.tensor<[2, 2, 3], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D11, D12, D13], f32>
  %18 = mo.rebind<() -> D7>(%7) : !mo.tensor<[D15, 1], f32> -> !mo.tensor<[D7, 1], f32>
  %19 = mo.broadcast_to<() -> D8, D9, D10>(%18, %16) : (!mo.tensor<[D7, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D8, D9, D10], f32>
  %20 = mo.rebind<() -> D1, D2, D3>(%17) : !mo.tensor<[D11, D12, D13], f32> -> !mo.tensor<[D1, D2, D3], f32>
  %21 = mo.rebind<() -> D4, D5, D6>(%19) : !mo.tensor<[D8, D9, D10], f32> -> !mo.tensor<[D4, D5, D6], f32>
  %22 = mo.batch_matmul(%20, %21) : (!mo.tensor<[D1, D2, D3], f32>, !mo.tensor<[D4, D5, D6], f32>) -> !mo.tensor<[2, 2, 1], f32>
  %23 = mo.static.reshape(%22) : !mo.tensor<[2, 2, 1], f32> -> !mo.tensor<[2, 2], f32>
  mo.output %23 : !mo.tensor<[2, 2], f32>
}

// -----

// COM: Do not constant fold parametric outputs in graphs. Depend on shape
// COM: inference to clean up first. We can go up to the add in this case.
// CHECK-LABEL: mo.graph
// CHECK-SAME: @partial_fold_parameterized_outputs<() -> O>() -> !mo.tensor<[O], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = dense_resource<folded_const> : !M.array<600xf32>} : !mo.tensor<[600], f32>
// CHECK-NEXT: kgen.param.declare A = <600>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind([[V0]]) : !mo.tensor<[600], f32> -> !mo.tensor<[A], f32>
// CHECK-NEXT: mo.output<A> [[V1]] : !mo.tensor<[A], f32>
mo.graph @partial_fold_parameterized_outputs<() -> O>() -> (!mo.tensor<[O], f32>) {
  %0 = mo.constant {value = dense_resource<large_const> : !M.array<600xf32>} : !mo.tensor<[600], f32>
  %1 = mo.add(%0, %0): !mo.tensor<[600], f32>
  %2 = mo.rebind<() -> A>(%1): !mo.tensor<[600], f32> -> !mo.tensor<[A], f32>
  mo.output<A> %2 : !mo.tensor<[A], f32>
}

{-#
  dialect_resources: {
    builtin: {
      large_const: "0x040000000000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C040"
    }
  }
#-}

// -----

// CHECK-LABEL: @if_cache_resources
// CHECK-SAME: ([[ARG0:%.*]]: !mo.tensor<[], bool>, [[ARG1:%.*]]: !mo.tensor<[600], f32>) -> !mo.tensor<[600], f32> {
mo.graph @if_cache_resources(%arg0: !mo.tensor<[], bool>, %arg1: !mo.tensor<[600], f32>) -> (!mo.tensor<[600], f32>) {
//  CHECK:   [[V0:%.*]] = mo.constant {value = dense_resource<folded_const> : !M.array<600xf32>} : !mo.tensor<[600], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.if([[ARG0]] : !mo.tensor<[], bool>) -> !mo.tensor<[600], f32> {
//  CHECK-NEXT:     [[V2:%.*]] = mo.add([[ARG1]], [[V0]]) : !mo.tensor<[600], f32>
//  CHECK-NEXT:     mo.yield [[V2]] : !mo.tensor<[600], f32>
//  CHECK-NEXT:   } else {
//  CHECK-NEXT:     [[V2]] = mo.mul([[ARG1]], [[V0]]) : !mo.tensor<[600], f32>
//  CHECK-NEXT:     mo.yield [[V2]] : !mo.tensor<[600], f32>
  %0 = mo.constant {value = dense_resource<large_const> : !M.array<600xf32>} : !mo.tensor<[600], f32>
  %1 = mo.if(%arg0 : !mo.tensor<[], bool>) -> !mo.tensor<[600], f32> {
    %2 = mo.add(%0, %0) : !mo.tensor<[600], f32>
    %3 = mo.add(%arg1, %2) : !mo.tensor<[600], f32>
    mo.yield %3 : !mo.tensor<[600], f32>
  } else {
    %2 = mo.add(%0, %0) : !mo.tensor<[600], f32>
    %3 = mo.mul(%arg1, %2) : !mo.tensor<[600], f32>
    mo.yield %3 : !mo.tensor<[600], f32>
  }
  mo.output %1 : !mo.tensor<[600], f32>
}

// Here we have a constant graph with 3 outputs:
//                ┌───────┐
//   ┌───────────►│Low-Mem│
//   │            └───────┘
//   │
// ┌─┴──┬─────────────┐
// │ %0 │             │
// └────┘         ┌───▼──┐
//                │Matmul│
//                └───▲──┘
// ┌────┐             │
// │ %1 │             │
// └─┬──┴─────────────┘
//   │
//   │            ┌────────┐
//   └───────────►│High-Mem│
//                └────────┘
//
// The `Matmul` has an extremely high memory usage so will be cut in pruning.
//
// However this creates two disjoint graphs (%0 --> low-mem) and (%1 --> high-mem).
// Because the folding policy is based on increasing in total memory usage,
// we may allow folding %1 --> high-mem because though it increases memory the
// folder erroneously has a higher threshold due to the inclusion of a disjoint
// graph whose inputs add to the folding memory budget.
//
// This test makes sure we don't fold under this scenario, e.g. the folder
// recognizes when things are disjoint.

// CHECK-LABEL: mo.graph @disjoint_set()
// CHECK-DAG: mo.constant {value = dense_resource<{{.*}}> : !M.array<600xf32>} : !mo.tensor<[1, 600], f32>
// CHECK-DAG: mo.constant {value = dense_resource<{{.*}}> : !M.array<600xf32>} : !mo.tensor<[600], f32>
// CHECK-DAG: mo.constant {value = dense_resource<{{.*}}> : !M.array<600xf32>} : !mo.tensor<[600], f32>

// No more constants
// CHECK-NOT: mo.constant
mo.graph @disjoint_set() -> (!mo.tensor<[600, 600], f32>, !mo.tensor<[1, 600], f32>, !mo.tensor<[3, 600], f32>) {
  %0 = mo.constant {value = dense_resource<large_const> : !M.array<600xf32>} : !mo.tensor<[600], f32>
  %1 = mo.constant {value = dense_resource<large_const_alt> : !M.array<600xf32>} : !mo.tensor<[600], f32>

  %00 = mo.static.broadcast_to(%0): !mo.tensor<[600], f32> -> !mo.tensor<[600, 600], f32>
  %11 = mo.static.broadcast_to(%1): !mo.tensor<[600], f32> -> !mo.tensor<[600, 600], f32>

  // Should be pruned.
  %2 = mo.matmul(%00, %11): (!mo.tensor<[600, 600], f32>, !mo.tensor<[600, 600], f32>) -> !mo.tensor<[600, 600], f32>

  // good_bcast does not increase memory usage while bad_bcast does
  // we have an extra add since we try to avoid materializing bcast outputs directly
  %good_bcast = mo.static.broadcast_to(%0): !mo.tensor<[600], f32> -> !mo.tensor<[1, 600], f32>
  %good_add = mo.add(%good_bcast, %good_bcast): !mo.tensor<[1, 600], f32>

  %bad_bcast = mo.static.broadcast_to(%1): !mo.tensor<[600], f32> -> !mo.tensor<[3, 600], f32>
  %bad_add = mo.add(%bad_bcast, %bad_bcast): !mo.tensor<[3, 600], f32>

  // We expect a subgraph where we have 600 elements --> 600 elements and one
  // with 600 elements --> 600 * 3 elements.
  //
  // The default threshold for folding is 2x the initial memory which for a
  // subgraph with two distinct components would be 2 * 600 * 2 = 4 * 600 elements.
  // This allows the materialization of both the 600 * 3 and 600 element path barely.
  //
  // However, these graphs are disjoint after pruning so should be considered separately.
  // In this case, only the 600 --> 600 element path should be folded (as it does not increase mem usage) but
  // the 600 --> 600 * 3 element path (which is greater than the 600 * 2 threshold) does not.
  mo.output %2, %good_add, %bad_add : !mo.tensor<[600, 600], f32>, !mo.tensor<[1, 600], f32>, !mo.tensor<[3, 600], f32>
}


{-#
  dialect_resources: {
    builtin: {
      large_const: "0x040000000000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C040",
      large_const_alt: "0x040000000000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C0400000000000000040000080400000C040"    }
  }
#-}

// -----

// CHECK-CONST-MO-SUBGRAPH-LABEL: @partial_fold_bcast_constant_subgraphs
// CHECK-CONST-MO-SUBGRAPH: mo.static.broadcast_to

// COM: For the reshape impl:
// CHECK-CONST-MO-SUBGRAPH: @copy_view

// COM: Do not constant fold broadcasts in the output, but allow them internally
// CHECK-LABEL: mo.graph
// CHECK-SAME: @partial_fold_bcast() -> !mo.tensor<[4, 2, 4], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : !M.array<8xf32>} : !mo.tensor<[1, 2, 4], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[1, 2, 4], f32> -> !mo.tensor<[4, 2, 4], f32>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[4, 2, 4], f32>
mo.graph @partial_fold_bcast() -> !mo.tensor<[4, 2, 4], f32> {
  %0 = mo.constant {value = #M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : !M.array<4xf32>} : !mo.tensor<[4], f32>

  // COM: Here we should constant fold up to the reshape.
  // COM: Note all of these increase memory but should be under the default
  // COM: threshold of 100 bytes.
  %1 = mo.static.broadcast_to(%0) : !mo.tensor<[4], f32> -> !mo.tensor<[2, 4], f32>
  %2 = mo.static.reshape(%1): !mo.tensor<[2, 4], f32> -> !mo.tensor<[1, 2, 4], f32>

  // COM: Do not fold up to the below because it is a broadcast op.
  %3 = mo.static.broadcast_to(%2) : !mo.tensor<[1, 2, 4], f32> -> !mo.tensor<[4, 2, 4], f32>
  mo.output %3 : !mo.tensor<[4, 2, 4], f32>
}
