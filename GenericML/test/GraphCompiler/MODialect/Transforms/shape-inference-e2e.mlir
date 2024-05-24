// Note: due to the size of the file, we want to filecheck with --enable-var-scope
// which prevents variables from being reused across CHECK-LABEL boundaries.
//
// RUN: mo-opt %s --split-input-file --shape-inference --hoist-param-exprs | FileCheck %s --enable-var-scope
// CHECK-NOT: kgen.func

mo.graph @static_ranks(%arg0: !mo.tensor<[?, ?], f32>, %arg1: !mo.tensor<[?, ?], f32>) -> !mo.tensor<?, f32> {
  %arg0c = mo.rebind(%arg0) : !mo.tensor<[?, ?], f32> -> !mo.tensor<?, f32>
  %arg1c = mo.rebind(%arg1) : !mo.tensor<[?, ?], f32> -> !mo.tensor<?, f32>
  %0 = mo.add(%arg0c, %arg1c) : !mo.tensor<?, f32>
  %1 = mo.gelu(%0) : !mo.tensor<?, f32>
  %2 = mo.sub(%1, %0) : !mo.tensor<?, f32>
  %3 = mo.relu(%2) : !mo.tensor<?, f32>
  mo.output %3 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @static_ranks<[[D11:.*]], [[D12:.*]], [[D13:.*]], [[D14:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D11]], [[D12]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D13]], [[D14]]], f32>) -> !mo.tensor<[[[D11]], [[D12]]], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.rebind([[VARG1]]) : !mo.tensor<[[[D13]], [[D14]]], f32> -> !mo.tensor<[[[D11]], [[D12]]], f32>
// CHECK-NEXT:   kgen.param.declare SORS_CH10: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[add] input shapes must match" : !kgen.string}, and(eq([[D11]], [[D13]]), eq([[D12]], [[D14]])))>
// CHECK-NEXT:   [[V1:%.*]] = mo.add([[VARG0]], [[V0]]) : !mo.tensor<[[[D11]], [[D12]]], f32>
// CHECK-NEXT:   [[V2:%.*]] = mo.gelu([[V1]]) : !mo.tensor<[[[D11]], [[D12]]], f32>
// CHECK-NEXT:   [[V3:%.*]] = mo.sub([[V2]], [[V1]]) : !mo.tensor<[[[D11]], [[D12]]], f32>
// CHECK-NEXT:   [[V4:%.*]] = mo.relu([[V3]]) : !mo.tensor<[[[D11]], [[D12]]], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[[[D11]], [[D12]]], f32>
// CHECK-NEXT: }

mo.graph @batch_matmul_static_rank(
    %arg0: !mo.tensor<[?, ?, ?], f32>,
    %arg1: !mo.tensor<[?, ?, ?], f32>
) -> !mo.tensor<[?, ?, ?], f32> {
  %out = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %out : !mo.tensor<[?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @batch_matmul_static_rank<[[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]], [[D6:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[[D1]], [[D2]], [[D6]]], f32> {
// CHECK-DAG:    kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[batch_matmul] inputs inner dimensions must match" : !kgen.string}, eq([[D3]], [[D5]]))>
// CHECK-DAG:    kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[batch_matmul] inputs batch dimensions must match" : !kgen.string}, eq([[D1]], [[D4]]))>
// CHECK-NEXT:   [[V1:%.*]] = mo.batch_matmul([[VARG0]], [[VARG1]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[[D1]], [[D2]], [[D6]]], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D1]], [[D2]], [[D6]]], f32>
// CHECK-NEXT: }

mo.graph @materialize_gather_axis_0_si32_e2e(
  %input: !mo.tensor<[?, 2], f32>,
  %indices: !mo.tensor<[2, ?], si32>
) -> !mo.tensor<[?, ?, ?], f32> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %result = mo.gather(%input, %indices, %axis) : (!mo.tensor<[?, 2], f32>, !mo.tensor<[2, ?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %result : !mo.tensor<[?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @materialize_gather_axis_0_si32_e2e<[[D6:.*]], [[D7:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D6]], 2], f32>, [[VARG1:%.*]]: !mo.tensor<[2, [[D7]]], si32>) -> !mo.tensor<[2, [[D7]], 2], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.gather([[VARG0]], [[VARG1]], [[V0]]) : (!mo.tensor<[[[D6]], 2], f32>, !mo.tensor<[2, [[D7]]], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[2, [[D7]], 2], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[2, [[D7]], 2], f32>
// CHECK-NEXT: }

mo.graph @materialize_gather_axis_neg_1_si32_e2e(
  %input: !mo.tensor<[?, 2], f32>,
  %indices: !mo.tensor<[2, ?], si32>
) -> !mo.tensor<[?, ?, ?], f32> {
  %axis = mo.constant {value = #M.dense_array<-1> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %result = mo.gather(%input, %indices, %axis) : (!mo.tensor<[?, 2], f32>, !mo.tensor<[2, ?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %result : !mo.tensor<[?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @materialize_gather_axis_neg_1_si32_e2e<[[D6:.*]], [[D7:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D6]], 2], f32>, [[VARG1:%.*]]: !mo.tensor<[2, [[D7]]], si32>) -> !mo.tensor<[[[D6]], 2, [[D7]]], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<-1> : tensor<1xsi32>} : !mo.tensor<[], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.gather([[VARG0]], [[VARG1]], [[V0]]) : (!mo.tensor<[[[D6]], 2], f32>, !mo.tensor<[2, [[D7]]], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[D6]], 2, [[D7]]], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D6]], 2, [[D7]]], f32>
// CHECK-NEXT: }

mo.graph @materialize_mean_axis_0_si32_e2e(
  %input: !mo.tensor<[?, 2], f32>
) -> !mo.tensor<[?, ?], f32> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %result = mo.mean(%input, %axis) : (!mo.tensor<[?, 2], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?], f32>
  mo.output %result : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @materialize_mean_axis_0_si32_e2e<[[D4:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D4]], 2], f32>) -> !mo.tensor<[1, 2], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.mean([[VARG0]], [[V0]]) : (!mo.tensor<[[[D4]], 2], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[1, 2], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[1, 2], f32>
// CHECK-NEXT: }

mo.graph @materialize_prod_axis_1_si64_e2e(
  %input: !mo.tensor<[?, 2], f32>
) -> !mo.tensor<[?, ?], f32> {
  %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %result = mo.reduce.mul(%input, %axis) : (!mo.tensor<[?, 2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %result : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @materialize_prod_axis_1_si64_e2e<[[D4:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D4]], 2], f32>) -> !mo.tensor<[[[D4]], 1], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.reduce.mul([[VARG0]], [[V0]]) : (!mo.tensor<[[[D4]], 2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D4]], 1], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D4]], 1], f32>
// CHECK-NEXT: }

mo.graph @materialize_sum_axis_neg_1_si64_e2e(
  %input: !mo.tensor<[?, 2], f32>
) -> !mo.tensor<[?, ?], f32> {
  %axis = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %result = mo.reduce.add(%input, %axis) : (!mo.tensor<[?, 2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %result : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @materialize_sum_axis_neg_1_si64_e2e<[[D4:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D4]], 2], f32>) -> !mo.tensor<[[[D4]], 1], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.reduce.add([[VARG0]], [[V0]]) : (!mo.tensor<[[[D4]], 2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D4]], 1], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D4]], 1], f32>
// CHECK-NEXT: }

mo.graph @unresolved_rebinds(
  %arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>
) -> !mo.tensor<?, f32> {
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<?, f32>
  %1 = mo.gelu(%0) : !mo.tensor<?, f32>
  %2 = mo.batch_matmul(%1, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  %3 = mo.relu(%2) : !mo.tensor<?, f32>
  %4 = mo.gelu(%3) : !mo.tensor<?, f32>
  %5 = mo.sub(%4, %3) : !mo.tensor<?, f32>
  mo.output %5 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @unresolved_rebinds<[[Sh11:.*]]: !mosh.ape, [[Sh12:.*]]: !mosh.ape -> [[Sh13:.*]]: !mosh.ape>([[VARG0:%.*]]: !mo.tensor<[[Sh11]], f32>, [[VARG1:%.*]]: !mo.tensor<[[Sh12]], f32>) -> !mo.tensor<[[Sh13]], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.rebind([[VARG1]]) : !mo.tensor<[[Sh12]], f32> -> !mo.tensor<[[Sh11]], f32>
// CHECK-NEXT:   kgen.param.declare SORS_EQ16: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[Sh11]], [[Sh12]])>
// CHECK-NEXT:   kgen.param.declare SORS_CH17: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[add] input shapes must match" : !kgen.string}, SORS_EQ16)>
// CHECK-NEXT:   [[V1:%.*]] = mo.add([[VARG0]], [[V0]]) : !mo.tensor<[[Sh11]], f32>
// CHECK-NEXT:   [[V2:%.*]] = mo.gelu([[V1]]) : !mo.tensor<[[Sh11]], f32>
// CHECK-NEXT:   mo.invoke_shape_func["batch_matmul_shape"]<() -> [[Sh9:.*]]: !mosh.ape>([[V2]], [[VARG1]]) {dataDeptTensors = []} : (!mo.tensor<[[Sh11]], f32>, !mo.tensor<[[Sh12]], f32>)
// CHECK-NEXT:   [[V4:%.*]] = mo.batch_matmul([[V2]], [[VARG1]]) : (!mo.tensor<[[Sh11]], f32>, !mo.tensor<[[Sh12]], f32>) -> !mo.tensor<[[Sh9]], f32>
// CHECK-NEXT:   [[V5:%.*]] = mo.relu([[V4]]) : !mo.tensor<[[P27:.*]], f32>
// CHECK-NEXT:   [[V6:%.*]] = mo.gelu([[V5]]) : !mo.tensor<[[P27:.*]], f32>
// CHECK-NEXT:   [[V7:%.*]] = mo.sub([[V6]], [[V5]]) : !mo.tensor<[[P27:.*]], f32>
// CHECK-NEXT:   mo.output<:!mosh.ape [[P27:.*]]> [[V7]] : !mo.tensor<[[P27:.*]], f32>
// CHECK-NEXT: }

mo.graph @broadcasted_binary_elwise(
  %arg0: !mo.tensor<[?, 12, ?], f32>, %arg1: !mo.tensor<[?, ?], f32>
) -> (!mo.tensor<[?, ?, ?], f32>) {
  %shape1 = mo.shape_of(%arg0) : (!mo.tensor<[?, 12, ?], f32>) -> !mo.tensor<[3], si64>
  %shape2 = mo.shape_of(%arg1) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<[2], si64>
  %shape = mo.broadcast_shape(%shape1, %shape2) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>

  %0 = mo.broadcast_to(%arg0, %shape) : (!mo.tensor<[?, 12, ?], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[?, ?, ?], f32>
  %1 = mo.broadcast_to(%arg1, %shape) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[?, ?, ?], f32>
  %3 = mo.add(%0, %1) : !mo.tensor<[?, ?, ?], f32>
  mo.output %3 : !mo.tensor<[?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcasted_binary_elwise<[[D21:.*]], [[D22:.*]], [[D23:.*]], [[D24:.*]] -> [[D27:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D21]], 12, [[D22]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D23]], [[D24]]], f32>) -> !mo.tensor<[[[D21]], 12, [[D27]]], f32> {
// CHECK-NEXT:   kgen.param.declare {{.*}}"mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(eq([[D23]], 1), eq([[D23]], 12)))>
// CHECK-NEXT:   kgen.param.declare {{.*}}"mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(eq([[D22]], [[D24]]), eq([[D22]], 1), eq([[D24]], 1)))>
// CHECK-NEXT:   kgen.param.declare {{.*}}"mo.assert"{message = "[broadcast_to] input dim must be either 1 or equal to corresponding output dim starting from the rightmost dim" : !kgen.string}, or(eq(cond(eq([[D22]], 1), [[D24]], [[D22]]), [[D22]]), eq([[D22]], 1)))>
// CHECK-DAG:    kgen.param.declare [[EXPR:.*]] = <cond(eq([[D22]], 1), [[D24]], [[D22]])>
// CHECK-DAG:    [[V0:%.*]] = mo.static.broadcast_to([[VARG0]]) : !mo.tensor<[[[D21]], 12, [[D22]]], f32> -> !mo.tensor<[[[D21]], 12, [[EXPR]]], f32>
// CHECK-DAG:    kgen.param.declare {{.*}} "mo.assert"{message = "[broadcast_to] input dim must be either 1 or equal to corresponding output dim starting from the rightmost dim" : !kgen.string}, or(eq(cond(eq([[D22]], 1), [[D24]], [[D22]]), [[D24]]), eq([[D24]], 1)))>
// CHECK-DAG:    kgen.param.declare {{.*}} "mo.assert"{message = "[broadcast_to] input dim must be either 1 or equal to corresponding output dim starting from the rightmost dim" : !kgen.string}, or(eq([[D23]], 1), eq([[D23]], 12)))>
// CHECK-DAG:    [[V1:%.*]] = mo.static.broadcast_to([[VARG1]]) : !mo.tensor<[[[D23]], [[D24]]], f32> -> !mo.tensor<[[[D21]], 12, [[EXPR]]], f32>
// CHECK-NEXT:   [[V2:%.*]] = mo.add([[V0]], [[V1]]) : !mo.tensor<[[[D21]], 12, [[EXPR:.*]]], f32>
// CHECK-NEXT:   [[V3:%.*]] = mo.rebind([[V2]]) : !mo.tensor<[[[D21]], 12, [[EXPR]]], f32> -> !mo.tensor<[[[D21]], 12, cond(eq([[D22]], 1), [[D24]], [[D22]])], f32>
// CHECK-NEXT:   mo.output<cond(eq([[D22]], 1), [[D24]], [[D22]])> [[V3]] : !mo.tensor<[[[D21]], 12, cond(eq([[D22]], 1), [[D24]], [[D22]])], f32>
// CHECK-NEXT: }

mo.graph @transpose_static_rank(
  %input0: !mo.tensor<[?, ?], f32>,
  %input1: !mo.tensor<[?, ?], f32>
) -> (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>) {
  %perm0 = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %perm1 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %output0 = mo.transpose(%input0, %perm0) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  %output1 = mo.transpose(%input1, %perm1) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %output0, %output1 : !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @transpose_static_rank<[[D9:.*]], [[D10:.*]], [[D11:.*]], [[D12:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D9]], [[D10]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D11]], [[D12]]], f32>) -> (!mo.tensor<[[[D9]], [[D10]]], f32>, !mo.tensor<[[[D12]], [[D11]]], f32>) {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.transpose([[VARG1]], [[V0]]) : (!mo.tensor<[[[D11]], [[D12]]], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[D12]], [[D11]]], f32>
// CHECK-NEXT:   mo.output [[VARG0]], [[V1]] : !mo.tensor<[[[D9]], [[D10]]], f32>, !mo.tensor<[[[D12]], [[D11]]], f32>
// CHECK-NEXT: }

mo.graph @scalar_to_tensor_reshapes_before_staticization(
    %arg: !mo.tensor<[?, ?], si32>
  ) -> !mo.tensor<[?, ?], si32> {
  %scalar_one = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %scalar_zero = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %tensor_zero = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %tensor_one = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %arg_shape = mo.shape_of(%arg) : (!mo.tensor<[?, ?], si32>) -> !mo.tensor<[2], si32>
  %arg_dim0 = mo.slice(%arg_shape, %tensor_zero, %tensor_one, %tensor_one) : (!mo.tensor<[2], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[?], si32>
  %empty_tensor = mo.range(%scalar_zero, %scalar_zero, %scalar_one) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %one_ = mo.gather(%tensor_one, %empty_tensor, %scalar_zero) : (!mo.tensor<[1], si32>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %one_one = mo.concat[%scalar_zero : !mo.tensor<[], si32>] (%one_, %one_) : (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[?], si64>
  %arg_dim0_rs1 = mo.reshape(%arg_dim0, %one_one) : (!mo.tensor<[?], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<[], si32>
  %arg_dim0_rs2 = mo.reshape(%arg_dim0_rs1, %tensor_one) : (!mo.tensor<[], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[?], si32>
  %arg_dim0_dim0 = mo.concat[%scalar_zero : !mo.tensor<[], si32>] (%arg_dim0_rs2, %arg_dim0_rs2)
  : (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[2], si32>
  %res = mo.reshape(%arg, %arg_dim0_dim0) : (!mo.tensor<[?, ?], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[?, ?], si32>
  mo.output %res : !mo.tensor<[?, ?], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @scalar_to_tensor_reshapes_before_staticization<[[D18:.*]], [[D19:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D18]], [[D19]]], si32>) -> !mo.tensor<[[[D18]], [[D18]]], si32> {
// CHECK-NEXT:   kgen.param.declare [[P29:.*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[reshape] input and output number of elements must match" : !kgen.string}, eq(mul_nuw([[D18]], [[D18]]), mul_nuw([[D18]], [[D19]])))>
// CHECK-NEXT:   [[V0:%.*]] = mo.rebind([[VARG0]]) : !mo.tensor<[[[D18]], [[D19]]], si32> -> !mo.tensor<[[[D18]], [[D18]]], si32>
// CHECK-NEXT:   mo.output [[V0]] : !mo.tensor<[[[D18]], [[D18]]], si32>
// CHECK-NEXT: }

mo.graph @binary_op_asserts(
  %arg1: !mo.tensor<[?, ?], f32>,
  %arg2: !mo.tensor<[?, ?], f32>
) -> (!mo.tensor<[?, ?], f32>) {
  %add11 = mo.add(%arg1, %arg1) : !mo.tensor<[?, ?], f32>
  %add12 = mo.add(%arg1, %arg2) : !mo.tensor<[?, ?], f32>
  %res = mo.add(%add11, %add12) : !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_op_asserts<[[D13:.*]], [[D14:.*]], [[D15:.*]], [[D16:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D13]], [[D14]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D15]], [[D16]]], f32>) -> !mo.tensor<[[[D13]], [[D14]]], f32> {
// CHECK:        [[V0:%.*]] = mo.add([[VARG0]], {{.*}}) : !mo.tensor<[[[D13]], [[D14]]], f32>
// CHECK-NEXT:   [[V1:%.*]] = mo.rebind([[VARG1]]) : !mo.tensor<[[[D15]], [[D16]]], f32> -> !mo.tensor<[[[D13]], [[D14]]], f32>
// CHECK-NEXT:   kgen.param.declare SORS_CH12: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[add] input shapes must match" : !kgen.string}, and(eq([[D13]], [[D15]]), eq([[D14]], [[D16]])))>
// CHECK-NEXT:   [[V2:%.*]] = mo.add([[VARG0]], [[V1]]) : !mo.tensor<[[[D13]], [[D14]]], f32>
// CHECK-NEXT:   [[V3:%.*]] = mo.add([[V0]], [[V2]]) : !mo.tensor<[[[D13]], [[D14]]], f32>
// CHECK-NEXT:   mo.output [[V3]] : !mo.tensor<[[[D13]], [[D14]]], f32>
// CHECK-NEXT: }

mo.graph @binary_op_no_assert(
  %arg1: !mo.tensor<[?, ?], f32>
) -> (!mo.tensor<[?, ?], f32>) {
  %add11 = mo.add(%arg1, %arg1) : !mo.tensor<[?, ?], f32>
  %res = mo.add(%arg1, %add11) : !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_op_no_assert<[[D7:.*]], [[D8:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D7]], [[D8]]], f32>) -> !mo.tensor<[[[D7]], [[D8]]], f32> {
// CHECK:        [[V0:%.*]] = mo.add([[VARG0]], {{.*}}) : !mo.tensor<[[[D7]], [[D8]]], f32>
// CHECK-NEXT:   [[V1:%.*]] = mo.add([[VARG0]], [[V0]]) : !mo.tensor<[[[D7]], [[D8]]], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D7]], [[D8]]], f32>
// CHECK-NEXT: }

mo.graph @batch_matmul_reused_rhs<D0, D1 -> Sh0: !mosh.ape>(
  %arg0: !mo.tensor<[3, D0, 4], f32>,
  %arg1: !mo.tensor<[3, 4, D1], f32>
) -> !mo.tensor<Sh0, f32> {
  %arg0reb = mo.rebind<() -> Sh1: !mosh.ape>(%arg0) : !mo.tensor<[3, D0, 4], f32> -> !mo.tensor<Sh1, f32>
  %arg1reb = mo.rebind<() -> Sh2: !mosh.ape>(%arg1) : !mo.tensor<[3, 4, D1], f32> -> !mo.tensor<Sh2, f32>
  %bmm1 = mo.batch_matmul<() -> Sh3: !mosh.ape>(%arg0reb, %arg1reb) : (!mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32>
  %bmm2 = mo.batch_matmul<() -> Sh4: !mosh.ape>(%bmm1, %arg1reb) : (!mo.tensor<Sh3, f32>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %bmm2 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @batch_matmul_reused_rhs<[[D0:.*]], [[D1:.*]]>([[VARG0:%.*]]: !mo.tensor<[3, [[D0]], 4], f32>, [[VARG1:%.*]]: !mo.tensor<[3, 4, [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.batch_matmul([[VARG0]], [[VARG1]]) : (!mo.tensor<[3, [[D0]], 4], f32>, !mo.tensor<[3, 4, [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32>
// CHECK-NEXT:   kgen.param.declare [[P25:.*]]: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[batch_matmul] inputs inner dimensions must match" : !kgen.string}, eq([[D1]], 4))>
// CHECK-NEXT:   [[V2:%.*]] = mo.batch_matmul([[V0]], [[VARG1]]) : (!mo.tensor<[3, [[D0]], [[D1]]], f32>, !mo.tensor<[3, 4, [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32>
// CHECK-NEXT:   mo.output [[V2]] : !mo.tensor<[3, [[D0]], [[D1]]], f32>
// CHECK-NEXT: }

mo.graph @linalg_solve_reused_rhs<D0, D1 -> Sh0: !mosh.ape>(
  %arg0: !mo.tensor<[3, D0, D0], f32>,
  %arg1: !mo.tensor<[3, D0, D1], f32>
) -> !mo.tensor<Sh0, f32> {
  %arg0reb = mo.rebind<() -> Sh1: !mosh.ape>(%arg0) : !mo.tensor<[3, D0, D0], f32> -> !mo.tensor<Sh1, f32>
  %arg1reb = mo.rebind<() -> Sh2: !mosh.ape>(%arg1) : !mo.tensor<[3, D0, D1], f32> -> !mo.tensor<Sh2, f32>
  %0 = mo.linalg.solve<() -> Sh3: !mosh.ape>(%arg0reb, %arg1reb) : (!mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32>
  %1 = mo.linalg.solve<() -> Sh4: !mosh.ape>(%0, %arg1reb) : (!mo.tensor<Sh3, f32>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %1 : !mo.tensor<Sh4, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @linalg_solve_reused_rhs<[[D0:.*]], [[D1:.*]]>([[VARG0:%.*]]: !mo.tensor<[3, [[D0]], [[D0]]], f32>, [[VARG1:%.*]]: !mo.tensor<[3, [[D0]], [[D1]]], f32>)
// CHECK-NEXT:   [[V0:%.*]] = mo.linalg.solve([[VARG0]], [[VARG1]]) : {{.*}} -> !mo.tensor<[3, [[D0]], [[D1]]], f32>
// CHECK-NEXT:   {{.*}}"mo.assert"{{.*}}"[matrix-solve] first input must be a (batch of) square matrix"{{.*}} eq([[D0]], [[D1]]))
// CHECK-NEXT:   [[V2:%.*]] = mo.linalg.solve([[V0]], [[VARG1]]) : {{.*}} -> !mo.tensor<[3, [[D0]], [[D1]]], f32>
// CHECK-NEXT:   mo.output [[V2]]
// CHECK-NEXT: }

mo.graph @slice_neg_stop(
  %input: !mo.tensor<[10], f32>
) -> !mo.tensor<[?], f32> {
  %start = mo.constant {value = #M.dense_array<3> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %stop = mo.constant {value = #M.dense_array<-1> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %step = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %result = mo.slice(%input, %start, %stop, %step)
    : (!mo.tensor<[10], f32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[?], f32>
  mo.output %result : !mo.tensor<[?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_neg_stop([[VARG0:%.*]]: !mo.tensor<[10], f32>) -> !mo.tensor<[3], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<3> : tensor<1xsi32>} : !mo.tensor<[1], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<-1> : tensor<1xsi32>} : !mo.tensor<[1], si32>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} : !mo.tensor<[1], si32>
// CHECK-NEXT:   [[V3:%.*]] = mo.slice([[VARG0]], [[V0]], [[V1]], [[V2]]) : (!mo.tensor<[10], f32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[3], f32>
// CHECK-NEXT:   mo.output [[V3]] : !mo.tensor<[3], f32>
// CHECK-NEXT: }

mo.graph @slice_clamped_start_stop_neg_step(
  %input: !mo.tensor<[10], f32>
) -> !mo.tensor<[?], f32> {
  %start = mo.constant {value = #M.dense_array<30> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %stop = mo.constant {value = #M.dense_array<-20> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %step = mo.constant {value = #M.dense_array<-2> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %result = mo.slice(%input, %start, %stop, %step)
    : (!mo.tensor<[10], f32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[?], f32>
  mo.output %result : !mo.tensor<[?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_clamped_start_stop_neg_step([[VARG0:%.*]]: !mo.tensor<[10], f32>) -> !mo.tensor<[5], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<30> : tensor<1xsi32>} : !mo.tensor<[1], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<-20> : tensor<1xsi32>} : !mo.tensor<[1], si32>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<-2> : tensor<1xsi32>} : !mo.tensor<[1], si32>
// CHECK-NEXT:   [[V3:%.*]] = mo.slice([[VARG0]], [[V0]], [[V1]], [[V2]]) : (!mo.tensor<[10], f32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[5], f32>
// CHECK-NEXT:   mo.output [[V3]] : !mo.tensor<[5], f32>
// CHECK-NEXT: }

mo.graph @slice_4d(
  %input: !mo.tensor<[10, 20, 30, 40], f32>
) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %start = mo.constant {value = #M.dense_array<3, -2, -100, 200> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  %stop = mo.constant {value = #M.dense_array<-1, 10, 200, -100> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  %step = mo.constant {value = #M.dense_array<2, -3, 4, -5> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  %result = mo.slice(%input, %start, %stop, %step)
    : (!mo.tensor<[10, 20, 30, 40], f32>, !mo.tensor<[4], si32>, !mo.tensor<[4], si32>, !mo.tensor<[4], si32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %result : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_4d([[VARG0:%.*]]: !mo.tensor<[10, 20, 30, 40], f32>) -> !mo.tensor<[3, 3, 8, 8], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<3, -2, -100, 200> : tensor<4xsi32>} : !mo.tensor<[4], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<-1, 10, 200, -100> : tensor<4xsi32>} : !mo.tensor<[4], si32>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<2, -3, 4, -5> : tensor<4xsi32>} : !mo.tensor<[4], si32>
// CHECK-NEXT:   [[V3:%.*]] = mo.slice([[VARG0]], [[V0]], [[V1]], [[V2]]) : (!mo.tensor<[10, 20, 30, 40], f32>, !mo.tensor<[4], si32>, !mo.tensor<[4], si32>, !mo.tensor<[4], si32>) -> !mo.tensor<[3, 3, 8, 8], f32>
// CHECK-NEXT:   mo.output [[V3]] : !mo.tensor<[3, 3, 8, 8], f32>
// CHECK-NEXT: }

// CHECK-LABEL: mo.graph @scatter
// CHECK-SAME:  <[[N:.*]]>(
// CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[N, 2, 3], f32>,
// CHECK-SAME:    %[[UPDATES:.*]]: !mo.tensor<[1, 1, 2], f32>,
// CHECK-SAME:    %[[INDICES:.*]]: !mo.tensor<[1, 1, 2], si32>
mo.graph @scatter<N -> D0, D1, D2>(
  %input: !mo.tensor<[N, 2, 3], f32>,
  %updates: !mo.tensor<[1, 1, 2], f32>,
  %indices: !mo.tensor<[1, 1, 2], si32>
) -> !mo.tensor<[D0, D1, D2], f32> {

  // CHECK: "mo.assert"{{.*}} ge([[N]], 1)
  // CHECK: %[[RES:.*]] = mo.scatter{{.*}} -> !mo.tensor<[[[N]], 2, 3], f32>
  %axis = mo.constant {value = #M.dense_array<-1> : tensor<si32>} : !mo.tensor<[], si32>
  %res = mo.scatter<() -> D3, D4, D5>(%input, %updates, %indices, %axis) : (
    !mo.tensor<[N, 2, 3], f32>, !mo.tensor<[1, 1, 2], f32>, !mo.tensor<[1, 1, 2], si32>, !mo.tensor<[], si32>
  ) -> !mo.tensor<[D3, D4, D5], f32>

  // CHECK-NEXT: mo.output %[[RES]]
  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}

mo.graph @scatter_nd(
  %input: !mo.tensor<[10, 1, 1], f32>,
  %updates: !mo.tensor<[5, 1, 1], f32>,
  %indices_1d: !mo.tensor<[5], si32>
) -> !mo.tensor<[?, ?, ?], f32> {
  %shape = mo.constant {value = #M.dense_array<5, 1> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %indices = mo.reshape(%indices_1d, %shape) : (!mo.tensor<[5], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[?, ?], si32>
  %res = mo.scatter_nd(%input, %updates, %indices)
    : (!mo.tensor<[10, 1, 1], f32>, !mo.tensor<[5, 1, 1], f32>, !mo.tensor<[?, ?], si32>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_nd([[VARG0:[^ ]*]]: !mo.tensor<[10, 1, 1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[5, 1, 1], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[5], si32>) -> !mo.tensor<[10, 1, 1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.static.reshape([[VARG2]]) : !mo.tensor<[5], si32> -> !mo.tensor<[5, 1], si32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.scatter_nd([[VARG0]], [[VARG1]], [[V0]]) : (!mo.tensor<[10, 1, 1], f32>, !mo.tensor<[5, 1, 1], f32>, !mo.tensor<[5, 1], si32>) -> !mo.tensor<[10, 1, 1], f32>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[10, 1, 1], f32>

mo.graph @scatter_nd_add(
  %input: !mo.tensor<[10, 1, 1], f32>,
  %updates: !mo.tensor<[5, 1, 1], f32>,
  %indices_1d: !mo.tensor<[5], si32>
) -> !mo.tensor<[?, ?, ?], f32> {
  %shape = mo.constant {value = #M.dense_array<5, 1> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %indices = mo.reshape(%indices_1d, %shape) : (!mo.tensor<[5], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[?, ?], si32>
  %res = mo.scatter_nd.add(%input, %updates, %indices)
    : (!mo.tensor<[10, 1, 1], f32>, !mo.tensor<[5, 1, 1], f32>, !mo.tensor<[?, ?], si32>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_nd_add([[VARG0:[^ ]*]]: !mo.tensor<[10, 1, 1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[5, 1, 1], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[5], si32>) -> !mo.tensor<[10, 1, 1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.static.reshape([[VARG2]]) : !mo.tensor<[5], si32> -> !mo.tensor<[5, 1], si32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.scatter_nd.add([[VARG0]], [[VARG1]], [[V0]]) : (!mo.tensor<[10, 1, 1], f32>, !mo.tensor<[5, 1, 1], f32>, !mo.tensor<[5, 1], si32>) -> !mo.tensor<[10, 1, 1], f32>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[10, 1, 1], f32>

mo.graph @conv2d_static_input_dims(
  %input: !mo.tensor<[10, 5, 5, 16], f32>,
  %filter: !mo.tensor<[3, 3, 16, 32], f32>
) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %ng = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %0 = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
    !mo.tensor<[10, 5, 5, 16], f32>,
    !mo.tensor<[3, 3, 16, 32], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>,
    !mo.tensor<[], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @conv2d_static_input_dims([[VARG0:%.*]]: !mo.tensor<[10, 5, 5, 16], f32>, [[VARG1:%.*]]: !mo.tensor<[3, 3, 16, 32], f32>) -> !mo.tensor<[10, 2, 2, 32], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V4:%.*]] = mo.conv([[VARG0]], [[VARG1]]) [strides = [[V0]], dilations = [[V1]], paddings = [[V2]], num_groups = [[V3]]] : (!mo.tensor<[10, 5, 5, 16], f32>, !mo.tensor<[3, 3, 16, 32], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[10, 2, 2, 32], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[10, 2, 2, 32], f32>
// CHECK-NEXT: }

mo.graph @conv2d_dyn_input_dims(
  %input: !mo.tensor<[?, 20, 30, 16], f32>,
  %filter: !mo.tensor<[3, 3, 16, 32], f32>
) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %ng = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %0 = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
    !mo.tensor<[?, 20, 30, 16], f32>,
    !mo.tensor<[3, 3, 16, 32], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>,
    !mo.tensor<[], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @conv2d_dyn_input_dims<[[D9:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D9]], 20, 30, 16], f32>, [[VARG1:%.*]]: !mo.tensor<[3, 3, 16, 32], f32>) -> !mo.tensor<[[[D9]], 7, 12, 32], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V4:%.*]] = mo.conv([[VARG0]], [[VARG1]]) [strides = [[V3]], dilations = [[V2]], paddings = [[V1]], num_groups = [[V0]]] : (!mo.tensor<[[[D9]], 20, 30, 16], f32>, !mo.tensor<[3, 3, 16, 32], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D9]], 7, 12, 32], f32>
// CHECK-NEXT:   mo.output [[V4]]
// CHECK-NEXT: }

// NOTE this is essentially the inverse of `conv2d_dyn_input_dims`, although we
// can't get the result shape to match the input shape in
// `conv2d_dyn_input_dims` due to roundings.
// CHECK-LABEL: mo.graph @conv_transpose_2d_dyn_input_dims
//  CHECK-SAME:   <[[D9:.*]]>(
mo.graph @conv_transpose_2d_dyn_input_dims(
  %input: !mo.tensor<[?, 7, 12, 32], f32>,
  %filter: !mo.tensor<[3, 3, 16, 32], f32>
) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %op = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: mo.conv_transpose{{.*}} -> !mo.tensor<[[[D9]], 19, 29, 16], f32>
  %0 = mo.conv_transpose(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<[?, 7, 12, 32], f32>,
    !mo.tensor<[3, 3, 16, 32], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>,
    !mo.tensor<[2], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}

mo.graph @max_pool2d_static_input_dims(
  %input: !mo.tensor<[10, 4, 4, 16], f32>
) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %fs = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %0 = mo.max_pool(%input) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (
    !mo.tensor<[10, 4, 4, 16], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @max_pool2d_static_input_dims([[VARG0:%.*]]: !mo.tensor<[10, 4, 4, 16], f32>) -> !mo.tensor<[10, 1, 1, 16], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
// CHECK-NEXT:   [[V4:%.*]] = mo.max_pool([[VARG0]]) [filter_shape = [[V0]], strides = [[V1]], dilations = [[V2]], paddings = [[V3]]] : (!mo.tensor<[10, 4, 4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[10, 1, 1, 16], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[10, 1, 1, 16], f32>
// CHECK-NEXT: }

mo.graph @max_pool2d_ceil_mode_true_static_input_dims(
  %input: !mo.tensor<[10, 4, 4, 16], f32>
) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %fs = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %0 = mo.max_pool_ceil_mode_true(%input) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (
    !mo.tensor<[10, 4, 4, 16], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @max_pool2d_ceil_mode_true_static_input_dims([[VARG0:%.*]]: !mo.tensor<[10, 4, 4, 16], f32>) -> !mo.tensor<[10, 2, 2, 16], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
// CHECK-NEXT:   [[V4:%.*]] = mo.max_pool_ceil_mode_true([[VARG0]]) [filter_shape = [[V0]], strides = [[V1]], dilations = [[V2]], paddings = [[V3]]] : (!mo.tensor<[10, 4, 4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[10, 2, 2, 16], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[10, 2, 2, 16], f32>
// CHECK-NEXT: }

mo.graph @avg_pool2d_dyn_input_dims(
  %input: !mo.tensor<[?, 20, 30, ?], f32>
) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %fs = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %0 = mo.avg_pool(%input) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] {count_boundary = false} : (
    !mo.tensor<[?, 20, 30, ?], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @avg_pool2d_dyn_input_dims<[[D7:.*]], [[D8:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D7]], 20, 30, [[D8]]], f32>) -> !mo.tensor<[[[D7]], 9, 14, [[D8]]], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V4:%.*]] = mo.avg_pool([[VARG0]]) [filter_shape = [[V3]], strides = [[V2]], dilations = [[V1]], paddings = [[V0]]] {count_boundary = false} : (!mo.tensor<[[[D7]], 20, 30, [[D8]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[D7]], 9, 14, [[D8]]], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[[[D7]], 9, 14, [[D8]]], f32>
// CHECK-NEXT: }

mo.graph @avg_pool2d_ceil_mode_true_dyn_input_dims(
  %input: !mo.tensor<[?, 20, 30, ?], f32>
) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %fs = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %0 = mo.avg_pool_ceil_mode_true(%input) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] {count_boundary = false} : (
    !mo.tensor<[?, 20, 30, ?], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @avg_pool2d_ceil_mode_true_dyn_input_dims<[[D7:.*]], [[D8:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D7]], 20, 30, [[D8]]], f32>) -> !mo.tensor<[[[D7]], 10, 15, [[D8]]], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V4:%.*]] = mo.avg_pool_ceil_mode_true([[VARG0]]) [filter_shape = [[V3]], strides = [[V2]], dilations = [[V1]], paddings = [[V0]]] {count_boundary = false} : (!mo.tensor<[[[D7]], 20, 30, [[D8]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[D7]], 10, 15, [[D8]]], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[[[D7]], 10, 15, [[D8]]], f32>
// CHECK-NEXT: }

mo.graph @arg_max(%input: !mo.tensor<[?, ?], si64>) -> !mo.tensor<[?, ?], si64> {
  %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %res = mo.arg_max(%input, %axis) :
    (!mo.tensor<[?, ?], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?], si64>
  mo.output %res : !mo.tensor<[?, ?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @arg_max<[[D5:.*]], [[D6:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D5]], [[D6]]], si64>) -> !mo.tensor<[[[D5]], 1], si64> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.arg_max([[VARG0]], [[V0]]) : (!mo.tensor<[[[D5]], [[D6]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D5]], 1], si64>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D5]], 1], si64>
// CHECK-NEXT: }

// This test ensures we can shape infer through a `mo.concat` op with non-equal
// dimensions (i.e., it doesn't rely on shape inference being able to prove the
// dimensions are equal). This is important because sometimes we can't prove as
// much as we want, but we still want to shape infer through `mo.concat`, which
// is an important op for shape inference in models like transformer.
mo.graph @concat_non_equal_dims(
  %input1: !mo.tensor<[?, 1, ?], f32>,
  %input2: !mo.tensor<[?, 2, ?], f32>,
  %input3: !mo.tensor<[?, 3, ?], f32>
) -> !mo.tensor<[?, ?, ?], f32> {
  %axis = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %res = mo.concat[%axis : !mo.tensor<[], si64>](%input1, %input2, %input3) :
  (!mo.tensor<[?, 1, ?], f32>, !mo.tensor<[?, 2, ?], f32>, !mo.tensor<[?, 3, ?], f32>)
    -> !mo.tensor<[?, ?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @concat_non_equal_dims<[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]]>([[INPUT1:%.*]]: !mo.tensor<[[[D0]], 1, [[D1]]], f32>, [[INPUT2:%.*]]: !mo.tensor<[[[D2]], 2, [[D3]]], f32>, [[INPUT3:%.*]]: !mo.tensor<[[[D4]], 3, [[D5]]], f32>) -> !mo.tensor<[[[D0]], 6, [[D1]]], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   {{.*}}"mo.assert"{{.*}} "[concat] input shapes must match except at concat axis" {{.*}} eq([[D0]], [[D2]])
// CHECK-NEXT:   {{.*}}"mo.assert"{{.*}} "[concat] input shapes must match except at concat axis" {{.*}} eq([[D1]], [[D3]])
// CHECK-NEXT:   {{.*}}"mo.assert"{{.*}} "[concat] input shapes must match except at concat axis" {{.*}} eq([[D0]], [[D4]])
// CHECK-NEXT:   {{.*}}"mo.assert"{{.*}} "[concat] input shapes must match except at concat axis" {{.*}} eq([[D1]], [[D5]])
// CHECK-NEXT:   [[RES:%.*]] = mo.concat{{.*}}([[INPUT1]], [[INPUT2]], [[INPUT3]]) : {{.*}} -> !mo.tensor<[[[D0]], 6, [[D1]]], f32>
// CHECK-NEXT:   mo.output [[RES]]
// CHECK-NEXT: }

mo.graph @pad_constant(
  %input: !mo.tensor<[2, 3], f32>,
  %constant: !mo.tensor<[], f32>
) -> !mo.tensor<[?, ?], f32> {
  %paddings = mo.constant {value = #M.dense_array<10, 20, 30, 40> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
  %res = mo.pad.constant(%input, %paddings, %constant) : (
    !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], si32>, !mo.tensor<[], f32>
  ) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pad_constant([[VARG0:%.*]]: !mo.tensor<[2, 3], f32>, [[VARG1:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[32, 73], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<10, 20, 30, 40> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.pad.constant([[VARG0]], [[V0]], [[VARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[32, 73], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[32, 73], f32>
// CHECK-NEXT: }

mo.graph @pad_reflect(
  %input: !mo.tensor<[2, 3], f32>
) -> !mo.tensor<[?, ?], f32> {
  %paddings = mo.constant {value = #M.dense_array<10, 20, 30, 40> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
  %res = mo.pad.reflect(%input, %paddings) : (
    !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], si32>
  ) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pad_reflect([[VARG0:%.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[32, 73], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<10, 20, 30, 40> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.pad.reflect([[VARG0]], [[V0]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[32, 73], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[32, 73], f32>
// CHECK-NEXT: }

mo.graph @pad_repeat(
  %input: !mo.tensor<[2, 3], f32>
) -> !mo.tensor<[?, ?], f32> {
  %paddings = mo.constant {value = #M.dense_array<10, 20, 30, 40> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
  %res = mo.pad.repeat(%input, %paddings) : (
    !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], si32>
  ) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pad_repeat([[VARG0:%.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[32, 73], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<10, 20, 30, 40> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.pad.repeat([[VARG0]], [[V0]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[32, 73], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[32, 73], f32>
// CHECK-NEXT: }

mo.graph @tile(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[?, ?], f32> {
  %repeats = mo.constant {value = #M.dense_array<10, 20> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %res = mo.tile(%input, %repeats) : (
    !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @tile([[VARG0:%.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[20, 60], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<10, 20> : tensor<2xsi32>} : !mo.tensor<[2], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.tile([[VARG0]], [[V0]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[20, 60], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[20, 60], f32>
// CHECK-NEXT: }

mo.graph @partially_parameterized_signature<in>(
  %input: !mo.tensor<[in, ?, ?], f32>
) -> !mo.tensor<[?, ?, ?], f32> {
  %axis = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %res  = mo.reduce.mul(%input, %axis)
    : (!mo.tensor<[in, ?, ?], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @partially_parameterized_signature<in, [[D6:.*]], [[D7:.*]]>([[VARG0:%.*]]: !mo.tensor<[in, [[D6]], [[D7]]], f32>) -> !mo.tensor<[in, 1, [[D7]]], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.reduce.mul([[VARG0]], [[V0]]) : (!mo.tensor<[in, [[D6]], [[D7]]], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[in, 1, [[D7]]], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[in, 1, [[D7]]], f32>
// CHECK-NEXT: }

mo.graph @binary_logical_op(
  %lhs: !mo.tensor<[?, ?], bool>,
  %rhs: !mo.tensor<[?, ?], bool>
) -> !mo.tensor<[?, ?], bool> {
  %and = mo.and(%lhs, %rhs) : !mo.tensor<[?, ?], bool>
  %xor = mo.xor(%and, %rhs) : !mo.tensor<[?, ?], bool>
  mo.output %xor : !mo.tensor<[?, ?], bool>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_logical_op<[[D7:.*]], [[D8:.*]], [[D9:.*]], [[D10:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D7]], [[D8]]], bool>, [[VARG1:%.*]]: !mo.tensor<[[[D9]], [[D10]]], bool>) -> !mo.tensor<[[[D7]], [[D8]]], bool> {
// CHECK-NEXT:   [[V0:%.*]] = mo.rebind([[VARG1]]) : !mo.tensor<[[[D9]], [[D10]]], bool> -> !mo.tensor<[[[D7]], [[D8]]], bool>
// CHECK-NEXT:   kgen.param.declare SORS_CH18: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[and] input shapes must match" : !kgen.string}, and(eq([[D10]], [[D8]]), eq([[D7]], [[D9]])))>
// CHECK-NEXT:   [[V1:%.*]] = mo.and([[VARG0]], [[V0]]) : !mo.tensor<[[[D7]], [[D8]]], bool>
// CHECK-NEXT:   kgen.param.declare SORS_CH6: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[xor] input shapes must match" : !kgen.string}, and(eq([[D10]], [[D8]]), eq([[D7]], [[D9]])))>
// CHECK-NEXT:   [[V2:%.*]] = mo.xor([[V1]], [[V0]]) : !mo.tensor<[[[D7]], [[D8]]], bool>
// CHECK-NEXT:   mo.output [[V2]] : !mo.tensor<[[[D7]], [[D8]]], bool>
// CHECK-NEXT: }

mo.graph @top_k(%input : !mo.tensor<[5, 5], si64>, %sorted : !mo.tensor<[], bool>) -> (!mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>) {
  %in = "mo.constant"() {value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>} : () -> !mo.tensor<[5, 5], si64>
  %K = "mo.constant"() {value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %axis = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %values, %indices = mo.top_k(%in, %K, %axis, %sorted) : (
    !mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>)
  mo.output %values, %indices : !mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @top_k([[VARG0:%.*]]: !mo.tensor<[5, 5], si64>, [[VARG1:%.*]]: !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>) {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>} : !mo.tensor<[5, 5], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<3> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   [[VVALUES:%.*]], [[VINDICES:%.*]] = mo.top_k([[V0]], [[V1]], [[V2]], [[VARG1]]) : (!mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>)
// CHECK-NEXT:   mo.output [[VVALUES]], [[VINDICES]] : !mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>
// CHECK-NEXT: }

// Ensure dynamic K properly generates an assert
// dummy is just meant to help define the param 'A'
mo.graph @top_k_dynamic<A>(%input : !mo.tensor<[5, 5], si64>, %sorted : !mo.tensor<[], bool>, %dummy : !mo.tensor<[A], bool>) -> (!mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>) {
  %in = "mo.constant"() {value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>} : () -> !mo.tensor<[5, 5], si64>
  %K = mo.shape_of(%dummy) : (!mo.tensor<[A], bool>) -> !mo.tensor<[1], si64>
  %K_reshape = mo.static.reshape(%K): !mo.tensor<[1], si64> -> !mo.tensor<[], si64>
  %axis = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %values, %indices = mo.top_k(%in, %K_reshape, %axis, %sorted) : (
    !mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>)
  mo.output %values, %indices : !mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @top_k_dynamic<A>([[VARG0:[^ ]*]]: !mo.tensor<[5, 5], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[], bool>, [[VARG2:[^ ]*]]: !mo.tensor<[A], bool>) -> (!mo.tensor<[5, A], si64>, !mo.tensor<[5, A], si64>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant {value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>} : !mo.tensor<[5, 5], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mosh.param.to_value = <A>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.index.to_tensor([[V2]]) -> !mo.tensor<[], si64>
// CHECK-NEXT: kgen.param.declare assert9: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[top/bottom-k] k must be between [0, dim_size]." : !kgen.string}, lt(A, 6))>
// CHECK-NEXT: [[VVALUES:[^ ]*]], [[VINDICES:[^ ]*]] = mo.top_k([[V1]], [[V3]], [[V0]], [[VARG1]]) : (!mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[5, A], si64>, !mo.tensor<[5, A], si64>)
// CHECK-NEXT: mo.output [[VVALUES]], [[VINDICES]] : !mo.tensor<[5, A], si64>, !mo.tensor<[5, A], si64>

mo.graph @bottom_k(%input : !mo.tensor<[5, 5], si64>, %sorted : !mo.tensor<[], bool>) -> (!mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>) {
  %in = "mo.constant"() {value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>} : () -> !mo.tensor<[5, 5], si64>
  %K = "mo.constant"() {value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %axis = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %values, %indices = mo.bottom_k(%in, %K, %axis, %sorted) : (
    !mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>)
  mo.output %values, %indices : !mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @bottom_k([[VARG0:%.*]]: !mo.tensor<[5, 5], si64>, [[VARG1:%.*]]: !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>) {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>} : !mo.tensor<[5, 5], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<3> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   [[VVALUES:%.*]], [[VINDICES:%.*]] = mo.bottom_k([[V0]], [[V1]], [[V2]], [[VARG1]]) : (!mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>)
// CHECK-NEXT:   mo.output [[VVALUES]], [[VINDICES]] : !mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>
// CHECK-NEXT: }

mo.graph @unsqueeze_shape_dynamic_shape(%input_shape: !mo.tensor<[?], si32>, %indices: !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32> {
  %res = mo.unsqueeze_shape(%input_shape, %indices) : (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32>
  mo.output %res : !mo.tensor<[?], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @unsqueeze_shape_dynamic_shape<[[D1:.*]], [[D2:.*]] -> [[D3:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D1]]], si32>, [[VARG1:%.*]]: !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[[[D3]]], si32> {
// CHECK-NEXT:   kgen.param.declare [[EXPR:.*]] = <add([[D1]], [[D2]])>
// CHECK-NEXT:   [[V0:%.*]] = mo.unsqueeze_shape([[VARG0]], [[VARG1]]) : (!mo.tensor<[[[D1]]], si32>, !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[[[EXPR]]], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.rebind([[V0]]) : !mo.tensor<[[[EXPR]]], si32> -> !mo.tensor<[add([[D1]], [[D2]])], si32>
// CHECK-NEXT:   mo.output<add([[D1]], [[D2]])> [[V1]] : !mo.tensor<[add([[D1]], [[D2]])], si32>
// CHECK-NEXT: }

mo.graph @unsqueeze_shape_dynamic_shape_with_reshape(%input_tensor: !mo.tensor<[?, ?, ?], si32>, %indices: !mo.tensor<[?], si32>) -> !mo.tensor<?, si32> {
  %shape_tensor = mo.shape_of(%input_tensor) : (!mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[3], si32>
  %result_shape = mo.unsqueeze_shape(%shape_tensor, %indices) : (!mo.tensor<[3], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32>
  %res = mo.reshape(%input_tensor, %result_shape) : (!mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<?, si32>
  mo.output %res : !mo.tensor<?, si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @unsqueeze_shape_dynamic_shape_with_reshape<[[D8:.*]], [[D9:.*]], [[D10:.*]], [[D11:.*]] -> [[Sh12:.*]]: !mosh.ape>([[VARG0:%.*]]: !mo.tensor<[[[D8]], [[D9]], [[D10]]], si32>, [[VARG1:%.*]]: !mo.tensor<[[[D11]]], si32>) -> !mo.tensor<Sh12, si32> {
// CHECK-NEXT:   [[V0:%.*]] = mosh.param.to_value: !mosh.ape = <[[[D8]], [[D9]], [[D10]]]>
// CHECK-NEXT:   [[V1:%.*]] = mo.shape.to_tensor([[V0]]) -> !mo.tensor<[3], si32>
// CHECK-NEXT:   [[V2:%.*]] = mo.unsqueeze_shape([[V1]], [[VARG1]]) : (!mo.tensor<[3], si32>, !mo.tensor<[[[D11]]], si32>) -> !mo.tensor<[param_expr_18], si32>
// CHECK-NEXT:   mo.invoke_shape_func["reshape_shape"]<() -> [[Sh5:.*]]: !mosh.ape>([[VARG0]], [[V2]]) {dataDeptTensors = [1 : i32]}
// CHECK-NEXT:   kgen.param.declare param_expr_18 = <add([[D11]], 3)>
// CHECK-NEXT:   [[V3:%.*]] = mo.reshape([[VARG0]], [[V2]]) : (!mo.tensor<[[[D8]], [[D9]], [[D10]]], si32>, !mo.tensor<[param_expr_18], si32>) -> !mo.tensor<[[Sh5]], si32>
// CHECK-NEXT:   mo.output<{{.*}} [[V3]]
// CHECK-NEXT: }

mo.graph @broadcast_shape_dynamic(%shape1: !mo.tensor<[?], si32>, %shape2: !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32> {
  %res = mo.broadcast_shape(%shape1, %shape2) : (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32>
  mo.output %res : !mo.tensor<[?], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_dynamic<[[D1:.*]], [[D2:.*]] -> [[D3:.*]]>(%[[ARG0:.*]]: !mo.tensor<[[[D1]]], si32>, %[[ARG1:.*]]: !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[[[D3]]], si32>
// CHECK-NEXT: kgen.param.declare [[EXPR:.*]] = <max([[D1]], [[D2]])>
// CHECK-NEXT: %[[V0:.*]] = mo.broadcast_shape(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[[[D1]]], si32>, !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[[[EXPR]]], si32>
// CHECK-NEXT: %[[V1:.*]] = mo.rebind(%[[V0]]) : !mo.tensor<[[[EXPR]]], si32> -> !mo.tensor<[max([[D1]], [[D2]])], si32>
// CHECK-NEXT: mo.output<{{.*}}> %[[V1]]

mo.graph @broadcast_shape_static(%shape1: !mo.tensor<[3], si32>, %shape2: !mo.tensor<[5], si32>) -> !mo.tensor<[?], si32> {
  %res = mo.broadcast_shape(%shape1, %shape2) : (!mo.tensor<[3], si32>, !mo.tensor<[5], si32>) -> !mo.tensor<[?], si32>
  mo.output %res : !mo.tensor<[?], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_static(%[[ARG0:.*]]: !mo.tensor<[3], si32>, %[[ARG1:.*]]: !mo.tensor<[5], si32>) -> !mo.tensor<[5], si32>
// CHECK-NEXT: %[[V0:.*]] = mo.broadcast_shape(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[3], si32>, !mo.tensor<[5], si32>) -> !mo.tensor<[5], si32>
// CHECK-NEXT: mo.output %[[V0]] : !mo.tensor<[5], si32>

mo.graph @gather_nd(%input: !mo.tensor<[?, ?, ?], si64>, %indices: !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, si64> {
  %result = mo.gather_nd(%input, %indices) {batchDims = 1} :
    (!mo.tensor<[?, ?, ?], si64>, !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, si64>
  mo.output %result : !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @gather_nd<[[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]] -> [[Sh6:.*]]: !mosh.ape>([[VARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], si64>, [[VARG1:%.*]]: !mo.tensor<[[[D4]], [[D5]]], si64>) -> !mo.tensor<Sh6, si64> {
// CHECK-NEXT:   mo.invoke_shape_func["gather_nd_shape"]<() -> [[SH12:.*]]: !mosh.ape>([[VARG0]], [[VARG1]]) {dataDeptTensors = [], kgenParams = {batch_dims = 1 : i64}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], si64>, !mo.tensor<[[[D4]], [[D5]]], si64>)
// CHECK-NEXT:   [[V0:%.*]] = mo.gather_nd([[VARG0]], [[VARG1]]) {batchDims = 1 : i64} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], si64>, !mo.tensor<[[[D4]], [[D5]]], si64>) -> !mo.tensor<[[SH12]], si64>
// CHECK-NEXT:   mo.output<{{.*}}> [[V0]]
// CHECK-NEXT: }

mo.graph @gather_nd_with_last_dim(%input: !mo.tensor<[?, ?, ?], si64>, %indices: !mo.tensor<[?, 1], si64>) -> !mo.tensor<?, si64> {
  %result = mo.gather_nd(%input, %indices) {batchDims = 1} :
    (!mo.tensor<[?, ?, ?], si64>, !mo.tensor<[?, 1], si64>) -> !mo.tensor<?, si64>
  mo.output %result : !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @gather_nd_with_last_dim<[[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], si64>, [[VARG1:%.*]]: !mo.tensor<[[[D4]], 1], si64>)
// CHECK-NEXT:   [[V0:%.*]] = mo.gather_nd([[VARG0]], [[VARG1]]) {batchDims = 1 : i64} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], si64>, !mo.tensor<[[[D4]], 1], si64>) -> !mo.tensor<[[[D1]], [[D3]]], si64>
// CHECK-NEXT:   mo.output [[V0]]
// CHECK-NEXT: }

mo.graph @squeeze_shape_dynamic_shape(%input_shape: !mo.tensor<[?], si32>, %indices: !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32> {
  %res = mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32>
  mo.output %res : !mo.tensor<[?], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @squeeze_shape_dynamic_shape<[[D1:.*]], [[D2:.*]] -> [[D3:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D1]]], si32>, [[VARG1:%.*]]: !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[[[D3]]], si32> {
// CHECK-NEXT:   kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[squeeze_shape] cannot remove more dimensions than there exists" : !kgen.string}, le([[D2]], [[D1]]))>
// CHECK-NEXT:   kgen.param.declare [[EXPR:.*]] = <add(mul_nuw([[D2]], -1), [[D1]])>
// CHECK-NEXT:   [[V0:%.*]] = mo.squeeze_shape([[VARG0]], [[VARG1]]) : (!mo.tensor<[[[D1]]], si32>, !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[[[EXPR]]], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.rebind([[V0]]) : !mo.tensor<[[[EXPR]]], si32> -> !mo.tensor<[add(mul_nuw([[D2]], -1), [[D1]])], si32>
// CHECK-NEXT:   mo.output<{{.*}}> [[V1]]
// CHECK-NEXT: }

mo.graph @squeeze_shape_dynamic_shape_with_reshape(%input_tensor: !mo.tensor<[?, ?, ?], si32>, %indices: !mo.tensor<[?], si32>) -> !mo.tensor<?, si32> {
  %shape_tensor = mo.shape_of(%input_tensor) : (!mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[3], si32>
  %result_shape = mo.squeeze_shape(%shape_tensor, %indices) : (!mo.tensor<[3], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32>
  %res = mo.reshape(%input_tensor, %result_shape) : (!mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<?, si32>
  mo.output %res : !mo.tensor<?, si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @squeeze_shape_dynamic_shape_with_reshape<[[D8:.*]], [[D9:.*]], [[D10:.*]], [[D11:.*]] -> [[Sh12:.*]]: !mosh.ape>([[VARG0:%.*]]: !mo.tensor<[[[D8]], [[D9]], [[D10]]], si32>, [[VARG1:%.*]]: !mo.tensor<[[[D11]]], si32>) -> !mo.tensor<Sh12, si32> {
// CHECK-NEXT:   [[V0:%.*]] = mosh.param.to_value: !mosh.ape = <[[[D8]], [[D9]], [[D10]]]>
// CHECK-NEXT:   [[V1:%.*]] = mo.shape.to_tensor([[V0]]) -> !mo.tensor<[3], si32>
// CHECK-NEXT:   kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[squeeze_shape] cannot remove more dimensions than there exists" : !kgen.string}, lt([[D11]], 4))>
// CHECK-NEXT:   [[V2:%.*]] = mo.squeeze_shape([[V1]], [[VARG1]]) : (!mo.tensor<[3], si32>, !mo.tensor<[[[D11]]], si32>) -> !mo.tensor<[param_expr_21], si32>
// CHECK-NEXT:   mo.invoke_shape_func["reshape_shape"]<() -> [[Sh5:.*]]: !mosh.ape>([[VARG0]], [[V2]]) {dataDeptTensors = [1 : i32]}
// CHECK-NEXT:   kgen.param.declare param_expr_21 = <add(mul_nuw([[D11]], -1), 3)>
// CHECK-NEXT:   [[V3:%.*]] = mo.reshape([[VARG0]], [[V2]]) : (!mo.tensor<[[[D8]], [[D9]], [[D10]]], si32>, !mo.tensor<[param_expr_21], si32>) -> !mo.tensor<[[Sh5]], si32>
// CHECK-NEXT:   mo.output<{{.*}}> [[V3]]
// CHECK-NEXT: }

mo.graph @slice_entire_dynamic_axis(%arg0: !mo.tensor<[10, ?, 20], f32>) -> (!mo.tensor<[?, ?, ?], f32>) {
  %axis0 = mo.constant {value = #M.dense_array<0> : !M.array<1xsi32>} : !mo.tensor<[], si32>
  %index0 = mo.constant {value = #M.dense_array<0> : !M.array<1xsi64>} : !mo.tensor<[1], si64>
  %index1 = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[1], si64>
  %index2 = mo.constant {value = #M.dense_array<2> : !M.array<1xsi64>} : !mo.tensor<[1], si64>
  %starts = mo.constant {value = #M.dense_array<0, 0, 0> : !M.array<3xsi64>} : !mo.tensor<[3], si64>
  %steps = mo.constant {value = #M.dense_array<1, 1, 1> : !M.array<3xsi64>} : !mo.tensor<[3], si64>
  %shape = mo.shape_of(%arg0) : (!mo.tensor<[10, ?, 20], f32>) -> !mo.tensor<[3], si64>
  %gather0 = mo.gather(%shape, %index0, %axis0) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[], si32>) -> !mo.tensor<[1], si64>
  %gather1 = mo.gather(%shape, %index1, %axis0) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[], si32>) -> !mo.tensor<[1], si64>
  %gather2 = mo.gather(%shape, %index2, %axis0) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[], si32>) -> !mo.tensor<[1], si64>
  %ends = mo.concat[%axis0 : !mo.tensor<[], si32>](%gather0, %gather1, %gather2) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[3], si64>
  %output = mo.slice(%arg0, %starts, %ends, %steps) : (!mo.tensor<[10, ?, 20], f32>, !mo.tensor<[3], si64>,!mo.tensor<[3], si64>,!mo.tensor<[3], si64>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %output : !mo.tensor<[?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_entire_dynamic_axis<[[D6:.*]]>([[VARG0:%.*]]: !mo.tensor<[10, [[D6]], 20], f32>) -> !mo.tensor<[10, [[D6]], 20], f32> {
// CHECK-NEXT:   mo.output [[VARG0]] : !mo.tensor<[10, [[D6]], 20], f32>
// CHECK-NEXT: }

mo.graph @slice_out_a_third_of_constant_axis(
  %1207: !mo.tensor<[50, ?, 2304], f32>
) -> !mo.tensor<[?, ?, ?], f32> {
  %2 = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %4 = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %12 = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %16 = mo.constant {value = #M.dense_array<50> : !M.array<1xsi64>} : !mo.tensor<[1], si64>
  %19 = mo.constant {value = #M.dense_array<0> : !M.array<1xsi32>} : !mo.tensor<[], si32>
  %23 = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[1], si64>
  %24 = mo.constant {value = #M.dense_array<0> : !M.array<1xsi64>} : !mo.tensor<[1], si64>
  %32 = mo.constant {value = #M.dense_array<2, 3, 1, 0> : !M.array<4xsi64>} : !mo.tensor<[4], si64>
  %34 = mo.constant {value = #M.dense_array<0> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %241 = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %243 = mo.constant {value = #M.dense_array<3> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %1210 = mo.shape_of(%1207) : (!mo.tensor<[50, ?, 2304], f32>) -> !mo.tensor<[3], si64>
  %1213 = mo.gather(%1210, %241, %34) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[1], si64>
  %1215 = mo.add(%1213, %12) : !mo.tensor<[1], si64>
  %1218 = mo.div(%1215, %243) : !mo.tensor<[1], si64>
  %1220 = mo.mul(%1218, %12) : !mo.tensor<[1], si64>
  %1230 = mo.shape_of(%1207) : (!mo.tensor<[50, ?, 2304], f32>) -> !mo.tensor<[3], si32>
  %1231 = mo.gather(%1230, %23, %19) : (!mo.tensor<[3], si32>, !mo.tensor<[1], si64>, !mo.tensor<[], si32>) -> !mo.tensor<[1], si32>
  %1232 = mo.concat[%19 : !mo.tensor<[], si32>] (%16, %1231) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si32>) -> !mo.tensor<[2], si64>
  %1233 = mo.gather(%1218, %24, %19) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[], si32>) -> !mo.tensor<[1], si64>
  %1234 = mo.gather(%1220, %24, %19) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[], si32>) -> !mo.tensor<[1], si64>
  %1235 = mo.concat[%19 : !mo.tensor<[], si32>] (%4, %1233) : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[3], si64>
  %1236 = mo.concat[%19 : !mo.tensor<[], si32>] (%1232, %1234) : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[3], si64>
  %1237 = mo.slice(%1207, %1235, %1236, %2) : (!mo.tensor<[50, ?, 2304], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %1237 : !mo.tensor<[?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_out_a_third_of_constant_axis<[[D7:.*]]>([[VARG0:%.*]]: !mo.tensor<[50, [[D7]], 2304], f32>) -> !mo.tensor<[50, [[D7]], 768], f32> {
// CHECK-DAG:    [[V0:%.*]] = mosh.param.to_value: !mosh.ape = <[50, [[D7]], 1536]>
// CHECK-DAG:    [[V1:%.*]] = mo.constant {value = #M.dense_array<0, 0, 768> : tensor<3xsi64>} : !mo.tensor<[3], si64>
// CHECK-DAG:    [[V2:%.*]] = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.shape.to_tensor([[V0]]) -> !mo.tensor<[3], si64>
// CHECK-NEXT:   [[V4:%.*]] = mo.slice([[VARG0]], [[V1]], [[V3]], [[V2]]) : (!mo.tensor<[50, [[D7]], 2304], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[50, [[D7]], 768], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[50, [[D7]], 768], f32>
// CHECK-NEXT: }

mo.graph @nested_broadcast_shapes(
  %cond: !mo.tensor<[?, ?], bool>,
  %else: !mo.tensor<[?, ?], si32>
) -> !mo.tensor<[?, ?], si32> {
  %then = mo.cast(%cond) : (!mo.tensor<[?, ?], bool>) -> !mo.tensor<[?, ?], si32>

  %cond_sh = mo.shape_of(%cond) : (!mo.tensor<[?, ?], bool>) -> !mo.tensor<[2], si32>
  %then_sh = mo.shape_of(%then) : (!mo.tensor<[?, ?], si32>) -> !mo.tensor<[2], si32>
  %else_sh = mo.shape_of(%else) : (!mo.tensor<[?, ?], si32>) -> !mo.tensor<[2], si32>

  %sh1 = mo.broadcast_shape(%cond_sh, %then_sh) : (!mo.tensor<[2], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2], si32>
  %sh2 = mo.broadcast_shape(%sh1, %else_sh) : (!mo.tensor<[2], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2], si32>

  %cond_b = mo.broadcast_to(%cond, %sh2) : (!mo.tensor<[?, ?], bool>, !mo.tensor<[2], si32>)   -> !mo.tensor<[?, ?], bool>
  %then_b = mo.broadcast_to(%then, %sh2) : (!mo.tensor<[?, ?], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[?, ?], si32>
  %else_b = mo.broadcast_to(%else, %sh2) : (!mo.tensor<[?, ?], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[?, ?], si32>

  %res = mo.select(%cond_b, %then_b, %else_b) : (!mo.tensor<[?, ?], bool>, !mo.tensor<[?, ?], si32>, !mo.tensor<[?, ?], si32>) -> !mo.tensor<[?, ?], si32>
  mo.output %res : !mo.tensor<[?, ?], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @nested_broadcast_shapes<[[D29:.*]], [[D30:.*]], [[D31:.*]], [[D32:.*]] -> [[D33:.*]], [[D34:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D29]], [[D30]]], bool>, [[VARG1:%.*]]: !mo.tensor<[[[D31]], [[D32]]], si32>) -> !mo.tensor<[[[D33]], [[D34]]], si32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.cast([[VARG0]]) : (!mo.tensor<[[[D29]], [[D30]]], bool>) -> !mo.tensor<[[[D29]], [[D30]]], si32>
// CHECK-NEXT:   kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(eq([[D29]], [[D31]]), eq([[D29]], 1), eq([[D31]], 1)))>
// CHECK-NEXT:   kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(eq([[D30]], [[D32]]), eq([[D30]], 1), eq([[D32]], 1)))>
// CHECK-NEXT:   kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dim must be either 1 or equal to corresponding output dim starting from the rightmost dim" : !kgen.string}
// CHECK-NEXT:   kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dim must be either 1 or equal to corresponding output dim starting from the rightmost dim" : !kgen.string}
// CHECK-NEXT:   [[V1:%.*]] = mo.static.broadcast_to([[VARG0]]) : !mo.tensor<[[[D29]], [[D30]]], bool> -> !mo.tensor<[param_expr_85, param_expr_86], bool>
// CHECK-NEXT:   kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dim must be either 1 or equal to corresponding output dim starting from the rightmost dim" : !kgen.string}
// CHECK-NEXT:   kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dim must be either 1 or equal to corresponding output dim starting from the rightmost dim" : !kgen.string}
// CHECK-NEXT:   [[V2:%.*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[[[D29]], [[D30]]], si32> -> !mo.tensor<[param_expr_85, param_expr_86], si32>
// CHECK-NEXT:   kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dim must be either 1 or equal to corresponding output dim starting from the rightmost dim" : !kgen.string}
// CHECK-NEXT:   kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dim must be either 1 or equal to corresponding output dim starting from the rightmost dim" : !kgen.string}
// CHECK-NEXT:   [[V3:%.*]] = mo.static.broadcast_to([[VARG1]]) : !mo.tensor<[[[D31]], [[D32]]], si32> -> !mo.tensor<[param_expr_85, param_expr_86], si32>
// CHECK-NEXT:   kgen.param.declare param_expr_85 = <cond(eq([[D29]], 1), [[D31]], [[D29]])>
// CHECK-NEXT:   kgen.param.declare param_expr_86 = <cond(eq([[D30]], 1), [[D32]], [[D30]])>
// CHECK-NEXT:   [[V4:%.*]] = mo.select([[V1]], [[V2]], [[V3]])
// CHECK-NEXT:   [[V5:%.*]] = mo.rebind([[V4]]) : {{.*}} -> !mo.tensor<[cond(eq([[D29]], 1), [[D31]], [[D29]]), cond(eq([[D30]], 1), [[D32]], [[D30]])], si32>
// CHECK-NEXT:   mo.output{{.*}} [[V5]]
// CHECK-NEXT: }

mo.graph @onnx_select_neg_ones_bcast(
  %in: !mo.tensor<[1, 32, ?, 10], si64>
) -> !mo.tensor<[1, 32, ?, 10], si64> {
  %ones = mo.constant {value = #M.dense_array<1, 1, 1, 1> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %neg_ones = mo.constant {value = #M.dense_array<-1, -1, -1, -1> : tensor<4xsi64>} : !mo.tensor<[4], si64>

  %add = mo.add(%in, %in) : !mo.tensor<[1, 32, ?, 10], si64>
  %add_shape = mo.shape_of(%add) : (!mo.tensor<[1, 32, ?, 10], si64>) -> !mo.tensor<[4], si32>

  %in_shape = mo.shape_of(%in) : (!mo.tensor<[1, 32, ?, 10], si64>) -> !mo.tensor<[4], si64>
  %cond = mo.equal(%in_shape, %neg_ones) : (!mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[4], bool>
  %selected_shape = mo.select(%cond, %ones, %in_shape) : (!mo.tensor<[4], bool>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[4], si64>

  %target_shape = mo.broadcast_shape(%add_shape, %selected_shape) : (!mo.tensor<[4], si32>, !mo.tensor<[4], si64>) -> !mo.tensor<[4], si64>
  %res = mo.broadcast_to(%in, %target_shape) : (!mo.tensor<[1, 32, ?, 10], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, ?, 10], si64>

  mo.output %res : !mo.tensor<[1, 32, ?, 10], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @onnx_select_neg_ones_bcast<[[D9:.*]]>([[VARG0:.*]]: !mo.tensor<[1, 32, [[D9]], 10], si64>) -> !mo.tensor<[1, 32, [[D9]], 10], si64> {
// CHECK-NEXT:   mo.output [[VARG0]] : !mo.tensor<[1, 32, [[D9]], 10], si64>
// CHECK-NEXT: }


// CHECK-LABEL: mo.graph @random_normal_const
// CHECK-SAME:  (
// CHECK-SAME:    %[[FLOAT:.*]]: !mo.tensor<[], f32>,
// CHECK-SAME:    %[[SEED:.*]]: !mo.tensor<[], si64>
// CHECK-SAME:  ) -> !mo.tensor<[128], f32> {
mo.graph @random_normal_const(
  %float: !mo.tensor<[], f32>,
  %seed: !mo.tensor<[], si64>
) -> !mo.tensor<[?], f32>{

  // CHECK-NEXT: %[[RES:.*]] = mo.static.random.normal(%[[FLOAT]], %[[FLOAT]], %[[SEED]]) : ({{.*}}) -> !mo.tensor<[128], f32>
  %size = mo.constant {value = #M.dense_array<128> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %res = mo.random.normal(%size, %float, %float, %seed) :
        (!mo.tensor<[1], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>,
        !mo.tensor<[], si64>) -> !mo.tensor<[?], f32>

  // CHECK-NEXT: mo.output %[[RES]]
  mo.output %res: !mo.tensor<[?], f32>
}

// CHECK-LABEL: mo.graph @random_normal_dyn
// CHECK-SAME:  <[[N:.*]]>(
// CHECK-SAME:    %[[DUMMY:.*]]: !mo.tensor<[[[N]]], f32>,
// CHECK-SAME:    %[[FLOAT:.*]]: !mo.tensor<[], f32>,
// CHECK-SAME:    %[[SEED:.*]]: !mo.tensor<[], si64>
// CHECK-SAME:  ) -> !mo.tensor<[[[N]]], f32> {
mo.graph @random_normal_dyn<N>(
  %dummy: !mo.tensor<[N], f32>,
  %float: !mo.tensor<[], f32>,
  %seed: !mo.tensor<[], si64>
) -> !mo.tensor<[?], f32>{

  // CHECK-NEXT: %[[RES:.*]] = mo.static.random.normal(%[[FLOAT]], %[[FLOAT]], %[[SEED]]) : ({{.*}}) -> !mo.tensor<[[[N]]], f32>
  %size = mo.shape_of(%dummy) : (!mo.tensor<[N], f32>) -> !mo.tensor<[1], si64>
  %res = mo.random.normal(%size, %float, %float, %seed) :
        (!mo.tensor<[1], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>,
        !mo.tensor<[], si64>) -> !mo.tensor<[?], f32>

  // CHECK-NEXT: mo.output %[[RES]]
  mo.output %res: !mo.tensor<[?], f32>
}

// CHECK-LABEL: mo.graph @reshape_neg_one_dim_in_const_shape
// CHECK-SAME:   <[[D0:.*]] -> [[D5:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]], 4], si32>
// CHECK-SAME:  ) -> !mo.tensor<[3, [[D5]], 2], si32> {
mo.graph @reshape_neg_one_dim_in_const_shape<D0 -> D4, D5, D6>(
  %input: !mo.tensor<[D0, 4], si32>
) -> !mo.tensor<[D4, D5, D6], si32> {

  // CHECK-NEXT: kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[reshape] input and output number of elements must match" : !kgen.string}, eq(mul_nuw(div(mul_nuw([[D0]], 2), 3), 6), mul_nuw([[D0]], 4)))>
  // CHECK-NEXT: kgen.param.declare [[EXPR:.*]] = <div(mul_nuw([[D0]], 2), 3)>
  // CHECK-NEXT: %[[RES:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[[[D0:.*]], 4], si32> -> !mo.tensor<[3, [[EXPR]], 2], si32>
  %shape = mo.constant {value = #M.dense_array<3, -1, 2> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %res = mo.reshape<() -> D1, D2, D3>(%input, %shape) : (!mo.tensor<[D0, 4], si32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D1, D2, D3], si32>

  // CHECK-NEXT: %[[REB:.*]] = mo.rebind(%[[RES]])
  // CHECK-NEXT: mo.output<{{.*}}> %[[REB]]
  mo.output<D1, D2, D3> %res : !mo.tensor<[D1, D2, D3], si32>
}

// CHECK-LABEL: mo.graph @reshape_flatten_const_shape
// CHECK-SAME:   <[[D0:.*]] -> [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]], 4], si32>
// CHECK-SAME:  ) -> !mo.tensor<[[[D2]]], si32> {
mo.graph @reshape_flatten_const_shape<D0 -> D2>(
  %input: !mo.tensor<[D0, 4], si32>
) -> !mo.tensor<[D2], si32> {

  // CHECK-NEXT: kgen.param.declare [[EXPR:.*]] = <mul_nuw([[D0]], 4)>
  // CHECK-NEXT: %[[RES:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[[[D0:.*]], 4], si32> -> !mo.tensor<[[[EXPR]]], si32>
  %shape = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %res = mo.reshape<() -> D1>(%input, %shape) : (!mo.tensor<[D0, 4], si32>, !mo.tensor<[1], si64>) -> !mo.tensor<[D1], si32>

  // CHECK-NEXT: %[[REB:.*]] = mo.rebind(%[[RES]])
  // CHECK-NEXT: mo.output<{{.*}}> %[[REB]]
  mo.output<D1> %res : !mo.tensor<[D1], si32>
}

// CHECK-LABEL: mo.graph @e2e_fold
// CHECK-SAME:  <[[M:.*]], [[N:.*]]>(
// CHECK-SAME:   %[[ARG0:.*]]: !mo.tensor<[[[M]], 1], f32>,
// CHECK-SAME:   %[[ARG1:.*]]: !mo.tensor<[1, [[N]]], f32>
// CHECK-SAME:   %[[ARG2:.*]]: !mo.tensor<[[[N]], [[M]]], f32>
// CHECK-SAME:  ) -> !mo.tensor<[[[N]], [[M]]], f32> {
mo.graph @e2e_fold<M, N -> E, F>(
  %arg0: !mo.tensor<[M, 1], f32>,
  %arg1: !mo.tensor<[1, N], f32>,
  %arg2: !mo.tensor<[N, M], f32>
) -> !mo.tensor<[E, F], f32> {
  %sht0 = mo.shape_of(%arg0) : (!mo.tensor<[M, 1], f32>) -> !mo.tensor<[2], si64>
  %sht1 = mo.shape_of(%arg1) : (!mo.tensor<[1, N], f32>) -> !mo.tensor<[2], si64>
  %s3 = mo.broadcast_shape(%sht0, %sht1) : (!mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2], si64>

  // CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[[[M]], 1], f32> -> !mo.tensor<[[[M]], [[N]]], f32>
  %3 = mo.broadcast_to<() -> A, B>(%arg0, %s3) : (
    !mo.tensor<[M, 1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[A, B], f32>

  // CHECK-NEXT: %[[RESHAPE:.*]] = mo.static.reshape(%[[BCAST]]) : !mo.tensor<[[[M]], [[N]]], f32> -> !mo.tensor<[[[N]], [[M]]], f32>
  %sht2 = mo.shape_of(%arg2) : (!mo.tensor<[N, M], f32>) -> !mo.tensor<[2], si64>
  %res = mo.reshape<() -> C, D>(%3, %sht2) : (!mo.tensor<[A, B], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[C, D], f32>

  // CHECK-NEXT: mo.output %[[RESHAPE]]
  mo.output<C, D> %res : !mo.tensor<[C, D], f32>
}

// CHECK-LABEL: mo.graph @consecutive_rebind_during_staticization
// CHECK-SAME:   %[[ARG1:.*]]: !mo.tensor
mo.graph @consecutive_rebind_during_staticization(%arg1: !mo.tensor<[?, ?], si32>) -> !mo.tensor<[?, ?], si32> {
  // This IR is taken from TF GPT2, before shape inference.
  %0 = mo.constant {value = #M.dense_array<> : tensor<0xsi32>} : !mo.tensor<[0], si32>
  %1 = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %2 = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %3 = mo.constant {value = #M.dense_array<-1> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %4 = mo.constant {value = #M.dense_array<1> : tensor<si32>} : !mo.tensor<[], si32>
  %5 = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %6 = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %7 = mo.shape_of(%arg1) : (!mo.tensor<[?, ?], si32>) -> !mo.tensor<[2], si32>
  %8 = mo.rebind(%0) : !mo.tensor<[0], si32> -> !mo.tensor<[?], si32>
  %9 = mo.slice(%7, %2, %6, %2) : (!mo.tensor<[2], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[?], si32>
  %10 = mo.shape_of(%9) : (!mo.tensor<[?], si32>) -> !mo.tensor<[1], si64>
  %11 = mo.shape_of(%10) : (!mo.tensor<[1], si64>) -> !mo.tensor<[1], si32>
  %12 = mo.static.reshape(%11) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %13 = mo.range(%4, %12, %4) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %14 = mo.gather(%10, %8, %1) : (!mo.tensor<[1], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %15 = mo.gather(%10, %13, %1) : (!mo.tensor<[1], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %16 = mo.concat[%1 : !mo.tensor<[], si32>] (%14, %15) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>

  // This reshape will be canonicalized into a static.reshape, then merged with
  // the subsequent static.reshape, and then become a rebind during shape
  // inference, and we end up having 2 consecutive rebinds starting from %20.
  %17 = mo.reshape(%9, %16) : (!mo.tensor<[?], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<[], si32>
  %18 = mo.rebind(%3) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %19 = mo.static.reshape(%17) : !mo.tensor<[], si32> -> !mo.tensor<[1], si32>
  %20 = mo.rebind(%19) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %21 = mo.concat[%5 : !mo.tensor<[], si64>] (%18, %20) : (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[2], si32>
  %22 = mo.reshape(%arg1, %21) : (!mo.tensor<[?, ?], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[?, ?], si32>

  // CHECK-NEXT: "mo.assert"{{.*}}"[reshape] cannot infer dimension when a specified output dimension is 0"{{.*}}ne({{.*}}, 0))>
  // CHECK-NEXT: mo.output{{.*}} %[[ARG1]]
  mo.output %22 : !mo.tensor<[?, ?], si32>
}

// This is a regression test taken from mistral-7b-hf-lptg-onnx, where
// - D0 := seq_len
// - D1 := seq_len + past_seq_len
// - D2 := batch_size
//
// Make sure everything folds to a single static broadcast.
// CHECK-LABEL: mo.graph @staticize_broadcast_on_constants
// CHECK-SAME:    <[[D0:.*]], [[D1:.*]], [[D2:.*]]>(
// CHECK-SAME:    %[[LHS:.*]]: !mo.tensor<[1, 1, [[D0]], [[D1]]], f32>,
// CHECK-SAME:    %[[RHS:.*]]: !mo.tensor<[[[D2]], 1, [[D0]], [[D1]]], f32>
//
// CHECK-NEXT:    %[[RES:.*]] = mo.static.broadcast_to(%[[LHS]]) : {{.*}} -> !mo.tensor<[[[D2]], 1, [[D0]], [[D1]]], f32>
// CHECK-NEXT:    mo.output %[[RES]]
mo.graph @staticize_broadcast_on_constants<D0, D1, D2 -> D3, D4, D5, D6>(
  %lhs: !mo.tensor<[1, 1, D0, D1], f32>,
  %rhs: !mo.tensor<[D2, 1, D0, D1], f32>
) -> !mo.tensor<[D3, D4, D5, D6], f32> {
  // Constants.
  %one = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %neg_one = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>

  // This is where things got blocked -- we turned off constant folding for
  // broadcast in #36054, so these broadcasts won't be folded into constants,
  // and we failed to staticize them, which blocked shape inference all the way
  // to the broadcast op at the end.
  %ones = mo.static.broadcast_to(%one) : !mo.tensor<[], si64> -> !mo.tensor<[4], si64>
  %neg_ones = mo.static.broadcast_to(%neg_one) : !mo.tensor<[], si64> -> !mo.tensor<[4], si64>

  // Input shapes.
  %lhs_shape = mo.shape_of(%lhs) : (!mo.tensor<[1, 1, D0, D1], f32>) -> !mo.tensor<[4], si64>
  %rhs_shape = mo.shape_of(%rhs) : (!mo.tensor<[D2, 1, D0, D1], f32>) -> !mo.tensor<[4], si64>

  // Not sure why but this seemingly redundant check shows up in the model.
  %bools = mo.equal(%rhs_shape, %neg_ones) : (!mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[4], bool>
  %ones_or_rhs_shape = mo.select(%bools, %ones, %rhs_shape) : (!mo.tensor<[4], bool>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[4], si64>

  // Do the broadcast.
  %bcast_shape = mo.broadcast_shape(%lhs_shape, %ones_or_rhs_shape) : (!mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[4], si64>
  %res = mo.broadcast_to<() -> D7, D8, D9, D10>(%lhs, %bcast_shape) : (!mo.tensor<[1, 1, D0, D1], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[D7, D8, D9, D10], f32>

  mo.output<D7, D8, D9, D10> %res : !mo.tensor<[D7, D8, D9, D10], f32>
}

// CHECK-LABEL: mo.graph @reshape_infer_dim_folded
//  CHECK-SAME:  <[[D0:.*]], [[D1:.*]]>(
//  CHECK-SAME:  %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>
mo.graph @reshape_infer_dim_folded<D0, D1 -> D2, D3>(
  %input: !mo.tensor<[D0, D1], f32>
) -> !mo.tensor<[D2, D3], f32> {

  // CHECK-NEXT: "mo.assert"{{.*}}"[reshape] cannot infer dimension when a specified output dimension is 0"{{.*}}ne([[D1]], 0))>
  // CHECK-NEXT: %[[RESULT:.*]] = mo.static.reshape(%[[INPUT]]) : {{.*}} -> !mo.tensor<[[[D1]], [[D0]]], f32>
  %symbolic_target_shape = mosh.param.to_value: !mosh.ape = <[D1, -1]>
  %target_shape = mo.shape.to_tensor(%symbolic_target_shape) -> !mo.tensor<[2], si32>
  %result = mo.reshape<() -> D4, D5>(%input, %target_shape) : (
    !mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[D4, D5], f32>

  // CHECK-NEXT: mo.output %[[RESULT]] : {{.*}}
  mo.output<D4, D5> %result : !mo.tensor<[D4, D5], f32>
}

// This is created from `replitv15-3B-hf-context-encoding-bf16-onnx`, as a
// regression test to make sure we can staticize `mo.negative`.
// CHECK-LABEL: mo.graph @staticize_mo_negative
//  CHECK-SAME:  <[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]] -> {{.*}}>(
//  CHECK-SAME:  %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], bf16>
// TODO(GRA-449)
//       CHECK:   kgen.param.declare [[PARAM_EXPR:.*]] = <div(mul_nuw([[D2]], -1), -1)>
//  CHECK-NEXT:   %[[RES:.*]] = mo.range{{.*}} -> !mo.tensor<[[[PARAM_EXPR]]], si64>
//  CHECK-NEXT:   %[[RES_REB:.*]] = mo.rebind(%[[RES]])
//  CHECK-NEXT:   mo.output{{.*}} %[[RES_REB:.*]]
mo.graph @staticize_mo_negative<D0, D1, D2, D3 -> D4>(
  %input: !mo.tensor<[D0, D1, D2, D3], bf16>
) -> !mo.tensor<[D4], si64> {

  // Constants
  %zero = mo.constant {value = #M.dense_array<0> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %two = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %neg_one = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>

  // Extracting `D2` and make it negative
  %input_shape = mo.shape_of(%input) : (!mo.tensor<[D0, D1, D2, D3], bf16>) -> !mo.tensor<[4], si64>
  %d2 = mo.gather(%input_shape, %two, %zero) : (!mo.tensor<[4], si64>, !mo.tensor<[1], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[1], si64>
  %d2_scalar = mo.static.reshape(%d2) : !mo.tensor<[1], si64> -> !mo.tensor<[], si64>
  %neg_d2 = mo.negative(%d2_scalar) : !mo.tensor<[], si64>

  // This part is contrived, in the original model the `mo.negative` was used to
  // create a negative index for slicing, but I didn't want to go that far for
  // this test, which focuses on being able to staticize `mo.negative`.
  %res = mo.range<() -> D5>(%zero, %neg_d2, %neg_one) : (!mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[D5], si64>

  mo.output<D5> %res : !mo.tensor<[D5], si64>
}

// This checks we can use range-based folding to simplify patterns like `lt(X +
// Y, X)`. We see such pattern generated from `mo.slice` shape function in
// `replitv15-3B-hf-context-encoding-bf16-onnx`.
// CHECK-LABEL: mo.graph @fold_non_trivial_lt
//  CHECK-SAME:  <[[X:.*]], [[Y:.*]]>(
//  CHECK-SAME:  %[[INPUT:.*]]: !mo.tensor<[[[X]], [[Y]]], f32>
//  CHECK-NEXT:   mo.output %[[INPUT]]
mo.graph @fold_non_trivial_lt<X, Y -> D0, D1>(
  %input: !mo.tensor<[X, Y], f32>
) -> !mo.tensor<[D0, D1], f32> {

  // NOTE this is somewhat contrived -- in general shape function (or maybe some
  // rare staticization) generates these parameter expressions, but this test
  // aims to check the simplification aspect, so we make this contrived test
  // case. The generation part is also expected to go through some changes, so
  // this avoids the churn. We'll likely add more e2e tests once the changes are
  // completed.

  // This is essentially `(X + Y < Y) ? (X + Y) : X
  kgen.param.declare P0 = <add(X, Y)>
  kgen.param.declare P1: i1 = <lt(P0, Y)>
  kgen.param.declare P2 = <cond(P1, P0, X)>

  // This is essentially `(2*X + Y < X) ? X : Y
  kgen.param.declare P3 = <add(X, Y, X)>
  kgen.param.declare P4: i1 = <lt(P0, X)>
  kgen.param.declare P5 = <cond(P1, X, Y)>

  // This reshape will be folded away since in/out shapes will become equal.
  %res = mo.static.reshape(%input) : !mo.tensor<[X, Y], f32> -> !mo.tensor<[P2, P5], f32>

  mo.output<P2, P5> %res : !mo.tensor<[P2, P5], f32>
}

// CHECK-LABEL: mo.graph @split_shape
//  CHECK-SAME:  <[[N:.*]]>(
//  CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[[[N]], 20], f32>
mo.graph @split_shape<N -> D0, D1, D2, D3>(
  %input: !mo.tensor<[N, 20], f32>
) -> (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>) {

  // CHECK-NEXT: %[[SPLIT_SIZES:.*]] = mo.constant {value = #M.dense_array<5, 15> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<-1> : tensor<si32>} : !mo.tensor<[], si32>
  %split_sizes = mo.constant {value = #M.dense_array<5, 15> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %axis = mo.constant {value = #M.dense_array<-1> : tensor<si32>} : !mo.tensor<[], si32>

  //      CHECK: %[[RES:.*]]:2 = mo.split[%[[AXIS]] :{{.*}}] (%[[INPUT]], %[[SPLIT_SIZES]])
  // CHECK-SAME:   : ({{.*}}) -> (!mo.tensor<[[[N]], 5], f32>, !mo.tensor<[[[N]], 15], f32>)
  %res:2 = mo.split<() -> D4, D5, D6, D7>[%axis: !mo.tensor<[], si32>](%input, %split_sizes) : (
    !mo.tensor<[N, 20], f32>, !mo.tensor<[2], si64>
  ) -> (!mo.tensor<[D4, D5], f32>, !mo.tensor<[D6, D7], f32>)

  // CHECK-NEXT: mo.output %[[RES]]#0, %[[RES]]#1
  mo.output<D4, D5, D6, D7> %res#0, %res#1 : !mo.tensor<[D4, D5], f32>, !mo.tensor<[D6, D7], f32>
}

// CHECK-LABEL: mo.graph @roi_align
//  CHECK-SAME:   <[[C:.*]], [[M:.*]]>(
//  CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[1, 2, 3, [[C]]], f32>,
//  CHECK-SAME:    %[[ROIS:.*]]: !mo.tensor<[[[M]], 5], f32>
mo.graph @roi_align<C, M -> D0, D1, D2, D3>(
  %input: !mo.tensor<[1, 2, 3, C], f32>,
  %rois: !mo.tensor<[M, 5], f32>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {

  %out_h = mo.constant {value = #M.dense_array<10> : tensor<si64>} : !mo.tensor<[], si64>
  %out_w = mo.constant {value = #M.dense_array<20> : tensor<si64>} : !mo.tensor<[], si64>
  %ss = mo.constant {value = #M.dense_array<1.0> : tensor<1xf32>} : !mo.tensor<[], f32>
  %sr = mo.constant {value = #M.dense_array<2.0> : tensor<1xf32>} : !mo.tensor<[], f32>

  // CHECK: %[[RES:.*]] = mo.roi_align{{.*}} -> !mo.tensor<[[[M]], 10, 20, [[C]]], f32>
  %res = mo.roi_align<() -> D4, D5, D6, D7>(%input, %rois, %out_h, %out_w, %ss, %sr) {aligned = false,  mode = "AVG"} : (
    !mo.tensor<[1, 2, 3, C], f32>, !mo.tensor<[M, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>
  ) -> !mo.tensor<[D4, D5, D6, D7], f32>

  // CHECK-NEXT: mo.output %[[RES]]
  mo.output<D4, D5, D6, D7> %res : !mo.tensor<[D4, D5, D6, D7], f32>
}
