// Note: due to the size of the file, we want to filecheck with --enable-var-scope
// which prevents variables from being reused across CHECK-LABEL boundaries.
//
// RUN: mo-opt %s --canonicalize --shape-inference=experimental=false --canonicalize --hoist-param-exprs | FileCheck %s --enable-var-scope
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
// CHECK-SAME: @static_ranks<[[D11:.*]], [[D12:.*]], [[D13:.*]], [[D14:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D11]], [[D12]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D13]], [[D14]]], f32>) -> !mo.tensor<[[[D11]], [[D12]]], f32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.rebind([[VARG1]]) : !mo.tensor<[[[D13]], [[D14]]], f32> -> !mo.tensor<[[[D11]], [[D12]]], f32>
// CHECK-NEXT:   kgen.param.declare SORS_CH10: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[add] input shapes must match" : !kgen.string}, and(eq([[D11]], [[D13]]), eq([[D12]], [[D14]])))>
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
// CHECK-SAME: @batch_matmul_static_rank<[[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]], [[D6:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[[D1]], [[D2]], [[D6]]], f32> no_inline {
// CHECK-NEXT:   kgen.param.declare [[P41:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "inner dimensions do not match" : !kgen.string}, eq([[D3]], [[D5]]))>
// CHECK-NEXT:   kgen.param.declare [[P45:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "batch sizes of left and right inputs are not equal" : !kgen.string}, eq([[D1]], [[D4]]))>
// CHECK-NEXT:   [[V0:%.*]] = mo.rebind([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32> -> !mo.tensor<[[[D1]], [[D3]], [[D6]]], f32>
// CHECK-NEXT:   [[V1:%.*]] = mo.batch_matmul([[VARG0]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[[D1]], [[D3]], [[D6]]], f32>) -> !mo.tensor<[[[D1]], [[D2]], [[D6]]], f32>
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
// CHECK-SAME: @materialize_gather_axis_0_si32_e2e<[[D6:.*]], [[D7:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D6]], 2], f32>, [[VARG1:%.*]]: !mo.tensor<[2, [[D7]]], si32>) -> !mo.tensor<[2, [[D7]], 2], f32> no_inline {
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
// CHECK-SAME: @materialize_gather_axis_neg_1_si32_e2e<[[D6:.*]], [[D7:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D6]], 2], f32>, [[VARG1:%.*]]: !mo.tensor<[2, [[D7]]], si32>) -> !mo.tensor<[[[D6]], 2, [[D7]]], f32> no_inline {
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
// CHECK-SAME: @materialize_mean_axis_0_si32_e2e<[[D4:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D4]], 2], f32>) -> !mo.tensor<[1, 2], f32> no_inline {
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
// CHECK-SAME: @materialize_prod_axis_1_si64_e2e<[[D4:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D4]], 2], f32>) -> !mo.tensor<[[[D4]], 1], f32> no_inline {
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
// CHECK-SAME: @materialize_sum_axis_neg_1_si64_e2e<[[D4:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D4]], 2], f32>) -> !mo.tensor<[[[D4]], 1], f32> no_inline {
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
// CHECK-SAME: @unresolved_rebinds<[[Sh11:.*]]: !mosh.ape, [[Sh12:.*]]: !mosh.ape -> [[Sh13:.*]]: !mosh.ape>([[VARG0:%.*]]: !mo.tensor<Sh11, f32>, [[VARG1:%.*]]: !mo.tensor<Sh12, f32>) -> !mo.tensor<Sh13, f32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.rebind([[VARG1]]) : !mo.tensor<[[Sh12]], f32> -> !mo.tensor<[[Sh11]], f32>
// CHECK-NEXT:   kgen.param.declare SORS_EQ16: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[Sh11]], [[Sh12]])>
// CHECK-NEXT:   kgen.param.declare SORS_CH17: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[add] input shapes must match" : !kgen.string}, SORS_EQ16)>
// CHECK-NEXT:   [[V1:%.*]] = mo.add([[VARG0]], [[V0]]) : !mo.tensor<[[Sh11]], f32>
// CHECK-NEXT:   [[V2:%.*]] = mo.gelu([[V1]]) : !mo.tensor<[[Sh11]], f32>
// CHECK-NEXT:   kgen.param.declare [[P31:.*]] = <apply(:(!mosh.ape) -> index "mosh.get_rank", [[Sh12]])>
// CHECK-NEXT:   kgen.param.declare [[P32:.*]] = <apply(:(!mosh.ape) -> index "mosh.get_rank", [[Sh11]])>
// CHECK-NEXT:   kgen.param.declare [[P34:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "ranks of left and right inputs are not equal" : !kgen.string}, eq([[P31:.*]], [[P32:.*]]))>
// CHECK-NEXT:   kgen.param.declare [[P36:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "input ranks are not greater than minimum rank for matmul-like" : !kgen.string}, ge([[P31:.*]], 3))>
// CHECK-NEXT:   kgen.param.declare [[P37:.*]] = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", [[Sh11]], -1)>
// CHECK-NEXT:   kgen.param.declare [[P38:.*]] = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", [[Sh12]], -2)>
// CHECK-NEXT:   kgen.param.declare [[P40:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "inner dimensions do not match" : !kgen.string}, eq([[P37:.*]], [[P38:.*]]))>
// CHECK-NEXT:   kgen.param.declare [[P41:.*]]: !mosh.ape = <apply(:(!mosh.ape, index, index) -> !mosh.ape "mosh.slice", [[Sh11]], 0, -2)>
// CHECK-NEXT:   kgen.param.declare [[P42:.*]]: !mosh.ape = <apply(:(!mosh.ape, index, index) -> !mosh.ape "mosh.slice", [[Sh12]], 0, -2)>
// CHECK-NEXT:   kgen.param.declare [[P43:.*]]: i1 = <apply(:(!mosh.ape, !mosh.ape) -> i1 "mosh.eq", [[P41:.*]], [[P42:.*]])>
// CHECK-NEXT:   kgen.param.declare [[P44:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "batch sizes of left and right inputs are not equal" : !kgen.string}, [[P43:.*]])>
// CHECK-NEXT:   kgen.param.declare [[P45:.*]] = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", [[Sh12]], -1)>
// CHECK-NEXT:   kgen.param.declare [[P47:.*]]: !mosh.ape = <apply(:(!mosh.ape, !mosh.ape) -> !mosh.ape "mosh.concat", [[P41:.*]], [[[P37:.*]], [[P45:.*]]])>
// CHECK-NEXT:   [[V3:%.*]] = mo.rebind([[VARG1]]) : !mo.tensor<[[Sh12]], f32> -> !mo.tensor<[[P47:.*]], f32>
// CHECK-NEXT:   kgen.param.declare [[P23:.*]] = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", [[Sh11]], -2)>
// CHECK-NEXT:   kgen.param.declare [[P24:.*]] = <apply(:(!mosh.ape, index) -> index "mosh.get_dim", [[P47:.*]], -1)>
// CHECK-NEXT:   kgen.param.declare [[P25:.*]]: !mosh.ape = <apply(:(!mosh.ape, index, index) -> !mosh.ape "mosh.slice", [[Sh11]], 0, -2)>
// CHECK-NEXT:   kgen.param.declare [[P27:.*]]: !mosh.ape = <apply(:(!mosh.ape, !mosh.ape) -> !mosh.ape "mosh.concat", [[P25:.*]], [[[P23:.*]], [[P24:.*]]])>
// CHECK-NEXT:   [[V4:%.*]] = mo.batch_matmul([[V2]], [[V3]]) : (!mo.tensor<[[Sh11]], f32>, !mo.tensor<[[P47:.*]], f32>) -> !mo.tensor<[[P27:.*]], f32>
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
// CHECK-SAME: @broadcasted_binary_elwise<[[D21:.*]], [[D22:.*]], [[D23:.*]], [[D24:.*]] -> [[D27:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D21]], 12, [[D22]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D23]], [[D24]]], f32>) -> !mo.tensor<[[[D21]], 12, [[D27]]], f32> no_inline {
// CHECK-NEXT:   kgen.param.declare dims_assert35: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions are not broadcastable" : !kgen.string}, or(eq([[D23]], 1), eq([[D23]], 12)))>
// CHECK-NEXT:   kgen.param.declare dims_assert40: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions are not broadcastable" : !kgen.string}, or(eq([[D22]], [[D24]]), eq([[D22]], 1), eq([[D24]], 1)))>
// CHECK-NEXT:   [[V0:%.*]] = mo.static.broadcast_to([[VARG0]]) : !mo.tensor<[[[D21]], 12, [[D22]]], f32> -> !mo.tensor<[[[D21]], 12, param_expr_41], f32>
// CHECK-NEXT:   [[V1:%.*]] = mo.static.broadcast_to([[VARG1]]) : !mo.tensor<[[[D23]], [[D24]]], f32> -> !mo.tensor<[[[D21]], 12, param_expr_41], f32>
// CHECK-NEXT:   kgen.param.declare param_expr_41 = <cond(eq([[D22]], 1), [[D24]], [[D22]])>
// CHECK-NEXT:   [[V2:%.*]] = mo.add([[V0]], [[V1]]) : !mo.tensor<[[[D21]], 12, param_expr_41], f32>
// CHECK-NEXT:   [[V3:%.*]] = mo.rebind([[V2]]) : !mo.tensor<[[[D21]], 12, param_expr_41], f32> -> !mo.tensor<[[[D21]], 12, cond(eq([[D22]], 1), [[D24]], [[D22]])], f32>
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
// CHECK-SAME: @transpose_static_rank<[[D9:.*]], [[D10:.*]], [[D11:.*]], [[D12:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D9]], [[D10]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D11]], [[D12]]], f32>) -> (!mo.tensor<[[[D9]], [[D10]]], f32>, !mo.tensor<[[[D12]], [[D11]]], f32>) no_inline {
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
// CHECK-SAME: @scalar_to_tensor_reshapes_before_staticization<[[D18:.*]], [[D19:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D18]], [[D19]]], si32>) -> !mo.tensor<[[[D18]], [[D18]]], si32> no_inline {
// CHECK-NEXT:   kgen.param.declare [[P29:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[reshape] input and output number of elements must match" : !kgen.string}, eq(mul_nuw([[D18]], [[D18]]), mul_nuw([[D18]], [[D19]])))>
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
// CHECK-SAME: @binary_op_asserts<[[D13:.*]], [[D14:.*]], [[D15:.*]], [[D16:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D13]], [[D14]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D15]], [[D16]]], f32>) -> !mo.tensor<[[[D13]], [[D14]]], f32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.add([[VARG0]], [[VARG0]]) : !mo.tensor<[[[D13]], [[D14]]], f32>
// CHECK-NEXT:   [[V1:%.*]] = mo.rebind([[VARG1]]) : !mo.tensor<[[[D15]], [[D16]]], f32> -> !mo.tensor<[[[D13]], [[D14]]], f32>
// CHECK-NEXT:   kgen.param.declare SORS_CH12: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[add] input shapes must match" : !kgen.string}, and(eq([[D13]], [[D15]]), eq([[D14]], [[D16]])))>
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
// CHECK-SAME: @binary_op_no_assert<[[D7:.*]], [[D8:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D7]], [[D8]]], f32>) -> !mo.tensor<[[[D7]], [[D8]]], f32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.add([[VARG0]], [[VARG0]]) : !mo.tensor<[[[D7]], [[D8]]], f32>
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
// CHECK-SAME: @batch_matmul_reused_rhs<[[D0:.*]], [[D1:.*]]>([[VARG0:%.*]]: !mo.tensor<[3, [[D0]], 4], f32>, [[VARG1:%.*]]: !mo.tensor<[3, 4, [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.batch_matmul([[VARG0]], [[VARG1]]) : (!mo.tensor<[3, [[D0]], 4], f32>, !mo.tensor<[3, 4, [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32>
// CHECK-NEXT:   kgen.param.declare [[P25:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "inner dimensions do not match" : !kgen.string}, eq([[D1]], 4))>
// CHECK-NEXT:   [[V1:%.*]] = mo.rebind([[VARG1]]) : !mo.tensor<[3, 4, [[D1]]], f32> -> !mo.tensor<[3, [[D1]], [[D1]]], f32>
// CHECK-NEXT:   [[V2:%.*]] = mo.batch_matmul([[V0]], [[V1]]) : (!mo.tensor<[3, [[D0]], [[D1]]], f32>, !mo.tensor<[3, [[D1]], [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32>
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
// CHECK-SAME: @linalg_solve_reused_rhs<[[D0:.*]], [[D1:.*]]>([[VARG0:%.*]]: !mo.tensor<[3, [[D0]], [[D0]]], f32>, [[VARG1:%.*]]: !mo.tensor<[3, [[D0]], [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.linalg.solve([[VARG0]], [[VARG1]]) : (!mo.tensor<[3, [[D0]], [[D0]]], f32>, !mo.tensor<[3, [[D0]], [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32>
// CHECK-NEXT:   kgen.param.declare [[P25:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "inner dimensions do not match" : !kgen.string}, eq([[D0]], [[D1]]))>
// CHECK-NEXT:   [[V1:%.*]] = mo.rebind([[VARG1]]) : !mo.tensor<[3, [[D0]], [[D1]]], f32> -> !mo.tensor<[3, [[D1]], [[D1]]], f32>
// CHECK-NEXT:   [[V2:%.*]] = mo.linalg.solve([[V0]], [[V1]]) : (!mo.tensor<[3, [[D0]], [[D1]]], f32>, !mo.tensor<[3, [[D1]], [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32>
// CHECK-NEXT:   mo.output [[V2]] : !mo.tensor<[3, [[D0]], [[D1]]], f32>
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
// CHECK-SAME: @slice_neg_stop([[VARG0:%.*]]: !mo.tensor<[10], f32>) -> !mo.tensor<[3], f32> no_inline {
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
// CHECK-SAME: @slice_clamped_start_stop_neg_step([[VARG0:%.*]]: !mo.tensor<[10], f32>) -> !mo.tensor<[5], f32> no_inline {
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
// CHECK-SAME: @slice_4d([[VARG0:%.*]]: !mo.tensor<[10, 20, 30, 40], f32>) -> !mo.tensor<[3, 3, 8, 8], f32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<3, -2, -100, 200> : tensor<4xsi32>} : !mo.tensor<[4], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<-1, 10, 200, -100> : tensor<4xsi32>} : !mo.tensor<[4], si32>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<2, -3, 4, -5> : tensor<4xsi32>} : !mo.tensor<[4], si32>
// CHECK-NEXT:   [[V3:%.*]] = mo.slice([[VARG0]], [[V0]], [[V1]], [[V2]]) : (!mo.tensor<[10, 20, 30, 40], f32>, !mo.tensor<[4], si32>, !mo.tensor<[4], si32>, !mo.tensor<[4], si32>) -> !mo.tensor<[3, 3, 8, 8], f32>
// CHECK-NEXT:   mo.output [[V3]] : !mo.tensor<[3, 3, 8, 8], f32>
// CHECK-NEXT: }

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
// CHECK-SAME: @scatter_nd([[VARG0:%.*]]: !mo.tensor<[10, 1, 1], f32>, [[VARG1:%.*]]: !mo.tensor<[5, 1, 1], f32>, [[VARG2:%.*]]: !mo.tensor<[5], si32>) -> !mo.tensor<[10, 1, 1], f32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.static.reshape([[VARG2]]) : !mo.tensor<[5], si32> -> !mo.tensor<[5, 1], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.scatter_nd([[VARG0]], [[VARG1]], [[V0]]) : (!mo.tensor<[10, 1, 1], f32>, !mo.tensor<[5, 1, 1], f32>, !mo.tensor<[5, 1], si32>) -> !mo.tensor<[10, 1, 1], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[10, 1, 1], f32>
// CHECK-NEXT: }

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
// CHECK-SAME: @scatter_nd_add([[VARG0:%.*]]: !mo.tensor<[10, 1, 1], f32>, [[VARG1:%.*]]: !mo.tensor<[5, 1, 1], f32>, [[VARG2:%.*]]: !mo.tensor<[5], si32>) -> !mo.tensor<[10, 1, 1], f32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.static.reshape([[VARG2]]) : !mo.tensor<[5], si32> -> !mo.tensor<[5, 1], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.scatter_nd.add([[VARG0]], [[VARG1]], [[V0]]) : (!mo.tensor<[10, 1, 1], f32>, !mo.tensor<[5, 1, 1], f32>, !mo.tensor<[5, 1], si32>) -> !mo.tensor<[10, 1, 1], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[10, 1, 1], f32>
// CHECK-NEXT: }

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
// CHECK-SAME: @conv2d_static_input_dims([[VARG0:%.*]]: !mo.tensor<[10, 5, 5, 16], f32>, [[VARG1:%.*]]: !mo.tensor<[3, 3, 16, 32], f32>) -> !mo.tensor<[10, 2, 2, 32], f32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V4:%.*]] = mo.conv([[VARG0]], [[VARG1]]) [strides = [[V0]], dilations = [[V1]], paddings = [[V2]], num_groups = [[V3]]] : (!mo.tensor<[10, 5, 5, 16], f32>, !mo.tensor<[3, 3, 16, 32], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[10, 2, 2, 32], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[10, 2, 2, 32], f32>
// CHECK-NEXT: }

mo.graph @conv2d_dyn_input_dims(
  %input: !mo.tensor<[?, 20, 30, ?], f32>,
  %filter: !mo.tensor<[3, 3, ?, ?], f32>
) -> !mo.tensor<[?, ?, ?, ?], f32> {
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %ng = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %0 = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
    !mo.tensor<[?, 20, 30, ?], f32>,
    !mo.tensor<[3, 3, ?, ?], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>,
    !mo.tensor<[], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @conv2d_dyn_input_dims<[[D9:.*]], [[D10:.*]], [[D11:.*]], [[D12:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D9]], 20, 30, [[D10]]], f32>, [[VARG1:%.*]]: !mo.tensor<[3, 3, [[D11]], [[D12]]], f32>) -> !mo.tensor<[[[D9]], 7, 12, [[D12]]], f32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
// CHECK-NEXT:   kgen.param.declare [[P39:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "Groups times filter channels must match input channels" : !kgen.string}, eq([[D10]], [[D11]]))>
// CHECK-NEXT:   [[V4:%.*]] = mo.conv([[VARG0]], [[VARG1]]) [strides = [[V3]], dilations = [[V2]], paddings = [[V1]], num_groups = [[V0]]] : (!mo.tensor<[[[D9]], 20, 30, [[D10]]], f32>, !mo.tensor<[3, 3, [[D11]], [[D12]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D9]], 7, 12, [[D12]]], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[[[D9]], 7, 12, [[D12]]], f32>
// CHECK-NEXT: }

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
// CHECK-SAME: @max_pool2d_static_input_dims([[VARG0:%.*]]: !mo.tensor<[10, 4, 4, 16], f32>) -> !mo.tensor<[10, 1, 1, 16], f32> no_inline {
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
// CHECK-SAME: @max_pool2d_ceil_mode_true_static_input_dims([[VARG0:%.*]]: !mo.tensor<[10, 4, 4, 16], f32>) -> !mo.tensor<[10, 2, 2, 16], f32> no_inline {
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
// CHECK-SAME: @avg_pool2d_dyn_input_dims<[[D7:.*]], [[D8:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D7]], 20, 30, [[D8]]], f32>) -> !mo.tensor<[[[D7]], 9, 14, [[D8]]], f32> no_inline {
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
// CHECK-SAME: @avg_pool2d_ceil_mode_true_dyn_input_dims<[[D7:.*]], [[D8:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D7]], 20, 30, [[D8]]], f32>) -> !mo.tensor<[[[D7]], 10, 15, [[D8]]], f32> no_inline {
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
// CHECK-SAME: @arg_max<[[D5:.*]], [[D6:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D5]], [[D6]]], si64>) -> !mo.tensor<[[[D5]], 1], si64> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.arg_max([[VARG0]], [[V0]]) : (!mo.tensor<[[[D5]], [[D6]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D5]], 1], si64>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D5]], 1], si64>
// CHECK-NEXT: }

mo.graph @concat(
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
// CHECK-SAME: @concat<[[D10:.*]], [[D11:.*]], [[D12:.*]], [[D13:.*]], [[D14:.*]], [[D15:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D10]], 1, [[D11]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D12]], 2, [[D13]]], f32>, [[VARG2:%.*]]: !mo.tensor<[[[D14]], 3, [[D15]]], f32>) -> !mo.tensor<[[[D10]], 6, [[D11]]], f32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   kgen.param.declare [[P50:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions before axis must be equal" : !kgen.string}, eq([[D10]], [[D12]]))>
// CHECK-NEXT:   kgen.param.declare [[P52:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions after axis must be equal" : !kgen.string}, eq([[D11]], [[D13]]))>
// CHECK-NEXT:   kgen.param.declare [[P62:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions before axis must be equal" : !kgen.string}, eq([[D10]], [[D14]]))>
// CHECK-NEXT:   kgen.param.declare [[P64:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions after axis must be equal" : !kgen.string}, eq([[D11]], [[D15]]))>
// CHECK-NEXT:   [[V1:%.*]] = mo.concat[[[V0]] : !mo.tensor<[], si64>] ([[VARG0]], [[VARG1]], [[VARG2]]) : (!mo.tensor<[[[D10]], 1, [[D11]]], f32>, !mo.tensor<[[[D12]], 2, [[D13]]], f32>, !mo.tensor<[[[D14]], 3, [[D15]]], f32>) -> !mo.tensor<[[[D10]], 6, [[D11]]], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D10]], 6, [[D11]]], f32>
// CHECK-NEXT: }

mo.graph @concat_neg_axis(
  %input1: !mo.tensor<[?, 1, ?], f32>,
  %input2: !mo.tensor<[?, 2, ?], f32>,
  %input3: !mo.tensor<[?, 3, ?], f32>
) -> !mo.tensor<[?, ?, ?], f32> {
  %axis = mo.constant {value = #M.dense_array<-2> : tensor<si64>} : !mo.tensor<[], si64>
  %res = mo.concat[%axis : !mo.tensor<[], si64>](%input1, %input2, %input3) :
    (!mo.tensor<[?, 1, ?], f32>, !mo.tensor<[?, 2, ?], f32>, !mo.tensor<[?, 3, ?], f32>)
    -> !mo.tensor<[?, ?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @concat_neg_axis<[[D10:.*]], [[D11:.*]], [[D12:.*]], [[D13:.*]], [[D14:.*]], [[D15:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D10]], 1, [[D11]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D12]], 2, [[D13]]], f32>, [[VARG2:%.*]]: !mo.tensor<[[[D14]], 3, [[D15]]], f32>) -> !mo.tensor<[[[D10]], 6, [[D11]]], f32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<-2> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   kgen.param.declare [[P50:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions before axis must be equal" : !kgen.string}, eq([[D10]], [[D12]]))>
// CHECK-NEXT:   kgen.param.declare [[P52:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions after axis must be equal" : !kgen.string}, eq([[D11]], [[D13]]))>
// CHECK-NEXT:   kgen.param.declare [[P62:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions before axis must be equal" : !kgen.string}, eq([[D10]], [[D14]]))>
// CHECK-NEXT:   kgen.param.declare [[P64:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions after axis must be equal" : !kgen.string}, eq([[D11]], [[D15]]))>
// CHECK-NEXT:   [[V1:%.*]] = mo.concat[[[V0]] : !mo.tensor<[], si64>] ([[VARG0]], [[VARG1]], [[VARG2]]) : (!mo.tensor<[[[D10]], 1, [[D11]]], f32>, !mo.tensor<[[[D12]], 2, [[D13]]], f32>, !mo.tensor<[[[D14]], 3, [[D15]]], f32>) -> !mo.tensor<[[[D10]], 6, [[D11]]], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D10]], 6, [[D11]]], f32>
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
// CHECK-SAME: @pad_constant([[VARG0:%.*]]: !mo.tensor<[2, 3], f32>, [[VARG1:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[32, 73], f32> no_inline {
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
// CHECK-SAME: @pad_reflect([[VARG0:%.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[32, 73], f32> no_inline {
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
// CHECK-SAME: @pad_repeat([[VARG0:%.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[32, 73], f32> no_inline {
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
// CHECK-SAME: @tile([[VARG0:%.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[20, 60], f32> no_inline {
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
// CHECK-SAME: @partially_parameterized_signature<in, [[D6:.*]], [[D7:.*]]>([[VARG0:%.*]]: !mo.tensor<[in, [[D6]], [[D7]]], f32>) -> !mo.tensor<[in, 1, [[D7]]], f32> no_inline {
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
// CHECK-SAME: @binary_logical_op<[[D7:.*]], [[D8:.*]], [[D9:.*]], [[D10:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D7]], [[D8]]], bool>, [[VARG1:%.*]]: !mo.tensor<[[[D9]], [[D10]]], bool>) -> !mo.tensor<[[[D7]], [[D8]]], bool> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.rebind([[VARG1]]) : !mo.tensor<[[[D9]], [[D10]]], bool> -> !mo.tensor<[[[D7]], [[D8]]], bool>
// CHECK-NEXT:   kgen.param.declare SORS_CH18: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[and] input shapes must match" : !kgen.string}, and(eq([[D10]], [[D8]]), eq([[D7]], [[D9]])))>
// CHECK-NEXT:   [[V1:%.*]] = mo.and([[VARG0]], [[V0]]) : !mo.tensor<[[[D7]], [[D8]]], bool>
// CHECK-NEXT:   kgen.param.declare SORS_CH6: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "[xor] input shapes must match" : !kgen.string}, and(eq([[D10]], [[D8]]), eq([[D7]], [[D9]])))>
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
// CHECK-SAME: @top_k([[VARG0:%.*]]: !mo.tensor<[5, 5], si64>, [[VARG1:%.*]]: !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>) no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant {value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>} : !mo.tensor<[5, 5], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<3> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
// CHECK-NEXT:   [[VVALUES:%.*]], [[VINDICES:%.*]] = mo.top_k([[V0]], [[V1]], [[V2]], [[VARG1]]) : (!mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>)
// CHECK-NEXT:   mo.output [[VVALUES]], [[VINDICES]] : !mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>
// CHECK-NEXT: }

mo.graph @bottom_k(%input : !mo.tensor<[5, 5], si64>, %sorted : !mo.tensor<[], bool>) -> (!mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>) {
  %in = "mo.constant"() {value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>} : () -> !mo.tensor<[5, 5], si64>
  %K = "mo.constant"() {value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %axis = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %values, %indices = mo.bottom_k(%in, %K, %axis, %sorted) : (
    !mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>)
  mo.output %values, %indices : !mo.tensor<[?, ?], si64>, !mo.tensor<[?, ?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @bottom_k([[VARG0:%.*]]: !mo.tensor<[5, 5], si64>, [[VARG1:%.*]]: !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>) no_inline {
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
// CHECK-SAME: @unsqueeze_shape_dynamic_shape<[[D1:.*]], [[D2:.*]] -> [[D3:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D1]]], si32>, [[VARG1:%.*]]: !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[[[D3]]], si32> no_inline {
// CHECK-NEXT:   kgen.param.declare param_expr_20 = <add([[D1]], [[D2]])>
// CHECK-NEXT:   [[V0:%.*]] = mo.unsqueeze_shape([[VARG0]], [[VARG1]]) : (!mo.tensor<[[[D1]]], si32>, !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[param_expr_20], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.rebind([[V0]]) : !mo.tensor<[param_expr_20], si32> -> !mo.tensor<[add([[D1]], [[D2]])], si32>
// CHECK-NEXT:   mo.output<add([[D1]], [[D2]])> [[V1]] : !mo.tensor<[add([[D1]], [[D2]])], si32>
// CHECK-NEXT: }

mo.graph @unsqueeze_shape_dynamic_shape_with_reshape(%input_tensor: !mo.tensor<[?, ?, ?], si32>, %indices: !mo.tensor<[?], si32>) -> !mo.tensor<?, si32> {
  %shape_tensor = mo.shape_of(%input_tensor) : (!mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[3], si32>
  %result_shape = mo.unsqueeze_shape(%shape_tensor, %indices) : (!mo.tensor<[3], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32>
  %res = mo.reshape(%input_tensor, %result_shape) : (!mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<?, si32>
  mo.output %res : !mo.tensor<?, si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @unsqueeze_shape_dynamic_shape_with_reshape<[[D8:.*]], [[D9:.*]], [[D10:.*]], [[D11:.*]] -> [[Sh12:.*]]: !mosh.ape>([[VARG0:%.*]]: !mo.tensor<[[[D8]], [[D9]], [[D10]]], si32>, [[VARG1:%.*]]: !mo.tensor<[[[D11]]], si32>) -> !mo.tensor<Sh12, si32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = kgen.param.constant: !mosh.ape = <[[[D8]], [[D9]], [[D10]]]>
// CHECK-NEXT:   [[V1:%.*]] = mo.shape.to_tensor([[V0]]) -> !mo.tensor<[3], si32>
// CHECK-NEXT:   [[V2:%.*]] = mo.unsqueeze_shape([[V1]], [[VARG1]]) : (!mo.tensor<[3], si32>, !mo.tensor<[[[D11]]], si32>) -> !mo.tensor<[param_expr_29], si32>
// CHECK-NEXT:   kgen.param.declare param_expr_29 = <add([[D11]], 3)>
// CHECK-NEXT:   [[V3:%.*]] = mo.reshape<() -> [[Sh5:.*]]: !mosh.ape>([[VARG0]], [[V2]]) : (!mo.tensor<[[[D8]], [[D9]], [[D10]]], si32>, !mo.tensor<[param_expr_29], si32>) -> !mo.tensor<Sh5, si32>
// CHECK-NEXT:   mo.output<:!mosh.ape [[Sh5:.*]]> [[V3]] : !mo.tensor<[[Sh5:.*]], si32>
// CHECK-NEXT: }

mo.graph @gather_nd(%input: !mo.tensor<[?, ?, ?], si64>, %indices: !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, si64> {
  %result = mo.gather_nd(%input, %indices) {batchDims = 1} :
    (!mo.tensor<[?, ?, ?], si64>, !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, si64>
  mo.output %result : !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @gather_nd<[[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]] -> [[Sh6:.*]]: !mosh.ape>([[VARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], si64>, [[VARG1:%.*]]: !mo.tensor<[[[D4]], [[D5]]], si64>) -> !mo.tensor<Sh6, si64> no_inline {
// CHECK-NEXT:   kgen.param.declare [[P22:.*]]: scalar<bool> = <apply(:(i1) -> !pop.scalar<bool> "pop.cast_from_builtin", lt([[D5]], 3))>
// CHECK-NEXT:   kgen.param.declare [[P24:.*]]: scalar<bool> = <apply(:(i1) -> !pop.scalar<bool> "pop.cast_from_builtin", ge([[D5]], 1))>
// CHECK-NEXT:   kgen.param.declare [[P26:.*]]: i1 = <apply(:(!pop.scalar<bool>) -> i1 "pop.cast_to_builtin", cond(ge([[D5]], 1), [[P22:.*]], [[P24:.*]]))>
// CHECK-NEXT:   kgen.param.declare [[P27:.*]]: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "1 <= indices_shape[-1] <= input_rank - batch_dims" : !kgen.string}, [[P26:.*]])>
// CHECK-NEXT:   kgen.param.declare [[P31:.*]]: !mosh.ape = <apply(:(!mosh.ape, index, index) -> !mosh.ape "mosh.slice", [[[D1]], [[D2]], [[D3]]], add([[D5]], 1), 3)>
// CHECK-NEXT:   kgen.param.declare [[P32:.*]]: !mosh.ape = <apply(:(!mosh.ape, !mosh.ape, !mosh.ape) -> !mosh.ape "mosh.concat", [[[D4]]], [], [[P31:.*]])>
// CHECK-NEXT:   [[V0:%.*]] = mo.gather_nd([[VARG0]], [[VARG1]]) {batchDims = 1 : i64} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], si64>, !mo.tensor<[[[D4]], [[D5]]], si64>) -> !mo.tensor<[[P32:.*]], si64>
// CHECK-NEXT:   mo.output<:!mosh.ape [[P32:.*]]> [[V0]] : !mo.tensor<[[P32:.*]], si64>
// CHECK-NEXT: }

mo.graph @gather_nd_with_last_dim(%input: !mo.tensor<[?, ?, ?], si64>, %indices: !mo.tensor<[?, 1], si64>) -> !mo.tensor<?, si64> {
  %result = mo.gather_nd(%input, %indices) {batchDims = 1} :
    (!mo.tensor<[?, ?, ?], si64>, !mo.tensor<[?, 1], si64>) -> !mo.tensor<?, si64>
  mo.output %result : !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @gather_nd_with_last_dim<[[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], si64>, [[VARG1:%.*]]: !mo.tensor<[[[D4]], 1], si64>) -> !mo.tensor<[[[D4]], [[D3]]], si64> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.gather_nd([[VARG0]], [[VARG1]]) {batchDims = 1 : i64} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], si64>, !mo.tensor<[[[D4]], 1], si64>) -> !mo.tensor<[[[D4]], [[D3]]], si64>
// CHECK-NEXT:   mo.output [[V0]] : !mo.tensor<[[[D4]], [[D3]]], si64>
// CHECK-NEXT: }

mo.graph @squeeze_shape_dynamic_shape(%input_shape: !mo.tensor<[?], si32>, %indices: !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32> {
  %res = mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32>
  mo.output %res : !mo.tensor<[?], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @squeeze_shape_dynamic_shape<[[D1:.*]], [[D2:.*]] -> [[D3:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D1]]], si32>, [[VARG1:%.*]]: !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[[[D3]]], si32> no_inline {
// CHECK-NEXT:   kgen.param.declare param_expr_20 = <add(mul([[D2]], -1), [[D1]])>
// CHECK-NEXT:   [[V0:%.*]] = mo.squeeze_shape([[VARG0]], [[VARG1]]) : (!mo.tensor<[[[D1]]], si32>, !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[param_expr_20], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.rebind([[V0]]) : !mo.tensor<[param_expr_20], si32> -> !mo.tensor<[add(mul([[D2]], -1), [[D1]])], si32>
// CHECK-NEXT:   mo.output<add(mul([[D2]], -1), [[D1]])> [[V1]] : !mo.tensor<[add(mul([[D2]], -1), [[D1]])], si32>
// CHECK-NEXT: }

mo.graph @squeeze_shape_dynamic_shape_with_reshape(%input_tensor: !mo.tensor<[?, ?, ?], si32>, %indices: !mo.tensor<[?], si32>) -> !mo.tensor<?, si32> {
  %shape_tensor = mo.shape_of(%input_tensor) : (!mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[3], si32>
  %result_shape = mo.squeeze_shape(%shape_tensor, %indices) : (!mo.tensor<[3], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32>
  %res = mo.reshape(%input_tensor, %result_shape) : (!mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<?, si32>
  mo.output %res : !mo.tensor<?, si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @squeeze_shape_dynamic_shape_with_reshape<[[D8:.*]], [[D9:.*]], [[D10:.*]], [[D11:.*]] -> [[Sh12:.*]]: !mosh.ape>([[VARG0:%.*]]: !mo.tensor<[[[D8]], [[D9]], [[D10]]], si32>, [[VARG1:%.*]]: !mo.tensor<[[[D11]]], si32>) -> !mo.tensor<Sh12, si32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = kgen.param.constant: !mosh.ape = <[[[D8]], [[D9]], [[D10]]]>
// CHECK-NEXT:   [[V1:%.*]] = mo.shape.to_tensor([[V0]]) -> !mo.tensor<[3], si32>
// CHECK-NEXT:   [[V2:%.*]] = mo.squeeze_shape([[V1]], [[VARG1]]) : (!mo.tensor<[3], si32>, !mo.tensor<[[[D11]]], si32>) -> !mo.tensor<[param_expr_29], si32>
// CHECK-NEXT:   kgen.param.declare param_expr_29 = <add(mul([[D11]], -1), 3)>
// CHECK-NEXT:   [[V3:%.*]] = mo.reshape<() -> [[Sh5:.*]]: !mosh.ape>([[VARG0]], [[V2]]) : (!mo.tensor<[[[D8]], [[D9]], [[D10]]], si32>, !mo.tensor<[param_expr_29], si32>) -> !mo.tensor<[[Sh5]], si32>
// CHECK-NEXT:   mo.output<:!mosh.ape [[Sh5]]> [[V3]] : !mo.tensor<[[Sh5]], si32>
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
// CHECK-SAME: @slice_entire_dynamic_axis<[[D6:.*]]>([[VARG0:%.*]]: !mo.tensor<[10, [[D6]], 20], f32>) -> !mo.tensor<[10, [[D6]], 20], f32> no_inline {
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
// CHECK-SAME: @slice_out_a_third_of_constant_axis<[[D7:.*]]>([[VARG0:%.*]]: !mo.tensor<[50, [[D7]], 2304], f32>) -> !mo.tensor<[50, [[D7]], 768], f32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = kgen.param.constant: !mosh.ape = <[50, [[D7]], 1536]>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant {value = #M.dense_array<0, 0, 768> : tensor<3xsi64>} : !mo.tensor<[3], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
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
// CHECK-SAME: @nested_broadcast_shapes<[[D29:.*]], [[D30:.*]], [[D31:.*]], [[D32:.*]] -> [[D33:.*]], [[D34:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D29]], [[D30]]], bool>, [[VARG1:%.*]]: !mo.tensor<[[[D31]], [[D32]]], si32>) -> !mo.tensor<[[[D33]], [[D34]]], si32> no_inline {
// CHECK-NEXT:   [[V0:%.*]] = mo.cast([[VARG0]]) : (!mo.tensor<[[[D29]], [[D30]]], bool>) -> !mo.tensor<[[[D29]], [[D30]]], si32>
// CHECK-NEXT:   kgen.param.declare dims_assert43: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions are not broadcastable" : !kgen.string}, or(eq([[D29]], [[D31]]), eq([[D29]], 1), eq([[D31]], 1)))>
// CHECK-NEXT:   kgen.param.declare dims_assert48: !mosh.chain = <apply(:(i1) -> !mosh.chain "mosh.assert"{message = "dimensions are not broadcastable" : !kgen.string}, or(eq([[D30]], [[D32]]), eq([[D30]], 1), eq([[D32]], 1)))>
// CHECK-NEXT:   [[V1:%.*]] = mo.static.broadcast_to([[VARG0]]) : !mo.tensor<[[[D29]], [[D30]]], bool> -> !mo.tensor<[param_expr_49, param_expr_50], bool>
// CHECK-NEXT:   [[V2:%.*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[[[D29]], [[D30]]], si32> -> !mo.tensor<[param_expr_49, param_expr_50], si32>
// CHECK-NEXT:   [[V3:%.*]] = mo.static.broadcast_to([[VARG1]]) : !mo.tensor<[[[D31]], [[D32]]], si32> -> !mo.tensor<[param_expr_49, param_expr_50], si32>
// CHECK-NEXT:   kgen.param.declare param_expr_49 = <cond(eq([[D29]], 1), [[D31]], [[D29]])>
// CHECK-NEXT:   kgen.param.declare param_expr_50 = <cond(eq([[D30]], 1), [[D32]], [[D30]])>
// CHECK-NEXT:   [[V4:%.*]] = mo.select([[V1]], [[V2]], [[V3]]) : (!mo.tensor<[param_expr_49, param_expr_50], bool>, !mo.tensor<[param_expr_49, param_expr_50], si32>, !mo.tensor<[param_expr_49, param_expr_50], si32>) -> !mo.tensor<[param_expr_49, param_expr_50], si32>
// CHECK-NEXT:   [[V5:%.*]] = mo.rebind([[V4]]) : !mo.tensor<[param_expr_49, param_expr_50], si32> -> !mo.tensor<[cond(eq([[D29]], 1), [[D31]], [[D29]]), cond(eq([[D30]], 1), [[D32]], [[D30]])], si32>
// CHECK-NEXT:   mo.output<cond(eq([[D29]], 1), [[D31]], [[D29]]), cond(eq([[D30]], 1), [[D32]], [[D30]])> [[V5]] : !mo.tensor<[cond(eq([[D29]], 1), [[D31]], [[D29]]), cond(eq([[D30]], 1), [[D32]], [[D30]])], si32>
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
// CHECK-SAME: @onnx_select_neg_ones_bcast<[[D9:.*]]>([[VARG0:.*]]: !mo.tensor<[1, 32, [[D9]], 10], si64>) -> !mo.tensor<[1, 32, [[D9]], 10], si64> no_inline {
// CHECK-NEXT:   mo.output [[VARG0]] : !mo.tensor<[1, 32, [[D9]], 10], si64>
// CHECK-NEXT: }
