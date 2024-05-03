// RUN: mo-opt --symbolic-optimize="symbolic-ffast-math=true symbolic-no-undefined-math=true" %s | FileCheck %s

// The first few tests have different data types and negative checks for constants.

// x + 0 -> x
mo.graph @add_with_cst_zero(%input: !mo.tensor<[1, 2, 3], f64>) -> !mo.tensor<[1, 2, 3], f64> {
  %zero = mo.constant {value = #M.dense_array<0.0, 0.0, 0.0, 0.0, 0.0, 0.0> : tensor<6xf64>} : !mo.tensor<[1, 2, 3], f64>
  %0 = mo.add(%input, %zero) : !mo.tensor<[1, 2, 3], f64>
  mo.output %0 : !mo.tensor<[1, 2, 3], f64>
}
// CHECK-LABEL: @add_with_cst_zero
// CHECK-NEXT: mo.output

// x + bcast(0) -> x
mo.graph @add_with_bcast_zero(%input: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %zero = mo.constant {value = #M.dense_array<0.0> : tensor<1xf32>} : !mo.tensor<[], f32>
  %bcast_zero = mo.static.broadcast_to(%zero) : !mo.tensor<[], f32> -> !mo.tensor<[1, 2, 3], f32>
  %0 = mo.add(%input, %bcast_zero) : !mo.tensor<[1, 2, 3], f32>
  mo.output %0 : !mo.tensor<[1, 2, 3], f32>
}
// CHECK-LABEL: @add_with_bcast_zero
// CHECK-NEXT: mo.output

// 0 + x -> x
mo.graph @add_with_cst_zero_reversed(%input: !mo.tensor<[1, 2, 3], si32>) -> !mo.tensor<[1, 2, 3], si32> {
  %zero = mo.constant {value = #M.dense_array<0, 0, 0, 0, 0, 0> : tensor<6xsi32>} : !mo.tensor<[1, 2, 3], si32>
  %0 = mo.add(%zero, %input) : !mo.tensor<[1, 2, 3], si32>
  mo.output %0 : !mo.tensor<[1, 2, 3], si32>
}
// CHECK-LABEL: @add_with_cst_zero_reversed
// CHECK-NEXT: mo.output

// bcast(0) + x -> x
mo.graph @add_with_bcast_zero_reversed(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %zero = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %bcast_zero = mo.static.broadcast_to(%zero) : !mo.tensor<[], si64> -> !mo.tensor<[1, 2, 3], si64>
  %0 = mo.add(%bcast_zero, %input) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @add_with_bcast_zero_reversed
// CHECK-NEXT: mo.output

mo.graph @add_with_non_zero(%input: !mo.tensor<[1, 2, 3], f64>) -> !mo.tensor<[1, 2, 3], f64> {
  %zero = mo.constant {value = #M.dense_array<0.0, 0.0, 0.0, 0.0, 0.0, 1.0> : tensor<6xf64>} : !mo.tensor<[1, 2, 3], f64>
  %0 = mo.add(%input, %zero) : !mo.tensor<[1, 2, 3], f64>
  mo.output %0 : !mo.tensor<[1, 2, 3], f64>
}
// CHECK-LABEL: @add_with_non_zero
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[ADD:.*]] = mo.add
// CHECK-NEXT: mo.output %[[ADD]]

// x - 0 -> x
mo.graph @sub_with_cst_zero(%input: !mo.tensor<[1, 2, 3], f64>) -> !mo.tensor<[1, 2, 3], f64> {
  %zero = mo.constant {value = #M.dense_array<0.0, 0.0, 0.0, 0.0, 0.0, 0.0> : tensor<6xf64>} : !mo.tensor<[1, 2, 3], f64>
  %0 = mo.sub(%input, %zero) : !mo.tensor<[1, 2, 3], f64>
  mo.output %0 : !mo.tensor<[1, 2, 3], f64>
}
// CHECK-LABEL: @sub_with_cst_zero
// CHECK-NEXT: mo.output

// x - x -> bcast(0)
mo.graph @sub_with_same(%input: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
  %0 = mo.sub(%input, %input) : !mo.tensor<[1, 2, 3], f32>
  mo.output %0 : !mo.tensor<[1, 2, 3], f32>
}
// CHECK-LABEL: @sub_with_same
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x * 1 -> x
mo.graph @mul_with_cst_one(%input: !mo.tensor<[1, 2, 3], f64>) -> !mo.tensor<[1, 2, 3], f64> {
  %one = mo.constant {value = #M.dense_array<1.0, 1.0, 1.0, 1.0, 1.0, 1.0> : tensor<6xf64>} : !mo.tensor<[1, 2, 3], f64>
  %0 = mo.mul(%input, %one) : !mo.tensor<[1, 2, 3], f64>
  mo.output %0 : !mo.tensor<[1, 2, 3], f64>
}
// CHECK-LABEL: @mul_with_cst_one
// CHECK-NEXT: mo.output

// x * 0 -> bcast(0)
mo.graph @mul_with_cst_zero(%input: !mo.tensor<[1, 2, 3], f64>) -> !mo.tensor<[1, 2, 3], f64> {
  %zero = mo.constant {value = #M.dense_array<0.0, 0.0, 0.0, 0.0, 0.0, 0.0> : tensor<6xf64>} : !mo.tensor<[1, 2, 3], f64>
  %0 = mo.mul(%input, %zero) : !mo.tensor<[1, 2, 3], f64>
  mo.output %0 : !mo.tensor<[1, 2, 3], f64>
}
// CHECK-LABEL: @mul_with_cst_zero
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// 0 / x -> bcast(0)
mo.graph @zero_div(%input: !mo.tensor<[1, 2, 3], f64>) -> !mo.tensor<[1, 2, 3], f64> {
  %zero = mo.constant {value = #M.dense_array<0.0, 0.0, 0.0, 0.0, 0.0, 0.0> : tensor<6xf64>} : !mo.tensor<[1, 2, 3], f64>
  %0 = mo.div(%zero, %input) : !mo.tensor<[1, 2, 3], f64>
  mo.output %0 : !mo.tensor<[1, 2, 3], f64>
}
// CHECK-LABEL: @zero_div
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x / 1 -> x
mo.graph @div_by_one(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %one = mo.constant {value = #M.dense_array<1, 1, 1, 1, 1, 1> : tensor<6xsi64>} : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.div(%input, %one) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @div_by_one
// CHECK-NEXT: mo.output

// x ** 1 -> x
mo.graph @pow_by_one(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %one = mo.constant {value = #M.dense_array<1, 1, 1, 1, 1, 1> : tensor<6xsi64>} : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.pow(%input : !mo.tensor<[1, 2, 3], si64>, %one : !mo.tensor<[1, 2, 3], si64>) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @pow_by_one
// CHECK-NEXT: mo.output

// x ** 0 -> bcast(1)
mo.graph @pow_by_zero(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %zero = mo.constant {value = #M.dense_array<0, 0, 0, 0, 0, 0> : tensor<6xsi64>} : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.pow(%input : !mo.tensor<[1, 2, 3], si64>, %zero : !mo.tensor<[1, 2, 3], si64>) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @pow_by_zero
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x ** 0 -> bcast(1)
mo.graph @pow_of_zero(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %zero = mo.constant {value = #M.dense_array<0, 0, 0, 0, 0, 0> : tensor<6xsi64>} : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.pow(%zero : !mo.tensor<[1, 2, 3], si64>, %input : !mo.tensor<[1, 2, 3], si64>) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @pow_of_zero
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x ** 0 -> bcast(1)
mo.graph @pow_of_one(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %one = mo.constant {value = #M.dense_array<1, 1, 1, 1, 1, 1> : tensor<6xsi64>} : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.pow(%one : !mo.tensor<[1, 2, 3], si64>, %input : !mo.tensor<[1, 2, 3], si64>) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @pow_of_one
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// max(x, x) -> x
mo.graph @max_same_operand(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %0 = mo.max(%input, %input) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @max_same_operand
// CHECK-NEXT: mo.output

// min(x, x) -> x
mo.graph @min_same_operand(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %0 = mo.min(%input, %input) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @min_same_operand
// CHECK-NEXT: mo.output

// min(x, max(x, y)) -> x
mo.graph @min_of_max(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %cst = mo.constant {value = #M.dense_array<1, 1, 1, 1, 1, 1> : tensor<6xsi64>} : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.max(%input, %cst) : !mo.tensor<[1, 2, 3], si64>
  %1 = mo.min(%0, %input) : !mo.tensor<[1, 2, 3], si64>
  mo.output %1 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @min_of_max
// CHECK-NEXT: mo.output

// max(x, min(x, y)) -> x
mo.graph @max_of_min(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %cst = mo.constant {value = #M.dense_array<1, 1, 1, 1, 1, 1> : tensor<6xsi64>} : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.min(%input, %cst) : !mo.tensor<[1, 2, 3], si64>
  %1 = mo.max(%0, %input) : !mo.tensor<[1, 2, 3], si64>
  mo.output %1 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @max_of_min
// CHECK-NEXT: mo.output

// mod(x, x) -> x
mo.graph @mod_same_operand(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %0 = mo.mod(%input, %input) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @mod_same_operand
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// mod(x, 1) -> bcast(0)
mo.graph @mod_by_1(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], si64> {
  %one = mo.constant {value = #M.dense_array<1, 1, 1, 1, 1, 1> : tensor<6xsi64>} : !mo.tensor<[1, 2, 3], si64>
  %0 = mo.mod(%input, %one) : !mo.tensor<[1, 2, 3], si64>
  mo.output %0 : !mo.tensor<[1, 2, 3], si64>
}
// CHECK-LABEL: @mod_by_1
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x >= x -> bcast(true)
mo.graph @ge_same_operand(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool> {
  %0 = mo.greater_equal(%input, %input) : (!mo.tensor<[1, 2, 3], si64>, !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool>
  mo.output %0 : !mo.tensor<[1, 2, 3], bool>
}
// CHECK-LABEL: @ge_same_operand
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x > x -> bcast(false)
mo.graph @gt_same_operand(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool> {
  %0 = mo.greater(%input, %input) : (!mo.tensor<[1, 2, 3], si64>, !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool>
  mo.output %0 : !mo.tensor<[1, 2, 3], bool>
}
// CHECK-LABEL: @gt_same_operand
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x != x -> bcast(false)
mo.graph @ne_same_operand(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool> {
  %0 = mo.not_equal(%input, %input) : (!mo.tensor<[1, 2, 3], si64>, !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool>
  mo.output %0 : !mo.tensor<[1, 2, 3], bool>
}
// CHECK-LABEL: @ne_same_operand
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x == x -> bcast(false)
mo.graph @eq_same_operand(%input: !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool> {
  %0 = mo.equal(%input, %input) : (!mo.tensor<[1, 2, 3], si64>, !mo.tensor<[1, 2, 3], si64>) -> !mo.tensor<[1, 2, 3], bool>
  mo.output %0 : !mo.tensor<[1, 2, 3], bool>
}
// CHECK-LABEL: @eq_same_operand
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]

// x && x -> bcast(true)
mo.graph @and_same_operand(%input: !mo.tensor<[1, 2, 3], bool>) -> !mo.tensor<[1, 2, 3], bool> {
  %0 = mo.and(%input, %input) : !mo.tensor<[1, 2, 3], bool>
  mo.output %0 : !mo.tensor<[1, 2, 3], bool>
}
// CHECK-LABEL: @and_same_operand
// CHECK-NEXT: mo.output

// x || x -> bcast(true)
mo.graph @or_same_operand(%input: !mo.tensor<[1, 2, 3], bool>) -> !mo.tensor<[1, 2, 3], bool> {
  %0 = mo.or(%input, %input) : !mo.tensor<[1, 2, 3], bool>
  mo.output %0 : !mo.tensor<[1, 2, 3], bool>
}
// CHECK-LABEL: @or_same_operand
// CHECK-NEXT: mo.output

// x ^ x -> bcast(true)
mo.graph @xor_same_operand(%input: !mo.tensor<[1, 2, 3], bool>) -> !mo.tensor<[1, 2, 3], bool> {
  %0 = mo.xor(%input, %input) : !mo.tensor<[1, 2, 3], bool>
  mo.output %0 : !mo.tensor<[1, 2, 3], bool>
}
// CHECK-LABEL: @xor_same_operand
// CHECK-NEXT: %[[CST:.*]] = mo.constant
// CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[CST]])
// CHECK-NEXT: mo.output %[[BCAST]]
