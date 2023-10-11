mo.graph @model(%arg0: !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) no_inline attributes {argument_names = ["input"], result_names = ["output"]} {
  %0 = mo.sqrt(%arg0): !mo.tensor<[5], f32>
  mo.output %0 : !mo.tensor<[5], f32>
}
