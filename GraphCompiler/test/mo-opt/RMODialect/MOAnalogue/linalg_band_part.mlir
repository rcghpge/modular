// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @band_part_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %num_lower: !mo.tensor<Sh1, si64>, %num_upper: !mo.tensor<Sh2, si64>, %exclude: !mo.tensor<Sh3, bool>) -> !mo.tensor<Sh4, f32> {
  %0 = rmo.mo.linalg.band_part<() -> Sh4: !mosh.ape>(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, bool>) -> !mo.tensor<Sh4, f32>
  mo.output %0 : !mo.tensor<Sh4, f32>
}

// -----

mo.graph @band_part_bad_num_lower<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %num_lower: !mo.tensor<Sh1, f32>, %num_upper: !mo.tensor<Sh2, si64>, %exclude: !mo.tensor<Sh3, bool>) -> !mo.tensor<Sh4, f32> {
  // expected-error @+1 {{'rmo.mo.linalg.band_part' op operand #1 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<Sh1, f32>'}}
  %0 = rmo.mo.linalg.band_part(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, bool>) -> !mo.tensor<Sh4, f32>
  mo.output %0 : !mo.tensor<Sh4, f32>
}

// -----

mo.graph @band_part_bad_num_upper<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %num_lower: !mo.tensor<Sh1, si64>, %num_upper: !mo.tensor<Sh2, f32>, %exclude: !mo.tensor<Sh3, bool>) -> !mo.tensor<Sh4, f32> {
  // expected-error @+1 {{'rmo.mo.linalg.band_part' op operand #2 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<Sh2, f32>'}}
  %0 = rmo.mo.linalg.band_part(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, f32>, !mo.tensor<Sh3, bool>) -> !mo.tensor<Sh4, f32>
  mo.output %0 : !mo.tensor<Sh4, f32>
}

// -----

mo.graph @band_part_bad_exclude<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %num_lower: !mo.tensor<Sh1, si64>, %num_upper: !mo.tensor<Sh2, si64>, %exclude: !mo.tensor<Sh3, f32>) -> !mo.tensor<Sh4, f32> {
  // expected-error @+1 {{'rmo.mo.linalg.band_part' op operand #3 must be rank 0 or unranked with boolean element type, but got '!mo.tensor<Sh3, f32>'}}
  %0 = rmo.mo.linalg.band_part(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, f32>) -> !mo.tensor<Sh4, f32>
  mo.output %0 : !mo.tensor<Sh4, f32>
}

// -----

mo.graph @band_part_bad_result<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %num_lower: !mo.tensor<Sh1, si64>, %num_upper: !mo.tensor<Sh2, si64>, %exclude: !mo.tensor<Sh3, bool>) -> !mo.tensor<Sh4, f16> {
  // expected-error @+1 {{'rmo.mo.linalg.band_part' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.linalg.band_part(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, bool>) -> !mo.tensor<Sh4, f16>
  mo.output %0 : !mo.tensor<Sh4, f16>
}
