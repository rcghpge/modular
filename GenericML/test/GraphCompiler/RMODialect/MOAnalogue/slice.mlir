// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @slice_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %start: !mo.tensor<Sh1, si64>, %stop: !mo.tensor<Sh2, si32>, %step: !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32> {
  %0 = rmo.mo.slice<() -> Sh4: !mosh.ape>(%arg0, %start, %stop, %step) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si32>, !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32>
  mo.output %0 : !mo.tensor<Sh4, f32>
}

// -----

mo.graph @slice_bad_start<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %start: !mo.tensor<Sh1, f32>, %stop: !mo.tensor<Sh2, si32>, %step: !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32> {
  // expected-error @+1 {{'rmo.mo.slice' op operand #1 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<Sh1, f32>'}}
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si32>, !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32>
  mo.output %0 : !mo.tensor<Sh4, f32>
}

// -----

mo.graph @slice_bad_step<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %start: !mo.tensor<Sh1, si64>, %stop: !mo.tensor<Sh2, f32>, %step: !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32> {
  // expected-error @+1 {{'rmo.mo.slice' op operand #2 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<Sh2, f32>'}}
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, f32>, !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32>
  mo.output %0 : !mo.tensor<Sh4, f32>
}

// -----

mo.graph @slice_bad_stop<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %start: !mo.tensor<Sh1, si64>, %stop: !mo.tensor<Sh2, si32>, %step: !mo.tensor<Sh3, f32>) -> !mo.tensor<Sh4, f32> {
  // expected-error @+1 {{'rmo.mo.slice' op operand #3 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<Sh3, f32>'}}
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si32>, !mo.tensor<Sh3, f32>) -> !mo.tensor<Sh4, f32>
  mo.output %0 : !mo.tensor<Sh4, f32>
}

// -----

mo.graph @slice_bad_result<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %start: !mo.tensor<Sh1, si64>, %stop: !mo.tensor<Sh2, si32>, %step: !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f16> {
  // expected-error @+1 {{'rmo.mo.slice' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si32>, !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f16>
  mo.output %0 : !mo.tensor<Sh4, f16>
}
