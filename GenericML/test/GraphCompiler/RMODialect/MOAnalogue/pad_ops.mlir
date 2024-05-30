// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @pad_reflect_ok(%arg0: !mo.tensor<?, f32>, %pads: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.pad.reflect(%arg0, %pads) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @pad_reflect_params_ok(%arg0: !mo.tensor<[?, ?, ?], f32>, %pads: !mo.tensor<[?], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.pad.reflect(%arg0, %pads) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @pad_reflect_static_ok(%arg0: !mo.tensor<[2, 3, 4], f32>, %pads: !mo.tensor<[6], si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.pad.reflect(%arg0, %pads) : (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[6], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @pad_repeat_ok(%arg0: !mo.tensor<?, f32>, %pads: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.pad.repeat(%arg0, %pads) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @pad_constant_ok(%arg0: !mo.tensor<?, f32>, %pads: !mo.tensor<?, si64>, %cst: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.pad.constant(%arg0, %pads, %cst) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @pad_reflect_bad_paddings(%arg0: !mo.tensor<?, f32>, %pads: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.pad.reflect' op operand #1 must be unranked or shape [N] and signed 64 or signed 32 integer elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.pad.reflect(%arg0, %pads) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @pad_reflect_bad_result(%arg0: !mo.tensor<?, f32>, %pads: !mo.tensor<?, si64>) -> !mo.tensor<?, f16> {
  // expected-error @+1 {{'rmo.mo.pad.reflect' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.pad.reflect(%arg0, %pads) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f16>
  mo.output %0 : !mo.tensor<?, f16>
}
