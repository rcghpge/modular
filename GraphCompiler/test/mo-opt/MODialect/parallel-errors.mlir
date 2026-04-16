// RUN: mo-opt %s --split-input-file --verify-diagnostics

// Second element of the input bundle has a different shape than the block arg.
// The first element matches, so this specifically tests the per-element check.
mo.graph @bundle_elem_shape_mismatch(
    %a: !mo.tensor<[3], f32>,
    %b: !mo.tensor<[4], f32>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[4], f32>) {
  %bundle = mo.tensor.bundle(%a, %b) : (!mo.tensor<[3], f32>, !mo.tensor<[4], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[4], f32>]>)
  // expected-error @below {{'mo.parallel' op bundle input #0 element #1 type '!mo.tensor<[4], f32>' does not match block argument type '!mo.tensor<[3], f32>'}}
  %res = mo.parallel (%arg) in (%bundle : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[4], f32>]>)
      -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[4], f32>]>) {
    mo.yield %arg : !mo.tensor<[3], f32>
  }
  %r0, %r1 = mo.tensor.unbundle(%res) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[4], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[4], f32>)
  mo.output %r0, %r1 : !mo.tensor<[3], f32>, !mo.tensor<[4], f32>
}

// -----

// Second element of the input bundle has a different dtype than the block arg.
mo.graph @bundle_elem_dtype_mismatch(
    %a: !mo.tensor<[3], f32>,
    %b: !mo.tensor<[3], f16>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f16>) {
  %bundle = mo.tensor.bundle(%a, %b) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f16>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f16>]>)
  // expected-error @below {{'mo.parallel' op bundle input #0 element #1 type '!mo.tensor<[3], f16>' does not match block argument type '!mo.tensor<[3], f32>'}}
  %res = mo.parallel (%arg) in (%bundle : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f16>]>)
      -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f16>]>) {
    mo.yield %arg : !mo.tensor<[3], f32>
  }
  %r0, %r1 = mo.tensor.unbundle(%res) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f16>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f16>)
  mo.output %r0, %r1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f16>
}

// -----

// Mismatch in the second input bundle (extra inputs), not the first.
// First bundle is homogeneous and fine; second bundle has a shape mismatch
// in element #1.
mo.graph @extra_input_bundle_mismatch(
    %a0: !mo.tensor<[3], f32>,
    %a1: !mo.tensor<[3], f32>,
    %w0: !mo.tensor<[3], f32>,
    %w1: !mo.tensor<[5], f32>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %d_in = mo.tensor.bundle(%a0, %a1) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %d_w = mo.tensor.bundle(%w0, %w1) : (!mo.tensor<[3], f32>, !mo.tensor<[5], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[5], f32>]>)
  // expected-error @below {{'mo.parallel' op bundle input #1 element #1 type '!mo.tensor<[5], f32>' does not match block argument type '!mo.tensor<[3], f32>'}}
  %res = mo.parallel (%arg, %w) in (%d_in : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>, %d_w : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[5], f32>]>)
      -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) {
    %sum = mo.add(%arg, %w) : !mo.tensor<[3], f32>
    mo.yield %sum : !mo.tensor<[3], f32>
  }
  %r0, %r1 = mo.tensor.unbundle(%res) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  mo.output %r0, %r1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// -----

// Buffer type mismatch: second buffer has a different dtype than the block arg.
mo.graph @buffer_type_mismatch(
    %buf0: !mo.buffer<[1], ui8, gpu:0>,
    %buf1: !mo.buffer<[1], ui16, gpu:1>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %a = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>} : !mo.tensor<[3], f32, gpu:0>
  %b = mo.constant {value = #M.dense_array<4.0, 5.0, 6.0> : tensor<3xf32>} : !mo.tensor<[3], f32, gpu:1>
  %bundle = mo.tensor.bundle(%a, %b) : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>) -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>)
  %ch = mo.chain.create()
  // expected-error @below {{'mo.parallel' op buffer #1 type '!mo.buffer<[1], ui16, gpu:1>' does not match block argument type '!mo.buffer<[1], ui8, gpu:0>'}}
  %res, %out_ch = mo.parallel (%arg, %sig) in (%bundle : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>)
      buffers(%buf0 : !mo.buffer<[1], ui8, gpu:0>, %buf1 : !mo.buffer<[1], ui16, gpu:1>)
      chain(%ch) -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>) {
    mo.yield %arg : !mo.tensor<[3], f32, gpu:0>
  }
  %r0, %r1 = mo.tensor.unbundle(%res) : (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>]>) -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  mo.output %r0, %r1 : !mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>
}
