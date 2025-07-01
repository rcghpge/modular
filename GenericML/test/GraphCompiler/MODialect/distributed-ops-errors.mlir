// RUN: mo-opt %s --split-input-file --verify-diagnostics

// Test allgather with incorrect number of outputs (should be num_inputs * num_inputs)
mo.graph @allgather_wrong_output_count(
    %arg0: !mo.tensor<[2, 3], f32, gpu:0>,
    %arg1: !mo.tensor<[2, 3], f32, gpu:1>,
    %sig0: !mo.buffer<[537919488], ui8, gpu:0>,
    %sig1: !mo.buffer<[537919488], ui8, gpu:1>)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>) {
  %chain_in = mo.chain.create()
  // expected-error@+1 {{number of outputs should be num_inputs * num_inputs for all-gather. Expected 4 outputs but got 2 (num_inputs=2)}}
  %0:3 = "mo.distributed.allgather"(%arg0, %arg1, %sig0, %sig1, %chain_in) :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>,
     !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>
}

// -----

// Test allgather with 3 inputs but only 6 outputs (should be 9)
mo.graph @allgather_three_inputs_wrong_outputs(
    %arg0: !mo.tensor<[4], f32, gpu:0>,
    %arg1: !mo.tensor<[4], f32, gpu:1>,
    %arg2: !mo.tensor<[4], f32, gpu:2>,
    %sig0: !mo.buffer<[537919488], ui8, gpu:0>,
    %sig1: !mo.buffer<[537919488], ui8, gpu:1>,
    %sig2: !mo.buffer<[537919488], ui8, gpu:2>)
    -> (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
        !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:1>) {
  %chain_in = mo.chain.create()
  // expected-error@+1 {{number of outputs should be num_inputs * num_inputs for all-gather. Expected 9 outputs but got 6 (num_inputs=3)}}
  %0:7 = "mo.distributed.allgather"(%arg0, %arg1, %arg2, %sig0, %sig1, %sig2, %chain_in) :
    (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:2>,
     !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.buffer<[537919488], ui8, gpu:2>, !mo.chain)
    -> (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
        !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 :
    !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
    !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:1>
}

// -----

// Test valid allgather with 1 input and 1 output (1 * 1 = 1)
mo.graph @allgather_single_device(
    %arg0: !mo.tensor<[10, 20], f32, gpu:0>,
    %sig0: !mo.buffer<[537919488], ui8, gpu:0>)
    -> (!mo.tensor<[10, 20], f32, gpu:0>) {
  %chain_in = mo.chain.create()
  %0:2 = "mo.distributed.allgather"(%arg0, %sig0, %chain_in) :
    (!mo.tensor<[10, 20], f32, gpu:0>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.chain)
    -> (!mo.tensor<[10, 20], f32, gpu:0>, !mo.chain)
  mo.output %0#0 : !mo.tensor<[10, 20], f32, gpu:0>
}

// -----

// Test valid allgather with 2 inputs and 4 outputs (2 * 2 = 4)
mo.graph @allgather_two_devices(
    %arg0: !mo.tensor<[5, 10], f32, gpu:0>,
    %arg1: !mo.tensor<[5, 10], f32, gpu:1>,
    %sig0: !mo.buffer<[537919488], ui8, gpu:0>,
    %sig1: !mo.buffer<[537919488], ui8, gpu:1>)
    -> (!mo.tensor<[5, 10], f32, gpu:0>, !mo.tensor<[5, 10], f32, gpu:0>,
        !mo.tensor<[5, 10], f32, gpu:1>, !mo.tensor<[5, 10], f32, gpu:1>) {
  %chain_in = mo.chain.create()
  %0:5 = "mo.distributed.allgather"(%arg0, %arg1, %sig0, %sig1, %chain_in) :
    (!mo.tensor<[5, 10], f32, gpu:0>, !mo.tensor<[5, 10], f32, gpu:1>,
     !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[5, 10], f32, gpu:0>, !mo.tensor<[5, 10], f32, gpu:0>,
        !mo.tensor<[5, 10], f32, gpu:1>, !mo.tensor<[5, 10], f32, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1, %0#2, %0#3 :
    !mo.tensor<[5, 10], f32, gpu:0>, !mo.tensor<[5, 10], f32, gpu:0>,
    !mo.tensor<[5, 10], f32, gpu:1>, !mo.tensor<[5, 10], f32, gpu:1>
}
