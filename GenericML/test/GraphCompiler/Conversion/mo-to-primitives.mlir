// RUN: mo-opt %s --mo-to-mogg --mo-to-primitives | FileCheck %s

// CHECK-LABEL:  mgp.model @constants_with_multiple_returns : mof
// CHECK:  } execute (
// CHECK-SAME: %[[ARG0:.*]]: !mef.chain, %[[ARG1:.*]]: !mgp.context, %{{.*}}: !mgp.opaque<"weights_registry">
// CHECK-SAME: %[[ARG2:[^:]*]]: !mgp.tensor<2xf32>, %[[ARG3:.*]]: !mgp.tensor<5x1xf32>
// CHECK-SAME: ) -> (!mef.chain, !mgp.tensor<2xf32>, !mgp.tensor<3xf32>, !mgp.tensor<3xf32>,
// CHECK-SAME: !mgp.tensor<5x1xf32>, !mgp.tensor<5x1xf32>, !mgp.tensor<4xsi8>) {
// CHECK-DAG:    %[[VAL0:.*]] = mgp.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040", align {{.*}}> : !M.array<12xui8>
// CHECK-DAG:    %[[VAL1:.*]] = mgp.tensor_spec.create  : !mgp.tensor_spec<3xf32>
// CHECK-DAG:    %[[VAL3:.*]] = mgp.tensor.create %[[ARG0]], %[[VAL0]], %[[VAL1]] : !mgp.tensor<3xf32>
// CHECK-DAG:    %[[VAL4:.*]] = mgp.buffer.constant dense_resource<weight_buffer> : !M.aligned_bytes<4, align 64>
// CHECK-DAG:    %[[VAL5:.*]] = mgp.tensor_spec.create  : !mgp.tensor_spec<4xsi8>
// CHECK:    %[[VAL7:.*]] = mgp.tensor.create %[[ARG0]], %[[VAL4]], %[[VAL5]] : !mgp.tensor<4xsi8>
// CHECK:    mef.output %[[ARG0]], %[[ARG2]], %[[VAL3]], %[[VAL3]], %[[ARG3]], %[[ARG3]], %[[VAL7]] : !mef.chain, !mgp.tensor<2xf32>, !mgp.tensor<3xf32>, !mgp.tensor<3xf32>, !mgp.tensor<5x1xf32>, !mgp.tensor<5x1xf32>, !mgp.tensor<4xsi8>
// CHECK:  }

mo.graph @constants_with_multiple_returns(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[5, 1], f32>) -> (
    !mo.tensor<[2], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[4], si8>
) {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>} : !mo.tensor<[3], f32>
  %1 = mo.constant {value = dense_resource<weight_buffer> : tensor<4xsi8>} : !mo.tensor<[4], si8>
  mo.output %arg0, %0, %0, %arg1, %arg1, %1 : !mo.tensor<[2], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[4], si8>
}

// CHECK-LABEL: mgp.model @foo : mof
// CHECK: } execute (
// CHECK-SAME: %[[ARG0:.*]]: !mef.chain, %[[ARG1:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:[^:]*]]: !mgp.tensor<1x?xf32>, %[[ARG3:.*]]: !mgp.tensor<?xf32>) -> !mef.chain {
// CHECK:   %[[VAL0:.*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <1x?xf32>
// CHECK:   %[[VAL1:.*]] = mgp.tensor_spec.get_dim[1] %[[VAL0]] : <1x?xf32>
// CHECK:   %[[VAL2:.*]] = mgp.tensor.extract.tensor_spec %[[ARG3]] : <?xf32>
// CHECK:   %[[VAL3:.*]] = mgp.tensor_spec.get_dim[0] %[[VAL2]] : <?xf32>
// CHECK:   mef.output %[[ARG0:.*]] : !mef.chain
mo.graph @foo<N>(%arg0: !mo.tensor<[1, N], f32>, %arg1: !mo.tensor<[N], f32>) {
  mo.output
}

// CHECK-LABEL: mgp.model @scalar_arg : mof
// CHECK: } execute (
// CHECK-SAME: %[[ARG0:.*]]: !mef.chain, %[[ARG1:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:[^:]*]]: !mgp.tensor<f32>) -> (!mef.chain, !mgp.tensor<f32>) {
// CHECK:   mef.output %[[ARG0]], %[[ARG2]] : !mef.chain, !mgp.tensor<f32>
mo.graph @scalar_arg(%arg0: !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>) {
  mo.output %arg0 : !mo.tensor<[], f32>
}

{-#
  dialect_resources: {
    builtin: {
      weight_buffer: "0x4000000001234567"
    }
  }
#-}

// See `isAsync` is properly forwarded from mogg kernel to
// CHECK-LABEL: mgp.model @test_lowered_async
// CHECK-COUNT-1: mogg.generic.compiled.region
// CHECK-COUNT-1: {canBeFusedInto = true, device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"ioii">, isAsync = true, isKGEN = true, summary = "mo.add_test"}
kgen.extern.generator export @"stub"<type: dtype, rank>(!kgen.pointer<struct<(pointer<none>, array<rank, index>, array<rank, index>) memoryOnly>> read_mem) -> !kgen.struct<(pointer<none>, array<rank, index>, array<rank, index>, i1)>mo.graph @test_lowered_async(%arg0: !mo.tensor<[], si64>, %arg1: !mo.tensor<[], si64>) -> !mo.tensor<[], si64> attributes {device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "x86_64-unknown-linux-gnu", arch = "icelake-server", features = "+adx,+aes,+avx,+avx2,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+crc32,+cx16,+cx8,+evex512,+f16c,+fma,+fsgsbase,+fxsr,+gfni,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+pconfig,+pku,+popcnt,+prfchw,+rdpid,+rdrnd,+rdseed,+sahf,+sgx,+sha,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+vaes,+vpclmulqdq,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves">>]>} {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[], si64>, %arg3: !mo.tensor<[], si64>)  -> !mo.tensor<[], si64>{
    %1 = mogg.output_placeholders : !mo.tensor<[], si64>
    %2 = "mogg.to_kgen"(%1) : (!mo.tensor<[], si64>) -> !kgen.pointer<struct<(pointer<none>, array<1, index>, array<1, index>) memoryOnly>>
    %3 = kgen.call @"stub"<:dtype si64, 1>(%2) : (!kgen.pointer<struct<(pointer<none>, array<1, index>, array<1, index>) memoryOnly>> read_mem) -> !kgen.struct<(pointer<none>, array<1, index>, array<1, index>, i1)>
    mogg.output %3 : !kgen.struct<(pointer<none>, array<1, index>, array<1, index>, i1)>
  } {device = #M.device_ref<"cpu", 0>, fusionBodyLambda = "elemwise_compute_stub", summary = "mo.add_test", isAsync=true}
  mo.output %0 : !mo.tensor<[], si64>
}


// CHECK-LABEL: mgp.model @test_buffer_dependencies

// CHECK: execute (%[[CH0:.*]]: !mef.chain
// CHECK: %[[CH1:.*]] = mogg.generic.compiled.region (%[[CH0]],
// CHECK: %[[MERGE1:.*]] = grt.chain.create (%[[CH1]], %[[CH0]]
// CHECK: %[[CH2:.*]] = mogg.generic.compiled.region (%[[MERGE1]]
// CHECK: %[[MERGE2:.*]] = grt.chain.create (%[[CH1]], %[[CH2]], %[[CH0]]
// CHECK: %[[CH3:.*]] = mogg.generic.compiled.region (%[[MERGE2]],

mo.graph @test_buffer_dependencies(%arg0: !mo.tensor<[], si64>) -> !mo.tensor<[], si64> attributes {device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "x86_64-unknown-linux-gnu", arch = "icelake-server", features = "+adx,+aes,+avx,+avx2,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+crc32,+cx16,+cx8,+evex512,+f16c,+fma,+fsgsbase,+fxsr,+gfni,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+pconfig,+pku,+popcnt,+prfchw,+rdpid,+rdrnd,+rdseed,+sahf,+sgx,+sha,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+vaes,+vpclmulqdq,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves">>]>} {
    %0 = mo.relu(%arg0): !mo.tensor<[], si64>
    %1 = mo.relu(%0): !mo.tensor<[], si64>
    %2 = mo.add(%0, %1): !mo.tensor<[], si64>
    mo.output %2 : !mo.tensor<[], si64>
}


// CHECK-LABEL: mgp.model @test_lowered_multiple_outputs_different_ordering
// Check associations between mgp.buffer and tensors are perfect, otherwise
// we would have a verifier telling us we have failed.
mo.graph @test_lowered_multiple_outputs_different_ordering(%arg0: !mo.tensor<[2], si64>, %arg1: !mo.tensor<[2], si64>) -> (!mo.tensor<[2], si64>, !mo.tensor<[3], si64>) attributes {device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "x86_64-unknown-linux-gnu", arch = "icelake-server", features = "+adx,+aes,+avx,+avx2,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+crc32,+cx16,+cx8,+evex512,+f16c,+fma,+fsgsbase,+fxsr,+gfni,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+pconfig,+pku,+popcnt,+prfchw,+rdpid,+rdrnd,+rdseed,+sahf,+sgx,+sha,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+vaes,+vpclmulqdq,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves">>]>} {
  %0:2 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[2], si64>, %arg3: !mo.tensor<[2], si64>)  -> (!mo.tensor<[2], si64>, !mo.tensor<[3], si64>){
    // Note we declare the order different than what we return
    %1:2 = mogg.output_placeholders : !mo.tensor<[3], si64>, !mo.tensor<[2], si64>
    mogg.output %1#1, %1#0 : !mo.tensor<[2], si64>, !mo.tensor<[3], si64>
  } {device = #M.device_ref<"cpu", 0>, fusionBodyLambda = "elemwise_compute_stub", summary = "mo.add_test", isAsync=true}
  mo.output %0#0, %0#1 : !mo.tensor<[2], si64>, !mo.tensor<[3], si64>
}
