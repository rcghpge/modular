// RUN: mo-opt %s --mo-to-mogg=kernel-library-paths="%kernel_api_test_lib" --mo-to-mogg | FileCheck %s

// CHECK-LABEL: @test_simple_add
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "OutputTensor"{{.*}}(!mo.tensor<[5], f32>)
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "InputTensor"{{.*}}(!mo.tensor<[5], f32>)
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "InputTensor"{{.*}}(!mo.tensor<[5], f32>)
mo.graph @test_simple_add(%0: !mo.tensor<[5], f32>, %1: !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>) {
  %2 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  mo.output %2 : !mo.tensor<[5], f32>
}

// CHECK-LABEL: @with_a_list_type
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "OutputTensor"{{.*}}(!mo.tensor<[30], si32>) -> !kgen.struct<{{.*}}>
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "InputTensor"{{.*}}(!mo.list<!mo.tensor<[10], si32>>) -> !kgen.pointer<{{.*}}>
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "InputTensor"{{.*}}(!mo.tensor<[], si64>) -> !kgen.struct<{{.*}}>
mo.graph @with_a_list_type(%l0: !mo.list<!mo.tensor<[10], si32>>) -> !mo.tensor<[30], si32> {
  %zero = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %res = mo.concat_from_list[%zero: !mo.tensor<[], si64>](%l0) : !mo.list<!mo.tensor<[10], si32>> -> !mo.tensor<[30], si32>
  mo.output %res : !mo.tensor<[30], si32>
}

// CHECK-LABEL: @with_variadic_inputs
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "FusedOutputTensor"{{.*}}(!mo.tensor<[6], si32>) -> !kgen.struct<{{.*}}>
// CHECK: "mogg.to_kgen"{{.*}}{tensorIOType = "InputTensor"{{.*}}(!mo.tensor<[], si32>) -> !kgen.struct<{{.*}}>
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "FusedInputTensor"{{.*}}(!mo.tensor<[1], si32>) -> !kgen.struct<{{.*}}>
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "FusedInputTensor"{{.*}}(!mo.tensor<[2], si32>) -> !kgen.struct<{{.*}}>
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "FusedInputTensor"{{.*}}(!mo.tensor<[3], si32>) -> !kgen.struct<{{.*}}>
mo.graph @with_variadic_inputs(%arg2: !mo.tensor<[2], si32>) -> (!mo.tensor<[6], si32>) {
  %ax = mo.constant {value = #M.dense_array<-1> : tensor<si32>} : !mo.tensor<[], si32>
  %arg1 = mo.constant {value = #M.dense_array<2> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %arg3 = mo.constant {value = #M.dense_array<3, 3, 3> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %out = mo.concat[%ax: !mo.tensor<[], si32>](%arg1, %arg2, %arg3) : (
    !mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>
  ) -> !mo.tensor<[6], si32>
  mo.output %out : !mo.tensor<[6], si32>
}

// CHECK-LABEL: @with_variadic_outputs
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "OutputTensor"{{.*}}(!mo.tensor<[1], si32>) -> !kgen.struct<{{.*}}>
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "OutputTensor"{{.*}}(!mo.tensor<[2], si32>) -> !kgen.struct<{{.*}}>
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "OutputTensor"{{.*}}(!mo.tensor<[3], si32>) -> !kgen.struct<{{.*}}>
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "InputTensor"{{.*}}(!mo.tensor<[6], si32>) -> !kgen.struct<{{.*}}>
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "InputTensor"{{.*}}(!mo.tensor<[3], si64>) -> !kgen.struct<{{.*}}>
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "InputTensor"{{.*}}(!mo.tensor<[], si32>) -> !kgen.struct<{{.*}}>
mo.graph @with_variadic_outputs() -> (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>) {
  %ax = mo.constant {value = #M.dense_array<-1> : tensor<si32>} : !mo.tensor<[], si32>
  %input = mo.constant {value = #M.dense_array<0, 1, 1, 2, 3, 5> : tensor<6xsi32>} : !mo.tensor<[6], si32>
  %splitSizes = mo.constant {value = #M.dense_array<1, 2, 3> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %out1, %out2, %out3 = mo.split[%ax: !mo.tensor<[], si32>](%input, %splitSizes) : (
    !mo.tensor<[6], si32>, !mo.tensor<[3], si64>
  ) -> (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>)
  mo.output %out1, %out2, %out3 : !mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>
}

// CHECK-LABEL: @multiple_outputs_and_opaque
// CHECK: mogg.to_kgen{{.*}}tensorIOType = "OutputTensor"
// CHECK: mogg.to_kgen{{.*}}tensorIOType = "OutputTensor"
// CHECK: mogg.to_kgen{{.*}}!mo.opaque
// CHECK: mogg.to_kgen{{.*}}tensorIOType = "InputTensor"
mo.graph @multiple_outputs_and_opaque(%op: !mo.opaque<"MyCustomScalarSI32">, %t2: !mo.tensor<[3], f32>) {
  %ch0 = mo.chain.create()
  %out:3 = mo.custom{symbol = "tensor_opaque_tensor_kernel"} (%op, %t2, %ch0) :
    (!mo.opaque<"MyCustomScalarSI32">, !mo.tensor<[3], f32>, !mo.chain) -> (!mo.tensor<[4], f32>, !mo.tensor<[5], f32>, !mo.chain)
  mo.output
}

// CHECK-LABEL: @shape_func
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "InputTensor"{{.*}}(!mo.tensor<[2, 3, 4], f32>)
// CHECK: "mogg.to_kgen"{{.*}}tensorIOType = "InputTensor"{{.*}}(!mo.tensor<[2, 4, 5], f32>)
mo.graph @shape_func(
  %lhs: !mo.tensor<[2, 3, 4], f32>,
  %rhs: !mo.tensor<[2, 4, 5], f32>
) {

  mo.invoke_shape_func["mo.batch_matmul"]<() -> D3, D4, D5>(%lhs, %rhs)
    {dataDeptTensors = []} :
    (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[2, 4, 5], f32>)

  mo.output
}
