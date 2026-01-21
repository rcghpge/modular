// RUN: mo-opt %s -mo-to-mogg -mogg-to-kgen -mo-to-primitives -lower-abi -canonicalize -outline-kernels -add-try-except-blocks -emit-mojo -dump-mojo 2>&1 | FileCheck %s

// Test that mo.concat with variadic inputs properly generates VariadicTensors in the emitted Mojo code

// CHECK: fn stub_0({{.*}}InputTensor[dtype = DType.int32, rank = 1], {{.*}}InputTensor[dtype = DType.int32, rank = 1], {{.*}}InputTensor[dtype = DType.int32, rank = 1], {{.*}}OutputTensor[dtype = DType.int32, rank = 1]{{.*}}) raises:
// CHECK: var {{.*}} = StaticTuple[size = 3]({{.*}}._ptr, {{.*}}._ptr, {{.*}}._ptr)
// CHECK: var {{.*}} = StaticTuple[size = 3]({{.*}}.shape(), {{.*}}.shape(), {{.*}}.shape())
// CHECK: comptime {{.*}} = StaticTuple[size = 3]({{.*}}.static_spec, {{.*}}.static_spec, {{.*}}.static_spec)
// CHECK: var {{.*}} = VariadicTensors[io_spec = {{.*}}.io_spec, static_specs = {{.*}}]({{.*}}, {{.*}})
// CHECK: MOGGKernelAPI.Concat.execute[target = "cpu"]({{.*}}, {{.*}}, {{.*}}, {{.*}})

mo.graph @concat(%arg0: !mo.tensor<[2], si32>, %arg1: !mo.tensor<[3], si32>, %arg2: !mo.tensor<[4], si32>) -> !mo.tensor<[9], si32> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %result = mo.concat[%axis: !mo.tensor<[], si32>](%arg0, %arg1, %arg2) : (
    !mo.tensor<[2], si32>, !mo.tensor<[3], si32>, !mo.tensor<[4], si32>
  ) -> !mo.tensor<[9], si32>
  mo.output %result : !mo.tensor<[9], si32>
}
