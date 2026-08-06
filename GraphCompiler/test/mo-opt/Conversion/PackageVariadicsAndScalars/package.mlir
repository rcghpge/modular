// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mogg-package-variadics-and-scalars | FileCheck %s

// PackageVariadicsAndScalars runs directly on the mogg codegen ops (the
// post-MAPToMOGG form, with plain !mogg.tensor_ operands). It re-runs operand
// inference against the callee mogg.kernel.decl and materializes the non-tensor
// operands the kernel expects: a scalar operand is wrapped in
// mogg._scalar.create, and a run of variadic tensor operands is packed into one
// mogg._tuple.create.

// The second input is a scalar kernel argument.
// CHECK-LABEL: func.func @scalar
// CHECK: %[[SC:.*]] = mogg._scalar.create(%{{.*}}) : (!mogg.tensor_) -> !mogg.scalar_
// CHECK: mogg._call.opaque @{{.*}}MapLowerScalar inputs(%{{.*}}, %[[SC]] : !mogg.tensor_, !mogg.scalar_) outputs(%{{.*}} : !mogg.tensor_) context(%{{.*}} : !mogg.context)
func.func @scalar(%in: !mogg.tensor_, %scalar: !mogg.tensor_, %out: !mogg.tensor_, %ctx: !mogg.context) {
  mogg._call.opaque @testing_kernels.testing_kernels.MapLowerScalar inputs(%in, %scalar : !mogg.tensor_, !mogg.tensor_) outputs(%out : !mogg.tensor_) context(%ctx : !mogg.context)
  return
}

// The two inputs satisfy one variadic kernel argument.
// CHECK-LABEL: func.func @variadic
// CHECK: %[[TUP:.*]] = mogg._tuple.create(%{{.*}}, %{{.*}}) : (!mogg.tensor_, !mogg.tensor_) -> !mogg.tuple_
// CHECK: mogg._call.opaque @{{.*}}MapLowerVariadic inputs(%[[TUP]] : !mogg.tuple_) outputs(%{{.*}} : !mogg.tensor_) context(%{{.*}} : !mogg.context)
func.func @variadic(%a: !mogg.tensor_, %b: !mogg.tensor_, %out: !mogg.tensor_, %ctx: !mogg.context) {
  mogg._call.opaque @testing_kernels.testing_kernels.MapLowerVariadic inputs(%a, %b : !mogg.tensor_, !mogg.tensor_) outputs(%out : !mogg.tensor_) context(%ctx : !mogg.context)
  return
}
