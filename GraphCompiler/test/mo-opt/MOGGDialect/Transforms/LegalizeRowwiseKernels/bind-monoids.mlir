// RUN: mo-opt %s --mogg-rowwise-bind-monoids | FileCheck %s

// Each fold's monoid dtype parameters are bound from its result types, and
// its output fields are named. `accum` is the widest contribution type.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.monoids.ArgMax registration = "mo.reduce.arg_max.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc_values", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"acc_indices", #lit.struct<{_mlir_value: dtype = si64}> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.monoids.Welford registration = "mo.reduce.mean_var.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"count", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"mean", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"M2", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

// CHECK-LABEL: func.func @bf16Sum
// CHECK:       attributes {accum = f32}
// CHECK:       } {fields = ["acc"], params = {dtype = f32}}
func.func @bf16Sum(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> f32 {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> bf16
      %a = arith.extf %v : bf16 to f32
      mogg.yield %a : f32
    }
    mogg.tensor.store %co[%row] = %acc : !mogg.tensor, !map.index<2>, f32
  }
  return
}

// ArgMax's index field has a fixed int64 dtype; only `dtype` is bound.
// CHECK-LABEL: func.func @argMax
// CHECK:       attributes {accum = f32}
// CHECK:       } {fields = ["acc_values", "acc_indices"], params = {dtype = f32}}
func.func @argMax(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %ov: !mogg.tensor, %oi: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %cv = %ov : !mogg.tensor, %ci = %oi : !mogg.tensor} device = "cpu" {
    %v, %i = mogg.rowwise.reduce @builtin_kernels.monoids.ArgMax (%i : !map.index<2>) -> (f32, si64) {
      %xv = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> f32
      mogg.yield %xv : f32
    }
    mogg.tensor.store %cv[%row] = %v : !mogg.tensor, !map.index<2>, f32
    mogg.tensor.store %ci[%row] = %i : !mogg.tensor, !map.index<2>, si64
  }
  return
}

// `accum` is the wider of the two folds' contributions.
// CHECK-LABEL: func.func @twoFolds
// CHECK:       attributes {accum = f32}
// CHECK:       } {fields = ["acc"], params = {dtype = f16}}
// CHECK:       } {fields = ["acc"], params = {dtype = f32}}
func.func @twoFolds(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %a = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> f16 {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> f16
      mogg.yield %v : f16
    }
    %b = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> f32 {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> f16
      %w = arith.extf %v : f16 to f32
      mogg.yield %w : f32
    }
    %af = arith.extf %a : f16 to f32
    %s = arith.addf %af, %b : f32
    mogg.tensor.store %co[%row] = %s : !mogg.tensor, !map.index<2>, f32
  }
  return
}
