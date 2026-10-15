// RUN: mo-opt %s --mogg-rowwise-select-primary-input | FileCheck %s

// The primary input is the capture the first fold loads at its own index.
// Those loads, in every fold, become `mogg.rowwise.primary`; other loads stay.
// `dtype` is its element type, and `is_cached` says whether more than one fold
// reads it.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.monoids.ArgMax registration = "mo.reduce.arg_max.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc_values", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"acc_indices", #lit.struct<{_mlir_value: dtype = si64}> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.monoids.Welford registration = "mo.reduce.mean_var.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"count", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"mean", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"M2", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

// CHECK-LABEL: func.func @twoInputs
// CHECK:       mogg.call.rowwise {{.*}} capturing {%[[X:cap[0-9]+]] = %arg{{[0-9]+}} : !mogg.tensor, %[[Y:cap[0-9]+]] = %arg{{[0-9]+}} : !mogg.tensor, {{.*}}} device = "cpu" attributes {dtype = f32, is_cached = false, primary = 0 : i64} {
// CHECK-NEXT:    mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%[[I:[[:alnum:]_]+]] : !map.index<2>) -> f32 {
// CHECK-NEXT:      %[[XV:.*]] = mogg.rowwise.primary : f32
// CHECK-NEXT:      %[[YV:.*]] = mogg.tensor.load %[[Y]][%[[I]]]
// CHECK-NEXT:      arith.mulf %[[XV]], %[[YV]]
// CHECK:         mogg.tensor.load %[[X]][%row]
func.func @twoInputs(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %y: !mogg.tensor, %out: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %cy = %y : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> f32 {
      %xv = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> f32
      %yv = mogg.tensor.load %cy[%i] : !mogg.tensor, !map.index<2> -> f32
      %p = arith.mulf %xv, %yv : f32
      mogg.yield %p : f32
    }
    %xr = mogg.tensor.load %cx[%row] : !mogg.tensor, !map.index<2> -> f32
    %s = arith.addf %acc, %xr : f32
    mogg.tensor.store %co[%row] = %s : !mogg.tensor, !map.index<2>, f32
  }
  return
}

// The first load picks the primary, whatever the capture order.
// CHECK-LABEL: func.func @firstLoad
// CHECK:       attributes {dtype = f32, is_cached = false, primary = 1 : i64}
func.func @firstLoad(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %y: !mogg.tensor, %out: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %cy = %y : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> f32 {
      %yv = mogg.tensor.load %cy[%i] : !mogg.tensor, !map.index<2> -> f32
      %xv = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> f32
      %p = arith.mulf %xv, %yv : f32
      mogg.yield %p : f32
    }
    mogg.tensor.store %co[%row] = %acc : !mogg.tensor, !map.index<2>, f32
  }
  return
}

// Every fold's element loads of the primary input are replaced.
// CHECK-LABEL: func.func @twoFolds
// CHECK:       attributes {dtype = f32, is_cached = true, primary = 0 : i64}
// CHECK:       mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum
// CHECK-NEXT:    mogg.rowwise.primary : f32
// CHECK:       mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum
// CHECK-NEXT:    mogg.rowwise.primary : f32
func.func @twoFolds(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %a = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> f32 {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> f32
      mogg.yield %v : f32
    }
    %b = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> f32 {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> f32
      %d = arith.subf %v, %a : f32
      mogg.yield %d : f32
    }
    mogg.tensor.store %co[%row] = %b : !mogg.tensor, !map.index<2>, f32
  }
  return
}
