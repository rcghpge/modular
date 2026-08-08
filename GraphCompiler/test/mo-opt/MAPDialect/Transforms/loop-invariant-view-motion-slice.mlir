// RUN: mo-opt %s --pass-pipeline='builtin.module(mo.graph(map.iter.opaque(map-loop-invariant-view-motion)))' | FileCheck %s

// LoopInvariantViewMotion hoists a map.index.slice chain feeding a map.load
// out of the loop body into a map.index.slice.stride op in the enclosing
// map.iter.opaque, rewriting the load to use the loop's own (untransformed)
// index directly. Covers: static and symbolic-dim shapes, a load with no
// transform at all (must be left untouched), an arbitrary number of
// independent views/loads in one body, a load mixed with an untransformed
// one, two loads sharing the exact same hoisted view (dedup), and a chain of
// two index-transform hops feeding a single load.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>

module {

// map.call.elementwise verifies its callee resolves to a mogg.kernel.decl
// with an `elementwise` function, so this stub must be present.
mogg.kernel.decl @builtin_kernels.elementwise.Add registration = "mo.add" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

// A single view feeding a single load: the common case.

// CHECK-LABEL: mo.graph @sliceStatic
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[INPUT:%[^ :,]+]]: !mo.tensor<[10, 64], f32>, [[STARTS:%[^ :,]+]]: !mo.tensor<[2], si64>, [[STOPS:%[^ :,]+]]: !mo.tensor<[2], si64>, [[STEPS:%[^ :,]+]]: !mo.tensor<[2], si64>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[3, 32], f32>) {
// CHECK-NEXT:      [[STRIDED:%[^ ]+]] = map.index.slice.stride<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> ([[INPUT]], [[STARTS]], [[STOPS]], [[STEPS]])
// CHECK-NEXT:      map.iter.foreach.body<[3, 32]>
// CHECK-NEXT:        index   ([[IDX:%[^ :,]+]]: !map.index<2>) {
// CHECK-NOT:       map.index.slice
// CHECK-NEXT:      [[VAL:%[^ ]+]] = map.load [[STRIDED]]{{\[}}[[IDX]]] : !mo.tensor<[3, 32], f32>
// CHECK-NEXT:      map.store [[OUT]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[3, 32], f32>
// CHECK-NEXT:    }
mo.graph @sliceStatic(%input: !mo.tensor<[10, 64], f32>,
                      %starts: !mo.tensor<[2], si64>,
                      %stops: !mo.tensor<[2], si64>,
                      %steps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 32], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[10, 64], f32>, %starts : !mo.tensor<[2], si64>, %stops : !mo.tensor<[2], si64>, %steps : !mo.tensor<[2], si64>)
      outputs (%out : !mo.tensor<[3, 32], f32>) {
    map.iter.foreach.body<[3, 32]>
        index (%idx: !map.index<2>) {
      %idx_ = map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> (%idx, %starts, %stops, %steps)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %val = map.load %input[%idx_] : !mo.tensor<[10, 64], f32>
      map.store %out[%idx] = %val : !mo.tensor<[3, 32], f32>
    }
  }
  mo.output %output : !mo.tensor<[3, 32], f32>
}

// Symbolic-dim shapes and dynamic (`?`) bounds: the hoisting logic doesn't
// care whether the spec is static or dynamic.

// CHECK-LABEL: mo.graph @sliceDynamic
// CHECK:         map.iter.opaque
// CHECK:           map.index.slice.stride<#map.index.slice.spec<[N, 64] -> [M, 32], starts = [?, ?], stops = [?, ?], steps = [?, ?]>>
// CHECK:           map.iter.foreach.body
// CHECK-NOT:         map.index.slice
mo.graph @sliceDynamic<N, M>(
    %input: !mo.tensor<[N, 64], f32>,
    %starts: !mo.tensor<[2], si64>,
    %stops: !mo.tensor<[2], si64>,
    %steps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[M, 32], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[N, 64], f32>, %starts : !mo.tensor<[2], si64>, %stops : !mo.tensor<[2], si64>, %steps : !mo.tensor<[2], si64>)
      outputs (%out : !mo.tensor<[M, 32], f32>) {
    map.iter.foreach.body<[M, 32]>
        index (%idx: !map.index<2>) {
      %idx_ = map.index.slice<#map.index.slice.spec<[N, 64] -> [M, 32], starts = [?, ?], stops = [?, ?], steps = [?, ?]>> (%idx, %starts, %stops, %steps)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %val = map.load %input[%idx_] : !mo.tensor<[N, 64], f32>
      map.store %out[%idx] = %val : !mo.tensor<[M, 32], f32>
    }
  }
  mo.output %output : !mo.tensor<[M, 32], f32>
}

// A load with no index-transform at all (already indexes with the loop's own
// index, as an elementwise op's loads would): left completely untouched, no
// map.index.slice.stride is introduced.

// CHECK-LABEL: mo.graph @noTransform
// CHECK:         [[RESULT3:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[X3:%[^ :,]+]]: !mo.tensor<[8, 8], f32>)
// CHECK-NEXT:    outputs ([[OUT3:%[^ :,]+]] : !mo.tensor<[8, 8], f32>) {
// CHECK-NEXT:      map.iter.foreach.body<[8, 8]>
// CHECK-NEXT:        index   ([[IDX3:%[^ :,]+]]: !map.index<2>) {
// CHECK-NEXT:      [[VAL3:%[^ ]+]] = map.load [[X3]]{{\[}}[[IDX3]]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:      map.store [[OUT3]]{{\[}}[[IDX3]]] = [[VAL3]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:    }
// CHECK-NOT:   map.index.slice.stride
mo.graph @noTransform(%x: !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32> {
  %output = map.iter.opaque
      inputs (%x : !mo.tensor<[8, 8], f32>)
      outputs (%out : !mo.tensor<[8, 8], f32>) {
    map.iter.foreach.body<[8, 8]>
        index (%idx: !map.index<2>) {
      %val = map.load %x[%idx] : !mo.tensor<[8, 8], f32>
      map.store %out[%idx] = %val : !mo.tensor<[8, 8], f32>
    }
  }
  mo.output %output : !mo.tensor<[8, 8], f32>
}

// Two independent views feeding two independent loads in the same body
// (arbitrary number of views / loads): both get hoisted, each to its own
// map.index.slice.stride, each rewritten independently.

// CHECK-LABEL: mo.graph @twoIndependentViews
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs
// CHECK-NEXT:    outputs ([[OUT4:%[^ :,]+]] : !mo.tensor<[3, 3], f32>) {
// CHECK-NEXT:      [[STRIDEDA:%[^ ]+]] = map.index.slice.stride<#map.index.slice.spec<[10, 10] -> [3, 3], starts = [0, 0], stops = [3, 3], steps = [1, 1]>>
// CHECK-NEXT:      [[STRIDEDB:%[^ ]+]] = map.index.slice.stride<#map.index.slice.spec<[20, 20] -> [3, 3], starts = [5, 5], stops = [8, 8], steps = [1, 1]>>
// CHECK-NEXT:      map.iter.foreach.body<[3, 3]>
// CHECK-NEXT:        index   ([[IDX4:%[^ :,]+]]: !map.index<2>) {
// CHECK-NOT:       map.index.slice
// CHECK:           [[AV:%[^ ]+]] = map.load [[STRIDEDA]]{{\[}}[[IDX4]]]
// CHECK:           [[BV:%[^ ]+]] = map.load [[STRIDEDB]]{{\[}}[[IDX4]]]
// CHECK:           [[SUM:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Add([[IDX4]], [[AV]], [[BV]])
// CHECK:           map.store [[OUT4]]{{\[}}[[IDX4]]] = [[SUM]]
mo.graph @twoIndependentViews(
    %a: !mo.tensor<[10, 10], f32>,
    %aStarts: !mo.tensor<[2], si64>,
    %aStops: !mo.tensor<[2], si64>,
    %aSteps: !mo.tensor<[2], si64>,
    %b: !mo.tensor<[20, 20], f32>,
    %bStarts: !mo.tensor<[2], si64>,
    %bStops: !mo.tensor<[2], si64>,
    %bSteps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 3], f32> {
  %output = map.iter.opaque
      inputs (%a : !mo.tensor<[10, 10], f32>, %aStarts : !mo.tensor<[2], si64>, %aStops : !mo.tensor<[2], si64>, %aSteps : !mo.tensor<[2], si64>,
              %b : !mo.tensor<[20, 20], f32>, %bStarts : !mo.tensor<[2], si64>, %bStops : !mo.tensor<[2], si64>, %bSteps : !mo.tensor<[2], si64>)
      outputs (%out : !mo.tensor<[3, 3], f32>) {
    map.iter.foreach.body<[3, 3]>
        index (%idx: !map.index<2>) {
      %aIdx = map.index.slice<#map.index.slice.spec<[10, 10] -> [3, 3], starts = [0, 0], stops = [3, 3], steps = [1, 1]>> (%idx, %aStarts, %aStops, %aSteps)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %aVal = map.load %a[%aIdx] : !mo.tensor<[10, 10], f32>
      %bIdx = map.index.slice<#map.index.slice.spec<[20, 20] -> [3, 3], starts = [5, 5], stops = [8, 8], steps = [1, 1]>> (%idx, %bStarts, %bStops, %bSteps)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %bVal = map.load %b[%bIdx] : !mo.tensor<[20, 20], f32>
      %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %aVal, %bVal)
                : !map.index<2>, f32, f32 -> f32
      map.store %out[%idx] = %sum : !mo.tensor<[3, 3], f32>
    }
  }
  mo.output %output : !mo.tensor<[3, 3], f32>
}

// A view mixed with a plain (untransformed) load: only the view's load is
// rewritten; the other is untouched.

// CHECK-LABEL: mo.graph @mixedTransformedAndPlain
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[X5:%[^ :,]+]]: !mo.tensor<[3, 3], f32>, [[A5:%[^ :,]+]]: !mo.tensor<[10, 10], f32>,
// CHECK:         [[STRIDED5:%[^ ]+]] = map.index.slice.stride
// CHECK-NEXT:    map.iter.foreach.body<[3, 3]>
// CHECK-NEXT:      index   ([[IDX5:%[^ :,]+]]: !map.index<2>) {
// CHECK-NOT:     map.index.slice
// CHECK:         [[XV5:%[^ ]+]] = map.load [[X5]]{{\[}}[[IDX5]]] : !mo.tensor<[3, 3], f32>
// CHECK:         [[AV5:%[^ ]+]] = map.load [[STRIDED5]]{{\[}}[[IDX5]]]
mo.graph @mixedTransformedAndPlain(
    %x: !mo.tensor<[3, 3], f32>,
    %a: !mo.tensor<[10, 10], f32>,
    %aStarts: !mo.tensor<[2], si64>,
    %aStops: !mo.tensor<[2], si64>,
    %aSteps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 3], f32> {
  %output = map.iter.opaque
      inputs (%x : !mo.tensor<[3, 3], f32>, %a : !mo.tensor<[10, 10], f32>, %aStarts : !mo.tensor<[2], si64>, %aStops : !mo.tensor<[2], si64>, %aSteps : !mo.tensor<[2], si64>)
      outputs (%out : !mo.tensor<[3, 3], f32>) {
    map.iter.foreach.body<[3, 3]>
        index (%idx: !map.index<2>) {
      %xVal = map.load %x[%idx] : !mo.tensor<[3, 3], f32>
      %aIdx = map.index.slice<#map.index.slice.spec<[10, 10] -> [3, 3], starts = [0, 0], stops = [3, 3], steps = [1, 1]>> (%idx, %aStarts, %aStops, %aSteps)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %aVal = map.load %a[%aIdx] : !mo.tensor<[10, 10], f32>
      %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xVal, %aVal)
                : !map.index<2>, f32, f32 -> f32
      map.store %out[%idx] = %sum : !mo.tensor<[3, 3], f32>
    }
  }
  mo.output %output : !mo.tensor<[3, 3], f32>
}

// Two loads of the *same* tensor sharing the exact same map.index.slice SSA
// value: hoisted once, both loads reuse the one map.index.slice.stride.

// CHECK-LABEL: mo.graph @sharedChain
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs
// CHECK-NEXT:    outputs ([[OUT6:%[^ :,]+]] : !mo.tensor<[3, 3], f32>) {
// CHECK-NEXT:      [[STRIDED6:%[^ ]+]] = map.index.slice.stride
// CHECK-NEXT:      map.iter.foreach.body<[3, 3]>
// CHECK-NEXT:        index   ([[IDX6:%[^ :,]+]]: !map.index<2>) {
// CHECK-NOT:       map.index.slice
// CHECK-NOT:       map.index.slice.stride
// CHECK:           [[V0:%[^ ]+]] = map.load [[STRIDED6]]{{\[}}[[IDX6]]]
// CHECK:           [[V1:%[^ ]+]] = map.load [[STRIDED6]]{{\[}}[[IDX6]]]
mo.graph @sharedChain(
    %a: !mo.tensor<[10, 10], f32>,
    %aStarts: !mo.tensor<[2], si64>,
    %aStops: !mo.tensor<[2], si64>,
    %aSteps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 3], f32> {
  %output = map.iter.opaque
      inputs (%a : !mo.tensor<[10, 10], f32>, %aStarts : !mo.tensor<[2], si64>, %aStops : !mo.tensor<[2], si64>, %aSteps : !mo.tensor<[2], si64>)
      outputs (%out : !mo.tensor<[3, 3], f32>) {
    map.iter.foreach.body<[3, 3]>
        index (%idx: !map.index<2>) {
      %aIdx = map.index.slice<#map.index.slice.spec<[10, 10] -> [3, 3], starts = [0, 0], stops = [3, 3], steps = [1, 1]>> (%idx, %aStarts, %aStops, %aSteps)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %v0 = map.load %a[%aIdx] : !mo.tensor<[10, 10], f32>
      %v1 = map.load %a[%aIdx] : !mo.tensor<[10, 10], f32>
      %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %v0, %v1)
                : !map.index<2>, f32, f32 -> f32
      map.store %out[%idx] = %sum : !mo.tensor<[3, 3], f32>
    }
  }
  mo.output %output : !mo.tensor<[3, 3], f32>
}

// A chain of two index-transform hops (a view feeding another view's index):
// both hops get hoisted into their own map.index.slice.stride, composed
// hop-by-hop starting from the original tensor, and the load addresses the
// fully-composed result directly with the outermost hop's own
// (untransformed) index.

// CHECK-LABEL: mo.graph @chainedSliceIndex
// CHECK:         [[RESULT7:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[INPUT7:%[^ :,]+]]: !mo.tensor<[10, 10], f32>, [[STARTS7:%[^ :,]+]]: !mo.tensor<[2], si64>, [[STOPS7:%[^ :,]+]]: !mo.tensor<[2], si64>, [[STEPS7:%[^ :,]+]]: !mo.tensor<[2], si64>)
// CHECK-NEXT:    outputs ([[OUT7:%[^ :,]+]] : !mo.tensor<[3, 3], f32>) {
// CHECK-NEXT:      [[INNER7:%[^ ]+]] = map.index.slice.stride<#map.index.slice.spec<[10, 10] -> [6, 6], starts = [0, 0], stops = [6, 6], steps = [1, 1]>> ([[INPUT7]], [[STARTS7]], [[STOPS7]], [[STEPS7]])
// CHECK-NEXT:      [[OUTER7:%[^ ]+]] = map.index.slice.stride<#map.index.slice.spec<[6, 6] -> [3, 3], starts = [0, 0], stops = [3, 3], steps = [1, 1]>> ([[INNER7]], [[STARTS7]], [[STOPS7]], [[STEPS7]])
// CHECK-NEXT:      map.iter.foreach.body<[3, 3]>
// CHECK-NEXT:        index   ([[IDX7:%[^ :,]+]]: !map.index<2>) {
// CHECK-NOT:       map.index.slice
// CHECK-NEXT:      [[VAL7:%[^ ]+]] = map.load [[OUTER7]]{{\[}}[[IDX7]]] : !mo.tensor<[3, 3], f32>
// CHECK-NEXT:      map.store [[OUT7]]{{\[}}[[IDX7]]] = [[VAL7]] : !mo.tensor<[3, 3], f32>
// CHECK-NEXT:    }
mo.graph @chainedSliceIndex(
    %input: !mo.tensor<[10, 10], f32>,
    %starts: !mo.tensor<[2], si64>,
    %stops: !mo.tensor<[2], si64>,
    %steps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 3], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[10, 10], f32>, %starts : !mo.tensor<[2], si64>, %stops : !mo.tensor<[2], si64>, %steps : !mo.tensor<[2], si64>)
      outputs (%out : !mo.tensor<[3, 3], f32>) {
    map.iter.foreach.body<[3, 3]>
        index (%idx: !map.index<2>) {
      %mid = map.index.slice<#map.index.slice.spec<[6, 6] -> [3, 3], starts = [0, 0], stops = [3, 3], steps = [1, 1]>> (%idx, %starts, %stops, %steps)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %inner = map.index.slice<#map.index.slice.spec<[10, 10] -> [6, 6], starts = [0, 0], stops = [6, 6], steps = [1, 1]>> (%mid, %starts, %stops, %steps)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %val = map.load %input[%inner] : !mo.tensor<[10, 10], f32>
      map.store %out[%idx] = %val : !mo.tensor<[3, 3], f32>
    }
  }
  mo.output %output : !mo.tensor<[3, 3], f32>
}

}
