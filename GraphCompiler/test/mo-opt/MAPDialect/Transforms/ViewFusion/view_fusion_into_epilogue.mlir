// RUN: mo-opt %s --map-fuse-views | FileCheck %s

// Hand-authored MAP-level IR (mirrors ViewFusion/view_fusion_into_prologue.mlir):
// exercises ViewFuser's fusion of a view directly into an existing
// map.iter.epilogue, the bias-add-with-broadcast case -- an epilogue already
// captures a bias tensor and reads it via map.load; when that tensor is
// itself the rank-expanded result of a broadcast kernel, ViewFuser splices
// the broadcast's body into the epilogue's own load site, the one edge
// EpilogueFuser's simpler splice-verbatim mechanism can't take (a transform
// op's ParentOneOf used to reject living inside an epilogue).

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>

module {

mogg.kernel.decl @builtin_kernels.elementwise.Add registration = "mo.add" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false
mogg.kernel.decl @kernels.producer registration = "kernels.producer" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @broadcastIntoExistingEpilogue
// CHECK-SAME:  (%[[MAIN:.*]]: !mo.tensor<[8, 8], f32>, %[[BIAS:.*]]: !mo.tensor<[8], f32>)
// CHECK-NOT:   map.iter.foreach
// CHECK:       %[[RES:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[MAIN]]: !mo.tensor<[8, 8], f32>, %[[BIAS]]: !mo.tensor<[8], f32>)
// CHECK-NEXT:  outputs (%[[O:.*]] : !mo.tensor<[8, 8], f32>) {
// CHECK-NEXT:  %[[EPI:.*]] = map.iter.epilogue (%[[VAL:.*]] : f32, %[[IDX:.*]] : !map.index<2>) capturing
// CHECK-SAME:      (%[[OCAP:.*]] = %[[O]] : !mo.tensor<[8, 8], f32>,
// CHECK-SAME:       %[[BCAP:.*]] = %[[BIAS]] : !mo.tensor<[8], f32>)
// CHECK-NEXT:  %[[BIDX:.*]] = map.index.broadcast<{{.*}}> (%[[IDX]])
// CHECK-NEXT:  %[[BV:.*]] = map.load %[[BCAP]][%[[BIDX]]] : !mo.tensor<[8], f32>
// CHECK-NEXT:  %[[SUM:.*]] = map.call.elementwise @{{.*}}.Add(%[[IDX]], %[[VAL]], %[[BV]])
// CHECK-NEXT:  map.store %[[OCAP]][%[[IDX]]] = %[[SUM]]
// CHECK-NEXT:  } : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:  map.call.opaque @kernels.producer[] inputs(%[[MAIN]]) outputs(%[[EPI]])
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[{{.*}}], operandIndices = [1]>>
// CHECK:       mo.output %[[RES]] : !mo.tensor<[8, 8], f32>
mo.graph @broadcastIntoExistingEpilogue(
    %main: !mo.tensor<[8, 8], f32>,
    %bias: !mo.tensor<[8], f32>
) -> !mo.tensor<[8, 8], f32> {
  %bcast = map.iter.foreach<[8, 8]>
      inputs  (%bias: !mo.tensor<[8], f32>)
      outputs (%bout: !mo.tensor<[8, 8], f32>)
      index   (%i: !map.index<2>)
      -> !mo.tensor<[8, 8], f32>
  {
    %bi = map.index.broadcast<#map.index.broadcast.spec<[8] -> [8, 8]>> (%i)
        : !map.index<2> -> !map.index<1>
    %bv = map.load %bias[%bi] : !mo.tensor<[8], f32>
    map.store %bout[%i] = %bv : !mo.tensor<[8, 8], f32>
  }
  %res = map.iter.opaque
      inputs (%main : !mo.tensor<[8, 8], f32>, %bcast : !mo.tensor<[8, 8], f32>)
      outputs (%o : !mo.tensor<[8, 8], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>)
        capturing (%ocap = %o : !mo.tensor<[8, 8], f32>, %bcap = %bcast : !mo.tensor<[8, 8], f32>) {
      %bv = map.load %bcap[%idx] : !mo.tensor<[8, 8], f32>
      %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %val, %bv)
                : !map.index<2>, f32, f32 -> f32
      map.store %ocap[%idx] = %sum : !mo.tensor<[8, 8], f32>
    } : !mo.tensor<[8, 8], f32>
    map.call.opaque @kernels.producer[] inputs (%main) outputs (%epi) {kernelName = "kernels.producer"} : (!mo.tensor<[8, 8], f32>) -> (!mo.tensor<[8, 8], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[8, 8], f32>
}

// A plain elementwise (non-view) producer feeding an existing epilogue is
// EpilogueFuser's own job -- ViewFuser must leave it alone, keeping the two
// fusers' jobs disjoint. Negative control for the case above: same shape of
// graph, but the producer has no IndexTransformOpInterface op in its body.

// CHECK-LABEL: mo.graph @elementwiseIntoExistingEpilogue_noFuse
// CHECK:       %[[EW:.*]] = map.iter.foreach<[8, 8]>
// CHECK:       map.iter.opaque
// CHECK:       map.iter.epilogue
// CHECK:       map.load
// CHECK-NOT:   map.index.broadcast
// CHECK:       map.store
mo.graph @elementwiseIntoExistingEpilogue_noFuse(
    %main: !mo.tensor<[8, 8], f32>,
    %a: !mo.tensor<[8, 8], f32>,
    %b: !mo.tensor<[8, 8], f32>
) -> !mo.tensor<[8, 8], f32> {
  %ew = map.iter.foreach<[8, 8]>
      inputs  (%a: !mo.tensor<[8, 8], f32>, %b: !mo.tensor<[8, 8], f32>)
      outputs (%ewout: !mo.tensor<[8, 8], f32>)
      index   (%i: !map.index<2>)
      -> !mo.tensor<[8, 8], f32>
  {
    %av = map.load %a[%i] : !mo.tensor<[8, 8], f32>
    %bv = map.load %b[%i] : !mo.tensor<[8, 8], f32>
    %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%i, %av, %bv)
               : !map.index<2>, f32, f32 -> f32
    map.store %ewout[%i] = %sum : !mo.tensor<[8, 8], f32>
  }
  %res = map.iter.opaque
      inputs (%main : !mo.tensor<[8, 8], f32>, %ew : !mo.tensor<[8, 8], f32>)
      outputs (%o : !mo.tensor<[8, 8], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>)
        capturing (%ocap = %o : !mo.tensor<[8, 8], f32>, %ecap = %ew : !mo.tensor<[8, 8], f32>) {
      %ev = map.load %ecap[%idx] : !mo.tensor<[8, 8], f32>
      %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %val, %ev)
                : !map.index<2>, f32, f32 -> f32
      map.store %ocap[%idx] = %sum : !mo.tensor<[8, 8], f32>
    } : !mo.tensor<[8, 8], f32>
    map.call.opaque @kernels.producer[] inputs (%main) outputs (%epi) {kernelName = "kernels.producer"} : (!mo.tensor<[8, 8], f32>) -> (!mo.tensor<[8, 8], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[8, 8], f32>
}

}
