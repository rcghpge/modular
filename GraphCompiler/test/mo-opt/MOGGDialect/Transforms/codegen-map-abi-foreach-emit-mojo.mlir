// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s


// CHECK: def {{[A-Za-z0-9_]+_map(_[0-9]+)?}}({{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], [[CTX:.*]]: {{.*}}DeviceContext) raises:
// CHECK:   var [[SHAPE:.*]] = layout.Coord(layout.Idx[2], layout.Idx[2])
// CHECK:   var {{.*}} = layout.row_major([[SHAPE]])
// CHECK:   var [[MTS0:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, False, extensibility.IO.Input, {{[0-9]+}}]({{.*}}, {{.*}})
// CHECK:   var [[MTS1:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, False, extensibility.IO.Input, {{[0-9]+}}]({{.*}}, {{.*}})
// CHECK:   var [[MTS2:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, True, extensibility.IO.Output, {{[0-9]+}}]({{.*}}, {{.*}})
// Exactly like the fusion functor, each capture is decomposed into (ptr, layout)
// plus its own static alignment, rebuilt as a TileTensor inside the closure, and
// accessed through the stride-aware primitive with that alignment threaded in.
// CHECK:   var [[PTR0:.*]] = [[MTS0]]._ptr
// CHECK:   var [[LAY0:.*]] = [[MTS0]].to_tile_tensor().layout
// CHECK:   comptime [[AL0:.*]] = [[MTS0]].alignment
// CHECK:   var [[PTR1:.*]] = [[MTS1]]._ptr
// CHECK:   var [[LAY1:.*]] = [[MTS1]].to_tile_tensor().layout
// CHECK:   comptime [[AL1:.*]] = [[MTS1]].alignment
// CHECK:   var [[PTR2:.*]] = [[MTS2]]._ptr
// CHECK:   var [[LAY2:.*]] = [[MTS2]].to_tile_tensor().layout
// CHECK:   comptime [[AL2:.*]] = [[MTS2]].alignment
// CHECK:   def [[LAMBDA:lambda___[0-9]+]][width : Int, element_alignment : Int = 1](idx: layout.Coord) {var}:
// CHECK:     var [[T0:.*]] = layout.TileTensor[dtype = DType.float32, LayoutType = type_of([[LAY0]]), origin = UntrackedOrigin[mut = False]](ptr = [[PTR0]], layout = [[LAY0]])
// CHECK:     var [[T1:.*]] = layout.TileTensor[dtype = DType.float32, LayoutType = type_of([[LAY1]]), origin = UntrackedOrigin[mut = False]](ptr = [[PTR1]], layout = [[LAY1]])
// CHECK:     var [[T2:.*]] = layout.TileTensor[dtype = DType.float32, LayoutType = type_of([[LAY2]]), origin = UntrackedOrigin[mut = True]](ptr = [[PTR2]], layout = [[LAY2]])
// CHECK:     var [[XV:.*]] = {{.*}}tile_tensor_strided_load[simd_width = width, element_alignment = element_alignment, tensor_alignment = [[AL0]]]([[T0]], idx)
// CHECK:     var [[YV:.*]] = {{.*}}tile_tensor_strided_load[simd_width = width, element_alignment = element_alignment, tensor_alignment = [[AL1]]]([[T1]], idx)
// CHECK:     var [[RES:.*]] = builtin_kernels.elementwise.Add.elementwise[dtype = DType.float32]([[XV]], [[YV]])
// CHECK:     {{.*}}tile_tensor_strided_store[simd_width = width, element_alignment = element_alignment, tensor_alignment = [[AL2]]]([[T2]], idx, rebind[SIMD[type_of([[T2]]).dtype, width]]([[RES]]))
// CHECK:   {{.*}}_foreach[simd_width = {{.*}}get_simd_width_for_dtypes[{{.*}}](), target = "cpu"]([[SHAPE]], [[LAMBDA]], [[CTX]])

mo.graph @add(%arg0: !mo.tensor<[2, 2], f32>, %arg1: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.add(%arg0, %arg1) : !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}
