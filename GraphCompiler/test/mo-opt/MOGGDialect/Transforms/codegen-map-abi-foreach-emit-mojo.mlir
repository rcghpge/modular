// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s


// CHECK: def {{[A-Za-z0-9_]+_map(_[0-9]+)?}}({{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], [[CTX:.*]]: {{.*}}DeviceContext) raises:
// CHECK:   var [[SHAPE:.*]] = layout.Coord(layout.Idx[2], layout.Idx[2])
// CHECK:   var [[LAYOUT:.*]] = layout.row_major([[SHAPE]])
// CHECK:   var [[MTS0:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, False, extensibility.IO.Input, {{[0-9]+}}]({{.*}}, {{.*}})
// CHECK:   var [[MTS1:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, False, extensibility.IO.Input, {{[0-9]+}}]({{.*}}, {{.*}})
// CHECK:   var [[MTS2:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, True, extensibility.IO.Output, {{[0-9]+}}]({{.*}}, {{.*}})
// CHECK:   var [[PTR0:.*]] = [[MTS0]]._ptr
// CHECK:   var [[PTR1:.*]] = [[MTS1]]._ptr
// CHECK:   var [[PTR2:.*]] = [[MTS2]]._ptr
// CHECK:   def [[LAMBDA:lambda___[0-9]+]][width : Int, element_alignment : Int = 1](idx: layout.Coord) {var}:
// CHECK:     var [[IN0:.*]] = layout.TileTensor[dtype = DType.float32, LayoutType = type_of([[LAYOUT]]), origin = UntrackedOrigin[mut = False]](ptr = [[PTR0]], layout = [[LAYOUT]])
// CHECK:     var [[IN1:.*]] = layout.TileTensor[dtype = DType.float32, LayoutType = type_of([[LAYOUT]]), origin = UntrackedOrigin[mut = False]](ptr = [[PTR1]], layout = [[LAYOUT]])
// CHECK:     var [[OUT:.*]] = layout.TileTensor[dtype = DType.float32, LayoutType = type_of([[LAYOUT]]), origin = UntrackedOrigin[mut = True]](ptr = [[PTR2]], layout = [[LAYOUT]])
// CHECK:     var [[XV:.*]] = [[IN0]].load[width = width, alignment = element_alignment](idx)
// CHECK:     var [[YV:.*]] = [[IN1]].load[width = width, alignment = element_alignment](idx)
// CHECK:     var [[RES:.*]] = builtin_kernels.elementwise.Add.elementwise[dtype = DType.float32]([[XV]], [[YV]])
// CHECK:     [[OUT]].store[width = width, alignment = element_alignment](idx, rebind[SIMD[type_of([[OUT]]).dtype, width]]([[RES]]))
// CHECK:   {{.*}}_foreach[simd_width = {{.*}}get_simd_width_for_dtypes[{{.*}}](), target = "cpu"]([[SHAPE]], [[LAMBDA]], [[CTX]])

mo.graph @add(%arg0: !mo.tensor<[2, 2], f32>, %arg1: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.add(%arg0, %arg1) : !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}
