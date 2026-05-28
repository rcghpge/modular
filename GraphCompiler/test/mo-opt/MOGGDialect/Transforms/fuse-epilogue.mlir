// RUN: mo-opt --mogg-import-packages --mo-to-mogg --mogg-fuse-elementwise --cse --mogg-fuse-views --cse --mogg-fuse-epilogues --cse --mogg-strip-kernel-decls %s | FileCheck %s

mo.graph @basicMatmul<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>, %arg3: !mo.tensor<[], f32>) -> (!mo.tensor<[M, N], f32>) {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>)  -> !mo.tensor<[M, N], f32>
  %bcast = mo.static.broadcast_to(%arg3) : !mo.tensor<[], f32> -> !mo.tensor<[M, N], f32>
  %1 = mo.add(%mm, %bcast) : !mo.tensor<[M, N], f32>
  mo.output %1 : !mo.tensor<[M, N], f32>
}

// CHECK-LABEL: mo.graph @basicMatmul
// CHECK: mogg.kernel
// CHECK: mo.output
// CHECK-NOT: mogg.kernel{{[^.]}}

mo.graph @basicMatmulInIfStatement<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>, %arg3: !mo.tensor<[], f32>, %arg4: !mo.tensor<[M, N], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[M, N], f32>) {
  %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[M, N], f32> {
    %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>)  -> !mo.tensor<[M, N], f32>
    %bcast = mo.static.broadcast_to(%arg3) : !mo.tensor<[], f32> -> !mo.tensor<[M, N], f32>
    %1 = mo.add(%mm, %bcast) : !mo.tensor<[M, N], f32>
    mo.yield %1 : !mo.tensor<[M, N], f32>
  } else {
    %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>)  -> !mo.tensor<[M, N], f32>
    %bcast = mo.static.broadcast_to(%arg3) : !mo.tensor<[], f32> -> !mo.tensor<[M, N], f32>
    %1 = mo.add(%mm, %bcast) : !mo.tensor<[M, N], f32>
    mo.yield %1 : !mo.tensor<[M, N], f32>
  }
  mo.output %res : !mo.tensor<[M, N], f32>
}

// CHECK-LABEL: mo.graph @basicMatmulInIfStatement
// CHECK: mo.if
// CHECK: mogg.kernel
// CHECK-NOT: mogg.kernel{{[^.]}}
// CHECK: } else {
// CHECK: mogg.kernel
// CHECK-NOT: mogg.kernel{{[^.]}}
// CHECK: mo.output

mo.graph @dontFuseAcrossBlocks<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>, %arg3: !mo.tensor<[], f32>, %arg4: !mo.tensor<[M, N], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[M, N], f32>) {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>)  -> !mo.tensor<[M, N], f32>
  %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[M, N], f32> {
    %bcast = mo.static.broadcast_to(%arg3) : !mo.tensor<[], f32> -> !mo.tensor<[M, N], f32>
    %1 = mo.add(%mm, %bcast) : !mo.tensor<[M, N], f32>
    mo.yield %1 : !mo.tensor<[M, N], f32>
  } else {
    %bcast = mo.static.broadcast_to(%arg3) : !mo.tensor<[], f32> -> !mo.tensor<[M, N], f32>
    %1 = mo.add(%mm, %bcast) : !mo.tensor<[M, N], f32>
    mo.yield %1 : !mo.tensor<[M, N], f32>
  }
  mo.output %res : !mo.tensor<[M, N], f32>
}

// CHECK-LABEL: mo.graph @dontFuseAcrossBlocks
// CHECK: mogg.kernel
// CHECK: mo.if
// CHECK: mogg.kernel
// CHECK: } else {
// CHECK: mogg.kernel

mo.graph @noFuseMultiUse<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>, %arg3: !mo.tensor<[], f32>, %arg4: !mo.tensor<[], f32>) -> (!mo.tensor<[M, N], f32>, !mo.tensor<[M, N], f32>) {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32, {layout = #mo.layout<KN>}>)  -> !mo.tensor<[M, N], f32>
  %bcast = mo.static.broadcast_to(%arg3) : !mo.tensor<[], f32> -> !mo.tensor<[M, N], f32>
  %1 = mo.add(%mm, %bcast) : !mo.tensor<[M, N], f32>
  %bcast2 = mo.static.broadcast_to(%arg4) : !mo.tensor<[], f32> -> !mo.tensor<[M, N], f32>
  %2 = mo.add(%mm, %bcast2) : !mo.tensor<[M, N], f32>
  mo.output %1, %2 : !mo.tensor<[M, N], f32>, !mo.tensor<[M, N], f32>
}

// CHECK-LABEL: mo.graph @noFuseMultiUse
// CHECK: mogg.kernel
// CHECK: mogg.kernel
// CHECK: mogg.kernel
// CHECK: mo.output

mo.graph @mixed_precision(%arg0: !mo.tensor<[2, 2], ui32>, %arg1: !mo.tensor<[2, 2], ui32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[2, 2], f32> {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[2, 2], ui32>, !mo.tensor<[2, 2], ui32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[2, 2], ui32>
  %cast = mo.cast(%mm) : (!mo.tensor<[2, 2], ui32>) -> !mo.tensor<[2, 2], f32>
  mo.output %cast : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: @mixed_precision
// CHECK: mogg.kernel
// CHECK: mo.output
// CHECK-NOT: mogg.kernel{{[^.]}}

mo.graph @checkBasicFusionBehavior<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32> {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32>
  %relu = mo.relu(%mm) : !mo.tensor<[M, N], f32>
  mo.output %relu : !mo.tensor<[M, N], f32>
}

// CHECK-LABEL: @checkBasicFusionBehavior
// CHECK: mogg.kernel
// CHECK: fusionInfo = Epilogue("mo.matmul":{{.*}}, "mo.relu":{{.*}})
// CHECK: mo.output
// CHECK-NOT: mogg.kernel{{[^.]}}

mo.graph @checkNoFuseInputInterfaceDisabled<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32> {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32>
  %relu = mo.relu(%mm) : !mo.tensor<[M, N], f32>
  mo.output %relu : !mo.tensor<[M, N], f32>
}

// CHECK-LABEL: @checkNoFuseInputInterfaceDisabled
// CHECK: mogg.kernel
// CHECK: fusionInfo = Epilogue("mo.matmul":{{.*}}, "mo.relu":{{.*}})
// CHECK: mo.output
// CHECK-NOT: mogg.kernel{{[^.]}}

mo.graph @checkNoFuseOutputInterfaceDisabled<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32> {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32>
  %relu = mo.relu(%mm) : !mo.tensor<[M, N], f32>
  mo.output %relu : !mo.tensor<[M, N], f32>
}

// CHECK-LABEL: @checkNoFuseOutputInterfaceDisabled
// CHECK: mogg.kernel
// CHECK: mo.output
// CHECK-NOT: mogg.kernel{{[^.]}}

mo.graph @checkNoFuseNoElementwiseAttr<N, M, K>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32> {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32>
  %relu = mo.relu(%mm) : !mo.tensor<[M, N], f32>
  mo.output %relu : !mo.tensor<[M, N], f32>
}

// CHECK-LABEL: @checkNoFuseNoElementwiseAttr
// CHECK: mogg.kernel
// CHECK: fusionInfo = Epilogue("mo.matmul":{{.*}}, "mo.relu":{{.*}})
// CHECK: mo.output
// CHECK-NOT: mogg.kernel{{[^.]}}

mo.graph @fused_mixed_precision_gpu_device(%arg0: !mo.tensor<[2, 2], f16>, %arg1: !mo.tensor<[2, 2], f16>) -> !mo.tensor<[2, 2], f32> {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[2, 2], f16>, !mo.tensor<[2, 2], f16>) -> !mo.tensor<[2, 2], f16>
  %cast = mo.cast(%mm) : (!mo.tensor<[2, 2], f16>) -> !mo.tensor<[2, 2], f32>
  mo.output %cast : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: @fused_mixed_precision_gpu_device
// CHECK: mogg.kernel
// CHECK: fusionInfo = Epilogue("mo.matmul":{{.*}}, "mo.cast":{{.*}})
// CHECK: mo.output
// CHECK-NOT: mogg.kernel{{[^.]}}

mo.graph @nofuse_mixed_precision_mixed_device(%arg0: !mo.tensor<[2, 2], f16>, %arg1: !mo.tensor<[2, 2], f16>) -> !mo.tensor<[2, 2], f32> {
  %mm = mo.matmul(%arg0, %arg1) : (!mo.tensor<[2, 2], f16>, !mo.tensor<[2, 2], f16>) -> !mo.tensor<[2, 2], f16>
  %cast = mo.cast(%mm) : (!mo.tensor<[2, 2], f16>) -> !mo.tensor<[2, 2], f32>
  mo.output %cast : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: @nofuse_mixed_precision_mixed_device
// CHECK: mogg.kernel
// CHECK: fusionInfo = Epilogue("mo.matmul":{{.*}}, "mo.cast":{{.*}})
// CHECK: mo.output
// CHECK-NOT: mogg.kernel{{[^.]}}
