// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @topk_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape -> Sh4: !mosh.ape, Sh5: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %k: !mo.tensor<Sh1, si64>, %axis: !mo.tensor<Sh2, si64>, %sorted: !mo.tensor<Sh3, bool>) -> (!mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>) {
  %values, %indices = rmo.mo.top_k<() -> Sh6: !mosh.ape, Sh7: !mosh.ape>(%input, %k, %axis, %sorted) : (
        !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, bool>
      ) -> (
        !mo.tensor<Sh6, f32>, !mo.tensor<Sh7, si64>
      )
  mo.output<:!mosh.ape Sh6, :!mosh.ape Sh7> %values, %indices : !mo.tensor<Sh6, f32>, !mo.tensor<Sh7, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @topk_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape -> Sh4: !mosh.ape, Sh5: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, si64>, %[[ARG2:.*]]: !mo.tensor<Sh2, si64>, %[[ARG3:.*]]: !mo.tensor<Sh3, bool>) -> (!mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>) {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH0:[A-Za-z0-9_]*]]: !mosh.ape>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[RMO_SH0]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<Sh2, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape(%[[ARG3]]) {{.*}} : (!mo.tensor<Sh3, bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   %[[VALUES:.*]], %[[INDICES:.*]] = mo.top_k<() -> [[RMO_SH1:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V0]], %[[V1]], %[[V2]], %[[V3]]) : (!mo.tensor<[[RMO_SH0]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[RMO_SH1]], f32>, !mo.tensor<[[RMO_SH1]], si64>)
//  CHECK-NEXT:   %[[V4:.*]] = rmo.rebind_tensor_shape<() -> [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>(%[[VALUES]]) {{.*}} : (!mo.tensor<[[RMO_SH1]], f32>) -> !mo.tensor<[[SH6]], f32>
//  CHECK-NEXT:   %[[V5:.*]] = rmo.rebind_tensor_shape<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>(%[[INDICES]]) {{.*}} : (!mo.tensor<[[RMO_SH1]], si64>) -> !mo.tensor<[[SH7]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH6]], :!mosh.ape [[SH7]]> %[[V4]], %[[V5]] : !mo.tensor<[[SH6]], f32>, !mo.tensor<[[SH7]], si64>

// -----

mo.graph @topk_dyn_params<D0, D1, Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> D2, D3, D4, D5>(%input: !mo.tensor<[D0, D1], f32>, %k: !mo.tensor<Sh0, si64>, %axis: !mo.tensor<Sh1, si64>, %sorted: !mo.tensor<Sh2, bool>) -> (!mo.tensor<[D2, D3], f32>, !mo.tensor<[D4, D5], si64>) {
  %values, %indices = rmo.mo.top_k<() -> TmpD0, TmpD1, TmpD2, TmpD3>(%input, %k, %axis, %sorted) : (
        !mo.tensor<[D0, D1], f32>, !mo.tensor<Sh0, si64>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, bool>
      ) -> (
        !mo.tensor<[TmpD0, TmpD1], f32>, !mo.tensor<[TmpD2, TmpD3], si64>
      )
  mo.output<TmpD0, TmpD1, TmpD2, TmpD3> %values, %indices : !mo.tensor<[TmpD0, TmpD1], f32>, !mo.tensor<[TmpD2, TmpD3], si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @topk_dyn_params<D0, D1, Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> D2, D3, D4, D5>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], f32>, %[[ARG1:.*]]: !mo.tensor<Sh0, si64>, %[[ARG2:.*]]: !mo.tensor<Sh1, si64>, %[[ARG3:.*]]: !mo.tensor<Sh2, bool>) -> (!mo.tensor<[D2, D3], f32>, !mo.tensor<[D4, D5], si64>) {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh0, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape(%[[ARG3]]) {{.*}} : (!mo.tensor<Sh2, bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   %[[VALUES:.*]], %[[INDICES:.*]] = mo.top_k<() -> [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]]>(%[[V0]], %[[V1]], %[[V2]], %[[V3]]) : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>, !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], si64>)
//  CHECK-NEXT:   %[[V4:.*]] = rmo.rebind_tensor_shape<() -> [[TmpD0:[A-Za-z0-9_]*]], [[TmpD1:[A-Za-z0-9_]*]]>(%[[VALUES]]) {{.*}} : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>) -> !mo.tensor<[[[TmpD0]], [[TmpD1]]], f32>
//  CHECK-NEXT:   %[[V5:.*]] = rmo.rebind_tensor_shape<() -> [[TmpD2:[A-Za-z0-9_]*]], [[TmpD3:[A-Za-z0-9_]*]]>(%[[INDICES]]) {{.*}} : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], si64>) -> !mo.tensor<[[[TmpD2]], [[TmpD3]]], si64>
//  CHECK-NEXT:   mo.output<[[TmpD0]], [[TmpD1]], [[TmpD2]], [[TmpD3]]> %[[V4]], %[[V5]] : !mo.tensor<[[[TmpD0]], [[TmpD1]]], f32>, !mo.tensor<[[[TmpD2]], [[TmpD3]]], si64>


// -----

mo.graph @topk_dyn_params_2<D0, D1, Sh0: !mosh.ape -> D2, D3, Sh1: !mosh.ape>(%input: !mo.tensor<[D0, D1], f32>, %k: !mo.tensor<Sh0, si64>, %axis: !mo.tensor<[], si64>, %sorted: !mo.tensor<[], bool>) -> (!mo.tensor<[D2, D3], f32>, !mo.tensor<Sh1, si64>) {
  %values, %indices = rmo.mo.top_k<() -> TmpD0, TmpD1, TmpSh0: !mosh.ape>(%input, %k, %axis, %sorted) : (
        !mo.tensor<[D0, D1], f32>, !mo.tensor<Sh0, si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[TmpD0, TmpD1], f32>, !mo.tensor<TmpSh0, si64>
      )
  mo.output<TmpD0, TmpD1, :!mosh.ape TmpSh0> %values, %indices : !mo.tensor<[TmpD0, TmpD1], f32>, !mo.tensor<TmpSh0, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @topk_dyn_params_2<D0, D1, Sh0: !mosh.ape -> D2, D3, Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], f32>, %[[ARG1:.*]]: !mo.tensor<Sh0, si64>, %[[ARG2:.*]]: !mo.tensor<[], si64>, %[[ARG3:.*]]: !mo.tensor<[], bool>) -> (!mo.tensor<[D2, D3], f32>, !mo.tensor<Sh1, si64>) {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh0, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[VALUES:.*]], %[[INDICES:.*]] = mo.top_k<() -> [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]]>(%[[V0]], %[[V1]], %[[ARG2]], %[[ARG3]]) : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>, !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], si64>)
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[TmpD0:[A-Za-z0-9_]*]], [[TmpD1:[A-Za-z0-9_]*]]>(%[[VALUES]]) {{.*}} : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>) -> !mo.tensor<[[[TmpD0]], [[TmpD1]]], f32>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape<() -> [[TmpSh0:[A-Za-z0-9_]*]]: !mosh.ape>(%[[INDICES]]) {{.*}} : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], si64>) -> !mo.tensor<[[TmpSh0]], si64>
//  CHECK-NEXT:   mo.output<[[TmpD0]], [[TmpD1]], :!mosh.ape [[TmpSh0]]> %[[V2]], %[[V3]] : !mo.tensor<[[[TmpD0]], [[TmpD1]]], f32>, !mo.tensor<[[TmpSh0]], si64>


// -----

mo.graph @topk_dyn_mix<D0, Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> D1, D2, D3>(%input: !mo.tensor<[D0, 3], f32>, %k: !mo.tensor<Sh0, si64>, %axis: !mo.tensor<Sh1, si64>, %sorted: !mo.tensor<Sh2, bool>) -> (!mo.tensor<[D1, 3], f32>, !mo.tensor<[D2, D3], si64>) {
  %values, %indices = rmo.mo.top_k<() -> TmpD0, TmpD1, TmpD2>(%input, %k, %axis, %sorted) : (
        !mo.tensor<[D0, 3], f32>, !mo.tensor<Sh0, si64>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, bool>
      ) -> (
        !mo.tensor<[TmpD0, 3], f32>, !mo.tensor<[TmpD1, TmpD2], si64>
      )
  mo.output<TmpD0, TmpD1, TmpD2> %values, %indices : !mo.tensor<[TmpD0, 3], f32>, !mo.tensor<[TmpD1, TmpD2], si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @topk_dyn_mix<D0, Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> D1, D2, D3>(%[[ARG0:.*]]: !mo.tensor<[D0, 3], f32>, %[[ARG1:.*]]: !mo.tensor<Sh0, si64>, %[[ARG2:.*]]: !mo.tensor<Sh1, si64>, %[[ARG3:.*]]: !mo.tensor<Sh2, bool>) -> (!mo.tensor<[D1, 3], f32>, !mo.tensor<[D2, D3], si64>) {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[D0, 3], f32>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh0, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape(%[[ARG3]]) {{.*}} : (!mo.tensor<Sh2, bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   %[[VALUES:.*]], %[[INDICES:.*]] = mo.top_k<() -> [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]]>(%[[V0]], %[[V1]], %[[V2]], %[[V3]]) : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>, !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], si64>)
//  CHECK-NEXT:   %[[V4:.*]] = rmo.rebind_tensor_shape<() -> [[TmpD0:[A-Za-z0-9_]*]]>(%[[VALUES]]) {{.*}} : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>) -> !mo.tensor<[[[TmpD0]], 3], f32>
//  CHECK-NEXT:   %[[V5:.*]] = rmo.rebind_tensor_shape<() -> [[TmpD1:[A-Za-z0-9_]*]], [[TmpD2:[A-Za-z0-9_]*]]>(%[[INDICES]]) {{.*}} : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], si64>) -> !mo.tensor<[[[TmpD1]], [[TmpD2]]], si64>
//  CHECK-NEXT:   mo.output<[[TmpD0]], [[TmpD1]], [[TmpD2]]> %[[V4]], %[[V5]] : !mo.tensor<[[[TmpD0]], 3], f32>, !mo.tensor<[[[TmpD1]], [[TmpD2]]], si64>

// -----

mo.graph @topk_static<() -> Sh0: !mosh.ape, Sh1: !mosh.ape>(%input: !mo.tensor<[4, 6], f32>, %k: !mo.tensor<[], si64>, %axis: !mo.tensor<[], si64>, %sorted: !mo.tensor<[], bool>) -> (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) {
  %values, %indices = rmo.mo.top_k<() -> TmpSh0: !mosh.ape, TmpSh1: !mosh.ape>(%input, %k, %axis, %sorted) : (
        !mo.tensor<[4, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<TmpSh0, f32>, !mo.tensor<TmpSh1, si64>
      )
  mo.output<:!mosh.ape TmpSh0, :!mosh.ape TmpSh1> %values, %indices : !mo.tensor<TmpSh0, f32>, !mo.tensor<TmpSh1, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @topk_static<() -> Sh0: !mosh.ape, Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[4, 6], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>, %[[ARG2:.*]]: !mo.tensor<[], si64>, %[[ARG3:.*]]: !mo.tensor<[], bool>) -> (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<[4, 6], f32>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>
//  CHECK-NEXT:   %[[VALUES:.*]], %[[INDICES:.*]] = mo.top_k<() -> [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]]>(%[[V0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]) : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>, !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], si64>)
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[TmpSh0:[A-Za-z0-9_]*]]: !mosh.ape>(%[[VALUES]]) {{.*}} : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>) -> !mo.tensor<[[TmpSh0]], f32>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape<() -> [[TmpSh1:[A-Za-z0-9_]*]]: !mosh.ape>(%[[INDICES]]) {{.*}} : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], si64>) -> !mo.tensor<[[TmpSh1]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[TmpSh0]], :!mosh.ape [[TmpSh1]]> %[[V1]], %[[V2]] : !mo.tensor<[[TmpSh0]], f32>, !mo.tensor<[[TmpSh1]], si64>

// -----

mo.graph @bottomk_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape -> Sh4: !mosh.ape, Sh5: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %k: !mo.tensor<Sh1, si64>, %axis: !mo.tensor<Sh2, si64>, %sorted: !mo.tensor<Sh3, bool>) -> (!mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>) {
  %values, %indices = rmo.mo.bottom_k<() -> Sh6: !mosh.ape, Sh7: !mosh.ape>(%input, %k, %axis, %sorted) : (
        !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, bool>
      ) -> (
        !mo.tensor<Sh6, f32>, !mo.tensor<Sh7, si64>
      )
  mo.output<:!mosh.ape Sh6, :!mosh.ape Sh7> %values, %indices : !mo.tensor<Sh6, f32>, !mo.tensor<Sh7, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @bottomk_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape -> Sh4: !mosh.ape, Sh5: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, si64>, %[[ARG2:.*]]: !mo.tensor<Sh2, si64>, %[[ARG3:.*]]: !mo.tensor<Sh3, bool>) -> (!mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>) {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH0:[A-Za-z0-9_]*]]: !mosh.ape>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[RMO_SH0]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<Sh2, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape(%[[ARG3]]) {{.*}} : (!mo.tensor<Sh3, bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   %[[VALUES:.*]], %[[INDICES:.*]] = mo.bottom_k<() -> [[RMO_SH1:[A-Za-z0-9_]*]]: !mosh.ape>(%[[V0]], %[[V1]], %[[V2]], %[[V3]]) : (!mo.tensor<[[RMO_SH0]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[RMO_SH1]], f32>, !mo.tensor<[[RMO_SH1]], si64>)
//  CHECK-NEXT:   %[[V4:.*]] = rmo.rebind_tensor_shape<() -> [[TmpSh6:[A-Za-z0-9_]*]]: !mosh.ape>(%[[VALUES]]) {{.*}} : (!mo.tensor<[[RMO_SH1]], f32>) -> !mo.tensor<[[TmpSh6]], f32>
//  CHECK-NEXT:   %[[V5:.*]] = rmo.rebind_tensor_shape<() -> [[TmpSh7:[A-Za-z0-9_]*]]: !mosh.ape>(%[[INDICES]]) {{.*}} : (!mo.tensor<[[RMO_SH1]], si64>) -> !mo.tensor<[[TmpSh7]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[TmpSh6]], :!mosh.ape [[TmpSh7]]> %[[V4]], %[[V5]] : !mo.tensor<[[TmpSh6]], f32>, !mo.tensor<[[TmpSh7]], si64>

// -----

mo.graph @bottomk_dyn_out_param<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape -> Sh4: !mosh.ape, D0, D1>(%input: !mo.tensor<Sh0, f32>, %k: !mo.tensor<Sh1, si64>, %axis: !mo.tensor<Sh2, si64>, %sorted: !mo.tensor<Sh3, bool>) -> (!mo.tensor<Sh4, f32>, !mo.tensor<[D0, D1], si64>) {
  %values, %indices = rmo.mo.bottom_k<() -> TmpSh0: !mosh.ape, TmpD0, TmpD1>(%input, %k, %axis, %sorted) : (
        !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, bool>
      ) -> (
        !mo.tensor<TmpSh0, f32>, !mo.tensor<[TmpD0, TmpD1], si64>
      )
  mo.output<:!mosh.ape TmpSh0, TmpD0, TmpD1> %values, %indices : !mo.tensor<TmpSh0, f32>, !mo.tensor<[TmpD0, TmpD1], si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @bottomk_dyn_out_param<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape -> Sh4: !mosh.ape, D0, D1>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>, %[[ARG1:.*]]: !mo.tensor<Sh1, si64>, %[[ARG2:.*]]: !mo.tensor<Sh2, si64>, %[[ARG3:.*]]: !mo.tensor<Sh3, bool>) -> (!mo.tensor<Sh4, f32>, !mo.tensor<[D0, D1], si64>) {
//  CHECK-NEXT:   %[[V0:.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D0:[A-Za-z0-9_]*]], [[RMO_D1:[A-Za-z0-9_]*]]>(%[[ARG0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape(%[[ARG1]]) {{.*}} : (!mo.tensor<Sh1, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V2:.*]] = rmo.rebind_tensor_shape(%[[ARG2]]) {{.*}} : (!mo.tensor<Sh2, si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   %[[V3:.*]] = rmo.rebind_tensor_shape(%[[ARG3]]) {{.*}} : (!mo.tensor<Sh3, bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   %[[VALUES:.*]], %[[INDICES:.*]] = mo.bottom_k<() -> [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]]>(%[[V0]], %[[V1]], %[[V2]], %[[V3]]) : (!mo.tensor<[[[RMO_D0]], [[RMO_D1]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>, !mo.tensor<[[[RMO_D2]], [[RMO_D3]]], si64>)
//  CHECK-NEXT:   %[[V4:.*]] = rmo.rebind_tensor_shape<() -> [[TmpSh0:[A-Za-z0-9_]*]]: !mosh.ape>(%[[VALUES]]) {{.*}} : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], f32>) -> !mo.tensor<[[TmpSh0]], f32>
//  CHECK-NEXT:   %[[V5:.*]] = rmo.rebind_tensor_shape<() -> [[TmpD0:[A-Za-z0-9_]*]], [[TmpD1:[A-Za-z0-9_]*]]>(%[[INDICES]]) {{.*}} : (!mo.tensor<[[[RMO_D2]], [[RMO_D3]]], si64>) -> !mo.tensor<[[[TmpD0]], [[TmpD1]]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[TmpSh0]], [[TmpD0]], [[TmpD1]]> %[[V4]], %[[V5]] : !mo.tensor<[[TmpSh0]], f32>, !mo.tensor<[[[TmpD0]], [[TmpD1]]], si64>
