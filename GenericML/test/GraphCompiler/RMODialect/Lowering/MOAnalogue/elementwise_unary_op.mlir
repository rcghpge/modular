// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @abs_dyn_dyn<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, f32> {
  %1 = rmo.mo.abs<() -> Sh2: !mosh.ape>(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @abs_dyn_dyn<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.abs(%[[ARG0]]) : !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH2:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> %[[V1]] : !mo.tensor<[[SH2]], f32>

// -----

mo.graph @ceil_dyn_params<Sh0: !mosh.ape -> D0, D1>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<[D0, D1], f32> {
  %1 = rmo.mo.ceil<() -> D2, D3>(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %1 : !mo.tensor<[D2, D3], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @ceil_dyn_params<Sh0: !mosh.ape -> D0, D1>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>) -> !mo.tensor<[D0, D1], f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.ceil(%[[ARG0]]) : !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[D2:.*]], [[D3:.*]]>(%[[V0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[[D2]], [[D3]]], f32>
// CHECK-NEXT: mo.output<[[D2]], [[D3]]> %[[V1]] : !mo.tensor<[[[D2]], [[D3]]], f32>

// -----

mo.graph @cos_dyn_mixed<Sh0: !mosh.ape -> D0>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<[D0, 10], f32> {
  %1 = rmo.mo.cos<() -> D1>(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[D1, 10], f32>
  mo.output<D1> %1 : !mo.tensor<[D1, 10], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @cos_dyn_mixed<Sh0: !mosh.ape -> D0>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>) -> !mo.tensor<[D0, 10], f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.cos(%[[ARG0]]) : !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[D1:.*]]>(%[[V0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[[[D1]], 10], f32>
// CHECK-NEXT: mo.output<[[D1]]> %[[V1]] : !mo.tensor<[[[D1]], 10], f32>

// -----

mo.graph @erf_dyn_static<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<[20, 10], f32> {
  %1 = rmo.mo.erf(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[20, 10], f32>
  mo.output %1 : !mo.tensor<[20, 10], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @erf_dyn_static<Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<Sh0, f32>) -> !mo.tensor<[20, 10], f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.erf(%[[ARG0]]) : !mo.tensor<Sh0, f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape(%[[V0]]) {{.*}} : (!mo.tensor<Sh0, f32>) -> !mo.tensor<[20, 10], f32>
// CHECK-NEXT: mo.output %[[V1]] : !mo.tensor<[20, 10], f32>

// -----

mo.graph @exp_params_dyn<D0, D1 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.exp<() -> Sh1: !mosh.ape>(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @exp_params_dyn<D0, D1 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<Sh0, f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.exp(%[[ARG0]]) : !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @floor_params_params<D0, D1 -> D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, D3], f32> {
  %1 = rmo.mo.floor<() -> D4, D5>(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D4, D5], f32>
  mo.output<D4, D5> %1 : !mo.tensor<[D4, D5], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @floor_params_params<D0, D1 -> D2, D3>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, D3], f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.floor(%[[ARG0]]) : !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[D4:.*]], [[D5:.*]]>(%[[V0]]) {{.*}} : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[[[D4]], [[D5]]], f32>
// CHECK-NEXT: mo.output<[[D4]], [[D5]]> %[[V1]] : !mo.tensor<[[[D4]], [[D5]]], f32>

// -----

mo.graph @log_params_static<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[20, 10], f32> {
  %1 = rmo.mo.log(%arg0) : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[20, 10], f32>
  mo.output %1 : !mo.tensor<[20, 10], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @log_params_static<D0, D1>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[20, 10], f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.log(%[[ARG0]]) : !mo.tensor<[D0, D1], f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape(%[[V0]]) {{.*}} : (!mo.tensor<[D0, D1], f32>) -> !mo.tensor<[20, 10], f32>
// CHECK-NEXT: mo.output %[[V1]] : !mo.tensor<[20, 10], f32>

// -----

mo.graph @log1p_mixed_dyn<D0 -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[20, D0], f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.log1p<() -> Sh1: !mosh.ape>(%arg0) : (!mo.tensor<[20, D0], f32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @log1p_mixed_dyn<D0 -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[20, D0], f32>) -> !mo.tensor<Sh0, f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.log1p(%[[ARG0]]) : !mo.tensor<[20, D0], f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[20, D0], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @logsoftmax_mixed_params<D0 -> D1, D2>(%arg0: !mo.tensor<[20, D0], f32>, %arg1: !mo.tensor<[], si64>) -> !mo.tensor<[D1, D2], f32> {
  %1 = rmo.mo.logsoftmax<() -> D3, D4>(%arg0, %arg1) : (!mo.tensor<[20, D0], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[D3, D4], f32>
  mo.output<D3, D4> %1 : !mo.tensor<[D3, D4], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @logsoftmax_mixed_params<D0 -> D1, D2>(%[[ARG0:.*]]: !mo.tensor<[20, D0], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[D1, D2], f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.logsoftmax(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[20, D0], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[20, D0], f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[D3:.*]], [[D4:.*]]>(%[[V0]]) {{.*}} : (!mo.tensor<[20, D0], f32>) -> !mo.tensor<[[[D3]], [[D4]]], f32>
// CHECK-NEXT: mo.output<[[D3]], [[D4]]> %[[V1]] : !mo.tensor<[[[D3]], [[D4]]], f32>

// -----

mo.graph @negative_mixed_mixed<D0 -> D1>(%arg0: !mo.tensor<[20, D0], f32>) -> !mo.tensor<[D1, 10], f32> {
  %1 = rmo.mo.negative<() -> D2>(%arg0) : (!mo.tensor<[20, D0], f32>) -> !mo.tensor<[D2, 10], f32>
  mo.output<D2> %1 : !mo.tensor<[D2, 10], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @negative_mixed_mixed<D0 -> D1>(%[[ARG0:.*]]: !mo.tensor<[20, D0], f32>) -> !mo.tensor<[D1, 10], f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.negative(%[[ARG0]]) : !mo.tensor<[20, D0], f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[D2:.*]]>(%[[V0]]) {{.*}} : (!mo.tensor<[20, D0], f32>) -> !mo.tensor<[[[D2]], 10], f32>
// CHECK-NEXT: mo.output<[[D2]]> %[[V1]] : !mo.tensor<[[[D2]], 10], f32>

// -----

mo.graph @relu_mixed_static<D0>(%arg0: !mo.tensor<[20, D0], f32>) -> !mo.tensor<[20, 10], f32> {
  %1 = rmo.mo.relu(%arg0) : (!mo.tensor<[20, D0], f32>) -> !mo.tensor<[20, 10], f32>
  mo.output %1 : !mo.tensor<[20, 10], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @relu_mixed_static<D0>(%[[ARG0:.*]]: !mo.tensor<[20, D0], f32>) -> !mo.tensor<[20, 10], f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.relu(%[[ARG0]]) : !mo.tensor<[20, D0], f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape(%[[V0]]) {{.*}} : (!mo.tensor<[20, D0], f32>) -> !mo.tensor<[20, 10], f32>
// CHECK-NEXT: mo.output %[[V1]] : !mo.tensor<[20, 10], f32>

// -----

mo.graph @round_static_dyn<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[20, 30], f32>) -> !mo.tensor<Sh0, f32> {
  %1 = rmo.mo.round<() -> Sh1: !mosh.ape>(%arg0) : (!mo.tensor<[20, 30], f32>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @round_static_dyn<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[20, 30], f32>) -> !mo.tensor<Sh0, f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.round(%[[ARG0]]) : !mo.tensor<[20, 30], f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[20, 30], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], f32>

// -----

mo.graph @rsqrt_static_mixed<() -> D0>(%arg0: !mo.tensor<[20, 30], f32>) -> !mo.tensor<[D0, 10], f32> {
  %1 = rmo.mo.rsqrt<() -> D1>(%arg0) : (!mo.tensor<[20, 30], f32>) -> !mo.tensor<[D1, 10], f32>
  mo.output<D1> %1 : !mo.tensor<[D1, 10], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @rsqrt_static_mixed<() -> D0>(%[[ARG0:.*]]: !mo.tensor<[20, 30], f32>) -> !mo.tensor<[D0, 10], f32> {
// CHECK-NEXT: %[[V0:.*]] = mo.rsqrt(%[[ARG0]]) : !mo.tensor<[20, 30], f32>
// CHECK-NEXT: %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[D1:.*]]>(%[[V0]]) {{.*}} : (!mo.tensor<[20, 30], f32>) -> !mo.tensor<[[[D1]], 10], f32>
// CHECK-NEXT: mo.output<[[D1]]> %[[V1]] : !mo.tensor<[[[D1]], 10], f32>
