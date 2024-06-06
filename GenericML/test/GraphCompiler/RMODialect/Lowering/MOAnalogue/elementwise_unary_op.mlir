// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @abs_dyn_dyn(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.abs(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @abs_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.abs([[V0]]) : !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH4:[^ ]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[SH3]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: }

// -----

mo.graph @ceil_dyn_params(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<[?, ?], f32> {
  %1 = rmo.mo.ceil(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %1 : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @ceil_dyn_params<[[SH1:[^ ]*]]: !mosh.ape -> [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[[D2]], [[D3]]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.ceil([[V0]]) : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D5:[^ ]*]], [[D6:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[[D5]], [[D6]]], f32>
// CHECK-NEXT: mo.output<[[D5]], [[D6]]> [[V2]] : !mo.tensor<[[[D5]], [[D6]]], f32>
// CHECK-NEXT: }

// -----

mo.graph @cos_dyn_mixed(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<[?, 10], f32> {
  %1 = rmo.mo.cos(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<[?, 10], f32>
  mo.output %1 : !mo.tensor<[?, 10], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @cos_dyn_mixed<[[SH1:[^ ]*]]: !mosh.ape -> [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[[D2]], 10], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.cos([[V0]]) : !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D4:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[SH3]], f32>) -> !mo.tensor<[[[D4]], 10], f32>
// CHECK-NEXT: mo.output<[[D4]]> [[V2]] : !mo.tensor<[[[D4]], 10], f32>
// CHECK-NEXT: }

// -----

mo.graph @erf_dyn_static(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<[20, 10], f32> {
  %1 = rmo.mo.erf(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<[20, 10], f32>
  mo.output %1 : !mo.tensor<[20, 10], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @erf_dyn_static<[[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[20, 10], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.erf([[V0]]) : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH2]], f32>) -> !mo.tensor<[20, 10], f32>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[20, 10], f32>
// CHECK-NEXT: }

// -----

mo.graph @exp_params_dyn(%arg0: !mo.tensor<[?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.exp(%arg0) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @exp_params_dyn<[[D1:[^ ]*]], [[D2:[^ ]*]] -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]]], f32>) -> !mo.tensor<[[SH3]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]], [[D5:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D4]], [[D5]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.exp([[V0]]) : !mo.tensor<[[[D4]], [[D5]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[[[D4]], [[D5]]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V2]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }

// -----

mo.graph @floor_params_params(%arg0: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32> {
  %1 = rmo.mo.floor(%arg0) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %1 : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @floor_params_params<[[D1:[^ ]*]], [[D2:[^ ]*]] -> [[D3:[^ ]*]], [[D4:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]]], f32>) -> !mo.tensor<[[[D3]], [[D4]]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]], [[D6:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D5]], [[D6]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.floor([[V0]]) : !mo.tensor<[[[D5]], [[D6]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D7:[^ ]*]], [[D8:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D5]], [[D6]]], f32>) -> !mo.tensor<[[[D7]], [[D8]]], f32>
// CHECK-NEXT: mo.output<[[D7]], [[D8]]> [[V2]] : !mo.tensor<[[[D7]], [[D8]]], f32>
// CHECK-NEXT: }

// -----

mo.graph @gelu_params_mixed(%arg0: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, 10], f32> {
  %1 = rmo.mo.gelu(%arg0) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, 10], f32>
  mo.output %1 : !mo.tensor<[?, 10], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @gelu_params_mixed<[[D1:[^ ]*]], [[D2:[^ ]*]] -> [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]]], f32>) -> !mo.tensor<[[[D3]], 10], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]], [[D5:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D4]], [[D5]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.gelu([[V0]]) : !mo.tensor<[[[D4]], [[D5]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D6:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[[[D4]], [[D5]]], f32>) -> !mo.tensor<[[[D6]], 10], f32>
// CHECK-NEXT: mo.output<[[D6]]> [[V2]] : !mo.tensor<[[[D6]], 10], f32>
// CHECK-NEXT: }

// -----

mo.graph @log_params_static(%arg0: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[20, 10], f32> {
  %1 = rmo.mo.log(%arg0) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<[20, 10], f32>
  mo.output %1 : !mo.tensor<[20, 10], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @log_params_static<[[D1:[^ ]*]], [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]]], f32>) -> !mo.tensor<[20, 10], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D3:[^ ]*]], [[D4:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D3]], [[D4]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.log([[V0]]) : !mo.tensor<[[[D3]], [[D4]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D3]], [[D4]]], f32>) -> !mo.tensor<[20, 10], f32>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[20, 10], f32>
// CHECK-NEXT: }

// -----

mo.graph @log1p_mixed_dyn(%arg0: !mo.tensor<[20, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.log1p(%arg0) : (!mo.tensor<[20, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @log1p_mixed_dyn<[[D1:[^ ]*]] -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[20, [[D1]]], f32>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[20, [[D1]]], f32> -> !mo.tensor<[20, [[D3]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.log1p([[V0]]) : !mo.tensor<[20, [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH4:[^ ]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[20, [[D3]]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V2]] : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: }

// -----

mo.graph @logsoftmax_mixed_params(%arg0: !mo.tensor<[20, ?], f32>) -> !mo.tensor<[?, ?], f32> {
  %1 = rmo.mo.logsoftmax(%arg0) : (!mo.tensor<[20, ?], f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %1 : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @logsoftmax_mixed_params<[[D1:[^ ]*]] -> [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[20, [[D1]]], f32>) -> !mo.tensor<[[[D2]], [[D3]]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]]>([[VARG0]]) : !mo.tensor<[20, [[D1]]], f32> -> !mo.tensor<[20, [[D4]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.logsoftmax([[V0]]) : (!mo.tensor<[20, [[D4]]], f32>) -> !mo.tensor<[20, [[D4]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D5:[^ ]*]], [[D6:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[20, [[D4]]], f32>) -> !mo.tensor<[[[D5]], [[D6]]], f32>
// CHECK-NEXT: mo.output<[[D5]], [[D6]]> [[V2]] : !mo.tensor<[[[D5]], [[D6]]], f32>
// CHECK-NEXT: }

// -----

mo.graph @negative_mixed_mixed(%arg0: !mo.tensor<[20, ?], f32>) -> !mo.tensor<[?, 10], f32> {
  %1 = rmo.mo.negative(%arg0) : (!mo.tensor<[20, ?], f32>) -> !mo.tensor<[?, 10], f32>
  mo.output %1 : !mo.tensor<[?, 10], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @negative_mixed_mixed<[[D1:[^ ]*]] -> [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[20, [[D1]]], f32>) -> !mo.tensor<[[[D2]], 10], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[20, [[D1]]], f32> -> !mo.tensor<[20, [[D3]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.negative([[V0]]) : !mo.tensor<[20, [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D4:[^ ]*]]>([[V1]]) {{.*}} : (!mo.tensor<[20, [[D3]]], f32>) -> !mo.tensor<[[[D4]], 10], f32>
// CHECK-NEXT: mo.output<[[D4]]> [[V2]] : !mo.tensor<[[[D4]], 10], f32>
// CHECK-NEXT: }

// -----

mo.graph @relu_mixed_static(%arg0: !mo.tensor<[20, ?], f32>) -> !mo.tensor<[20, 10], f32> {
  %1 = rmo.mo.relu(%arg0) : (!mo.tensor<[20, ?], f32>) -> !mo.tensor<[20, 10], f32>
  mo.output %1 : !mo.tensor<[20, 10], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @relu_mixed_static<[[D1:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[20, [[D1]]], f32>) -> !mo.tensor<[20, 10], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D2:[^ ]*]]>([[VARG0]]) : !mo.tensor<[20, [[D1]]], f32> -> !mo.tensor<[20, [[D2]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.relu([[V0]]) : !mo.tensor<[20, [[D2]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[20, [[D2]]], f32>) -> !mo.tensor<[20, 10], f32>
// CHECK-NEXT: mo.output [[V2]] : !mo.tensor<[20, 10], f32>
// CHECK-NEXT: }

// -----

mo.graph @round_static_dyn(%arg0: !mo.tensor<[20, 30], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.round(%arg0) : (!mo.tensor<[20, 30], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @round_static_dyn<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[20, 30], f32>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.round([[VARG0]]) : !mo.tensor<[20, 30], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[20, 30], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }

// -----

mo.graph @roundeven_static_params(%arg0: !mo.tensor<[20, 30], f32>) -> !mo.tensor<[?, ?], f32> {
  %1 = rmo.mo.roundeven(%arg0) : (!mo.tensor<[20, 30], f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %1 : !mo.tensor<[?, ?], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @roundeven_static_params<() -> [[D1:[^ ]*]], [[D2:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[20, 30], f32>) -> !mo.tensor<[[[D1]], [[D2]]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.roundeven([[VARG0]]) : !mo.tensor<[20, 30], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D3:[^ ]*]], [[D4:[^ ]*]]>([[V0]]) {{.*}} : (!mo.tensor<[20, 30], f32>) -> !mo.tensor<[[[D3]], [[D4]]], f32>
// CHECK-NEXT: mo.output<[[D3]], [[D4]]> [[V1]] : !mo.tensor<[[[D3]], [[D4]]], f32>
// CHECK-NEXT: }

// -----

mo.graph @rsqrt_static_mixed(%arg0: !mo.tensor<[20, 30], f32>) -> !mo.tensor<[?, 10], f32> {
  %1 = rmo.mo.rsqrt(%arg0) : (!mo.tensor<[20, 30], f32>) -> !mo.tensor<[?, 10], f32>
  mo.output %1 : !mo.tensor<[?, 10], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @rsqrt_static_mixed<() -> [[D1:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[20, 30], f32>) -> !mo.tensor<[[[D1]], 10], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rsqrt([[VARG0]]) : !mo.tensor<[20, 30], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D2:[^ ]*]]>([[V0]]) {{.*}} : (!mo.tensor<[20, 30], f32>) -> !mo.tensor<[[[D2]], 10], f32>
// CHECK-NEXT: mo.output<[[D2]]> [[V1]] : !mo.tensor<[[[D2]], 10], f32>
// CHECK-NEXT: }
