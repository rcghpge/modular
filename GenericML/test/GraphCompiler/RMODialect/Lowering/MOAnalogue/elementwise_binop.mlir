// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @add_dyn_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.add(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.add([[V0]], [[V2]]) : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @add_dyn_params(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.add(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_dyn_params<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]] -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32>) -> !mo.tensor<[[SH5]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.add([[V2]], [[V1]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH10:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[SH10]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH10]]> [[V4]] : !mo.tensor<[[SH10]], f32>
// -----

mo.graph @add_params_params(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.add(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_params_params<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]], [[D6:[^ ]*]] -> [[SH7:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[SH7]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D11:[^ ]*]], [[D12:[^ ]*]], [[D13:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32> -> !mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.add([[V0]], [[V2]]) : !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH14:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>) -> !mo.tensor<[[SH14]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH14]]> [[V4]] : !mo.tensor<[[SH14]], f32>

// -----

mo.graph @add_params_mix(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.add(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_params_mix<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]] -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], 2], f32>) -> !mo.tensor<[[SH6]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D10:[^ ]*]], [[D11:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], 2], f32> -> !mo.tensor<[[[D10]], [[D11]], 2], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D10]], [[D11]], 2], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.add([[V2]], [[V3]]) : !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH12:[^ ]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], 2], f32>) -> !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH12]]> [[V5]] : !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: }

// -----

mo.graph @add_static_static(%arg0: !mo.tensor<[1, 2, 3], f32>, %arg1: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.add(%arg0, %arg1) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_static_static<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.add([[VARG0]], [[VARG1]]) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }

// -----

mo.graph @div_dyn_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.div(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @div_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.div([[V0]], [[V2]]) : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @div_dyn_params(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.div(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @div_dyn_params<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]] -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32>) -> !mo.tensor<[[SH5]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.div([[V2]], [[V1]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH10:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[SH10]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH10]]> [[V4]] : !mo.tensor<[[SH10]], f32>
// -----

mo.graph @div_params_params(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.div(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @div_params_params<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]], [[D6:[^ ]*]] -> [[SH7:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[SH7]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D11:[^ ]*]], [[D12:[^ ]*]], [[D13:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32> -> !mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.div([[V0]], [[V2]]) : !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH14:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>) -> !mo.tensor<[[SH14]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH14]]> [[V4]] : !mo.tensor<[[SH14]], f32>

// -----

mo.graph @div_params_mix(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.div(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @div_params_mix<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]] -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], 2], f32>) -> !mo.tensor<[[SH6]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D10:[^ ]*]], [[D11:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], 2], f32> -> !mo.tensor<[[[D10]], [[D11]], 2], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D10]], [[D11]], 2], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.div([[V2]], [[V3]]) : !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH12:[^ ]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], 2], f32>) -> !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH12]]> [[V5]] : !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: }

// -----

mo.graph @div_static_static(%arg0: !mo.tensor<[1, 2, 3], f32>, %arg1: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.div(%arg0, %arg1) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @div_static_static<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.div([[VARG0]], [[VARG1]]) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }

// -----

mo.graph @max_dyn_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.max(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @max_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.max([[V0]], [[V2]]) : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @max_dyn_params(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.max(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @max_dyn_params<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]] -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32>) -> !mo.tensor<[[SH5]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.max([[V2]], [[V1]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH10:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[SH10]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH10]]> [[V4]] : !mo.tensor<[[SH10]], f32>
// -----

mo.graph @max_params_params(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.max(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @max_params_params<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]], [[D6:[^ ]*]] -> [[SH7:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[SH7]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D11:[^ ]*]], [[D12:[^ ]*]], [[D13:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32> -> !mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.max([[V0]], [[V2]]) : !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH14:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>) -> !mo.tensor<[[SH14]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH14]]> [[V4]] : !mo.tensor<[[SH14]], f32>

// -----

mo.graph @max_params_mix(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.max(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @max_params_mix<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]] -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], 2], f32>) -> !mo.tensor<[[SH6]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D10:[^ ]*]], [[D11:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], 2], f32> -> !mo.tensor<[[[D10]], [[D11]], 2], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D10]], [[D11]], 2], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.max([[V2]], [[V3]]) : !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH12:[^ ]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], 2], f32>) -> !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH12]]> [[V5]] : !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: }

// -----

mo.graph @max_static_static(%arg0: !mo.tensor<[1, 2, 3], f32>, %arg1: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.max(%arg0, %arg1) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @max_static_static<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.max([[VARG0]], [[VARG1]]) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }

// -----

mo.graph @min_dyn_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.min(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @min_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.min([[V0]], [[V2]]) : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @min_dyn_params(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.min(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @min_dyn_params<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]] -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32>) -> !mo.tensor<[[SH5]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.min([[V2]], [[V1]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH10:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[SH10]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH10]]> [[V4]] : !mo.tensor<[[SH10]], f32>
// -----

mo.graph @min_params_params(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.min(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @min_params_params<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]], [[D6:[^ ]*]] -> [[SH7:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[SH7]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D11:[^ ]*]], [[D12:[^ ]*]], [[D13:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32> -> !mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.min([[V0]], [[V2]]) : !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH14:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>) -> !mo.tensor<[[SH14]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH14]]> [[V4]] : !mo.tensor<[[SH14]], f32>

// -----

mo.graph @min_params_mix(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.min(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @min_params_mix<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]] -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], 2], f32>) -> !mo.tensor<[[SH6]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D10:[^ ]*]], [[D11:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], 2], f32> -> !mo.tensor<[[[D10]], [[D11]], 2], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D10]], [[D11]], 2], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.min([[V2]], [[V3]]) : !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH12:[^ ]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], 2], f32>) -> !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH12]]> [[V5]] : !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: }

// -----

mo.graph @min_static_static(%arg0: !mo.tensor<[1, 2, 3], f32>, %arg1: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.min(%arg0, %arg1) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @min_static_static<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.min([[VARG0]], [[VARG1]]) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }

// -----

mo.graph @mod_dyn_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.mod(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @mod_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.mod([[V0]], [[V2]]) : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @mod_dyn_params(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.mod(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @mod_dyn_params<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]] -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32>) -> !mo.tensor<[[SH5]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.mod([[V2]], [[V1]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH10:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[SH10]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH10]]> [[V4]] : !mo.tensor<[[SH10]], f32>
// -----

mo.graph @mod_params_params(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.mod(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @mod_params_params<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]], [[D6:[^ ]*]] -> [[SH7:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[SH7]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D11:[^ ]*]], [[D12:[^ ]*]], [[D13:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32> -> !mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.mod([[V0]], [[V2]]) : !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH14:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>) -> !mo.tensor<[[SH14]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH14]]> [[V4]] : !mo.tensor<[[SH14]], f32>

// -----

mo.graph @mod_params_mix(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.mod(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @mod_params_mix<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]] -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], 2], f32>) -> !mo.tensor<[[SH6]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D10:[^ ]*]], [[D11:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], 2], f32> -> !mo.tensor<[[[D10]], [[D11]], 2], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D10]], [[D11]], 2], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.mod([[V2]], [[V3]]) : !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH12:[^ ]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], 2], f32>) -> !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH12]]> [[V5]] : !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: }

// -----

mo.graph @mod_static_static(%arg0: !mo.tensor<[1, 2, 3], f32>, %arg1: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.mod(%arg0, %arg1) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @mod_static_static<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.mod([[VARG0]], [[VARG1]]) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }

// -----

mo.graph @mul_dyn_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.mul(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @mul_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.mul([[V0]], [[V2]]) : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @mul_dyn_params(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.mul(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @mul_dyn_params<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]] -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32>) -> !mo.tensor<[[SH5]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.mul([[V2]], [[V1]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH10:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[SH10]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH10]]> [[V4]] : !mo.tensor<[[SH10]], f32>
// -----

mo.graph @mul_params_params(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.mul(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @mul_params_params<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]], [[D6:[^ ]*]] -> [[SH7:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[SH7]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D11:[^ ]*]], [[D12:[^ ]*]], [[D13:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32> -> !mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.mul([[V0]], [[V2]]) : !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH14:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>) -> !mo.tensor<[[SH14]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH14]]> [[V4]] : !mo.tensor<[[SH14]], f32>

// -----

mo.graph @mul_params_mix(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.mul(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @mul_params_mix<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]] -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], 2], f32>) -> !mo.tensor<[[SH6]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D10:[^ ]*]], [[D11:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], 2], f32> -> !mo.tensor<[[[D10]], [[D11]], 2], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D10]], [[D11]], 2], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.mul([[V2]], [[V3]]) : !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH12:[^ ]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], 2], f32>) -> !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH12]]> [[V5]] : !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: }

// -----

mo.graph @mul_static_static(%arg0: !mo.tensor<[1, 2, 3], f32>, %arg1: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.mul(%arg0, %arg1) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @mul_static_static<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.mul([[VARG0]], [[VARG1]]) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }

// -----

mo.graph @pow_dyn_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.pow(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pow_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.pow([[V0]] {{.*}}, [[V2]] {{.*}}) : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @pow_dyn_params(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.pow(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pow_dyn_params<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]] -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32>) -> !mo.tensor<[[SH5]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.pow([[V2]] {{.*}}, [[V1]] {{.*}}) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH10:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[SH10]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH10]]> [[V4]] : !mo.tensor<[[SH10]], f32>
// -----

mo.graph @pow_params_params(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.pow(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pow_params_params<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]], [[D6:[^ ]*]] -> [[SH7:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[SH7]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D11:[^ ]*]], [[D12:[^ ]*]], [[D13:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32> -> !mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.pow([[V0]] {{.*}}, [[V2]] {{.*}}) : !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH14:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>) -> !mo.tensor<[[SH14]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH14]]> [[V4]] : !mo.tensor<[[SH14]], f32>

// -----

mo.graph @pow_params_mix(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.pow(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pow_params_mix<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]] -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], 2], f32>) -> !mo.tensor<[[SH6]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D10:[^ ]*]], [[D11:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], 2], f32> -> !mo.tensor<[[[D10]], [[D11]], 2], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D10]], [[D11]], 2], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.pow([[V2]] {{.*}}, [[V3]] {{.*}}) : !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH12:[^ ]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], 2], f32>) -> !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH12]]> [[V5]] : !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: }

// -----

mo.graph @pow_static_static(%arg0: !mo.tensor<[1, 2, 3], f32>, %arg1: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.pow(%arg0, %arg1) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pow_static_static<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.pow([[VARG0]] {{.*}}, [[VARG1]] {{.*}}) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }

// -----

mo.graph @pow_static_static_mixed_precision(%arg0: !mo.tensor<[1, 2, 3], f32>, %arg1: !mo.tensor<[1, 2, 3], si32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.pow(%arg0, %arg1) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], si32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pow_static_static_mixed_precision<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 2, 3], si32>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.pow([[VARG0]] {{.*}}, [[VARG1]] {{.*}}) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }

// -----

mo.graph @sub_dyn_dyn(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.sub(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sub_dyn_dyn<[[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH2]], f32>) -> !mo.tensor<[[SH3]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH5]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.sub([[V0]], [[V2]]) : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[SH4]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V4]] : !mo.tensor<[[SH6]], f32>

// -----

mo.graph @sub_dyn_params(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.sub(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sub_dyn_params<[[SH1:[^ ]*]]: !mosh.ape, [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]] -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32>) -> !mo.tensor<[[SH5]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[SH6]], f32>) -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.sub([[V2]], [[V1]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH10:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[SH10]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH10]]> [[V4]] : !mo.tensor<[[SH10]], f32>
// -----

mo.graph @sub_params_params(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.sub(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sub_params_params<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]], [[D6:[^ ]*]] -> [[SH7:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[SH7]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D11:[^ ]*]], [[D12:[^ ]*]], [[D13:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32> -> !mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D11]], [[D12]], [[D13]]], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.sub([[V0]], [[V2]]) : !mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH14:[^ ]*]]: !mosh.ape>([[V3]]) {{.*}} : (!mo.tensor<[[[D8]], [[D9]], [[D10]]], f32>) -> !mo.tensor<[[SH14]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH14]]> [[V4]] : !mo.tensor<[[SH14]], f32>

// -----

mo.graph @sub_params_mix(%arg0: !mo.tensor<[?, ?, ?], f32>, %arg1: !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.sub(%arg0, %arg1) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, 2], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sub_params_mix<[[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]], [[D4:[^ ]*]], [[D5:[^ ]*]] -> [[SH6:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D4]], [[D5]], 2], f32>) -> !mo.tensor<[[SH6]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.rebind<() -> [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D10:[^ ]*]], [[D11:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D4]], [[D5]], 2], f32> -> !mo.tensor<[[[D10]], [[D11]], 2], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape([[V0]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[[D10]], [[D11]], 2], f32>) -> !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.sub([[V2]], [[V3]]) : !mo.tensor<[[[D7]], [[D8]], 2], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH12:[^ ]*]]: !mosh.ape>([[V4]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]], 2], f32>) -> !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH12]]> [[V5]] : !mo.tensor<[[SH12]], f32>
// CHECK-NEXT: }

// -----

mo.graph @sub_static_static(%arg0: !mo.tensor<[1, 2, 3], f32>, %arg1: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.sub(%arg0, %arg1) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @sub_static_static<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.sub([[VARG0]], [[VARG1]]) : !mo.tensor<[1, 2, 3], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }
