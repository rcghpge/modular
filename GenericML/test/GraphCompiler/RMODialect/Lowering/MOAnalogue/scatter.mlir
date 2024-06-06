// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_dyn_dyn<[[SH5:[^ ]*]]: !mosh.ape, [[SH6:[^ ]*]]: !mosh.ape, [[SH7:[^ ]*]]: !mosh.ape -> [[SH8:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH5]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH6]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH7]], si64>) -> !mo.tensor<[[SH8]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH5]], f32> -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH6]], f32> -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG2]]) : !mo.tensor<[[SH7]], si64> -> !mo.tensor<[[SH3]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[SH2]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[SH1]], f32>, !mo.tensor<[[SH2]], f32>, !mo.tensor<[[SH2]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH4:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V6]] : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_dyn_dyn(%input: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter(%input, %updates, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_dyn_params<[[SH7:[^ ]*]]: !mosh.ape, [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]] -> [[SH12:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH7]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D8]], [[D9]]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D10]], [[D11]]], si64>) -> !mo.tensor<[[SH12]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH7]], f32> -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D8]], [[D9]]], f32> -> !mo.tensor<[[[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]], [[D5:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D10]], [[D11]]], si64> -> !mo.tensor<[[[D4]], [[D5]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D4]], [[D5]]], si64>) -> !mo.tensor<[[[D2]], [[D3]]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[SH1]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>, !mo.tensor<[[[D2]], [[D3]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_dyn_params(%input: !mo.tensor<?, f32>, %updates: !mo.tensor<[?, ?], f32>, %indices: !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter(%input, %updates, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_params_dyn<[[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]], [[SH10:[^ ]*]]: !mosh.ape, [[SH11:[^ ]*]]: !mosh.ape -> [[SH12:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH10]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH11]], si64>) -> !mo.tensor<[[SH12]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH10]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG2]]) : !mo.tensor<[[SH11]], si64> -> !mo.tensor<[[SH5]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[SH4]], f32>, !mo.tensor<[[SH4]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_params_dyn(%input: !mo.tensor<[?, ?, ?], f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter(%input, %updates, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_params_params<[[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]], [[D12:[^ ]*]], [[D13:[^ ]*]], [[D14:[^ ]*]], [[D15:[^ ]*]] -> [[SH16:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D12]], [[D13]]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D14]], [[D15]]], si64>) -> !mo.tensor<[[SH16]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]], [[D5:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D12]], [[D13]]], f32> -> !mo.tensor<[[[D4]], [[D5]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D6:[^ ]*]], [[D7:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D14]], [[D15]]], si64> -> !mo.tensor<[[[D6]], [[D7]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D6]], [[D7]]], si64>) -> !mo.tensor<[[[D4]], [[D5]]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]]], f32>, !mo.tensor<[[[D4]], [[D5]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH8:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH8]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH8]]> [[V6]] : !mo.tensor<[[SH8]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_params_params(%input: !mo.tensor<[?, ?, ?], f32>, %updates: !mo.tensor<[?, ?], f32>, %indices: !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter(%input, %updates, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_params_mix<[[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]] -> [[SH12:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D10]], 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D11]], 3], si64>) -> !mo.tensor<[[SH12]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D10]], 3], f32> -> !mo.tensor<[[[D4]], 3], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D11]], 3], si64> -> !mo.tensor<[[[D5]], 3], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D5]], 3], si64>) -> !mo.tensor<[[[D4]], 3], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], 3], f32>, !mo.tensor<[[[D4]], 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_params_mix(%input: !mo.tensor<[?, ?, ?], f32>, %updates: !mo.tensor<[?, 3], f32>, %indices: !mo.tensor<[?, 3], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter(%input, %updates, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, 3], f32>, !mo.tensor<[?, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_mix_mix<[[D6:[^ ]*]], [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]] -> [[SH10:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D6]], 8, [[D7]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D8]], 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D9]], 3], si64>) -> !mo.tensor<[[SH10]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D6]], 8, [[D7]]], f32> -> !mo.tensor<[[[D1]], 8, [[D2]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D3:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D8]], 3], f32> -> !mo.tensor<[[[D3]], 3], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D9]], 3], si64> -> !mo.tensor<[[[D4]], 3], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D4]], 3], si64>) -> !mo.tensor<[[[D3]], 3], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], 8, [[D2]]], f32>, !mo.tensor<[[[D3]], 3], f32>, !mo.tensor<[[[D3]], 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], 8, [[D2]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH5:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], 8, [[D2]]], f32>) -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH5]]> [[V6]] : !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_mix_mix(%input: !mo.tensor<[?, 8, ?], f32>, %updates: !mo.tensor<[?, 3], f32>, %indices: !mo.tensor<[?, 3], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter(%input, %updates, %indices, %axis) : (!mo.tensor<[?, 8, ?], f32>, !mo.tensor<[?, 3], f32>, !mo.tensor<[?, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_static_static<() -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[3, 8, 4], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[5, 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[5, 3], si64>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.scatter([[VARG0]], [[VARG1]], [[VARG2]], [[V0]]) : (!mo.tensor<[3, 8, 4], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 8, 4], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH1:[^ ]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[3, 8, 4], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> [[V2]] : !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_static_static(%input: !mo.tensor<[3, 8, 4], f32>, %updates: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[5, 3], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter(%input, %updates, %indices, %axis) : (!mo.tensor<[3, 8, 4], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_add_dyn_dyn<[[SH5:[^ ]*]]: !mosh.ape, [[SH6:[^ ]*]]: !mosh.ape, [[SH7:[^ ]*]]: !mosh.ape -> [[SH8:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH5]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH6]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH7]], si64>) -> !mo.tensor<[[SH8]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH5]], f32> -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH6]], f32> -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG2]]) : !mo.tensor<[[SH7]], si64> -> !mo.tensor<[[SH3]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[SH2]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.add([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[SH1]], f32>, !mo.tensor<[[SH2]], f32>, !mo.tensor<[[SH2]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH4:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V6]] : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_add_dyn_dyn(%input: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.add(%input, %updates, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_add_dyn_params<[[SH7:[^ ]*]]: !mosh.ape, [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]] -> [[SH12:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH7]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D8]], [[D9]]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D10]], [[D11]]], si64>) -> !mo.tensor<[[SH12]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH7]], f32> -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D8]], [[D9]]], f32> -> !mo.tensor<[[[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]], [[D5:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D10]], [[D11]]], si64> -> !mo.tensor<[[[D4]], [[D5]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D4]], [[D5]]], si64>) -> !mo.tensor<[[[D2]], [[D3]]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.add([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[SH1]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>, !mo.tensor<[[[D2]], [[D3]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_add_dyn_params(%input: !mo.tensor<?, f32>, %updates: !mo.tensor<[?, ?], f32>, %indices: !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.add(%input, %updates, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_add_params_dyn<[[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]], [[SH10:[^ ]*]]: !mosh.ape, [[SH11:[^ ]*]]: !mosh.ape -> [[SH12:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH10]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH11]], si64>) -> !mo.tensor<[[SH12]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH10]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG2]]) : !mo.tensor<[[SH11]], si64> -> !mo.tensor<[[SH5]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.add([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[SH4]], f32>, !mo.tensor<[[SH4]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_add_params_dyn(%input: !mo.tensor<[?, ?, ?], f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.add(%input, %updates, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_add_params_params<[[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]], [[D12:[^ ]*]], [[D13:[^ ]*]], [[D14:[^ ]*]], [[D15:[^ ]*]] -> [[SH16:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D12]], [[D13]]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D14]], [[D15]]], si64>) -> !mo.tensor<[[SH16]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]], [[D5:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D12]], [[D13]]], f32> -> !mo.tensor<[[[D4]], [[D5]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D6:[^ ]*]], [[D7:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D14]], [[D15]]], si64> -> !mo.tensor<[[[D6]], [[D7]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D6]], [[D7]]], si64>) -> !mo.tensor<[[[D4]], [[D5]]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.add([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]]], f32>, !mo.tensor<[[[D4]], [[D5]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH8:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH8]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH8]]> [[V6]] : !mo.tensor<[[SH8]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_add_params_params(%input: !mo.tensor<[?, ?, ?], f32>, %updates: !mo.tensor<[?, ?], f32>, %indices: !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.add(%input, %updates, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_add_params_mix<[[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]] -> [[SH12:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D10]], 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D11]], 3], si64>) -> !mo.tensor<[[SH12]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D10]], 3], f32> -> !mo.tensor<[[[D4]], 3], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D11]], 3], si64> -> !mo.tensor<[[[D5]], 3], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D5]], 3], si64>) -> !mo.tensor<[[[D4]], 3], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.add([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], 3], f32>, !mo.tensor<[[[D4]], 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_add_params_mix(%input: !mo.tensor<[?, ?, ?], f32>, %updates: !mo.tensor<[?, 3], f32>, %indices: !mo.tensor<[?, 3], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.add(%input, %updates, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, 3], f32>, !mo.tensor<[?, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_add_mix_mix<[[D6:[^ ]*]], [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]] -> [[SH10:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D6]], 8, [[D7]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D8]], 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D9]], 3], si64>) -> !mo.tensor<[[SH10]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D6]], 8, [[D7]]], f32> -> !mo.tensor<[[[D1]], 8, [[D2]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D3:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D8]], 3], f32> -> !mo.tensor<[[[D3]], 3], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D9]], 3], si64> -> !mo.tensor<[[[D4]], 3], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D4]], 3], si64>) -> !mo.tensor<[[[D3]], 3], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.add([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], 8, [[D2]]], f32>, !mo.tensor<[[[D3]], 3], f32>, !mo.tensor<[[[D3]], 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], 8, [[D2]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH5:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], 8, [[D2]]], f32>) -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH5]]> [[V6]] : !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_add_mix_mix(%input: !mo.tensor<[?, 8, ?], f32>, %updates: !mo.tensor<[?, 3], f32>, %indices: !mo.tensor<[?, 3], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.add(%input, %updates, %indices, %axis) : (!mo.tensor<[?, 8, ?], f32>, !mo.tensor<[?, 3], f32>, !mo.tensor<[?, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_add_static_static<() -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[3, 8, 4], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[5, 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[5, 3], si64>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.scatter.add([[VARG0]], [[VARG1]], [[VARG2]], [[V0]]) : (!mo.tensor<[3, 8, 4], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 8, 4], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH1:[^ ]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[3, 8, 4], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> [[V2]] : !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_add_static_static(%input: !mo.tensor<[3, 8, 4], f32>, %updates: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[5, 3], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.add(%input, %updates, %indices, %axis) : (!mo.tensor<[3, 8, 4], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_max_dyn_dyn<[[SH5:[^ ]*]]: !mosh.ape, [[SH6:[^ ]*]]: !mosh.ape, [[SH7:[^ ]*]]: !mosh.ape -> [[SH8:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH5]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH6]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH7]], si64>) -> !mo.tensor<[[SH8]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH5]], f32> -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH6]], f32> -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG2]]) : !mo.tensor<[[SH7]], si64> -> !mo.tensor<[[SH3]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[SH2]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.max([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[SH1]], f32>, !mo.tensor<[[SH2]], f32>, !mo.tensor<[[SH2]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH4:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V6]] : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_max_dyn_dyn(%input: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.max(%input, %updates, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_max_dyn_params<[[SH7:[^ ]*]]: !mosh.ape, [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]] -> [[SH12:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH7]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D8]], [[D9]]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D10]], [[D11]]], si64>) -> !mo.tensor<[[SH12]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH7]], f32> -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D8]], [[D9]]], f32> -> !mo.tensor<[[[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]], [[D5:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D10]], [[D11]]], si64> -> !mo.tensor<[[[D4]], [[D5]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D4]], [[D5]]], si64>) -> !mo.tensor<[[[D2]], [[D3]]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.max([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[SH1]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>, !mo.tensor<[[[D2]], [[D3]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_max_dyn_params(%input: !mo.tensor<?, f32>, %updates: !mo.tensor<[?, ?], f32>, %indices: !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.max(%input, %updates, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_max_params_dyn<[[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]], [[SH10:[^ ]*]]: !mosh.ape, [[SH11:[^ ]*]]: !mosh.ape -> [[SH12:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH10]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH11]], si64>) -> !mo.tensor<[[SH12]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH10]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG2]]) : !mo.tensor<[[SH11]], si64> -> !mo.tensor<[[SH5]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.max([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[SH4]], f32>, !mo.tensor<[[SH4]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_max_params_dyn(%input: !mo.tensor<[?, ?, ?], f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.max(%input, %updates, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_max_params_params<[[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]], [[D12:[^ ]*]], [[D13:[^ ]*]], [[D14:[^ ]*]], [[D15:[^ ]*]] -> [[SH16:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D12]], [[D13]]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D14]], [[D15]]], si64>) -> !mo.tensor<[[SH16]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]], [[D5:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D12]], [[D13]]], f32> -> !mo.tensor<[[[D4]], [[D5]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D6:[^ ]*]], [[D7:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D14]], [[D15]]], si64> -> !mo.tensor<[[[D6]], [[D7]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D6]], [[D7]]], si64>) -> !mo.tensor<[[[D4]], [[D5]]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.max([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]]], f32>, !mo.tensor<[[[D4]], [[D5]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH8:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH8]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH8]]> [[V6]] : !mo.tensor<[[SH8]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_max_params_params(%input: !mo.tensor<[?, ?, ?], f32>, %updates: !mo.tensor<[?, ?], f32>, %indices: !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.max(%input, %updates, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_max_params_mix<[[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]] -> [[SH12:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D10]], 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D11]], 3], si64>) -> !mo.tensor<[[SH12]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D10]], 3], f32> -> !mo.tensor<[[[D4]], 3], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D11]], 3], si64> -> !mo.tensor<[[[D5]], 3], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D5]], 3], si64>) -> !mo.tensor<[[[D4]], 3], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.max([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], 3], f32>, !mo.tensor<[[[D4]], 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_max_params_mix(%input: !mo.tensor<[?, ?, ?], f32>, %updates: !mo.tensor<[?, 3], f32>, %indices: !mo.tensor<[?, 3], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.max(%input, %updates, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, 3], f32>, !mo.tensor<[?, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_max_mix_mix<[[D6:[^ ]*]], [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]] -> [[SH10:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D6]], 8, [[D7]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D8]], 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D9]], 3], si64>) -> !mo.tensor<[[SH10]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D6]], 8, [[D7]]], f32> -> !mo.tensor<[[[D1]], 8, [[D2]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D3:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D8]], 3], f32> -> !mo.tensor<[[[D3]], 3], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D9]], 3], si64> -> !mo.tensor<[[[D4]], 3], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D4]], 3], si64>) -> !mo.tensor<[[[D3]], 3], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.max([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], 8, [[D2]]], f32>, !mo.tensor<[[[D3]], 3], f32>, !mo.tensor<[[[D3]], 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], 8, [[D2]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH5:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], 8, [[D2]]], f32>) -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH5]]> [[V6]] : !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_max_mix_mix(%input: !mo.tensor<[?, 8, ?], f32>, %updates: !mo.tensor<[?, 3], f32>, %indices: !mo.tensor<[?, 3], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.max(%input, %updates, %indices, %axis) : (!mo.tensor<[?, 8, ?], f32>, !mo.tensor<[?, 3], f32>, !mo.tensor<[?, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_max_static_static<() -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[3, 8, 4], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[5, 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[5, 3], si64>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.scatter.max([[VARG0]], [[VARG1]], [[VARG2]], [[V0]]) : (!mo.tensor<[3, 8, 4], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 8, 4], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH1:[^ ]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[3, 8, 4], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> [[V2]] : !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_max_static_static(%input: !mo.tensor<[3, 8, 4], f32>, %updates: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[5, 3], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.max(%input, %updates, %indices, %axis) : (!mo.tensor<[3, 8, 4], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_min_dyn_dyn<[[SH5:[^ ]*]]: !mosh.ape, [[SH6:[^ ]*]]: !mosh.ape, [[SH7:[^ ]*]]: !mosh.ape -> [[SH8:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH5]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH6]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH7]], si64>) -> !mo.tensor<[[SH8]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH5]], f32> -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH6]], f32> -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG2]]) : !mo.tensor<[[SH7]], si64> -> !mo.tensor<[[SH3]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[SH2]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.min([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[SH1]], f32>, !mo.tensor<[[SH2]], f32>, !mo.tensor<[[SH2]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH4:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V6]] : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_min_dyn_dyn(%input: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.min(%input, %updates, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_min_dyn_params<[[SH7:[^ ]*]]: !mosh.ape, [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]] -> [[SH12:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH7]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D8]], [[D9]]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D10]], [[D11]]], si64>) -> !mo.tensor<[[SH12]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH7]], f32> -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D8]], [[D9]]], f32> -> !mo.tensor<[[[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]], [[D5:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D10]], [[D11]]], si64> -> !mo.tensor<[[[D4]], [[D5]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D4]], [[D5]]], si64>) -> !mo.tensor<[[[D2]], [[D3]]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.min([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[SH1]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>, !mo.tensor<[[[D2]], [[D3]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_min_dyn_params(%input: !mo.tensor<?, f32>, %updates: !mo.tensor<[?, ?], f32>, %indices: !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.min(%input, %updates, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_min_params_dyn<[[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]], [[SH10:[^ ]*]]: !mosh.ape, [[SH11:[^ ]*]]: !mosh.ape -> [[SH12:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH10]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH11]], si64>) -> !mo.tensor<[[SH12]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH10]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG2]]) : !mo.tensor<[[SH11]], si64> -> !mo.tensor<[[SH5]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.min([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[SH4]], f32>, !mo.tensor<[[SH4]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_min_params_dyn(%input: !mo.tensor<[?, ?, ?], f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.min(%input, %updates, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_min_params_params<[[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]], [[D12:[^ ]*]], [[D13:[^ ]*]], [[D14:[^ ]*]], [[D15:[^ ]*]] -> [[SH16:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D12]], [[D13]]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D14]], [[D15]]], si64>) -> !mo.tensor<[[SH16]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]], [[D5:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D12]], [[D13]]], f32> -> !mo.tensor<[[[D4]], [[D5]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D6:[^ ]*]], [[D7:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D14]], [[D15]]], si64> -> !mo.tensor<[[[D6]], [[D7]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D6]], [[D7]]], si64>) -> !mo.tensor<[[[D4]], [[D5]]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.min([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]]], f32>, !mo.tensor<[[[D4]], [[D5]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH8:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH8]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH8]]> [[V6]] : !mo.tensor<[[SH8]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_min_params_params(%input: !mo.tensor<[?, ?, ?], f32>, %updates: !mo.tensor<[?, ?], f32>, %indices: !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.min(%input, %updates, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_min_params_mix<[[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]] -> [[SH12:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D10]], 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D11]], 3], si64>) -> !mo.tensor<[[SH12]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D10]], 3], f32> -> !mo.tensor<[[[D4]], 3], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D11]], 3], si64> -> !mo.tensor<[[[D5]], 3], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D5]], 3], si64>) -> !mo.tensor<[[[D4]], 3], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.min([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], 3], f32>, !mo.tensor<[[[D4]], 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_min_params_mix(%input: !mo.tensor<[?, ?, ?], f32>, %updates: !mo.tensor<[?, 3], f32>, %indices: !mo.tensor<[?, 3], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.min(%input, %updates, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, 3], f32>, !mo.tensor<[?, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_min_mix_mix<[[D6:[^ ]*]], [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]] -> [[SH10:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D6]], 8, [[D7]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D8]], 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D9]], 3], si64>) -> !mo.tensor<[[SH10]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D6]], 8, [[D7]]], f32> -> !mo.tensor<[[[D1]], 8, [[D2]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D3:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D8]], 3], f32> -> !mo.tensor<[[[D3]], 3], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D9]], 3], si64> -> !mo.tensor<[[[D4]], 3], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D4]], 3], si64>) -> !mo.tensor<[[[D3]], 3], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.min([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], 8, [[D2]]], f32>, !mo.tensor<[[[D3]], 3], f32>, !mo.tensor<[[[D3]], 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], 8, [[D2]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH5:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], 8, [[D2]]], f32>) -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH5]]> [[V6]] : !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_min_mix_mix(%input: !mo.tensor<[?, 8, ?], f32>, %updates: !mo.tensor<[?, 3], f32>, %indices: !mo.tensor<[?, 3], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.min(%input, %updates, %indices, %axis) : (!mo.tensor<[?, 8, ?], f32>, !mo.tensor<[?, 3], f32>, !mo.tensor<[?, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_min_static_static<() -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[3, 8, 4], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[5, 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[5, 3], si64>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.scatter.min([[VARG0]], [[VARG1]], [[VARG2]], [[V0]]) : (!mo.tensor<[3, 8, 4], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 8, 4], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH1:[^ ]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[3, 8, 4], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> [[V2]] : !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_min_static_static(%input: !mo.tensor<[3, 8, 4], f32>, %updates: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[5, 3], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.min(%input, %updates, %indices, %axis) : (!mo.tensor<[3, 8, 4], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_mul_dyn_dyn<[[SH5:[^ ]*]]: !mosh.ape, [[SH6:[^ ]*]]: !mosh.ape, [[SH7:[^ ]*]]: !mosh.ape -> [[SH8:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH5]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH6]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH7]], si64>) -> !mo.tensor<[[SH8]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH5]], f32> -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH6]], f32> -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG2]]) : !mo.tensor<[[SH7]], si64> -> !mo.tensor<[[SH3]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH3]], si64>) -> !mo.tensor<[[SH2]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.mul([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[SH1]], f32>, !mo.tensor<[[SH2]], f32>, !mo.tensor<[[SH2]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH4:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V6]] : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_mul_dyn_dyn(%input: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.mul(%input, %updates, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_mul_dyn_params<[[SH7:[^ ]*]]: !mosh.ape, [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]] -> [[SH12:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH7]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D8]], [[D9]]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D10]], [[D11]]], si64>) -> !mo.tensor<[[SH12]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0]]) : !mo.tensor<[[SH7]], f32> -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D8]], [[D9]]], f32> -> !mo.tensor<[[[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]], [[D5:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D10]], [[D11]]], si64> -> !mo.tensor<[[[D4]], [[D5]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D4]], [[D5]]], si64>) -> !mo.tensor<[[[D2]], [[D3]]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.mul([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[SH1]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>, !mo.tensor<[[[D2]], [[D3]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_mul_dyn_params(%input: !mo.tensor<?, f32>, %updates: !mo.tensor<[?, ?], f32>, %indices: !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.mul(%input, %updates, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_mul_params_dyn<[[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]], [[SH10:[^ ]*]]: !mosh.ape, [[SH11:[^ ]*]]: !mosh.ape -> [[SH12:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH10]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH11]], si64>) -> !mo.tensor<[[SH12]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[SH4:[^ ]*]]: !mosh.ape>([[VARG1]]) : !mo.tensor<[[SH10]], f32> -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[SH5:[^ ]*]]: !mosh.ape>([[VARG2]]) : !mo.tensor<[[SH11]], si64> -> !mo.tensor<[[SH5]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH5]], si64>) -> !mo.tensor<[[SH4]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.mul([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[SH4]], f32>, !mo.tensor<[[SH4]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_mul_params_dyn(%input: !mo.tensor<[?, ?, ?], f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.mul(%input, %updates, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_mul_params_params<[[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]], [[D12:[^ ]*]], [[D13:[^ ]*]], [[D14:[^ ]*]], [[D15:[^ ]*]] -> [[SH16:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D12]], [[D13]]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D14]], [[D15]]], si64>) -> !mo.tensor<[[SH16]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]], [[D5:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D12]], [[D13]]], f32> -> !mo.tensor<[[[D4]], [[D5]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D6:[^ ]*]], [[D7:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D14]], [[D15]]], si64> -> !mo.tensor<[[[D6]], [[D7]]], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D6]], [[D7]]], si64>) -> !mo.tensor<[[[D4]], [[D5]]], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.mul([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]]], f32>, !mo.tensor<[[[D4]], [[D5]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH8:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH8]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH8]]> [[V6]] : !mo.tensor<[[SH8]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_mul_params_params(%input: !mo.tensor<[?, ?, ?], f32>, %updates: !mo.tensor<[?, ?], f32>, %indices: !mo.tensor<[?, ?], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.mul(%input, %updates, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_mul_params_mix<[[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]], [[D10:[^ ]*]], [[D11:[^ ]*]] -> [[SH12:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D10]], 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D11]], 3], si64>) -> !mo.tensor<[[SH12]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32> -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D10]], 3], f32> -> !mo.tensor<[[[D4]], 3], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D5:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D11]], 3], si64> -> !mo.tensor<[[[D5]], 3], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D5]], 3], si64>) -> !mo.tensor<[[[D4]], 3], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.mul([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], 3], f32>, !mo.tensor<[[[D4]], 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH6:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>) -> !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH6]]> [[V6]] : !mo.tensor<[[SH6]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_mul_params_mix(%input: !mo.tensor<[?, ?, ?], f32>, %updates: !mo.tensor<[?, 3], f32>, %indices: !mo.tensor<[?, 3], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.mul(%input, %updates, %indices, %axis) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, 3], f32>, !mo.tensor<[?, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_mul_mix_mix<[[D6:[^ ]*]], [[D7:[^ ]*]], [[D8:[^ ]*]], [[D9:[^ ]*]] -> [[SH10:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D6]], 8, [[D7]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D8]], 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D9]], 3], si64>) -> !mo.tensor<[[SH10]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.rebind<() -> [[D1:[^ ]*]], [[D2:[^ ]*]]>([[VARG0]]) : !mo.tensor<[[[D6]], 8, [[D7]]], f32> -> !mo.tensor<[[[D1]], 8, [[D2]]], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.rebind<() -> [[D3:[^ ]*]]>([[VARG1]]) : !mo.tensor<[[[D8]], 3], f32> -> !mo.tensor<[[[D3]], 3], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.rebind<() -> [[D4:[^ ]*]]>([[VARG2]]) : !mo.tensor<[[[D9]], 3], si64> -> !mo.tensor<[[[D4]], 3], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[[D4]], 3], si64>) -> !mo.tensor<[[[D3]], 3], si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.scatter.mul([[V1]], [[V2]], [[V4]], [[V0]]) : (!mo.tensor<[[[D1]], 8, [[D2]]], f32>, !mo.tensor<[[[D3]], 3], f32>, !mo.tensor<[[[D3]], 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D1]], 8, [[D2]]], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH5:[^ ]*]]: !mosh.ape>([[V5]]) {{.*}} : (!mo.tensor<[[[D1]], 8, [[D2]]], f32>) -> !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH5]]> [[V6]] : !mo.tensor<[[SH5]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_mul_mix_mix(%input: !mo.tensor<[?, 8, ?], f32>, %updates: !mo.tensor<[?, 3], f32>, %indices: !mo.tensor<[?, 3], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.mul(%input, %updates, %indices, %axis) : (!mo.tensor<[?, 8, ?], f32>, !mo.tensor<[?, 3], f32>, !mo.tensor<[?, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_mul_static_static<() -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[3, 8, 4], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[5, 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[5, 3], si64>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.scatter.mul([[VARG0]], [[VARG1]], [[VARG2]], [[V0]]) : (!mo.tensor<[3, 8, 4], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 8, 4], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH1:[^ ]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[3, 8, 4], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> [[V2]] : !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: }
// CHECK-NEXT: }
mo.graph @scatter_mul_static_static(%input: !mo.tensor<[3, 8, 4], f32>, %updates: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[5, 3], si64>) -> !mo.tensor<?, f32> {
     %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.mul(%input, %updates, %indices, %axis) : (!mo.tensor<[3, 8, 4], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
