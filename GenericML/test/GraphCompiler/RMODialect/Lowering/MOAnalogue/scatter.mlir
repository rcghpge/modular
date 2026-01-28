// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_dyn_dyn<[[SH0:[^ ]*]]: !mosh.ape, [[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG2]]) {{.*}} : (!mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH1]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.scatter([[VARG0]], [[VARG1]], [[V1]], [[V0]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], f32>, !mo.tensor<[[SH1]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH4:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V3]] : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: }
mo.graph @scatter_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter<() -> Sh4: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %1 : !mo.tensor<Sh4, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_dyn_params<[[SH0:[^ ]*]]: !mosh.ape, [[D0:[^ ]*]], [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]] -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D0]], [[D1]]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D2]], [[D3]]], si64>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG2]]) {{.*}} : (!mo.tensor<[[[D2]], [[D3]]], si64>) -> !mo.tensor<[[[D0]], [[D1]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.scatter([[VARG0]], [[VARG1]], [[V1]], [[V0]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D0]], [[D1]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V3]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }
mo.graph @scatter_dyn_params<Sh0: !mosh.ape, D0, D1, D2, D3 -> Sh1: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<[D0, D1], f32>, %indices: !mo.tensor<[D2, D3], si64>) -> !mo.tensor<Sh1, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter<() -> Sh2: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_static_static<() -> [[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[3, 8], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[5, 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[5, 3], si64>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.scatter([[VARG0]], [[VARG1]], [[VARG2]], [[V0]]) : (!mo.tensor<[3, 8], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 8], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH1:[^ ]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> [[V2]] : !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: }
mo.graph @scatter_static_static<() -> Sh0: !mosh.ape>(%input: !mo.tensor<[3, 8], f32>, %updates: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[5, 3], si64>) -> !mo.tensor<Sh0, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter<() -> Sh1: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<[3, 8], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_add_dyn_dyn<[[SH0:[^ ]*]]: !mosh.ape, [[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG2]]) {{.*}} : (!mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH1]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.scatter.add([[VARG0]], [[VARG1]], [[V1]], [[V0]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], f32>, !mo.tensor<[[SH1]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH4:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V3]] : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: }
mo.graph @scatter_add_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.add<() -> Sh4: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %1 : !mo.tensor<Sh4, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_add_dyn_params<[[SH0:[^ ]*]]: !mosh.ape, [[D0:[^ ]*]], [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]] -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D0]], [[D1]]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D2]], [[D3]]], si64>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG2]]) {{.*}} : (!mo.tensor<[[[D2]], [[D3]]], si64>) -> !mo.tensor<[[[D0]], [[D1]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.scatter.add([[VARG0]], [[VARG1]], [[V1]], [[V0]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D0]], [[D1]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V3]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }
mo.graph @scatter_add_dyn_params<Sh0: !mosh.ape, D0, D1, D2, D3 -> Sh1: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<[D0, D1], f32>, %indices: !mo.tensor<[D2, D3], si64>) -> !mo.tensor<Sh1, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.add<() -> Sh2: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_add_params_dyn<[[D0:[^ ]*]], [[D1:[^ ]*]], [[D2:[^ ]*]], [[SH0:[^ ]*]]: !mosh.ape, [[SH1:[^ ]*]]: !mosh.ape -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG2]]) {{.*}} : (!mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[SH0]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.scatter.add([[VARG0]], [[VARG1]], [[V1]], [[V0]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, !mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH0]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH3:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>) -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH3]]> [[V3]] : !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: }
mo.graph @scatter_add_params_dyn<D0, D1, D2, Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%input: !mo.tensor<[D0, D1, D2], f32>, %updates: !mo.tensor<Sh0, f32>, %indices: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.add<() -> Sh3: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_add_static_static<() -> [[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[3, 8], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[5, 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[5, 3], si64>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.scatter.add([[VARG0]], [[VARG1]], [[VARG2]], [[V0]]) : (!mo.tensor<[3, 8], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 8], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH1:[^ ]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> [[V2]] : !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: }
mo.graph @scatter_add_static_static<() -> Sh0: !mosh.ape>(%input: !mo.tensor<[3, 8], f32>, %updates: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[5, 3], si64>) -> !mo.tensor<Sh0, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.add<() -> Sh1: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<[3, 8], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_max_dyn_dyn<[[SH0:[^ ]*]]: !mosh.ape, [[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG2]]) {{.*}} : (!mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH1]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.scatter.max([[VARG0]], [[VARG1]], [[V1]], [[V0]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], f32>, !mo.tensor<[[SH1]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH4:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V3]] : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: }
mo.graph @scatter_max_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.max<() -> Sh4: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %1 : !mo.tensor<Sh4, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_max_dyn_params<[[SH0:[^ ]*]]: !mosh.ape, [[D0:[^ ]*]], [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]] -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D0]], [[D1]]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D2]], [[D3]]], si64>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG2]]) {{.*}} : (!mo.tensor<[[[D2]], [[D3]]], si64>) -> !mo.tensor<[[[D0]], [[D1]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.scatter.max([[VARG0]], [[VARG1]], [[V1]], [[V0]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D0]], [[D1]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V3]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }
mo.graph @scatter_max_dyn_params<Sh0: !mosh.ape, D0, D1, D2, D3 -> Sh1: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<[D0, D1], f32>, %indices: !mo.tensor<[D2, D3], si64>) -> !mo.tensor<Sh1, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.max<() -> Sh2: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_max_params_dyn<[[D0:[^ ]*]], [[D1:[^ ]*]], [[D2:[^ ]*]], [[SH0:[^ ]*]]: !mosh.ape, [[SH1:[^ ]*]]: !mosh.ape -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG2]]) {{.*}} : (!mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[SH0]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.scatter.max([[VARG0]], [[VARG1]], [[V1]], [[V0]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, !mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH0]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH3:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>) -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH3]]> [[V3]] : !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: }
mo.graph @scatter_max_params_dyn<D0, D1, D2, Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%input: !mo.tensor<[D0, D1, D2], f32>, %updates: !mo.tensor<Sh0, f32>, %indices: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.max<() -> Sh3: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_max_static_static<() -> [[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[3, 8], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[5, 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[5, 3], si64>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.scatter.max([[VARG0]], [[VARG1]], [[VARG2]], [[V0]]) : (!mo.tensor<[3, 8], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 8], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH1:[^ ]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> [[V2]] : !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: }
mo.graph @scatter_max_static_static<() -> Sh0: !mosh.ape>(%input: !mo.tensor<[3, 8], f32>, %updates: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[5, 3], si64>) -> !mo.tensor<Sh0, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.max<() -> Sh1: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<[3, 8], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_min_dyn_dyn<[[SH0:[^ ]*]]: !mosh.ape, [[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG2]]) {{.*}} : (!mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH1]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.scatter.min([[VARG0]], [[VARG1]], [[V1]], [[V0]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], f32>, !mo.tensor<[[SH1]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH4:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V3]] : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: }
mo.graph @scatter_min_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.min<() -> Sh4: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %1 : !mo.tensor<Sh4, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_min_dyn_params<[[SH0:[^ ]*]]: !mosh.ape, [[D0:[^ ]*]], [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]] -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D0]], [[D1]]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D2]], [[D3]]], si64>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG2]]) {{.*}} : (!mo.tensor<[[[D2]], [[D3]]], si64>) -> !mo.tensor<[[[D0]], [[D1]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.scatter.min([[VARG0]], [[VARG1]], [[V1]], [[V0]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D0]], [[D1]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V3]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }
mo.graph @scatter_min_dyn_params<Sh0: !mosh.ape, D0, D1, D2, D3 -> Sh1: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<[D0, D1], f32>, %indices: !mo.tensor<[D2, D3], si64>) -> !mo.tensor<Sh1, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.min<() -> Sh2: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_min_params_dyn<[[D0:[^ ]*]], [[D1:[^ ]*]], [[D2:[^ ]*]], [[SH0:[^ ]*]]: !mosh.ape, [[SH1:[^ ]*]]: !mosh.ape -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG2]]) {{.*}} : (!mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[SH0]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.scatter.min([[VARG0]], [[VARG1]], [[V1]], [[V0]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, !mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH0]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH3:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>) -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH3]]> [[V3]] : !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: }
mo.graph @scatter_min_params_dyn<D0, D1, D2, Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%input: !mo.tensor<[D0, D1, D2], f32>, %updates: !mo.tensor<Sh0, f32>, %indices: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.min<() -> Sh3: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_min_static_static<() -> [[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[3, 8], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[5, 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[5, 3], si64>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.scatter.min([[VARG0]], [[VARG1]], [[VARG2]], [[V0]]) : (!mo.tensor<[3, 8], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 8], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH1:[^ ]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> [[V2]] : !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: }
mo.graph @scatter_min_static_static<() -> Sh0: !mosh.ape>(%input: !mo.tensor<[3, 8], f32>, %updates: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[5, 3], si64>) -> !mo.tensor<Sh0, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.min<() -> Sh1: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<[3, 8], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_mul_dyn_dyn<[[SH0:[^ ]*]]: !mosh.ape, [[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape -> [[SH3:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH1]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH3]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG2]]) {{.*}} : (!mo.tensor<[[SH2]], si64>) -> !mo.tensor<[[SH1]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.scatter.mul([[VARG0]], [[VARG1]], [[V1]], [[V0]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], f32>, !mo.tensor<[[SH1]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH4:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH4]]> [[V3]] : !mo.tensor<[[SH4]], f32>
// CHECK-NEXT: }
mo.graph @scatter_mul_dyn_dyn<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.mul<() -> Sh4: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh4, f32>
  mo.output<:!mosh.ape Sh4> %1 : !mo.tensor<Sh4, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_mul_dyn_params<[[SH0:[^ ]*]]: !mosh.ape, [[D0:[^ ]*]], [[D1:[^ ]*]], [[D2:[^ ]*]], [[D3:[^ ]*]] -> [[SH1:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[[D0]], [[D1]]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[[D2]], [[D3]]], si64>) -> !mo.tensor<[[SH1]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG2]]) {{.*}} : (!mo.tensor<[[[D2]], [[D3]]], si64>) -> !mo.tensor<[[[D0]], [[D1]]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.scatter.mul([[VARG0]], [[VARG1]], [[V1]], [[V0]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D0]], [[D1]]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[SH0]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH2:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> [[V3]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }
mo.graph @scatter_mul_dyn_params<Sh0: !mosh.ape, D0, D1, D2, D3 -> Sh1: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<[D0, D1], f32>, %indices: !mo.tensor<[D2, D3], si64>) -> !mo.tensor<Sh1, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.mul<() -> Sh2: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_mul_params_dyn<[[D0:[^ ]*]], [[D1:[^ ]*]], [[D2:[^ ]*]], [[SH0:[^ ]*]]: !mosh.ape, [[SH1:[^ ]*]]: !mosh.ape -> [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG2]]) {{.*}} : (!mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[SH0]], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.scatter.mul([[VARG0]], [[VARG1]], [[V1]], [[V0]]) : (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, !mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH0]], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH3:[^ ]*]]: !mosh.ape>([[V2]]) {{.*}} : (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>) -> !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH3]]> [[V3]] : !mo.tensor<[[SH3]], f32>
// CHECK-NEXT: }
mo.graph @scatter_mul_params_dyn<D0, D1, D2, Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%input: !mo.tensor<[D0, D1, D2], f32>, %updates: !mo.tensor<Sh0, f32>, %indices: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.mul<() -> Sh3: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_mul_static_static<() -> [[SH0:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[3, 8], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[5, 3], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[5, 3], si64>) -> !mo.tensor<[[SH0]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.scatter.mul([[VARG0]], [[VARG1]], [[VARG2]], [[V0]]) : (!mo.tensor<[3, 8], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 8], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[SH1:[^ ]*]]: !mosh.ape>([[V1]]) {{.*}} : (!mo.tensor<[3, 8], f32>) -> !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> [[V2]] : !mo.tensor<[[SH1]], f32>
// CHECK-NEXT: }
mo.graph @scatter_mul_static_static<() -> Sh0: !mosh.ape>(%input: !mo.tensor<[3, 8], f32>, %updates: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[5, 3], si64>) -> !mo.tensor<Sh0, f32> {
     %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter.mul<() -> Sh1: !mosh.ape>(%input, %updates, %indices, %axis) : (!mo.tensor<[3, 8], f32>, !mo.tensor<[5, 3], f32>, !mo.tensor<[5, 3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32>
  mo.output<:!mosh.ape Sh1> %1 : !mo.tensor<Sh1, f32>
}
