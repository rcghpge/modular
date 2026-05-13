// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @conv_dyn_params<D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12 -> D13, D14, D15, D16, D17>(%input : !mo.tensor<[D0, D1, D2, D3, D4], f32>, %filter : !mo.tensor<[D5, D6, D7, D8, D9], f32>, %st: !mo.tensor<[D10], si64>, %di: !mo.tensor<[D11], si64>, %pa: !mo.tensor<[D12], si64>, %ng: !mo.tensor<[], si64>) -> (!mo.tensor<[D13, D14, D15, D16, D17], f32>) {
  %0 = rmo.mo.conv<() -> D18, D19, D20, D21, D22>(%input, %filter) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
    !mo.tensor<[D0, D1, D2, D3, D4], f32>, !mo.tensor<[D5, D6, D7, D8, D9], f32>, !mo.tensor<[D10], si64>, !mo.tensor<[D11], si64>, !mo.tensor<[D12], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[D18, D19, D20, D21, D22], f32>
  mo.output<D18, D19, D20, D21, D22> %0 : !mo.tensor<[D18, D19, D20, D21, D22], f32>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @conv_dyn_params<D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12 -> D13, D14, D15, D16, D17>([[ARG0:%.*]]: !mo.tensor<[D0, D1, D2, D3, D4], f32>, [[ARG1:%.*]]: !mo.tensor<[D5, D6, D7, D8, D9], f32>, [[ARG2:%.*]]: !mo.tensor<[D10], si64>, [[ARG3:%.*]]: !mo.tensor<[D11], si64>, [[ARG4:%.*]]: !mo.tensor<[D12], si64>, [[ARG5:%.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[D13, D14, D15, D16, D17], f32> {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape([[ARG3]]) {{.*}} : (!mo.tensor<[D11], si64>) -> !mo.tensor<[D10], si64>
//  CHECK-NEXT:   [[V1:%.*]] = mo.conv<() -> [[RMO_D1:[A-Za-z0-9_]*]], [[RMO_D2:[A-Za-z0-9_]*]], [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) {filter_layout = ""} [strides = [[ARG2]], dilations = [[V0]], paddings = [[ARG4]], num_groups = [[ARG5]]] : (!mo.tensor<[D0, D1, D2, D3, D4], f32>, !mo.tensor<[D5, D6, D7, D8, D9], f32>, !mo.tensor<[D10], si64>, !mo.tensor<[D10], si64>, !mo.tensor<[D12], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> D18, D19, D20, D21, D22>([[V1]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32>) -> !mo.tensor<[D18, D19, D20, D21, D22], f32>
//  CHECK-NEXT:   mo.output<D18, D19, D20, D21, D22> [[V2]] : !mo.tensor<[D18, D19, D20, D21, D22], f32>

