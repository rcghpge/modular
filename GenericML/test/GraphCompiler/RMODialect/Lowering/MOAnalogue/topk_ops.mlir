// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

mo.graph @topk_dyn(%input: !mo.tensor<?, f32>, %k: !mo.tensor<?, si64>, %axis: !mo.tensor<?, si64>, %sorted: !mo.tensor<?, bool>) -> (!mo.tensor<?, f32>, !mo.tensor<?, si64>) {
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, bool>
      ) -> (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>
      )
  mo.output %values, %indices : !mo.tensor<?, f32>, !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @topk_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape, [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG3:%.*]]: !mo.tensor<[[SH4]], bool>) -> (!mo.tensor<[[SH5]], f32>, !mo.tensor<[[SH6]], si64>) {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH8]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH9]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH4]], bool> -> !mo.tensor<[[SH10]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH13:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[SH7]], f32>) -> !mo.tensor<[[RMO_SH13]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH8]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH9]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH10]], bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   [[VALUES:%.*]], [[INDICES:%.*]] = mo.top_k<() -> [[RMO_SH14:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]], [[V5]], [[V6]], [[V7]]) : (!mo.tensor<[[RMO_SH13]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[RMO_SH14]], f32>, !mo.tensor<[[RMO_SH14]], si64>)
//  CHECK-NEXT:   [[V8:%.*]] = rmo.rebind_tensor_shape<() -> [[SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[VALUES]]) {{.*}} : (!mo.tensor<[[RMO_SH14]], f32>) -> !mo.tensor<[[SH11]], f32>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[INDICES]]) {{.*}} : (!mo.tensor<[[RMO_SH14]], si64>) -> !mo.tensor<[[SH12]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH11]], :!mosh.ape [[SH12]]> [[V8]], [[V9]] : !mo.tensor<[[SH11]], f32>, !mo.tensor<[[SH12]], si64>

// -----

mo.graph @topk_dyn_params(%input: !mo.tensor<[?, ?], f32>, %k: !mo.tensor<?, si64>, %axis: !mo.tensor<?, si64>, %sorted: !mo.tensor<?, bool>) -> (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], si64>) {
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<[?, ?], f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, bool>
      ) -> (
        !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], si64>
      )
  mo.output %values, %indices : !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @topk_dyn_params<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape, [[SH5:[A-Za-z0-9_]*]]: !mosh.ape -> [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]], [[D9:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG2:%.*]]: !mo.tensor<[[SH4]], si64>, [[ARG3:%.*]]: !mo.tensor<[[SH5]], bool>) -> (!mo.tensor<[[[D6]], [[D7]]], f32>, !mo.tensor<[[[D8]], [[D9]]], si64>) {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D10]], [[D11]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH12]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH13:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH4]], si64> -> !mo.tensor<[[SH13]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH14:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH5]], bool> -> !mo.tensor<[[SH14]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D19:[A-Za-z0-9_]*]], [[RMO_D20:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[D10]], [[D11]]], f32>) -> !mo.tensor<[[[RMO_D19]], [[RMO_D20]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH12]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH13]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH14]], bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   [[VALUES:%.*]], [[INDICES:%.*]] = mo.top_k<() -> [[RMO_D21:[A-Za-z0-9_]*]], [[RMO_D22:[A-Za-z0-9_]*]]>([[V4]], [[V5]], [[V6]], [[V7]]) : (!mo.tensor<[[[RMO_D19]], [[RMO_D20]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[[RMO_D21]], [[RMO_D22]]], f32>, !mo.tensor<[[[RMO_D21]], [[RMO_D22]]], si64>)
//  CHECK-NEXT:   [[V8:%.*]] = rmo.rebind_tensor_shape<() -> [[D15:[A-Za-z0-9_]*]], [[D16:[A-Za-z0-9_]*]]>([[VALUES]]) {{.*}} : (!mo.tensor<[[[RMO_D21]], [[RMO_D22]]], f32>) -> !mo.tensor<[[[D15]], [[D16]]], f32>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape<() -> [[D17:[A-Za-z0-9_]*]], [[D18:[A-Za-z0-9_]*]]>([[INDICES]]) {{.*}} : (!mo.tensor<[[[RMO_D21]], [[RMO_D22]]], si64>) -> !mo.tensor<[[[D17]], [[D18]]], si64>
//  CHECK-NEXT:   mo.output<[[D15]], [[D16]], [[D17]], [[D18]]> [[V8]], [[V9]] : !mo.tensor<[[[D15]], [[D16]]], f32>, !mo.tensor<[[[D17]], [[D18]]], si64>


// -----

mo.graph @topk_dyn_params_2(%input: !mo.tensor<[?, ?], f32>, %k: !mo.tensor<?, si64>, %axis: !mo.tensor<[], si64>, %sorted: !mo.tensor<[], bool>) -> (!mo.tensor<[?, ?], f32>, !mo.tensor<?, si64>) {
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<[?, ?], f32>, !mo.tensor<?, si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[?, ?], f32>, !mo.tensor<?, si64>
      )
  mo.output %values, %indices : !mo.tensor<[?, ?], f32>, !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @topk_dyn_params_2<[[D1:[A-Za-z0-9_]*]], [[D2:[A-Za-z0-9_]*]], [[SH3:[A-Za-z0-9_]*]]: !mosh.ape -> [[D4:[A-Za-z0-9_]*]], [[D5:[A-Za-z0-9_]*]], [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG2:%.*]]: !mo.tensor<[], si64>, [[ARG3:%.*]]: !mo.tensor<[], bool>) -> (!mo.tensor<[[[D4]], [[D5]]], f32>, !mo.tensor<[[SH6]], si64>) {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D7:[A-Za-z0-9_]*]], [[D8:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], [[D2]]], f32> -> !mo.tensor<[[[D7]], [[D8]]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH9]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D13:[A-Za-z0-9_]*]], [[RMO_D14:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[D7]], [[D8]]], f32>) -> !mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>
//  CHECK-NEXT:   [[V3:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH9]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[VALUES:%.*]], [[INDICES:%.*]] = mo.top_k<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]]>([[V2]], [[V3]], [[ARG2]], [[ARG3]]) : (!mo.tensor<[[[RMO_D13]], [[RMO_D14]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>, !mo.tensor<[[[RMO_D15]], [[RMO_D16]]], si64>)
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[D10:[A-Za-z0-9_]*]], [[D11:[A-Za-z0-9_]*]]>([[VALUES]]) {{.*}} : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>) -> !mo.tensor<[[[D10]], [[D11]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[INDICES]]) {{.*}} : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]]], si64>) -> !mo.tensor<[[SH12]], si64>
//  CHECK-NEXT:   mo.output<[[D10]], [[D11]], :!mosh.ape [[SH12]]> [[V4]], [[V5]] : !mo.tensor<[[[D10]], [[D11]]], f32>, !mo.tensor<[[SH12]], si64>


// -----

mo.graph @topk_dyn_mix(%input: !mo.tensor<[?, 3], f32>, %k: !mo.tensor<?, si64>, %axis: !mo.tensor<?, si64>, %sorted: !mo.tensor<?, bool>) -> (!mo.tensor<[?, 3], f32>, !mo.tensor<[?, ?], si64>) {
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<[?, 3], f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, bool>
      ) -> (
        !mo.tensor<[?, 3], f32>, !mo.tensor<[?, ?], si64>
      )
  mo.output %values, %indices : !mo.tensor<[?, 3], f32>, !mo.tensor<[?, ?], si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @topk_dyn_mix<[[D1:[A-Za-z0-9_]*]], [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[D5:[A-Za-z0-9_]*]], [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[[D1]], 3], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG3:%.*]]: !mo.tensor<[[SH4]], bool>) -> (!mo.tensor<[[[D5]], 3], f32>, !mo.tensor<[[[D6]], [[D7]]], si64>) {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[D8:[A-Za-z0-9_]*]]>([[ARG0]]) : !mo.tensor<[[[D1]], 3], f32> -> !mo.tensor<[[[D8]], 3], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH9]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH10]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH4]], bool> -> !mo.tensor<[[SH11]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[[D8]], 3], f32>) -> !mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH9]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH10]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH11]], bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   [[VALUES:%.*]], [[INDICES:%.*]] = mo.top_k<() -> [[RMO_D17:[A-Za-z0-9_]*]], [[RMO_D18:[A-Za-z0-9_]*]]>([[V4]], [[V5]], [[V6]], [[V7]]) : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[[RMO_D17]], [[RMO_D18]]], f32>, !mo.tensor<[[[RMO_D17]], [[RMO_D18]]], si64>)
//  CHECK-NEXT:   [[V8:%.*]] = rmo.rebind_tensor_shape<() -> [[D12:[A-Za-z0-9_]*]]>([[VALUES]]) {{.*}} : (!mo.tensor<[[[RMO_D17]], [[RMO_D18]]], f32>) -> !mo.tensor<[[[D12]], 3], f32>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape<() -> [[D13:[A-Za-z0-9_]*]], [[D14:[A-Za-z0-9_]*]]>([[INDICES]]) {{.*}} : (!mo.tensor<[[[RMO_D17]], [[RMO_D18]]], si64>) -> !mo.tensor<[[[D13]], [[D14]]], si64>
//  CHECK-NEXT:   mo.output<[[D12]], [[D13]], [[D14]]> [[V8]], [[V9]] : !mo.tensor<[[[D12]], 3], f32>, !mo.tensor<[[[D13]], [[D14]]], si64>

// -----

mo.graph @topk_static(%input: !mo.tensor<[4, 6], f32>, %k: !mo.tensor<[], si64>, %axis: !mo.tensor<[], si64>, %sorted: !mo.tensor<[], bool>) -> (!mo.tensor<?, f32>, !mo.tensor<?, si64>) {
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<[4, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>
      )
  mo.output %values, %indices : !mo.tensor<?, f32>, !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @topk_static<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[4, 6], f32>, [[ARG1:%.*]]: !mo.tensor<[], si64>, [[ARG2:%.*]]: !mo.tensor<[], si64>, [[ARG3:%.*]]: !mo.tensor<[], bool>) -> (!mo.tensor<[[SH1]], f32>, !mo.tensor<[[SH2]], si64>) {
//  CHECK-NEXT:   [[V0:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D5:[A-Za-z0-9_]*]], [[RMO_D6:[A-Za-z0-9_]*]]>([[ARG0]]) {{.*}} : (!mo.tensor<[4, 6], f32>) -> !mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>
//  CHECK-NEXT:   [[VALUES:%.*]], [[INDICES:%.*]] = mo.top_k<() -> [[RMO_D7:[A-Za-z0-9_]*]], [[RMO_D8:[A-Za-z0-9_]*]]>([[V0]], [[ARG1]], [[ARG2]], [[ARG3]]) : (!mo.tensor<[[[RMO_D5]], [[RMO_D6]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>, !mo.tensor<[[[RMO_D7]], [[RMO_D8]]], si64>)
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH3:[A-Za-z0-9_]*]]: !mosh.ape>([[VALUES]]) {{.*}} : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]]], f32>) -> !mo.tensor<[[SH3]], f32>
//  CHECK-NEXT:   [[V2:%.*]] = rmo.rebind_tensor_shape<() -> [[SH4:[A-Za-z0-9_]*]]: !mosh.ape>([[INDICES]]) {{.*}} : (!mo.tensor<[[[RMO_D7]], [[RMO_D8]]], si64>) -> !mo.tensor<[[SH4]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH3]], :!mosh.ape [[SH4]]> [[V1]], [[V2]] : !mo.tensor<[[SH3]], f32>, !mo.tensor<[[SH4]], si64>

// -----

mo.graph @bottomk_dyn(%input: !mo.tensor<?, f32>, %k: !mo.tensor<?, si64>, %axis: !mo.tensor<?, si64>, %sorted: !mo.tensor<?, bool>) -> (!mo.tensor<?, f32>, !mo.tensor<?, si64>) {
  %values, %indices = rmo.mo.bottom_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, bool>
      ) -> (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>
      )
  mo.output %values, %indices : !mo.tensor<?, f32>, !mo.tensor<?, si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @bottomk_dyn<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape, [[SH6:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG3:%.*]]: !mo.tensor<[[SH4]], bool>) -> (!mo.tensor<[[SH5]], f32>, !mo.tensor<[[SH6]], si64>) {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH7:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH7]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH8]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH9]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH4]], bool> -> !mo.tensor<[[SH10]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_SH13:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[SH7]], f32>) -> !mo.tensor<[[RMO_SH13]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH8]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH9]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH10]], bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   [[VALUES:%.*]], [[INDICES:%.*]] = mo.bottom_k<() -> [[RMO_SH14:[A-Za-z0-9_]*]]: !mosh.ape>([[V4]], [[V5]], [[V6]], [[V7]]) : (!mo.tensor<[[RMO_SH13]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[RMO_SH14]], f32>, !mo.tensor<[[RMO_SH14]], si64>)
//  CHECK-NEXT:   [[V8:%.*]] = rmo.rebind_tensor_shape<() -> [[SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[VALUES]]) {{.*}} : (!mo.tensor<[[RMO_SH14]], f32>) -> !mo.tensor<[[SH11]], f32>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[INDICES]]) {{.*}} : (!mo.tensor<[[RMO_SH14]], si64>) -> !mo.tensor<[[SH12]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH11]], :!mosh.ape [[SH12]]> [[V8]], [[V9]] : !mo.tensor<[[SH11]], f32>, !mo.tensor<[[SH12]], si64>

// -----

mo.graph @bottomk_dyn_out_param(%input: !mo.tensor<?, f32>, %k: !mo.tensor<?, si64>, %axis: !mo.tensor<?, si64>, %sorted: !mo.tensor<?, bool>) -> (!mo.tensor<?, f32>, !mo.tensor<[?, ?], si64>) {
  %values, %indices = rmo.mo.bottom_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, bool>
      ) -> (
        !mo.tensor<?, f32>, !mo.tensor<[?, ?], si64>
      )
  mo.output %values, %indices : !mo.tensor<?, f32>, !mo.tensor<[?, ?], si64>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @bottomk_dyn_out_param<[[SH1:[A-Za-z0-9_]*]]: !mosh.ape, [[SH2:[A-Za-z0-9_]*]]: !mosh.ape, [[SH3:[A-Za-z0-9_]*]]: !mosh.ape, [[SH4:[A-Za-z0-9_]*]]: !mosh.ape -> [[SH5:[A-Za-z0-9_]*]]: !mosh.ape, [[D6:[A-Za-z0-9_]*]], [[D7:[A-Za-z0-9_]*]]>([[ARG0:%.*]]: !mo.tensor<[[SH1]], f32>, [[ARG1:%.*]]: !mo.tensor<[[SH2]], si64>, [[ARG2:%.*]]: !mo.tensor<[[SH3]], si64>, [[ARG3:%.*]]: !mo.tensor<[[SH4]], bool>) -> (!mo.tensor<[[SH5]], f32>, !mo.tensor<[[[D6]], [[D7]]], si64>) {
//  CHECK-NEXT:   [[V0:%.*]] = mo.rebind<() -> [[SH8:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH8]], f32>
//  CHECK-NEXT:   [[V1:%.*]] = mo.rebind<() -> [[SH9:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG1]]) : !mo.tensor<[[SH2]], si64> -> !mo.tensor<[[SH9]], si64>
//  CHECK-NEXT:   [[V2:%.*]] = mo.rebind<() -> [[SH10:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG2]]) : !mo.tensor<[[SH3]], si64> -> !mo.tensor<[[SH10]], si64>
//  CHECK-NEXT:   [[V3:%.*]] = mo.rebind<() -> [[SH11:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG3]]) : !mo.tensor<[[SH4]], bool> -> !mo.tensor<[[SH11]], bool>
//  CHECK-NEXT:   [[V4:%.*]] = rmo.rebind_tensor_shape<() -> [[RMO_D15:[A-Za-z0-9_]*]], [[RMO_D16:[A-Za-z0-9_]*]]>([[V0]]) {{.*}} : (!mo.tensor<[[SH8]], f32>) -> !mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>
//  CHECK-NEXT:   [[V5:%.*]] = rmo.rebind_tensor_shape([[V1]]) {{.*}} : (!mo.tensor<[[SH9]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V6:%.*]] = rmo.rebind_tensor_shape([[V2]]) {{.*}} : (!mo.tensor<[[SH10]], si64>) -> !mo.tensor<[], si64>
//  CHECK-NEXT:   [[V7:%.*]] = rmo.rebind_tensor_shape([[V3]]) {{.*}} : (!mo.tensor<[[SH11]], bool>) -> !mo.tensor<[], bool>
//  CHECK-NEXT:   [[VALUES:%.*]], [[INDICES:%.*]] = mo.bottom_k<() -> [[RMO_D17:[A-Za-z0-9_]*]], [[RMO_D18:[A-Za-z0-9_]*]]>([[V4]], [[V5]], [[V6]], [[V7]]) : (!mo.tensor<[[[RMO_D15]], [[RMO_D16]]], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[[RMO_D17]], [[RMO_D18]]], f32>, !mo.tensor<[[[RMO_D17]], [[RMO_D18]]], si64>)
//  CHECK-NEXT:   [[V8:%.*]] = rmo.rebind_tensor_shape<() -> [[SH12:[A-Za-z0-9_]*]]: !mosh.ape>([[VALUES]]) {{.*}} : (!mo.tensor<[[[RMO_D17]], [[RMO_D18]]], f32>) -> !mo.tensor<[[SH12]], f32>
//  CHECK-NEXT:   [[V9:%.*]] = rmo.rebind_tensor_shape<() -> [[D13:[A-Za-z0-9_]*]], [[D14:[A-Za-z0-9_]*]]>([[INDICES]]) {{.*}} : (!mo.tensor<[[[RMO_D17]], [[RMO_D18]]], si64>) -> !mo.tensor<[[[D13]], [[D14]]], si64>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH12]], [[D13]], [[D14]]> [[V8]], [[V9]] : !mo.tensor<[[SH12]], f32>, !mo.tensor<[[[D13]], [[D14]]], si64>
