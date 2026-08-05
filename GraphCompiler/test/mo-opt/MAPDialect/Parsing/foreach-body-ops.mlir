// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt

module {

// CHECK-LABEL: mo.graph @foreachBodyStatic
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[X:%[^ :,]+]] : !mo.tensor<[8, 16, 128], f32>, [[Y:%[^ :,]+]] : !mo.tensor<[8, 16, 128], f32>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[8, 16, 128], f32>) {
// CHECK-NEXT:      map.iter.foreach.body<[8, 16, 128]>
// CHECK-NEXT:        index   ([[IDX:%[^ :,]+]]: !map.index<3>) {
// CHECK-NEXT:      [[XV:%[^ ]+]] = map.load [[X]][[[IDX]]] : !mo.tensor<[8, 16, 128], f32>
// CHECK-NEXT:      [[YV:%[^ ]+]] = map.load [[Y]][[[IDX]]] : !mo.tensor<[8, 16, 128], f32>
// CHECK-NEXT:      [[SUM:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Add([[IDX]], [[XV]], [[YV]]) : !map.index<3>, f32, f32 -> f32
// CHECK-NEXT:      map.store [[OUT]][[[IDX]]] = [[SUM]] : !mo.tensor<[8, 16, 128], f32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  mo.output [[RESULT]] : !mo.tensor<[8, 16, 128], f32>
mo.graph @foreachBodyStatic(%x: !mo.tensor<[8, 16, 128], f32>,
                            %y: !mo.tensor<[8, 16, 128], f32>)
    -> !mo.tensor<[8, 16, 128], f32> {
  %r = map.iter.opaque
      inputs (%x : !mo.tensor<[8, 16, 128], f32>, %y : !mo.tensor<[8, 16, 128], f32>)
      outputs (%out : !mo.tensor<[8, 16, 128], f32>) {
    map.iter.foreach.body<[8, 16, 128]>
        index (%idx: !map.index<3>) {
      %xv = map.load %x[%idx] : !mo.tensor<[8, 16, 128], f32>
      %yv = map.load %y[%idx] : !mo.tensor<[8, 16, 128], f32>
      %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xv, %yv)
                : !map.index<3>, f32, f32 -> f32
      map.store %out[%idx] = %sum : !mo.tensor<[8, 16, 128], f32>
    }
  }
  mo.output %r : !mo.tensor<[8, 16, 128], f32>
}

// CHECK-LABEL: mo.graph @foreachBodyDynamic
// CHECK:         [[RESULT2:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[X2:%[^ :,]+]] : !mo.tensor<[D0, D1], f32>, [[Y2:%[^ :,]+]] : !mo.tensor<[D0, D1], f32>)
// CHECK-NEXT:    outputs ([[OUT2:%[^ :,]+]] : !mo.tensor<[D0, D1], f32>) {
// CHECK-NEXT:      map.iter.foreach.body<[D0, D1]>
// CHECK-NEXT:        index   ([[IDX2:%[^ :,]+]]: !map.index<2>) {
mo.graph @foreachBodyDynamic<D0, D1>(
    %x: !mo.tensor<[D0, D1], f32>,
    %y: !mo.tensor<[D0, D1], f32>)
    -> !mo.tensor<[D0, D1], f32> {
  %r = map.iter.opaque
      inputs (%x : !mo.tensor<[D0, D1], f32>, %y : !mo.tensor<[D0, D1], f32>)
      outputs (%out : !mo.tensor<[D0, D1], f32>) {
    map.iter.foreach.body<[D0, D1]>
        index (%idx: !map.index<2>) {
      %xv = map.load %x[%idx] : !mo.tensor<[D0, D1], f32>
      %yv = map.load %y[%idx] : !mo.tensor<[D0, D1], f32>
      %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xv, %yv)
                : !map.index<2>, f32, f32 -> f32
      map.store %out[%idx] = %sum : !mo.tensor<[D0, D1], f32>
    }
  }
  mo.output %r : !mo.tensor<[D0, D1], f32>
}

}
