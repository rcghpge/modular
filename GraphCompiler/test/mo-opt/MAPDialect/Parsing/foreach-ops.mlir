// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt

// Round-trip the map.iter.foreach family of ops.
// Covers both fully-static shapes and MOSH symbolic-dim shapes.
// Value names are matched with capture variables so changes in the printer's
// SSA numbering don't break the test.

module {

// Static shape: verify map.iter.foreach<[8, 16, 128]> round-trips unchanged.
// CHECK-LABEL: mo.graph @elementwiseAdd
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.foreach<[8, 16, 128]>
// CHECK-NEXT:    inputs  ([[X:%[^ :,]+]]: !mo.tensor<[8, 16, 128], f32>, [[Y:%[^ :,]+]]: !mo.tensor<[8, 16, 128], f32>)
// CHECK-NEXT:    outputs ([[OUTARG:%[^ :,]+]]: !mo.tensor<[8, 16, 128], f32>)
// CHECK-NEXT:    index   ([[IDX:%[^ :,]+]]: !map.index<3>) -> !mo.tensor<[8, 16, 128], f32> {
// CHECK-NEXT:   [[XV:%[^ ]+]] = map.load [[X]][[[IDX]]] : !mo.tensor<[8, 16, 128], f32>
// CHECK-NEXT:   [[YV:%[^ ]+]] = map.load [[Y]][[[IDX]]] : !mo.tensor<[8, 16, 128], f32>
// CHECK-NEXT:   [[RES:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Add([[IDX]], [[XV]], [[YV]]) : !map.index<3>, f32, f32 -> f32
// CHECK-NEXT:   map.store [[OUTARG]][[[IDX]]] = [[RES]] : !mo.tensor<[8, 16, 128], f32>
// CHECK-NEXT: }
// CHECK-NEXT:   mo.output [[RESULT]] : !mo.tensor<[8, 16, 128], f32>
mo.graph @elementwiseAdd(%x: !mo.tensor<[8, 16, 128], f32>,
                         %y: !mo.tensor<[8, 16, 128], f32>)
    -> !mo.tensor<[8, 16, 128], f32> {
  %output = map.iter.foreach<[8, 16, 128]>
      inputs  (%x: !mo.tensor<[8, 16, 128], f32>,
               %y: !mo.tensor<[8, 16, 128], f32>)
      outputs (%out: !mo.tensor<[8, 16, 128], f32>)
      index   (%idx: !map.index<3>)
      -> !mo.tensor<[8, 16, 128], f32>
  {
    %xv = map.load %x[%idx] : !mo.tensor<[8, 16, 128], f32>
    %yv = map.load %y[%idx] : !mo.tensor<[8, 16, 128], f32>
    %outv = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xv, %yv)
                : !map.index<3>, f32, f32 -> f32
    map.store %out[%idx] = %outv : !mo.tensor<[8, 16, 128], f32>
  }
  mo.output %output : !mo.tensor<[8, 16, 128], f32>
}

// Symbolic shape: named MOSH dims in the shape attr.
// CHECK-LABEL: mo.graph @elementwiseAddDynamic
// CHECK:         [[RESULT2:%[^ ]+]] = map.iter.foreach<[D0, D1]>
// CHECK-NEXT:    inputs  ([[X2:%[^ :,]+]]: !mo.tensor<[D0, D1], f32>, [[Y2:%[^ :,]+]]: !mo.tensor<[D0, D1], f32>)
// CHECK-NEXT:    outputs ([[OUTARG2:%[^ :,]+]]: !mo.tensor<[D0, D1], f32>)
// CHECK-NEXT:    index   ([[IDX2:%[^ :,]+]]: !map.index<2>) -> !mo.tensor<[D0, D1], f32> {
mo.graph @elementwiseAddDynamic<D0, D1>(
    %x: !mo.tensor<[D0, D1], f32>,
    %y: !mo.tensor<[D0, D1], f32>)
    -> !mo.tensor<[D0, D1], f32> {
  %output = map.iter.foreach<[D0, D1]>
      inputs  (%x: !mo.tensor<[D0, D1], f32>,
               %y: !mo.tensor<[D0, D1], f32>)
      outputs (%out: !mo.tensor<[D0, D1], f32>)
      index   (%idx: !map.index<2>)
      -> !mo.tensor<[D0, D1], f32>
  {
    %xv = map.load %x[%idx] : !mo.tensor<[D0, D1], f32>
    %yv = map.load %y[%idx] : !mo.tensor<[D0, D1], f32>
    %outv = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xv, %yv)
                : !map.index<2>, f32, f32 -> f32
    map.store %out[%idx] = %outv : !mo.tensor<[D0, D1], f32>
  }
  mo.output %output : !mo.tensor<[D0, D1], f32>
}

}
