// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck %s

// CHECK: mo.graph @buffer_type
// CHECK-SAME:   {{.*}}: !mo.buffer<[10, 20], f32>,
// CHECK-SAME:   {{.*}}: !mo.buffer<[D0, D1], f32>,
// CHECK-SAME:   {{.*}}: !mo.buffer<Sh0, f32>)
mo.graph @buffer_type<D0, D1, Sh0: !mosh.ape>(
  %b0: !mo.buffer<[10, 20], f32>,
  %b3: !mo.buffer<[D0, D1], f32>,
  %b4: !mo.buffer<Sh0, f32>
) {
  mo.output
}
