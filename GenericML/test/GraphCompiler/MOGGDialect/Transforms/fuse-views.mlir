// RUN: mo-opt --mo-to-mogg --simplify-kernel-operands --fuse-elementwise --canonicalize --cse --fuse-views --canonicalize --cse %s | FileCheck %s

// Expect a kernel with 3 bcasts, 2 for index, 1 for updates
// CHECK-LABEL: mo.graph @advanced_indexing_setitem
// CHECK: summary = "mo.static.reshape:{{[0-9]+}}+mo.static.broadcast_to:{{[0-9]+}}+mo.static.broadcast_to:{{[0-9]+}}+mo.static.broadcast_to:{{[0-9]+}}+custom__advanced_indexing_setitem:{{[0-9]+}}"
mo.graph @advanced_indexing_setitem() -> !mo.tensor<[2, 3, 4, 5], si32> {
  %start = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %stop = mo.constant {value = #M.dense_array<120> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %step = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[], si32>

  // Setup your buffer
  %input = mo.range(%start, %stop, %step) : (
    !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>
  ) -> !mo.tensor<[120], si32>
  %input_reshape = mo.static.reshape(%input): !mo.tensor<[120], si32> -> !mo.tensor<[2, 3, 4, 5], si32>

  %data = mo.constant {value = #M.dense_array<-1, -2, -3, -4, -5, -6, -7, -8, -9> : tensor<9xsi32>} : !mo.tensor<[3, 3], si32>
  %data_bcast = mo.static.broadcast_to(%data): !mo.tensor<[3, 3], si32> -> !mo.tensor<[2, 3, 3, 3], si32>
  %index_tensor1 = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[1, 1], si32>
  %index_tensor1_bcast = mo.static.broadcast_to(%index_tensor1): !mo.tensor<[1, 1], si32> -> !mo.tensor<[3, 3], si32>
  %index_tensor2 = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[1, 1], si32>
  %index_tensor2_bcast = mo.static.broadcast_to(%index_tensor2): !mo.tensor<[1, 1], si32> -> !mo.tensor<[3, 3], si32>

  %out = mo.custom {
    symbol = "advanced_indexing_setitem",
    device = #M.device_ref<"cpu", 0>,
    parameters = { start_axis = 2 }
  } (%input_reshape, %data_bcast, %index_tensor1_bcast, %index_tensor2_bcast) : (
    !mo.tensor<[2, 3, 4, 5], si32>, !mo.tensor<[2, 3, 3, 3], si32>, !mo.tensor<[3, 3], si32>, !mo.tensor<[3, 3], si32>
  ) -> !mo.tensor<[2, 3, 4, 5], si32>

  mo.output %out: !mo.tensor<[2, 3, 4, 5], si32>
}

// CHECK-LABEL: @forced_view_fusion
// CHECK: mogg.kernel
// CHECK: summary = "mo.slice:{{[0-9]+}}+mo.matmul:{{[0-9]+}}"
// CHECK-NOT: mogg.kernel
mo.graph @forced_view_fusion(%input: !mo.tensor<[10, 2], si32>) -> (!mo.tensor<[2, 2], si32>) {
	%start = mo.constant {value = #M.dense_array<8, 0> : tensor<2xsi32>} : !mo.tensor<[2], si32>
	%stop = mo.constant {value = #M.dense_array<10, 2> : tensor<2xsi32>} : !mo.tensor<[2], si32>
	%step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi32>} : !mo.tensor<[2], si32>
	%res = mo.slice(%input, %start, %stop, %step) : (!mo.tensor<[10, 2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 2], si32>
  %mm = mo.matmul(%res, %res) : (!mo.tensor<[2, 2], si32>, !mo.tensor<[2, 2], si32>)  -> !mo.tensor<[2, 2], si32>
	mo.output %mm : !mo.tensor<[2, 2], si32>
}
