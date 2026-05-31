// TODO(GEX-1573): Convert the following sniff test to a complete sniff test.
// RUN: mo-opt %s --mogg-import-packages --mo-to-mogg

// Just check we can compile it.

module {
  mo.graph @dlrm<D473, D474>(%arg0: !mo.tensor<[D473, 8, 100], si32>, %arg1: !mo.tensor<[D474, 256], f32>, %arg2: !mo.tensor<[200000, 32], f32>, %arg3: !mo.tensor<[200000, 32], f32>, %arg4: !mo.tensor<[200000, 32], f32>, %arg5: !mo.tensor<[200000, 32], f32>, %arg6: !mo.tensor<[200000, 32], f32>, %arg7: !mo.tensor<[200000, 32], f32>, %arg8: !mo.tensor<[200000, 32], f32>, %arg9: !mo.tensor<[200000, 32], f32>, %arg10: !mo.tensor<[256, 64], f32, {layout = #mo.layout<KN>}>, %arg11: !mo.tensor<[77, 256], f32, {layout = #mo.layout<KN>}>, %arg12: !mo.tensor<[128, 32], f32, {layout = #mo.layout<KN>}>, %arg13: !mo.tensor<[256, 128], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[D474, 1], f32> {
    %0 = mo.constant {value = #M.dense_array<2> : !M.array<1xsi64>} : !mo.tensor<[], si64>
    %1 = mo.constant {value = #M.dense_array<0> : !M.array<1xsi64>} : !mo.tensor<[], si64>
    %2 = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
    %3 = mo.constant {value = #M.dense_array<9223372036854775807, 800> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %4 = mo.constant {value = #M.dense_array<0, 700> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %5 = mo.constant {value = #M.dense_array<9223372036854775807, 700> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %6 = mo.constant {value = #M.dense_array<0, 600> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %7 = mo.constant {value = #M.dense_array<9223372036854775807, 600> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %8 = mo.constant {value = #M.dense_array<0, 500> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %9 = mo.constant {value = #M.dense_array<9223372036854775807, 500> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %10 = mo.constant {value = #M.dense_array<0, 400> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %11 = mo.constant {value = #M.dense_array<9223372036854775807, 400> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %12 = mo.constant {value = #M.dense_array<0, 300> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %13 = mo.constant {value = #M.dense_array<9223372036854775807, 300> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %14 = mo.constant {value = #M.dense_array<0, 200> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %15 = mo.constant {value = #M.dense_array<9223372036854775807, 200> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %16 = mo.constant {value = #M.dense_array<0, 100> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %17 = mo.constant {value = #M.dense_array<9223372036854775807, 100> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %18 = mo.constant {value = #M.dense_array<0, 0> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %19 = mo.constant {value = #M.dense_array<1, 1> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %20 = mo.constant {value = #M.dense_array<81> : tensor<si64>} : !mo.tensor<[], si64>
    %21 = mo.constant {value = #M.dense_array<1, 0> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
    %22 = mo.constant {value = #M.dense_array<0> : !M.array<1xsi64>} : !mo.tensor<[], si64>
    %23 = mo.constant {value = #M.dense_array<-1> : tensor<si32>} : !mo.tensor<[], si32>
    %24 = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
    %25 = mo.constant {value = #M.dense_array<-0.121941715, -0.112631559, 0.199184537, 0.260294795, -0.190447032, 0.214445293, 0.0505436063, 0.254368067, 0.18954441, -0.156480938, -0.0834032595, -1.882630e-01, -0.125319079, -0.225907743, 0.122629285, -0.0715062469, -0.184827894, -0.228427023, 0.229661107, 0.0153187215, -0.118268147, 0.159035563, 0.0799599587, 0.0667982101, 0.0901305377, -0.0790597349, 0.04434672, -0.278808236, 0.300504982, 0.0512512624, 0.156755388, -0.141259581, -0.0541819781, -0.207770735, -0.232766122, -0.180529654, 0.185262382, -0.155863777, -5.34206629E-4, 0.194931746, -3.772080e-02, 0.188996345, 0.128953159, 0.273819387, -0.0088250339, 0.191499621, 0.204237163, -0.0390079916, -0.292554468, 0.260827839, -0.114037767, 0.249398828, -0.0697858631, -0.271265894, -0.0694437474, 0.141885012, 8.175060e-02, 0.289748192, 0.165302575, 0.228590071, -0.0586845726, -0.22602132, 0.19075048, 0.0257503986> : tensor<64x1xf32>} : !mo.tensor<[64, 1], f32, {layout = #mo.layout<KN>}>
    %26 = mo.constant {value = #M.dense_array<0> : tensor<i8>} : !mo.tensor<[], bool>
    %27 = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
    %28 = mo.constant {value = #M.dense_array<1> : tensor<si32>} : !mo.tensor<[], si32>
    %29 = mo.static.reshape(%arg0) : !mo.tensor<[D473, 8, 100], si32> -> !mo.tensor<[D473, 800], si32>
    %37 = mo.slice(%29, %18, %17, %19) : (!mo.tensor<[D473, 800], si32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D473, 100], si32>
    %43 = mo.slice(%29, %16, %15, %19) : (!mo.tensor<[D473, 800], si32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D473, 100], si32>
    %49 = mo.slice(%29, %14, %13, %19) : (!mo.tensor<[D473, 800], si32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D473, 100], si32>
    %55 = mo.slice(%29, %12, %11, %19) : (!mo.tensor<[D473, 800], si32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D473, 100], si32>
    %61 = mo.slice(%29, %10, %9, %19) : (!mo.tensor<[D473, 800], si32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D473, 100], si32>
    %67 = mo.slice(%29, %8, %7, %19) : (!mo.tensor<[D473, 800], si32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D473, 100], si32>
    %73 = mo.slice(%29, %6, %5, %19) : (!mo.tensor<[D473, 800], si32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D473, 100], si32>
    %79 = mo.slice(%29, %4, %3, %19) : (!mo.tensor<[D473, 800], si32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D473, 100], si32>
    %80 = "mo.gather_sum"(%arg9, %37) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[D473, 100], si32>) -> !mo.tensor<[D473, 1, 32], f32>
    %81 = mo.static.reshape(%80) : !mo.tensor<[D473, 1, 32], f32> -> !mo.tensor<[D473, 32], f32>
    %82 = "mo.gather_sum"(%arg8, %43) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[D473, 100], si32>) -> !mo.tensor<[D473, 1, 32], f32>
    %83 = mo.static.reshape(%82) : !mo.tensor<[D473, 1, 32], f32> -> !mo.tensor<[D473, 32], f32>
    %84 = "mo.gather_sum"(%arg7, %49) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[D473, 100], si32>) -> !mo.tensor<[D473, 1, 32], f32>
    %85 = mo.static.reshape(%84) : !mo.tensor<[D473, 1, 32], f32> -> !mo.tensor<[D473, 32], f32>
    %86 = "mo.gather_sum"(%arg6, %55) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[D473, 100], si32>) -> !mo.tensor<[D473, 1, 32], f32>
    %87 = mo.static.reshape(%86) : !mo.tensor<[D473, 1, 32], f32> -> !mo.tensor<[D473, 32], f32>
    %88 = "mo.gather_sum"(%arg5, %61) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[D473, 100], si32>) -> !mo.tensor<[D473, 1, 32], f32>
    %89 = mo.static.reshape(%88) : !mo.tensor<[D473, 1, 32], f32> -> !mo.tensor<[D473, 32], f32>
    %90 = "mo.gather_sum"(%arg4, %67) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[D473, 100], si32>) -> !mo.tensor<[D473, 1, 32], f32>
    %91 = mo.static.reshape(%90) : !mo.tensor<[D473, 1, 32], f32> -> !mo.tensor<[D473, 32], f32>
    %92 = "mo.gather_sum"(%arg3, %73) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[D473, 100], si32>) -> !mo.tensor<[D473, 1, 32], f32>
    %93 = mo.static.reshape(%92) : !mo.tensor<[D473, 1, 32], f32> -> !mo.tensor<[D473, 32], f32>
    %94 = "mo.gather_sum"(%arg2, %79) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[D473, 100], si32>) -> !mo.tensor<[D473, 1, 32], f32>
    %95 = mo.static.reshape(%94) : !mo.tensor<[D473, 1, 32], f32> -> !mo.tensor<[D473, 32], f32>
    %96 = mo.matmul(%arg1, %arg13) : (!mo.tensor<[D474, 256], f32>, !mo.tensor<[256, 128], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[D474, 128], f32>
    %97 = mo.relu(%96) : !mo.tensor<[D474, 128], f32>
    %98 = mo.matmul(%97, %arg12) : (!mo.tensor<[D474, 128], f32>, !mo.tensor<[128, 32], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[D474, 32], f32>
    %99 = mo.relu(%98) : !mo.tensor<[D474, 32], f32>
    kgen.param.declare assert555: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] input shapes must match except at concat axis" : !kgen.string}, eq(:scalar<si64> D473, D474))>
    kgen.param.declare assert557: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] input shapes must match except at concat axis" : !kgen.string}, eq(:scalar<si64> D473, D474))>
    kgen.param.declare assert559: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] input shapes must match except at concat axis" : !kgen.string}, eq(:scalar<si64> D473, D474))>
    kgen.param.declare assert561: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] input shapes must match except at concat axis" : !kgen.string}, eq(:scalar<si64> D473, D474))>
    kgen.param.declare assert563: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] input shapes must match except at concat axis" : !kgen.string}, eq(:scalar<si64> D473, D474))>
    kgen.param.declare assert565: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] input shapes must match except at concat axis" : !kgen.string}, eq(:scalar<si64> D473, D474))>
    kgen.param.declare assert567: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] input shapes must match except at concat axis" : !kgen.string}, eq(:scalar<si64> D473, D474))>
    kgen.param.declare assert569: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] input shapes must match except at concat axis" : !kgen.string}, eq(:scalar<si64> D473, D474))>
    %100 = mo.concat[1] (%99, %81, %83, %85, %87, %89, %91, %93, %95) : (!mo.tensor<[D474, 32], f32>, !mo.tensor<[D473, 32], f32>, !mo.tensor<[D473, 32], f32>, !mo.tensor<[D473, 32], f32>, !mo.tensor<[D473, 32], f32>, !mo.tensor<[D473, 32], f32>, !mo.tensor<[D473, 32], f32>, !mo.tensor<[D473, 32], f32>, !mo.tensor<[D473, 32], f32>) -> !mo.tensor<[D474, 288], f32>
    %101 = mo.static.reshape(%100) : !mo.tensor<[D474, 288], f32> -> !mo.tensor<[D474, 9, 32], f32>
    %102 = mo.batch_matmul(%101, %101) {transpose_b = true} : (!mo.tensor<[D474, 9, 32], f32>, !mo.tensor<[D474, 9, 32], f32>) -> !mo.tensor<[D474, 9, 9], f32>
    %103 = mo.merge_dim[1] (%102) : (!mo.tensor<[D474, 9, 9], f32>) -> !mo.tensor<[D474, 81], f32>
    %104 = mo.merge_dim[0] (%103) : (!mo.tensor<[D474, 81], f32>) -> !mo.tensor<[param_expr_627], f32>
    %105 = mo.static.broadcast_to(%24) : !mo.tensor<[], f32> -> !mo.tensor<[9, 9], f32>
    %106 = mo.linalg.band_part(%105, %27, %23, %26) : (!mo.tensor<[9, 9], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[9, 9], f32>
    %107 = mo.merge_dim[0] (%106) : (!mo.tensor<[9, 9], f32>) -> !mo.tensor<[81], f32>
    mo.invoke_shape_func["mo.arg_nonzero"]<() -> W619, F621>(%107) {dataDeptTensors = [0 : i32]} : (!mo.tensor<[81], f32>)
    %108 = mo.arg_nonzero(%107) : (!mo.tensor<[81], f32>) -> !mo.tensor<[W619, 1], si64>
    %109 = mosh.param.to_value = <mul_no_wrap(D474, 81)>
    %110 = mo.index.to_tensor(%109) -> !mo.tensor<[], si64>
    %111 = mo.range(%22, %110, %20) : (!mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[D474], si64>
    %112 = mo.add_singleton_dim[1] (%111) : (!mo.tensor<[D474], si64>) -> !mo.tensor<[D474, 1], si64>
    %113 = mo.static.broadcast_to(%108) : !mo.tensor<[W619, 1], si64> -> !mo.tensor<[W619, D474], si64>
    %114 = mo.transpose(%113, %21) : (!mo.tensor<[W619, D474], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D474, W619], si64>
    %115 = mo.static.broadcast_to(%112) : !mo.tensor<[D474, 1], si64> -> !mo.tensor<[D474, W619], si64>
    %116 = mo.add(%114, %115) : !mo.tensor<[D474, W619], si64>
    %117 = mo.merge_dim[0] (%116) : (!mo.tensor<[D474, W619], si64>) -> !mo.tensor<[param_expr_625], si64>
    kgen.param.declare param_expr_627: scalar<si64> = <mul_no_wrap(D474, 81)>
    %118 = mo.gather(%104, %117) {axis = 0 : index} : (!mo.tensor<[param_expr_627], f32>, !mo.tensor<[param_expr_625], si64>) -> !mo.tensor<[param_expr_625], f32>
    kgen.param.declare assert599: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[reshape] input and output number of elements must match: #mosh<ape[D52]> : !mosh.ape != #mosh<ape[var_div596, 45]> : !mosh.ape" : !kgen.string}, to_builtin(:scalar<bool> eq(:scalar<si64> mul_no_wrap(div(mul_no_wrap(D474, W619), 45), 45), mul_no_wrap(D474, W619))))>
    kgen.param.declare param_expr_625: scalar<si64> = <mul_no_wrap(D474, W619)>
    kgen.param.declare param_expr_626: scalar<si64> = <cond(eq(:scalar<si64> div(mul_no_wrap(D474, W619), 45), 0), 0, div(mul_no_wrap(D474, W619), cond(eq(:scalar<si64> div(mul_no_wrap(D474, W619), 45), 0), 1, div(mul_no_wrap(D474, W619), 45))))>
    %119 = mo.split_dim[0] (%118) : (!mo.tensor<[param_expr_625], f32>) -> !mo.tensor<[param_expr_624, param_expr_626], f32>
    kgen.param.declare assert602: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "[concat] input shapes must match except at concat axis" : !kgen.string}, to_builtin(:scalar<bool> eq(:scalar<si64> div(mul_no_wrap(D474, W619), 45), D474)))>
    kgen.param.declare param_expr_624: scalar<si64> = <div(mul_no_wrap(D474, W619), 45)>
    %120 = mo.rebind(%119) : !mo.tensor<[param_expr_624, param_expr_626], f32> -> !mo.tensor<[param_expr_624, 45], f32>
    %121 = mo.concat[1] (%99, %120) : (!mo.tensor<[D474, 32], f32>, !mo.tensor<[param_expr_624, 45], f32>) -> !mo.tensor<[D474, 77], f32>
    %122 = mo.matmul(%121, %arg11) : (!mo.tensor<[D474, 77], f32>, !mo.tensor<[77, 256], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[D474, 256], f32>
    %123 = mo.relu(%122) : !mo.tensor<[D474, 256], f32>
    %124 = mo.matmul(%123, %arg10) : (!mo.tensor<[D474, 256], f32>, !mo.tensor<[256, 64], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[D474, 64], f32>
    %125 = mo.relu(%124) : !mo.tensor<[D474, 64], f32>
    %126 = mo.layout.transform(%25) {kgenParams = {a_shape = dense<[-9223372036854775808, 64]> : vector<2xi64>, a_type = f32, b_shape = dense<[64, 1]> : vector<2xi64>, b_type = f32, c_shape = dense<[-9223372036854775808, 1]> : vector<2xi64>, c_type = f32}} : !mo.tensor<[64, 1], f32, {layout = #mo.layout<KN>}> -> !mo.tensor<[64, 24], f32, {layout = #mo.layout<KNkni>}>
    %127 = mo.matmul(%125, %126) : (!mo.tensor<[D474, 64], f32>, !mo.tensor<[64, 24], f32, {layout = #mo.layout<KNkni>}>) -> !mo.tensor<[D474, 1], f32>
    %128 = mo.sin(%127) : !mo.tensor<[D474, 1], f32>
    mo.output %128 : !mo.tensor<[D474, 1], f32>
  }
}
