// RUN: mo-opt %s --mo-to-mgp  --allow-unregistered-dialect | FileCheck %s

// Just check we can compile it.

// CHECK: mgp.model @dlrm

mo.graph @dlrm(%arg0: !mo.tensor<[?, 8, 100], si32>, %arg1: !mo.tensor<[?, 256], f32>, %arg2: !mo.tensor<[200000, 32], f32>, %arg3: !mo.tensor<[200000, 32], f32>, %arg4: !mo.tensor<[200000, 32], f32>, %arg5: !mo.tensor<[200000, 32], f32>, %arg6: !mo.tensor<[200000, 32], f32>, %arg7: !mo.tensor<[200000, 32], f32>, %arg8: !mo.tensor<[200000, 32], f32>, %arg9: !mo.tensor<[200000, 32], f32>, %arg10: !mo.tensor<[256, 64], f32>, %arg11: !mo.tensor<[77, 256], f32>, %arg12: !mo.tensor<[128, 32], f32>, %arg13: !mo.tensor<[256, 128], f32>) -> !mo.tensor<[?, 1], f32> {
  %0 = mo.constant {value = #M.dense_array<> : tensor<0xsi32>} : !mo.tensor<[0], si32>
  %1 = mo.constant {value = #M.dense_array<2> : tensor<si32>} : !mo.tensor<[], si32>
  %2 = mo.constant {value = #M.dense_array<1> : tensor<si32>} : !mo.tensor<[], si32>
  %3 = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %4 = mo.constant {value = #M.dense_array<32> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %5 = mo.constant {value = #M.dense_array<9> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %6 = mo.constant {value = #M.dense_array<false> : tensor<i1>} : !mo.tensor<[], bool>
  %7 = mo.constant {value = #M.dense_array<0, 2, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %8 = mo.constant {value = #M.dense_array<3> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %9 = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %10 = mo.constant {value = #M.dense_array<1, 1, 1, 1, 1, 1, 1, 1> : tensor<8xsi64>} : !mo.tensor<[8], si64>
  %11 = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %12 = mo.constant {value = #M.dense_array<-0.121941715, -0.112631559, 0.199184537, 0.260294795, -0.190447032, 0.214445293, 0.0505436063, 0.254368067, 0.18954441, -0.156480938, -0.0834032595, -1.882630e-01, -0.125319079, -0.225907743, 0.122629285, -0.0715062469, -0.184827894, -0.228427023, 0.229661107, 0.0153187215, -0.118268147, 0.159035563, 0.0799599587, 0.0667982101, 0.0901305377, -0.0790597349, 0.04434672, -0.278808236, 0.300504982, 0.0512512624, 0.156755388, -0.141259581, -0.0541819781, -0.207770735, -0.232766122, -0.180529654, 0.185262382, -0.155863777, -5.34206629E-4, 0.194931746, -3.772080e-02, 0.188996345, 0.128953159, 0.273819387, -0.0088250339, 0.191499621, 0.204237163, -0.0390079916, -0.292554468, 0.260827839, -0.114037767, 0.249398828, -0.0697858631, -0.271265894, -0.0694437474, 0.141885012, 8.175060e-02, 0.289748192, 0.165302575, 0.228590071, -0.0586845726, -0.22602132, 0.19075048, 0.0257503986> : tensor<64x1xf32>} : !mo.tensor<[64, 1], f32>
  %13 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %14 = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %15 = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %16 = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %17 = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
  %18 = mo.constant {value = #M.dense_array<-1> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %19 = mo.constant {value = #M.dense_array<-1, 45> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %20 = mo.constant {value = #M.dense_array<-1> : tensor<si32>} : !mo.tensor<[], si32>
  %21 = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %22 = mo.constant {value = #M.dense_array<0.000000e+00> : !M.array<1xf32>} : !mo.tensor<[], f32>
  %23 = mo.static.broadcast_to(%22) : !mo.tensor<[], f32> -> !mo.tensor<[128], f32>
  %24 = mo.static.broadcast_to(%22) : !mo.tensor<[], f32> -> !mo.tensor<[32], f32>
  %25 = mo.static.broadcast_to(%22) : !mo.tensor<[], f32> -> !mo.tensor<[256], f32>
  %26 = mo.static.broadcast_to(%22) : !mo.tensor<[], f32> -> !mo.tensor<[64], f32>
  %27:8 = mo.split[%11 : !mo.tensor<[], si64>] (%arg0, %10) : (!mo.tensor<[?, 8, 100], si32>, !mo.tensor<[8], si64>) -> (!mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?, ?, ?], si32>)
  %28 = mo.shape_of(%27#0) : (!mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[3], si64>
  %29 = mo.shape_of(%28) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %30 = mo.static.reshape(%29) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %31 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %32 = mo.range(%1, %30, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %33 = mo.gather(%28, %31, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %34 = mo.gather(%28, %32, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %35 = mo.concat[%3 : !mo.tensor<[], si32>] (%33, %34) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %36 = mo.reshape(%27#0, %35) : (!mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 100], si32>
  %37 = mo.shape_of(%27#1) : (!mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[3], si64>
  %38 = mo.shape_of(%37) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %39 = mo.static.reshape(%38) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %40 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %41 = mo.range(%1, %39, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %42 = mo.gather(%37, %40, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %43 = mo.gather(%37, %41, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %44 = mo.concat[%3 : !mo.tensor<[], si32>] (%42, %43) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %45 = mo.reshape(%27#1, %44) : (!mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 100], si32>
  %46 = mo.shape_of(%27#2) : (!mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[3], si64>
  %47 = mo.shape_of(%46) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %48 = mo.static.reshape(%47) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %49 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %50 = mo.range(%1, %48, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %51 = mo.gather(%46, %49, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %52 = mo.gather(%46, %50, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %53 = mo.concat[%3 : !mo.tensor<[], si32>] (%51, %52) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %54 = mo.reshape(%27#2, %53) : (!mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 100], si32>
  %55 = mo.shape_of(%27#3) : (!mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[3], si64>
  %56 = mo.shape_of(%55) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %57 = mo.static.reshape(%56) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %58 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %59 = mo.range(%1, %57, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %60 = mo.gather(%55, %58, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %61 = mo.gather(%55, %59, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %62 = mo.concat[%3 : !mo.tensor<[], si32>] (%60, %61) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %63 = mo.reshape(%27#3, %62) : (!mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 100], si32>
  %64 = mo.shape_of(%27#4) : (!mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[3], si64>
  %65 = mo.shape_of(%64) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %66 = mo.static.reshape(%65) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %67 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %68 = mo.range(%1, %66, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %69 = mo.gather(%64, %67, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %70 = mo.gather(%64, %68, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %71 = mo.concat[%3 : !mo.tensor<[], si32>] (%69, %70) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %72 = mo.reshape(%27#4, %71) : (!mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 100], si32>
  %73 = mo.shape_of(%27#5) : (!mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[3], si64>
  %74 = mo.shape_of(%73) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %75 = mo.static.reshape(%74) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %76 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %77 = mo.range(%1, %75, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %78 = mo.gather(%73, %76, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %79 = mo.gather(%73, %77, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %80 = mo.concat[%3 : !mo.tensor<[], si32>] (%78, %79) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %81 = mo.reshape(%27#5, %80) : (!mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 100], si32>
  %82 = mo.shape_of(%27#6) : (!mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[3], si64>
  %83 = mo.shape_of(%82) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %84 = mo.static.reshape(%83) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %85 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %86 = mo.range(%1, %84, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %87 = mo.gather(%82, %85, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %88 = mo.gather(%82, %86, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %89 = mo.concat[%3 : !mo.tensor<[], si32>] (%87, %88) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %90 = mo.reshape(%27#6, %89) : (!mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 100], si32>
  %91 = mo.shape_of(%27#7) : (!mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[3], si64>
  %92 = mo.shape_of(%91) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %93 = mo.static.reshape(%92) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %94 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %95 = mo.range(%1, %93, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %96 = mo.gather(%91, %94, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %97 = mo.gather(%91, %95, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %98 = mo.concat[%3 : !mo.tensor<[], si32>] (%96, %97) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %99 = mo.reshape(%27#7, %98) : (!mo.tensor<[?, ?, ?], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 100], si32>
  %100 = mo.gather(%arg9, %36, %14) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[?, 100], si32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, 100, 32], f32>
  %101 = mo.reduce.add(%100, %2) : (!mo.tensor<[?, 100, 32], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?], f32>
  %102 = mo.shape_of(%101) : (!mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[3], si64>
  %103 = mo.shape_of(%102) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %104 = mo.static.reshape(%103) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %105 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %106 = mo.range(%1, %104, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %107 = mo.gather(%102, %105, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %108 = mo.gather(%102, %106, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %109 = mo.concat[%3 : !mo.tensor<[], si32>] (%107, %108) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %110 = mo.reshape(%101, %109) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 32], f32>
  %111 = mo.gather(%arg8, %45, %14) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[?, 100], si32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, 100, 32], f32>
  %112 = mo.reduce.add(%111, %2) : (!mo.tensor<[?, 100, 32], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?], f32>
  %113 = mo.shape_of(%112) : (!mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[3], si64>
  %114 = mo.shape_of(%113) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %115 = mo.static.reshape(%114) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %116 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %117 = mo.range(%1, %115, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %118 = mo.gather(%113, %116, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %119 = mo.gather(%113, %117, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %120 = mo.concat[%3 : !mo.tensor<[], si32>] (%118, %119) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %121 = mo.reshape(%112, %120) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 32], f32>
  %122 = mo.gather(%arg7, %54, %14) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[?, 100], si32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, 100, 32], f32>
  %123 = mo.reduce.add(%122, %2) : (!mo.tensor<[?, 100, 32], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?], f32>
  %124 = mo.shape_of(%123) : (!mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[3], si64>
  %125 = mo.shape_of(%124) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %126 = mo.static.reshape(%125) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %127 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %128 = mo.range(%1, %126, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %129 = mo.gather(%124, %127, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %130 = mo.gather(%124, %128, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %131 = mo.concat[%3 : !mo.tensor<[], si32>] (%129, %130) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %132 = mo.reshape(%123, %131) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 32], f32>
  %133 = mo.gather(%arg6, %63, %14) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[?, 100], si32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, 100, 32], f32>
  %134 = mo.reduce.add(%133, %2) : (!mo.tensor<[?, 100, 32], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?], f32>
  %135 = mo.shape_of(%134) : (!mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[3], si64>
  %136 = mo.shape_of(%135) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %137 = mo.static.reshape(%136) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %138 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %139 = mo.range(%1, %137, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %140 = mo.gather(%135, %138, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %141 = mo.gather(%135, %139, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %142 = mo.concat[%3 : !mo.tensor<[], si32>] (%140, %141) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %143 = mo.reshape(%134, %142) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 32], f32>
  %144 = mo.gather(%arg5, %72, %14) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[?, 100], si32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, 100, 32], f32>
  %145 = mo.reduce.add(%144, %2) : (!mo.tensor<[?, 100, 32], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?], f32>
  %146 = mo.shape_of(%145) : (!mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[3], si64>
  %147 = mo.shape_of(%146) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %148 = mo.static.reshape(%147) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %149 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %150 = mo.range(%1, %148, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %151 = mo.gather(%146, %149, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %152 = mo.gather(%146, %150, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %153 = mo.concat[%3 : !mo.tensor<[], si32>] (%151, %152) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %154 = mo.reshape(%145, %153) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 32], f32>
  %155 = mo.gather(%arg4, %81, %14) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[?, 100], si32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, 100, 32], f32>
  %156 = mo.reduce.add(%155, %2) : (!mo.tensor<[?, 100, 32], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?], f32>
  %157 = mo.shape_of(%156) : (!mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[3], si64>
  %158 = mo.shape_of(%157) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %159 = mo.static.reshape(%158) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %160 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %161 = mo.range(%1, %159, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %162 = mo.gather(%157, %160, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %163 = mo.gather(%157, %161, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %164 = mo.concat[%3 : !mo.tensor<[], si32>] (%162, %163) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %165 = mo.reshape(%156, %164) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 32], f32>
  %166 = mo.gather(%arg3, %90, %14) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[?, 100], si32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, 100, 32], f32>
  %167 = mo.reduce.add(%166, %2) : (!mo.tensor<[?, 100, 32], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?], f32>
  %168 = mo.shape_of(%167) : (!mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[3], si64>
  %169 = mo.shape_of(%168) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %170 = mo.static.reshape(%169) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %171 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %172 = mo.range(%1, %170, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %173 = mo.gather(%168, %171, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %174 = mo.gather(%168, %172, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %175 = mo.concat[%3 : !mo.tensor<[], si32>] (%173, %174) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %176 = mo.reshape(%167, %175) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 32], f32>
  %177 = mo.gather(%arg2, %99, %14) : (!mo.tensor<[200000, 32], f32>, !mo.tensor<[?, 100], si32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, 100, 32], f32>
  %178 = mo.reduce.add(%177, %2) : (!mo.tensor<[?, 100, 32], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[?, ?, ?], f32>
  %179 = mo.shape_of(%178) : (!mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[3], si64>
  %180 = mo.shape_of(%179) : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si32>
  %181 = mo.static.reshape(%180) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %182 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %183 = mo.range(%1, %181, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %184 = mo.gather(%179, %182, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %185 = mo.gather(%179, %183, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %186 = mo.concat[%3 : !mo.tensor<[], si32>] (%184, %185) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %187 = mo.reshape(%178, %186) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, 32], f32>
  %188 = mo.matmul(%arg1, %arg13) : (!mo.tensor<[?, 256], f32>, !mo.tensor<[256, 128], f32>) -> !mo.tensor<[?, 128], f32>
  %189 = mo.shape_of(%188) : (!mo.tensor<[?, 128], f32>) -> !mo.tensor<[2], si64>
  %190 = mo.shape_of(%23) : (!mo.tensor<[128], f32>) -> !mo.tensor<[1], si64>
  %191 = mo.broadcast_shape(%189, %190) : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %192 = mo.broadcast_to(%188, %191) : (!mo.tensor<[?, 128], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, 128], f32>
  %193 = mo.broadcast_to(%23, %191) : (!mo.tensor<[128], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, 128], f32>
  %194 = mo.add(%192, %193) : !mo.tensor<[?, 128], f32>
  %195 = mo.relu(%194) : !mo.tensor<[?, 128], f32>
  %196 = mo.matmul(%195, %arg12) : (!mo.tensor<[?, 128], f32>, !mo.tensor<[128, 32], f32>) -> !mo.tensor<[?, 32], f32>
  %197 = mo.shape_of(%196) : (!mo.tensor<[?, 32], f32>) -> !mo.tensor<[2], si64>
  %198 = mo.shape_of(%24) : (!mo.tensor<[32], f32>) -> !mo.tensor<[1], si64>
  %199 = mo.broadcast_shape(%197, %198) : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %200 = mo.broadcast_to(%196, %199) : (!mo.tensor<[?, 32], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, 32], f32>
  %201 = mo.broadcast_to(%24, %199) : (!mo.tensor<[32], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, 32], f32>
  %202 = mo.add(%200, %201) : !mo.tensor<[?, 32], f32>
  %203 = mo.relu(%202) : !mo.tensor<[?, 32], f32>
  %204 = mo.shape_of(%203) : (!mo.tensor<[?, 32], f32>) -> !mo.tensor<[2], si32>
  %205 = mo.slice(%204, %16, %15, %15) : (!mo.tensor<[2], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[?], si32>
  %206 = mo.shape_of(%205) : (!mo.tensor<[?], si32>) -> !mo.tensor<[1], si64>
  %207 = mo.shape_of(%206) : (!mo.tensor<[1], si64>) -> !mo.tensor<[1], si32>
  %208 = mo.static.reshape(%207) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %209 = mo.rebind(%0) : !mo.tensor<[0], si32> -> !mo.tensor<[?], si32>
  %210 = mo.range(%2, %208, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %211 = mo.gather(%206, %209, %3) : (!mo.tensor<[1], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %212 = mo.gather(%206, %210, %3) : (!mo.tensor<[1], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %213 = mo.concat[%3 : !mo.tensor<[], si32>] (%211, %212) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %214 = mo.reshape(%205, %213) : (!mo.tensor<[?], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<[], si32>
  %215 = mo.static.reshape(%214) : !mo.tensor<[], si32> -> !mo.tensor<[1], si32>
  %216 = mo.rebind(%215) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %217 = mo.rebind(%5) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %218 = mo.rebind(%4) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %219 = mo.concat[%14 : !mo.tensor<[], si64>] (%216, %217, %218) : (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[3], si32>
  %220 = mo.concat[%2 : !mo.tensor<[], si32>] (%203, %110, %121, %132, %143, %154, %165, %176, %187) : (!mo.tensor<[?, 32], f32>, !mo.tensor<[?, 32], f32>, !mo.tensor<[?, 32], f32>, !mo.tensor<[?, 32], f32>, !mo.tensor<[?, 32], f32>, !mo.tensor<[?, 32], f32>, !mo.tensor<[?, 32], f32>, !mo.tensor<[?, 32], f32>, !mo.tensor<[?, 32], f32>) -> !mo.tensor<[?, 288], f32>
  %221 = mo.reshape(%220, %219) : (!mo.tensor<[?, 288], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[?, 9, 32], f32>
  %222 = mo.shape_of(%221) : (!mo.tensor<[?, 9, 32], f32>) -> !mo.tensor<[3], si64>
  %223 = mo.slice(%222, %9, %21, %21) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
  %224 = mo.slice(%222, %21, %8, %21) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %225 = mo.broadcast_shape(%223, %223) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
  %226 = mo.concat[%14 : !mo.tensor<[], si64>] (%225, %224) : (!mo.tensor<[1], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
  %227 = mo.broadcast_to(%221, %226) : (!mo.tensor<[?, 9, 32], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[?, ?, ?], f32>
  %228 = mo.transpose(%227, %7) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[?, ?, ?], f32>
  %229 = mo.batch_matmul(%227, %228) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, 9, 9], f32>
  %230 = mo.shape_of(%229) : (!mo.tensor<[?, 9, 9], f32>) -> !mo.tensor<[3], si32>
  %231 = mo.reduce.mul(%230, %3) : (!mo.tensor<[3], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %232 = mo.shape_of(%231) : (!mo.tensor<[?], si32>) -> !mo.tensor<[1], si64>
  %233 = mo.shape_of(%232) : (!mo.tensor<[1], si64>) -> !mo.tensor<[1], si32>
  %234 = mo.static.reshape(%233) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %235 = mo.rebind(%0) : !mo.tensor<[0], si32> -> !mo.tensor<[?], si32>
  %236 = mo.range(%2, %234, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %237 = mo.gather(%232, %235, %3) : (!mo.tensor<[1], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %238 = mo.gather(%232, %236, %3) : (!mo.tensor<[1], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %239 = mo.concat[%3 : !mo.tensor<[], si32>] (%237, %238) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %240 = mo.reshape(%231, %239) : (!mo.tensor<[?], si32>, !mo.tensor<[?], si64>) -> !mo.tensor<[], si32>
  %241 = mo.static.reshape(%240) : !mo.tensor<[], si32> -> !mo.tensor<[1], si32>
  %242 = mo.reshape(%229, %241) : (!mo.tensor<[?, 9, 9], f32>, !mo.tensor<[1], si32>) -> !mo.tensor<[?], f32>
  %243 = mo.broadcast_to(%17, %230) : (!mo.tensor<[], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[?, ?, ?], f32>
  %244 = mo.linalg.band_part(%243, %3, %20, %6) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[?, ?, ?], f32>
  %245 = mo.reshape(%244, %18) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[1], si32>) -> !mo.tensor<[?], f32>
  %246 = mo.arg_nonzero(%245) : (!mo.tensor<[?], f32>) -> !mo.tensor<[?, 1], si64>
  %247 = mo.shape_of(%246) : (!mo.tensor<[?, 1], si64>) -> !mo.tensor<[2], si64>
  %248 = mo.shape_of(%247) : (!mo.tensor<[2], si64>) -> !mo.tensor<[1], si32>
  %249 = mo.static.reshape(%248) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %250 = mo.rebind(%16) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %251 = mo.range(%1, %249, %2) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %252 = mo.gather(%247, %250, %3) : (!mo.tensor<[2], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %253 = mo.gather(%247, %251, %3) : (!mo.tensor<[2], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %254 = mo.concat[%3 : !mo.tensor<[], si32>] (%252, %253) : (!mo.tensor<[?], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %255 = mo.reshape(%246, %254) : (!mo.tensor<[?, 1], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  %256 = mo.gather(%242, %255, %3) : (!mo.tensor<[?], f32>, !mo.tensor<[?], si64>, !mo.tensor<[], si32>) -> !mo.tensor<[?], f32>
  %257 = mo.reshape(%256, %19) : (!mo.tensor<[?], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[?, 45], f32>
  %258 = mo.concat[%2 : !mo.tensor<[], si32>] (%203, %257) : (!mo.tensor<[?, 32], f32>, !mo.tensor<[?, 45], f32>) -> !mo.tensor<[?, 77], f32>
  %259 = mo.matmul(%258, %arg11) : (!mo.tensor<[?, 77], f32>, !mo.tensor<[77, 256], f32>) -> !mo.tensor<[?, 256], f32>
  %260 = mo.shape_of(%259) : (!mo.tensor<[?, 256], f32>) -> !mo.tensor<[2], si64>
  %261 = mo.shape_of(%25) : (!mo.tensor<[256], f32>) -> !mo.tensor<[1], si64>
  %262 = mo.broadcast_shape(%260, %261) : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %263 = mo.broadcast_to(%259, %262) : (!mo.tensor<[?, 256], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, 256], f32>
  %264 = mo.broadcast_to(%25, %262) : (!mo.tensor<[256], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, 256], f32>
  %265 = mo.add(%263, %264) : !mo.tensor<[?, 256], f32>
  %266 = mo.relu(%265) : !mo.tensor<[?, 256], f32>
  %267 = mo.matmul(%266, %arg10) : (!mo.tensor<[?, 256], f32>, !mo.tensor<[256, 64], f32>) -> !mo.tensor<[?, 64], f32>
  %268 = mo.shape_of(%267) : (!mo.tensor<[?, 64], f32>) -> !mo.tensor<[2], si64>
  %269 = mo.shape_of(%26) : (!mo.tensor<[64], f32>) -> !mo.tensor<[1], si64>
  %270 = mo.broadcast_shape(%268, %269) : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %271 = mo.broadcast_to(%267, %270) : (!mo.tensor<[?, 64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, 64], f32>
  %272 = mo.broadcast_to(%26, %270) : (!mo.tensor<[64], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, 64], f32>
  %273 = mo.add(%271, %272) : !mo.tensor<[?, 64], f32>
  %274 = mo.relu(%273) : !mo.tensor<[?, 64], f32>
  %275 = mo.matmul(%274, %12) : (!mo.tensor<[?, 64], f32>, !mo.tensor<[64, 1], f32>) -> !mo.tensor<[?, 1], f32>
  %276 = mo.shape_of(%275) : (!mo.tensor<[?, 1], f32>) -> !mo.tensor<[2], si64>
  %277 = mo.shape_of(%13) : (!mo.tensor<[1], f32>) -> !mo.tensor<[1], si64>
  %278 = mo.broadcast_shape(%276, %277) : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %279 = mo.broadcast_to(%275, %278) : (!mo.tensor<[?, 1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, 1], f32>
  %280 = mo.broadcast_to(%13, %278) : (!mo.tensor<[1], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, 1], f32>
  %281 = mo.add(%279, %280) : !mo.tensor<[?, 1], f32>
  %282 = mo.sigmoid(%281) : !mo.tensor<[?, 1], f32>
  mo.output %282 : !mo.tensor<[?, 1], f32>
}
