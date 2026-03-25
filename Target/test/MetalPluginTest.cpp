//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "../lib/Metal/Common.h"
#include "Fixture.h"

// Extend test-cases per-plugin beyond the common ones in PluginTest.cpp

static std::string code = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct FloatBufPtr      { device float*       ptr; };
struct ConstFloatBufPtr { device const float* ptr; };

kernel void vec_add_msl(
    device FloatBufPtr&       out_ref [[buffer(0)]],
    device ConstFloatBufPtr&  in0_ref [[buffer(1)]],
    device ConstFloatBufPtr&  in1_ref [[buffer(2)]],
    constant uint&            length  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    device float*       output = out_ref.ptr;
    device const float* input0 = in0_ref.ptr;
    device const float* input1 = in1_ref.ptr;
    if (tid < length)
        output[tid] = input0[tid] + input1[tid];
}
)MSL";

TEST_F(DriverPluginTest, CustomStatusMessage) {
  char errorMessage[1024];
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(M_DRIVER_STATUS_UNIMPLEMENTED, errorMessage,
                                 sizeof(errorMessage)));
  ASSERT_STREQ(errorMessage, "Unimplemented");
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(M_DRIVER_STATUS_UNINITIALIZED, errorMessage,
                                 sizeof(errorMessage)));
  ASSERT_STREQ(errorMessage, "Uninitialized");
}

TEST_F(DriverPluginContextTest, FunctionLoad) {
  for (M_driver_context_t context : contexts) {
    M_driver_function_t function;
    ASSERT_SUCCESS(function_load(driver, context, code.data(), code.size(),
                                 "vec_add_msl", -1, "", -1, &function));
    ASSERT_SUCCESS(function_unload(driver, context, function));
  }
}
