//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Fixture.h"

static std::string code = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void vec_add_msl(
    device float* output [[buffer(0)]],
    device const float* input0 [[buffer(1)]],
    device const float* input1 [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < length) {
        output[tid] = input0[tid] + input1[tid];
    }
}
)MSL";

TEST_F(DriverPluginContextTest, FunctionLoad) {
  for (M_driver_context_t context : contexts) {
    M_driver_function_t function;
    ASSERT_SUCCESS(function_load(driver, context, code.data(), code.size(),
                                 "vec_add_msl", -1, "", -1, &function));
    ASSERT_SUCCESS(function_unload(driver, context, function));
  }
}
