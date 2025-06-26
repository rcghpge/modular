//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/EngineAPI/python/Types.h"
#include "gtest/gtest.h"

using namespace M::core;

TEST(DTypeNanobindConversionTest, RoundTrips) {
  EXPECT_EQ(mapNanobindType(mapDType(M_BOOL)), M_BOOL);
  EXPECT_EQ(mapNanobindType(mapDType(M_INT16)), M_INT16);
  EXPECT_EQ(mapNanobindType(mapDType(M_INT32)), M_INT32);
  EXPECT_EQ(mapNanobindType(mapDType(M_INT64)), M_INT64);
  EXPECT_EQ(mapNanobindType(mapDType(M_UINT8)), M_UINT8);
  EXPECT_EQ(mapNanobindType(mapDType(M_UINT16)), M_UINT16);
  EXPECT_EQ(mapNanobindType(mapDType(M_UINT64)), M_UINT64);
  EXPECT_EQ(mapNanobindType(mapDType(M_BFLOAT16)), M_BFLOAT16);
  EXPECT_EQ(mapNanobindType(mapDType(M_FLOAT16)), M_FLOAT16);
  EXPECT_EQ(mapNanobindType(mapDType(M_FLOAT32)), M_FLOAT32);
  EXPECT_EQ(mapNanobindType(mapDType(M_FLOAT64)), M_FLOAT64);

  // NOTE: nanobind's dlpack does not support fp8, hence why they re omitted
  // here.

  EXPECT_THROW(mapDType(M_UNKNOWN), std::runtime_error);
}
