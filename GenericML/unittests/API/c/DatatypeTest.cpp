//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// C API datatype tests
//
//===----------------------------------------------------------------------===//

#include "max/c/common.h"
#include "gtest/gtest.h"

TEST(ModularEngineTest, DtypeSize) {
  EXPECT_EQ(M_sizeOf(M_UNKNOWN), 0u);
  EXPECT_EQ(M_sizeOf(M_INT8), sizeof(int8_t));
  EXPECT_EQ(M_sizeOf(M_INT16), sizeof(int16_t));
  EXPECT_EQ(M_sizeOf(M_INT32), sizeof(int32_t));
  EXPECT_EQ(M_sizeOf(M_INT64), sizeof(int64_t));
  EXPECT_EQ(M_sizeOf(M_UINT8), sizeof(uint8_t));
  EXPECT_EQ(M_sizeOf(M_UINT16), sizeof(uint16_t));
  EXPECT_EQ(M_sizeOf(M_UINT32), sizeof(uint32_t));
  EXPECT_EQ(M_sizeOf(M_UINT64), sizeof(uint64_t));
  EXPECT_EQ(M_sizeOf(M_FLOAT16), sizeof(uint16_t));
  EXPECT_EQ(M_sizeOf(M_FLOAT32), sizeof(float));
  EXPECT_EQ(M_sizeOf(M_FLOAT64), sizeof(double));
  EXPECT_EQ(M_sizeOf(M_BFLOAT16), sizeof(uint16_t));
  EXPECT_EQ(M_sizeOf(M_FLOAT8_E4M3FN), sizeof(uint8_t));
  EXPECT_EQ(M_sizeOf(M_FLOAT8_E4M3FNUZ), sizeof(uint8_t));
  EXPECT_EQ(M_sizeOf(M_FLOAT8_E5M2), sizeof(uint8_t));
  EXPECT_EQ(M_sizeOf(M_FLOAT8_E5M2FNUZ), sizeof(uint8_t));
  EXPECT_EQ(M_sizeOf(M_BOOL), sizeof(bool));
}
