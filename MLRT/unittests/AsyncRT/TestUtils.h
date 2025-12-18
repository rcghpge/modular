//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Test helpers for gunit tests that make it more easy to handle
// `ErrorOr<T>` and `ErrorOrSuccess` with less boilerplate.

#include "gmock/gmock.h"

#define ASSIGN_NO_ERR(type, var, expr)                                         \
  type var;                                                                    \
  do {                                                                         \
    auto _err_or_val_ = (expr);                                                \
    ASSERT_FALSE(_err_or_val_.isError()) << _err_or_val_.getError() << "\n";   \
    var = _err_or_val_.takeValue();                                            \
  } while (0)

#define EXPECT_NO_ERR(expr)                                                    \
  do {                                                                         \
    auto err = (expr);                                                         \
    EXPECT_FALSE(err.isError()) << err.getError() << "\n";                     \
  } while (0)

#define EXPECT_ERR(err)                                                        \
  do {                                                                         \
    EXPECT_TRUE((err).isError());                                              \
  } while (0)

#define EXPECT_ERR_MSG(expr, msg)                                              \
  do {                                                                         \
    auto err = (expr);                                                         \
    EXPECT_TRUE(err.isError());                                                \
    EXPECT_THAT(err.getError(), testing::HasSubstr(msg));                      \
  } while (0)
