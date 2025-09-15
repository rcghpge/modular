//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Value manipulation tests
//
//===----------------------------------------------------------------------===//

#include "max/c/value.h"
#include "max/c/common.h"
#include "max/cpp/aliases.h"
#include "gtest/gtest.h"

using namespace M;

// Test fixture
//===----------------------------------------------------------------------===//

namespace {

struct AIEngineValueTest : ::testing::Test {
  StatusPtr status{M_newStatus()};

  RuntimeConfigPtr config{M_newRuntimeConfig()};

  RuntimeContextPtr context{M_newRuntimeContext(config.get(), status.get())};
};
} // namespace

TEST_F(AIEngineValueTest, ListTest) {
  ValuePtr listValue{M_createListAsyncValue(context.get())};
  ValuePtr tupleValue{M_createTupleAsyncValue(context.get())};

  EXPECT_EQ(M_getValueType(listValue.get()), M_ValueType::M_LIST_VALUE);

  M_AsyncList *list = M_getListFromValue(listValue.get());

  M_appendToList(list, tupleValue.get());

  ASSERT_EQ(M_getListSize(list), 1u);

  ValuePtr retValue0{M_getListValue(list, 0)};

  EXPECT_EQ(M_getValueType(retValue0.get()), M_ValueType::M_TUPLE_VALUE);

  M_freeList(list);
}

TEST_F(AIEngineValueTest, TupleTest) {
  ValuePtr tupleValue{M_createTupleAsyncValue(context.get())};

  EXPECT_EQ(M_getValueType(tupleValue.get()), M_ValueType::M_TUPLE_VALUE);

  M_AsyncTuple *tuple = M_getTupleFromValue(tupleValue.get());

  ASSERT_EQ(M_getTupleSize(tuple), 0u);

  M_freeTuple(tuple);
}
