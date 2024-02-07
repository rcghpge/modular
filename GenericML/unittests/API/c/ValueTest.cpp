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
#include "max/c/tensor.h"
#include "max/cpp/aliases.h"
#include "gtest/gtest.h"

using namespace M;

// Test fixture
//===----------------------------------------------------------------------===//

namespace {

struct AIEngineValueTest : ::testing::Test {
  StatusPtr status = {M_newStatus(), &M_freeStatus};

  RuntimeConfigPtr config = {M_newRuntimeConfig(), &M_freeRuntimeConfig};

  RuntimeContextPtr context = {M_newRuntimeContext(config.get(), status.get()),
                               &M_freeRuntimeContext};
};
} // namespace

TEST_F(AIEngineValueTest, DictTest) {
  ValuePtr dictValue = {M_createDictAsyncValue(context.get()), M_freeValue};

  EXPECT_EQ(M_getValueType(dictValue.get()), M_ValueType::M_DICT_VALUE);

  M_AsyncDict *dict = M_getDictFromValue(dictValue.get());

  ValuePtr noneValue = {M_createNoneAsyncValue(context.get()), M_freeValue};

  std::vector<float> inputTensor(10);
  int64_t shape[] = {2, 5};
  TensorSpecPtr tensorSpecPtr = {
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "t0"),
      &M_freeTensorSpec};
  ValuePtr tensorValue = {M_createBorrowedTensor(inputTensor.data(),
                                                 tensorSpecPtr.get(),
                                                 context.get()),
                          M_freeValue};

  M_insertIntoDict(dict, noneValue.get(), tensorValue.get());

  std::string stringKey = "a";
  ValuePtr stringValue = {
      M_createStringAsyncValue(stringKey.c_str(), context.get()), M_freeValue};

  ValuePtr doubleValue = {M_createDoubleAsyncValue(2.0f, context.get()),
                          M_freeValue};

  M_insertIntoDict(dict, stringValue.get(), doubleValue.get());

  ASSERT_EQ(M_getDictSize(dict), 2u);

  ValuePtr retValue0 = {M_getDictValue(dict, 0), M_freeValue};
  TensorPtr retTensor = {M_getTensorFromValue(retValue0.get()), M_freeTensor};

  ValuePtr retKey1 = {M_getDictKey(dict, 1), M_freeValue};
  ValuePtr retValue1 = {M_getDictValue(dict, 1), M_freeValue};

  EXPECT_EQ(M_getTensorData(retTensor.get()), inputTensor.data());
  EXPECT_EQ(std::string(M_getStringFromValue(retKey1.get())), "a");
  EXPECT_EQ(M_getDoubleFromValue(retValue1.get()), 2.0f);

  M_freeDict(dict);
}

TEST_F(AIEngineValueTest, ListTest) {
  ValuePtr listValue = {M_createListAsyncValue(context.get()), M_freeValue};
  ValuePtr tupleValue = {M_createTupleAsyncValue(context.get()), M_freeValue};
  ValuePtr dictValue = {M_createDictAsyncValue(context.get()), M_freeValue};

  EXPECT_EQ(M_getValueType(listValue.get()), M_ValueType::M_LIST_VALUE);

  M_AsyncList *list = M_getListFromValue(listValue.get());

  M_appendToList(list, tupleValue.get());
  M_appendToList(list, dictValue.get());

  ASSERT_EQ(M_getListSize(list), 2u);

  ValuePtr retValue0 = {M_getListValue(list, 0), M_freeValue};
  ValuePtr retValue1 = {M_getListValue(list, 1), M_freeValue};

  EXPECT_EQ(M_getValueType(retValue0.get()), M_ValueType::M_TUPLE_VALUE);
  EXPECT_EQ(M_getValueType(retValue1.get()), M_ValueType::M_DICT_VALUE);

  M_freeList(list);
}

TEST_F(AIEngineValueTest, TupleTest) {
  ValuePtr tupleValue = {M_createTupleAsyncValue(context.get()), M_freeValue};

  EXPECT_EQ(M_getValueType(tupleValue.get()), M_ValueType::M_TUPLE_VALUE);

  M_AsyncTuple *tuple = M_getTupleFromValue(tupleValue.get());

  ValuePtr longValue = {M_createLongAsyncValue(2u, context.get()), M_freeValue};
  ValuePtr boolValue = {M_createBoolAsyncValue(false, context.get()),
                        M_freeValue};

  M_borrowIntoTuple(tuple, longValue.get());
  M_borrowIntoTuple(tuple, boolValue.get());

  ASSERT_EQ(M_getTupleSize(tuple), 2u);

  ValuePtr retValue0 = {M_getTupleValue(tuple, 0), M_freeValue};
  ValuePtr retValue1 = {M_getTupleValue(tuple, 1), M_freeValue};

  EXPECT_EQ(M_getValueType(retValue0.get()), M_ValueType::M_LONG_VALUE);
  EXPECT_EQ(M_getValueType(retValue1.get()), M_ValueType::M_BOOL_VALUE);

  EXPECT_EQ(M_getLongFromValue(retValue0.get()), 2u);
  EXPECT_EQ(M_getBoolFromValue(retValue1.get()), false);

  M_freeTuple(tuple);
}

TEST_F(AIEngineValueTest, ValueMapTest) {
  TensorMapPtr tensorMap = {M_newAsyncTensorMap(context.get()),
                            &M_freeAsyncTensorMap};

  std::string name = "value";
  ValuePtr longValue = {M_createLongAsyncValue(8u, context.get()), M_freeValue};

  M_borrowValueInto(tensorMap.get(), name.c_str(), longValue.get(),
                    status.get());

  EXPECT_FALSE(M_isError(status.get()));

  ValuePtr value1 = {
      M_getValueByNameFrom(tensorMap.get(), name.c_str(), status.get()),
      M_freeValue};

  EXPECT_EQ(M_getLongFromValue(value1.get()), 8u);

  // Check that iterating works as well
  M_TensorMapIterator *iter =
      M_getTensorMapIterator(tensorMap.get(), status.get());
  ASSERT_FALSE(M_isEndOfTensorMap(tensorMap.get(), iter));
  ValuePtr value2 = {M_getValueFromMapIterator(iter), M_freeValue};

  EXPECT_EQ(M_getLongFromValue(value2.get()), 8u);

  M_advanceTensorMapIterator(iter);
  EXPECT_TRUE(M_isEndOfTensorMap(tensorMap.get(), iter));
  M_freeTensorMapIterator(iter);

  // Check that a value can't be inserted with the same name.
  M_borrowValueInto(tensorMap.get(), name.c_str(), longValue.get(),
                    status.get());
  EXPECT_TRUE(M_isError(status.get()));
  EXPECT_EQ(M_getError(status.get()),
            "Value with name " + name + " already exists in the map!");
}
