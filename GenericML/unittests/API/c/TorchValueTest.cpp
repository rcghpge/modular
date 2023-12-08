//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Value manipulation tests
//
//===----------------------------------------------------------------------===//

#include "PyTorch/API/modular/c/pytorch/value.h"
#include "modular/c/common.h"
#include "modular/c/tensor.h"
#include "modular/c/value.h"
#include "modular/cpp/aliases.h"
#include "gtest/gtest.h"

using namespace M;

// Test fixture
//===----------------------------------------------------------------------===//

namespace {

struct AIEngineTorchValueTest : ::testing::Test {
  StatusPtr status = {M_newStatus(), &M_freeStatus};

  RuntimeConfigPtr config = {M_newRuntimeConfig(), &M_freeRuntimeConfig};

  RuntimeContextPtr context = {M_newRuntimeContext(config.get(), status.get()),
                               &M_freeRuntimeContext};
};
} // namespace

TEST_F(AIEngineTorchValueTest, DictTest) {
  ValuePtr dictValue = {M_Torch_createDictAsyncValue(context.get()),
                        M_freeValue};

  EXPECT_EQ(M_Torch_getValueType(dictValue.get()),
            M_Torch_ValueType::M_TORCH_DICT_VALUE);

  M_Torch_AsyncDict *dict = M_Torch_getDictFromValue(dictValue.get());

  ValuePtr noneValue = {M_Torch_createNoneAsyncValue(context.get()),
                        M_freeValue};

  std::vector<float> inputTensor(10);
  int64_t shape[] = {2, 5};
  TensorSpecPtr tensorSpecPtr = {
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "t0"),
      &M_freeTensorSpec};
  ValuePtr tensorValue = {M_createBorrowedTensor(inputTensor.data(),
                                                 tensorSpecPtr.get(),
                                                 context.get()),
                          M_freeValue};

  M_Torch_borrowIntoDict(dict, noneValue.get(), tensorValue.get());

  std::string stringKey = "a";
  ValuePtr stringValue = {
      M_createStringAsyncValue(stringKey.c_str(), context.get()), M_freeValue};

  ValuePtr doubleValue = {M_createDoubleAsyncValue(2.0f, context.get()),
                          M_freeValue};

  M_Torch_borrowIntoDict(dict, stringValue.get(), doubleValue.get());

  ASSERT_EQ(M_Torch_getDictSize(dict), 2u);

  ValuePtr retValue0 = {M_Torch_getDictValue(dict, 0), M_freeValue};
  TensorPtr retTensor = {M_getTensorFromValue(retValue0.get()), M_freeTensor};

  ValuePtr retKey1 = {M_Torch_getDictKey(dict, 1), M_freeValue};
  ValuePtr retValue1 = {M_Torch_getDictValue(dict, 1), M_freeValue};

  EXPECT_EQ(M_getTensorData(retTensor.get()), inputTensor.data());
  EXPECT_EQ(std::string(M_getStringFromValue(retKey1.get())), "a");
  EXPECT_EQ(M_getDoubleFromValue(retValue1.get()), 2.0f);

  M_Torch_freeDict(dict);
}

TEST_F(AIEngineTorchValueTest, ListTest) {
  ValuePtr listValue = {M_Torch_createListAsyncValue(context.get()),
                        M_freeValue};
  ValuePtr tupleValue = {M_Torch_createTupleAsyncValue(context.get()),
                         M_freeValue};
  ValuePtr dictValue = {M_Torch_createDictAsyncValue(context.get()),
                        M_freeValue};

  EXPECT_EQ(M_Torch_getValueType(listValue.get()),
            M_Torch_ValueType::M_TORCH_LIST_VALUE);

  M_Torch_AsyncList *list = M_Torch_getListFromValue(listValue.get());

  M_Torch_borrowIntoList(list, tupleValue.get());
  M_Torch_borrowIntoList(list, dictValue.get());

  ASSERT_EQ(M_Torch_getListSize(list), 2u);

  ValuePtr retValue0 = {M_Torch_getListValue(list, 0), M_freeValue};
  ValuePtr retValue1 = {M_Torch_getListValue(list, 1), M_freeValue};

  EXPECT_EQ(M_Torch_getValueType(retValue0.get()),
            M_Torch_ValueType::M_TORCH_TUPLE_VALUE);
  EXPECT_EQ(M_Torch_getValueType(retValue1.get()),
            M_Torch_ValueType::M_TORCH_DICT_VALUE);

  M_Torch_freeList(list);
}

TEST_F(AIEngineTorchValueTest, TupleTest) {
  ValuePtr tupleValue = {M_Torch_createTupleAsyncValue(context.get()),
                         M_freeValue};

  EXPECT_EQ(M_Torch_getValueType(tupleValue.get()),
            M_Torch_ValueType::M_TORCH_TUPLE_VALUE);

  M_Torch_AsyncTuple *tuple = M_Torch_getTupleFromValue(tupleValue.get());

  ValuePtr longValue = {M_createLongAsyncValue(2u, context.get()), M_freeValue};
  ValuePtr boolValue = {M_createBoolAsyncValue(false, context.get()),
                        M_freeValue};

  M_Torch_borrowIntoTuple(tuple, longValue.get());
  M_Torch_borrowIntoTuple(tuple, boolValue.get());

  ASSERT_EQ(M_Torch_getTupleSize(tuple), 2u);

  ValuePtr retValue0 = {M_Torch_getTupleValue(tuple, 0), M_freeValue};
  ValuePtr retValue1 = {M_Torch_getTupleValue(tuple, 1), M_freeValue};

  EXPECT_EQ(M_Torch_getValueType(retValue0.get()),
            M_Torch_ValueType::M_TORCH_LONG_VALUE);
  EXPECT_EQ(M_Torch_getValueType(retValue1.get()),
            M_Torch_ValueType::M_TORCH_BOOL_VALUE);

  EXPECT_EQ(M_getLongFromValue(retValue0.get()), 2u);
  EXPECT_EQ(M_getBoolFromValue(retValue1.get()), false);

  M_Torch_freeTuple(tuple);
}

TEST_F(AIEngineTorchValueTest, ValueMapTest) {
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
