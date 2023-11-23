//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Some unit tests for List, Dict, Tuple and friends.
//
//===----------------------------------------------------------------------===//

#include "PyTorch/Support/Value.h"
#include "GenericML/Support/Tensor.h"
#include "LLCL/Runtime/AnyAsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "PyTorch/Support/Dict.h"
#include "PyTorch/Support/FormatYAML.h"
#include "PyTorch/Support/List.h"

#include "gtest/gtest.h"
#include <llvm/Support/YAMLTraits.h>

using namespace M;
using namespace M::Torch;

namespace {

struct ValueTest : ::testing::Test {};

static LLCL::Runtime *getRuntime() {
  static LLCL::Runtime *runtime = []() {
    std::unique_ptr<Allocator> allocator =
        LLCL::createProfilingAllocator(LLCL::createMallocAllocator());
    std::unique_ptr<LLCL::WorkQueue> workQueue =
        LLCL::createSingleThreadWorkQueue();
    return new LLCL::Runtime(std::move(allocator), std::move(workQueue));
  }();
  return runtime;
}

static GML::Tensor makeTensor(size_t size) {
  M::TensorSpec spec({size}, DType::f32);
  GML::Tensor result =
      GML::Tensor::createOwned(std::move(spec), getRuntime(), /*alignment=*/{});
  return result;
}

static List makeTensorList(size_t size) {
  List list;
  for (size_t i = 0; i < size; ++i) {
    TensorSpec spec({i}, M::DType::f32);

    GML::Tensor tensor = makeTensor(i);
    list.getElements().push_back(
        LLCL::AnyAsyncValueRef::createReady<GML::Tensor>(getRuntime(),
                                                         std::move(tensor)));
  }

  return list;
}

template <typename T>
static std::string toYamlString(T t) {
  std::string yamlString;
  llvm::raw_string_ostream yamlStream(yamlString);
  llvm::yaml::Output yamlOutput(yamlStream);
  yamlOutput << t;
  yamlStream.flush();
  return yamlString;
}

TEST_F(ValueTest, ListTest) {
  List list1 = makeTensorList(3);

  List list2 = makeTensorList(3);

  // Lists contain distinct tensors, and are not the same.
  ASSERT_NE(list1, list2);

  List list3;

  for (M::Torch::Variant &elem : list1.getElements()) {
    list3.getElements().push_back(
        LLCL::AnyAsyncValueRef::copy(elem.getValue().getPointer()));
  }

  // Lists contain the same tensors, and are the same.
  ASSERT_EQ(list1, list3);
}

TEST_F(ValueTest, TupleOfNoneTest) {
  Tuple tuple1;
  Tuple tuple2;
  Tuple tuple3;

  tuple1.getElements().push_back(
      LLCL::AnyAsyncValueRef::createReady<None>(getRuntime()));
  tuple2.getElements().push_back(
      LLCL::AnyAsyncValueRef::createReady<None>(getRuntime()));
  tuple3.getElements().push_back(
      LLCL::AnyAsyncValueRef::createReady<None>(getRuntime()));

  tuple1.getElements().push_back(
      LLCL::AnyAsyncValueRef::createReady<None>(getRuntime()));
  tuple2.getElements().push_back(
      LLCL::AnyAsyncValueRef::createReady<None>(getRuntime()));

  ASSERT_EQ(tuple1, tuple2);
  ASSERT_NE(tuple1, tuple3);
  ASSERT_NE(tuple2, tuple3);

  ASSERT_EQ(std::to_string(tuple1), "Tuple(<none>, <none>)");
  ASSERT_EQ(std::to_string(tuple3), "Tuple(<none>)");
}

TEST_F(ValueTest, DictTest) {
  Dict dict1;
  Dict dict2;
  Dict dict3;

  dict1.getElements()[LLCL::AnyAsyncValueRef::createReady<std::string>(
      getRuntime(), "a")] =
      LLCL::AnyAsyncValueRef::createReady<None>(getRuntime());

  dict2.getElements()[LLCL::AnyAsyncValueRef::createReady<std::string>(
      getRuntime(), "a")] =
      LLCL::AnyAsyncValueRef::createReady<None>(getRuntime());

  dict3.getElements()[LLCL::AnyAsyncValueRef::createReady<std::string>(
      getRuntime(), "b")] =
      LLCL::AnyAsyncValueRef::createReady<None>(getRuntime());

  GML::Tensor tensor = makeTensor(1);
  dict3.getElements()[LLCL::AnyAsyncValueRef::createReady<std::string>(
      getRuntime(), "b")] =
      LLCL::AnyAsyncValueRef::createReady<GML::Tensor>(getRuntime(),
                                                       std::move(tensor));

  ASSERT_EQ(std::to_string(dict1), "Dict{\"a\": <none>}");
}

TEST_F(ValueTest, VariantDictKeys) {
  List list;
  list.getElements().push_back(
      LLCL::AnyAsyncValueRef::createReady<std::string>(getRuntime(), "a"));

  list.getElements().push_back(
      LLCL::AnyAsyncValueRef::createReady<int64_t>(getRuntime(), 2));

  Dict dict1;
  Dict dict2;

  dict1.getElements()[Variant(
      LLCL::AnyAsyncValueRef::createReady<List>(getRuntime(), list))] =
      LLCL::AnyAsyncValueRef::createReady<None>(getRuntime());

  dict2.getElements()[Variant(
      LLCL::AnyAsyncValueRef::createReady<List>(getRuntime(), list))] =
      LLCL::AnyAsyncValueRef::createReady<None>(getRuntime());

  ASSERT_EQ(dict1, dict2);
  ASSERT_EQ(std::to_string(dict1), "Dict{List[\"a\", 2]: <none>}");
}

TEST_F(ValueTest, YamlTest0) {

  Dict dict1;

  dict1.getElements()[Variant(
      LLCL::AnyAsyncValueRef::createReady<std::string>(getRuntime(), "a"))] =
      LLCL::AnyAsyncValueRef::createReady<None>(getRuntime());

  dict1.getElements()[Variant(
      LLCL::AnyAsyncValueRef::createReady<double>(getRuntime(), 1.0))] =
      LLCL::AnyAsyncValueRef::createReady<int64_t>(getRuntime(), 5);

  std::string ystring = toYamlString(dict1);

  // Test YAML string.
  ASSERT_EQ(ystring, "---\n"
                     "dict:\n"
                     "  - key:\n"
                     "      string:          a\n"
                     "    value:\n"
                     "      none:            {}\n"
                     "  - key:\n"
                     "      double:          1\n"
                     "    value:\n"
                     "      long:            5\n"
                     "...\n");

  llvm::yaml::Input yin(ystring);

  Dict dict2;
  yin >> dict2;

  // Test round tripping through YAML.
  ASSERT_EQ(dict1, dict2);
}

TEST_F(ValueTest, YamlTest1) {

  List list;
  list.getElements().push_back(
      LLCL::AnyAsyncValueRef::createReady<std::string>(getRuntime(), "a"));

  list.getElements().push_back(
      LLCL::AnyAsyncValueRef::createReady<int64_t>(getRuntime(), 2));

  Dict dict1;

  dict1.getElements()[Variant(
      LLCL::AnyAsyncValueRef::createReady<List>(getRuntime(), list))] =
      LLCL::AnyAsyncValueRef::createReady<None>(getRuntime());

  // Test YAML string.
  std::string ystring = toYamlString(dict1);

  ASSERT_EQ(ystring, "---\n"
                     "dict:\n"
                     "  - key:\n"
                     "      list:\n"
                     "        list:\n"
                     "          - string:          a\n"
                     "          - long:            2\n"
                     "    value:\n"
                     "      none:            {}\n"
                     "...\n");

  llvm::yaml::Input yin(ystring);

  Dict dict2;
  yin >> dict2;

  ASSERT_EQ(dict1, dict2);
}

} // namespace
