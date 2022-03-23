//===- TestPrimitives.cpp -------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'test' dialect primitives provided by the GraphRT
// library.  These are intended for writing executor testcases.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/PrimitiveFrame.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "GraphRT/Primitives.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "llvm/ADT/StringRef.h"
using namespace M;

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

static Chain print_hello(Chain chain) {
  printf("hello bef executor!\n");
  return chain;
}
static void memory_leak_one_int32(Runtime &runtime) {
  // We allocate an integer to intentionally cause a memory leak for testing.
  runtime.allocate<int32_t>();
}

//===----------------------------------------------------------------------===//
// String, Array, and Dict Tests
//===----------------------------------------------------------------------===//

static std::string get_string(BEFAttr<StringRef> value) { return value->str(); }

static std::string append_string(const std::string &lhs,
                                 const std::string &rhs) {
  return lhs + rhs;
}

static Chain print_string(const std::string &lhs, Chain chain) {
  printf("string = ");
  fwrite(lhs.data(), 1, lhs.size(), stdout);
  fputc('\n', stdout);
  return chain;
}

static Chain print_array_i32(Chain chain, BEFAttr<ArrayRef<int32_t>> values) {
  printf("array = [");
  for (size_t i = 0, e = values->size(); i != e; ++i) {
    if (i != 0)
      printf(", ");
    printf("%d", values[i]);
  }
  printf("]\n");
  return chain;
}

static Chain print_dict_i32(Chain chain, BEFAttr<StringRef> key,
                            BEFDictionaryAttr dict) {
  printf("dictionary[%.*s] = ", (int)key->size(), key->begin());
  auto value = dict.find<int32_t>(*key);
  if (!value)
    printf("NOTFOUND\n");
  else
    printf("%d\n", value->get());

  return chain;
}

static Chain print_all_dict(Chain chain, BEFDictionaryAttr dict) {
  printf("dictionary = ");
  llvm::interleave(
      dict,
      [](const BEFDictionaryAttr::Element &elt) {
        if (elt.getName() == "aString") {
          auto value = *(elt.getValue<StringRef>());
          printf("aString : %.*s", (int)value.size(), value.data());
        } else {
          printf("%.*s : %d", (int)elt.getName().size(), elt.getName().data(),
                 *(elt.getValue<int32_t>()));
        }
      },
      []() { printf(", "); });

  fputc('\n', stdout);
  return chain;
}

//===----------------------------------------------------------------------===//
// Dynamic attributes
//===----------------------------------------------------------------------===//

static Chain print_dynamic_attr(Chain chain, BEFDynamicAttr value) {
  if (auto intAttr = value.dyn_cast<int32_t>())
    printf("int32 = %d\n", intAttr->get());
  else if (auto strAttr = value.dyn_cast<StringRef>()) {
    printf("string = ");
    fwrite(strAttr.getValue()->data(), 1, strAttr.getValue()->size(), stdout);
    fputc('\n', stdout);
  } else if (auto intArray = value.dyn_cast<ArrayRef<int32_t>>()) {
    print_array_i32(chain, *intArray);
  } else if (auto dict = value.dyn_cast<Detail::Dictionary>()) {
    print_all_dict(chain, *dict);
  } else {
    printf("unsupported attribute kind\n");
  }
  return chain;
}

//===----------------------------------------------------------------------===//
// Value Tracking
//===----------------------------------------------------------------------===//

/// This class is just an integer but it prints on construction and destruction
/// so we can write tests showing that the values get created and destroyed at
/// the right times.
class VTValue {
public:
  explicit VTValue(int32_t v) : v(v) {
    printf("constructed vt.value(%d)\n", v);
  }

  VTValue(const VTValue &rhs) : v(rhs.v) {
    printf("copy constructed vt.value(%d)\n", v);
  }

  VTValue(VTValue &&rhs) : v(rhs.v) {
    rhs.v = -1;
    printf("move constructed vt.value(%d)\n", v);
  }

  ~VTValue() {
    if (v == -1)
      return;
    printf("destroyed vt.value(%d)\n", v);
  }

  int32_t v;
};

static VTValue vt_constant(BEFAttr<int32_t> v) { return VTValue(*v); }

static VTValue vt_add(const VTValue &lhs, const VTValue &rhs) {
  return VTValue(lhs.v + rhs.v);
}

static Chain vt_print(const VTValue &v) {
  printf("print vt.value(%d)\n", v.v);
  return Chain();
}

/// Share input args with results without copying.
static std::pair<AsyncValueRef<VTValue>, AsyncValueRef<VTValue>>
vt_share_to_two(AsyncValueRef<VTValue> in) {
  auto result1 = in.copy();
  return {std::move(result1), std::move(in)};
}

/// Share input args with results without copying.  This is the same as above,
/// but uses RCRef<AsyncValue> instead of AsyncValueRef<>.
static std::pair<RCRef<AsyncValue>, RCRef<AsyncValue>>
vt_share_to_two_2(RCRef<AsyncValue> in) {
  auto result1 = in.copy();
  return {std::move(result1), std::move(in)};
}

/// Share input args with results without copying.
static std::tuple<AsyncValueRef<VTValue>, AsyncValueRef<VTValue>,
                  AsyncValueRef<VTValue>, AsyncValueRef<VTValue>>
vt_share_to_four(AsyncValueRef<VTValue> in) {
  auto result1 = in.copy(), result2 = in.copy(), result3 = in.copy();
  return {std::move(result1), std::move(result2), std::move(result3),
          std::move(in)};
}

/// Share input args with results without copying.  This is the same as above,
/// but uses RCRef<AsyncValue> instead of AsyncValueRef<>.
static std::tuple<RCRef<AsyncValue>, RCRef<AsyncValue>, RCRef<AsyncValue>,
                  RCRef<AsyncValue>>
vt_share_to_four_2(RCRef<AsyncValue> in) {
  auto result1 = in.copy(), result2 = in.copy(), result3 = in.copy();
  return {std::move(result1), std::move(result2), std::move(result3),
          std::move(in)};
}

static void registerValueTracking(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "test.vt.constant", vt_constant);
  ADD_PRIMITIVE(registry, "test.vt.add", vt_add);
  ADD_PRIMITIVE(registry, "test.vt.print", vt_print);
  ADD_PRIMITIVE(registry, "test.vt_share_to_two", vt_share_to_two);
  ADD_PRIMITIVE(registry, "test.vt_share_to_two_2", vt_share_to_two_2);
  ADD_PRIMITIVE(registry, "test.vt_share_to_four", vt_share_to_four);
  ADD_PRIMITIVE(registry, "test.vt_share_to_four_2", vt_share_to_four_2);
}

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

// This produces only failure/success as Error.
static ErrorOr<int32_t> fail() { return Error("something bad happened"); }

// This produces a normal output and an error output.
static std::pair<ErrorOr<int32_t>, ErrorOr<int32_t>> partial_fail() {
  return {42, Error("something partially bad happened")};
}

//===----------------------------------------------------------------------===//
// Cancellation
//===----------------------------------------------------------------------===//

// Primitive for testing cancellation logic.
static Chain cancel(Chain inChain, Runtime &runtime) {
  // Do NOT do this in a normal primitive. Runtime::cancelExecution() should
  // only be called by clients external to the BEFExecutor.
  runtime.cancelExecution("Cancelled!");
  return inChain;
}

//===----------------------------------------------------------------------===//
// Async Operations
//===----------------------------------------------------------------------===//

static AsyncValueRef<int32_t> async_add_i32(int32_t lhs, int32_t rhs,
                                            Runtime &runtime) {
  // Even though a single scalar add is trivial, we can do it on a background
  // thread if we'd like!
  return addTask(runtime, [lhs, rhs] { return lhs + rhs; });
}

// It is silly to do this for add of course, but this shows how to handle this.
static void async_add2_i32(AsyncValueRef<int32_t> lhs,
                           AsyncValueRef<int32_t> rhs, Result<int32_t> result) {
  // If this were a non-strict primitive, this could get invoked before lhs/rhs
  // are resolved, and either or both may resolve to an error, so we need to
  // handle that.
  andThen(
      [result = result.copy()](AsyncValueRef<int32_t> lhs,
                               AsyncValueRef<int32_t> rhs) {
        // This actually only needs to check for error if this is defined as
        // non-strict.
        if (lhs->isError())
          result->setToError(lhs->getError().copy());
        else if (rhs->isError())
          result->setToError(rhs->getError().copy());
        else
          result.emplace(lhs.get() + rhs.get());
      },
      std::move(lhs), std::move(rhs));
}

/// Example that returns an indirect value with IndirectResult, nominally
/// because this is implemented in async style and returns different types
/// depending on the resolved types of lhs/rhs
static void async_add_any(RCRef<AsyncValue> lhs, RCRef<AsyncValue> rhs,
                          IndirectResult result) {
  if (lhs->isType<int32_t>() && rhs->isType<int32_t>())
    result.emplace<int32_t>(lhs->get<int32_t>() + rhs->get<int32_t>());
  else if (lhs->isType<std::string>() && rhs->isType<std::string>())
    result.emplace<std::string>(lhs->get<std::string>() +
                                rhs->get<std::string>());
  else
    result->setToError("invalid operands to async_add_any");
}

//===----------------------------------------------------------------------===//
// TestVector Primitives
//===----------------------------------------------------------------------===//

// This is intended to test parallel operations and other "advanced" use-cases.
namespace {
/// This class is a designed-to-be-simple data type that we can pass around to
/// illustrate some more advanced usecases for async value and related types.
struct TestVector {
  TestVector() = default;
  TestVector(const TestVector &) = default;
  TestVector(size_t n) { values.resize(n); }
  SmallVector<int32_t> values;
};
} // end anonymous namespace.

static TestVector vec_atoi(BEFAttr<int32_t> number) {
  TestVector result;
  result.values.reserve(*number);
  for (int32_t i = 0, e = *number; i != e; ++i)
    result.values.push_back(i);
  return result;
}

static TestVector vec_constant(BEFAttr<int32_t> size, BEFAttr<int32_t> value) {
  TestVector result;
  result.values.reserve(*size);
  for (int32_t i = 0, e = *size; i != e; ++i)
    result.values.push_back(*value);
  return result;
}

static ErrorOr<TestVector> vec_add(const TestVector &lhs,
                                   const TestVector &rhs) {
  if (lhs.values.size() != rhs.values.size())
    return Error("vector length mismatch");

  TestVector result;
  result.values.reserve(lhs.values.size());
  for (int32_t i = 0, e = lhs.values.size(); i != e; ++i)
    result.values.push_back(lhs.values[i] + rhs.values[i]);
  return result;
}

/// This version of vector addition does the entire add of the data on one
/// background task.
static void vec_add_async(AsyncValueRef<TestVector> lhs,
                          AsyncValueRef<TestVector> rhs,
                          Result<TestVector> result, Runtime &runtime) {
  if (lhs.get().values.size() != rhs.get().values.size()) {
    result->setToError("vector length mismatch");
    return;
  }

  addTask(runtime, [lhs = std::move(lhs), rhs = std::move(rhs),
                    result = result.copy()]() {
    auto &lhsValues = lhs.get().values, &rhsValues = rhs.get().values;
    TestVector resultValues;
    for (int32_t i = 0, e = lhsValues.size(); i != e; ++i)
      resultValues.values.push_back(lhsValues[i] + rhsValues[i]);
    // After we've populated the vector, set it into the result value.
    result.emplace(std::move(resultValues));
  });
}

/// This version of vector addition does the addition with each element in
/// parallel as separate tasks.
///
/// This is similar to vec_add_async, but uses `parallelForEachNCompletion`.
static void vec_add_async2(AsyncValueRef<TestVector> lhs,
                           AsyncValueRef<TestVector> rhs,
                           Result<TestVector> result, Runtime &runtime) {
  size_t lhsValueSize = lhs.get().values.size();
  if (lhsValueSize != rhs.get().values.size())
    return result->setToError("vector length mismatch");

  // Move the AsyncValue to "constructed" state with the specified # elements.
  result.construct(lhsValueSize);

  // Process each element in parallel.
  parallelForEachNCustomCompletion(
      runtime, lhsValueSize,
      [lhs = std::move(lhs),
       rhs = std::move(rhs)](size_t n, AsyncValueRef<TestVector> &result) {
        result.get().values[n] = lhs.get().values[n] + rhs.get().values[n];
      },
      [](AsyncValueRef<TestVector> &result) { result->markReady(); },
      // This is the state we capture, which is passed into each closure.
      result.copy());
}

/// This version of vector addition does the addition with each element in
/// parallel as separate tasks.
///
/// This is the same as vec_add_async2, but captures the result vector in a
/// slightly different way that doesn't need the for loop itself to capture
/// state, simplifying signatures.
static void vec_add_async3(AsyncValueRef<TestVector> lhs,
                           AsyncValueRef<TestVector> rhs,
                           Result<TestVector> result, Runtime &runtime) {
  size_t lhsValueSize = lhs.get().values.size();
  if (lhsValueSize != rhs.get().values.size())
    return result->setToError("vector length mismatch");

  // Move the AsyncValue to "constructed" state with the specified # elements.
  result.construct(lhsValueSize);

  // Process each element in parallel.
  parallelForEachNCustomCompletion(
      runtime, lhsValueSize,
      // Capturing result by reference here is safe since the completion handler
      // guarantees it stays alive across execution of all elements.
      [&result = result.get(), lhs = std::move(lhs),
       rhs = std::move(rhs)](size_t n) {
        result.values[n] = lhs.get().values[n] + rhs.get().values[n];
      },
      [result = result.copy()]() { result->markReady(); });
}

/// This version of vector addition does the addition with each element in
/// parallel as separate tasks.
///
/// This is the same as vec_add_async3, but uses a convenience method to mark
/// the resulting values as ready.
static void vec_add_async4(AsyncValueRef<TestVector> lhs,
                           AsyncValueRef<TestVector> rhs,
                           Result<TestVector> result, Runtime &runtime) {
  size_t lhsValueSize = lhs.get().values.size();
  if (lhsValueSize != rhs.get().values.size())
    return result->setToError("vector length mismatch");

  // Move the AsyncValue to "constructed" state with the specified # elements.
  result.construct(lhsValueSize);

  // Process each element in parallel.
  parallelForEachNMarkReady(runtime, lhsValueSize, result.copy(),
                            [result = result.copy(), lhs = std::move(lhs),
                             rhs = std::move(rhs)](size_t n) {
                              result.get().values[n] =
                                  lhs.get().values[n] + rhs.get().values[n];
                            });
}

static Chain vec_print(const TestVector &vec, Chain chain) {
  printf("vec = ");
  for (size_t i = 0, e = vec.values.size(); i != e; ++i) {
    if (i != 0)
      printf(", ");
    printf("%d", vec.values[i]);
  }
  fputc('\n', stdout);
  return chain;
}

static void registerTestVecPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "test.vec.atoi", vec_atoi);
  ADD_PRIMITIVE(registry, "test.vec.constant", vec_constant);
  ADD_PRIMITIVE(registry, "test.vec.add", vec_add);
  ADD_PRIMITIVE(registry, "test.vec.add.async", vec_add_async);
  ADD_PRIMITIVE(registry, "test.vec.add.async2", vec_add_async2);
  ADD_PRIMITIVE(registry, "test.vec.add.async3", vec_add_async3);
  ADD_PRIMITIVE(registry, "test.vec.add.async4", vec_add_async4);
  ADD_PRIMITIVE(registry, "test.vec.print", vec_print);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// This registers the 'test' dialect primitives, intended for writing
/// executor testcases.
void M::registerGraphRTTestPrimitives(PrimitiveRegistry &registry) {
  // Misc.
  ADD_PRIMITIVE(registry, "test.print_hello", print_hello);
  ADD_PRIMITIVE(registry, "test.memory_leak_one_int32", memory_leak_one_int32);

  // String and Array Tests
  ADD_PRIMITIVE(registry, "test.get_string", get_string);
  ADD_PRIMITIVE(registry, "test.append_string", append_string);
  ADD_PRIMITIVE(registry, "test.print_string", print_string);
  ADD_PRIMITIVE(registry, "test.print_array_i32", print_array_i32);
  ADD_PRIMITIVE(registry, "test.print_dict_i32", print_dict_i32);
  ADD_PRIMITIVE(registry, "test.print_all_dict", print_all_dict);

  ADD_PRIMITIVE(registry, "test.print.dynamic.attr", print_dynamic_attr);

  registerValueTracking(registry);
  registerTestVecPrimitives(registry);

  // Error handling.
  ADD_PRIMITIVE(registry, "test.fail", fail);
  ADD_PRIMITIVE(registry, "test.partial_fail", partial_fail);

  // Cancellation.
  ADD_PRIMITIVE(registry, "test.cancel", cancel);

  ADD_PRIMITIVE(registry, "test.async.add.i32", async_add_i32);
  ADD_PRIMITIVE(registry, "test.async.add2.i32", async_add2_i32);
  ADD_PRIMITIVE(registry, "test.async.add.any", async_add_any);
}
