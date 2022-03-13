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
// String and Array Tests
//===----------------------------------------------------------------------===//

static std::string get_string(BEFAttribute<StringRef> value) {
  return value->str();
}

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

static Chain print_array_i32(Chain chain,
                             BEFAttribute<ArrayRef<int32_t>> values) {
  printf("array = ");
  for (size_t i = 0, e = values->size(); i != e; ++i) {
    if (i != 0)
      printf(", ");
    printf("%d", values[i]);
  }
  fputc('\n', stdout);
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

static VTValue vt_constant(BEFAttribute<int32_t> v) { return VTValue(*v); }

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
  // TODO: Registration of these types should get folded into auto-magic as part
  // of the ADD_PRIMITIVE macro.  Issue #33.
  AsyncValue::registerType<VTValue>();
  AsyncValue::registerType<Chain>();

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

//===----------------------------------------------------------------------===//
// TestVector Primitives
//===----------------------------------------------------------------------===//

// This is intended to test parallel operations and other "advanced" use-cases.
namespace {
/// This class is a designed-to-be-simple data type that we can pass around to
/// illustrate some more advanced usecases for async value and related types.
struct TestVector {
  SmallVector<int32_t> values;
};
} // end anonymous namespace.

static TestVector vec_atoi(BEFAttribute<int32_t> number) {
  TestVector result;
  result.values.reserve(*number);
  for (int32_t i = 0, e = *number; i != e; ++i)
    result.values.push_back(i);
  return result;
}

static TestVector vec_constant(BEFAttribute<int32_t> size,
                               BEFAttribute<int32_t> value) {
  TestVector result;
  result.values.reserve(*size);
  for (int32_t i = 0, e = *size; i != e; ++i)
    result.values.push_back(*value);
  return result;
}

static ErrorOr<TestVector> vec_add(const TestVector &lhs,
                                   const TestVector &rhs) {
  if (lhs.values.size() != rhs.values.size())
    return Error("test.vec.add length mismatch");

  TestVector result;
  result.values.reserve(lhs.values.size());
  for (int32_t i = 0, e = lhs.values.size(); i != e; ++i)
    result.values.push_back(lhs.values[i] + rhs.values[i]);
  return result;
}

static AsyncValueRef<TestVector> vec_add_async(AsyncValueRef<TestVector> lhs,
                                               AsyncValueRef<TestVector> rhs,
                                               Runtime &runtime) {
  if (lhs.get().values.size() != rhs.get().values.size())
    return AsyncValue::createError(runtime, "test.vec.add length mismatch");

  auto result = AsyncValueRef<TestVector>::createConstructed(runtime);
  addTask(runtime, [lhs = std::move(lhs), rhs = std::move(rhs),
                    result = result.copy()]() {
    auto &lhsValues = lhs.get().values, &rhsValues = rhs.get().values;
    auto &resultValues = result.get().values;
    for (int32_t i = 0, e = lhsValues.size(); i != e; ++i)
      resultValues.push_back(lhsValues[i] + rhsValues[i]);
    // After we've populated the map, we can return it.
    result->markReady();
  });

  return result;
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
  // TODO: Registration of these types should get folded into auto-magic as part
  // of the ADD_PRIMITIVE macro.  Issue #33.
  AsyncValue::registerType<TestVector>();

  ADD_PRIMITIVE(registry, "test.vec.atoi", vec_atoi);
  ADD_PRIMITIVE(registry, "test.vec.constant", vec_constant);
  ADD_PRIMITIVE(registry, "test.vec.add", vec_add);
  ADD_PRIMITIVE(registry, "test.vec.add.async", vec_add_async);
  ADD_PRIMITIVE(registry, "test.vec.print", vec_print);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// This registers the 'test' dialect primitives, intended for writing
/// executor testcases.
void M::registerGraphRTTestPrimitives(PrimitiveRegistry &registry) {
  // TODO: Registration of these types should get folded into auto-magic as part
  // of the ADD_PRIMITIVE macro.  Issue #33.

  // Misc.
  AsyncValue::registerType<Chain>();
  AsyncValue::registerType<int32_t>();
  ADD_PRIMITIVE(registry, "test.print_hello", print_hello);
  ADD_PRIMITIVE(registry, "test.memory_leak_one_int32", memory_leak_one_int32);

  // String and Array Tests
  AsyncValue::registerType<std::string>();
  ADD_PRIMITIVE(registry, "test.get_string", get_string);
  ADD_PRIMITIVE(registry, "test.append_string", append_string);
  ADD_PRIMITIVE(registry, "test.print_string", print_string);
  ADD_PRIMITIVE(registry, "test.print_array_i32", print_array_i32);

  registerValueTracking(registry);
  registerTestVecPrimitives(registry);

  // Error handling.
  ADD_PRIMITIVE(registry, "test.fail", fail);
  ADD_PRIMITIVE(registry, "test.partial_fail", partial_fail);

  // Cancellation.
  ADD_PRIMITIVE(registry, "test.cancel", cancel);

  ADD_PRIMITIVE(registry, "test.async.add.i32", async_add_i32);
}
