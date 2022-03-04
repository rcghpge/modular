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
vt_share_to_two(AsyncValueRef<VTValue> &in) {
  auto result1 = in.copy();
  return {std::move(result1), std::move(in)};
}

/// Share input args with results without copying.  This is the same as above,
/// but uses RCRef<AsyncValue> instead of AsyncValueRef<>.
static std::pair<RCRef<AsyncValue>, RCRef<AsyncValue>>
vt_share_to_two_2(RCRef<AsyncValue> &in) {
  auto result1 = in.copy();
  return {std::move(result1), std::move(in)};
}

static void registerValueTracking(PrimitiveRegistry &registry) {
  registry.add("test.vt.constant", M_PRIMITIVE(vt_constant));
  registry.add("test.vt.add", M_PRIMITIVE(vt_add));
  registry.add("test.vt.print", M_PRIMITIVE(vt_print));
  registry.add("test.vt_share_to_two", M_PRIMITIVE(vt_share_to_two));
  registry.add("test.vt_share_to_two_2", M_PRIMITIVE(vt_share_to_two_2));
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

/// This registers the 'test' dialect primitives, intended for writing executor
/// testcases.
void M::registerGraphRTTestPrimitives(PrimitiveRegistry &registry) {
  // Misc.
  registry.add("test.print_hello", M_PRIMITIVE(print_hello));
  registry.add("test.memory_leak_one_int32",
               M_PRIMITIVE(memory_leak_one_int32));

  // String and Array Tests
  registry.add("test.get_string", M_PRIMITIVE(get_string));
  registry.add("test.append_string", M_PRIMITIVE(append_string));
  registry.add("test.print_string", M_PRIMITIVE(print_string));
  registry.add("test.print_array_i32", M_PRIMITIVE(print_array_i32));

  registerValueTracking(registry);

  // Error handling.
  registry.add("test.fail", M_PRIMITIVE(fail));
  registry.add("test.partial_fail", M_PRIMITIVE(partial_fail));

  // Cancellation.
  registry.add("test.cancel", M_PRIMITIVE(cancel));

  registry.add("test.async.add.i32", M_PRIMITIVE(async_add_i32));
}
