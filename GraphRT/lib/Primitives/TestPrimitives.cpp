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
#include "LLCL/ForwardDecls.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/Chain.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/StringRef.h"
using namespace M;

//===----------------------------------------------------------------------===//
// Helper Classes For Primitive Implementations
//===----------------------------------------------------------------------===//

// Utility template for tag dispatching.
template <typename T>
struct TypeTag {};

//===----------------------------------------------------------------------===//
// Primitive Registration Wrapper Generator
//===----------------------------------------------------------------------===//

// M_PRIMITIVE is a macro that makes defining primitives more straightforward.
// For simple primitives with a few arguments and a single movable result,
// you can define the function using the native C++ types directly:
//
//   int32_t Add(int32_t a, int32_t b) { return a + b; }
//
// For non-strict primitives where one or more of the arguments are
// asynchronous, you can use the "Argument" wrapper type as your function
// arguments and return the result by value. The Argument<> internally wraps an
// AsyncValue:
//
//   int32_t Add(Argument<int32_t> a, Argument<int32_t> b) { return *a + *b; }
//
// All primitives defined with M_PRIMITIVE accept an implicit Chain as their
// last argument. For example, the Add primitive above could be called like
// this:
//
//   %c0 = bef.chain.new
//   %z = "Add"(%x, %y, %c0) : (i32, i32, !bef.chain) -> i32
//
// For primitives with multiple results, you can return std::pair/tuple as the
// result type:
//
//   // q = (n / d), r = n % d;
//   std::pair<int32_t, int32_t> DivRem(int32_t n, int32_t d) {
//     return {n / d, n % d};
//   }
//
// Or you can use the "Result" wrapper type. Results should appear after
// arguments:
//
//   // q = (n / d), r = n % d;
//   void DivRem(Argument<int32_t> n, Argument<int32_t> d,
//               Result<int32_t> q, Result<int32_t> r) {
//     q.Emplace(*n / *d);
//     r.Emplace(*n % *d);
//   }
//
// There is also an "Attribute" wrapper for primitives that need to consume
// attribute values. Note that attribute values are essentially literals, so
// these should generally only be used for things that never change for a given
// executable, as opposed to values that will be computed at runtime. Attributes
// should appear after arguments and results:
//
//   int32_t MakeInt(Attribute<int32_t> value) {
//     return *value;
//   }
//
// Similarly StringAttribute, ArrayAttribute<T> and AggregateAttr are also
// provided. They work the same way as attribute but for arrays of T or
// characters or nested arrays of heterogeneous types.
//
// For primitives that may fail at runtime, for sync primitives the preferred
// way is to return ErrorOr<T> to report the error:
//
//   ErrorOr<std::string> ReadFile(std::string path) {
//     auto* f = OpenFile(*path);
//     if (!f) {
//       return Error("Could not open file");
//     }
//     <Read file here>
//     return bytes;
//   }
//
// PrimitiveErrorHandler may be used for sync primitives:
//
//   void ReadFile(Argument<std::string> path, Result<std::string> bytes,
//                 PrimitiveErrorHandler handler) {
//     auto* f = OpenFile(*path);
//     if (!f) {
//       handler.ReportError("Could not open file");
//       return;
//     }
//     <Read file here>
//   }
//
// WARNING: PrimitiveErrorHandler can't be used asynchronously because it holds
// a pointer to the PrimitiveFrame, which is destroyed when the primitive
// returns.
//
// Primitivess can also take the PrimitiveFrame or Runtime if they need access
// to anything else the above wrapper types don't provide.
//
// See the definitions of the wrapper types below for more details.
//

// TODO(C++'17): Switch to template when we can use C++17 with 'auto' template
// requirements.
#define M_PRIMITIVE(...)                                                       \
  ::M::Detail::PrimitiveImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::invoke

namespace M {
namespace Detail {
// This class implements the M_PRIMITIVE macro.
template <typename FnType, FnType implFn>
struct PrimitiveImpl;

template <typename Return, typename... Args, Return (*implFn)(Args...)>
struct PrimitiveImpl<Return (*)(Args...), implFn> {
  /// This is the main entry point that gets registered as an impl.  This works
  /// by recursively processing the argument list, then handling the result
  /// list, generating a function with the standard PrimitiveImpl signature that
  /// handles each part of the function signature as it goes.
  static void invoke(PrimitiveFrame &frame) {
    CallHelper<Args..., TypeTag<int>>::template invoke<
        /*argIdx=*/0, /*resultIdx=*/0, /*attrIdx=*/0, /*funcIdx=*/0,
        /*hasErrorHandler=*/false, /*hasInChain=*/false>(frame);
  }

  /// Helper that introspects the arguments to derive the signature and
  /// cast parts of the PrimitiveFrame to their appropriate type before passing
  /// them to implFn. Works by recursively unpacking the arguments.
  template <typename... RemainingArgs>
  struct CallHelper;

  //===--------------------------------------------------------------------===//
  // Argument handling
  //===--------------------------------------------------------------------===//

  // TODO: Handle Argument/Attribute/etc.

  /// If this requires the PrimitiveFrame for any reason, pass it along.
  template <typename... Tail>
  struct CallHelper<PrimitiveFrame *, Tail...> {
    template <int argIdx, int resultIdx, int attrIdx, int funcIdx,
              bool hasErrorHandler, bool hasInChain, typename... PreviousArgs>
    static void invoke(PrimitiveFrame &frame, const PreviousArgs &...pargs) {
      CallHelper<Tail...>::template invoke<argIdx, resultIdx, attrIdx, funcIdx,
                                           hasErrorHandler, hasInChain>(
          frame, pargs..., &frame);
    }
  };

  /// If this requires a Runtime, pass it along.
  template <typename... Tail>
  struct CallHelper<Runtime *, Tail...> {
    template <int argIdx, int resultIdx, int attrIdx, int funcIdx,
              bool hasErrorHandler, bool hasInChain, typename... PreviousArgs>
    static void invoke(PrimitiveFrame &frame, const PreviousArgs &...pargs) {
      CallHelper<Tail...>::template invoke<argIdx, resultIdx, attrIdx, funcIdx,
                                           hasErrorHandler, hasInChain>(
          frame, pargs..., &frame.getRuntime());
    }
  };

  // Treat any other type as an Argument.
  template <typename Head, typename... Tail>
  struct CallHelper<Head, Tail...> {
    using ArgTy = std::decay_t<Head>;

    template <int argIdx, int resultIdx, int attrIdx, int funcIdx,
              bool hasErrorHandler, bool hasInChain, typename... PreviousArgs>
    static void invoke(PrimitiveFrame &frame, const PreviousArgs &...pargs) {
      static_assert(argIdx != -1,
                    "Do not place Arguments after RemainingArguments");
      static_assert(resultIdx == 0, "Arguments should appear before results");
      static_assert(attrIdx == 0,
                    "Arguments and results should appear before attributes");
      static_assert(funcIdx == 0,
                    "Arguments and results should appear before functions");
      auto &&arg = frame.getArg<ArgTy>(argIdx);

      constexpr bool isArgChain = std::is_same<ArgTy, LLCL::Chain>();
      CallHelper<Tail...>::template invoke<argIdx + 1, resultIdx, attrIdx,
                                           funcIdx, hasErrorHandler,
                                           (hasInChain || isArgChain)>(
          frame, pargs..., arg);
    }
  };

  // Base case: No arguments left.
  // TypeTag<T> is a dummy template parameter to work around the restriction
  // of GCC that fully specialized template is not allowed in a template class.
  template <typename T>
  struct CallHelper<TypeTag<T>> {
    // Verify the result index for non-void Return type
    template <typename ReturnT, int resultIdx>
    struct AssertIndex {
      static void verify(PrimitiveFrame &frame) {
        static_assert(resultIdx == 0,
                      "Don't mix return by value and Result<> syntax.");
      }
    };

    // Verify the result index for void Return type.
    template <int resultIdx>
    struct AssertIndex<void, resultIdx> {
      static void verify(PrimitiveFrame &frame) {
        assert((resultIdx == frame.getNumResults() || resultIdx == -1) &&
               "Extra results passed to primitive");
      }
    };

    template <int argIdx, int resultIdx, int attrIdx, int funcIdx,
              bool hasErrorHandler, bool hasInChain, typename... PreviousArgs>
    static void invoke(PrimitiveFrame &frame, const PreviousArgs &...pargs) {
      AssertIndex<Return, resultIdx>::verify(frame);

      assert((argIdx == frame.getNumArgs() || argIdx == -1 ||
              (!hasInChain && argIdx == frame.getNumArgs() - 1 &&
               frame.getArg(argIdx)->template isType<LLCL::Chain>())) &&
             "Extra arguments passed to primitive");
      assert((attrIdx == frame.getNumAttrs() || attrIdx == -1) &&
             "Extra attributes passed to primitive");
      assert((funcIdx == frame.getNumFunctions() || funcIdx == -1) &&
             "Extra functions passed to primitive");
      ReturnHelper<hasErrorHandler, Return>::invoke(frame, pargs...);
    }
  };

  //===--------------------------------------------------------------------===//
  // Result handling
  //===--------------------------------------------------------------------===//

  // Handles the return value of the primitive with `handleReturn` if non-void.
  // Otherwise ignores the return value.
  template <bool hasErrorHandler, typename T>
  struct ReturnHelper {
    static void invoke(PrimitiveFrame &frame, const Args &...args) {
      static_assert(
          !hasErrorHandler,
          "Do not return by value when using PrimitiveErrorHandler. Set the "
          "return value with Result<> instead. "
          "Bad: Chain myFunction(PrimitiveErrorHandler handler). "
          "Good: void myFunction(Result<Chain> out, PrimitiveErrorHandler "
          "handler).");
      handleReturn(frame, implFn(args...));
    }
  };
  template <bool hasErrorHandler>
  struct ReturnHelper<hasErrorHandler, void> {
    static void invoke(PrimitiveFrame &frame, const Args &...args) {
      implFn(args...);
    }
  };

  // These functions provide custom behavior for concrete result types that may
  // come up.

  /// Store result as an AsyncValue output in PrimitiveFrame by creating a
  /// ConcreteAsyncValue.
  template <typename T>
  static void processResult(PrimitiveFrame &frame, size_t index, T &&t) {
    frame.emplaceResult<std::decay_t<T>>(index, std::forward<T>(t));
  }

  /// Support functions that return ErrorOr<T>: if it contains an error, call
  /// frame.fail() to report the error message and set an error in the output
  /// AsyncValue. Otherwise, store the return value as normal.
  template <typename T>
  static void processResult(PrimitiveFrame &frame, size_t index,
                            ErrorOr<T> &&errorOr) {
    if (errorOr)
      processResult(frame, index, std::move(errorOr.get()));
    else
      frame.fail(errorOr.takeErrorStorage());
  }

  /// Don't allocate an async value for chain results, just use the one already
  /// stored in Runtime.
  static void processResult(PrimitiveFrame &frame, size_t index,
                            LLCL::Chain t) {
    frame.setResult(index, frame.getRuntime().getReadyChain());
  }

  /// Handle explicit AsyncValueRef results.
  template <typename T>
  static void processResult(PrimitiveFrame &frame, size_t index,
                            AsyncValueRef<T> t) {
    frame.setResult(index, std::move(t));
  }

  /// Store the function result back to the output AsyncValue in the
  /// PrimitiveFrame.
  template <typename T>
  static void handleReturn(PrimitiveFrame &frame, T &&t) {
    assert(frame.getNumResults() == 1 && "Incorrect number of results");
    processResult(frame, 0, std::forward<T>(t));
  }

  /// Support functions that return std::pair<>, storing the two results in
  /// separate output results.
  template <typename T1, typename T2>
  static void handleReturn(PrimitiveFrame &frame, std::pair<T1, T2> &&pair) {
    assert(frame.getNumResults() == 2 && "Incorrect number of results");
    processResult(frame, 0, std::move(pair.first));
    processResult(frame, 1, std::move(pair.second));
  }

  /// TODO: Support std::tuple results.
};
} // end namespace Detail
} // end namespace M

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

using LLCL::Chain;

static Chain test_print_hello(Chain chain) {
  printf("hello bef executor!\n");
  fflush(stdout);
  return chain;
}

static void test_constant_i32(PrimitiveFrame &frame) {
  assert(frame.getNumArgs() == 0 && frame.getNumResults() == 1 &&
         frame.getNumAttrs() == 1);
  frame.emplaceResult<int32_t>(0, frame.getAttr<int32_t>(0));
}

static Chain test_print_i32(int32_t value, Chain chain) {
  printf("int32 = %d\n", value);
  fflush(stdout);
  return chain;
}

static int32_t test_add_i32(int32_t lhs, int32_t rhs) { return lhs + rhs; }

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

// This produces only failure/success as Error.
static ErrorOr<int32_t> test_fail() { return Error("something bad happened"); }

// This produces a normal output and an error output.
static std::pair<ErrorOr<int32_t>, ErrorOr<int32_t>> test_partial_fail() {
  return {42, Error("something partially bad happened")};
}

//===----------------------------------------------------------------------===//
// Async Operations
//===----------------------------------------------------------------------===//

static AsyncValueRef<int32_t> test_async_add_i32(Runtime *runtime, int32_t lhs,
                                                 int32_t rhs) {
  // Even though a single scalar add is trivial, we can do it on a background
  // thread if we'd like!
  return addTask(*runtime, [lhs, rhs] { return lhs + rhs; });
}

/// This registers the 'test' dialect primitives, intended for writing executor
/// testcases.
void M::registerGraphRTTestPrimitives(PrimitiveRegistry &registry) {
  registry.add("test.print_hello", M_PRIMITIVE(test_print_hello));
  registry.add("test.constant.i32", test_constant_i32);
  registry.add("test.print.i32", M_PRIMITIVE(test_print_i32));
  registry.add("test.add.i32", M_PRIMITIVE(test_add_i32));

  registry.add("test.fail", M_PRIMITIVE(test_fail));
  registry.add("test.partial_fail", M_PRIMITIVE(test_partial_fail));

  registry.add("test.async.add.i32", M_PRIMITIVE(test_async_add_i32));
}
