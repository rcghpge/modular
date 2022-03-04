//===- GraphRT/BEFExecutor/PrimitiveRegistry.h ------------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_BEFEXECUTOR_PRIMITIVE_REGISTRY_H
#define GRAPHRT_BEFEXECUTOR_PRIMITIVE_REGISTRY_H

#include "GraphRT/BEFExecutor/PrimitiveFrame.h"
#include "LLCL/ForwardDecls.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/Chain.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "Support/ErrorOr.h"

namespace M {

/// This represents a mapping between the names of the MLIR opcodes to the
/// implementations of those functions, along with type mappings.
class PrimitiveRegistry final {
public:
  PrimitiveRegistry();
  ~PrimitiveRegistry();

  /// Primitive implementations use this signature, acting on the data provided
  /// in the argument frame, and returning results through it.
  using PrimitiveImplFn = void (*)(PrimitiveFrame &frame);

  /// Add an implementation of a primitive to the registry.  Consider using
  /// the M_PRIMITIVE macro to enable various pretty forms for this.
  void add(StringLiteral name, PrimitiveImplFn fn);

  /// Finalize the registry. Calling `get()` before the registry is finalized
  /// will result in undefined behavior.
  void finalize();

  /// Look up a primitive with a specified name out of the registry, returning
  /// null if no match is found.
  PrimitiveImplFn get(StringRef name) const;

private:
  PrimitiveRegistry(const PrimitiveRegistry &) = delete;
  PrimitiveRegistry &operator=(const PrimitiveRegistry &) = delete;

  struct Impl;
  std::unique_ptr<Impl> impl;
};

//===----------------------------------------------------------------------===//
// Helper Classes For Primitive Implementations
//===----------------------------------------------------------------------===//

// Utility template for tag dispatching.
template <typename T>
struct TypeTag {};

/// RemainingArguments collects all remaining arguments in an ArrayRef. There
/// can be at most one RemainingArguments/RepeatedArguments, and it must appear
/// after all other Arguments.
//
/// This should only be used when argument types have potentially heterogenous
/// types, for example GRT::CallOp.
class RemainingArguments {
public:
  explicit RemainingArguments(MutableArrayRef<RCRef<AsyncValue>> arguments)
      : arguments(arguments) {}

  MutableArrayRef<RCRef<AsyncValue>> get() const { return arguments; }
  size_t size() const { return arguments.size(); }
  RCRef<AsyncValue> &operator[](size_t i) const { return arguments[i]; }

private:
  MutableArrayRef<RCRef<AsyncValue>> arguments;
};

/// RemainingResults collects all remaining results in a MutableArrayRef. There
/// can be at most one RemainingResults, and it must appear after all other
/// Results.
///
/// This should be used when a primitive has potentially heterogenous types, for
/// example, GRT::CallOp.
class RemainingResults {
public:
  explicit RemainingResults(MutableArrayRef<RCRef<AsyncValue>> results)
      : results(results) {}

  MutableArrayRef<RCRef<AsyncValue>> get() const { return results; }
  size_t size() const { return results.size(); }
  RCRef<AsyncValue> &operator[](size_t i) const { return results[i]; }

private:
  MutableArrayRef<RCRef<AsyncValue>> results;
};

/// Primitives use this to declare that they have an attribute input with the
/// specified C++ type.
template <typename T>
class BEFAttribute {
public:
  explicit BEFAttribute(const void *value)
      : value(*reinterpret_cast<const T *>(value)) {}

  const T &get() const { return value; }
  const T *operator->() const { return &value; }
  const T &operator*() const { return value; }

private:
  const T &value;
};

//===----------------------------------------------------------------------===//
// Primitive Registration Wrapper Generator
//===----------------------------------------------------------------------===//

// M_PRIMITIVE is a macro that makes defining primitives more straightforward.
// For simple primitives with a few arguments and a single movable result,
// you can define the function using the native C++ types directly:
//
//   int32_t add(int32_t a, int32_t b) { return a + b; }
//
// For non-strict primitives where one or more of the arguments are
// asynchronous, you can use the "Argument" wrapper type as your function
// arguments and return the result by value. The Argument<> internally wraps an
// AsyncValue:
//
//   int32_t add(Argument<int32_t> a, Argument<int32_t> b) { return *a + *b; }
//
// These can be useful if you want to move() the async value itself out of the
// argument instead of just getting access to the value it contains.
//
// All primitives defined with M_PRIMITIVE accept an implicit Chain as their
// last argument. For example, the Add primitive above could be called like
// this:
//
//   %c0 = bef.chain.new
//   %z = "test.add"(%x, %y, %c0) : (i32, i32, !bef.chain) -> i32
//
// For primitives with multiple results, you can return std::pair/tuple as the
// result type:
//
//   // q = (n / d), r = n % d;
//   std::pair<int32_t, int32_t> div_rem(int32_t n, int32_t d) {
//     return {n / d, n % d};
//   }
//
// Or you can use the "Result" wrapper type. Results should appear after
// arguments:
//
//   // q = (n / d), r = n % d;
//   void div_rem(Argument<int32_t> n, Argument<int32_t> d,
//                Result<int32_t> q, Result<int32_t> r) {
//     q.emplace(*n / *d);
//     r.emplace(*n % *d);
//   }
//
// There is also an "Attribute" wrapper for primitives that need to consume
// attribute values. Note that attribute values are essentially literals, so
// these should generally only be used for things that never change for a given
// executable, as opposed to values that will be computed at runtime. Attributes
// should appear after arguments and results:
//
//   int32_t make_int(BEFAttribute<int32_t> value) {
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
//   ErrorOr<std::string> read_file(std::string path) {
//     auto* f = OpenFile(*path);
//     if (!f) {
//       return Error("Could not open file");
//     }
//     <Read file here>
//     return bytes;
//   }
//
// Primitives can also take the PrimitiveFrame or Runtime if they need access
// to anything else the above wrapper types don't provide.
//
// See the definitions of the wrapper types below for more details.
//

// TODO(C++'17): Switch to template when we can use C++17 with 'auto' template
// requirements.
#define M_PRIMITIVE(...)                                                       \
  ::M::Detail::PrimitiveImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::invoke

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
        /*hasInChain=*/false>(frame);
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
  struct CallHelper<PrimitiveFrame &, Tail...> {
    template <int argIdx, int resultIdx, int attrIdx, int funcIdx,
              bool hasInChain, typename... PreviousArgs>
    static void invoke(PrimitiveFrame &frame, PreviousArgs &&...pargs) {
      CallHelper<Tail...>::template invoke<argIdx, resultIdx, attrIdx, funcIdx,
                                           hasInChain, PreviousArgs...,
                                           PrimitiveFrame &>(frame, pargs...,
                                                             frame);
    }
  };

  /// If this requires a Runtime, pass it along.
  template <typename... Tail>
  struct CallHelper<Runtime &, Tail...> {
    template <int argIdx, int resultIdx, int attrIdx, int funcIdx,
              bool hasInChain, typename... PreviousArgs>
    static void invoke(PrimitiveFrame &frame, PreviousArgs &&...pargs) {
      CallHelper<Tail...>::template invoke<argIdx, resultIdx, attrIdx, funcIdx,
                                           hasInChain, PreviousArgs...,
                                           Runtime &>(frame, pargs...,
                                                      frame.getRuntime());
    }
  };

  /// RemainingArguments provides an ArrayRef containing all remaining
  /// arguments. Useful for variadic primitives with heterogenous elements.
  template <typename... Tail>
  struct CallHelper<RemainingArguments, Tail...> {
    template <int argIdx, int resultIdx, int attrIdx, int funcIdx,
              bool hasInChain, typename... PreviousArgs>
    static void invoke(PrimitiveFrame &frame, const PreviousArgs &...pargs) {
      static_assert(
          argIdx != -1,
          "Do not use more than one RemainingArguments/RepeatedArguments");
      static_assert(resultIdx == 0, "Arguments should appear before results.");
      static_assert(attrIdx == 0,
                    "Arguments and results should appear before attributes.");
      static_assert(funcIdx == 0,
                    "Arguments and results should appear before funtions.");
      RemainingArguments args(frame.getArguments().drop_front(argIdx));

      CallHelper<Tail...>::template invoke<-1, resultIdx, attrIdx, funcIdx,
                                           hasInChain>(frame, pargs..., args);
    }
  };

  /// RemainingResults provides an MutableArrayRef<AsyncValue*> containing all
  /// remaining results. Useful for variadic primitives.
  template <typename... Tail>
  struct CallHelper<RemainingResults, Tail...> {
    template <int argIdx, int resultIdx, int attrIdx, int funcIdx,
              bool hasInChain, typename... PreviousArgs>
    static void invoke(PrimitiveFrame &frame, const PreviousArgs &...pargs) {
      static_assert(resultIdx != -1,
                    "Do not use more than one RemainingResults");
      static_assert(attrIdx == 0,
                    "Arguments and results should appear before attributes.");
      static_assert(funcIdx == 0,
                    "Arguments and results should appear before funtions.");

      RemainingResults results(frame.getResults().drop_front(resultIdx));
      CallHelper<Tail...>::template invoke<argIdx, -1, attrIdx, funcIdx,
                                           hasInChain>(frame, pargs...,
                                                       results);
    }
  };

  /// If this takes a `const Function *`, then pass it down.
  template <typename... Tail>
  struct CallHelper<const Function *, Tail...> {
    template <int argIdx, int resultIdx, int attrIdx, int funcIdx,
              bool hasInChain, typename... PreviousArgs>
    static void invoke(PrimitiveFrame &frame, PreviousArgs &&...pargs) {
      CallHelper<Tail...>::template invoke<
          argIdx, resultIdx, attrIdx, funcIdx + 1, hasInChain, PreviousArgs...>(
          frame, pargs..., frame.getFunction(funcIdx));
    }
  };

  // Specialization to cast a single attribute (Head).
  template <typename Head, typename... Tail>
  struct CallHelper<BEFAttribute<Head>, Tail...> {
    template <int argIdx, int resultIdx, int attrIdx, int funcIdx,
              bool hasInChain, typename... PreviousArgs>
    static void invoke(PrimitiveFrame &frame, const PreviousArgs &...pargs) {
      static_assert(attrIdx != -1,
                    "Do not place Attributes after RemainingAttributes");
      BEFAttribute<Head> arg(frame.getRawAttr(attrIdx));
      CallHelper<Tail...>::template invoke<argIdx, resultIdx, attrIdx + 1,
                                           funcIdx, hasInChain>(frame, pargs...,
                                                                arg);
    }
  };

  // Treat any other type as an Argument.
  template <typename Head, typename... Tail>
  struct CallHelper<Head, Tail...> {
    using ArgTy = std::decay_t<Head>;

    template <int argIdx, int resultIdx, int attrIdx, int funcIdx,
              bool hasInChain, typename... PreviousArgs>
    static void invoke(PrimitiveFrame &frame, PreviousArgs &&...pargs) {
      static_assert(argIdx != -1,
                    "Do not place Arguments after RemainingArguments");
      static_assert(resultIdx == 0, "Arguments should appear before results");
      static_assert(attrIdx == 0,
                    "Arguments and results should appear before attributes");
      static_assert(funcIdx == 0,
                    "Arguments and results should appear before functions");
      auto &&arg = frame.getArg<ArgTy>(argIdx);

      constexpr bool isArgChain = std::is_same<ArgTy, Chain>();
      CallHelper<Tail...>::template invoke<argIdx + 1, resultIdx, attrIdx,
                                           funcIdx, (hasInChain || isArgChain)>(
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
        assert((resultIdx == (int)frame.getNumResults() || resultIdx == -1) &&
               "Extra results passed to primitive");
      }
    };

    template <int argIdx, int resultIdx, int attrIdx, int funcIdx,
              bool hasInChain, typename... PreviousArgs>
    static void invoke(PrimitiveFrame &frame, PreviousArgs &&...pargs) {
      AssertIndex<Return, resultIdx>::verify(frame);

      assert((argIdx == (int)frame.getNumArgs() || argIdx == -1 ||
              (!hasInChain && argIdx == frame.getNumArgs() - 1 &&
               frame.getArg(argIdx)->template isType<Chain>())) &&
             "Extra arguments passed to primitive");
      assert((attrIdx == frame.getNumAttrs() || attrIdx == -1) &&
             "Extra attributes passed to primitive");
      assert((funcIdx == frame.getNumFunctions() || funcIdx == -1) &&
             "Extra functions passed to primitive");
      ReturnHelper<Return>::invoke(frame, pargs...);
    }
  };

  //===--------------------------------------------------------------------===//
  // Result handling
  //===--------------------------------------------------------------------===//

  // Handles the return value of the primitive with `handleReturn` if non-void.
  // Otherwise ignores the return value.
  template <typename T>
  struct ReturnHelper {
    static void invoke(PrimitiveFrame &frame, const Args &...args) {
      handleReturn(frame, implFn(args...));
    }
  };
  template <>
  struct ReturnHelper<void> {
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
  static void processResult(PrimitiveFrame &frame, size_t index, Chain t) {
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

} // namespace M

#endif // GRAPHRT_BEFEXECUTOR_PRIMITIVE_REGISTRY_H