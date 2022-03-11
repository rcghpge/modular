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
#include "llvm/ADT/StringRef.h"

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

/// Given a pointer to the start of a BEF array or string attribute, return the
/// length of array or string by decoding the prefix encoded VBR.
size_t decodeBEFArrayLength(const void *dataPtr);

// Utility template for tag dispatching.
template <typename... T>
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

  /// Allocate all of the unset result values as IndirectAsyncValues.
  void allocateIndirectResults(Runtime &runtime) {
    for (auto &elt : results) {
      if (!elt)
        elt = AsyncValue::createIndirect(runtime);
    }
  }

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

/// String attributes have a special encoding that includes their length,
/// allowing projection into a StringRef.
template <>
class BEFAttribute<StringRef> {
public:
  explicit BEFAttribute(const void *value)
      : value(static_cast<const char *>(value), decodeBEFArrayLength(value)) {}

  const StringRef &get() const { return value; }
  const StringRef *operator->() const { return &value; }
  const StringRef &operator*() const { return value; }

private:
  StringRef value;
};

/// Array attributes have a special encoding that includes their length,
/// allowing projection into a ArrayRef.
template <typename T>
class BEFAttribute<ArrayRef<T>> {
public:
  explicit BEFAttribute(const void *value)
      : value(static_cast<const T *>(value), decodeBEFArrayLength(value)) {}

  const ArrayRef<T> &get() const { return value; }
  const ArrayRef<T> *operator->() const { return &value; }
  const ArrayRef<T> &operator*() const { return value; }

  size_t size() const { return value.size(); }
  const T &operator[](size_t i) const { return value[i]; }

private:
  ArrayRef<T> value;
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
// For primitives where you want low-level access to AsyncValue's, you can
// declare them with either AsyncValueRef (if you know the element type) or with
// RCRef<AsyncValue> (if you are type generic):
//
//   int32_t add(AsyncValueRef<int32_t> a, RCRef<AsyncValue> b) {
//     return a.get() + b.get<int32_t>();
//   }
//
// These can be useful if you want to move() the async value itself out of the
// argument instead of just getting access to the value it contains.
//
// All primitives defined with M_PRIMITIVE accept an implicit Chain as their
// last argument. For example, the `add` primitive above could be called like
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
//   void div_rem(int32_t n, int32_t d,
//                Result<int32_t> q, Result<int32_t> r) {
//     q.emplace(n / d);
//     r.emplace(n % d);
//   }
//
// There is also a "BEFAttribute" wrapper for primitives that need to consume
// attribute values.  Attribute values are essentially literals - they are used
// for things that never change for a given BEF executable, as opposed to values
// that will be computed at runtime. Attributes must be listed after arguments
// and results:
//
//   int32_t make_int(BEFAttribute<int32_t> value) {
//     return *value;
//   }
//
// BEF supports character strings and arrays, which you can access with
// StringRef and ArrayRef in BEFAttribute:
//
//   int32_t sum_ints(BEFAttribute<ArrayRef<int32_t>> value) {
//     return std::accumulate(value->begin(), value->end(), 0);
//   }
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
// You can also return RCRef<AsyncValue> and AsyncValueRef<T> values, which
// allow lazy completion of values and lazy production of error results.
//
// Primitives can also take the PrimitiveFrame or Runtime if they need access
// to anything else the above wrapper types don't provide.
//
// See the definitions of the wrapper types above for more details.
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
    processArgsAndInvoke<
        ValueIndices</*argIdx=*/0, /*resultIdx=*/0, /*attrIdx=*/0,
                     /*funcIdx=*/0>,
        /*hasInChain=*/false>(TypeTag<Args...>(), frame);
  }

  //===--------------------------------------------------------------------===//
  // Argument handling
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Argument handling: Value Type Definitions
  //===--------------------------------------------------------------------===//

  /// This enum represents the different types of values that may be passed as
  /// arguments during the invocation of a primitive.
  enum class ValueType {
    kArgument,
    kResult,
    kAttribute,
    kFunction,
    kFramework,
  };

  /// This class represents a static container for the various indices used
  /// during argument handling.
  template <int argIdxVal, int resultIdxVal, int attrIdxVal, int funcIdxVal>
  struct ValueIndices {
    /// A set of indices that correspond to lists of various value types. When
    /// each list has been fully consumed, the index is set to -1.
    static constexpr int argIdx = argIdxVal;
    static constexpr int resultIdx = resultIdxVal;
    static constexpr int attrIdx = attrIdxVal;
    static constexpr int funcIdx = funcIdxVal;
  };

  /// This class provides a common static abstraction of the different value
  /// types.
  template <ValueType valueTypeParam, bool valueIsTerminal = false>
  struct ValueBase {
    /// A flag indicating the type of the value.
    static constexpr ValueType valueType = valueTypeParam;

    /// A flag indicating whether the value is a terminal value, i.e. it
    /// consumes the remaining values of this type.
    static constexpr bool isTerminal = valueIsTerminal;

    /// A flag indicating whether the value contains a chain.
    static constexpr bool isChain = false;

    /// Updates the given index if the type matches the one for this value,
    /// otherwise returns the current index.
    template <ValueType type, int idx>
    static constexpr int updateIndex() {
      return (valueType == type) ? (isTerminal ? -1 : (idx + 1)) : idx;
    }
    /// Updates the given indices for the current value.
    template <typename IndicesT>
    using UpdatedIndicesT =
        ValueIndices<updateIndex<ValueType::kArgument, IndicesT::argIdx>(),
                     updateIndex<ValueType::kResult, IndicesT::resultIdx>(),
                     updateIndex<ValueType::kAttribute, IndicesT::attrIdx>(),
                     updateIndex<ValueType::kFunction, IndicesT::funcIdx>()>;

    /// Verify the given value indices based on the current value type. The
    /// default implementation assumes all are valid, but derived value types
    /// may override to provide additional checks and guarantees.
    template <typename IndicesT>
    static constexpr void verifyCurrentIndex() {}
  };

  //===--------------------------------------------------------------------===//
  // Argument Values

  /// This class defines an Argument value.
  template <bool isTerminal = false>
  struct ArgValue : public ValueBase<ValueType::kArgument, isTerminal> {
    template <typename IndicesT>
    static constexpr void verifyCurrentIndex() {
      static_assert(isTerminal || IndicesT::argIdx != -1,
                    "Do not place Arguments after RemainingArguments");
      static_assert(
          !isTerminal || IndicesT::argIdx != -1,
          "Do not use more than one RemainingArguments/RepeatedArguments");

      static_assert(IndicesT::resultIdx == 0,
                    "Arguments should appear before results");
      static_assert(IndicesT::attrIdx == 0,
                    "Arguments and results should appear before attributes");
      static_assert(IndicesT::funcIdx == 0,
                    "Arguments and results should appear before functions");
    }
  };
  /// This class defines an Argument value that terminates the argument list,
  /// and consumes the remaining arguments.
  struct TerminalArgValue : public ArgValue</*isTerminal=*/true> {};

  //===--------------------------------------------------------------------===//
  // Result Values

  /// This class represents a Result value.
  template <bool isTerminal>
  struct ResultValue : public ValueBase<ValueType::kResult, isTerminal> {
    template <typename IndicesT>
    static constexpr void verifyCurrentIndex() {
      static_assert(isTerminal || IndicesT::resultIdx != -1,
                    "Do not place Results after RemainingResults");
      static_assert(!isTerminal || IndicesT::resultIdx != -1,
                    "Do not use more than one RemainingResults");
      static_assert(IndicesT::attrIdx == 0,
                    "Arguments and results should appear before attributes.");
      static_assert(IndicesT::funcIdx == 0,
                    "Arguments and results should appear before funtions.");
    }
  };
  /// This class defines a Result value that terminates the result list,
  /// and consumes the remaining results.
  struct TerminalResultValue : public ResultValue</*isTerminal=*/true> {};

  //===--------------------------------------------------------------------===//
  // Attribute Values

  /// This class defines an Attribute value.
  struct AttrValue : public ValueBase<ValueType::kAttribute> {
    template <typename IndicesT>
    static void verifyCurrentIndex() {
      static_assert(IndicesT::attrIdx != -1,
                    "Do not place Attributes after RemainingAttributes");
    }
  };

  //===--------------------------------------------------------------------===//
  // Function Values

  /// This class defines a Function value.
  struct FunctionValue : public ValueBase<ValueType::kFunction> {};

  //===--------------------------------------------------------------------===//
  // Framework Values

  /// This class defines a Framework value, such as a PrimitiveFrame or
  /// Runtime.
  struct FrameworkValue : public ValueBase<ValueType::kFramework> {};

  //===--------------------------------------------------------------------===//
  // Argument handling: Value Definitions
  //===--------------------------------------------------------------------===//

  /// This class defines a concrete type of value, with `T` representing the
  /// expected C++ class. Each specialization should derive from one of the
  /// specific value types above, and also provide a `process` function that
  /// computes the concrete value. This function should generally be of (or
  /// similar to) the following form:
  ///
  ///   template <typename IndicesT>
  ///   static T process(PrimitiveFrame &frame) {
  ///     // Compute and return the value of `T` here.
  ///     // `IndicesT` is an instance of ValueIndices containing the current
  ///     // indices for each value type.
  ///   }
  ///
  /// Note: The `Dummy` type is used to avoid GCC errors for defining struct
  /// specializations outside of a namespace.
  template <typename T, typename Dummy = void>
  struct Value;

  //===--------------------------------------------------------------------===//
  // Argument Values

  /// By default, treat any unknown type as an Argument.
  template <typename T, typename Dummy>
  struct Value : public ArgValue<> {
    using ArgTy = std::decay_t<T>;

    template <typename IndicesT>
    static auto &&process(PrimitiveFrame &frame) {
      return frame.getArg<ArgTy>(IndicesT::argIdx);
    }

    /// Mark if the value is a chain.
    static constexpr bool isChain = std::is_same<ArgTy, Chain>::value;
  };

  /// If this wants an AsyncValueRef, then they want raw low-level access to an
  /// argument, provide it.
  template <typename T, typename Dummy>
  struct Value<AsyncValueRef<T>, Dummy> : public ArgValue<> {
    template <typename IndicesT>
    static AsyncValueRef<T> process(PrimitiveFrame &frame) {
      return std::move(frame.getArg(IndicesT::argIdx));
    }

    /// Mark if the value is a chain.
    static constexpr bool isChain = std::is_same<T, Chain>::value;
  };

  /// If this wants an AsyncValueRef, then they want raw low-level access to an
  /// argument, provide it.
  template <typename Dummy>
  struct Value<RCRef<AsyncValue>, Dummy> : public ArgValue<> {
    template <typename IndicesT>
    static RCRef<AsyncValue> process(PrimitiveFrame &frame) {
      return std::move(frame.getArg(IndicesT::argIdx));
    }
  };

  /// RemainingArguments provides an ArrayRef containing all remaining
  /// arguments. Useful for variadic primitives with heterogenous elements.
  template <typename Dummy>
  struct Value<RemainingArguments, Dummy> : public TerminalArgValue {
    template <typename IndicesT>
    static RemainingArguments process(PrimitiveFrame &frame) {
      return RemainingArguments(
          frame.getArguments().drop_front(IndicesT::argIdx));
    }
  };

  //===--------------------------------------------------------------------===//
  // Result Values

  /// RemainingResults provides a view containing all remaining results. Useful
  /// for variadic primitives.
  template <typename Dummy>
  struct Value<RemainingResults, Dummy> : public TerminalResultValue {
    template <typename IndicesT>
    static RemainingResults process(PrimitiveFrame &frame) {
      return RemainingResults(
          frame.getResults().drop_front(IndicesT::resultIdx));
    }
  };

  //===--------------------------------------------------------------------===//
  // Attribute Values

  /// Specialization to cast a single attribute (T).
  template <typename T, typename Dummy>
  struct Value<BEFAttribute<T>, Dummy> : public AttrValue {
    template <typename IndicesT>
    static BEFAttribute<T> process(PrimitiveFrame &frame) {
      return BEFAttribute<T>(frame.getRawAttr(IndicesT::attrIdx));
    }
  };

  //===--------------------------------------------------------------------===//
  // Function Values

  /// If this takes a `const Function *`, then pass it down.
  template <typename Dummy>
  struct Value<const Function *, Dummy> : public FunctionValue {
    template <typename IndicesT>
    static const Function *process(PrimitiveFrame &frame) {
      return frame.getFunction(IndicesT::funcIdx);
    }
  };

  //===--------------------------------------------------------------------===//
  // Framework Values

  /// If this requires the PrimitiveFrame for any reason, pass it along.
  template <typename Dummy>
  struct Value<PrimitiveFrame &, Dummy> : public FrameworkValue {
    template <typename IndicesT>
    static PrimitiveFrame &process(PrimitiveFrame &frame) {
      return frame;
    }
  };

  /// If this requires a Runtime, pass it along.
  template <typename Dummy>
  struct Value<Runtime &, Dummy> : public FrameworkValue {
    template <typename IndicesT>
    static Runtime &process(PrimitiveFrame &frame) {
      return frame.getRuntime();
    }
  };

  //===--------------------------------------------------------------------===//
  // Argument handling: Processing
  //===--------------------------------------------------------------------===//

  /// This function handles the nitty gritty processing of the arguments. We
  /// recursively unwrap each argument, verify the current value indices,
  /// process the argument, and append it to the working argument list.
  template <typename IndicesT, bool hasInChain, typename ArgT,
            typename... ArgsToProcessT, typename... ArgsT>
  static void processArgsAndInvoke(TypeTag<ArgT, ArgsToProcessT...>,
                                   PrimitiveFrame &frame, ArgsT &&...args) {
    Value<ArgT, void>::template verifyCurrentIndex<IndicesT>();
    processArgsAndInvoke<
        typename Value<ArgT, void>::template UpdatedIndicesT<IndicesT>,
        (hasInChain || Value<ArgT, void>::isChain)>(
        TypeTag<ArgsToProcessT...>(), frame, std::forward<ArgsT>(args)...,
        Value<ArgT, void>::template process<IndicesT>(frame));
  }

  /// This function is the base case for argument processing function above. It
  /// is invoked after all arguments have finished processing, and handles some
  /// final validation before invoking the primitive.
  template <typename IndicesT, bool hasInChain, typename... ArgsT>
  static void processArgsAndInvoke(TypeTag<>, PrimitiveFrame &frame,
                                   ArgsT &&...args) {
    verifyResultIndex<IndicesT::resultIdx>(frame, TypeTag<Return>());

    assert((IndicesT::argIdx == (int)frame.getNumArgs() ||
            IndicesT::argIdx == -1 ||
            (!hasInChain &&
             static_cast<size_t>(IndicesT::argIdx) == frame.getNumArgs() - 1 &&
             frame.getArg(IndicesT::argIdx)->template isType<Chain>())) &&
           "Extra arguments passed to primitive");
    assert(
        (IndicesT::attrIdx == frame.getNumAttrs() || IndicesT::attrIdx == -1) &&
        "Extra attributes passed to primitive");
    assert((IndicesT::funcIdx == frame.getNumFunctions() ||
            IndicesT::funcIdx == -1) &&
           "Extra functions passed to primitive");
    invoke<Return>(frame, std::forward<ArgsT>(args)...);
  }

  // Verify the result index for non-void Return type
  template <int resultIdx, typename T>
  static void verifyResultIndex(PrimitiveFrame &frame, TypeTag<T>) {
    static_assert(resultIdx == 0,
                  "Don't mix return by value and Result<> syntax.");
  }

  // Verify the result index for void Return type.
  template <int resultIdx>
  static void verifyResultIndex(PrimitiveFrame &frame, TypeTag<void>) {
    assert((resultIdx == (int)frame.getNumResults() || resultIdx == -1) &&
           "Extra results passed to primitive");
  }

  //===--------------------------------------------------------------------===//
  // Result handling
  //===--------------------------------------------------------------------===//

  // Handles the return value of the primitive with `handleReturn` if
  // non-void. Otherwise ignores the return value.
  template <typename ReturnT, typename... ActualArgs>
  static std::enable_if_t<!std::is_same<ReturnT, void>::value>
  invoke(PrimitiveFrame &frame, ActualArgs &&...args) {
    handleReturn(frame, implFn(std::forward<ActualArgs>(args)...));
  }
  template <typename ReturnT, typename... ActualArgs>
  static std::enable_if_t<std::is_same<ReturnT, void>::value>
  invoke(PrimitiveFrame &frame, ActualArgs &&...args) {
    implFn(std::forward<ActualArgs>(args)...);
  }

  // These functions provide custom behavior for concrete result types that
  // may come up.

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

  /// Don't allocate an async value for chain results, just use the one
  /// already stored in Runtime.
  static void processResult(PrimitiveFrame &frame, size_t index, Chain t) {
    frame.setResult(index, frame.getRuntime().getReadyChain());
  }

  /// Handle explicit RCRef<AsyncValue> results.
  static void processResult(PrimitiveFrame &frame, size_t index,
                            RCRef<AsyncValue> t) {
    frame.setResult(index, std::move(t));
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