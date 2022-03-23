//===- GraphRT/BEFExecutor/PrimitiveRegistry.h ----------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_BEFEXECUTOR_PRIMITIVE_REGISTRY_H
#define GRAPHRT_BEFEXECUTOR_PRIMITIVE_REGISTRY_H

#include "GraphRT/BEF/BEFEncoding.h"
#include "GraphRT/BEFExecutor/PrimitiveFrame.h"
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
  /// the ADD_PRIMITIVE macro to enable various pretty forms for this.
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
/// types, for example GRT::InvokeOp.
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

/// The Result class is used by primitive implementations for typed asynchronous
/// results.
template <typename T>
class Result {
public:
  /// On construction, the result is initialized with an AsyncValue of the
  /// correct type.
  explicit Result(RCRef<AsyncValue> &result, Runtime &runtime)
      : result(result) {
    assert(!result && "result already initialized?");
    result = AsyncValue::allocate<T>(runtime);
  }
  Result(const Result &&other) : result(other.result) {}

  /// Return an owning reference to the AsyncValueRef.  This is typically used
  /// when copying into a closure.
  AsyncValueRef<T> copy() const { return AsyncValueRef<T>(result.copy()); }

  /// Return a reference to the underlying element in the AsyncValue if it is
  /// constructed.
  T &get() const { return result->get<T>(); }

  /// Construct the result, but do not mark it ready.
  template <typename... Args>
  void construct(Args &&...args) {
    result->construct<T, Args...>(std::forward<Args>(args)...);
  }

  /// Construct the result in place and mark it as ready.
  template <typename... Args>
  void emplace(Args &&...args) {
    result->emplace<T, Args...>(std::forward<Args>(args)...);
  }

  /// Return a raw pointer to the AsyncValue.
  AsyncValue *getPointer() const { return result.getPointer(); }
  /// Return a reference to the low-level async value without copying it.
  RCRef<AsyncValue> &getAsyncValue() { return result; }

  /// Allow dereferencing this to get access to setError().
  AsyncValue &operator*() const { return *getPointer(); }
  AsyncValue *operator->() const { return getPointer(); }

private:
  Result(const Result &) = delete;
  void operator=(const Result &) = delete;
  RCRef<AsyncValue> &result;
};

/// The IndirectResult class is used for primitives that need a result but don't
/// know its static type.  It will be initialized to an AsyncValue that is in
/// indirect state, allowing it to be late-bound, e.g. asynchronously.
class IndirectResult {
public:
  /// On construction, this is initialized with an indirect AsyncValue.
  explicit IndirectResult(RCRef<AsyncValue> &result, Runtime &runtime)
      : result(result) {
    assert(!result && "result already initialized?");
    result = AsyncValue::createIndirect(runtime);
  }
  IndirectResult(const IndirectResult &&other) : result(other.result) {}

  /// Return an owning reference to the AsyncValueRef.  This is typically used
  /// when copying into a closure.
  RCRef<AsyncValue> copy() const { return result.copy(); }

  /// Resolve an IndirectAsyncValue to point to the specified new value,
  /// resolving any waiters whenever newValue becomes ready.
  void resolve(RCRef<AsyncValue> newValue) {
    result->resolveIndirect(std::move(newValue));
  }

  /// Construct an AsyncValue for the result and resolve the indirect result.
  template <typename T, typename... Args>
  void emplace(Args &&...args) {
    resolve(AsyncValue::createReady<T, Args...>(result->getRuntime(),
                                                std::forward<Args>(args)...));
  }

  /// Return a raw pointer to the AsyncValue.
  AsyncValue *getPointer() const { return result.getPointer(); }
  /// Return a reference to the low-level async value without copying it.
  RCRef<AsyncValue> &getAsyncValue() { return result; }

  /// Allow dereferencing this to get access to setError().
  AsyncValue &operator*() const { return *getPointer(); }
  AsyncValue *operator->() const { return getPointer(); }

private:
  IndirectResult(const IndirectResult &) = delete;
  void operator=(const IndirectResult &) = delete;
  RCRef<AsyncValue> &result;
};

/// RemainingResults collects all remaining results in a MutableArrayRef. There
/// can be at most one RemainingResults, and it must appear after all other
/// Results.
///
/// This should be used when a primitive has potentially heterogenous types, for
/// example, GRT::InvokeOp.
class RemainingResults {
public:
  explicit RemainingResults(MutableArrayRef<RCRef<AsyncValue>> results,
                            Runtime &runtime)
      : runtime(runtime), results(results) {}

  MutableArrayRef<RCRef<AsyncValue>> get() const { return results; }
  size_t size() const { return results.size(); }
  RCRef<AsyncValue> &operator[](size_t i) const { return results[i]; }

  /// Allocate all of the unset result values as IndirectAsyncValues.
  void allocateIndirectResults() {
    for (auto &elt : results) {
      if (!elt)
        elt = AsyncValue::createIndirect(runtime);
    }
  }

private:
  Runtime &runtime;
  MutableArrayRef<RCRef<AsyncValue>> results;
};

/// This CRTP class is used to provide common interfaces across the various
/// BEFAttr implementations.
template <typename ConcreteType, typename StoredType>
class BEFAttrCommon {
public:
  BEFAttrCommon(StoredType init) : value(init) {}

  using ValueType = typename std::remove_reference<StoredType>::type;

  const StoredType &get() const { return value; }
  const ValueType *operator->() const { return &value; }
  const StoredType &operator*() const { return value; }

  /// Given a pointer to the start of attribute value, return the attribute
  /// kind.
  static BEF::AttributeKind getAttrKind(const void *value) {
    // The attribute kind is encoded in the byte before the attribute value.
    return static_cast<BEF::AttributeKind>(
        *(static_cast<const uint8_t *>(value) - 1));
  }

  /// Check that the dynamic attribute kind encoded into the BEF file matches
  /// the C++ type for this.
  static inline void checkAttributeKind(const void *value) {
    assert(ConcreteType::matchesAttrKind(value) &&
           "attribute passed to primitive has unexpected order;"
           " mismatch between mlir syntax and primitive definition");
  }

private:
  StoredType value;
};

/// Primitives use this to declare that they have an attribute input with the
/// specified C++ type.
template <typename T>
class BEFAttr : public BEFAttrCommon<BEFAttr<T>, const T &> {
public:
  using BaseClass = BEFAttrCommon<BEFAttr<T>, const T &>;

  explicit BEFAttr(const void *value)
      : BaseClass(*reinterpret_cast<const T *>(value)) {
    // Check that the C++ type matches the encoded type for the attribute.
    static_assert(BEF::AttrKindForCXXType<T>::kind !=
                      BEF::AttributeKind::kInvalid &&
                  "C++ type passed to BEFAttr isn't supported");
    this->checkAttributeKind(value);
  }

  static bool matchesAttrKind(const void *pointer) {
    return BaseClass::getAttrKind(pointer) == BEF::AttrKindForCXXType<T>::kind;
  }
};

/// String attributes have a special encoding that includes their length,
/// allowing projection into a StringRef.
template <>
class BEFAttr<StringRef> : public BEFAttrCommon<BEFAttr<StringRef>, StringRef> {
public:
  using BaseClass = BEFAttrCommon<BEFAttr<StringRef>, StringRef>;

  explicit BEFAttr(const void *pointer)
      : BaseClass(StringRef(static_cast<const char *>(pointer),
                            decodeBEFArrayLength(pointer))) {
    this->checkAttributeKind(pointer);
  }

  static bool matchesAttrKind(const void *pointer) {
    return BaseClass::getAttrKind(pointer) == BEF::AttributeKind::kString;
  }
};

/// Array attributes have a special encoding that includes their length,
/// allowing projection into a ArrayRef.
template <typename T>
class BEFAttr<ArrayRef<T>>
    : public BEFAttrCommon<BEFAttr<ArrayRef<T>>, ArrayRef<T>> {
public:
  using BaseClass = BEFAttrCommon<BEFAttr<ArrayRef<T>>, ArrayRef<T>>;

  explicit BEFAttr(const void *pointer)
      : BaseClass(ArrayRef<T>(static_cast<const T *>(pointer),
                              decodeBEFArrayLength(pointer))) {
    this->checkAttributeKind(pointer);
  }

  static bool matchesAttrKind(const void *pointer) {
    // If the element type matches, this matches.
    BEF::AttributeKind kind = BaseClass::getAttrKind(pointer);
    if (kind == BEF::getArrayOf(BEF::AttrKindForCXXType<T>::kind))
      return true;
    // If the element type is invalid and the length is zero this matches.
    // This is because MLIR2BEF doesn't know the element type for empty arrays.
    if (kind == BEF::getArrayOf(BEF::AttributeKind::kInvalid) &&
        decodeBEFArrayLength(pointer) == 0)
      return true;
    return false;
  }

  size_t size() const { return this->get().size(); }
  const T &operator[](size_t i) const { return this->get()[i]; }
};

namespace Detail {
class Dynamic {};
} // namespace Detail

/// Dynamic attributes allow checked casting to other attribute kinds.
template <>
class BEFAttr<Detail::Dynamic>
    : public BEFAttrCommon<BEFAttr<Detail::Dynamic>, const void *> {
public:
  using BaseClass = BEFAttrCommon<BEFAttr<Detail::Dynamic>, const void *>;

  explicit BEFAttr(const void *pointer) : BaseClass(pointer) {}

  /// Return the AttributeKind for this attribute.
  BEF::AttributeKind getAttrKind() const {
    return BEFAttrCommon::getAttrKind(get());
  }

  /// Do a checked conversion to the specified attribute kind.
  template <typename T>
  Optional<BEFAttr<T>> dyn_cast() const {
    // Test to see if this has the right attribute kind.
    if (!BEFAttr<T>::matchesAttrKind(get()))
      return {};
    return BEFAttr<T>(get());
  }

  template <typename T>
  BEFAttr<T> cast() const {
    auto result = this->dyn_cast<T>();
    assert(result && "dynamic attribute is not of this kind");
    return *result;
  }

  /// Dynamic attributes are compatible with everything.
  static bool matchesAttrKind(const void *pointer) { return true; }
};

using BEFDynamicAttr = BEFAttr<Detail::Dynamic>;

//===----------------------------------------------------------------------===//
// Primitive Registration Wrapper Generator
//===----------------------------------------------------------------------===//

/// This macro is used to register an implementation function for a primitive
/// with a specified PrimitiveRegistry.  It handles generating a wrapper thunk
/// that makes it much easier (and less bug prone) to work with low-level
/// `PrimitiveFrame`.
///
/// For example:
///    ADD_PRIMITIVE(registery, "grt.invoke", grt_invoke_impl);
///
#define ADD_PRIMITIVE(registry, name, ...)                                     \
  do {                                                                         \
    registry.add(name, ::M::Detail::PrimitiveImpl<decltype(&__VA_ARGS__),      \
                                                  &__VA_ARGS__>::invoke);      \
    ::M::Detail::PrimitiveImpl<decltype(&__VA_ARGS__),                         \
                               &__VA_ARGS__>::registerTypes();                 \
  } while (0)

namespace Detail {
// This class implements the ADD_PRIMITIVE macro.
template <typename FnType, FnType implFn>
struct PrimitiveImpl;

template <typename Return, typename... Args, Return (*implFn)(Args...)>
struct PrimitiveImpl<Return (*)(Args...), implFn> {
  //===--------------------------------------------------------------------===//
  // Type Registration
  //===--------------------------------------------------------------------===//

  /// This class defines a helper for registering a type to enable use within
  /// AsyncValue.
  ///
  /// Note: The `Dummy` type is used to avoid GCC errors for defining struct
  /// specializations outside of a namespace.
  template <typename T, typename Dummy = void>
  struct TypeRegistration {
    static void registerType() { AsyncValue::registerType<T>(); }
  };
  template <typename Dummy>
  struct TypeRegistration<void, Dummy> {
    static void registerType() {}
  };

  /// A utility registration base class that recursively invokes registration on
  /// the given set of types.
  template <typename... Ts>
  struct ForwardingTypeRegistration {
    static void registerType() {
      // FIXME-C++17: Use a proper fold expression here.
      (void)std::initializer_list<int>{
          0, (TypeRegistration<Ts>::registerType(), 0)...};
    }
  };
  template <typename T1, typename T2>
  struct TypeRegistration<std::pair<T1, T2>>
      : public ForwardingTypeRegistration<T1, T2> {};
  template <typename... Ts>
  struct TypeRegistration<std::tuple<Ts...>>
      : public ForwardingTypeRegistration<Ts...> {};
  template <typename T>
  struct TypeRegistration<AsyncValueRef<T>>
      : public ForwardingTypeRegistration<T> {};
  template <typename T>
  struct TypeRegistration<ErrorOr<T>> : public ForwardingTypeRegistration<T> {};

  //===--------------------------------------------------------------------===//
  // Entry Point

  /// This method is an entry point used to register the necessary types with
  /// AsyncValue needed to properly invoke this primitive.
  static void registerTypes() {
    // We only need to register result types here, given that the argument types
    // require a valid producer.
    TypeRegistration<std::decay_t<Return>>::registerType();
  }

  //===--------------------------------------------------------------------===//
  // Invocation
  //===--------------------------------------------------------------------===//

  /// This is the main entry point that gets registered as an impl.  This works
  /// by recursively processing the argument list, then handling the result
  /// list, generating a function with the standard PrimitiveImpl signature that
  /// handles each part of the function signature as it goes.
  static void invoke(PrimitiveFrame &frame) {
    processArgsAndInvoke<
        ValueIndices</*argIdx=*/0, /*resultIdx=*/0, /*attrIdx=*/0,
                     /*invIdx=*/0>,
        /*hasInChain=*/false>(TypeTag<Args...>(), frame);
  }

  //===--------------------------------------------------------------------===//
  // Argument handling
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Argument handling: Value Type Definitions

  /// This enum represents the different types of values that may be passed as
  /// arguments during the invocation of a primitive.
  enum class ValueType {
    kArgument,
    kResult,
    kAttribute,
    kInvokable,
    kFramework,
  };

  /// This class represents a static container for the various indices used
  /// during argument handling.
  template <int argIdxVal, int resultIdxVal, int attrIdxVal, int invIdxVal>
  struct ValueIndices {
    /// A set of indices that correspond to lists of various value types. When
    /// each list has been fully consumed, the index is set to -1.
    static constexpr int argIdx = argIdxVal;
    static constexpr int resultIdx = resultIdxVal;
    static constexpr int attrIdx = attrIdxVal;
    static constexpr int invIdx = invIdxVal;
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
                     updateIndex<ValueType::kInvokable, IndicesT::invIdx>()>;

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
      static_assert(IndicesT::invIdx == 0,
                    "Arguments and results should appear before invokables");
    }
  };
  /// This class defines an Argument value that terminates the argument list,
  /// and consumes the remaining arguments.
  struct TerminalArgValue : public ArgValue</*isTerminal=*/true> {};

  //===--------------------------------------------------------------------===//
  // Result Values

  /// This class represents a Result value.
  template <bool isTerminal = false>
  struct ResultValue : public ValueBase<ValueType::kResult, isTerminal> {
    template <typename IndicesT>
    static constexpr void verifyCurrentIndex() {
      static_assert(isTerminal || IndicesT::resultIdx != -1,
                    "Do not place Results after RemainingResults");
      static_assert(!isTerminal || IndicesT::resultIdx != -1,
                    "Do not use more than one RemainingResults");
      static_assert(IndicesT::attrIdx == 0,
                    "Arguments and results should appear before attributes");
      static_assert(IndicesT::invIdx == 0,
                    "Arguments and results should appear before invokables");
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
  // Invokable Values

  /// This class defines a Invokable value.
  struct InvokableValue : public ValueBase<ValueType::kInvokable> {};

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

  /// Result<T> provides an initialized AsyncValue with a specific type.
  template <typename T, typename Dummy>
  struct Value<Result<T>, Dummy> : public ResultValue<> {
    template <typename IndicesT>
    static Result<T> process(PrimitiveFrame &frame) {
      return Result<T>(frame.getResults()[IndicesT::resultIdx],
                       frame.getRuntime());
    }
  };

  /// IndirectResult provides an initialized indirect AsyncValue.
  template <typename Dummy>
  struct Value<IndirectResult, Dummy> : public ResultValue<> {
    template <typename IndicesT>
    static IndirectResult process(PrimitiveFrame &frame) {
      return IndirectResult(frame.getResults()[IndicesT::resultIdx],
                            frame.getRuntime());
    }
  };

  /// RemainingResults provides a view containing all remaining results.
  /// Useful for variadic primitives.
  template <typename Dummy>
  struct Value<RemainingResults, Dummy> : public TerminalResultValue {
    template <typename IndicesT>
    static RemainingResults process(PrimitiveFrame &frame) {
      return RemainingResults(
          frame.getResults().drop_front(IndicesT::resultIdx),
          frame.getRuntime());
    }
  };

  //===--------------------------------------------------------------------===//
  // Attribute Values

  /// Specialization to cast a single attribute (T).
  template <typename T, typename Dummy>
  struct Value<BEFAttr<T>, Dummy> : public AttrValue {
    template <typename IndicesT>
    static BEFAttr<T> process(PrimitiveFrame &frame) {
      return BEFAttr<T>(frame.getRawAttr(IndicesT::attrIdx));
    }
  };

  //===--------------------------------------------------------------------===//
  // Invokable Values

  /// If this takes a `const Invokable *`, then pass it down.
  template <typename Dummy>
  struct Value<const Invokable *, Dummy> : public InvokableValue {
    template <typename IndicesT>
    static const Invokable *process(PrimitiveFrame &frame) {
      return frame.getInvokable(IndicesT::invIdx);
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
    assert((IndicesT::invIdx == frame.getNumInvokables() ||
            IndicesT::invIdx == -1) &&
           "Extra invokables passed to primitive");
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
    frame.setResult(index, frame.getRuntime().getReadyChain().copy());
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

  /// Store the primitive result back to the output AsyncValue in the
  /// PrimitiveFrame.
  template <typename T>
  static void handleReturn(PrimitiveFrame &frame, T &&t) {
    assert(frame.getNumResults() == 1 && "Incorrect number of results");
    processResult(frame, 0, std::forward<T>(t));
  }

  /// Support implementations that return std::pair<>, storing the two results
  /// in separate output results.
  template <typename T1, typename T2>
  static void handleReturn(PrimitiveFrame &frame, std::pair<T1, T2> &&pair) {
    assert(frame.getNumResults() == 2 && "Incorrect number of results");
    processResult(frame, 0, std::move(pair.first));
    processResult(frame, 1, std::move(pair.second));
  }

  /// Support implementations that return std::tuple<>, storing each element in
  /// separate output results.
  template <typename... Ts>
  static void handleReturn(PrimitiveFrame &frame, std::tuple<Ts...> &&tuple) {
    assert(frame.getNumResults() == sizeof...(Ts) &&
           "Incorrect number of results");

    auto applyFn = [&](auto &&...args) {
      // Process each result in-order.
      // Note: In C++17, this can be updated to use proper fold expressions, and
      // remove the initializer_list.
      int resultIdx = 0;
      (void)std::initializer_list<int>{
          0, (processResult(frame, resultIdx++, std::move(args)), 0)...};
    };
    llvm::apply_tuple(applyFn, std::move(tuple));
  }
};
} // end namespace Detail

} // namespace M

#endif // GRAPHRT_BEFEXECUTOR_PRIMITIVE_REGISTRY_H