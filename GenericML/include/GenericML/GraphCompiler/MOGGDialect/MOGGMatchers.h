//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides MO/MOGG specific Pattern matching indended to be
// used when lowering MO ops to more complex kernels.
//
// This is not a final solution, longer term we would like this infrastructure
// to live in MOJO, so this is intended to be the lightest weight solution
// until such a time that mojo is capable of expressing this kind of
// functionality
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGMATCHERS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGMATCHERS_H

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {

namespace detail {

/// RecursivePatternMatcher that composes.
template <typename... OperandMatchers>
struct UnregisteredRecursivePatternMatcher {
  UnregisteredRecursivePatternMatcher(StringRef opName,
                                      OperandMatchers... matchers)
      : opName(opName), operandMatchers(matchers...) {}
  bool match(Operation *op) {
    if (op->getName().getStringRef() != opName &&
        op->getNumOperands() != sizeof...(OperandMatchers))
      return false;
    bool res = true;
    mlir::detail::enumerate(operandMatchers, [&](size_t idx, auto &matcher) {
      res &= mlir::detail::matchOperandOrValueAtIndex(op, idx, matcher);
    });
    return res;
  }

private:
  StringRef opName;
  std::tuple<OperandMatchers...> operandMatchers;
};

template <typename T, typename Matcher>
struct Capture {
  Capture(T *bindDest, Matcher matcher)
      : bindDest(*bindDest), matcher(matcher) {}
  bool match(T op) {
    if (!matcher.match(op))
      return false;

    bindDest = op;
    return true;
  }

private:
  T &bindDest;
  Matcher matcher;
};

// CTAD Declaration to make the compiler happy
template <typename T, typename Matcher>
Capture(T, Matcher) -> Capture<T, Matcher>;

template <typename U, typename Matcher>
struct StringAttrMatcher {
  StringAttrMatcher(StringRef name, U value, Matcher matcher)
      : name(name), value(value), matcher(matcher) {}
  bool match(Operation *op) {
    return op->hasAttr(name) && op->getAttrOfType<StringAttr>(name) == value &&
           matcher.match(op);
  }

private:
  StringRef name;
  U value;
  Matcher matcher;
};

// CTAD Declaration to make the compiler happy
template <typename U, typename Matcher>
StringAttrMatcher(U, Matcher) -> StringAttrMatcher<U, Matcher>;

template <typename T>
struct MOScalarConstantMatcher {
  MOScalarConstantMatcher(T targetValue) : targetValue(targetValue) {}

  bool match(Value val) {
    auto tensorVal = dyn_cast<M::MO::TensorValue>(val);
    if (!tensorVal)
      return false;

    std::optional<SmallVector<T>> maybeInt =
        M::MO::tryGetConstantOfRank<T>(tensorVal, 0);
    if (maybeInt.has_value())
      return maybeInt.value().front() == targetValue;

    return false;
  }

private:
  T targetValue;
};

// CTAD Declaration to make the compiler happy
template <typename T>
MOScalarConstantMatcher(T) -> MOScalarConstantMatcher<T>;

template <typename Matcher>
struct MORankedValueMatcher {
  MORankedValueMatcher(int64_t targetRank, Matcher matcher)
      : targetRank(targetRank), matcher(matcher) {}
  bool match(Value op) {
    if (auto type = dyn_cast<ShapedType>(op.getType())) {
      return type.hasRank() && type.getRank() == targetRank &&
             matcher.match(op);
    }
    if (auto type = dyn_cast<M::MO::TensorType>(op.getType())) {
      return type.hasRank() &&
             static_cast<int64_t>(type.getRank()) == targetRank &&
             matcher.match(op);
    }
    return false;
  }

private:
  int64_t targetRank;
  Matcher matcher;
};

template <typename T, typename Matcher>
struct TypeMatcher {
  TypeMatcher(Matcher matcher) : matcher(matcher) {}
  bool match(Value op) {
    if (dyn_cast<T>(op.getType()))
      return matcher.match(op);
    return false;
  }

private:
  Matcher matcher;
};

/// Statically switch to a Value matcher.
template <typename MatcherClass>
std::enable_if_t<llvm::is_detected<detail::has_operation_or_value_matcher_t,
                                   MatcherClass, Value>::value,
                 bool>
matchOperandOrValue(Value op, MatcherClass &matcher) {
  return matcher.match(op);
}

/// Statically switch to an Operation matcher.
template <typename MatcherClass>
std::enable_if_t<llvm::is_detected<detail::has_operation_or_value_matcher_t,
                                   MatcherClass, Operation *>::value,
                 bool>
matchOperandOrValue(Value op, MatcherClass &matcher) {
  if (auto *defOp = op.getDefiningOp())
    return matcher.match(defOp);
  return false;
}

template <typename Matcher1, typename Matcher2>
struct OptionalMatcher {
  OptionalMatcher(Matcher1 matcher1, Matcher2 matcher2)
      : matcher1(matcher1), matcher2(matcher2) {}
  bool match(Value op) {
    return matchOperandOrValue(op, matcher1) ||
           matchOperandOrValue(op, matcher2);
  }

private:
  Matcher1 matcher1;
  Matcher2 matcher2;
};

} // namespace detail

template <typename OpType, typename... Matchers>
auto m_Op(Operation **bindDest, Matchers... matchers) {
  return detail::Capture(bindDest, m_Op<OpType, Matchers...>(matchers...));
}

template <typename... Matchers>
auto m_Op(StringRef opName, Matchers... matchers) {
  return detail::UnregisteredRecursivePatternMatcher<Matchers...>(opName,
                                                                  matchers...);
}

template <typename... Matchers>
auto m_Op(Operation **bindDest, StringRef opName, Matchers... matchers) {
  return detail::Capture(bindDest, m_Op(opName, matchers...));
}

template <typename Matcher>
auto m_StringAttr(StringRef attrName, StringRef opName, Matcher matcher) {
  return detail::StringAttrMatcher(attrName, opName, matcher);
}

template <typename... Matchers>
auto m_StringAttr(Operation **bindDest, StringRef attrName, StringRef opName,
                  Matchers... matchers) {
  return detail::Capture(bindDest, m_StringAttr(attrName, opName, matchers...));
}

template <typename Matcher>
auto m_Ranked(int64_t targetRank, Matcher matcher) {
  return detail::MORankedValueMatcher<Matcher>(targetRank, matcher);
}

template <typename T>
auto m_ScalarConstant(T target_value) {
  return detail::MOScalarConstantMatcher<T>(target_value);
}

template <typename T>
auto m_ScalarConstant(mlir::Value *val, T target_value) {
  return detail::Capture(val, m_ScalarConstant(target_value));
}

template <typename T, typename Matcher>
auto m_Typed(Matcher matcher) {
  return detail::TypeMatcher<T, Matcher>(matcher);
}

template <typename T, typename Matcher>
auto m_Typed(Value *val, Matcher matcher) {
  return detail::Capture(val, m_Typed<T, Matcher>(matcher));
}

template <typename Matcher>
auto m_MOVector(Matcher matcher) {
  return m_Ranked(1, m_Typed<M::MO::TensorType, Matcher>(matcher));
}

template <typename Matcher>
auto m_MOMatrix(Matcher matcher) {
  return m_Ranked(2, m_Typed<M::MO::TensorType, Matcher>(matcher));
}

template <typename Matcher1, typename Matcher2>
auto m_Optional(Matcher1 matcher1, Matcher2 matcher2) {
  return detail::OptionalMatcher<Matcher1, Matcher2>(matcher1, matcher2);
}

template <typename OpType, typename Matcher1, typename Matcher2>
auto m_BinaryCommutativeOp(Matcher1 matcher1, Matcher2 matcher2) {
  return m_Optional(m_Op<OpType>(matcher1, matcher2),
                    m_Op<OpType>(matcher2, matcher1));
}

} // namespace mlir

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGMATCHERS_H
