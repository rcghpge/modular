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

// TODO(1044): Refactor to support more types of constants and
// make this more composable.
struct MOScalarConstantIntMatcher {
  MOScalarConstantIntMatcher(int64_t targetValue) : targetValue(targetValue) {}

  bool match(Value val);

private:
  int64_t targetValue;
};

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

detail::Capture<Value, detail::MOScalarConstantIntMatcher>
m_ScalarConstantInt(Value *val, int64_t targetValue);

detail::MOScalarConstantIntMatcher m_ScalarConstantInt(int64_t targetValue);

template <typename T, typename Matcher>
auto m_Typed(Matcher matcher) {
  return detail::TypeMatcher<T, Matcher>(matcher);
}

template <typename T, typename Matcher>
auto m_Typed(Value *val, Matcher matcher) {
  return detail::Capture(val, m_Typed<T, Matcher>(matcher));
}

template <typename T>
auto m_MOMatrix(T *input) {
  return m_Ranked(2,
                  m_Typed<M::MO::TensorType>(input, mlir::matchers::m_Any()));
}

} // namespace mlir

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGMATCHERS_H
