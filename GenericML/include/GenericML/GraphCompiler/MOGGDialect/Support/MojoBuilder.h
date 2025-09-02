//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Mojo text emission helpers.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_SUPPORT_MOJOEMITTER_H
#define GENERICML_GRAPHCOMPILER_SUPPORT_MOJOEMITTER_H

#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "llvm/Support/FormatVariadic.h"

#include <iostream>

namespace M::MOGG {

/// A MojoValue is essentially a string that represents a value in Mojo. That
/// can be a variable (equivalent of lvalue) or a value (rvalue). This class
/// offers some helpers and syntax sugar for doing things with MojoValue
/// objects.
class MojoValue {
public:
  MojoValue() = default;
  MojoValue(std::string &&str) : text(std::move(str)) {}
  MojoValue(StringLiteral str) : text(str) {}
  MojoValue(StringRef str) : text(str) {}
  MojoValue(int64_t integer) : text(std::to_string(integer)) {}
  MojoValue(uint64_t integer) : text(std::to_string(integer)) {}
  MojoValue(bool val) : text(val ? "True" : "False") {}
  MojoValue(const char *str) : text(str) {}
  ~MojoValue() = default;

  static MojoValue None() { return MojoValue("None"); }
  MojoValue rebind(const MojoValue &type) {
    return llvm::formatv("rebind[{}]({})", type.text, text).str();
  }
  MojoValue get(const MojoValue &member) const {
    return llvm::formatv("{}.{}", text, member.text).str();
  }
  MojoValue operator[](const MojoValue &idx) const {
    return llvm::formatv("{}[{}]", text, idx.text).str();
  }
  MojoValue operator()(ArrayRef<MojoValue> parameters = {},
                       ArrayRef<MojoValue> operands = {}) const;
  MojoValue equal(const MojoValue &value) const {
    return llvm::formatv("{} = {}", text, value.text).str();
  }
  MojoValue quote() const { return llvm::formatv("\"{}\"", text).str(); }

  std::string text;
};

inline auto &operator<<(std::ostream &os, const MojoValue &value) {
  return os << value.text;
}

struct FunctionBody {
  SmallVector<MojoValue> parameters;
  SmallVector<MojoValue> arguments;
};

/// This is the equivalent of mlir::OpBuilder but for MojoValue instead of
/// mlir::Operation and mlir::Value.
class MojoBuilder {
public:
  MojoBuilder() : variableCounter(0) {}
  ~MojoBuilder() = default;

  /// Creates a function with the specified parameter and argument types. The
  /// returned object is a RAII guard that will recover the original indentation
  /// level when it goes out of scope.
  void fn(const MojoValue &name, ArrayRef<MojoValue> parameterTypes,
          ArrayRef<MojoValue> argumentTypes, bool exported,
          std::function<void(const FunctionBody &)> &&funcBody);

  [[nodiscard]] MojoBuilder &try_(std::function<void()> &&tryBlock);
  void except(std::function<void(const MojoValue &)> &&exceptBlock);

  /// Creates a variable and assigns the provided value to it. The returned
  /// value is the name of the variable.
  [[nodiscard]] MojoValue var(const MojoValue &value);

  /// Creates an alias and assigns the provided value to it. The returned value
  /// is the name of the parameter.
  [[nodiscard]] MojoValue alias(const MojoValue &value);

  /// Calls a function. This is different from MojoValue::operator() in that it
  /// does not return anything and it actually writes something in the
  /// underlying string stream.
  void call(const MojoValue &funcName, ArrayRef<MojoValue> parameters = {},
            ArrayRef<MojoValue> operands = {});

  /// Apply the assign operator `=` to a specific value.
  /// This is different from `var`, here assign doesn't declare anything.
  /// It can be used to set field / array entries for example.
  /// eg: `foo[idx] = 42`.
  void assign(const MojoValue &dst, const MojoValue &value);

  /// Adds a Mojo comment.
  void comment(const std::string &text);

  /// Adds a new line.
  void newline() { ss << "\n"; }

  /// Emits the keyword `pass`.
  void pass();

  /// Keep the value alive longer `_ = value`
  void keepAlive(const MojoValue &value);

  /// Imports the specified module.
  void import(const MojoValue &mod);

  /// Imports the specified object from the module.
  void import(const MojoValue &from, const MojoValue &import);

  /// Returns the generated Mojo module as a string.
  std::string str() const { return ss.str(); };

  void increaseIdentation() {
    indentation.insert(indentation.size(), "    ", 4);
  }

  void decreaseIdentation() { indentation.erase(indentation.size() - 4, 4); }
  MojoValue newVar(StringLiteral prefix = "val") {
    return llvm::formatv("{}_{}", prefix, variableCounter++).str();
  }

private:
  void indent() { ss << indentation; }
  template <typename... Ts>
  void statement(Ts &&...args) {
    indent();
    (ss << ... << args);
    newline();
  }

  std::stringstream ss;
  std::string indentation;
  uint32_t variableCounter;
};

} // namespace M::MOGG

#endif
