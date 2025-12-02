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
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/AssertStream.h"
#include "llvm/Support/FormatVariadic.h"

namespace M::MOGG {

class MojoBuilder;

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
  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  MojoValue(T integer) : text(std::to_string(integer)) {}
  MojoValue(const llvm::APSInt &integer) {
    SmallVector<char> chars;
    integer.toString(chars);
    this->text.assign(chars.begin(), chars.end());
  }
  MojoValue(bool val) : text(val ? "True" : "False") {}
  MojoValue(const char *str) : text(str) {}
  ~MojoValue() = default;

  static MojoValue None() { return MojoValue("None"); }
  MojoValue rebind(const MojoValue &type) const {
    return llvm::formatv("rebind[{}]({})", type.text, text).str();
  }
  MojoValue get(const MojoValue &member) const {
    return llvm::formatv("{}.{}", text, member.text).str();
  }
  MojoValue operator[](const MojoValue &idx) const {
    return llvm::formatv("{}[{}]", text, idx.text).str();
  }
  MojoValue operator[](ArrayRef<MojoValue> values) const;
  MojoValue operator()(ArrayRef<MojoValue> parameters = {},
                       ArrayRef<MojoValue> operands = {}) const;
  MojoValue equal(const MojoValue &value) const {
    return llvm::formatv("{} = {}", text, value.text).str();
  }
  MojoValue quote() const {
    std::string res;
    llvm::raw_string_ostream os(res);
    os << '"';
    KGEN::printAsMojoStringLiteral(text, os);
    os << '"';
    return MojoValue(res);
  }
  MojoValue operator+(const MojoValue &other) const {
    return llvm::formatv("{} + {}", text, other.text).str();
  }
  MojoValue operator*(const MojoValue &other) const {
    return llvm::formatv("{} * {}", text, other.text).str();
  }
  MojoValue operator%(const MojoValue &other) const {
    return llvm::formatv("{} % {}", text, other.text).str();
  }
  MojoValue operator<<(const MojoValue &other) const {
    return llvm::formatv("{} << {}", text, other.text).str();
  }
  MojoValue operator>>(const MojoValue &other) const {
    return llvm::formatv("{} >> {}", text, other.text).str();
  }
  MojoValue bitwise_logical_and(const MojoValue &other) const {
    return llvm::formatv("{} & {}", text, other.text).str();
  }
  MojoValue bitwise_logical_or(const MojoValue &other) const {
    return llvm::formatv("{} | {}", text, other.text).str();
  }
  MojoValue bitwise_logical_xor(const MojoValue &other) const {
    return llvm::formatv("{} ^ {}", text, other.text).str();
  }
  MojoValue div(const MojoValue &other) const {
    return llvm::formatv("{} / {}", text, other.text).str();
  }
  MojoValue idiv(const MojoValue &other) const {
    return llvm::formatv("{} // {}", text, other.text).str();
  }
  MojoValue sdiv(const MojoValue &other) const {
    return llvm::formatv("-((-{}) // {})", text, other.text).str();
  }
  MojoValue eq(const MojoValue &other) const {
    return llvm::formatv("{} == {}", text, other.text).str();
  }
  MojoValue neq(const MojoValue &other) const {
    return llvm::formatv("{} != {}", text, other.text).str();
  }
  MojoValue ge(const MojoValue &other) const {
    return llvm::formatv("{} >= {}", text, other.text).str();
  }
  MojoValue gt(const MojoValue &other) const {
    return llvm::formatv("{} > {}", text, other.text).str();
  }
  MojoValue le(const MojoValue &other) const {
    return llvm::formatv("{} <= {}", text, other.text).str();
  }
  MojoValue lt(const MojoValue &other) const {
    return llvm::formatv("{} < {}", text, other.text).str();
  }

  std::string text;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const MojoValue &value) {
  return os << value.text;
}

struct FunctionBody {
  SmallVector<MojoValue> parameters;
  SmallVector<MojoValue> arguments;
};

// Class used to represent a unique mojo identifier.
// It contains a static prefix, followed by a unique index `bar_45`
// In most cases the MojoBuilder takes care of handling unique names so we don't
// need to manipulate it.
class UniqueMojoIdentifier {
  StringLiteral prefix;
  std::optional<uint32_t> idx;

public:
  UniqueMojoIdentifier(StringLiteral prefix)
      : prefix(prefix), idx(std::nullopt) {}

private:
  UniqueMojoIdentifier(StringLiteral prefix, std::optional<uint32_t> idx)
      : prefix(prefix), idx(idx) {}

  // Get the corresponding identifier.
  std::string str() const {
    ASSERT_STREAM(idx.has_value(), << "UniqueMojoIdentifier must be resolved");
    return llvm::formatv("{}_{}", prefix, *idx).str();
  }

  friend class MojoBuilder;
};

// Represent a parameter / operand definition for a function / lambda.
struct NamedMojoDef {
  UniqueMojoIdentifier name;
  MojoValue type;
  std::optional<MojoValue> defaultValue;

  NamedMojoDef(UniqueMojoIdentifier name, MojoValue type,
               std::optional<MojoValue> defaultValue = std::nullopt)
      : name(name), type(type), defaultValue(defaultValue) {}

  NamedMojoDef(StringLiteral name, MojoValue type,
               std::optional<MojoValue> defaultValue = std::nullopt)
      : NamedMojoDef(UniqueMojoIdentifier(name), type, defaultValue) {}
};

/// This is the equivalent of mlir::OpBuilder but for MojoValue instead of
/// mlir::Operation and mlir::Value.
class MojoBuilder {
public:
  MojoBuilder() : ss(sourceStr), variableCounter(0) {}
  ~MojoBuilder() = default;

  /// Creates a function with the specified parameter and argument types, with
  /// optional raises. The returned object is a RAII guard that will recover the
  /// original indentation level when it goes out of scope.
  void fn(const MojoValue &name, ArrayRef<NamedMojoDef> parameters,
          ArrayRef<NamedMojoDef> arguments, std::optional<MojoValue> returnType,
          bool exported, bool raises,
          std::function<void(const FunctionBody &)> &&funcBody);

  /// Creates a lambda with the specified parameter and argument types. The
  /// returned object is a RAII guard that will recover the original indentation
  /// level when it goes out of scope.
  /// Returns the parameter name for the newly defined lambda.
  MojoValue lambda(StringLiteral lambdaName, ArrayRef<NamedMojoDef> parameters,
                   ArrayRef<NamedMojoDef> arguments,
                   std::optional<MojoValue> returnType, bool isInline,
                   bool isParameter,
                   std::function<void(const FunctionBody &)> &&lambdaBody);

  [[nodiscard]] MojoBuilder &try_(std::function<void()> &&tryBlock);
  void except(std::function<void(const MojoValue &)> &&exceptBlock);

  MojoBuilder &if_(const MojoValue &cond, std::function<void()> &&ifBlock);
  void else_(std::function<void()> &&elseBlock);

  /// Creates a variable and assigns the provided value to it. The returned
  /// value is the name of the variable.
  [[nodiscard]] MojoValue var(const MojoValue &value);

  /// Creates a variable declaration with a type annotation. The returned value
  /// is the name of the variable.
  [[nodiscard]] MojoValue varDecl(const MojoValue &type);

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

  /// Returns a mojo value
  void returnsValue(const MojoValue &value);

  /// Keep the value alive longer `_ = value`
  void keepAlive(const MojoValue &value);

  /// Imports the specified module.
  void import(const MojoValue &mod);

  /// Imports the specified object from the module.
  void import(const MojoValue &from, const MojoValue &import);

  /// Returns the generated Mojo module as a string.
  const std::string &str() const { return sourceStr; }

  void increaseIndentation() {
    indentation.insert(indentation.size(), "    ", 4);
  }

  void decreaseIndentation() { indentation.erase(indentation.size() - 4, 4); }

  UniqueMojoIdentifier getIdentifier(StringLiteral prefix = "val") {
    return UniqueMojoIdentifier(prefix, variableCounter++);
  }

  UniqueMojoIdentifier resolveIdentifier(UniqueMojoIdentifier name) {
    if (name.idx.has_value())
      return name;
    else
      return getIdentifier(name.prefix);
  }

  MojoValue newVar(StringLiteral name = "val") {
    return resolveIdentifier(name).str();
  }

  MojoValue newVar(UniqueMojoIdentifier name) {
    return resolveIdentifier(name).str();
  }

private:
  void indent() { ss << indentation; }
  template <typename... Ts>
  void statement(Ts &&...args) {
    indent();
    (ss << ... << args);
    newline();
  }

  std::string funcDefBase(ArrayRef<NamedMojoDef> parameters,
                          ArrayRef<NamedMojoDef> operands,
                          std::optional<MojoValue> retType, FunctionBody &body);

  std::string funcDefBase(ArrayRef<NamedMojoDef> parameters,
                          ArrayRef<NamedMojoDef> operands,
                          std::optional<MojoValue> retType, bool raises,
                          FunctionBody &body);

  std::string sourceStr;
  llvm::raw_string_ostream ss;
  std::string indentation;
  uint32_t variableCounter;
};

} // namespace M::MOGG

#endif
