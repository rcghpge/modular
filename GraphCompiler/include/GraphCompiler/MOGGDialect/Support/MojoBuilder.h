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

#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/ADT/SmartVariant.h"
#include "Support/AssertStream.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormatVariadic.h"

#include <stack>

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
  /// Formats a finite floating-point value as a Mojo float literal.
  MojoValue(const llvm::APFloat &value) {
    assert(value.isFinite() && "inf/nan have no Mojo float literal spelling");
    llvm::SmallString<32> str;
    value.toString(str);
    // `APFloat::toString` renders an integral value as bare digits with no
    // fractional part or exponent (e.g. "2"), so append ".0" to make it lex
    // as a float literal.
    if (str.find_first_of(".eE") == llvm::StringRef::npos)
      str += ".0";
    text = str.str();
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
  /// `value[unsafe_offset=idx]` — dereferences a pointer at `idx` using the
  /// keyword form since the positional `ptr[idx]` form is deprecated.
  MojoValue pointerOffsetGet(size_t idx) const {
    return llvm::formatv("{}[unsafe_offset={}]", text, idx).str();
  }
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
  /// Escape and quote the text for use as a Mojo single-line string literal.
  /// Handles \\, ", \n, and \r. Returns a MojoValue safe to embed in "...".
  MojoValue escape() const;
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

  /// Returns the transfer of this value (using the postfix `^` operator).
  /// Used to transfer ownership of move-only types.
  MojoValue transfer() const { return llvm::formatv("{}^", text).str(); }

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

/// Legacy lowering of a BindLambdaOp output to a `FusionStruct` value.
/// `bindMethodName` selects the ManagedTensorSlice binding helper
/// (`_bind_to_fused_input` etc.) used to attach the struct to a tensor.
struct FusionStructInfo {
  MojoValue instance;
  StringLiteral bindMethodName;
  MojoValue typeName;
};

/// How a BindLambdaOp output was lowered, stored in
/// `EmitMojoState::lambdaLowerings`.
using EmittedLambdaLowering = FusionStructInfo;

// Class used to represent a unique mojo identifier.
// It contains a static prefix, followed by a unique index `bar_45`
// In most cases the MojoBuilder takes care of handling unique names so we don't
// need to manipulate it.
class UniqueMojoIdentifier {
  std::string prefix;
  std::optional<uint32_t> idx;

public:
  UniqueMojoIdentifier(const std::string &prefix)
      : prefix(prefix), idx(std::nullopt) {}

private:
  UniqueMojoIdentifier(const std::string &prefix, std::optional<uint32_t> idx)
      : prefix(prefix), idx(idx) {}

  // Get the corresponding identifier.
  std::string str() const {
    ASSERT_STREAM(idx.has_value(), << "UniqueMojoIdentifier must be resolved");
    return llvm::formatv("{}{}", prefix, *idx).str();
  }

  friend class MojoBuilder;
};

// Represent a parameter / operand definition for a function / lambda.
struct NamedMojoDef {
  UniqueMojoIdentifier name;
  MojoValue type;
  std::optional<MojoValue> defaultValue;
  /// When true, the name is emitted verbatim (no counter suffix) and no type
  /// annotation is emitted if `type` is empty. Used for method signatures
  /// that must match a Mojo trait (e.g. `self`, `idx`, `dtype`, `rank`).
  bool verbatim = false;
  /// Argument convention emitted before the name (e.g. `mut`, `owned`). The
  /// convention is NOT part of the value reference, so the `FunctionBody`
  /// argument the body uses remains the bare name. Empty means the default
  /// (borrowed/read) convention with no keyword.
  llvm::StringRef argConvention = "";

  NamedMojoDef(UniqueMojoIdentifier name, MojoValue type,
               std::optional<MojoValue> defaultValue = std::nullopt)
      : name(name), type(type), defaultValue(defaultValue) {}

  NamedMojoDef(const std::string &name, MojoValue type,
               std::optional<MojoValue> defaultValue = std::nullopt)
      : NamedMojoDef(UniqueMojoIdentifier(name), type, defaultValue) {}

  /// Build a NamedMojoDef whose name is emitted verbatim. Passing an empty
  /// `type` (default `MojoValue()`) suppresses the `: type` annotation,
  /// which is needed for the implicit `self` argument of struct methods.
  static NamedMojoDef
  asVerbatim(llvm::StringRef name, MojoValue type = MojoValue(""),
             std::optional<MojoValue> defaultValue = std::nullopt) {
    NamedMojoDef def(name.str(), type, defaultValue);
    def.verbatim = true;
    return def;
  }
};

using MojoValueCache = std::stack<llvm::StringMap<MojoValue>>;

/// This is the equivalent of mlir::OpBuilder but for MojoValue instead of
/// mlir::Operation and mlir::Value.
class MojoBuilder {
public:
  MojoBuilder() : ss(sourceStr), variableCounters({0}) {}
  ~MojoBuilder() {
    ASSERT_STREAM(comptimeCache.empty(),
                  << "Expected caches of MojoValue to be empty");
  }

  /// Creates a function with the specified parameter and argument types, with
  /// optional raises and `@always_inline` decorator. The returned object is a
  /// RAII guard that will recover the original indentation level when it goes
  /// out of scope.
  void def(const MojoValue &name, ArrayRef<NamedMojoDef> parameters,
           ArrayRef<NamedMojoDef> arguments,
           std::optional<MojoValue> returnType, bool exported, bool raises,
           bool isAlwaysInline,
           std::function<void(const FunctionBody &)> &&funcBody);

  /// Creates a lambda with the specified parameter and argument types. The
  /// returned object is a RAII guard that will recover the original indentation
  /// level when it goes out of scope.
  /// Returns the parameter name for the newly defined lambda.
  /// When isAsync is true, emits `async fn` instead of `fn`.
  /// `raises` and `captureConvention` are independent: `raises` controls
  /// whether the lambda's own body can raise; do not infer one from the
  /// other. When `captureConvention` is non-empty (e.g. "var"), the lambda
  /// gets that blanket capture clause (`... {<captureConvention>} ...`),
  /// which Mojo requires for closures capturing enclosing values by name.
  /// Empty means no capture clause (e.g. `@__parameter` closures).
  MojoValue lambda(StringLiteral lambdaName, ArrayRef<NamedMojoDef> parameters,
                   ArrayRef<NamedMojoDef> arguments,
                   std::optional<MojoValue> returnType, bool isInline,
                   bool isParameter, bool isAsync, bool raises,
                   llvm::StringRef captureConvention,
                   std::function<void(const FunctionBody &)> &&lambdaBody);

  [[nodiscard]] MojoBuilder &try_(std::function<void()> &&tryBlock);
  void except(std::function<void(const MojoValue &)> &&exceptBlock);

  MojoBuilder &if_(const MojoValue &cond, std::function<void()> &&ifBlock);
  [[nodiscard]] MojoBuilder &elif_(const MojoValue &cond,
                                   std::function<void()> &&elifBlock);
  void else_(std::function<void()> &&elseBlock);

  /// Creates a `with` block with the given context manager expression.
  MojoBuilder &with_(const MojoValue &contextExpr,
                     std::function<void()> &&body);

  /// Creates a variable and assigns the provided value to it. The returned
  /// value is the name of the variable. When cache is enabled, the builder will
  /// re-use a past variable if the values are the same (like CSE). Set it to
  /// false if the value has a side effect.
  [[nodiscard]] MojoValue var(const MojoValue &value, bool cache = true);

  /// Creates a variable declaration with a type annotation. The returned value
  /// is the name of the variable.
  [[nodiscard]] MojoValue varDecl(const MojoValue &type);

  /// Creates an comptime and assigns the provided value to it. The returned
  /// value is the name of the parameter. There is no cache option here because
  /// it is always on by default (can't have side effects in comptime).
  [[nodiscard]] MojoValue comptime(const MojoValue &value,
                                   const MojoValue *type = nullptr);

  /// Creates a list literal of the supplied MojoValues.
  [[nodiscard]] MojoValue listLiteral(ArrayRef<MojoValue> values);

  /// Calls a function. This is different from MojoValue::operator() in that it
  /// does not return anything and it actually writes something in the
  /// underlying string stream.
  void call(const MojoValue &funcName, ArrayRef<MojoValue> parameters = {},
            ArrayRef<MojoValue> operands = {});

  /// Print a statement as-is, followed by a newline.
  void verbatim(const MojoValue &text);

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

  /// Raises an error value.
  void raise(const MojoValue &value);

  /// Keep the value alive longer `_ = value`
  void keepAlive(const MojoValue &value);

  /// Emits a struct definition with the given name and fields.
  /// Each field is a (name, type) pair.
  /// If fieldwiseInit is true, emits @fieldwise_init decorator.
  /// Example output (with fieldwiseInit=true):
  ///   @fieldwise_init
  ///   struct MyStruct:
  ///       var field1: Type1
  ///       var field2: Type2
  void structDecl(const MojoValue &name,
                  ArrayRef<std::pair<MojoValue, MojoValue>> fields,
                  bool fieldwiseInit = true);

  /// Emits a struct definition with parameters and fields.
  /// Each parameter is a (name, type) pair, and each field is a (name, type)
  /// pair. If fieldwiseInit is true, emits @fieldwise_init decorator.
  /// Example output (with fieldwiseInit=true):
  ///   @fieldwise_init
  ///   struct MyStruct[param1: Type1, param2: Type2]:
  ///       var field1: FieldType1
  ///       var field2: FieldType2
  void structDecl(const MojoValue &name,
                  ArrayRef<std::pair<MojoValue, MojoValue>> parameters,
                  ArrayRef<std::pair<MojoValue, MojoValue>> fields,
                  bool fieldwiseInit = true);

  /// Emits a struct definition with trait conformances, fields, and a body
  /// callback for emitting method definitions. Always emits @fieldwise_init.
  /// Example output:
  ///   @fieldwise_init
  ///   struct MyStruct(MyTrait):
  ///       var field1: Type1
  ///       <body callback output>
  void structDecl(const MojoValue &name, ArrayRef<MojoValue> traits,
                  ArrayRef<std::pair<MojoValue, MojoValue>> fields,
                  std::function<void()> body);

  /// Emits a struct definition with compile-time parameters, trait
  /// conformances, fields, and a body callback. Always emits @fieldwise_init.
  /// Example output:
  ///   @fieldwise_init
  ///   struct MyStruct[param1: Type1](MyTrait):
  ///       var field1: FieldType1
  ///       <body callback output>
  void structDecl(const MojoValue &name,
                  ArrayRef<std::pair<MojoValue, MojoValue>> parameters,
                  ArrayRef<MojoValue> traits,
                  ArrayRef<std::pair<MojoValue, MojoValue>> fields,
                  std::function<void()> body);

  /// Result of a scoped `structDecl` call: the resolved names of the struct's
  /// compile-time bracket parameters, in the same order as the input
  /// `parameters`, mirroring how `FunctionBody` returns resolved def()
  /// parameter/argument names.
  struct StructBody {
    SmallVector<MojoValue> parameters;
  };

  /// Emits `struct Name[params](traits):`, opens the struct's ONE-LEVEL body
  /// scope, then hands control BACK to the caller via `body`. Unlike the other
  /// `structDecl` overloads (which take a pre-declared `fields` list), this one
  /// does NOT pre-declare members: inside `body`, the caller declares fields
  /// with the EXISTING `varDecl(type)`, methods with the EXISTING `def(...)`,
  /// and associated constants with the EXISTING `comptime(value, type)` -- in
  /// any order, as many times as needed. `parameters` use `NamedMojoDef` (the
  /// caller supplies at least a name PREFIX; the builder resolves each to a
  /// fresh unique name exactly like `def()`'s own parameters/arguments), and
  /// the resolved names are handed back via `StructBody` since field/method
  /// type strings commonly need to reference them (e.g. `Self.dtype_3`).
  /// Always emits `@fieldwise_init`, matching the two other body-callback
  /// overloads.
  void structDecl(const MojoValue &name, ArrayRef<NamedMojoDef> parameters,
                  ArrayRef<MojoValue> traits,
                  std::function<void(const StructBody &)> &&body);

  /// Imports the specified module.
  void import(const MojoValue &mod);

  /// Imports the specified module as an alias
  void importAs(const MojoValue &from, const MojoValue &as);

  /// Imports the specified object from the module.
  void importFrom(const MojoValue &from, const MojoValue &import);

  /// Returns the generated Mojo module as a string.
  const std::string &str() const { return sourceStr; }

  void increaseIndentation() {
    indentation.insert(indentation.size(), "    ", 4);
  }

  void decreaseIndentation() { indentation.erase(indentation.size() - 4, 4); }

  UniqueMojoIdentifier getIdentifier(const std::string &prefix = "val") {
    auto prefixWithIndentationLevel =
        prefix + std::string(variableCounters.size(), '_');
    return UniqueMojoIdentifier(prefixWithIndentationLevel,
                                variableCounters.top()++);
  }

  UniqueMojoIdentifier resolveIdentifier(UniqueMojoIdentifier name) {
    if (name.idx.has_value())
      return name;
    else
      return getIdentifier(name.prefix);
  }

  MojoValue newVar(const std::string &name = "val") {
    return resolveIdentifier(name).str();
  }

  MojoValue newVar(UniqueMojoIdentifier name) {
    return resolveIdentifier(name).str();
  }

  void pushCounters() { variableCounters.push({0}); }

  void popCounters() { variableCounters.pop(); }

  void pushCache() {
    varCache.push({});
    comptimeCache.push({});
  }
  void popCache() {
    varCache.pop();
    comptimeCache.pop();
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
                          std::optional<MojoValue> retType,
                          StringRef effectsClause, FunctionBody &body);

  /// Shared implementation for all structDecl overloads. Emits a struct
  /// header with optional parameters and trait conformances, followed by
  /// field declarations. When \p body is non-null, the callback is invoked
  /// after the fields to emit method definitions inside the struct body.
  void structDeclImpl(const MojoValue &name,
                      ArrayRef<std::pair<MojoValue, MojoValue>> parameters,
                      ArrayRef<MojoValue> traits,
                      ArrayRef<std::pair<MojoValue, MojoValue>> fields,
                      bool fieldwiseInit, std::function<void()> body);

  std::string sourceStr;
  llvm::raw_string_ostream ss;
  std::string indentation;

  std::stack<uint32_t> variableCounters;

  // The MojoValue cache is effectively doing partial CSE for Mojo text. Certain
  // EmitsMojoInterface operations will generate many variables which are not
  // represented in the IR. This duplicates a ton of Mojo text. This cache
  // catches those duplications and re-uses them when it can. We treat each
  // scope (if, functions, lambdas) as a new variable scope which is why this is
  // a std::stack. It is pushed on entry and popped on exit.
  MojoValueCache varCache;
  MojoValueCache comptimeCache;
};

} // namespace M::MOGG

#endif
