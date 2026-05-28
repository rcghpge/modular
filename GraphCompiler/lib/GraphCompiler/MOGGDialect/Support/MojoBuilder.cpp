//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/MojoBuilder.h"

namespace M::MOGG {

MojoValue MojoValue::operator[](ArrayRef<MojoValue> values) const {
  std::string str;
  llvm::raw_string_ostream stream(str);
  stream << text;
  if (!values.empty()) {
    stream << "[";
    llvm::interleaveComma(values, stream);
    stream << "]";
  }

  return str;
}

MojoValue MojoValue::operator()(ArrayRef<MojoValue> parameters,
                                ArrayRef<MojoValue> operands) const {
  std::string str;
  llvm::raw_string_ostream stream(str);
  stream << text;
  if (!parameters.empty()) {
    stream << "[";
    llvm::interleaveComma(parameters, stream);
    stream << "]";
  }

  stream << "(";
  llvm::interleaveComma(operands, stream);
  stream << ")";
  return str;
}

MojoValue MojoValue::escape() const {
  std::string res = "\"";
  for (char c : text) {
    switch (c) {
    case '\\':
      res += "\\\\";
      break;
    case '"':
      res += "\\\"";
      break;
    case '\n':
      res += "\\n";
      break;
    case '\r':
      res += "\\r";
      break;
    default:
      res += c;
      break;
    }
  }
  res += '"';
  return MojoValue(res);
}

MojoValue createTypedVar(const MojoValue &name, const MojoValue &type) {
  return llvm::formatv("{}: {}", name.text, type.text).str();
}

class MojoBuilderScopeGuard {
public:
  MojoBuilderScopeGuard(MojoBuilder *e, bool deferExceptCounters = false)
      : emitter(e), posBeforeBody(0) {
    if (!deferExceptCounters) {
      emitter->increaseIndentation();
      emitter->pushCache();
      // Record position after indentation setup for empty body detection.
      posBeforeBody = emitter->str().size();
    }
    emitter->pushCounters();
  }
  MojoBuilderScopeGuard() = delete;
  MojoBuilderScopeGuard(const MojoBuilderScopeGuard &other) = delete;
  ~MojoBuilderScopeGuard() {
    // Mojo requires non-empty function bodies. Emit 'pass' if nothing was
    // written.
    if (emitter->str().size() == posBeforeBody)
      emitter->pass();
    emitter->decreaseIndentation();
    emitter->newline();
    emitter->popCache();
    emitter->popCounters();
  }
  MojoBuilder *emitter;
  size_t posBeforeBody;

  void doDeferredActions() {
    emitter->increaseIndentation();
    emitter->pushCache();
    // Record position after indentation setup for empty body detection.
    posBeforeBody = emitter->str().size();
  }
};

std::string MojoBuilder::funcDefBase(ArrayRef<NamedMojoDef> parameters,
                                     ArrayRef<NamedMojoDef> operands,
                                     std::optional<MojoValue> retType,
                                     FunctionBody &body) {
  return funcDefBase(parameters, operands, retType, /*effectsClause=*/"", body);
}

std::string MojoBuilder::funcDefBase(ArrayRef<NamedMojoDef> parameters,
                                     ArrayRef<NamedMojoDef> operands,
                                     std::optional<MojoValue> retType,
                                     StringRef effectsClause,
                                     FunctionBody &body) {
  std::string str;
  llvm::raw_string_ostream stream(str);
  auto emitNamedDef = [&](const NamedMojoDef &def, StringRef typeSep,
                          SmallVectorImpl<MojoValue> &out) {
    MojoValue name =
        def.verbatim ? MojoValue(def.name.prefix) : newVar(def.name);
    stream << name;
    if (!(def.verbatim && def.type.text.empty()))
      stream << typeSep << def.type;
    if (def.defaultValue.has_value())
      stream << " = " << *def.defaultValue;
    out.push_back(name);
  };
  if (!parameters.empty()) {
    stream << "[";
    llvm::interleaveComma(parameters, stream, [&](const NamedMojoDef &param) {
      emitNamedDef(param, " : ", body.parameters);
    });
    stream << "]";
  }

  stream << "(";
  llvm::interleaveComma(operands, stream, [&](const NamedMojoDef &operand) {
    emitNamedDef(operand, ": ", body.arguments);
  });
  stream << ")";

  if (!effectsClause.empty())
    stream << " " << effectsClause;

  if (retType.has_value())
    stream << " -> " << *retType;

  stream << ":";
  return str;
}

void MojoBuilder::def(const MojoValue &name, ArrayRef<NamedMojoDef> parameters,
                      ArrayRef<NamedMojoDef> arguments,
                      std::optional<MojoValue> returnType, bool exported,
                      bool raises, bool isAlwaysInline,
                      std::function<void(const FunctionBody &)> &&funcBody) {
  newline();

  FunctionBody body;

  MojoBuilderScopeGuard guard(this, true);

  if (isAlwaysInline)
    statement("@always_inline");
  if (exported)
    statement("@export");
  statement("def ", name,
            funcDefBase(parameters, arguments, returnType,
                        /*effectsClause=*/raises ? "raises" : "", body));

  // This is necessary to increase indentation and the value cache. The reason
  // we defer them is because we create the guard early to ensure that argument
  // names use new counters.
  guard.doDeferredActions();
  funcBody(body);
}

MojoValue
MojoBuilder::lambda(StringLiteral lambdaName, ArrayRef<NamedMojoDef> parameters,
                    ArrayRef<NamedMojoDef> arguments,
                    std::optional<MojoValue> returnType, bool isInline,
                    bool isParameter, bool isAsync,
                    std::function<void(const FunctionBody &)> &&lambdaBody) {
  MojoValue lambdaParam = newVar(lambdaName.str());
  FunctionBody body;

  if (isParameter)
    statement("@parameter");
  if (isInline)
    statement("@always_inline");
  statement(isAsync ? "async def " : "def ", lambdaParam,
            funcDefBase(parameters, arguments, returnType, body));

  MojoBuilderScopeGuard guard(this);
  lambdaBody(body);
  return lambdaParam;
}

MojoValue MojoBuilder::registerPassableElementwiseDef(
    StringLiteral funcNameHint, ArrayRef<NamedMojoDef> parameters,
    ArrayRef<NamedMojoDef> arguments, ArrayRef<MojoValue> capturedVarNames,
    std::optional<MojoValue> returnType,
    std::function<void(const FunctionBody &)> &&defBody) {
  MojoValue funcName = newVar(funcNameHint.str());
  FunctionBody body;

  // Only the `{var ...}` clause; register-passability follows from the
  // captures. The `register_passable` keyword is gone from the frontend (now
  // warns and is ignored), so don't emit it.
  std::string captureClause;
  llvm::raw_string_ostream cos(captureClause);
  cos << "{";
  llvm::interleaveComma(capturedVarNames, cos,
                        [&](const MojoValue &v) { cos << "var " << v.text; });
  cos << "}";

  statement(
      "def ", funcName,
      funcDefBase(parameters, arguments, returnType, captureClause, body));

  MojoBuilderScopeGuard guard(this);
  defBody(body);
  return funcName;
}

MojoBuilder &MojoBuilder::try_(std::function<void()> &&tryBlock) {
  statement("try:");
  MojoBuilderScopeGuard guard(this);
  tryBlock();

  return *this;
}

void MojoBuilder::except(std::function<void(const MojoValue &)> &&exceptBlock) {
  auto e = newVar("e");
  statement("except ", e, ":");
  MojoBuilderScopeGuard guard(this);
  exceptBlock(e);
}

MojoBuilder &MojoBuilder::if_(const MojoValue &cond,
                              std::function<void()> &&ifBlock) {
  statement("if ", cond, ":");
  MojoBuilderScopeGuard guard(this);
  ifBlock();
  return *this;
}

MojoBuilder &MojoBuilder::with_(const MojoValue &contextExpr,
                                std::function<void()> &&body) {
  statement("with ", contextExpr, ":");
  MojoBuilderScopeGuard guard(this);
  body();
  return *this;
}

MojoBuilder &MojoBuilder::elif_(const MojoValue &cond,
                                std::function<void()> &&elifBlock) {
  statement("elif ", cond, ":");
  MojoBuilderScopeGuard guard(this);
  elifBlock();
  return *this;
}
void MojoBuilder::else_(std::function<void()> &&elseBlock) {
  statement("else:");
  MojoBuilderScopeGuard guard(this);
  elseBlock();
}

MojoValue MojoBuilder::var(const MojoValue &value, bool cache) {
  std::pair<llvm::StringMapIterBase<M::MOGG::MojoValue, false>, bool> it;
  if (cache) {
    it = varCache.top().insert({value.text, ""});
    if (!it.second)
      return it.first->second;
  }

  auto name = newVar("var");
  statement("var ", name, " = ", value);

  if (cache)
    it.first->second = name;

  return name;
}
MojoValue MojoBuilder::varDecl(const MojoValue &type) {
  auto name = newVar("var");
  statement("var ", createTypedVar(name, type));

  return name;
}
MojoValue MojoBuilder::comptime(const MojoValue &value, const MojoValue *type) {
  auto it = comptimeCache.top().insert({value.text, ""});
  if (!it.second)
    return it.first->second;

  auto name = newVar("param");

  if (type)
    statement("comptime ", name, ": ", *type, " = ", value);
  else
    statement("comptime ", name, " = ", value);

  it.first->second = name;

  return name;
}
MojoValue MojoBuilder::listLiteral(ArrayRef<MojoValue> values) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "[";
  llvm::interleaveComma(values, os);
  os << "]";

  return MojoValue(std::move(str));
}
void MojoBuilder::call(const MojoValue &funcName,
                       ArrayRef<MojoValue> parameters,
                       ArrayRef<MojoValue> operands) {
  statement(funcName(parameters, operands));
}

void MojoBuilder::verbatim(const MojoValue &text) { statement(text); }

void MojoBuilder::assign(const MojoValue &dst, const MojoValue &value) {
  statement(dst, " = ", value);
}

void MojoBuilder::comment(const std::string &text) { statement("# ", text); }

void MojoBuilder::pass() { statement("pass"); }

void MojoBuilder::returnsValue(const MojoValue &value) {
  statement("return ", value);
}

void MojoBuilder::raise(const MojoValue &value) { statement("raise ", value); }

void MojoBuilder::keepAlive(const MojoValue &value) {
  statement("_ = ", value);
}

void MojoBuilder::import(const MojoValue &mod) {
  statement("import ", mod.text);
}

void MojoBuilder::importAs(const MojoValue &mod, const MojoValue &as) {
  statement("import ", mod.text, " as ", as.text);
}

void MojoBuilder::importFrom(const MojoValue &from, const MojoValue &import) {
  statement("from ", from.text, " import ", import.text);
}

void MojoBuilder::structDecl(const MojoValue &name,
                             ArrayRef<std::pair<MojoValue, MojoValue>> fields,
                             bool fieldwiseInit) {
  structDeclImpl(name, /*parameters=*/{}, /*traits=*/{}, fields, fieldwiseInit,
                 nullptr);
}

void MojoBuilder::structDecl(
    const MojoValue &name, ArrayRef<std::pair<MojoValue, MojoValue>> parameters,
    ArrayRef<std::pair<MojoValue, MojoValue>> fields, bool fieldwiseInit) {
  structDeclImpl(name, parameters, /*traits=*/{}, fields, fieldwiseInit,
                 nullptr);
}

void MojoBuilder::structDecl(const MojoValue &name, ArrayRef<MojoValue> traits,
                             ArrayRef<std::pair<MojoValue, MojoValue>> fields,
                             std::function<void()> body) {
  structDeclImpl(name, /*parameters=*/{}, traits, fields,
                 /*fieldwiseInit=*/true, std::move(body));
}

void MojoBuilder::structDecl(
    const MojoValue &name, ArrayRef<std::pair<MojoValue, MojoValue>> parameters,
    ArrayRef<MojoValue> traits,
    ArrayRef<std::pair<MojoValue, MojoValue>> fields,
    std::function<void()> body) {
  structDeclImpl(name, parameters, traits, fields,
                 /*fieldwiseInit=*/true, std::move(body));
}

void MojoBuilder::structDeclImpl(
    const MojoValue &name, ArrayRef<std::pair<MojoValue, MojoValue>> parameters,
    ArrayRef<MojoValue> traits,
    ArrayRef<std::pair<MojoValue, MojoValue>> fields, bool fieldwiseInit,
    std::function<void()> body) {
  newline();
  if (fieldwiseInit)
    statement("@fieldwise_init");

  std::string headerStr;
  llvm::raw_string_ostream os(headerStr);
  os << "struct " << name.text;
  if (!parameters.empty()) {
    os << "[";
    llvm::interleaveComma(parameters, os, [&](const auto &p) {
      os << p.first.text << ": " << p.second.text;
    });
    os << "]";
  }
  if (!traits.empty()) {
    os << "(";
    llvm::interleaveComma(traits, os, [&](const auto &t) { os << t.text; });
    os << ")";
  }
  os << ":";
  statement(headerStr);

  increaseIndentation();
  if (body)
    pushCache();

  for (const auto &[fieldName, fieldType] : fields) {
    statement("var ", fieldName, ": ", fieldType);
  }

  if (body) {
    newline();
    body();
    popCache();
  }

  decreaseIndentation();
  newline();
}

} // namespace M::MOGG
