//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoBuilder.h"

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

MojoValue createTypedVar(const MojoValue &name, const MojoValue &type) {
  return llvm::formatv("{}: {}", name.text, type.text).str();
}

class MojoBuilderScopeGuard {
public:
  MojoBuilderScopeGuard(MojoBuilder *e) : emitter(e) {
    emitter->increaseIdentation();
  }
  MojoBuilderScopeGuard() = delete;
  MojoBuilderScopeGuard(const MojoBuilderScopeGuard &other) = delete;
  ~MojoBuilderScopeGuard() {
    emitter->decreaseIdentation();
    emitter->newline();
  }
  MojoBuilder *emitter;
};

std::string MojoBuilder::funcDefBase(ArrayRef<NamedMojoDef> parameters,
                                     ArrayRef<NamedMojoDef> operands,
                                     std::optional<MojoValue> retType,
                                     FunctionBody &body) {
  return funcDefBase(parameters, operands, retType, false, body);
}

std::string MojoBuilder::funcDefBase(ArrayRef<NamedMojoDef> parameters,
                                     ArrayRef<NamedMojoDef> operands,
                                     std::optional<MojoValue> retType,
                                     bool raises, FunctionBody &body) {
  std::string str;
  llvm::raw_string_ostream stream(str);
  if (!parameters.empty()) {
    stream << "[";
    llvm::interleaveComma(parameters, stream, [&](const NamedMojoDef &param) {
      MojoValue paramName = newVar(param.name);
      stream << paramName << " : " << param.type;
      if (param.defaultValue.has_value())
        stream << " = " << *param.defaultValue;
      body.parameters.push_back(paramName);
    });
    stream << "]";
  }

  stream << "(";
  llvm::interleaveComma(operands, stream, [&](const NamedMojoDef &operand) {
    MojoValue operandName = newVar(operand.name);
    stream << operandName << ": " << operand.type;
    if (operand.defaultValue.has_value())
      stream << " = " << *operand.defaultValue;
    body.arguments.push_back(operandName);
  });
  stream << ")";

  if (retType.has_value())
    stream << " -> " << *retType;

  if (raises)
    stream << " raises";

  stream << ":";
  return str;
}

void MojoBuilder::fn(const MojoValue &name, ArrayRef<NamedMojoDef> parameters,
                     ArrayRef<NamedMojoDef> arguments,
                     std::optional<MojoValue> returnType, bool exported,
                     bool raises,
                     std::function<void(const FunctionBody &)> &&funcBody) {
  newline();

  FunctionBody body;

  if (exported)
    statement("@export");
  statement("fn ", name,
            funcDefBase(parameters, arguments, returnType, raises, body));

  MojoBuilderScopeGuard guard(this);
  funcBody(body);
}

MojoValue
MojoBuilder::lambda(StringLiteral lambdaName, ArrayRef<NamedMojoDef> parameters,
                    ArrayRef<NamedMojoDef> arguments,
                    std::optional<MojoValue> returnType, bool isInline,
                    bool isParameter,
                    std::function<void(const FunctionBody &)> &&lambdaBody) {
  MojoValue lambdaParam = newVar(lambdaName);
  FunctionBody body;

  if (isParameter)
    statement("@parameter");
  if (isInline)
    statement("@always_inline");
  statement("fn ", lambdaParam,
            funcDefBase(parameters, arguments, returnType, false, body));

  MojoBuilderScopeGuard guard(this);
  lambdaBody(body);
  return lambdaParam;
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

MojoValue MojoBuilder::var(const MojoValue &value) {
  auto name = newVar("var");
  statement("var ", name, " = ", value);

  return name;
}
MojoValue MojoBuilder::varDecl(const MojoValue &type) {
  auto name = newVar("var");
  statement("var ", createTypedVar(name, type));

  return name;
}
MojoValue MojoBuilder::alias(const MojoValue &value) {
  auto name = newVar("param");
  statement("alias ", name, " = ", value);

  return name;
}
void MojoBuilder::call(const MojoValue &funcName,
                       ArrayRef<MojoValue> parameters,
                       ArrayRef<MojoValue> operands) {
  statement(funcName(parameters, operands));
}

void MojoBuilder::assign(const MojoValue &dst, const MojoValue &value) {
  statement(dst, " = ", value);
}

void MojoBuilder::comment(const std::string &text) { statement("# ", text); }

void MojoBuilder::pass() { statement("pass"); }

void MojoBuilder::returnsValue(const MojoValue &value) {
  statement("return ", value);
}

void MojoBuilder::keepAlive(const MojoValue &value) {
  statement("_ = ", value);
}

void MojoBuilder::import(const MojoValue &mod) {
  statement("import ", mod.text);
}

void MojoBuilder::import(const MojoValue &from, const MojoValue &import) {
  statement("from ", from.text, " import ", import.text);
}

} // namespace M::MOGG
