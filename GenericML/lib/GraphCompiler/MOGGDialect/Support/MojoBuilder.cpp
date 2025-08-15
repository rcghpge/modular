//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoBuilder.h"

namespace M::MOGG {

MojoValue MojoValue::operator()(ArrayRef<MojoValue> parameters,
                                ArrayRef<MojoValue> operands) const {
  std::stringstream stream;
  stream << text;
  if (!parameters.empty()) {
    stream << "[";
    for (auto parameter : parameters.drop_back())
      stream << parameter << ", ";
    stream << parameters.back() << "]";
  }

  if (operands.empty()) {
    stream << "()";
  } else {
    stream << "(";
    for (auto operand : operands.drop_back())
      stream << operand << ", ";
    stream << operands.back() << ")";
  }

  return stream.str();
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

void MojoBuilder::fn(const MojoValue &name, ArrayRef<MojoValue> parameterTypes,
                     ArrayRef<MojoValue> argumentTypes, bool exported,
                     std::function<void(const FunctionBody &)> &&funcBody) {
  newline();

  FunctionBody body;
  SmallVector<MojoValue> typedParameters;
  for (auto &&type : parameterTypes) {
    body.parameters.push_back(newVar("arg"));
    typedParameters.push_back(createTypedVar(body.parameters.back(), type));
  }

  SmallVector<MojoValue> typedArguments;
  for (auto &&type : argumentTypes) {
    body.arguments.push_back(newVar("arg"));
    typedArguments.push_back(createTypedVar(body.arguments.back(), type));
  }

  if (exported)
    statement("@export");
  statement("fn ", name(typedParameters, typedArguments), ":");

  MojoBuilderScopeGuard guard(this);
  funcBody(body);
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

void MojoBuilder::comment(const std::string &text) { statement("# ", text); }

void MojoBuilder::pass() { statement("pass"); }

void MojoBuilder::import(const MojoValue &mod) {
  statement("import ", mod.text);
}

void MojoBuilder::import(const MojoValue &from, const MojoValue &import) {
  statement("from ", from.text, " import ", import.text);
}

} // namespace M::MOGG
