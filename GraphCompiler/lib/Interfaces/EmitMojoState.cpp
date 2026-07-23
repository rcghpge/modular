//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENPogUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/POPDialect/POPOps.h"

using namespace M;

namespace M::MOGG {

void EmitMojoState::setFusionStructLoweringFor(Value val, MojoValue instance,
                                               StringLiteral bindMethod,
                                               MojoValue typeName) {
  [[maybe_unused]] auto [it, inserted] = lambdaLowerings.try_emplace(
      val,
      FusionStructInfo{std::move(instance), bindMethod, std::move(typeName)});
  ASSERT_STREAM(inserted, << "value already bound to a lambda lowering");
}

/// Remove parameters from inferByInputs that appear in the function's result
/// type but not in the actual input argument types.
static void removeResultOnlyParams(KGEN::LIT::FnTypeGeneratorType sig,
                                   llvm::DenseSet<StringRef> &inferByInputs) {
  // Collect params from result type.
  SmallVector<KGEN::ParamDeclRefAttr> resultParamRefs;
  bool hasConstExpr = false;
  size_t requiredSignatureDepth = 0;
  Type userResultType = sig.getUserResultType();
  KGEN::collectParameterReferences(userResultType, resultParamRefs,
                                   hasConstExpr, requiredSignatureDepth);
  llvm::DenseSet<StringRef> resultParams;
  for (auto paramRef : resultParamRefs)
    resultParams.insert(paramRef.getName().getValue());

  // Collect params from actual input arguments (excluding byref result slot).
  ArrayRef<Type> argTypes = sig.getArguments();
  if (sig.hasMemoryOnlyResult())
    argTypes = argTypes.drop_back();
  if (sig.isThrows())
    argTypes = argTypes.drop_back();

  llvm::DenseSet<StringRef> inputParams;
  for (Type argType : argTypes) {
    SmallVector<KGEN::ParamDeclRefAttr> argParamRefs;
    KGEN::collectParameterReferences(argType, argParamRefs, hasConstExpr,
                                     requiredSignatureDepth);
    for (auto paramRef : argParamRefs)
      inputParams.insert(paramRef.getName().getValue());
  }

  // Remove any parameters that are only in the result type, not in inputs.
  SmallVector<StringRef> toRemove;
  for (StringRef paramName : inferByInputs) {
    if (resultParams.contains(paramName) && !inputParams.contains(paramName))
      toRemove.push_back(paramName);
  }
  for (StringRef paramName : toRemove)
    inferByInputs.erase(paramName);
}

#define EMITS_MOJO_EXTERNALLY(OpTy)                                            \
  LogicalResult emitMojo(OpTy op, EmitMojoState &state);
#define EMIT_MOJO_EXTERNALLY(OpTy)                                             \
  if (auto asOp = dyn_cast<OpTy>(op)) {                                        \
    if (emitMojo(asOp, state).failed()) {                                      \
      return op->emitError("Could not lower this operation to Mojo");          \
    }                                                                          \
    return success();                                                          \
  }

EMITS_MOJO_EXTERNALLY(mlir::index::AddOp);
EMITS_MOJO_EXTERNALLY(mlir::index::CastSOp);
EMITS_MOJO_EXTERNALLY(mlir::index::CmpOp);
EMITS_MOJO_EXTERNALLY(mlir::index::ConstantOp);
EMITS_MOJO_EXTERNALLY(mlir::index::BoolConstantOp);
EMITS_MOJO_EXTERNALLY(mlir::index::DivSOp);
EMITS_MOJO_EXTERNALLY(mlir::index::DivUOp);
EMITS_MOJO_EXTERNALLY(mlir::index::CeilDivSOp);
EMITS_MOJO_EXTERNALLY(mlir::index::CeilDivUOp);
EMITS_MOJO_EXTERNALLY(mlir::index::MinSOp);
EMITS_MOJO_EXTERNALLY(mlir::index::MaxSOp);
EMITS_MOJO_EXTERNALLY(mlir::index::MulOp);
EMITS_MOJO_EXTERNALLY(mlir::index::RemSOp);
EMITS_MOJO_EXTERNALLY(mlir::index::RemUOp);

EMITS_MOJO_EXTERNALLY(KGEN::POP::AndOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::AddOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::MulOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::DivOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::CastFromBuiltinOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::CastToBuiltinOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::CmpOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::RemOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::SelectOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::ShlOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::ShrOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::MinOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::MaxOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::OrOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::XOrOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::SIMDAndOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::SIMDSelectOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::SIMDOrOp);
EMITS_MOJO_EXTERNALLY(KGEN::POP::SIMDXOrOp);

EMITS_MOJO_EXTERNALLY(KGEN::ParamConstantOp);

LogicalResult EmitMojoState::lowerOpToMojo(Operation *op) {
  if (auto emitMojoInterface = dyn_cast<MOGG::EmitsMojoInterface>(op)) {
    if (enableOpTraceComments)
      builder.comment(llvm::formatv("Emitting Mojo for: {0}", op->getName()));

    if (emitMojoInterface.emitMojo(*this).failed()) {
      return emitMojoInterface->emitError(
          "Could not lower this operation to Mojo");
    }
    return success();
  }

  auto &state = *this;

  EMIT_MOJO_EXTERNALLY(mlir::index::AddOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::CastSOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::CmpOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::ConstantOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::BoolConstantOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::DivSOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::DivUOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::CeilDivSOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::CeilDivUOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::MinSOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::MaxSOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::MulOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::RemSOp);
  EMIT_MOJO_EXTERNALLY(mlir::index::RemUOp);

  EMIT_MOJO_EXTERNALLY(KGEN::POP::AndOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::AddOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::MulOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::DivOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::CastFromBuiltinOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::CastToBuiltinOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::CmpOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::RemOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::SelectOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::ShlOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::ShrOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::MinOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::MaxOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::OrOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::XOrOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::SIMDAndOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::SIMDSelectOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::SIMDOrOp);
  EMIT_MOJO_EXTERNALLY(KGEN::POP::SIMDXOrOp);

  EMIT_MOJO_EXTERNALLY(KGEN::ParamConstantOp);

  return op->emitError(
      "This operation doesn't implement EmitsMojoInterface and it is not a "
      "supported index or pop dialect operation");
}

//===----------------------------------------------------------------------===//
// index dialect implementations
//===----------------------------------------------------------------------===//

LogicalResult emitMojo(mlir::index::AddOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] + operands[1]));
  return success();
}

LogicalResult emitMojo(mlir::index::CastSOp op, EmitMojoState &state) {
  /// TODO: Check if this has any ramifications. The reason this op is emitted
  /// at all in MGP is just to please the type system (pop operations need pop
  /// types to operate on, not index). Therefore, it's likely that keeping them
  /// as-is (Int) is perfectly fine for the index calculations we do in MGP
  /// (shape function stuff only).
  state.setMojoValueFor(op.getResult(), state.getMojoValueFor(op.getInput()));
  return success();
}

LogicalResult emitMojo(mlir::index::CmpOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  MojoValue replacementValue;
  switch (op.getPred()) {
  case mlir::index::IndexCmpPredicate::EQ:
    replacementValue = operands[0].eq(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::NE:
    replacementValue = operands[0].neq(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::SGE:
    replacementValue = operands[0].ge(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::SGT:
    replacementValue = operands[0].gt(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::SLE:
    replacementValue = operands[0].le(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::SLT:
    replacementValue = operands[0].lt(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::UGE:
    replacementValue = operands[0].ge(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::UGT:
    replacementValue = operands[0].gt(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::ULE:
    replacementValue = operands[0].le(operands[1]);
    break;
  case mlir::index::IndexCmpPredicate::ULT:
    replacementValue = operands[0].lt(operands[1]);
    break;
  }
  state.setMojoValueFor(op.getResult(), state.builder.var(replacementValue));

  return success();
}

LogicalResult emitMojo(mlir::index::ConstantOp op, EmitMojoState &state) {
  state.setMojoValueFor(op.getResult(), op.getValue().getSExtValue());
  return success();
}

LogicalResult emitMojo(mlir::index::BoolConstantOp op, EmitMojoState &state) {
  state.setMojoValueFor(op.getResult(), op.getValue());
  return success();
}

LogicalResult emitMojo(mlir::index::DivSOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0].idiv(operands[1])));
  return success();
}

LogicalResult emitMojo(mlir::index::DivUOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0].idiv(operands[1])));
  return success();
}

LogicalResult emitMojo(mlir::index::CeilDivSOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0].sdiv(operands[1])));
  return success();
}

LogicalResult emitMojo(mlir::index::CeilDivUOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0].sdiv(operands[1])));
  return success();
}

LogicalResult emitMojo(mlir::index::MinSOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  static const MojoValue minFunc = "min";
  state.setMojoValueFor(op.getResult(), state.builder.var(minFunc(
                                            {}, {operands[0], operands[1]})));
  return success();
}

LogicalResult emitMojo(mlir::index::MaxSOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  static const MojoValue maxFunc = kFnMax;
  state.setMojoValueFor(op.getResult(), state.builder.var(maxFunc(
                                            {}, {operands[0], operands[1]})));
  return success();
}

LogicalResult emitMojo(mlir::index::MulOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] * operands[1]));
  return success();
}

LogicalResult emitMojo(mlir::index::RemSOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] % operands[1]));
  return success();
}

LogicalResult emitMojo(mlir::index::RemUOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] % operands[1]));
  return success();
}

//===----------------------------------------------------------------------===//
// POP dialect implementations
//===----------------------------------------------------------------------===//

LogicalResult emitMojo(KGEN::POP::AndOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(
      op.getResult(),
      state.builder.var(operands[0].bitwise_logical_and(operands[1])));
  return success();
}

LogicalResult emitMojo(KGEN::POP::AddOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] + operands[1]));
  return success();
}

LogicalResult emitMojo(KGEN::POP::MulOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] * operands[1]));
  return success();
}

LogicalResult emitMojo(KGEN::POP::DivOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  if (op.getType().isFloat()) {
    state.setMojoValueFor(op.getResult(),
                          state.builder.var(operands[0].div(operands[1])));
  } else {
    state.setMojoValueFor(op.getResult(),
                          state.builder.var(operands[0].idiv(operands[1])));
  }
  return success();
}

LogicalResult emitMojo(KGEN::POP::CastFromBuiltinOp op, EmitMojoState &state) {
  state.setMojoValueFor(op.getResult(), state.getMojoValueFor(op.getInput()));
  return success();
}

LogicalResult emitMojo(KGEN::POP::CastToBuiltinOp op, EmitMojoState &state) {
  state.setMojoValueFor(op.getResult(), state.getMojoValueFor(op.getInput()));
  return success();
}

LogicalResult emitMojo(KGEN::POP::CmpOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  MojoValue replacementValue;
  switch (op.getPred()) {
  case KGEN::CmpPredicate::EQ:
    replacementValue = operands[0].eq(operands[1]);
    break;
  case KGEN::CmpPredicate::NE:
    replacementValue = operands[0].neq(operands[1]);
    break;
  case KGEN::CmpPredicate::GE:
    replacementValue = operands[0].ge(operands[1]);
    break;
  case KGEN::CmpPredicate::GT:
    replacementValue = operands[0].gt(operands[1]);
    break;
  case KGEN::CmpPredicate::LE:
    replacementValue = operands[0].le(operands[1]);
    break;
  case KGEN::CmpPredicate::LT:
    replacementValue = operands[0].lt(operands[1]);
    break;
  }
  state.setMojoValueFor(op.getResult(), state.builder.var(replacementValue));

  return success();
}

LogicalResult emitMojo(KGEN::POP::RemOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] % operands[1]));
  return success();
}

LogicalResult emitMojo(KGEN::POP::SelectOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  auto primDecl = state.registry.lookupPrimitive(op.getOperationName());
  // sym_name on PrimitiveDeclOp is the fully-qualified Mojo source name.
  MojoValue selectFunc(primDecl.getSymNameAttr().strref());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(selectFunc({}, operands)));
  return success();
}

LogicalResult emitMojo(KGEN::POP::ShlOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] << operands[1]));
  return success();
}

LogicalResult emitMojo(KGEN::POP::ShrOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(operands[0] >> operands[1]));
  return success();
}

LogicalResult emitMojo(KGEN::POP::MinOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  static const MojoValue minFunc = "min";
  state.setMojoValueFor(op.getResult(), state.builder.var(minFunc(
                                            {}, {operands[0], operands[1]})));
  return success();
}

LogicalResult emitMojo(KGEN::POP::MaxOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  static const MojoValue maxFunc = kFnMax;
  state.setMojoValueFor(op.getResult(), state.builder.var(maxFunc(
                                            {}, {operands[0], operands[1]})));
  return success();
}

LogicalResult emitMojo(KGEN::POP::OrOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(
      op.getResult(),
      state.builder.var(operands[0].bitwise_logical_or(operands[1])));
  return success();
}

LogicalResult emitMojo(KGEN::POP::XOrOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(
      op.getResult(),
      state.builder.var(operands[0].bitwise_logical_xor(operands[1])));
  return success();
}

LogicalResult emitMojo(KGEN::POP::SIMDAndOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(
      op.getResult(),
      state.builder.var(operands[0].bitwise_logical_and(operands[1])));
  return success();
}

LogicalResult emitMojo(KGEN::POP::SIMDSelectOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  auto primDecl = state.registry.lookupPrimitive(op.getOperationName());
  // sym_name on PrimitiveDeclOp is the fully-qualified Mojo source name.
  MojoValue selectFunc(primDecl.getSymNameAttr().strref());
  state.setMojoValueFor(op.getResult(),
                        state.builder.var(selectFunc({}, operands)));
  return success();
}

LogicalResult emitMojo(KGEN::POP::SIMDOrOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(
      op.getResult(),
      state.builder.var(operands[0].bitwise_logical_or(operands[1])));
  return success();
}

LogicalResult emitMojo(KGEN::POP::SIMDXOrOp op, EmitMojoState &state) {
  auto operands = state.getMojoValuesFor(op->getOperands());
  state.setMojoValueFor(
      op.getResult(),
      state.builder.var(operands[0].bitwise_logical_xor(operands[1])));
  return success();
}

//===----------------------------------------------------------------------===//
// KGEN::ParamConstantOp
//===----------------------------------------------------------------------===//

LogicalResult emitMojo(KGEN::ParamConstantOp op, EmitMojoState &state) {
  if (auto vals = dyn_cast<KGEN::SIMDAttr>(op.getValue())) {
    ASSERT_STREAM(vals, << "Expected a SIMD parameter constant");
    ASSERT_STREAM(vals.getValues().size() == 1,
                  << "Only scalars are supported");
    KGEN::DTypeValue val = vals.getValues().front();
    KGEN::KGENDType valDType = val.getDType();
    if (valDType.isBool())
      state.setMojoValueFor(op.getResult(), val.getBoolVal());
    else
      state.setMojoValueFor(op.getResult(), val.getIntVal());
  } else if (auto val = dyn_cast<IntegerAttr>(op.getValue())) {
    auto apInt = val.getValue();
    if (val.getValue().getBitWidth() == 1) {
      state.setMojoValueFor(op.getResult(), apInt.getBoolValue());
    } else {
      state.setMojoValueFor(
          op.getResult(),
          APSInt(apInt, cast<IntegerType>(val.getType()).isSigned()));
    }
  } else {
    ASSERT_STREAM(false, << "Unsupported param constant type "
                         << op.getValue().getType());
    llvm_unreachable("Unsupported mlir type");
  }

  return success();
}

/// Core parameter-inference logic shared by all inferKernelParameters
/// overloads.
static ErrorOr<EmitMojoState::InferredParameters>
inferParameters(ArrayRef<mlir::Attribute> fnLevelParams,
                ArrayRef<mlir::Attribute> structLevelParams,
                mlir::ArrayAttr inputParamMappings,
                mlir::DictionaryAttr defaultAttrs,
                KGEN::LIT::FnTypeGeneratorType sig,
                std::function<ErrorOr<MojoValue>(StringRef)> &inferCallback) {

  // Collect the set of params inferable from inputs by walking the function's
  // argument types directly. A param is inferable when an argument type
  // references it -- either as a named `ParamDeclRefAttr` or as a depth-0
  // `ParamIndexRefAttr` (the form elaborated bytecode uses for bound
  // parameter values).
  //
  // For IndexRefs, `*(0, K)` indexes into the generator's *full* parameter
  // list, which Mojo lays out as `structLevelParams + fnLevelParams` (struct
  // params first). `fnLevelParams` alone (from
  // `KernelFunctionAttr::params` / `FnOp::getInputParams()`) is just the
  // fn-level subset, so we build a merged lookup table to recover the right
  // name. A wrong lookup would silently misreport e.g. "freq_dtype" as
  // inferable when the kernel really references "dtype" -- breaking parameter
  // inference for kernels with both struct-level and fn-level params.
  llvm::SmallVector<mlir::Attribute> mergedParams;
  mergedParams.reserve(structLevelParams.size() + fnLevelParams.size());
  mergedParams.append(structLevelParams.begin(), structLevelParams.end());
  mergedParams.append(fnLevelParams.begin(), fnLevelParams.end());

  llvm::DenseSet<StringRef> inferByInputs;
  std::function<void(mlir::Attribute)> walkAttr;
  std::function<void(mlir::Type)> walkType;
  walkAttr = [&](mlir::Attribute attr) {
    if (!attr)
      return;
    if (auto sugar = dyn_cast<KGEN::SugarAttr>(attr)) {
      walkAttr(sugar.getCanonical());
      return;
    }
    if (auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(attr)) {
      inferByInputs.insert(declRef.getName().getValue());
      return;
    }
    if (auto indexRef = dyn_cast<KGEN::ParamIndexRefAttr>(attr)) {
      // Depth-0 references the fn's enclosing generator scope, which is the
      // merged `structLevelParams + fnLevelParams` list.
      if (indexRef.getDepth() == 0 &&
          indexRef.getIndex() < mergedParams.size()) {
        auto pd = cast<KGEN::ParamDeclAttr>(mergedParams[indexRef.getIndex()]);
        inferByInputs.insert(pd.getName().getValue());
      }
      return;
    }
    attr.walkImmediateSubElements([&](mlir::Attribute a) { walkAttr(a); },
                                  [&](mlir::Type t) { walkType(t); });
  };
  walkType = [&](mlir::Type type) {
    if (!type)
      return;
    type.walkImmediateSubElements([&](mlir::Attribute a) { walkAttr(a); },
                                  [&](mlir::Type t) { walkType(t); });
  };
  for (size_t i = 0, n = sig.getNumArguments(); i < n; ++i)
    walkType(sig.getArgument(i));

  // Remove result-only parameters from the inferByInputs set.
  removeResultOnlyParams(sig, inferByInputs);

  auto isIOSpecType = [](Type ty) {
    auto structType = dyn_cast<KGEN::LIT::StructType>(ty);
    return structType &&
           isMojoIOSpecTypeName(structType.getSymbol().getNestedReferences());
  };

  auto inferParametersInner =
      [&](ArrayRef<mlir::Attribute> params,
          SmallVector<MojoValue> &inferredParams) -> ErrorOrSuccess {
    for (auto paramAttr : params) {
      auto param = cast<KGEN::ParamDeclAttr>(paramAttr);
      StringRef paramName = param.getName().getValue();

      if (isa<KGEN::LIT::OriginType>(param.getType()) ||
          isIOSpecType(param.getType()))
        continue;
      if (paramName.contains("."))
        continue;

      ErrorOr<MojoValue> maybeVal = inferCallback(paramName);
      if (inferByInputs.contains(paramName) && maybeVal.isError())
        continue;

      if (maybeVal.isError()) {
        if (defaultAttrs && defaultAttrs.contains(paramName))
          continue;
        return maybeVal.takeError();
      }
      inferredParams.push_back(MojoValue(paramName).equal(*maybeVal));
    }
    return success();
  };

  EmitMojoState::InferredParameters ret;
  auto errOr = inferParametersInner(fnLevelParams, ret.functionParameters);
  if (errOr.isError())
    return errOr.takeError();
  errOr = inferParametersInner(structLevelParams, ret.structParameters);
  if (errOr.isError())
    return errOr.takeError();
  return ret;
}

std::optional<StringRef>
EmitMojoState::getResultParamNameFor(const KernelDeclFunctionAdaptor &fnAdaptor,
                                     StringRef paramName) {
  if (!fnAdaptor.attr.getResultParams())
    return std::nullopt;
  // Each entry in resultParamMappings corresponds to one result type. It is a
  // DictionaryAttr {struct-param-name: value} if the result is a struct whose
  // declaration was visible to ImportPackages; otherwise it is a UnitAttr
  // placeholder. The value at `paramName` is a ParamDeclRefAttr only when the
  // struct parameter is bound to a free function/struct parameter (i.e. the
  // dtype is unresolved). If the value is concrete (e.g. DTypeConstantAttr),
  // there is no name to return.
  for (auto resultParam : fnAdaptor.attr.getResultParams()) {
    auto resultParamDict = dyn_cast<DictionaryAttr>(resultParam);
    if (!resultParamDict)
      continue;
    auto attr = resultParamDict.get(paramName);
    if (auto funcParamName = dyn_cast_or_null<KGEN::ParamDeclRefAttr>(attr))
      return funcParamName.getName();
  }
  return std::nullopt;
}

// Build a {param-name -> UnitAttr} dictionary listing fn-level params with
// a declared default value. Sourced from the function's `PogListAttr`
// metadata; needed by `inferParameters` to know which missing inferences
// are tolerable.
static mlir::DictionaryAttr
computeDefaultParamsDict(KernelDeclFunctionAdaptor *fnAdaptor) {
  if (!fnAdaptor || !fnAdaptor->attr.getParams() ||
      fnAdaptor->attr.getParams().empty())
    return nullptr;
  auto fnParams = fnAdaptor->attr.getParams().getValue();
  KGEN::PogListAttr pogs = fnAdaptor->attr.getSignature().getParamListAttrs();
  size_t numImplicit =
      fnAdaptor->attr.getSignature().getNumImplicitOriginDecls();
  size_t totalPogs = pogs.getPogs().size();
  // Fn-level params are at the end of the merged pog list (struct-level
  // params come first).
  size_t fnStart =
      totalPogs >= fnParams.size() ? totalPogs - fnParams.size() : 0;
  size_t fnEnd =
      fnParams.size() > numImplicit ? fnParams.size() - numImplicit : 0;

  mlir::Builder builder(fnAdaptor->attr.getSignature().getContext());
  mlir::NamedAttrList defaultNames;
  for (size_t fnIdx = 0; fnIdx < fnEnd; ++fnIdx) {
    size_t pogIdx = fnStart + fnIdx;
    if (pogIdx >= totalPogs || !pogs.getDefault(pogIdx))
      continue;
    auto paramDecl = mlir::cast<KGEN::ParamDeclAttr>(fnParams[fnIdx]);
    defaultNames.append(
        mlir::StringAttr::get(
            fnAdaptor->attr.getSignature().getContext(),
            KGEN::demangleParameterName(paramDecl.getName().strref())),
        builder.getUnitAttr());
  }
  if (defaultNames.empty())
    return nullptr;
  return mlir::DictionaryAttr::get(fnAdaptor->attr.getSignature().getContext(),
                                   defaultNames.getAttrs());
}

ErrorOr<EmitMojoState::InferredParameters> EmitMojoState::inferKernelParameters(
    KernelDeclAdaptor &adaptor, llvm::StringRef funcLabel,
    std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback) {
  KernelDeclFunctionAdaptor *fnAdaptor =
      const_cast<KernelDeclFunctionAdaptor *>(
          adaptor.getFunctionByLabel(funcLabel));
  ASSERT_STREAM(fnAdaptor,
                "Function label '" << funcLabel << "' not found in adaptor");

  // Pull fn-level params from the function adaptor itself. Each variant
  // (execute / elementwise / shape / update_view) keeps its own param list
  // because their signatures can differ (e.g. elementwise typically omits
  // `target`/`_trace_name`). For an elementwise label that falls back to the
  // synthesised-from-elementwise path, the adaptor's `params` already reflects
  // the elementwise variant.
  ArrayRef<mlir::Attribute> fnParams;
  if (fnAdaptor->attr.getParams())
    fnParams = fnAdaptor->attr.getParams().getValue();

  ArrayRef<mlir::Attribute> structParams;
  if (adaptor.structParams)
    structParams = adaptor.structParams.getValue();

  // Default-params dict is computed from the *execute* fn for parity with
  // the WIP behavior; defaults are declared on the execute signature.
  mlir::DictionaryAttr defaultParams = computeDefaultParamsDict(
      adaptor.executeFunction ? &*adaptor.executeFunction : nullptr);
  return inferParameters(fnParams, structParams, fnAdaptor->attr.getArgParams(),
                         defaultParams, fnAdaptor->attr.getSignature(),
                         inferCallback);
}

ErrorOr<EmitMojoState::InferredParameters> EmitMojoState::inferKernelParameters(
    PrimitiveDeclOp primDecl,
    std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback) {
  // Primitives have no struct-level params. arg-param mappings let
  // inferParameters tolerate failures for parameters that are inferable from
  // input argument types at the Mojo call site.
  MOGG::KernelFunctionAttr fn = primDecl.getFunction();
  return inferParameters(fn.getParams().getValue(),
                         /*structLevelParams=*/{}, fn.getArgParams(),
                         /*defaultAttrs=*/nullptr, fn.getSignature(),
                         inferCallback);
}

ErrorOr<EmitMojoState::InferredParameters> EmitMojoState::inferKernelParameters(
    ShapeFunctionDeclOp shapeDecl,
    std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback) {
  // Standalone shape functions have no struct-level params.
  MOGG::KernelFunctionAttr fn = shapeDecl.getFunction();
  return inferParameters(fn.getParams().getValue(),
                         /*structLevelParams=*/{}, fn.getArgParams(),
                         /*defaultAttrs=*/nullptr, fn.getSignature(),
                         inferCallback);
}

// Returns the trace name from fusionInfo, or empty string if not set.
static MojoValue getTraceNameValue(FusionInfoAttr fusionInfo) {
  // GEX-979: pass in trace description for fused kernel without
  // invalidating cache.
  if (!fusionInfo)
    return MojoValue("").quote();
  return MojoValue(fusionInfo.toString()).quote();
}

static MojoValue getHasEpilogueFusionValue(mlir::Operation *kernelCallOp) {
  // The kernel call op may be nested inside a WithTraceOp (MOGG ABI), so
  // BindLambdaOps may live in a parent block. Walk up the block hierarchy
  // (stopping at KernelStubOp) until we find BindLambdaOps.
  auto hasFusion = [](Block *block) {
    auto bindOps = block->getOps<MOGG::BindLambdaOp>();
    // Fused in affine view does not set the flag.
    //
    // TODO: we should come up with a better way, it works because current
    // affine view fusion is very limited and it does not change the contiguity
    // of the buffer, which might not be the case after treating all view
    // operations as index transform.
    return llvm::any_of(bindOps, [](MOGG::BindLambdaOp bindOp) {
      return llvm::any_of(bindOp.getLambda().getOps(), [](Operation &op) {
        return isa<MOGG::CallElementwiseOp>(&op);
      });
    });
  };

  Block *block = kernelCallOp->getBlock();
  while (block) {
    if (!block->getOps<MOGG::BindLambdaOp>().empty())
      return hasFusion(block);
    auto *parentOp = block->getParentOp();
    if (!parentOp || isa<MOGG::KernelStubOp>(parentOp))
      break;
    block = parentOp->getBlock();
  }
  return false;
}

static MojoValue getHasPrologueFusionValue(mlir::Operation *callOp) {
  auto kernelStubOp = callOp->getParentOfType<KernelStubOp>();
  if (!kernelStubOp)
    return true;
  FusionInfoAttr fusionInfo = kernelStubOp.getFusionInfoAttr();
  if (!fusionInfo)
    return true;
  return !mlir::isa<FusionLeafAttr>(fusionInfo.getRoot());
}

ErrorOr<MojoValue> EmitMojoState::inferDefaultKernelParamValue(
    Operation *callOp, StringRef paramName, DictionaryAttr customParamDict) {

  auto inferFromMlirParam = [&](Attribute customParam) -> ErrorOr<MojoValue> {
    // Convert it to mojo
    // TODO: Convert custom mlir parameter to the right mojo parameter type is
    // hardcoded here and brittle. Besides the param name, we would also need
    // the type from the kernel infos to convert it correctly.
    if (auto strAttr = dyn_cast<StringAttr>(customParam))
      return MojoValue(strAttr.getValue()).quote();
    else if (auto boolAttr = dyn_cast<BoolAttr>(customParam))
      return boolAttr.getValue();
    else if (auto intAttr = dyn_cast<IntegerAttr>(customParam))
      // Use getValue().getSExtValue() instead of getInt() because getInt()
      // asserts that the type must be signless or index, but we may have
      // signed integer types (like si32).
      return intAttr.getValue().getSExtValue();
    else if (auto dtypeAttr = dyn_cast<KGEN::DTypeConstantAttr>(customParam))
      return EmitMojoState::mlirDTypeToMojoType(dtypeAttr.getDType());
    else if (auto arrAttr = dyn_cast<ArrayAttr>(customParam);
             arrAttr && !arrAttr.empty() && isa<IntegerAttr>(arrAttr[0])) {
      SmallVector<int64_t> values;
      for (auto val : arrAttr)
        values.push_back(cast<IntegerAttr>(val).getValue().getSExtValue());
      return EmitMojoState::getAsIntTupleMojoValue(values);
    } else
      return Error(llvm::formatv("Unsupported custom param MLIR type for `{}`",
                                 paramName));
  };

  // Look at the provided list of kernel parameters first
  if (auto kernelCallOp = dyn_cast<KernelCallInterface>(callOp);
      kernelCallOp && kernelCallOp.getKernelParams()) {
    if (auto customParam = kernelCallOp.getKernelParams().get(paramName))
      return inferFromMlirParam(customParam);
  }

  if (customParamDict) {
    if (auto customParam = customParamDict.get(paramName))
      return inferFromMlirParam(customParam);
  }

  // Some special parameters are common to all kernels.
  /// TODO(GEX-2847): Do not check it on the KernelStubOp but instead directly
  /// on the op itself in case these get sequence fused. Today, this won't
  /// happen because we outline the stubs before sequence fusion and the ones
  /// that do get sequence fused will always be on the host and have matching
  /// devices.
  if (paramName == MOGG::kMoggTargetParameterName) {
    auto kernelStubOp = callOp->getParentOfType<KernelStubOp>();
    return MojoValue(MO::getDeviceOrDefaultAny(kernelStubOp).getLabel())
        .quote();
  }
  if (paramName == MOGG::kMoggTraceNameParameterName) {
    auto kernelStubOp = callOp->getParentOfType<KernelStubOp>();
    return getTraceNameValue(kernelStubOp.getFusionInfoAttr());
  }
  if (paramName == MOGG::kMoggHasEpilogueFusionParameterName)
    return getHasEpilogueFusionValue(callOp);
  if (paramName == MOGG::kMoggHasPrologueFusionParameterName)
    return getHasPrologueFusionValue(callOp);

  return Error(llvm::formatv("Failed to infer parameter `{}`", paramName));
}

MojoValue EmitMojoState::getShapeAsIndexListMojoValue(MOSH::ShapeAttr shapeAttr,
                                                      ValueRange dynamicDims) {
  SmallVector<MojoValue> mojoDims;
  unsigned nextDymDim = 0;
  for (TypedAttr dim : shapeAttr.getValues()) {
    if (auto staticDim = MOSH::ShapeAttr::getStaticDim(dim)) {
      mojoDims.push_back(*staticDim);
    } else {
      ASSERT_STREAM(nextDymDim < dynamicDims.size(),
                    "Missing dynamic dimension");
      mojoDims.push_back(getMojoValueFor(dynamicDims[nextDymDim++]));
    }
  }
  if (mojoDims.empty())
    mojoDims.push_back(MojoValue(static_cast<int64_t>(1)));

  MojoValue rankParam =
      MojoValue(shapeAttr.getRank() == 0 ? 1 : shapeAttr.getRank());

  return MojoValue(kStructIndexList)({rankParam}, mojoDims);
}

MojoValue EmitMojoState::getShapeAsCoordMojoValue(MOSH::ShapeAttr shapeAttr,
                                                  ValueRange dynamicDims) {
  SmallVector<MojoValue> coordElems;
  unsigned nextDynDim = 0;
  for (TypedAttr dim : shapeAttr.getValues()) {
    if (auto staticDim = MOSH::ShapeAttr::getStaticDim(dim)) {
      // Static dim: compile-time `Idx[N]` component.
      coordElems.push_back(MojoValue(kAliasIdx)[*staticDim]);
    } else {
      // Dynamic dim: runtime `Scalar[DType.int](<dim>)` component. The dynamic
      // element type must be Scalar[DType.int] (not Int) so it matches the
      // element-type the ManagedTensorSlice layout machinery rebinds to.
      ASSERT_STREAM(nextDynDim < dynamicDims.size(),
                    "Missing dynamic dimension");
      MojoValue scalarInt =
          MojoValue(kStructScalar)[MojoValue(kStructDType).get(kEnumDTypeInt)];
      coordElems.push_back(
          scalarInt({}, {getMojoValueFor(dynamicDims[nextDynDim++])}));
    }
  }
  // Rank-0 tensors are represented as rank-1 with a single static `1`.
  if (coordElems.empty())
    coordElems.push_back(
        MojoValue(kAliasIdx)[MojoValue(static_cast<int64_t>(1))]);

  return MojoValue(kStructCoord)({}, coordElems);
}

void EmitMojoState::addDebugPrintTensorSpecs(const MojoValue &tensorValue,
                                             const MojoValue &indicesValue,
                                             bool isInput,
                                             unsigned operandIndex) {
  if (!enableDebugDumpFusedTensorSpecs)
    return;

  // We check if all indices are 0 to only print once.
  ManagedTensorSliceRef mts(tensorValue);
  builder.if_(MojoValue(kFnAllZeros)({}, indicesValue), [&]() {
    builder.call(
        kFnPrint, {},
        {MojoValue(isInput ? kParameterInput : kParameterOutput).quote(),
         MojoValue("#" + std::to_string(operandIndex) + ":").quote(),
         MojoValue("shape =").quote(), mts.staticShapeTuple(),
         MojoValue("strides =").quote(), mts.staticStridesTuple(),
         MojoValue("alignment =").quote(), mts.alignment()});
  });
}

void EmitMojoState::addDebugPrintLambdaParam(MOGG::BindLambdaKind lambdaKind,
                                             const MojoValue &widthParam,
                                             const MojoValue &alignmentParam,
                                             const MojoValue &indicesValue) {
  if (!enableDebugDumpFusedTensorSpecs)
    return;

  // We check if all indices are 0 to only print once.
  builder.if_(MojoValue(kFnAllZeros)({}, indicesValue), [&]() {
    builder.call(kFnPrint, {},
                 {
                     MojoValue(stringifyBindLambdaKind(lambdaKind)).quote(),
                     MojoValue("width =").quote(),
                     widthParam,
                     MojoValue("alignment =").quote(),
                     alignmentParam,
                 });
  });
}

MojoValue EmitMojoState::mlirOpaqueTypeToMojoType(MO::OpaqueType opaqueType) {
  // ImportPackagesPass rewrites !mo.opaque<"Leaf"> to !mo.opaque<"a.b.Leaf">
  // (dotted Mojo qualified name), so the symbol can be used directly as a
  // Mojo identifier with no further rewriting.
  return MojoValue(opaqueType.getSymbol().getValue());
}

SmallVector<MojoValue>
EmitMojoState::getOpaqueTypeParameters(MO::OpaqueType opaqueType) {
  SmallVector<MojoValue> params;
  DictionaryAttr paramDict = opaqueType.getParameters();
  if (!paramDict || paramDict.empty())
    return params;

  for (NamedAttribute param : paramDict) {
    // Emit named parameters like SIMDPair[S0=4, S1=8].
    StringRef name = param.getName().getValue();
    std::string valueStr;
    if (auto intAttr = dyn_cast<IntegerAttr>(param.getValue())) {
      // Use getValue().getSExtValue() instead of getInt() because getInt()
      // asserts that the type must be signless or index, but we may have
      // signed integer types (like si32).
      valueStr = std::to_string(intAttr.getValue().getSExtValue());
    } else {
      // Fallback: use the attribute's string representation.
      llvm::raw_string_ostream ss(valueStr);
      param.getValue().print(ss);
    }
    params.push_back(MojoValue(name.str() + "=" + valueStr));
  }
  return params;
}

} // namespace M::MOGG
