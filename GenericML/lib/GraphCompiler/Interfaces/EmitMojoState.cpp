//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Interfaces/EmitMojoState.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGInterfaces.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "KGEN/POPDialect/POPOps.h"

using namespace M;

namespace M::MOGG {

/// Remove parameters from inferByInputs that appear in the function's result
/// type but not in the actual input argument types.
///
/// This walks both the user result type and all input argument types (excluding
/// byref result/error slots), collects all ParamDeclRefAttr references from
/// each, and removes from inferByInputs any parameter names that appear in the
/// result type but not in any input argument type.
static void removeResultOnlyParams(KGEN::LIT::FnOp function,
                                   llvm::DenseSet<StringRef> &inferByInputs) {
  // Collect params from result type.
  SmallVector<KGEN::ParamDeclRefAttr> resultParamRefs;
  bool hasConstExpr = false;
  Type userResultType = function.getUserResultType();
  KGEN::collectParameterReferences(userResultType, resultParamRefs,
                                   hasConstExpr);
  llvm::DenseSet<StringRef> resultParams;
  for (auto paramRef : resultParamRefs)
    resultParams.insert(paramRef.getName().getValue());

  // Collect params from actual input arguments (excluding byref result slot).
  auto sig = function.getFuncTypeGenerator().getBody();
  ArrayRef<Type> argTypes = function.getArgumentTypes();
  if (sig.hasMemoryOnlyResult())
    argTypes = argTypes.drop_back();
  if (sig.isThrows())
    argTypes = argTypes.drop_back();

  llvm::DenseSet<StringRef> inputParams;
  for (Type argType : argTypes) {
    SmallVector<KGEN::ParamDeclRefAttr> argParamRefs;
    KGEN::collectParameterReferences(argType, argParamRefs, hasConstExpr);
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
  static const MojoValue maxFunc = "max";
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
  case KGEN::POP::CmpPredicate::EQ:
    replacementValue = operands[0].eq(operands[1]);
    break;
  case KGEN::POP::CmpPredicate::NE:
    replacementValue = operands[0].neq(operands[1]);
    break;
  case KGEN::POP::CmpPredicate::GE:
    replacementValue = operands[0].ge(operands[1]);
    break;
  case KGEN::POP::CmpPredicate::GT:
    replacementValue = operands[0].gt(operands[1]);
    break;
  case KGEN::POP::CmpPredicate::LE:
    replacementValue = operands[0].le(operands[1]);
    break;
  case KGEN::POP::CmpPredicate::LT:
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
  auto selectFunc = state.getOriginalMojoNameFor(
      state.analysis.getRegisteredLITMojoKernelFor(op.getOperationName()));
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
  static const MojoValue maxFunc = "max";
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
  auto selectFunc = state.getOriginalMojoNameFor(
      state.analysis.getRegisteredLITMojoKernelFor(op.getOperationName()));
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
  if (auto vals = dyn_cast<KGEN::POP::SIMDAttr>(op.getValue())) {
    ASSERT_STREAM(vals, << "Expected a SIMD parameter constant");
    ASSERT_STREAM(vals.getValues().size() == 1,
                  << "Only scalars are supported");
    KGEN::POP::DTypeValue val = vals.getValues().front();
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

ErrorOr<EmitMojoState::InferedParameters> EmitMojoState::inferKernelParameters(
    KGEN::LIT::FnOp function,
    std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback) {
  // First get the set of all parameters that can be inferred from the inputs.
  llvm::DenseSet<StringRef> inferByInputs;

  DictionaryAttr defaultAttrs = function->getAttrOfType<DictionaryAttr>(
      KGEN::MOGGPreElab::kKernelDefaultParameterAttrName);

  if (auto inputParams = function->getAttrOfType<ArrayAttr>(
          KGEN::MOGGPreElab::kKernelValueParameterAttrName)) {
    for (auto inputParam : inputParams) {
      if (auto inputParamDict = dyn_cast<DictionaryAttr>(inputParam))
        for (auto param : inputParamDict)
          if (auto funcParamName =
                  dyn_cast<KGEN::ParamDeclRefAttr>(param.getValue())) {
            inferByInputs.insert(funcParamName.getName().getValue());
          }
    }
  }

  // TODO(GEX-2731): The annotate-kernels pass incorrectly marks result-only
  // parameters as input parameters. For example, shape functions return
  // !mogg.shape_func_result<rank>, where 'output_rank' is a parameter of the
  // result type. This gets incorrectly added to the input parameter inference
  // list by annotate-kernels, but should only be inferred from the result type.
  // Remove any such parameters that appear only in the result type but not in
  // actual input arguments. The proper fix would be to correct
  // annotate-kernels.
  removeResultOnlyParams(function, inferByInputs);

  auto isIOSpecType = [](Type ty) {
    auto structType = dyn_cast<KGEN::LIT::StructType>(ty);
    return structType && HardcodedMojoSymbols::isMojoIOSpecTypeName(
                             structType.getSymbol().getNestedReferences());
  };

  // Now go through all params and infer those that need to be inferred.
  auto inferParametersInner =
      [&](ArrayRef<ParamDeclAttr> params,
          SmallVector<MojoValue> &inferredParams) -> ErrorOrSuccess {
    for (auto param : params) {
      StringRef paramName = param.getName().getValue();

      // TODO(GEX-2623): IOSpec / origin parameters are currently removed from
      // the parameters list because in KGEN-land empty-strust params are
      // removed during LITToKGEN. But they are still present in EmitMojo path
      // (and would cause an error: failed to infer parameter). Instead just
      // skip them for now, they'll be inferred automatically anyway. This is
      // quite brittle as it assumes all IOSpec / origin are always inferred
      // (which is true fortunately today). This test could be removed once
      // EmitMojo is complete.
      if (isa<KGEN::LIT::OriginType>(param.getType()) ||
          isIOSpecType(param.getType()))
        continue;

      // For parameters that can be inferred from inputs (e.g., from tensor
      // types), we still try to get explicit values first. This is important
      // for opaque types where Mojo can't infer the parameter from type
      // arguments like SIMDPair[P0, _]. If we have an explicit value in
      // kernel_param, use it.
      ErrorOr<MojoValue> maybeVal = inferCallback(paramName);

      // If this parameter can be inferred from inputs AND we don't have an
      // explicit value, skip it and let Mojo infer.
      if (inferByInputs.contains(paramName) && maybeVal.isError())
        continue;

      if (maybeVal.isError()) {
        if (defaultAttrs && defaultAttrs.contains(paramName)) {
          // If there is already a default value we can skip it.
          // TODO: It might be worth making the distinction between "we didn't
          // found a parameter" vs we found it but the logic to infer its value
          // failed. In the latter case we should still emit an error.
          continue;
        }
        function.emitError(Twine("Failed to infer kernel parameter `") +
                           paramName + "`: " + maybeVal.getError());
        return maybeVal.takeError();
      }

      inferredParams.push_back(MojoValue(paramName).equal(*maybeVal));
    }

    return success();
  };

  EmitMojoState::InferedParameters ret;
  auto errOr =
      inferParametersInner(function.getInputParams(), ret.functionParameters);
  if (errOr.isError())
    return errOr.takeError();

  auto structDeclOp = function->getParentOfType<KGEN::LIT::StructDeclOp>();
  if (!structDeclOp)
    return ret;

  errOr =
      inferParametersInner(structDeclOp.getInputParams(), ret.structParameters);
  if (errOr.isError())
    return errOr.takeError();

  return ret;
}

// Returns the trace name from fusionInfo, or empty string if not set.
static MojoValue getTraceNameValue(FusionInfoAttr fusionInfo) {
  // GEX-979: pass in trace description for fused kernel without
  // invalidating cache.
  if (!fusionInfo)
    return MojoValue("").quote();
  return MojoValue(fusionInfo.toString()).quote();
}

static MojoValue getLambasHaveFusionValue(mlir::Operation *kernelCallOp) {
  auto bindOps = kernelCallOp->getBlock()->getOps<MOGG::BindLambdaOp>();
  return llvm::any_of(bindOps, [](MOGG::BindLambdaOp bindOp) {
    // Fused in affine view does not set the flag.
    //
    // TODO: we should come up with a better way, it works because current
    // affine view fusion is very limited and it does not change the contiguity
    // of the buffer, which might not be the case after treating all view
    // operations as index transform.
    return llvm::any_of(bindOp.getLambda().getOps(), [](Operation &op) {
      return isa<MOGG::CallElementwiseOp>(&op);
    });
  });
}

ErrorOr<MojoValue> EmitMojoState::inferDefaultKernelParamValue(
    Operation *callOp, StringRef paramName, DictionaryAttr customParamDict) {

  auto inferFromMliParam = [](Attribute customParam) -> ErrorOr<MojoValue> {
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
      return EmitMojoState::getAsDimListMojoValue(values);
    } else
      return Error("Unsupported custom param MLIR type");
  };

  // Look at the provided list of kernel parameters first
  if (auto kernelCallOp = dyn_cast<KernelCallInterface>(callOp);
      kernelCallOp && kernelCallOp.getKernelParams()) {
    if (auto customParam = kernelCallOp.getKernelParams().get(paramName))
      return inferFromMliParam(customParam);
  }

  if (customParamDict) {
    if (auto customParam = customParamDict.get(paramName))
      return inferFromMliParam(customParam);
  }

  // Some special parameters are common to all kernels.
  /// TODO(GEX-2847): Do not check it on the KernelStubOp but instead directly
  /// on the op itself in case these get sequence fused. Today, this won't
  /// happen because we outline the stubs before sequence fusion and the ones
  /// that do get sequence fused will always be on the host and have matching
  /// devices.
  if (paramName == KGEN::MOGGPreElab::kMOGGTargetParameterName) {
    auto kernelStubOp = callOp->getParentOfType<KernelStubOp>();
    return MojoValue(MO::getDeviceOrDefaultAny(kernelStubOp).getLabel())
        .quote();
  }
  if (paramName == KGEN::MOGGPreElab::kMOGGTraceNameParameterName) {
    auto kernelStubOp = callOp->getParentOfType<KernelStubOp>();
    return getTraceNameValue(kernelStubOp.getFusionInfoAttr());
  }
  if (paramName == KGEN::MOGGPreElab::kMOGGLambdasHaveFusionParameterName)
    return getLambasHaveFusionValue(callOp);

  return Error("Failed to infer parameter");
}

MojoValue EmitMojoState::getShapeAsIndexListMojoValue(MOSH::ShapeAttr shapeAttr,
                                                      ValueRange dynamicDims) {
  SmallVector<MojoValue> mojoDims;
  unsigned nextDymDim = 0;
  for (TypedAttr dim : shapeAttr.getValues()) {
    if (auto staticDim = dyn_cast<IntegerAttr>(dim)) {
      mojoDims.push_back(MojoValue(staticDim.getSInt()));
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

  return MojoValue(HardcodedMojoSymbols::struct_IndexList)({rankParam},
                                                           mojoDims);
}

void EmitMojoState::addDebugPrintTensorSpecs(const MojoValue &tensorValue,
                                             const MojoValue &indicesValue,
                                             bool isInput,
                                             unsigned operandIndex) {
  if (!enableDebugDumpFusedTensorSpecs)
    return;

  // We check if all indices are 0 to only print once.
  builder.if_(
      MojoValue(HardcodedMojoSymbols::func_all_zeros)({}, indicesValue), [&]() {
        builder.call(
            "print", {},
            {MojoValue(isInput ? "input" : "output").quote(),
             MojoValue("#" + std::to_string(operandIndex) + ":").quote(),
             MojoValue("shape =").quote(), tensorValue.get("_static_shape"),
             MojoValue("strides =").quote(), tensorValue.get("_static_strides"),
             MojoValue("alignment =").quote(), tensorValue.get("alignment")});
      });
}

void EmitMojoState::addDebugPrintLambdaParam(MOGG::BindLambdaKind lambdaKind,
                                             const MojoValue &widthParam,
                                             const MojoValue &alignmentParam,
                                             const MojoValue &indicesValue) {
  if (!enableDebugDumpFusedTensorSpecs)
    return;

  // We check if all indices are 0 to only print once.
  builder.if_(
      MojoValue(HardcodedMojoSymbols::func_all_zeros)({}, indicesValue), [&]() {
        builder.call("print", {},
                     {
                         MojoValue(stringifyBindLambdaKind(lambdaKind)).quote(),
                         MojoValue("width =").quote(),
                         widthParam,
                         MojoValue("alignment =").quote(),
                         alignmentParam,
                     });
      });
}

MojoValue
EmitMojoState::mlirOpaqueTypeToMojoType(MO::OpaqueType opaqueType) const {
  StringRef symbol = opaqueType.getSymbol().getValue();

  // Look up the fully qualified type name from the library analysis.
  // This mapping was built by walking all struct declarations in the loaded
  // kernel library modules (MOGGKernelAPI and custom kernel libraries).
  if (auto fullyQualifiedName =
          analysis.getFullyQualifiedOpaqueTypeName(symbol)) {
    return MojoValue(fullyQualifiedName->str());
  }

  // If not found, the type is not defined in any loaded kernel library.
  // This is likely a configuration error - the kernel library defining
  // this type should be included via kernel-library-paths.
  ASSERT_STREAM(false, "Opaque type '" << symbol
                                       << "' not found in any loaded kernel "
                                          "library. Ensure the library "
                                          "defining this type is included.");
  return MojoValue(symbol.str());
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
