//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Interfaces/LoweringState.h"

using namespace M;
using namespace M::MGP;

//===----------------------------------------------------------------------===//
// MOGraphLoweringState
//===----------------------------------------------------------------------===//

MGP::CpuTensorSpecDimsDynamicOp
MOGraphLoweringState::getDynamicDims(Value mgpTensor, Location loc) {
  OpBuilder &exeBuilder = getExecuteOpBuilder();
  auto extractOp = exeBuilder.create<MGP::CpuTensorExtractOp>(loc, mgpTensor);
  TypedValue<MGP::CpuTensorSpecType> spec = extractOp.getTensorSpecResult();
  return exeBuilder.create<MGP::CpuTensorSpecDimsDynamicOp>(
      loc, spec, spec.getType().getRawDims());
}

MGP::CpuTensorExtractShapeOp MOGraphLoweringState::getShape(Value mgpTensor,
                                                            Location loc) {
  OpBuilder &exeBuilder = getExecuteOpBuilder();
  return exeBuilder.create<MGP::CpuTensorExtractShapeOp>(loc, mgpTensor);
}

ErrorOr<bool> MOGraphLoweringState::lowerArg(BlockArgument &inArg,
                                             BlockArgument &outArg) {
  bool hasParams = false;
  Location loc = inArg.getLoc();
  if (auto inTensorArg = dyn_cast<MO::TensorValue>(inArg)) {
    MO::TensorType argType = inTensorArg.getType();
    if (argType.hasStaticShape()) {
      // No parameters to deal with.
    } else if (argType.hasUnknownShape()) {
      // Also no parameters to deal with, however this argument won't be
      // usable by any MO ops.
    } else if (auto inShapeParam =
                   dyn_cast<KGEN::ParamDeclRefAttr>(argType.getShapeAttr())) {
      // Argument has dynamic rank. The parameter needs to be represented by
      // the runtime MGP 'shape' of the tensor, ie a tensor_spec with unknown
      // dtype. However, we only need to capture the mapping for the first
      // occurrence of the parameter.
      hasParams = true;
      if (!params.lookupLowered(inShapeParam))
        params.emplaceLowered(inShapeParam, getShape(outArg, loc));
    } else {
      // Argument has at least one dynamic dimension. The parameters need to
      // be represented by the corresponding runtime dimensions. Since
      // parameters may be shared we only update the parameter map for the
      // first occurrence of each parameter. However, since dimensions are
      // extract as a group, we always do the extraction, and rely on DCE to
      // cleanup if all the parameters turn out to have already been mapped.
      hasParams = true;
      MGP::CpuTensorSpecDimsDynamicOp dynDimsOp = getDynamicDims(outArg, loc);
      // CpuTensorSpecDimsDynamicOp returns indexes only for dimensions
      // which are unknown at compile time. So build the corresponding
      // vector of parameters. Note we do not support arbitrary expressions in
      // the MO shape signature.
      SmallVector<KGEN::ParamDeclRefAttr> inDimParams;
      for (TypedAttr inDim : argType.getDims()) {
        if (auto inDimParam = dyn_cast<KGEN::ParamDeclRefAttr>(inDim))
          inDimParams.emplace_back(inDimParam);
      }
      if (dynDimsOp->getNumResults() != inDimParams.size()) {
        // TODO(#14531) handle param expr
        return Error("Expressions in parameter arguments are not supported");
      }
      for (auto [inDimParam, outDim] :
           llvm::zip(inDimParams, dynDimsOp->getResults()))
        // Only update the parameter mapping for the first parameter
        // occurrence.
        if (!params.lookupLowered(inDimParam))
          params.emplaceLowered(inDimParam, outDim);
    }
  }
  return hasParams;
}

ErrorOr<Value> MOGraphLoweringState::lowerExpression(Location loc,
                                                     TypedAttr attr) {
  auto newValOr = params.lowerExpression(getExecuteOpBuilder(), loc, attr,
                                         lowerExpressionImpl);
  if (newValOr.isError())
    return newValOr.takeError();
  auto newVal = std::move(*newValOr);

  // If we got a placeholder, we need to remember it, in case we map an SSA
  // value to it.
  if (auto placeholder = newVal.getDefiningOp<ParamPlaceholderOp>())
    if (!placeholderMap.contains(placeholder))
      placeholderMap.try_emplace(placeholder, SmallVector<Value>{});
  return newVal;
}

ErrorOrSuccess
MOGraphLoweringState::lowerParamDecl(KGEN::ParamDeclAttr declAttr,
                                     Value value) {
  Value maybePlaceholder = params.lowerParamDecl(declAttr, value);

  if (maybePlaceholder) {
    // If we just resolved a placeholder, we need to make sure no one will
    // ever map to it in the future.
    if (placeholderMap.contains(maybePlaceholder)) {
      for (Value key : placeholderMap.at(maybePlaceholder)) {
        execute.loweredValues.erase(key);
        execute.loweredValues.try_emplace(key, value);
      }
      placeholderMap.erase(maybePlaceholder);
    }

    // We also need to replace all its existing uses.
    maybePlaceholder.replaceAllUsesWith(value);
    maybePlaceholder.getDefiningOp<ParamPlaceholderOp>().erase();
  }
  return success();
}

ErrorOr<SmallVector<Value>>
MOGraphLoweringState::lowerShapeAttrDynDims(MOSH::ShapeAttr shapeAttr,
                                            Location loc) {
  SmallVector<Value> runtimeDims;
  for (TypedAttr dim : shapeAttr.getValues()) {
    if (!isa<IntegerAttr>(dim)) {
      auto vOr = lowerExpression(loc, dim);
      if (vOr.isError())
        return vOr.takeError();
      runtimeDims.emplace_back(std::move(*vOr));
    }
  }
  return runtimeDims;
}

ErrorOr<StringRef> MOGraphLoweringState::pocToMopIndexOp(KGEN::POC poc) {
  // TODO: support more opcodes
  switch (poc) {
  case KGEN::POC::Add:
    return MIP::AddOp::getOperationName();
  case KGEN::POC::Div:
    return MIP::DivOp::getOperationName();
  case KGEN::POC::EQ:
    return MIP::EqualOp::getOperationName();
  case KGEN::POC::LT:
    return MIP::LessThanOp::getOperationName();
  case KGEN::POC::LE:
    return MIP::LessThanEqualOp::getOperationName();
  case KGEN::POC::Max:
    return MIP::MaxOp::getOperationName();
  case KGEN::POC::Min:
    return MIP::MinOp::getOperationName();
  case KGEN::POC::Mod:
    return MIP::ModOp::getOperationName();
  case KGEN::POC::Mul:
  case KGEN::POC::MulNuw:
    return MIP::MulOp::getOperationName();
  case KGEN::POC::And:
    return MIP::AndOp::getOperationName();
  case KGEN::POC::Or:
    return MIP::OrOp::getOperationName();
  case KGEN::POC::Xor:
    return MIP::XorOp::getOperationName();
  case KGEN::POC::Cond:
    return MIP::SelectOp::getOperationName();
  default:
    return Error(Twine("unsupported POC '") + stringifyPOC(poc) + "'");
  }
}

ErrorOr<Value>
MOGraphLoweringState::lowerParamOperator(OpBuilder &builder, Location loc,
                                         KGEN::ParamOperatorAttr opAttr,
                                         ValueRange operands) {
  auto mopNameOr = pocToMopIndexOp(opAttr.getOpcode());
  if (mopNameOr.isError())
    return mopNameOr.takeError();
  auto opNameAttr = builder.getStringAttr(*mopNameOr);
  Type resTy = opAttr.getType();

  // For non binary op
  if (opNameAttr == MIP::SelectOp::getOperationName())
    return builder.create(loc, opNameAttr, operands, {resTy})->getResult(0);

  // For potentially variadic binary ops
  //   add(a, b, c)
  //   ===>
  //   mip.add(mip.add(a, b), c)
  Value res = operands.front();
  do {
    operands = operands.drop_front();
    res = builder.create(loc, opNameAttr, {res, operands.front()}, {resTy})
              ->getResult(0);
  } while (operands.size() > 1);
  return res;
}

Operation *MOGraphLoweringState::lowerIntAttr(OpBuilder &builder, Location loc,
                                              IntegerAttr intAttr) {
  if (auto boolAttr = dyn_cast<BoolAttr>(intAttr)) {
    return builder.create<MIP::ConstantBoolOp>(loc, intAttr.getType(),
                                               boolAttr);
  }
  IntegerType si64Ty = builder.getIntegerType(64, /*isSigned=*/true);
  auto newIntAttr = builder.getIntegerAttr(si64Ty, intAttr.getInt());
  return builder.create<MIP::ConstantIndexOp>(loc, intAttr.getType(),
                                              newIntAttr);
}

ErrorOr<Value>
MOGraphLoweringState::lowerExpressionImpl(ParameterLoweringState &state,
                                          OpBuilder &builder, Location loc,
                                          TypedAttr attr, ValueRange operands) {
  Operation *newOp;
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    newOp = lowerIntAttr(builder, loc, intAttr);
  } else if (auto opAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr)) {
    return lowerParamOperator(builder, loc, opAttr, operands);
  } else if (auto opAttr = dyn_cast<KGEN::MLIROpAttr>(attr)) {
    StringAttr opName = opAttr.getName();
    if (opName != MOSH::AssertOp::getOperationName()) {
      return Error(Twine("cannot lower mlir operation '") + opName.strref() +
                   "' in parameter expression");
    }

    // We must be dealing with an assert.
    newOp = builder.create<MGP::AssertOp>(loc, operands,
                                          opAttr.getAttrs().getValue());
  } else if (auto shapeAttr = dyn_cast<MOSH::ShapeAttr>(attr)) {
    // Lower dimensions recursively.
    SmallVector<Value> dimVals;
    for (TypedAttr dim : shapeAttr.getValues()) {
      auto dimValOr =
          state.lowerExpression(builder, loc, dim, lowerExpressionImpl);
      if (dimValOr.isError())
        return dimValOr.takeError();
      dimVals.emplace_back(std::move(*dimValOr));
    }

    // There is no runtime shape type in MGP, so we map shape attributes to
    // a fully dynamic spec.
    SmallVector<std::optional<uint64_t>> dummyShape(dimVals.size());
    auto specTy =
        MGP::CpuTensorSpecType::get(loc.getContext(), dummyShape, DType::si64);
    auto specOp =
        builder.create<MGP::CpuTensorSpecCreateOp>(loc, specTy, dimVals);
    return specOp.getResult();
  } else {
    return Error("cannot lower parameter expression");
  }

  return newOp->getResult(0);
}
