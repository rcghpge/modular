//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Interfaces/LoweringState.h"

using namespace M;
using namespace M::MGP;

//===----------------------------------------------------------------------===//
// Bufferizer
//===----------------------------------------------------------------------===//

BufferizedTensor Bufferizer::extract(OpBuilder &builder, Location loc,
                                     Value loweredTensor) {
  auto extractOp = builder.create<MGP::CpuTensorExtractOp>(loc, loweredTensor);
  return BufferizedTensor(extractOp.getBufferResult(),
                          extractOp.getTensorSpecResult());
}

Value Bufferizer::getShape(OpBuilder &builder, Location loc,
                           Value loweredTensor) {
  return builder.create<MGP::CpuTensorExtractShapeOp>(loc, loweredTensor);
}

Value Bufferizer::createTensor(OpBuilder &builder, Location loc, Value inChain,
                               BufferizedTensor &bufferized) {
  auto tensorSpecType = cast<MGP::CpuTensorSpecType>(bufferized.spec.getType());
  auto tensorType =
      MGP::CpuTensorType::get(builder.getContext(), tensorSpecType);
  return builder.create<MGP::CpuTensorCreateOp>(loc,
                                                /*result=*/tensorType,
                                                /*inChain=*/inChain,
                                                /*buffer=*/bufferized.buffer,
                                                /*tensorSpec=*/bufferized.spec);
}

Value Bufferizer::createBufferStatic(OpBuilder &builder, Location loc,
                                     Value inChain, uint64_t size,
                                     uint64_t alignment) {
  return builder.create<MGP::CpuBufferAllocStaticOp>(loc,
                                                     /*inChain=*/inChain,
                                                     /*aSize=*/size,
                                                     /*bRawAlign=*/alignment);
}

BufferizedTensor
Bufferizer::createBuffer(OpBuilder &builder, Location loc, Value inChain,
                         TypedValue<MGP::CpuTensorSpecType> spec,
                         uint64_t alignment) {
  std::optional<uint64_t> optSize = spec.getType().getSizeInBytes();
  if (optSize) {
    // We don't actually need the spec to allocate. It may end up being
    // eliminated by DCE.
    return BufferizedTensor(
        createBufferStatic(builder, loc, inChain, *optSize, alignment), spec);
  } else {
    auto allocOp =
        builder.create<MGP::CpuBufferAllocSpecOp>(loc,
                                                  /*inChain=*/inChain,
                                                  /*runtimeSpec=*/spec,
                                                  /*rawAlign=*/alignment);
    return BufferizedTensor(allocOp.getOutBuffer(), spec);
  }
}

//===----------------------------------------------------------------------===//
// MOGraphLoweringState
//===----------------------------------------------------------------------===//

MGP::CpuTensorSpecDimsDynamicOp
MOGraphLoweringState::getDynamicDims(Value loweredTensor, Location loc) {
  BufferizedTensor bufferized =
      bufferizer.extract(getExecuteOpBuilder(), loc, loweredTensor);
  // NOTE: We ignore the extracted buffer.
  return getExecuteOpBuilder().create<MGP::CpuTensorSpecDimsDynamicOp>(
      loc, bufferized.spec, bufferized.spec.getType().getRawDims());
}

Value MOGraphLoweringState::getShape(Value loweredTensor, Location loc) {
  return bufferizer.getShape(getExecuteOpBuilder(), loc, loweredTensor);
}

ErrorOr<TypedValue<MGP::CpuTensorSpecType>>
MOGraphLoweringState::createTensorSpec(OpBuilder &builder, Location loc,
                                       MO::TensorType moTensorType) {
  if (!moTensorType.hasRank())
    return Error("dynamically ranked MO tensors not supported");
  MGP::CpuTensorSpecType mgpSpecType =
      MGP::createCpuTensorSpecType(moTensorType);
  if (mgpSpecType.getSizeInBytes()) {
    // Statically shaped.
    return builder.create<MGP::CpuTensorSpecCreateOp>(loc,
                                                      /*spec=*/mgpSpecType);
  }

  // Dynamically shaped.
  auto runtimeDimsOr =
      lowerShapeAttrDynDims(moTensorType.getTypedShapeAttr(), loc);
  if (runtimeDimsOr.isError())
    return runtimeDimsOr.takeError();
  return builder.create<MGP::CpuTensorSpecCreateOp>(
      loc,
      /*spec=*/mgpSpecType,
      /*runtimeDims=*/*runtimeDimsOr);
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

//===----------------------------------------------------------------------===//
// MOOpLoweringState
//===----------------------------------------------------------------------===//

BufferizedTensor MOOpLoweringState::getBufferizedInput(Value moTensor) {
  // Find the lowered form for the MO tensor value.
  mlir::Value loweredTensor = loweringState.lookupExecuteLoweredValue(moTensor);
  assert(loweredTensor != nullptr &&
         "referenced SSA value has not been lowered");
  return loweringState.bufferizer.extract(loweringState.getExecuteOpBuilder(),
                                          loc, loweredTensor);
}

SmallVector<BufferizedTensor> MOOpLoweringState::getBufferizedInputs() {
  return llvm::map_to_vector(
      opBeingLowered->getOperands(),
      [&](Value moTensor) { return getBufferizedInput(moTensor); });
}

ErrorOr<SmallVector<TypedValue<MGP::CpuTensorSpecType>>>
MOOpLoweringState::createOutputSpecs() {
  SmallVector<MO::TensorType> resultMoTypes =
      castToMoTensorTypes(opBeingLowered->getResultTypes());
  SmallVector<TypedValue<MGP::CpuTensorSpecType>> specs;
  specs.reserve(resultMoTypes.size());
  for (MO::TensorType ty : resultMoTypes) {
    ErrorOr<TypedValue<MGP::CpuTensorSpecType>> specOr =
        loweringState.createTensorSpec(loweringState.getExecuteOpBuilder(), loc,
                                       ty);
    if (specOr)
      return specOr.takeError();
    specs.emplace_back(*specOr);
  }
  return specs;
}

ErrorOr<SmallVector<BufferizedTensor>>
MOOpLoweringState::createOutputBuffers(Value inChain) {
  ErrorOr<SmallVector<TypedValue<MGP::CpuTensorSpecType>>> specsOr =
      createOutputSpecs();
  if (specsOr)
    return specsOr.takeError();
  return llvm::map_to_vector(
      *specsOr, [&](TypedValue<MGP::CpuTensorSpecType> spec) {
        return loweringState.bufferizer.createBuffer(
            loweringState.getExecuteOpBuilder(), loc, inChain, spec,
            loweringState.getMinimumCpuAlignment());
      });
}

void MOOpLoweringState::createOutputTensors(
    Value inChain, SmallVector<TypedValue<MGP::CpuTensorSpecType>> &outputSpecs,
    SmallVector<Value> &outputBuffers) {
  for (const auto &[idx, spec, buffer] :
       llvm::enumerate(outputSpecs, outputBuffers)) {
    BufferizedTensor bufferizedOutput(buffer, spec);
    Value loweredTensor = loweringState.bufferizer.createTensor(
        loweringState.getExecuteOpBuilder(), loc, inChain, bufferizedOutput);
    // Update the MO tensor to now refer to this lowered tensor.
    loweringState.emplaceExecuteLoweredValue(opBeingLowered->getResult(idx),
                                             loweredTensor);
  }
}

void MOOpLoweringState::createOutputTensors(
    Value inChain, SmallVector<BufferizedTensor> &bufferizedOutputs) {
  for (const auto &[idx, bufferizedOutput] :
       llvm::enumerate(bufferizedOutputs)) {
    Value loweredTensor = loweringState.bufferizer.createTensor(
        loweringState.getExecuteOpBuilder(), loc, inChain, bufferizedOutput);
    // Update the MO tensor to now refer to this lowered tensor.
    loweringState.emplaceExecuteLoweredValue(opBeingLowered->getResult(idx),
                                             loweredTensor);
  }
}
