//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Interfaces/MOGGKernelOpLoweringState.h"

using namespace M;
using namespace M::MGP;

MOGGKernelOpLoweringState::MOGGKernelOpLoweringState(
    Operation *op, MGP::MOGraphLoweringState &loweringState)
    : moggOp(op), loweringState(loweringState),
      opLoweringState(loweringState, op), builder(opLoweringState.builder),
      device(MO::getDeviceOrDefaultAny(op)) {}

void MOGGKernelOpLoweringState::addOperandAndMapToIndex(
    Value operand, InOutSignatureAttr::InOutSemantics sig) {
  outsideValueToOperandIndex[operand] = operands.size();
  operands.emplace_back(operand);
  operandSemantics.emplace_back(sig);
}

ErrorOr<Value>
MOGGKernelOpLoweringState::getDynamicDimension(StringAttr dim,
                                               bool isInsideStub) {
  auto outsideSSAIt = parameterToOutsideSSAValue.find(dim);
  if (outsideSSAIt == parameterToOutsideSSAValue.end()) {
    return Error(
        llvm::formatv("Failed to find the runtime value of the parametric "
                      "dimension `{}`",
                      dim));
  }
  Value outsideSSAValue = outsideSSAIt->second;

  if (isInsideStub) {
    if (!outsideValueToStubOperandIndex.contains(outsideSSAValue)) {
      return Error(llvm::formatv(
          "Failed to find the stub argument for the runtime value "
          "of the parametric dimension `{}`",
          dim));
    }
    return kernelStub.getBody().getArgument(
        outsideValueToStubOperandIndex[outsideSSAValue]);
  } else {
    if (!outsideValueToOperandIndex.contains(outsideSSAValue)) {
      return Error(llvm::formatv(
          "Failed to find the kernel argument for the runtime value "
          "of the parametric dimension `{}`",
          dim));
    }
    return mgpRegionBody->getArgument(
        outsideValueToOperandIndex[outsideSSAValue]);
  }
}

ErrorOr<SmallVector<Value>>
MOGGKernelOpLoweringState::getTensorDynamicDimensions(MO::TensorType asTensor,
                                                      bool isInsideStub) {
  SmallVector<Value> dynamicDimensions;
  for (auto dim : asTensor.getDims()) {
    if (auto dimParamAttr = dyn_cast<KGEN::ParamDeclRefAttr>(dim)) {
      ErrorOr<Value> dimValueOr =
          getDynamicDimension(dimParamAttr.getName(), isInsideStub);
      if (dimValueOr.isError())
        return dimValueOr.takeError();
      dynamicDimensions.push_back(*dimValueOr);
    }
  }

  return dynamicDimensions;
}

void MOGGKernelOpLoweringState::passDynDimensionToKernelStub(Location loc,
                                                             StringAttr dim) {
  Value outsideSSAValue = parameterToOutsideSSAValue.at(dim);

  // Check if the argument isn't already defined.
  if (outsideValueToStubOperandIndex.contains(outsideSSAValue))
    return;

  // Pass the dynamic dimension as an argument of the stub.
  Value regionArg =
      mgpRegionBody->getArgument(outsideValueToOperandIndex[outsideSSAValue]);
  kernelStub.getBody().addArgument(regionArg.getType(), loc);
  outsideValueToStubOperandIndex[outsideSSAValue] = stubCallOperands.size();
  stubCallOperands.emplace_back(regionArg);
}

void MOGGKernelOpLoweringState::passTensorDynDimensionsToKernelStub(
    Location loc, MO::TensorType asTensor) {
  for (auto dim : asTensor.getDims())
    if (auto dimParamAttr = dyn_cast<KGEN::ParamDeclRefAttr>(dim))
      passDynDimensionToKernelStub(loc, dimParamAttr.getName());
}

void MOGGKernelOpLoweringState::lowerDeviceContextPlaceholder(
    MOGG::DeviceContextPlaceholderOp moggContext) {
  Value moggContextArg = kernelStub.getBodyBlock().addArgument(
      moggContext.getType(), moggContext.getLoc());
  moggContext.replaceAllUsesWith(moggContextArg);

  Value mgpContext = loweringState.getExecuteDevice(moggContext.getDevice());
  mgpRegion->insertOperands(mgpRegion.getNumOperands(), {mgpContext});
  Value mgpContextArg =
      mgpRegionBody->addArgument(mgpContext.getType(), mgpContext.getLoc());
  // The MGP context to MOGG context conversion happens at the stub (no
  // need for a conversion op since it's 1:1).
  callStubOp->insertOperands(callStubOp.getNumOperands(), mgpContextArg);

  // Update the kernel InOut semantics if needed.
  if (mgpRegion.getInOutSignature().size() != 0) {
    SmallVector<InOutSignatureAttr::InOutSemantics> newSemantics(
        1, InOutSignatureAttr::InOutSemantics::kNone);
    InOutSignatureAttr oldSemantics = mgpRegion.getInOutSignature();
    mgpRegion.setInOutSignatureAttr(oldSemantics.append(newSemantics));
  }

  moggContext.erase();
}

LogicalResult
MOGGKernelOpLoweringState::lowerBindLambdaOp(MOGG::BindLambdaOp bindOp) {
  mlir::MLIRContext *ctx = moggOp->getContext();
  auto loc = moggOp->getLoc();

  // Only replace I/O lambdas.
  if (bindOp.getBindKind() != MOGG::BindLambdaKind::InputLambda &&
      bindOp.getBindKind() != MOGG::BindLambdaKind::OutputLambda)
    return success();

  Value tensorVal = bindOp.getFusibleTensors().front();
  Value lambdaVal = bindOp.getOutput();

  // If the TensorFusedOp was already created, we don't need to change
  // anything else.
  // TODO: This is very confusing here. In MOGG dialect we just build the
  // lambdas with the BindLambdaOp, and this op once lowered might require
  // some transforms to generate a correct kernel call (eg pass an empty
  // tensor, pass the same tensor, or reshape it). This transformation is
  // implicit in MOGG IR, and the whole logic here figure this out and build
  // the correct FusedTensor object. But there is one special case: some
  // kernels require a scratchpad buffer when fused with a cast operation, and
  // compared to the transformations above, this scratchpad buffer is explicit
  // in the MOGG IR. We should either have all transformations implicit or all
  // explicit in MOGG IR
  if (isa<MOGG::TensorFusedOp>(*lambdaVal.getUsers().begin()))
    return success();

  // We always pass an empty tensor to the kernels for fused input tensors or
  // output lambdas with a cast.
  bool lambdaHasCast = MO::castFromTensorLike(tensorVal.getType()).getDtype() !=
                       MO::castFromTensorLike(lambdaVal.getType()).getDtype();
  bool substituteWithEmptyTensor =
      bindOp.getBindKind() == MOGG::BindLambdaKind::InputLambda ||
      lambdaHasCast;

  builder.setInsertionPointAfter(bindOp);

  auto mgpBufferType = MGP::DeviceBufferType::get(device);

  // Replace the output of the lambda with an empty tensor.
  if (substituteWithEmptyTensor) {
    Value emptyBuffer =
        MOGG::EmptyBufferOp::create(builder, bindOp->getLoc(), mgpBufferType);
    auto dynamicDimensionsOr =
        getTensorDynamicDimensions(MO::castFromTensorLike(lambdaVal.getType()),
                                   /*isInsideStub=*/true);
    if (dynamicDimensionsOr.isError())
      return moggOp->emitError(dynamicDimensionsOr.getError());
    MO::TensorType tensorType = MO::castFromTensorLike(lambdaVal.getType());
    tensorType =
        tensorType.cloneWith(MO::IOKindAttr::get(ctx, MO::IOKind::Input))
            .cloneWithMut(BoolAttr::get(ctx, false));
    tensorVal = MOGG::TensorInitOp::create(
        builder, loc, tensorType, emptyBuffer, *dynamicDimensionsOr,
        /*mut=*/BoolAttr::get(ctx, false),
        /*io=*/MO::IOKindAttr::get(ctx, MO::IOKind::Input),
        /*exclusive=*/BoolAttr::get(ctx, true));
  }

  else if (bindOp.hasAffineViewOperation()) {
    // The tensor needs to be reshaped to match the expected kernel operand
    // shape.
    auto dynamicDimensionsOr =
        getTensorDynamicDimensions(MO::castFromTensorLike(lambdaVal.getType()),
                                   /*isInsideStub=*/true);
    if (dynamicDimensionsOr.isError())
      return moggOp->emitError(dynamicDimensionsOr.getError());
    tensorVal = MOGG::TensorReshapeOp::create(builder, loc, lambdaVal.getType(),
                                              tensorVal, *dynamicDimensionsOr);
  }

  // Build the Fused tensor object.
  auto fusedTensorOp = MOGG::TensorFusedOp::create(
      builder, bindOp->getLoc(), lambdaVal.getType(), lambdaVal, tensorVal);

  // And finally replace it for the users of the lambda.
  lambdaVal.replaceAllUsesExcept(fusedTensorOp.getResult(), fusedTensorOp);
  return success();
}

void MOGGKernelOpLoweringState::lowerConstantOp(MO::ConstantOp constantOp) {
  mlir::MLIRContext *ctx = moggOp->getContext();
  auto loc = moggOp->getLoc();

  // Exclusive is True unless it's used by a view kernel.
  bool isExclusive =
      llvm::none_of(constantOp.getResult().getUsers(),
                    [](Operation *use) { return isa<MOGG::CallViewOp>(use); });

  builder.setInsertionPoint(callStubOp);
  auto currentResources = mgpRegion.getConstantResources();
  SmallVector<Attribute> newResources(currentResources.begin(),
                                      currentResources.end());
  newResources.push_back(
      *loweringState.lowerElementsAttr(constantOp.getValue()));
  mgpRegion.setConstantResourcesAttr(ArrayAttr::get(ctx, newResources));
  Value bufferConstant = MGP::BufferConstantOp::create(
      builder, constantOp->getLoc(), cast<ElementsAttr>(newResources.back()));
  callStubOp->insertOperands(callStubOp->getNumOperands(), bufferConstant);
  auto bufferValue = kernelStub.getBodyBlock().addArgument(
      bufferConstant.getType(), bufferConstant.getLoc());
  auto newFunctionType =
      FunctionType::get(ctx, kernelStub.getBodyBlock().getArgumentTypes(),
                        kernelStub.getFunctionType().getResults());
  kernelStub.setFunctionType(newFunctionType);

  builder.setInsertionPoint(constantOp);
  SmallVector<Value> dynamicDimensions;
  MO::TensorType tensorType =
      MO::castFromTensorLike(constantOp.getResult().getType());
  tensorType = tensorType.cloneWith(MO::IOKindAttr::get(ctx, MO::IOKind::Input))
                   .cloneWithMut(BoolAttr::get(ctx, false));
  Value constantValue = MOGG::TensorInitOp::create(
      builder, loc, tensorType, bufferValue,
      /*dynamicDimensions=*/ArrayRef<Value>{},
      /*mut=*/BoolAttr::get(ctx, false),
      /*io=*/MO::IOKindAttr::get(ctx, MO::IOKind::Input),
      BoolAttr::get(ctx, isExclusive));
  constantOp.getResult().replaceAllUsesWith(constantValue);
  constantOp->erase();
}

LogicalResult
MOGGKernelOpLoweringState::lowerGetDimOp(MOGG::GetDimOp getDimOp) {
  ErrorOr<Value> dimValueOr =
      getDynamicDimension(getDimOp.getDimension(), /*isInsideStub=*/true);
  if (dimValueOr.isError())
    return moggOp->emitError(dimValueOr.getError());
  getDimOp.getResult().replaceAllUsesWith(*dimValueOr);
  getDimOp->erase();
  return success();
}

LogicalResult MOGGKernelOpLoweringState::lowerTensorGetShapeOp(
    MOGG::TensorGetShapeOp getShapeOp) {
  // Get the dynamic dimensions
  auto asTensor = MO::castFromTensorLike(getShapeOp.getTensor());
  auto dynamicDimensionsOr = getTensorDynamicDimensions(asTensor,
                                                        /*isInsideStub=*/true);
  if (dynamicDimensionsOr.isError())
    return moggOp->emitError(dynamicDimensionsOr.getError());

  // Build a ShapeInitOp to materialize the new shape mojo value
  builder.setInsertionPoint(getShapeOp);
  Value shapeValue = MOGG::ShapeInitOp::create(builder, getShapeOp.getLoc(),
                                               *dynamicDimensionsOr,
                                               asTensor.getTypedShapeAttr());

  getShapeOp.getResult().replaceAllUsesWith(shapeValue);
  getShapeOp->erase();
  return success();
}

LogicalResult MOGGKernelOpLoweringState::lowerKernelStub() {
  auto walkResult = kernelStub.walk([&](mlir::Operation *op) {
    if (auto moggContext = dyn_cast<MOGG::DeviceContextPlaceholderOp>(op))
      lowerDeviceContextPlaceholder(moggContext);
    else if (auto bindOp = dyn_cast<MOGG::BindLambdaOp>(op)) {
      if (failed(lowerBindLambdaOp(bindOp)))
        return WalkResult::interrupt();
    } else if (auto constantOp = dyn_cast<MO::ConstantOp>(op))
      lowerConstantOp(constantOp);
    else if (auto tensorGetShapeOp = dyn_cast<MOGG::TensorGetShapeOp>(op)) {
      if (failed(lowerTensorGetShapeOp(tensorGetShapeOp)))
        return WalkResult::interrupt();
    } else if (auto getDimOp = dyn_cast<MOGG::GetDimOp>(op)) {
      if (failed(lowerGetDimOp(getDimOp)))
        return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  return success(!walkResult.wasInterrupted());
}

LogicalResult
MOGGKernelOpLoweringState::lowerKernelOp(MOGG::KernelOp moggKernelOp) {
  mlir::MLIRContext *ctx = moggKernelOp.getContext();
  auto loc = moggKernelOp.getLoc();

  SmallVector<KGEN::ParamDeclRefAttr> usedKernelParameters =
      M::MOGG::getKernelUsedParameters(moggKernelOp);

  // Outline the kernel into a stub by creating the op and taking the body of
  // the kernel op we are lowering. Output and scratch placeholders are part of
  // the MGP memory management analysis which means they will come from the
  // outside.
  Block *kernelBody = &moggKernelOp.getExecBody().front();
  auto outputPlaceholdersOp = moggKernelOp.getOutputPlaceholder();
  if (outputPlaceholdersOp) {
    for (auto value : outputPlaceholdersOp.getResults())
      value.replaceAllUsesWith(kernelBody->addArgument(value.getType(), loc));
  }

  auto scratchPlaceholdersOp = moggKernelOp.getScratchPlaceholder();
  if (scratchPlaceholdersOp) {
    for (auto value : scratchPlaceholdersOp.getResults())
      value.replaceAllUsesWith(kernelBody->addArgument(value.getType(), loc));
  }

  // If the kernel returns opaque types, they must be part of the stub's
  // signature.
  uint64_t resultTypeIndex;
  SmallVector<MO::OpaqueType> resultTypes;
  for (auto [i, resultType] : llvm::enumerate(moggKernelOp.getResultTypes())) {
    if (auto opaqueType = dyn_cast<MO::OpaqueType>(resultType)) {
      resultTypeIndex = i;
      resultTypes.push_back(opaqueType);
    }
  }

  if (resultTypes.size() > 1) {
    moggKernelOp.emitError(
        "MGP lowering does not support multiple returned opaque types");
    return failure();
  }

  // Create the stub op at the top of the module.
  /// TODO: Find a better way of naming stubs. This is not thread safe.
  static int stubCounter = 0;
  auto stubName =
      StringAttr::get(ctx, llvm::formatv("stub_{}", stubCounter++).str());

  // Determine the return type for the stub
  Type stubReturnType = nullptr;
  auto terminator = moggKernelOp.getTerminator();
  if (!resultTypes.empty()) {
    stubReturnType = resultTypes.front();
  } else if (auto regOutput = terminator.getKernelReturnOutput()) {
    stubReturnType = regOutput;
  }

  // Create the stub outside of the model op to ensure it is properly registered
  // into the SymbolTable of the ModuleOp.
  auto insertionPoint = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(loweringState.getModelOp()->getBlock());
  // First, we create a () -> () function type because we don't yet set the
  // value of certain things like IOKind.
  auto functionType = TypeAttr::get(FunctionType::get(ctx, {}, {}));
  kernelStub = MOGG::KernelStubOp::create(
      builder, loc, stubName, functionType, moggKernelOp.getSummaryAttr(),
      moggKernelOp.getIsAsyncAttr(), ArrayAttr::get(ctx, {}),
      MOGG::ABIKindAttr::get(ctx, MOGG::ABIKind::MOGG));
  kernelStub.getBody().takeBody(moggKernelOp.getExecBody());
  builder.restoreInsertionPoint(insertionPoint);

  // This is a list of input operands in MGP but without the chain.
  auto maybeInChain = moggKernelOp.maybeGetInChain();
  auto loweredInputOperands = opLoweringState.getLoweredOperands();
  Value chain = loweringState.getCurrentExecuteChain();
  if (maybeInChain) {
    chain = loweringState.lookupExecuteLoweredValue(*maybeInChain);
    loweredInputOperands.pop_back();
  }
  chain = opLoweringState.getGuardChain(chain);

  // This is a list of all the outputs of the kernel but lowered to MGP.
  SmallVector<MGP::MOOutputStorage> loweredOutputsStorage;
  auto loweredOutputsStorageOr = opLoweringState.createOutputStorage(chain);
  if (failed(loweredOutputsStorageOr))
    return failure();
  loweredOutputsStorage = *std::move(loweredOutputsStorageOr);

  // Scratch output placeholders turn into DPS kernel inputs. This makes it
  // possible to do memory scheduling on temporary storage. Lower the output of
  // the scratch placeholders into MGP types.
  SmallVector<MGP::MOOutputStorage> scratchOutputsStorage;
  if (scratchPlaceholdersOp) {
    for (Type t : scratchPlaceholdersOp.getResultTypes()) {
      auto specResult =
          opLoweringState.createTensorSpec(cast<MO::TensorType>(t));
      if (failed(specResult))
        return failure();

      auto bufOr = opLoweringState.createBuffer(chain, *specResult, t);
      if (failed(bufOr))
        return failure();
      scratchOutputsStorage.emplace_back(*bufOr);
    }
  }

  SmallVector<mlir::Type> mgpKernelResTypes;

  // 1. Add the chain with the input signature.
  addOperandAndMapToIndex(chain, InOutSignatureAttr::kIn);

  // 2. Add the inputs with the input or mutable input signature.
  // Use zip_first because we removed the original chain from the list.
  auto operandsRange = moggKernelOp.getOperands();
  auto operandsToZip = maybeInChain ? operandsRange.drop_back() : operandsRange;
  for (auto &&[origOperand, loweredInputOperand] :
       llvm::zip_first(operandsToZip, loweredInputOperands)) {
    auto sig = isa<MO::BufferType>(origOperand.getType())
                   ? InOutSignatureAttr::kMut
                   : InOutSignatureAttr::kIn;
    addOperandAndMapToIndex(loweredInputOperand, sig);
  }

  // 3. Add the outputs with the output signature.
  for (auto &&outputStorage : loweredOutputsStorage)
    addOperandAndMapToIndex(outputStorage.buffer, InOutSignatureAttr::kOut);

  // 4. Add scratch buffers with the none signature because it is only used
  // internally.
  for (auto &&scratchStorage : scratchOutputsStorage)
    addOperandAndMapToIndex(scratchStorage.buffer, InOutSignatureAttr::kNone);

  // 5. In MGP, parameters turn into values. Add the values of input parameters
  // with the input signature.
  for (auto param : usedKernelParameters) {
    auto symbolicDimOr = loweringState.lowerExpression(
        opLoweringState.opBeingLowered->getParentRegion(), opLoweringState.loc,
        param);
    Value symbol = symbolicDimOr.takeValue();
    addOperandAndMapToIndex(symbol, InOutSignatureAttr::kIn);
    parameterToOutsideSSAValue[param.getName()] = symbol;
  }

  // 6. Add a chain to the result types.
  mgpKernelResTypes.push_back(chain.getType());

  // 7. The original kernel could also return an opaque type which is allocated
  // inside of its own body. If that's the case, add it to the output of the
  // mogg.generic.compiled.region op since it is not in DPS form. The current
  // contract is that if a kernel op returns an opaque type, it must have been
  // allocated internally. Tensors can't be returned in a non-DPS form.
  // We use MO::OpaqueType here to match what the stub returns.
  if (!resultTypes.empty()) {
    mgpKernelResTypes.push_back(resultTypes.front());
  }

  // 8. The original kernel may be setting some parameters (shape functions do
  // just that). These will returned from the mogg.generic.compiled.region.
  for (size_t i = 0; i < moggKernelOp.getOutputParamDecls().size(); ++i)
    mgpKernelResTypes.push_back(
        moggKernelOp.getOutputParamDecls()[i].getType());

  mgpRegion = MOGG::GenericCompiledRegionOp::create(
      builder, loc, TypeRange{mgpKernelResTypes}, operands,
      moggKernelOp.getOutputParamDeclsAttr(),
      InOutSignatureAttr::get(ctx, operandSemantics),
      moggKernelOp.getSummaryAttr(), // summary
      moggKernelOp.getIsAsyncAttr(), // isAsync
      ArrayAttr::get(ctx, {}),       // KernelOp has no constant resources
      // Sequence fuse GPU launching to reduce primitive launch overhead.
      BoolAttr::get(ctx, !MO::isOnHost(moggKernelOp)), // canFuseIntoOther
      BoolAttr::get(ctx, true),                        // canBeFusedInto
      BoolAttr::get(ctx, false),                       // canDuplicate
      BoolAttr::get(ctx, true)                         // isStructuredKernel
  );
  mgpRegion.setDeviceRef(device);
  kernelStub.setDeviceRef(device);

  // Add a body to the created operation and mirror its operands with block
  // arguments.
  mgpRegionBody = &mgpRegion.getBody().emplaceBlock();
  builder.setInsertionPoint(mgpRegionBody, mgpRegionBody->begin());
  SmallVector<Location> locs =
      llvm::map_to_vector(operands, [](Value v) { return v.getLoc(); });
  mgpRegionBody->addArguments(ValueRange(operands).getTypes(), locs);

  // Call the previously created kernel stub by creating ops that bridge the gap
  // between MGP and MO/MOGG.
  builder.setInsertionPointToStart(mgpRegionBody);

  // Analyze which stub arguments are used as outputs in kernel terminator
  // operations
  DenseSet<Value> outputArguments;
  kernelStub.walk([&](Operation *op) {
    if (auto terminator = dyn_cast<MOGG::KernelTerminator>(op)) {
      for (Value output : terminator.getKernelDPSOutputs()) {
        outputArguments.insert(output);
      }
    }
  });

  // Explicitly track operand counts for correct mapping between stub and region
  // arguments. The stub body arguments are structured as:
  // [input operands..., output placeholders..., scratch placeholders...]
  // The region body arguments are structured as:
  // [chain, input operands..., output storage buffers..., scratch storage
  // buffers..., parameters...]
  const uint64_t numInputOperands = loweredInputOperands.size();
  const uint64_t numOutputOperands = loweredOutputsStorage.size();
  const uint64_t numScratchOperands = scratchOutputsStorage.size();

  // Helper to calculate region argument index based on stub argument position.
  // Region args: [chain, inputs..., outputs..., scratch...]
  // Stub args: [inputs..., outputs..., scratch...] (same order, no chain)
  auto getRegionArgIndex = [&](uint64_t stubArgIdx) -> uint64_t {
    // Add 1 for the chain argument at position 0
    return 1 + stubArgIdx;
  };

  // Track the count of tensor-only arguments seen (for input/output boundary)
  uint64_t tensorArgIdx = 0;
  // Track overall stub argument index for region argument mapping
  uint64_t stubArgIdx = 0;
  for (auto stubArgument : kernelStub.getBody().front().getArguments()) {
    auto asTensor = MO::dynCastFromTensorLike(stubArgument.getType());
    if (!asTensor) {
      // Non-tensor arguments: opaque types are passed directly from region,
      // device contexts use chain placeholder (handled later by
      // lowerDeviceContextPlaceholder).
      if (isa<MO::OpaqueType>(stubArgument.getType())) {
        uint64_t regionArgIndex = getRegionArgIndex(stubArgIdx);
        if (regionArgIndex >= mgpRegionBody->getArguments().size()) {
          return moggKernelOp.emitError(
              "Region argument index out of bounds for opaque");
        }
        stubCallOperands.emplace_back(
            mgpRegionBody->getArguments()[regionArgIndex]);
      } else {
        // Device context or other types - use chain placeholder
        stubCallOperands.emplace_back(mgpRegionBody->getArguments()[0]);
      }
      ++stubArgIdx;
      continue;
    }

    auto dynamicDimensionsOr =
        getTensorDynamicDimensions(asTensor, /*isInsideStub=*/false);
    if (dynamicDimensionsOr.isError())
      return moggKernelOp.emitError(dynamicDimensionsOr.getError());

    // Determine if this is an output: either in output/scratch range or
    // explicitly marked. Use tensorArgIdx for this comparison since
    // numInputOperands/numOutputOperands count tensor-only operands.
    bool isOutput = (tensorArgIdx >= numInputOperands) ||
                    outputArguments.contains(stubArgument);
    MO::IOKind io = isOutput ? MO::IOKind::Output : MO::IOKind::Input;
    bool mut = isOutput || isa<MO::BufferType>(stubArgument.getType());

    // Exclusive is True unless it's used by a view kernel
    bool isExclusive =
        llvm::none_of(stubArgument.getUsers(), [](Operation *use) {
          return isa<MOGG::CallViewOp>(use);
        });

    // Update stub argument type
    auto ioAttr = MO::IOKindAttr::get(ctx, io);
    auto mutAttr = BoolAttr::get(ctx, mut);
    stubArgument.setType(asTensor.cloneWith(ioAttr).cloneWithMut(mutAttr));

    // Verify tensor index is within expected bounds
    if (tensorArgIdx >=
        numInputOperands + numOutputOperands + numScratchOperands) {
      return moggKernelOp.emitError(
          "Stub argument index exceeds expected tensor argument count");
    }

    // Get corresponding region argument using stubArgIdx (accounts for opaques)
    uint64_t regionArgIndex = getRegionArgIndex(stubArgIdx);
    if (regionArgIndex >= mgpRegionBody->getArguments().size()) {
      return moggKernelOp.emitError("Region argument index out of bounds");
    }
    Value regionValue = mgpRegionBody->getArguments()[regionArgIndex];

    // Create tensor initialization
    auto tensorInitOp = MOGG::TensorInitOp::create(
        builder, loc, stubArgument.getType(), regionValue, *dynamicDimensionsOr,
        mutAttr, ioAttr, BoolAttr::get(ctx, isExclusive));
    stubCallOperands.emplace_back(tensorInitOp);

    ++tensorArgIdx;
    ++stubArgIdx;
  }

  // Add extra arguments for the dynamic dimensions for some ops
  kernelStub.walk([&](mlir::Operation *stubOp) {
    if (auto bindOp = dyn_cast<MOGG::BindLambdaOp>(stubOp)) {
      // We only need to get the dynamic dimensions when materializing an
      // intermediate tensor for the fused kernel operand. This happens only for
      // Input and AffineView fusion.
      if (bindOp.getBindKind() != MOGG::BindLambdaKind::InputLambda &&
          !bindOp.hasAffineViewOperation())
        return;

      Value fusedOutput = bindOp.getOutput();
      passTensorDynDimensionsToKernelStub(
          bindOp.getLoc(), MO::castFromTensorLike(fusedOutput.getType()));
    }

    else if (auto tensorShapeOp = dyn_cast<MOGG::TensorGetShapeOp>(stubOp)) {
      // The dynamic dimensions of TensorGetShapeOp must be lowered.
      passTensorDynDimensionsToKernelStub(
          tensorShapeOp.getLoc(),
          MO::castFromTensorLike(tensorShapeOp.getTensor()));
    }

    else if (auto getDimOp = dyn_cast<MOGG::GetDimOp>(stubOp)) {
      passDynDimensionToKernelStub(getDimOp.getLoc(), getDimOp.getDimension());
    }

    // Handle dynamic dimensions from result types of all operations.
    // This is needed for operations like mogg.experimental.kernel that have
    // tensor result types with dynamic dimensions.
    for (Type resultType : stubOp->getResultTypes()) {
      if (auto tensorType = MO::dynCastFromTensorLike(resultType)) {
        passTensorDynDimensionsToKernelStub(stubOp->getLoc(), tensorType);
      }
    }
  });

  assert(stubCallOperands.size() ==
         (kernelStub.getBodyBlock().getNumArguments()));

  // Create the call to the outlined kernel stub and use whatever return value
  // (if any) in the output op of the new kernel.
  callStubOp = MOGG::CallStubOp::create(
      builder, loc, stubReturnType ? TypeRange{stubReturnType} : TypeRange{},
      SymbolRefAttr::get(stubName), stubCallOperands);
  auto readyChain = MOGG::AsyncReadyOp::create(builder, loc);

  SmallVector<Value> outputs;
  outputs.push_back(readyChain);

  // Check if the terminator returns a shape result that needs unpacking
  if (callStubOp.getNumResults() > 0) {
    Value result = callStubOp.getResults().front();
    if (auto shapeResultType =
            dyn_cast<MOGG::ShapeFuncResultType>(result.getType())) {
      // For shape functions, extract individual dimensions from the shape
      // result
      size_t numDims = shapeResultType.getRank();

      // Create individual getitem operations for each dimension
      for (size_t i = 0; i < numDims; ++i) {
        auto getItemOp = MOGG::ShapeResultGetItemOp::create(
            builder, loc, builder.getIntegerType(64, true), result,
            builder.getIntegerAttr(builder.getIntegerType(64, true), i));
        outputs.push_back(getItemOp.getResult());
      }
    } else {
      // Regular case: add call stub results directly
      for (Value result : callStubOp.getResults()) {
        outputs.push_back(result);
      }
    }
  }

  MOGG::OutputOp::create(builder, loc, outputs);

  // Some operations needs to be lowered to work with EmitMojo.
  if (failed(lowerKernelStub()))
    return failure();

  // The following is to inform the lowering state of the values we have
  // created.
  builder.setInsertionPointAfter(mgpRegion);
  if (!resultTypes.empty()) {
    Value oldResult = moggKernelOp.getResult(resultTypeIndex);
    Value newResult = mgpRegion.getResult(1);
    loweringState.emplaceExecuteLoweredValue(oldResult, newResult);
  }

  opLoweringState.createOutputs(mgpRegion->getResult(0), loweredOutputsStorage);
  if (auto chain = moggKernelOp.maybeGetOutChain()) {
    loweringState.emplaceExecuteLoweredValue(std::move(*chain),
                                             mgpRegion.getResult(0));
  }

  // Handle parameter binding for shape function kernels
  // Doing this is necessary for mo-to-primtiives to properly map the output
  // dims (SSA values) to downstream kernels with tensors making use of these
  // params
  ArrayRef<KGEN::ParamDeclAttr> decls = moggKernelOp.getOutputParamDecls();
  if (!decls.empty() && terminator.getKernelReturnOutput()) {
    // For shape function kernels, the newKernel results are [chain, dim0, dim1,
    // dim2, ...] The dimensions come from unpacking the shape result inside the
    // kernel
    if (mgpRegion.getNumResults() < decls.size() + 1) {
      moggKernelOp.emitOpError() << "Expected at least " << (decls.size() + 1)
                                 << " results for shape function kernel, got "
                                 << mgpRegion.getNumResults();
      return failure();
    }
    for (auto [idx, decl] : llvm::enumerate(decls)) {
      // Get the dimension result from the kernel
      Value dimResult = mgpRegion.getResult(idx + 1);

      // Convert to dimension value
      auto asDimension = loweringState.indexToDimension(
          loweringState.getExecuteOpBuilder(), loc, dimResult);

      if (auto err = loweringState.lowerParamDecl(
              moggKernelOp.getOperation()->getParentRegion(), decl,
              asDimension)) {
        moggKernelOp.emitOpError() << "Failed to bind parameter "
                                   << decl.getName() << ": " << err.getError();
        return failure();
      }
    }
  }

  // Remove the placeholder ops that we replaced with DPS operands.
  if (outputPlaceholdersOp)
    outputPlaceholdersOp.erase();
  if (scratchPlaceholdersOp)
    scratchPlaceholdersOp.erase();

  mlir::BitVector unpackItAll(mgpRegionBody->getNumArguments(), true);
  mlir::BitVector packItAll(mgpRegion.getNumResults(), true);
  loweringState.lowerMOGGABI(mgpRegion, unpackItAll, packItAll);

  // We can now safely set the function type because we have added all arguments
  // and attributes.
  kernelStub.setFunctionType(
      FunctionType::get(ctx, kernelStub.getBodyBlock().getArgumentTypes(),
                        stubReturnType ? stubReturnType : TypeRange{}));

  return success();
}
