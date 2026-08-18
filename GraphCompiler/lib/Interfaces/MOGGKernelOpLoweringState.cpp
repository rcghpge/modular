//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/MOGGKernelOpLoweringState.h"

#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/MOGGDialect/Analysis/TraitContractDerivation.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/AssertStream.h"

using namespace M;
using namespace M::MGP;

/// Returns the element type of a `!mogg.async<T>`, or the type itself
/// otherwise. MGP region body arguments are async-wrapped, but the outlined
/// kernel stub is called with unpacked values, so its parameter types use the
/// element type.
static Type getAsyncElementOrSelf(Type type) {
  if (auto asyncType = dyn_cast<MOGG::AsyncType>(type))
    return asyncType.getElementType();
  return type;
}

void KernelOperandsMapping::addOperand(Value operandValue,
                                       bool isMGPRegionOperand,
                                       bool isKernelStubOperand,
                                       InOutSignatureAttr::InOutSemantics sig,
                                       Value keyValue, Type stubOperandType) {
  if (!keyValue)
    keyValue = operandValue;

  ASSERT_STREAM(isMGPRegionOperand || isKernelStubOperand,
                "The operand must be for the MGP Region and/or the KernelStub");

  unsigned operandIdx = operands.size();
  OperandValue operand;
  operand.operandValue = operandValue;
  operand.kernelStubOperandType = stubOperandType;

  if (isMGPRegionOperand) {
    operand.mgpRegionOperandIdx = mgpRegionOperands.size();
    mgpRegionOperands.push_back(operandValue);
    mgpRegionOperandSemantics.push_back(sig);
  }
  if (isKernelStubOperand) {
    operand.kernelOperandIdx = kernelStubOperands.size();
    // If the operand is forwarded from the MGP region to the stub, set it to
    // null, it'll be updated later. Otherwise if it's only a stub argument,
    // set it now.
    Value stubOperand = isMGPRegionOperand ? nullptr : operandValue;
    kernelStubOperands.push_back(stubOperand);
  }

  operands.push_back(operand);
  valToOperandsMap[keyValue] = operandIdx;
}

void KernelOperandsMapping::addOperand(KGEN::ParamDeclRefAttr paramName,
                                       Value paramValue,
                                       bool isKernelStubOperand) {
  unsigned operandIdx = operands.size();
  addOperand(paramValue, /*isMGPRegionOperand=*/true, isKernelStubOperand);
  ASSERT_STREAM(!parameterToOperandsMap.contains(paramName.getName()),
                "redefinition of the same parameter");
  parameterToOperandsMap[paramName.getName()] = operandIdx;
}

void KernelOperandsMapping::remapKernelStubOperand(unsigned operandIdx,
                                                   Value newValue) {
  OperandValue &operand = operands[operandIdx];
  auto kernelStubOperandIdx = operand.kernelOperandIdx.value();
  kernelStubOperands[kernelStubOperandIdx] = newValue;
}

Value KernelOperandsMapping::get(unsigned operandIdx, bool isInsideStub) const {
  const OperandValue &operand = operands[operandIdx];

  if (isInsideStub) {
    ASSERT_STREAM(operand.kernelStubOperand,
                  "No kernel stub argument mapped for "
                      << operand.operandValue);
    return operand.kernelStubOperand;
  } else {
    ASSERT_STREAM(operand.mgpRegionOperand,
                  "No mgp region argument mapped for " << operand.operandValue);
    return operand.mgpRegionOperand;
  }
}

Value KernelOperandsMapping::get(Value val, bool isInsideStub) const {
  auto it = valToOperandsMap.find(val);
  ASSERT_STREAM(it != valToOperandsMap.end(), "No mapping found for " << val);
  return get(it->second, isInsideStub);
}

Value KernelOperandsMapping::get(StringAttr paramName,
                                 bool isInsideStub) const {
  auto it = parameterToOperandsMap.find(paramName);
  ASSERT_STREAM(it != parameterToOperandsMap.end(),
                "No mapping found for parameter " << paramName);
  return get(it->second, isInsideStub);
}

SmallVector<Value>
KernelOperandsMapping::getTensorDynamicDimensions(MO::TensorType asTensor,
                                                  bool isInsideStub) {
  SmallVector<Value> dynamicDimensions;
  for (auto dim : asTensor.getDims()) {
    if (isa<KGEN::SIMDAttr>(dim))
      continue;
    auto dimParamAttr = dyn_cast<KGEN::ParamDeclRefAttr>(dim);
    ASSERT_STREAM(dimParamAttr,
                  "getTensorDynamicDimensions: unsupported dim attr kind. "
                  "Expected IntegerAttr or KGEN::ParamDeclRefAttr but got: "
                      << dim << " (type=" << dim.getType() << ")"
                      << " in tensor type " << asTensor << ".");
    if (dimParamAttr)
      dynamicDimensions.push_back(get(dimParamAttr.getName(), isInsideStub));
  }

  return dynamicDimensions;
}

void KernelOperandsMapping::setupMGPRegionArgs(Block *mgpRegion) {
  // Add the arguments. Body block arguments are `!mogg.async<T>` wrappers
  // around the operand types; they are unpacked to `T` by `lowerMOGGABI`.
  SmallVector<Location> locs = llvm::map_to_vector(
      mgpRegionOperands, [](Value v) { return v.getLoc(); });
  SmallVector<Type> argTypes =
      llvm::map_to_vector(mgpRegionOperands, [](Value v) -> Type {
        return MOGG::AsyncType::get(v.getType());
      });
  mgpRegion->addArguments(argTypes, locs);

  // And now update the mapping with the created block arguments.
  for (auto &operand : operands) {
    if (!operand.mgpRegionOperandIdx.has_value())
      continue;

    BlockArgument blockArg =
        mgpRegion->getArgument(operand.mgpRegionOperandIdx.value());

    operand.mgpRegionOperand = blockArg;
    if (auto stubOperandIdx = operand.kernelOperandIdx) {
      kernelStubOperands[*stubOperandIdx] = blockArg;
    }
  }
}

void KernelOperandsMapping::setupKernelStubArgs(Block *kernelStubBody) {
  // The KernelStub already has arguments for the existing kernel operands.
  // But some type signatures must be updated for the tensor objects.
  // And all extra other arguments must be added
  // Fix and cleanup the argument
  for (auto &operandInfos : operands) {
    if (!operandInfos.kernelOperandIdx.has_value())
      continue;

    unsigned idx = operandInfos.kernelOperandIdx.value();
    Value operand = kernelStubOperands[idx];
    // The stub is called with unpacked values (lowerMOGGABI unpacks the async
    // region arguments), so its parameter types are the element types.
    Type operandType = operandInfos.kernelStubOperandType
                           ? operandInfos.kernelStubOperandType
                           : getAsyncElementOrSelf(operand.getType());

    if (idx < kernelStubBody->getNumArguments()) {
      // Update the argument type if needed.
      if (kernelStubBody->getArgument(idx).getType() != operandType)
        kernelStubBody->getArgument(idx).setType(operandType);
    } else {
      // Add a new argument.
      kernelStubBody->addArgument(operandType, operand.getLoc());
    }

    // And now update the mapping
    operandInfos.kernelStubOperand = kernelStubBody->getArgument(idx);
  }
}

MOGGKernelOpLoweringState::MOGGKernelOpLoweringState(
    Operation *op, MGP::MOGraphLoweringState &loweringState)
    : moggOp(op), loweringState(loweringState),
      opLoweringState(loweringState, op), builder(opLoweringState.builder),
      device(MO::getDeviceOrDefaultAny(op)) {}

FailureOr<KernelStubInfo>
MOGGKernelOpLoweringState::createKernelStub(MOGG::KernelOp kernelOp) {
  mlir::MLIRContext *ctx = kernelOp.getContext();
  auto loc = kernelOp.getLoc();

  // Find output and scratch placeholders before taking the body
  MOGG::OutputPlaceholdersOp outputPlaceholdersOp = nullptr;
  MOGG::ScratchPlaceholdersOp scratchPlaceholdersOp = nullptr;
  kernelOp.walk([&](Operation *op) {
    if (auto outputOp = dyn_cast<MOGG::OutputPlaceholdersOp>(op))
      outputPlaceholdersOp = outputOp;
    else if (auto scratchOp = dyn_cast<MOGG::ScratchPlaceholdersOp>(op))
      scratchPlaceholdersOp = scratchOp;
  });

  // Save the number of kernel inputs before we modify the body
  size_t numKernelInputs = kernelOp.getInputs().size();

  // Create the preferred stub name.
  auto stubName = MOGG::getKernelStubName(kernelOp.getFusionInfoAttr(),
                                          MOGG::ABIKind::MOGG);

  // Create the stub outside of the model op to ensure it is properly registered
  // into the SymbolTable of the ModuleOp.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loweringState.getModelOp()->getBlock());

  // First, we create a () -> () function type because we don't yet set the
  // value of certain things like IOKind.
  auto functionType = TypeAttr::get(FunctionType::get(ctx, {}, {}));
  kernelStub = MOGG::KernelStubOp::create(
      builder, loc, stubName, functionType, kernelOp.getFusionInfoAttr(),
      kernelOp.getIsAsyncAttr(), ArrayAttr::get(ctx, {}),
      MOGG::ABIKindAttr::get(ctx, MOGG::ABIKind::MOGG), nullptr, nullptr);
  kernelStub.getBody().takeBody(kernelOp.getExecBody());

  // Wrap the kernel terminator in a WithTraceOp so that Trace instrumentation
  // and error context are handled structurally (not inline in emitMojoMOGGABI).
  {
    auto &bodyBlock = kernelStub.getBodyBlock();
    auto *kernelTerminator = bodyBlock.getTerminator();
    OpBuilder localBuilder(ctx);
    localBuilder.setInsertionPoint(kernelTerminator);
    auto withTraceOp =
        MOGG::WithTraceOp::create(localBuilder, kernelTerminator->getLoc());
    Block &innerBlock = withTraceOp.getBody().emplaceBlock();
    kernelTerminator->moveBefore(&innerBlock, innerBlock.end());
    localBuilder.setInsertionPointAfter(withTraceOp);
    MOGG::OutputOp::create(localBuilder, kernelStub.getLoc(), ValueRange{});
  }

  kernelStub.setDeviceRef(device);

  // Register the stub through the module's symbol table so the cached
  // `SymbolTableAnalysis` stays consistent -- it is never invalidated by the
  // pass manager, so a stub added only via the builder would be invisible to
  // downstream passes that read the cached table. `create` above linked the
  // stub at module-block start; re-insert it through the table to update the
  // table's name map and resolve any collision.
  mlir::SymbolTable &moduleSymbolTable = loweringState.getModuleSymbolTable();
  stubName = moduleSymbolTable.insert(kernelStub);

  return KernelStubInfo{kernelStub, stubName, numKernelInputs,
                        outputPlaceholdersOp, scratchPlaceholdersOp};
}

LogicalResult MOGGKernelOpLoweringState::setupInputOperandsMapping(
    Value chain, ArrayRef<Value> loweredInputOperands,
    MOGG::KernelIOContractDeriver &contractDeriver) {
  // 1. Add the chain with the input signature. (Only pass it to the region, not
  // the stub)
  operandsMapping.addOperand(chain, /*isMGPRegionOperand=*/true,
                             /*isKernelStubOperand=*/false,
                             /*sig=*/InOutSignatureAttr::kIn);

  // 2. Add the inputs with the input or mutable input signature.
  //
  // The per-input trait contract is derived from the kernel declaration
  // through each stub argument's use by the kernel call, so
  // operand-list transforms (fusion, dedup) cannot invalidate it (GEX-4071).
  for (auto &&[origOperand, loweredInputOperand] : llvm::zip_first(
           kernelStub.getBody().getArguments(), loweredInputOperands)) {
    auto sig = isa<MO::BufferType>(origOperand.getType())
                   ? InOutSignatureAttr::kMut
                   : InOutSignatureAttr::kIn;
    operandsMapping.addOperand(loweredInputOperand, /*isMGPRegionOperand=*/true,
                               /*isKernelStubOperand=*/true, sig);
    if (MO::dynCastFromTensorLike(origOperand.getType())) {
      FailureOr<MOGG::TensorArgContractAttr> contract =
          contractDeriver.derive(origOperand);
      if (failed(contract))
        return failure();
      tensorOperandInfos.push_back(
          TensorOperandInfos{.stubValue = origOperand,
                             .operandIdx = operandsMapping.getNumOperands() - 1,
                             .isOutput = false,
                             .traitContract = *contract});
    }
  }
  return success();
}

Value MOGGKernelOpLoweringState::buildMojoIOTensorObject(
    Value bufferValue, const TensorOperandInfos &tensorInfos) {
  mlir::MLIRContext *ctx = moggOp->getContext();
  auto loc = moggOp->getLoc();
  Value stubValue = tensorInfos.stubValue;

  auto asTensor = MO::castFromTensorLike(stubValue.getType());
  auto dynamicDimensions = operandsMapping.getTensorDynamicDimensions(
      asTensor, /*isInsideStub=*/false);

  MO::IOKind io = tensorInfos.isOutput ? MO::IOKind::Output : MO::IOKind::Input;
  bool mut = tensorInfos.isOutput || isa<MO::BufferType>(stubValue.getType());

  // Build the new tensor type.
  auto ioAttr = MO::IOKindAttr::get(ctx, io);
  auto mutAttr = BoolAttr::get(ctx, mut);
  MO::TensorType newTensorTy = asTensor.cloneWith(ioAttr).cloneWithMut(mutAttr);

  // A trait-bound argument carries an `#mogg.tensor_arg` contract. Build
  // `mogg.tensor_arg.__init__` carrying that contract instead of the legacy
  // `mogg.tensor.__init__`. The contract is the single source of truth for the
  // role: nothing downstream re-derives the role from the result tensor type's
  // `{io, mut}`. Both representations use the same io/mut-cloned `newTensorTy`;
  // the `{io, mut}` carried on the trait result type is vestigial -- do not
  // read it back as the role. The `$device_context` operand is left
  // unpopulated here; the GPU emit path wires it later.
  if (tensorInfos.traitContract) {
    return MOGG::TensorArgInitOp::create(
               builder, loc, newTensorTy, bufferValue, dynamicDimensions,
               /*device_context=*/Value(), tensorInfos.traitContract)
        .getResult();
  }

  // Legacy `ManagedTensorSlice` representation.
  return MOGG::TensorInitOp::create(builder, loc, newTensorTy, bufferValue,
                                    dynamicDimensions, mutAttr, ioAttr)
      .getResult();
}

LogicalResult MOGGKernelOpLoweringState::findExtraKernelArguments(
    Value chainValue, ArrayRef<KGEN::ParamDeclRefAttr> usedKernelParameters,
    MOGG::KernelIOContractDeriver &contractDeriver) {
  llvm::DenseSet<StringAttr> paramsInKernelStub;
  auto markTensorDimsInStub = [&](MO::TensorType asTensor) {
    for (auto dim : asTensor.getDims())
      if (auto dimParamAttr = dyn_cast<KGEN::ParamDeclRefAttr>(dim))
        paramsInKernelStub.insert(dimParamAttr.getName());
  };

  MOGG::OutputPlaceholdersOp outputPlaceholdersOp;
  MOGG::ScratchPlaceholdersOp scratchPlaceholdersOp;
  llvm::SmallVector<MOGG::DeviceContextPlaceholderOp> devicePlaceholderOps;

  // Walk through the IR first to find all extra needed operands: parameters, or
  // special ops.
  kernelStub.walk([&](mlir::Operation *stubOp) {
    if (auto bindOp = dyn_cast<MOGG::BindLambdaOp>(stubOp)) {
      // We only need to get the dynamic dimensions when materializing an
      // intermediate tensor for the fused kernel operand. This happens only for
      // Input and AffineView fusion.
      if (bindOp.getBindKind() != MOGG::BindLambdaKind::InputLambda &&
          !bindOp.hasAffineViewOperation())
        return;

      Value fusedOutput = bindOp.getOutput();
      // Ensure the dynamic dimensions of the tensor are available in the stub.
      markTensorDimsInStub(MO::castFromTensorLike(fusedOutput.getType()));
    }

    else if (auto tensorShapeOp = dyn_cast<MOGG::TensorGetShapeOp>(stubOp)) {
      // The dynamic dimensions of TensorGetShapeOp must be lowered.
      markTensorDimsInStub(MO::castFromTensorLike(tensorShapeOp.getTensor()));
    }

    else if (auto getDimOp = dyn_cast<MOGG::GetDimOp>(stubOp)) {
      paramsInKernelStub.insert(getDimOp.getDimension());
    }

    // The OutputPlaceholdersOp must be replaced with the output buffers.
    else if (auto outputOp = dyn_cast<MOGG::OutputPlaceholdersOp>(stubOp)) {
      outputPlaceholdersOp = outputOp;
    }

    // The ScratchPlaceholdersOp must be replaced with the scratchpad buffers.
    else if (auto scratchOp = dyn_cast<MOGG::ScratchPlaceholdersOp>(stubOp)) {
      scratchPlaceholdersOp = scratchOp;
    }

    // The DevicePlaceholderOp must be replaced with the device context object.
    else if (auto deviceOp =
                 dyn_cast<MOGG::DeviceContextPlaceholderOp>(stubOp)) {
      devicePlaceholderOps.push_back(deviceOp);
    }

    // Handle dynamic dimensions from result types of all operations.
    // This is needed for operations like mogg.experimental.kernel that have
    // tensor result types with dynamic dimensions.
    for (Type resultType : stubOp->getResultTypes()) {
      if (auto tensorType = MO::dynCastFromTensorLike(resultType)) {
        markTensorDimsInStub(tensorType);
      }
    }
  });

  // Allocate and pass the output buffers to the kernels if needed.
  if (outputPlaceholdersOp) {
    auto loweredOutputsStorageOr =
        opLoweringState.createOutputStorage(chainValue);
    if (failed(loweredOutputsStorageOr))
      return failure();
    loweredOutputsStorage = *std::move(loweredOutputsStorageOr);
    // Per-output trait contracts are derived from the kernel declaration
    // through each output placeholder's use of the kernel call terminator
    // (GEX-4071).
    for (auto [outputStorage, value] :
         llvm::zip(loweredOutputsStorage, outputPlaceholdersOp.getResults())) {
      // Add the kernel operand for the output buffer.
      operandsMapping.addOperand(outputStorage.buffer,
                                 /*isMGPRegionOperand=*/true,
                                 /*isKernelStubOperand=*/true,
                                 InOutSignatureAttr::kOut, /*keyValue=*/value);
      FailureOr<MOGG::TensorArgContractAttr> contract =
          contractDeriver.derive(value);
      if (failed(contract))
        return failure();
      tensorOperandInfos.push_back(
          TensorOperandInfos{.stubValue = value,
                             .operandIdx = operandsMapping.getNumOperands() - 1,
                             .isOutput = true,
                             .traitContract = *contract});
    }
  }

  // Allocate and pass the scratch buffers to the kernels if needed.
  if (scratchPlaceholdersOp) {

    // Add the kernel operand for the scratchpad buffer.
    // Use the none signature because it is only used.
    for (auto value : scratchPlaceholdersOp.getResults()) {

      auto specOr = opLoweringState.createTensorSpec(
          cast<MO::TensorType>(value.getType()));
      if (failed(specOr))
        return failure();

      auto bufOr =
          opLoweringState.createBuffer(chainValue, *specOr, value.getType());
      if (failed(bufOr))
        return failure();

      operandsMapping.addOperand(bufOr->buffer,
                                 /*isMGPRegionOperand=*/true,
                                 /*isKernelStubOperand=*/true,
                                 InOutSignatureAttr::kNone, /*keyValue=*/value);
      tensorOperandInfos.push_back(
          TensorOperandInfos{.stubValue = value,
                             .operandIdx = operandsMapping.getNumOperands() - 1,
                             .isOutput = true});
    }
  }

  // Turn parameters into values and pass them to the kernel.
  for (auto param : usedKernelParameters) {
    auto symbolicDimOr = loweringState.lowerExpression(
        opLoweringState.opBeingLowered->getParentRegion(), opLoweringState.loc,
        param);
    Value symbol = symbolicDimOr.takeValue();

    operandsMapping.addOperand(
        param, symbol,
        /*=isKernelStubOperand*/ paramsInKernelStub.contains(param.getName()));
  }

  // Add the DeviceContext argument
  for (auto devicePlaceholderOp : devicePlaceholderOps) {
    Value moggContext = devicePlaceholderOp.getResult();
    Value mgpContext =
        loweringState.getExecuteDevice(devicePlaceholderOp.getDevice());
    operandsMapping.addOperand(mgpContext,
                               /*isMGPRegionOperand=*/true,
                               /*isKernelStubOperand=*/true,
                               InOutSignatureAttr::kNone,
                               /*keyValue=*/moggContext,
                               /*stubOperandType=*/moggContext.getType());
  }

  return success();
}

void MOGGKernelOpLoweringState::findExtraKernelStubOnlyArguments() {
  SmallVector<Attribute> newKernelResources;

  kernelStub.walk([&](MO::ConstantOp constantOp) {
    // Add the constant resource to the kernel, and create an MGP Buffer.
    auto constantAttr = *loweringState.lowerElementsAttr(constantOp.getValue());
    newKernelResources.push_back(constantAttr);
    Value bufferConstant = MGP::BufferConstantOp::create(
        builder, constantOp->getLoc(), cast<ElementsAttr>(constantAttr));
    operandsMapping.addOperand(bufferConstant,
                               /*isMGPRegionOperand=*/false,
                               /*isKernelStubOperand=*/true,
                               InOutSignatureAttr::kNone,
                               /*keyValue=*/constantOp.getResult());
  });

  // Attach all new resources to the constant kernel.
  if (!newKernelResources.empty()) {
    SmallVector<Attribute> allKernelResources(
        mgpRegion.getConstantResources().begin(),
        mgpRegion.getConstantResources().end());
    allKernelResources.append(std::move(newKernelResources));
    mgpRegion.setConstantResourcesAttr(
        ArrayAttr::get(mgpRegion->getContext(), allKernelResources));
  }
}

void MOGGKernelOpLoweringState::lowerBindLambdaOp(MOGG::BindLambdaOp bindOp) {
  mlir::MLIRContext *ctx = moggOp->getContext();
  auto loc = moggOp->getLoc();

  // Only replace I/O lambdas.
  if (bindOp.getBindKind() != MOGG::BindLambdaKind::InputLambda &&
      bindOp.getBindKind() != MOGG::BindLambdaKind::OutputLambda)
    return;

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
  for (Operation *user : lambdaVal.getUsers()) {
    if (isa<MOGG::TensorFusedOp>(user))
      return;
  }
  if (bindOp.getFusibleTensors().empty())
    return;

  bool lambdaHasCast = MO::castFromTensorLike(tensorVal.getType()).getDtype() !=
                       MO::castFromTensorLike(lambdaVal.getType()).getDtype();

  // We always pass an empty tensor to the kernels for fused input tensors or
  // output lambdas with a cast.
  bool substituteWithEmptyTensor =
      bindOp.getBindKind() == MOGG::BindLambdaKind::InputLambda ||
      lambdaHasCast;

  builder.setInsertionPointAfter(bindOp);

  auto mgpBufferType = MGP::DeviceBufferType::get(device);

  if (substituteWithEmptyTensor) {
    Value emptyBuffer =
        MOGG::EmptyBufferOp::create(builder, bindOp->getLoc(), mgpBufferType);
    auto dynamicDimensions = operandsMapping.getTensorDynamicDimensions(
        MO::castFromTensorLike(lambdaVal.getType()),
        /*isInsideStub=*/true);

    MO::TensorType tensorType = MO::castFromTensorLike(lambdaVal.getType());
    tensorType =
        tensorType.cloneWith(MO::IOKindAttr::get(ctx, MO::IOKind::Input))
            .cloneWithMut(BoolAttr::get(ctx, false));
    tensorVal = MOGG::TensorInitOp::create(
        builder, loc, tensorType, emptyBuffer, dynamicDimensions,
        /*mut=*/BoolAttr::get(ctx, false),
        /*io=*/MO::IOKindAttr::get(ctx, MO::IOKind::Input));
  }

  else if (bindOp.hasAffineViewOperation()) {
    // The tensor needs to be reshaped to match the expected kernel operand
    // shape.
    auto dynamicDimensions = operandsMapping.getTensorDynamicDimensions(
        MO::castFromTensorLike(lambdaVal.getType()),
        /*isInsideStub=*/true);
    tensorVal = MOGG::TensorReshapeOp::create(builder, loc, lambdaVal.getType(),
                                              tensorVal, dynamicDimensions);
  }

  // Build the Fused tensor object.
  auto fusedTensorOp = MOGG::TensorFusedOp::create(
      builder, bindOp->getLoc(), lambdaVal.getType(), lambdaVal, tensorVal);

  // And finally replace it for the users of the lambda.
  lambdaVal.replaceAllUsesExcept(fusedTensorOp.getResult(), fusedTensorOp);
}

LogicalResult MOGGKernelOpLoweringState::lowerConstant(
    MO::ConstantOp constantOp, MOGG::KernelIOContractDeriver &contractDeriver) {
  mlir::MLIRContext *ctx = moggOp->getContext();
  auto loc = moggOp->getLoc();
  Value result = constantOp.getResult();

  // Per-use contract; see KernelIOContractDeriver::deriveUse.
  SmallVector<OpOperand *> traitUses, legacyUses;
  for (OpOperand &use : result.getUses()) {
    FailureOr<MOGG::TensorArgContractAttr> contract =
        contractDeriver.deriveUse(use);
    if (failed(contract))
      return failure();
    (*contract ? traitUses : legacyUses).push_back(&use);
  }

  // Replace the op with the corresponding kernel arguments.
  Value bufferValue = operandsMapping.get(result, /*isInsideStub=*/true);

  builder.setInsertionPoint(constantOp);
  MO::TensorType tensorType = MO::castFromTensorLike(result.getType());
  tensorType = tensorType.cloneWith(MO::IOKindAttr::get(ctx, MO::IOKind::Input))
                   .cloneWithMut(BoolAttr::get(ctx, false));
  assert(tensorType.hasStaticShape() &&
         "in-stub constants materialize no runtime dimensions");

  if (!traitUses.empty()) {
    // Neutral input capture: the execute declaration, not this init, selects
    // the dense vs fused wrapper.
    auto captureContract = MOGG::TensorArgContractAttr::get(
        ctx, MOGG::TensorArgRoleKind::input, /*fused=*/false,
        MOGG::TensorArgAccessKind::none, MOGG::TensorArgAccessKind::none);
    Value capture = MOGG::TensorArgInitOp::create(
        builder, loc, tensorType, bufferValue,
        /*dynamicDimensions=*/ArrayRef<Value>{},
        /*device_context=*/Value(), captureContract);
    for (OpOperand *use : traitUses)
      use->set(capture);
  }
  if (!legacyUses.empty()) {
    Value slice = MOGG::TensorInitOp::create(
        builder, loc, tensorType, bufferValue,
        /*dynamicDimensions=*/ArrayRef<Value>{},
        /*mut=*/BoolAttr::get(ctx, false),
        /*io=*/MO::IOKindAttr::get(ctx, MO::IOKind::Input));
    for (OpOperand *use : legacyUses)
      use->set(slice);
  }
  constantOp->erase();
  return success();
}

void MOGGKernelOpLoweringState::lowerGetDimOp(MOGG::GetDimOp getDimOp) {
  auto dimValue =
      operandsMapping.get(getDimOp.getDimension(), /*isInsideStub=*/true);
  getDimOp.getResult().replaceAllUsesWith(dimValue);
  getDimOp->erase();
}

void MOGGKernelOpLoweringState::lowerTensorGetShapeOp(
    MOGG::TensorGetShapeOp getShapeOp) {
  // Get the dynamic dimensions
  auto asTensor = MO::castFromTensorLike(getShapeOp.getTensor());
  auto dynamicDimensions =
      operandsMapping.getTensorDynamicDimensions(asTensor,
                                                 /*isInsideStub=*/true);

  // Build a ShapeInitOp to materialize the new shape mojo value
  builder.setInsertionPoint(getShapeOp);
  Value shapeValue =
      MOGG::ShapeInitOp::create(builder, getShapeOp.getLoc(), dynamicDimensions,
                                asTensor.getTypedShapeAttr());

  getShapeOp.getResult().replaceAllUsesWith(shapeValue);
  getShapeOp->erase();
}

void MOGGKernelOpLoweringState::lowerKernelOpWithOperandsMapping(
    Operation *op) {
  for (Value val : op->getResults()) {
    Value newVal = operandsMapping.get(val, /*isInsideStub=*/true);
    val.replaceAllUsesWith(newVal);
  }
  op->erase();
}

void MOGGKernelOpLoweringState::materializeInStubTensorFusedOps() {
  kernelStub.walk([&](mlir::Operation *op) {
    if (auto bindOp = dyn_cast<MOGG::BindLambdaOp>(op))
      lowerBindLambdaOp(bindOp);
  });
}

void MOGGKernelOpLoweringState::wireTraitFusedCapturesToStubArgs() {
  llvm::DenseMap<Value, const TensorOperandInfos *> infoByStubValue;
  for (const TensorOperandInfos &tensorInfos : tensorOperandInfos)
    infoByStubValue[tensorInfos.stubValue] = &tensorInfos;

  // Wire each fused trait operand through its own kernel I/O value: a bind
  // lambda's fusible that maps to a fused trait contract is rewired to that
  // operand's in-stub buffer capture. Keying on the fusible's identity (not
  // an ordinal among fused operands) keeps every fusible of a multi-operand
  // producer on its own capture.
  kernelStub.walk([&](MOGG::TensorFusedOp fusedOp) {
    MOGG::BindLambdaOp bindOp = fusedOp.getBindLambdaOp();
    if (!bindOp)
      return;
    for (Value fusible : bindOp.getFusibleTensors()) {
      const TensorOperandInfos *info = infoByStubValue.lookup(fusible);
      if (!info || !info->traitContract || !info->traitContract.getFused())
        continue;

      Value stubCapture =
          operandsMapping.get(info->operandIdx, /*isInsideStub=*/true);

      if (bindOp.getBindKind() == MOGG::BindLambdaKind::OutputLambda)
        fusedOp.getTensorMutable().assign(stubCapture);

      fusible.replaceUsesWithIf(stubCapture, [&](OpOperand &use) {
        return use.getOwner() == bindOp.getOperation();
      });
    }
  });
}

LogicalResult MOGGKernelOpLoweringState::lowerKernelStub(
    MOGG::KernelIOContractDeriver &contractDeriver) {
  // Walk through the IR and lower some ops to MGP.
  auto walkResult = kernelStub.walk([&](mlir::Operation *op) {
    if (auto bindOp = dyn_cast<MOGG::BindLambdaOp>(op))
      lowerBindLambdaOp(bindOp);
    else if (auto constantOp = dyn_cast<MO::ConstantOp>(op)) {
      if (failed(lowerConstant(constantOp, contractDeriver)))
        return WalkResult::interrupt();
    } else if (auto tensorGetShapeOp = dyn_cast<MOGG::TensorGetShapeOp>(op))
      lowerTensorGetShapeOp(tensorGetShapeOp);
    else if (auto getDimOp = dyn_cast<MOGG::GetDimOp>(op))
      lowerGetDimOp(getDimOp);
    else if (isa<MOGG::OutputPlaceholdersOp, MOGG::ScratchPlaceholdersOp,
                 MOGG::DeviceContextPlaceholderOp>(op))
      lowerKernelOpWithOperandsMapping(op);
    return WalkResult::advance();
  });
  return failure(walkResult.wasInterrupted());
}

LogicalResult
MOGGKernelOpLoweringState::lowerKernelOp(MOGG::KernelOp moggKernelOp) {
  mlir::MLIRContext *ctx = moggKernelOp.getContext();
  auto loc = moggKernelOp.getLoc();

  SmallVector<KGEN::ParamDeclRefAttr> usedKernelParameters =
      M::MOGG::getKernelUsedParameters(moggKernelOp);

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

  // Determine the return type for the stub
  Type stubReturnType = nullptr;
  auto terminator = moggKernelOp.getTerminator();
  if (!resultTypes.empty()) {
    stubReturnType = resultTypes.front();
  } else if (auto regOutput = terminator.getKernelReturnOutput()) {
    stubReturnType = regOutput;
  }

  // Create the stub by outlining the kernel body, naming it after the ops it
  // fuses (e.g. "relu_add_mogg") so a kernel error points at the work it
  // runs. `mogg.kernel` always carries fusionInfo.
  auto stubInfoOr = createKernelStub(moggKernelOp);
  if (failed(stubInfoOr))
    return failure();
  auto stubInfo = *stubInfoOr;
  StringAttr stubName = stubInfo.stubName;

  // This is a list of input operands in MGP but without the chain.
  auto maybeInChain = moggKernelOp.maybeGetInChain();
  auto loweredInputOperands = opLoweringState.getLoweredOperands();
  Value chain = loweringState.getCurrentExecuteChain();
  if (maybeInChain) {
    Value lowered = loweringState.lookupExecuteLoweredValue(*maybeInChain);
    ASSERT_STREAM(lowered, << "no lowered value for kernel input chain: "
                           << *maybeInChain);
    chain = lowered;
    loweredInputOperands.pop_back();
  }
  chain = opLoweringState.getGuardChain(chain);

  // One deriver serves inputs, outputs, and in-stub constants so declaration
  // adaptors are decoded once during this lowering.
  MOGG::KernelIOContractDeriver contractDeriver(
      moggKernelOp->getParentOfType<mlir::ModuleOp>());
  if (failed(setupInputOperandsMapping(chain, loweredInputOperands,
                                       contractDeriver)))
    return failure();

  // 3. Pass and map all other kernel arguments (dynamic dimensions, device
  // context, etc).
  if (failed(findExtraKernelArguments(chain, usedKernelParameters,
                                      contractDeriver)))
    return failure();

  // 4. Add a chain to the result types.
  SmallVector<mlir::Type> mgpKernelResTypes;
  mgpKernelResTypes.push_back(chain.getType());

  // 5. The original kernel could also return an opaque type which is allocated
  // inside of its own body. If that's the case, add it to the output of the
  // mogg.generic.compiled.region op since it is not in DPS form. The current
  // contract is that if a kernel op returns an opaque type, it must have been
  // allocated internally. Tensors can't be returned in a non-DPS form.
  // We use MO::OpaqueType here to match what the stub returns.
  if (!resultTypes.empty()) {
    mgpKernelResTypes.push_back(resultTypes.front());
  }

  // 6. The original kernel may be setting some parameters (shape functions do
  // just that). These will returned from the mogg.generic.compiled.region.
  for (size_t i = 0; i < moggKernelOp.getOutputParamDecls().size(); ++i)
    mgpKernelResTypes.push_back(
        moggKernelOp.getOutputParamDecls()[i].getType());

  mgpRegion = MOGG::GenericCompiledRegionOp::create(
      builder, loc, TypeRange{mgpKernelResTypes},
      operandsMapping.getMGPRegionOperands(),
      moggKernelOp.getOutputParamDeclsAttr(),
      InOutSignatureAttr::get(ctx,
                              operandsMapping.getMGPRegionOperandsSemantics()),
      moggKernelOp.getFusionInfoAttr(), // fusionInfo
      moggKernelOp.getIsAsyncAttr(),    // isAsync
      ArrayAttr::get(ctx, {}),          // KernelOp has no constant resources
      // Sequence fuse GPU launching to reduce primitive launch overhead.
      BoolAttr::get(ctx, !MO::isOnHost(moggKernelOp)), // canFuseIntoOther
      BoolAttr::get(ctx, true),                        // canBeFusedInto
      BoolAttr::get(ctx, false),                       // canDuplicate
      BoolAttr::get(ctx, true));                       // isStructuredKernel
  mgpRegion.setDeviceRef(device);

  // Add a body to the created operation and mirror its operands with block
  // arguments.
  mgpRegionBody = &mgpRegion.getBody().emplaceBlock();
  operandsMapping.setupMGPRegionArgs(mgpRegionBody);

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

  // Convert all tensor operands to buffer captures for the stub boundary.
  for (const auto &tensorInfos : tensorOperandInfos) {
    Value regionArg =
        operandsMapping.get(tensorInfos.operandIdx, /*isInsideStub=*/false);

    builder.setInsertionPointToEnd(mgpRegionBody);
    Value captureValue = buildMojoIOTensorObject(regionArg, tensorInfos);
    operandsMapping.remapKernelStubOperand(tensorInfos.operandIdx,
                                           captureValue);
  }

  // Find extra KernelStub only arguments
  findExtraKernelStubOnlyArguments();

  // Setup the KernelStub region and call.
  operandsMapping.setupKernelStubArgs(&kernelStub.getBody().front());
  // `lowerBindLambdaOp` may materialize tensors with dynamic dimensions inside
  // the stub. Do this after stub arguments are finalized so those dimensions
  // have stub-side mappings.
  materializeInStubTensorFusedOps();
  wireTraitFusedCapturesToStubArgs();
  builder.setInsertionPointToEnd(mgpRegionBody);
  llvm::ArrayRef<Value> stubCallOperands =
      operandsMapping.getKernelStubOperands();
  assert(stubCallOperands.size() ==
         (kernelStub.getBodyBlock().getNumArguments()));

  // For trait-based arguments, stamp per-stub-arg buffer-capture comptime
  // metadata for tensor operands (region passes `_MoggTensorBufferCapture`;
  // execute-facing trait args are assembled inside the stub). Shape-function
  // no-role args keep the legacy generic `DenseTensor` stub parameter via
  // `mogg.trait_stub_arg_bounds`.
  {
    size_t numStubArgs = kernelStub.getBodyBlock().getNumArguments();
    SmallVector<Attribute> captureAttrs(numStubArgs, builder.getStringAttr(""));
    SmallVector<std::string> shapeBounds(numStubArgs);
    bool anyCapture = false;
    bool anyShapeBound = false;
    for (const auto &tensorInfos : tensorOperandInfos) {
      if (!tensorInfos.traitContract)
        continue;
      std::optional<unsigned> stubArgIdx =
          operandsMapping.getKernelStubArgIndex(tensorInfos.operandIdx);
      if (!stubArgIdx || *stubArgIdx >= numStubArgs)
        continue;
      if (tensorInfos.traitContract.getRole() ==
          MOGG::TensorArgRoleKind::none) {
        shapeBounds[*stubArgIdx] = "DenseTensor";
        anyShapeBound = true;
        continue;
      }
      auto asTensor = MO::castFromTensorLike(tensorInfos.stubValue.getType());
      std::string dtypeExpr =
          MOGG::EmitMojoState::mlirDTypeToMojoType(asTensor.getDtype()).text;
      captureAttrs[*stubArgIdx] = builder.getDictionaryAttr({
          {builder.getStringAttr("dtype"), builder.getStringAttr(dtypeExpr)},
      });
      anyCapture = true;
    }
    if (anyCapture)
      kernelStub.setTraitStubBufferCaptureAttr(
          builder.getArrayAttr(captureAttrs));
    if (anyShapeBound) {
      SmallVector<Attribute> boundAttrs;
      for (const std::string &b : shapeBounds)
        boundAttrs.push_back(builder.getStringAttr(b));
      kernelStub.setTraitStubArgBoundsAttr(builder.getArrayAttr(boundAttrs));
    }
  }

  // Create the call to the outlined kernel stub and use whatever return value
  // (if any) in the output op of the new kernel.
  callStubOp = MOGG::CallStubOp::create(
      builder, loc, stubReturnType ? TypeRange{stubReturnType} : TypeRange{},
      FlatSymbolRefAttr::get(stubName), stubCallOperands,
      /*affinity_device_id=*/mlir::IntegerAttr{});
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
      Type dimType = MOSH::ShapeAttr::getDimType(builder.getContext());
      for (size_t i = 0; i < numDims; ++i) {
        auto getItemOp = MOGG::ShapeResultGetItemOp::create(
            builder, loc, dimType, result,
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

  // Lower remaining stub ops (placeholders, constants, etc.) for EmitMojo.
  if (failed(lowerKernelStub(contractDeriver)))
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
      auto asDimension = MGP::indexToDimension(
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

  loweringState.lowerMOGGABI(mgpRegion);

  // We can now safely set the function type because we have added all arguments
  // and attributes.
  kernelStub.setFunctionType(
      FunctionType::get(ctx, kernelStub.getBodyBlock().getArgumentTypes(),
                        stubReturnType ? stubReturnType : TypeRange{}));

  return success();
}
