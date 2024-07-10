//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Interfaces/LoweringState.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphRT/MIPDialect/MIPDialect.h"
#include "Support/AssertStream.h"
#include <llvm/ADT/STLExtras.h>
#include <mlir/Support/LLVM.h>

using namespace M;
using namespace M::MGP;

//===----------------------------------------------------------------------===//
// MOGraphLoweringState
//===----------------------------------------------------------------------===//

CompileDiagnosticOr<DeviceSpecCollectionAttr>
MOGraphLoweringState::getDeviceSpecs() {
  ErrorOr<DeviceSpecCollectionAttr> deviceSpecsOr =
      MO::getDeviceSpecsOrDefault(graph);
  if (deviceSpecsOr)
    return graph.emitOpError(
               "unable to determine the devices specs for graph: ")
           << deviceSpecsOr.getError() << "\n";
  return *deviceSpecsOr;
}

TypedValue<MGP::TensorSpecType>
MOGraphLoweringState::getShape(Location loc, Value loweredTensor,
                               DeviceRefAttr device) {
  OpBuilder &builder = getExecuteOpBuilder();
  return getBufferBuilder(device)->tensorExtractTensorSpec(
      builder, loc, getExecuteMgpContext(), loweredTensor,
      /*isShapeOnly=*/true);
}

MGP::CpuTensorSpecDimsDynamicOp
MOGraphLoweringState::getDynamicDims(Location loc, Value loweredTensor,
                                     DeviceRefAttr device) {
  OpBuilder &builder = getExecuteOpBuilder();
  // NOTE: We could use getShape here, however usually we'll also require the
  // loweredTensor's buffer in which case we'll be emitting a full
  // tensor.extract op anyway.
  TypedValue<MGP::TensorSpecType> tensorSpec =
      getBufferBuilder(device)->tensorExtractTensorSpec(
          builder, loc, getExecuteMgpContext(), loweredTensor);
  return builder.create<MGP::CpuTensorSpecDimsDynamicOp>(
      loc, /*spec=*/tensorSpec, /*rawDims=*/tensorSpec.getType().getRawDims());
}

ErrorOr<bool> MOGraphLoweringState::lowerArg(Region *parentRegion,
                                             BlockArgument &inArg,
                                             BlockArgument &outArg) {
  bool hasParams = false;
  Location loc = inArg.getLoc();
  if (auto argType = MO::dynCastFromTensorLike(inArg.getType())) {
    DeviceRefAttr argDevice = MO::getDeviceOrDefault(graph, argType);
    if (argType.hasStaticShape()) {
      // No parameters to deal with.
      /* no-op */
    } else if (argType.hasUnknownShape()) {
      // Also no parameters to deal with, however this argument won't be
      // usable by any MO ops.
      /* no-op */
    } else if (auto inShapeParam =
                   dyn_cast<KGEN::ParamDeclRefAttr>(argType.getShapeAttr())) {
      // Argument has dynamic rank. The parameter needs to be represented by
      // the runtime MGP 'shape' of the tensor, ie a tensor_spec with unknown
      // dtype. However, we only need to capture the mapping for the first
      // occurrence of the parameter.
      hasParams = true;
      if (!params.lookupLowered(parentRegion, inShapeParam))
        params.emplaceLowered(parentRegion, inShapeParam,
                              getShape(loc, outArg, argDevice));
    } else {
      // Argument has at least one dynamic dimension. The parameters need to
      // be represented by the corresponding runtime dimensions. Since
      // parameters may be shared we only update the parameter map for the
      // first occurrence of each parameter. However, since dimensions are
      // extract as a group, we always do the extraction, and rely on DCE to
      // cleanup if all the parameters turn out to have already been mapped.
      hasParams = true;
      MGP::CpuTensorSpecDimsDynamicOp dynDimsOp =
          getDynamicDims(loc, outArg, argDevice);
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
        if (!params.lookupLowered(parentRegion, inDimParam))
          params.emplaceLowered(parentRegion, inDimParam, outDim);
    }
  }
  return hasParams;
}

ErrorOr<Value> MOGraphLoweringState::lowerExpression(Region *parentRegion,
                                                     Location loc,
                                                     TypedAttr attr) {
  auto newValOr = params.lowerExpression(
      getExecuteOpBuilder(), loc, parentRegion, attr,
      [this](auto &&...args) { return this->lowerExpressionImpl(args...); });

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

ErrorOrSuccess MOGraphLoweringState::lowerParamDecl(
    Region *parentRegion, KGEN::ParamDeclAttr declAttr, Value value) {
  Value maybePlaceholder = params.lowerParamDecl(parentRegion, declAttr, value);

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

ErrorOr<SmallVector<Value>> MOGraphLoweringState::lowerShapeAttrDynDims(
    Region *parentRegion, MOSH::ShapeAttr shapeAttr, Location loc) {
  SmallVector<Value> runtimeDims;
  for (TypedAttr dim : shapeAttr.getValues()) {
    if (!isa<IntegerAttr>(dim)) {
      auto vOr = lowerExpression(parentRegion, loc, dim);
      if (vOr.isError())
        return vOr.takeError();
      runtimeDims.emplace_back(std::move(*vOr));
    }
  }
  return runtimeDims;
}

int64_t MOGraphLoweringState::extractShapeAttrStaticDimsProduct(
    MOSH::ShapeAttr shapeAttr, Location loc) {
  int64_t product = 1;
  for (TypedAttr dim : shapeAttr.getValues()) {
    if (auto intDim = dyn_cast<IntegerAttr>(dim)) {
      product *= intDim.getValue().getSExtValue();
    }
  }
  return product;
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

/// sequenced assert lowering helper functions
namespace {

bool isMipOp(Operation *op) { return isa<MIP::MIPDialect>(op->getDialect()); }

DenseSet<Operation *> getAncestors(Operation *root) {

  DenseSet<Operation *> ancestors;
  SmallVector<Operation *> worklist;

  worklist.push_back(root);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    if (!op || !isMipOp(op))
      continue;

    ancestors.insert(op);

    for (Value val : op->getOperands()) {
      Operation *defOp = val.getDefiningOp();

      if (!ancestors.contains(defOp))
        worklist.push_back(defOp);
    }
  }

  return ancestors;
}

SmallVector<OpOperand *>
getBoundaryOpOperands(Operation *root,
                      const llvm::DenseSet<Operation *> &prevAncestors) {

  DenseSet<OpOperand *> boundaryOpOperands;
  SmallVector<Operation *> worklist;

  worklist.push_back(root);

  while (!worklist.empty()) {

    Operation *op = worklist.pop_back_val();

    if (!op || !isMipOp(op) || prevAncestors.contains(op))
      continue;

    for (auto &&opOperand : op->getOpOperands()) {
      auto defOp = opOperand.get().getDefiningOp();
      const bool isConstant = defOp->hasTrait<OpTrait::ConstantLike>();

      if (prevAncestors.contains(defOp) && !isConstant)
        boundaryOpOperands.insert(&opOperand);
      else
        worklist.push_back(defOp);
    }
  }

  return {boundaryOpOperands.begin(), boundaryOpOperands.end()};
}

/// Given the operand chain and condition to an assertion gather all
/// expressions rooted at the current condition and the condition from the
/// previous producing assertion. The values (OpOperands really) that cross
/// from the previous assertion's tree to the current assertion's tree.
SmallVector<OpOperand *> getSequencedAssertBoundaryOperands(
    Value cond, const SmallVector<Operation *> &parentAsserts) {

  DenseSet<Operation *> allAncestors;

  for (auto parentAssert : parentAsserts) {
    auto prevAssertCondRoot =
        dyn_cast<MGP::AssertOp>(parentAssert).getCond().getDefiningOp();

    auto prevAncestors = getAncestors(prevAssertCondRoot);

    allAncestors.insert(prevAncestors.begin(), prevAncestors.end());
  }

  return getBoundaryOpOperands(cond.getDefiningOp(), allAncestors);
}

SmallVector<Operation *>
getParentAsserts(Operation *currentAssert,
                 const DenseMap<Operation *, Operation *> &parentMap) {
  SmallVector<Operation *> parents = {currentAssert};

  auto it = parentMap.find(currentAssert);
  bool hasParent = it != parentMap.end();

  while (hasParent) {
    currentAssert = it->second;
    parents.push_back(currentAssert);
    it = parentMap.find(currentAssert);
    hasParent = it != parentMap.end();
  }

  return parents;
}

} // namespace

ErrorOr<Value> MOGraphLoweringState::lowerExpressionImpl(
    ParameterLoweringState &state, OpBuilder &builder, Location loc,
    Region *parentRegion, TypedAttr attr, ValueRange operands) {
  Operation *newOp;
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    newOp = lowerIntAttr(builder, loc, intAttr);
  } else if (auto opAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr)) {
    return lowerParamOperator(builder, loc, opAttr, operands);
  } else if (auto opAttr = dyn_cast<KGEN::MLIROpAttr>(attr)) {
    StringAttr opName = opAttr.getName();
    if (opName != MO::AssertOp::getOperationName()) {
      return Error(Twine("cannot lower mlir operation '") + opName.strref() +
                   "' in parameter expression");
    }

    auto [cond, maybeParentAssert] = [&]() -> std::pair<Value, Operation *> {
      bool hasChain = isa<MEF::ChainType>(operands[0].getType());

      if (hasChain) {
        Value chain = operands[0];
        Value cond = operands[1];

        ASSERT_STREAM(
            isa<MGP::AssertOp>(chain.getDefiningOp()),
            << "Encountered a mo.assert with a chain that was not produced by "
               "another mo.assert!");

        SmallVector<Operation *> parentAsserts =
            getParentAsserts(chain.getDefiningOp(), execute.parentAssert);

        // collects the values (OpOperands really) that span from the tree
        // of expressions forming condition for the assertion we're looking
        // at and the assertion that produced the chain
        auto bOperands =
            getSequencedAssertBoundaryOperands(cond, parentAsserts);

        // This logic will materialize guards only in the case where there were
        // shared values between the expression tree rooted at the previous
        // assertion and the current
        //
        // This is desirable as the mgp.guard op doesn't have a Mojo
        // implementation (and consequently KGEN lowering), blocking the
        // possibility of sequence fusion
        if (!bOperands.empty()) {
          SmallVector<Value> v;
          for (OpOperand *opOperand : bOperands) {
            v.push_back(opOperand->get());
          }

          ValueRange guardedValRange{v};

          auto guardOp = builder.create<MGP::GuardOp>(
              loc, guardedValRange.getTypes(), chain, guardedValRange);

          for (auto &&[guarded, original] :
               llvm::zip(guardOp.getResults(), bOperands)) {
            original->assign(guarded);
          }
        }

        return {cond, chain.getDefiningOp()};
      } else {
        return {operands[0], nullptr};
      }
    }();

    // We must be dealing with an assert.
    newOp = builder.create<MGP::AssertOp>(loc, ValueRange{cond},
                                          opAttr.getAttrs().getValue());

    if (maybeParentAssert)
      execute.parentAssert[newOp] = maybeParentAssert;

    setCurrentExecuteChain(cast<MGP::AssertOp>(newOp).getOutChain());

  } else if (auto shapeAttr = dyn_cast<MOSH::ShapeAttr>(attr)) {
    // Lower dimensions recursively.
    SmallVector<Value> dimVals;
    for (TypedAttr dim : shapeAttr.getValues()) {
      auto dimValOr = state.lowerExpression(
          builder, loc, parentRegion, dim, [this](auto &&...args) {
            return this->lowerExpressionImpl(args...);
          });
      if (dimValOr.isError())
        return dimValOr.takeError();
      dimVals.emplace_back(std::move(*dimValOr));
    }

    // There is no runtime shape type in MGP, so we map shape attributes to
    // a fully dynamic spec.
    SmallVector<std::optional<uint64_t>> dummyShape(dimVals.size());
    auto specTy =
        MGP::TensorSpecType::get(loc.getContext(), dummyShape, DType::si64);
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

SmallVector<Value> MOOpLoweringState::getLoweredOperands() {
  SmallVector<Value> loweredOperands;
  for (Value operand : opBeingLowered->getOperands()) {
    Value loweredInput = loweringState.lookupExecuteLoweredValue(operand);

    if (isa<MO::TensorType>(operand.getType()) ||
        isa<MO::BufferType>(operand.getType())) {
      BufferizedTensor bufferizedInput = getBufferizedInput(operand);
      loweredOperands.push_back(bufferizedInput.buffer);
    } else {
      loweredOperands.push_back(loweredInput);
    }
  }
  return loweredOperands;
}

FailureOr<SmallVector<MOOutputStorage>>
MOOpLoweringState::createOutputStorage(Value inChain) {
  FailureOr<SmallVector<TypedValue<MGP::TensorSpecType>>> outSpecsOr =
      createOutputTensorSpecs();
  if (failed(outSpecsOr))
    return failure();
  return createOutputStorage(inChain, outSpecsOr.value());
}

FailureOr<SmallVector<MOOutputStorage>> MOOpLoweringState::createOutputStorage(
    Value inChain, ArrayRef<TypedValue<MGP::TensorSpecType>> outSpecs) {
  SmallVector<BufferizedTensor> bufferizedTensors;
  bufferizedTensors.reserve(outSpecs.size());

  // We only create the mgp storage for the pre-allocated tensors. For those
  // have in-place allocation, we skip them.
  DenseSet<int> inplaceAllocIndicesSet;
  if (auto kernel = dyn_cast<MOGG::KernelOp>(opBeingLowered))
    inplaceAllocIndicesSet = kernel.getInplaceAllocIndicesSet();

  SmallVector<Value> results;
  for (auto [idx, res] : llvm::enumerate(opBeingLowered->getResults())) {
    if (!inplaceAllocIndicesSet.contains(idx))
      results.push_back(opBeingLowered->getResult(idx));
  }

  for (auto [spec, result] : llvm::zip(outSpecs, results)) {
    auto bufOr = createBuffer(inChain, spec, result.getType());
    if (failed(bufOr))
      return failure();
    bufferizedTensors.emplace_back(*bufOr);
  }
  return bufferizedTensors;
}

Value MOOpLoweringState::kernelExecute(
    StringRef kernelName, Value inChain, ArrayRef<Value> args,
    ArrayRef<InOutSignatureAttr::InOutSemantics> inOutSemantics,
    ArrayRef<Value> inDimensions, uint64_t numOutDimensions,
    SmallVectorImpl<Value> &outDimensions, StringRef summary, bool isAsync) {
  MGP::BufferBuilder *bufferBuilder =
      loweringState.getBufferBuilder(MO::getDeviceOrDefaultAny(opBeingLowered));
  return bufferBuilder->kernelExecute(
      builder, loc, context, kernelName, inChain, args, inOutSemantics,
      inDimensions, numOutDimensions, outDimensions, summary, isAsync);
}

void MOOpLoweringState::createOutputs(
    Value inChain, SmallVector<MOOutputStorage> &loweredOutputsStorage) {
  if (auto kernel = dyn_cast<MOGG::KernelOp>(opBeingLowered)) {
    DenseSet<int> inplaceAllocIndicesSet = kernel.getInplaceAllocIndicesSet();
    int index = 0;
    for (const auto &[idx, res] : llvm::enumerate(kernel.getResults())) {
      if (!inplaceAllocIndicesSet.contains(idx)) {
        auto outputStorage = loweredOutputsStorage[index];
        createOutputTensor(res, inChain, outputStorage.buffer,
                           outputStorage.tensorSpec);
        index++;
      }
    }
  } else {
    for (const auto &[idx, outputStorage] :
         llvm::enumerate(loweredOutputsStorage)) {
      createOutputTensor(opBeingLowered->getResult(idx), inChain,
                         outputStorage.buffer, outputStorage.tensorSpec);
    }
  }
}

BufferizedTensor MOOpLoweringState::getBufferizedInput(Value moTensor) {
  Value loweredTensor = loweringState.lookupExecuteLoweredValue(moTensor);
  ASSERT_STREAM(loweredTensor, << "referenced SSA value has not been lowered");

  MO::TensorType ty = MO::castFromTensorLike(moTensor.getType());
  BufferBuilder *bufferBuilder = getStorageBufferBuilder(ty);
  auto buffer =
      bufferBuilder->tensorExtractBuffer(builder, loc, context, loweredTensor);
  auto tensorSpec = bufferBuilder->tensorExtractTensorSpec(
      builder, loc, context, loweredTensor);
  return BufferizedTensor(buffer, tensorSpec);
}

FailureOr<TypedValue<MGP::TensorSpecType>>
MOOpLoweringState::createTensorSpec(MO::TensorType moTensorType) {
  if (!moTensorType.hasRank()) {
    opBeingLowered->emitOpError(
        "dynamically ranked MO tensors are not yet supported");
    return failure();
  }
  MGP::TensorSpecType tensorSpecType =
      MGP::createCpuTensorSpecType(moTensorType);
  if (tensorSpecType.getSizeInBytes()) {
    // Statically shaped.
    return builder
        .create<MGP::CpuTensorSpecCreateOp>(loc,
                                            /*spec=*/tensorSpecType)
        .getResult();
  }

  // Dynamically shaped.
  auto runtimeDimsOr = loweringState.lowerShapeAttrDynDims(
      opBeingLowered->getParentRegion(), moTensorType.getTypedShapeAttr(), loc);
  if (runtimeDimsOr) {
    opBeingLowered->emitOpError(runtimeDimsOr.getError());
    return failure();
  }
  return builder
      .create<MGP::CpuTensorSpecCreateOp>(loc,
                                          /*spec=*/tensorSpecType,
                                          /*runtimeDims=*/*runtimeDimsOr)
      .getResult();
}

FailureOr<SmallVector<TypedValue<MGP::TensorSpecType>>>
MOOpLoweringState::createOutputTensorSpecs() {
  SmallVector<TypedValue<MGP::TensorSpecType>> tensorSpecs;
  tensorSpecs.reserve(opBeingLowered->getNumResults());

  DenseSet<int> inplaceAllocIndicesSet;
  if (auto kernel = dyn_cast<MOGG::KernelOp>(opBeingLowered))
    inplaceAllocIndicesSet = kernel.getInplaceAllocIndicesSet();

  for (auto [idx, type] :
       llvm::enumerate(opBeingLowered->getResults().getType())) {
    if (inplaceAllocIndicesSet.contains(idx))
      continue;

    auto moTensorType = dyn_cast<MO::TensorType>(type);
    ASSERT_STREAM(moTensorType, << "result of op is not a !mo.tensor");
    FailureOr<TypedValue<MGP::TensorSpecType>> tensorSpecOr =
        createTensorSpec(moTensorType);
    if (failed(tensorSpecOr))
      return failure();
    tensorSpecs.emplace_back(*tensorSpecOr);
  }
  return tensorSpecs;
}

FailureOr<BufferizedTensor> MOOpLoweringState::createBuffer(
    Value inChain, TypedValue<TensorSpecType> tensorSpec, Type moTensorType) {
  BufferBuilder *bufferBuilder = getStorageBufferBuilder(moTensorType);
  std::optional<uint64_t> optSize = tensorSpec.getType().getSizeInBytes();
  if (optSize) {
    // Statically sized. We don't actually need the tensor spec value to
    // allocate, and it may end up being eliminated by DCE.
    return BufferizedTensor(
        bufferBuilder->bufferAllocStatic(builder, loc, context, inChain,
                                         *optSize, minimumCpuAlignment),
        tensorSpec);
  } else {
    Value numBytes = builder.create<MGP::CpuTensorSpecSizeOp>(loc, tensorSpec);
    return BufferizedTensor(
        bufferBuilder->bufferAllocDynamic(builder, loc, context, inChain,
                                          numBytes, minimumCpuAlignment),
        tensorSpec);
  }
}

void MOOpLoweringState::createOutputTensor(
    Value moTensor, Value inChain, Value buffer,
    TypedValue<MGP::TensorSpecType> tensorSpec) {
  Value loweredTensor =
      getStorageBufferBuilder(moTensor.getType())
          ->tensorCreate(builder, loc, context, inChain, buffer, tensorSpec);
  loweringState.emplaceExecuteLoweredValue(moTensor, loweredTensor);
}

void MOOpLoweringState::createOutputTensors(
    Value inChain, SmallVector<Value> &buffers,
    SmallVector<TypedValue<MGP::TensorSpecType>> &tensorSpecs) {
  for (const auto &[idx, buffer, tensorSpec] :
       llvm::enumerate(buffers, tensorSpecs))
    createOutputTensor(opBeingLowered->getResult(idx), inChain, buffer,
                       tensorSpec);
}

BufferBuilder *MOOpLoweringState::getStorageBufferBuilder(Type moTensorType) {
  auto type = dyn_cast<MO::TensorType>(moTensorType);
  ASSERT_STREAM(type, << "expecting !mo.tensor type");
  return loweringState.getBufferBuilder(
      MO::getDeviceOrDefault(loweringState.getGraph(), type));
}

TypeBuilder *MOOpLoweringState::getStorageTypeBuilder(Type moTensorType) {
  auto type = dyn_cast<MO::TensorType>(moTensorType);
  ASSERT_STREAM(type, << "expecting !mo.tensor type");
  return loweringState.getTypeBuilder(
      MO::getDeviceOrDefault(loweringState.getGraph(), type));
}
