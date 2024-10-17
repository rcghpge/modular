//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Interfaces/LoweringState.h"
#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/AssertStream.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>

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
           << llvm::Twine(deviceSpecsOr.getError()) << "\n";
  return *deviceSpecsOr;
}

TypedValue<MGP::TensorSpecType>
MOGraphLoweringState::getShape(Location loc, Value loweredTensor,
                               DeviceRefAttr device) {
  OpBuilder &builder = getExecuteOpBuilder();
  return getBufferBuilder(device)->tensorExtractTensorSpec(builder, loc,
                                                           loweredTensor);
}

MGP::CpuTensorSpecDimsDynamicOp
MOGraphLoweringState::getDynamicDims(Location loc, Value loweredTensor,
                                     DeviceRefAttr device) {
  OpBuilder &builder = getExecuteOpBuilder();
  // NOTE: We could use getShape here, however usually we'll also require the
  // loweredTensor's buffer in which case we'll be emitting a full
  // tensor.extract op anyway.
  TypedValue<MGP::TensorSpecType> tensorSpec =
      getBufferBuilder(device)->tensorExtractTensorSpec(builder, loc,
                                                        loweredTensor);
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
    return mlir::index::AddOp::getOperationName();
  case KGEN::POC::Div:
    return mlir::index::DivSOp::getOperationName();
  case KGEN::POC::EQ:
  case KGEN::POC::LT:
  case KGEN::POC::LE:
    return mlir::index::CmpOp::getOperationName();
  case KGEN::POC::Max:
    return mlir::index::MaxSOp::getOperationName();
  case KGEN::POC::Min:
    return mlir::index::MinSOp::getOperationName();
  case KGEN::POC::Mod:
    return mlir::index::RemSOp::getOperationName();
  case KGEN::POC::Mul:
  case KGEN::POC::MulNuw:
    return mlir::index::MulOp::getOperationName();
  case KGEN::POC::And:
    return KGEN::POP::AndOp::getOperationName();
  case KGEN::POC::Or:
    return KGEN::POP::OrOp::getOperationName();
  case KGEN::POC::Xor:
    return KGEN::POP::XOrOp::getOperationName();
  case KGEN::POC::Cond:
    return KGEN::POP::SelectOp::getOperationName();
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
  if (opNameAttr == KGEN::POP::SelectOp::getOperationName())
    return builder.create(loc, opNameAttr, operands, {resTy})->getResult(0);

  if (opNameAttr == mlir::index::CmpOp::getOperationName()) {
    auto getIdxCmpPredicate = [](KGEN::POC poc) {
      switch (poc) {
      case KGEN::POC::EQ:
        return mlir::index::IndexCmpPredicate::EQ;
      case KGEN::POC::LT:
        return mlir::index::IndexCmpPredicate::SLT;
      case KGEN::POC::LE:
        return mlir::index::IndexCmpPredicate::SLE;
      default:
        llvm_unreachable("unrecognized predicate");
      }
    };

    assert(operands.size() == 2 && "Cmp Operation must be a binary operation");
    return builder.create<mlir::index::CmpOp>(
        loc, getIdxCmpPredicate(opAttr.getOpcode()),
        /*lhs=*/operands.front(),
        /*rhs=*/operands.back());
  }

  // For potentially variadic binary ops
  //   add(a, b, c)
  //   ===>
  //   index.add(index.add(a, b), c)
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
    return builder.create<mlir::index::BoolConstantOp>(loc,
                                                       boolAttr.getValue());
  }
  return builder.create<mlir::index::ConstantOp>(loc, intAttr.getInt());
}

/// sequenced assert lowering helper functions
namespace {

bool isIndexOp(Operation *op) {
  // We uses POP::And/Or/XOrOp instead of index::And/Or/XOr so that we can
  // handle more than IndexType.
  return isa<mlir::index::IndexDialect>(op->getDialect()) ||
         isa<KGEN::POP::SelectOp, KGEN::POP::AndOp, KGEN::POP::OrOp,
             KGEN::POP::XOrOp>(op);
}

DenseSet<Operation *> getAncestors(Operation *root) {

  DenseSet<Operation *> ancestors;
  SmallVector<Operation *> worklist;

  worklist.push_back(root);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    if (!op || !isIndexOp(op))
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

    if (!op || !isIndexOp(op) || prevAncestors.contains(op))
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
    MGP::AssertOp assertOp = builder.create<MGP::AssertOp>(
        loc, ValueRange{cond}, opAttr.getAttrs().getValue());

    newOp = assertOp;
    if (maybeParentAssert)
      execute.parentAssert[newOp] = maybeParentAssert;
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

Value MOOpLoweringState::getGuardChain(Value chain) {
  SmallVector<Value> inputChains;
  for (Value operand : opBeingLowered->getOperands()) {
    if (Value c = loweringState.getChainGuardingValueOrNull(operand))
      inputChains.push_back(c);
  }
  if (!inputChains.empty()) {
    inputChains.push_back(chain);
    chain = create<GRT::ChainCreateOp>(inputChains).getResult();
  }
  return chain;
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
    if (!inplaceAllocIndicesSet.contains(idx) &&
        !isa<MO::ChainType>(res.getType()))
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

void MOOpLoweringState::createOutputs(
    Value inChain, SmallVector<MOOutputStorage> &loweredOutputsStorage) {
  if (auto kernel = dyn_cast<MOGG::KernelOp>(opBeingLowered)) {
    DenseSet<int> inplaceAllocIndicesSet = kernel.getInplaceAllocIndicesSet();
    int index = 0;
    for (const auto &[idx, res] : llvm::enumerate(kernel.getResults())) {
      // ignore chain types as there are currently special carveouts in all
      // instances of KernelOp lowerings
      if (!inplaceAllocIndicesSet.contains(idx) &&
          !isa<MO::ChainType>(res.getType())) {
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
  auto buffer = bufferBuilder->tensorExtractBuffer(builder, loc, loweredTensor);
  auto tensorSpec =
      bufferBuilder->tensorExtractTensorSpec(builder, loc, loweredTensor);

  // we are extracting a buffer from a tensor, so the buffer needs to be guarded
  // by the same chain as the tensor
  loweringState.guardValueWithChain(
      buffer, loweringState.getChainGuardingValueOrNull(loweredTensor));

  return BufferizedTensor(buffer, tensorSpec);
}

FailureOr<TypedValue<MGP::TensorSpecType>>
MOOpLoweringState::createTensorSpec(MO::TensorType moTensorType) {
  if (!moTensorType.hasRank())
    return opBeingLowered->emitOpError(
        "dynamically ranked MO tensors are not yet supported");
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
  if (runtimeDimsOr)
    return opBeingLowered->emitOpError(runtimeDimsOr.getError());
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

    if (isa<MO::ChainType>(type))
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
  Value numBytes = builder.create<MGP::CpuTensorSpecSizeOp>(loc, tensorSpec);

  // If static size is known, it's used for the output type
  std::optional<uint64_t> optSize = tensorSpec.getType().getSizeInBytes();

  auto bufferDevice = bufferBuilder->getDevice();
  auto preferredAlignment =
      M::AsyncRT::getPreferredAlignment(bufferDevice.getLabel().str());
  if (bufferDevice.getLabel() == M::kCPULabel)
    preferredAlignment = std::gcd(preferredAlignment, minimumCpuAlignment);
  auto deviceContext = loweringState.getExecuteDevice(bufferDevice);
  return BufferizedTensor(
      bufferBuilder->bufferAlloc(builder, loc, context, inChain, numBytes,
                                 deviceContext, preferredAlignment, optSize),
      tensorSpec);
}

void MOOpLoweringState::createOutputTensor(
    Value moTensor, Value inChain, Value buffer,
    TypedValue<MGP::TensorSpecType> tensorSpec) {
  Value loweredTensor =
      getStorageBufferBuilder(moTensor.getType())
          ->tensorCreate(builder, loc, context, inChain, buffer, tensorSpec);
  loweringState.emplaceExecuteLoweredBufferValue(moTensor, loweredTensor,
                                                 inChain);
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
