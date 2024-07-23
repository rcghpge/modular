//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Converts MO ops to MGP and MEF ops.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/Conversion/DeviceBufferizationRegistry.h"
#include "GenericML/GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/MGPTransforms/FixupIsolatedFromAbove.h"
#include "GenericML/Support/HostDeviceSpecs.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "GraphRT/MEFDialect/MEFOps.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/MArchTarget/Host.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/ML/CommonAttributes.h"
#include "Support/STLExtras.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include <cassert>

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_MOTOPRIMITIVES
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

//===----------------------------------------------------------------------===//
// MOGraphLowering implementation
//===----------------------------------------------------------------------===//

namespace {

/// This class orchestrates lowering of one mo.graph operation to a mgp.model,
/// consisting of nad 'init' phase, and 'execute' phase, and a model descriptor.
///
/// The construction state for the init/execute invokables are kept in 'init'
/// and 'execute' members which are mutated as each operation in the input graph
/// is lowered.
class MOGraphLowering {
public:
  MOGraphLowering(MO::GraphOp graph, llvm::StringRef compiledFrameworkLabel,
                  size_t minimumCpuAlignment)
      : loweringState(graph, compiledFrameworkLabel, minimumCpuAlignment) {}

  /// Lower the mo.graph to mgp.model.
  LogicalResult run();

  // Establish type builders for all the (virtual) devices needed by the model.
  LogicalResult setupTypeBuildersForAllDevices();

  // Establish buffer builders for all the (virtual) devices needed by the
  // model. Each device will have its 'runtime' created in the 'init' block
  // and held in the !mgp.context.
  CompileDiagnosticOrSuccess setupBufferBuildersForAllDevices();

  LogicalResult createModelOp();

  /// Lowers a MO::ListType to a MGP::ListType.
  MGP::ListType lowerListType(MO::ListType listTy);

  /// Lowers the mo.output op to the mef.output for the final results.
  LogicalResult lowerOutput(MO::OutputOp output);

  /// Lowers the mo.yield to the mef.output for the if block results.
  LogicalResult lowerYield(MO::YieldOp output);

  /// Returns an attribute to represent elementAttr for the remainder of
  /// compilation:
  ///  - We commit to a specific byte layout, ie from here on the attribute
  ///    is considered an ArrayRef<uint8_t> with no further interpretation.
  ///  - Large attributes are outlined as dense_resources.
  ///  - The required runtime alignment is captured as an !M.aligned_bytes type.
  ///  - The runtime alignment will be at least 'minimumCpuAlignment' from
  ///    the pass options.
  ///  - If elementAttr already has an !M.aligned_bytes type annotation, that
  ///    may increase the alignment. Otherwise, if the attribute itself has some
  ///    indication of alignment (eg a dense_resource's blob's alignment) that
  ///    may increase the alignment.
  ///  - However the result's underlying data may not respect the alignment.
  ///    Instead we assume alignment is enforced on serialization to MEF.
  ///    This helps us avoid gratuitous memory copies.
  ///  - No data is copied.
  ErrorOr<ElementsAttr> lowerElementsAttr(ElementsAttr elementsAttr);

  /// Lower a parameter declaration op.
  LogicalResult lowerParamDeclare(Region *parentRegion,
                                  KGEN::ParamDeclareOp declOp);

  /// Lower a MOSH::ParamToValueOp op.
  LogicalResult lowerParamToValue(Region *parentRegion,
                                  MOSH::ParamToValueOp paramToValueOp);

  /// Lower an index -> scalar-like tensor cast op.
  LogicalResult lowerIndexToTensorOp(MO::IndexToTensorOp toTensorOp);

  /// Lower a shape -> shape-like tensor cast op.
  LogicalResult lowerShapeToTensorOp(MO::Shape::ToTensorOp toTensorOp);

  /// Lower a tensor rebind.
  LogicalResult lowerRebind(MO::RebindOp rebindOp);

  /// Lowers mo.constant op.
  LogicalResult lowerConstant(MO::ConstantOp constOp);

  /// Lowers mo.constant.scalar op.
  LogicalResult lowerConstantScalar(MO::ConstantScalarOp constOp);

  /// Lowers mo.reshape op.
  LogicalResult lowerReshape(MO::ReshapeOp reshapeOp);

  /// Lowers mo.static.reshape op.
  LogicalResult lowerStaticReshape(MO::Static::ReshapeOp reshapeOp);

  /// Lowers mo.debug.tensor.print op.
  LogicalResult lowerDebugTensorPrint(MO::DebugTensorPrintOp printOp);

  /// Lowers mo.transfer op.
  LogicalResult lowerTransfer(MO::TransferOp transferOp);

  /// Lowers mo.if op.
  LogicalResult lowerIf(MO::IfOp ifOp);

  /// Lowers mo.while op.
  LogicalResult lowerWhile(MO::WhileOp whileOp);

  /// Lowers mo.list.append op.
  LogicalResult lowerListAppend(MO::ListAppendOp listAppendOp);

  /// Lowers mo.list.concat op.
  LogicalResult lowerListConcat(MO::ListConcatOp listConcatOp);

  /// Lowers mo.list.create op.
  LogicalResult lowerListCreate(MO::ListCreateOp listCreateOp);

  /// Lowers mo.list.get op.
  LogicalResult lowerListGet(MO::ListGetOp listGetOp);

  /// Lowers mo.list.insert op.
  LogicalResult lowerListInsert(MO::ListInsertOp listInsertOp);

  // Lowers mo.list.slice op.
  LogicalResult lowerListSlice(MO::ListSliceOp listSliceOp);

  /// Lowers mo.guard op
  LogicalResult lowerGuard(MO::GuardOp guardOp);

  /// Lowers mo.chain.create op
  LogicalResult lowerChainCreate(MO::ChainCreateOp);

private:
  /// Lowers a Block
  LogicalResult lowerBlock(mlir::Block *block);

  MGP::MOGraphLoweringState loweringState;
};

} // namespace

LogicalResult MOGraphLowering::setupTypeBuildersForAllDevices() {
  CompileDiagnosticOr<DeviceSpecCollectionAttr> deviceSpecsOr =
      loweringState.getDeviceSpecs();
  if (deviceSpecsOr.isError())
    return deviceSpecsOr.takeError();

  for (DeviceSpecAttr deviceSpec : deviceSpecsOr.takeValue().getDevices()) {
    if (failed(MO::makeTypeBuilder(deviceSpec, loweringState)))
      return failure();
  }
  return success();
}

CompileDiagnosticOrSuccess MOGraphLowering::setupBufferBuildersForAllDevices() {
  CompileDiagnosticOr<DeviceSpecCollectionAttr> deviceSpecsOr =
      loweringState.getDeviceSpecs();
  if (deviceSpecsOr.isError())
    return deviceSpecsOr.takeError();
  auto deviceSpecs = deviceSpecsOr.takeValue();

  // Make sure the 'host' device buffer builder is established first since
  // other builders need to refer to it.
  for (DeviceSpecAttr deviceSpec : deviceSpecs.getDevices()) {
    if (deviceSpec.getRef() != deviceSpecs.getHost())
      continue;
    if (auto err = MO::makeBufferBuilder(deviceSpecs.getHost(), deviceSpec,
                                         loweringState);
        err.isError())
      return err;
  }

  // Now establish buffer builders for all other devices.
  for (DeviceSpecAttr deviceSpec : deviceSpecs.getDevices()) {
    if (deviceSpec.getRef() == deviceSpecs.getHost())
      continue;
    if (auto err = MO::makeBufferBuilder(deviceSpecs.getHost(), deviceSpec,
                                         loweringState);
        err.isError())
      return err;
  }
  return success();
}

// Create the mgp.model and the regions for the init/execute phases.
LogicalResult MOGraphLowering::createModelOp() {
  MO::GraphOp &graph = loweringState.getGraph();
  OpBuilder b(graph);

  // Establish type builders for all the devices.
  // CAUTION: Must be done before any types are lowered, eg the next paragraph.
  if (failed(setupTypeBuildersForAllDevices()))
    return failure();

  // Convert the tensor types for the model signature over.
  SmallVector<Type> inputTypes, outputTypes;
  if (failed(loweringState.getTypeConverter().convertTypes(
          graph.getArgumentTypes(), inputTypes))) {
    return graph.emitOpError("failed to convert mo graph input types")
           << graph.getArgumentTypes() << " is not supported";
  } else if (failed(loweringState.getTypeConverter().convertTypes(
                 graph.getResultTypes(), outputTypes))) {
    return graph.emitOpError("failed to convert mo graph output types -- ")
           << graph.getResultTypes() << " is not supported";
  }

  // Preserve the MO::GraphOp argument locations in the MGP::ModelOp 'execute'
  // block argument locations.
  SmallVector<Location> inputLocs;
  for (BlockArgument arg : graph.getBody().front().getArguments())
    inputLocs.push_back(arg.getLoc());

  // Collect additional attributes to include with the MGP::ModelOp to guide
  // the MEF executor.
  SmallVector<NamedAttribute> extraAttrs;

  ArrayAttr argNamesAttr =
      graph->getAttrOfType<ArrayAttr>(kMgpModelArgumentNames);
  ArrayAttr resultNamesAttr =
      graph->getAttrOfType<ArrayAttr>(kMgpModelResultNames);
  ASSERT_STREAM(
      (argNamesAttr == nullptr) == (resultNamesAttr == nullptr),
      << "expected either both or neither of the `argument_names` and "
         "`result_names` attributes.");
  if (argNamesAttr) {
    // Set the `argument_names` and `result_names` attributes.
    extraAttrs.emplace_back(b.getStringAttr(kMgpModelArgumentNames),
                            argNamesAttr);
    extraAttrs.emplace_back(b.getStringAttr(kMgpModelResultNames),
                            resultNamesAttr);
  }

  // Create the mgp.model and the regions for the init/execute phases.
  b.setInsertionPoint(graph);
  // Get all the devices
  ErrorOrSuccess err(success());
  auto devices = llvm::map_to_vector(
      loweringState.getDeviceSpecs().takeValue().getDevices(),
      [&err](DeviceSpecAttr attr) {
        auto refOr = toRuntimeDeviceRef(attr.getRef());
        if (refOr.isError()) {
          err = refOr.takeError();
          return DeviceRef();
        } else {
          return refOr.takeValue();
        }
      });
  if (err.isError())
    return err;
  auto modelOp = b.create<MGP::ModelOp>(
      graph.getLoc(),
      /*name=*/graph.getNameAttr(),
      /*frameworkName=*/loweringState.getCompiledFrameworkLabel(),
      /*devices=*/devices,
      /*inputTypes=*/inputTypes,
      /*outputTypes=*/outputTypes,
      /*inputArgLocs=*/inputLocs,
      /*extraAttrs=*/extraAttrs);

  loweringState.setModelOp(modelOp);

  // Initialize the init state.
  loweringState.getInitOpBuilder().setInsertionPointToEnd(
      loweringState.getModelOp().getInitBlock());
  // Get the chain and context values for the body.
  // This field tracks the current output chain as we lower into the init body.
  loweringState.setInitChain(
      loweringState.getInitOpBuilder().create<GRT::ChainCreateOp>(
          graph.getLoc()));

  // Create the !mgp.context. Its numSlots attribute will be incremented
  // as needed.
  loweringState.setInitMgpContext(
      loweringState.getInitOpBuilder().create<MGP::ContextCreateOp>(
          graph.getLoc(), /*numSlots=*/0));

  // Initialize the execute state.
  Block *executeBlock = loweringState.getModelOp().getExecuteBlock();
  loweringState.getExecuteOpBuilder().setInsertionPointToEnd(executeBlock);

  // Extract chain and MGP context from arguments
  loweringState.setRootExecuteChain(executeBlock->getArgument(0));
  loweringState.setCurrentExecuteChain(executeBlock->getArgument(0));
  loweringState.setExecuteMgpContext(executeBlock->getArgument(1));

  // Establish buffer builders for all the devices. This may require some
  // extra device-specific setup in the 'init' block.
  // CAUTION: Must be done after the 'init' and 'execute' mgp.contexts have
  // been established, but before we lower any device-specific ops.
  if (auto err = setupBufferBuildersForAllDevices(); err.isError())
    return err.takeError();

  return success();
}

MGP::ListType MOGraphLowering::lowerListType(MO::ListType listTy) {
  auto outputTensorTy = dyn_cast<MO::TensorType>(listTy.getElementType());
  // TODO(#29947): Support other element types
  ASSERT_STREAM(outputTensorTy, << "only support lowering list of tensors");

  MLIRContext *ctx = listTy.getContext();
  MGP::TensorSpecType outputTensorSpecTy =
      MGP::createCpuTensorSpecType(outputTensorTy);
  DeviceRefAttr tensorDeviceRef =
      MO::getDeviceOrDefault(loweringState.getGraph(), outputTensorTy);
  MGP::TypeBuilder *typeBuilder = loweringState.getTypeBuilder(tensorDeviceRef);
  auto mgpOutputTensorTy = typeBuilder->tensorType(ctx, outputTensorSpecTy);
  return MGP::ListType::get(ctx, mgpOutputTensorTy);
}

LogicalResult MOGraphLowering::lowerOutput(MO::OutputOp output) {
  // Init phase lowering.

  // Return the final chain.
  loweringState.getInitOpBuilder().create<MEF::OutputOp>(
      output.getLoc(), ArrayRef<Value>{loweringState.getInitChain(),
                                       loweringState.getInitMgpContext()});

  // Execute phase lowering.

  // Figure out what values to return.  We always return a chain.
  SmallVector<Value> operands;
  operands.push_back(loweringState.getCurrentExecuteChain());

  for (Value operand : output.getOperands())
    operands.push_back(loweringState.lookupExecuteLoweredValue(operand));

  // Finally, we can create the mef.output op.
  loweringState.getExecuteOpBuilder().create<MEF::OutputOp>(output.getLoc(),
                                                            operands);
  return success();
}

LogicalResult MOGraphLowering::lowerYield(MO::YieldOp yield) {
  // Execute phase lowering.

  // Figure out what values to return. We always implicitly return the current
  // 'execute' block chain.
  SmallVector<Value> operands;
  operands.push_back(loweringState.getCurrentExecuteChain());

  for (Value operand : yield.getOperands())
    operands.push_back(loweringState.lookupExecuteLoweredValue(operand));

  // Finally, we can create the mef.output op.
  loweringState.getExecuteOpBuilder().create<MEF::OutputOp>(yield.getLoc(),
                                                            operands);
  return success();
}

/// Minimum byte size for a dense_resource.
static constexpr size_t kLargeDataThreshold = 512;

ErrorOr<ElementsAttr>
MOGraphLowering::lowerElementsAttr(ElementsAttr elementsAttr) {
  MLIRContext *ctx = loweringState.getExecuteOpBuilder().getContext();

  // All attributes must have a minimum alignment of this value.
  uint64_t alignment =
      static_cast<uint64_t>(loweringState.getMinimumCpuAlignment());

  auto hasAlignedAttr = dyn_cast<HasAlignedBytesInterface>(elementsAttr);
  if (hasAlignedAttr == nullptr)
    return Error("unsupported constant attribute type");
  auto origType = hasAlignedAttr.getAlignedBytesType();

  // Increase alignment to respect what the attribute requests.
  alignment = std::max(alignment, origType.getAlign());

  // Incorporate the alignment and size into an AlignedBytesType.
  auto finalType = AlignedBytesType::get(ctx, origType.getSize(), alignment);

  if (auto arrayAttr = dyn_cast<ArrayElementsAttr>(elementsAttr)) {
    // In the following we'll reuse the underlying array elements data as is,
    // ignoring its current alignment. The desired MEF alignment is instead
    // conveyed via the aligned_bytes type.
    if (arrayAttr.getRawData().size() < kLargeDataThreshold) {
      // Small attributes can remain inline.
      return AlignedBytesAttr::get(ctx, arrayAttr.getRawData(), alignment);
    } else {
      // Push large attributes to dense_resources.
      std::string name =
          "const_" + std::to_string(loweringState.getExecuteNextConstId());
      ArrayRef<char> data(
          reinterpret_cast<const char *>(arrayAttr.getRawData().data()),
          arrayAttr.getRawData().size());
      return DenseResourceElementsAttr::get(
          finalType,
          DenseResourceElementsHandle::getManagerInterface(ctx).insert(
              name, mlir::UnmanagedAsmResourceBlob::allocateWithAlign(
                        data, alignment)));
    }
  }

  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(elementsAttr)) {
    // We'll not try to allocate a new blob. Instead we'll reuse the original
    // blob and rely on the type annotation to capture our desired alignment.
    return DenseResourceElementsAttr::get(finalType,
                                          resourceAttr.getRawHandle());
  }

  if (auto alignedAttr = dyn_cast<AlignedBytesAttr>(elementsAttr))
    return AlignedBytesAttr::get(ctx, alignedAttr.getData(), alignment);

  llvm_unreachable(
      "missing case for implementation of HasAlignedBytesInterface");
}

LogicalResult MOGraphLowering::lowerParamDeclare(Region *parentRegion,
                                                 KGEN::ParamDeclareOp declOp) {
  // We first lower the expression.
  auto valueOr = loweringState.lowerExpression(parentRegion, declOp->getLoc(),
                                               declOp.getValue());
  if (valueOr.isError())
    return declOp->emitOpError(valueOr.getError());

  Value value = *valueOr;
  // Then map the parameter declaration to it.
  if (auto err = loweringState.lowerParamDecl(parentRegion,
                                              declOp.getParamDecl(), value))
    return declOp->emitOpError(err.getError());

  return success();
}

LogicalResult
MOGraphLowering::lowerParamToValue(Region *parentRegion,
                                   MOSH::ParamToValueOp paramToValueOp) {
  auto valueOr = loweringState.lowerExpression(
      parentRegion, paramToValueOp.getLoc(), paramToValueOp.getValue());
  if (valueOr.isError())
    return paramToValueOp->emitOpError(valueOr.getError());

  loweringState.emplaceExecuteLoweredValue(paramToValueOp.getResult(),
                                           *valueOr);
  return success();
}

LogicalResult
MOGraphLowering::lowerIndexToTensorOp(MO::IndexToTensorOp toTensorOp) {
  ASSERT_STREAM(MO::isOnHost(toTensorOp),
                << "mo.index.to_tensor is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, toTensorOp);

  // Setup output buffer.
  FailureOr<SmallVector<MGP::MOOutputStorage>> loweredOutputStorageOr =
      opLoweringState.createOutputStorage(loweringState.getExecuteChain());
  if (failed(loweredOutputStorageOr))
    return failure();
  auto &outputStorage = (*loweredOutputStorageOr)[0];
  auto bufType =
      dyn_cast<MGP::DeviceBufferType>(outputStorage.buffer.getType());
  ASSERT_STREAM(bufType && bufType.getDevice().getLabel() == "cpu",
                << "expecting !mgp.buffer on host but it is on "
                << bufType.getDevice());

  // Fill the buffer with the index value.
  Value loweredIndexVal =
      loweringState.lookupExecuteLoweredValue(toTensorOp.getInput());
  SmallVector<Value> values;
  values.emplace_back(loweredIndexVal);
  auto outChain = opLoweringState.bufferSetWithIndex(
      outputStorage.buffer, values, toTensorOp.getResult().getType());

  // Setup output tensor.
  opLoweringState.createOutputs(outChain, *loweredOutputStorageOr);
  return success();
}

LogicalResult
MOGraphLowering::lowerShapeToTensorOp(MO::Shape::ToTensorOp toTensorOp) {
  ASSERT_STREAM(MO::isOnHost(toTensorOp),
                << "mo.shape.to_tensor is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, toTensorOp);

  // Setup output buffer.
  FailureOr<SmallVector<MGP::MOOutputStorage>> loweredOutputsStorageOr =
      opLoweringState.createOutputStorage(loweringState.getExecuteChain());
  if (failed(loweredOutputsStorageOr))
    return failure();
  auto &outputStorage = (*loweredOutputsStorageOr)[0];

  // Fill the buffer with the runtime dimensions.
  auto loweredShape =
      loweringState.lookupExecuteLoweredValue(toTensorOp.getInput());
  auto loweredTensorSpec = cast<TypedValue<MGP::TensorSpecType>>(loweredShape);
  // Note loweredTensorSpec is the encoding for the shape we are trying to
  // reify as a tensor, whereas outputStorage.tensorSpec just describes
  // the statically known shape of our output tensor.
  auto dynDimsOp = opLoweringState.create<MGP::CpuTensorSpecDimsDynamicOp>(
      /*spec=*/loweredTensorSpec,
      /*rawDims=*/loweredTensorSpec.getType().getRawDims());
  SmallVector<Value> values = dynDimsOp.getResults();
  auto outChain = opLoweringState.bufferSetWithIndex(
      outputStorage.buffer, values, toTensorOp.getResult().getType());

  // Setup output tensor.
  opLoweringState.createOutputs(outChain, *loweredOutputsStorageOr);
  return success();
}

/// Returns the lowered tensor that the rebind's result should map to.
static Value getRebindResTensor(MO::RebindOp rebindOp,
                                MGP::MOGraphLoweringState &loweringState) {
  MGP::MOOpLoweringState opLoweringState(loweringState, rebindOp);

  MO::TensorValue inputTensor = rebindOp.getInput();
  MO::TensorType inputTy = inputTensor.getType();
  MO::TensorType resTy = rebindOp.getResult().getType();

  // If the input and output specs are the same, we can just map to the already
  // lowered tensor.
  auto outSpecTy = MGP::createCpuTensorSpecType(resTy);
  auto inSpecTy = MGP::createCpuTensorSpecType(inputTy);
  Value loweredInput = loweringState.lookupExecuteLoweredValue(inputTensor);
  if (outSpecTy == inSpecTy)
    return loweredInput;

  // We will collect the runtime dimensions for the result spec.
  SmallVector<Value> runtimeDims;
  MGP::CpuTensorSpecDimsDynamicOp dynDimsOp;
  OpBuilder &exeBuilder = loweringState.getExecuteOpBuilder();
  Location loc = opLoweringState.loc;
  for (auto [idx, inDim, resDim] :
       llvm::enumerate(inputTy.getDims(), resTy.getDims())) {

    // If the result dimension is static, nothing to do.
    if (isa<IntegerAttr>(resDim))
      continue;

    // If the input dimension is static, and the result is dynamic, we create a
    // constant runtime dimension.
    if (auto intAttr = dyn_cast<IntegerAttr>(inDim)) {
      IntegerType si64Ty = exeBuilder.getIntegerType(64, /*isSigned=*/true);
      auto newIntAttr = exeBuilder.getIntegerAttr(si64Ty, intAttr.getInt());
      auto constOp = opLoweringState.create<MIP::ConstantIndexOp>(
          intAttr.getType(), newIntAttr);
      runtimeDims.emplace_back(constOp.getResult());
      continue;
    }

    // At this point we are dealing with dynamic input and result dimensions.
    // We extract the dynamic dimensions only once.
    if (!dynDimsOp)
      dynDimsOp = loweringState.getDynamicDims(
          loc, loweredInput,
          getDeviceOrDefault(loweringState.getGraph(), inputTy));

    // Forward the appropriate dynamic dimension from the input.
    runtimeDims.emplace_back(
        dynDimsOp->getResult(inSpecTy.getDynamicDimIndex(idx)));
  }

  // Create the output tensor.
  Value buffer = opLoweringState.tensorExtract(loweredInput, inputTy).first;
  TypedValue<MGP::TensorSpecType> tensorSpec =
      opLoweringState.create<MGP::CpuTensorSpecCreateOp>(outSpecTy,
                                                         runtimeDims);
  return opLoweringState.tensorCreate(loweringState.getExecuteChain(), buffer,
                                      tensorSpec, inputTy);
}

LogicalResult MOGraphLowering::lowerRebind(MO::RebindOp rebindOp) {
  MO::TensorValue inputTensor = rebindOp.getInput();
  MO::TensorValue resTensor = rebindOp.getResult();

  MO::TensorType inputTy = inputTensor.getType();
  MO::TensorType resTy = resTensor.getType();
  if (inputTy.hasRank() != resTy.hasRank())
    return rebindOp.emitOpError(
        "rebind lowering doesn't support unknown shapes");

  Value loweredOutput = getRebindResTensor(rebindOp, loweringState);
  loweringState.emplaceExecuteLoweredValue(resTensor, loweredOutput);

  // Establish the lowered form of the rebind's result parameters.
  // TODO: Note this may end up extracting the dynamic dimensions from the
  // result tensor even though the same dimension values were just passed
  // as operands to a tensor_spec.create op.
  ArrayRef<KGEN::ParamDeclAttr> decls = rebindOp.getOutputParamDecls();
  if (!decls.empty()) {
    MGP::CpuTensorSpecDimsDynamicOp dynDims = loweringState.getDynamicDims(
        rebindOp.getLoc(), loweredOutput,
        getDeviceOrDefault(loweringState.getGraph(), resTy));

    MOSH::ShapeAttr resShape = resTy.getTypedShapeAttr();
    for (KGEN::ParamDeclAttr decl : decls) {
      int64_t idx = findDeclIdx(resShape, decl);
      if (idx == -1) {
        return rebindOp.emitOpError(
            "rebind declares parameter not appearing in its result");
      }

      MGP::TensorSpecType specTy = dynDims.getSpec().getType();
      Value rtVal = dynDims->getResult(specTy.getDynamicDimIndex(idx));
      if (auto err = loweringState.lowerParamDecl(rebindOp->getParentRegion(),
                                                  decl, rtVal))
        return rebindOp->emitOpError(err.getError());
    }
  }

  return success();
}

LogicalResult MOGraphLowering::lowerConstant(MO::ConstantOp constOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, constOp);

  ErrorOr<ElementsAttr> elementsOr = lowerElementsAttr(constOp.getValue());
  if (elementsOr)
    return constOp->emitOpError(elementsOr.getError());

  Value buffer = opLoweringState.bufferConstant(loweringState.getExecuteChain(),
                                                *elementsOr,
                                                constOp.getResult().getType());
  FailureOr<TypedValue<MGP::TensorSpecType>> tensorSpecOr =
      opLoweringState.createTensorSpec(constOp.getType());
  if (failed(tensorSpecOr))
    return failure();
  opLoweringState.createOutputTensor(constOp.getResult(),
                                     loweringState.getExecuteChain(), buffer,
                                     *tensorSpecOr);
  return success();
}

LogicalResult
MOGraphLowering::lowerConstantScalar(MO::ConstantScalarOp constOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, constOp);

  // currently only si32 and i1 constants are supported
  if (auto boolAttr = dyn_cast<mlir::BoolAttr>(constOp.getValue())) {
    IntegerAttr intAttr = boolAttr;
    auto grtConstOp =
        opLoweringState.create<GRT::ConstantI1Op>(intAttr.getType(), intAttr);
    loweringState.emplaceExecuteLoweredValue(constOp, grtConstOp);

  } else if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
    if (intAttr.getType().isSignedInteger(32)) {
      auto grtConstOp = opLoweringState.create<GRT::ConstantSI32Op>(
          intAttr.getType(), intAttr);
      loweringState.emplaceExecuteLoweredValue(constOp, grtConstOp);
    } else {
      return constOp->emitOpError(
          "Only si32 and i1 constant scalar are currently lowerable");
    }
  } else {
    return constOp->emitOpError(
        "Only si32 and i1 constant scalar are currently lowerable");
  }

  return success();
}

LogicalResult
MOGraphLowering::lowerStaticReshape(MO::Static::ReshapeOp reshapeOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, reshapeOp);

  // Get the bufferized input.
  MGP::BufferizedTensor bufferizedInput =
      opLoweringState.getBufferizedInput(reshapeOp.getInput());
  Value buffer = bufferizedInput.buffer;

  // Create a tensor result with spec matching the result type.
  FailureOr<TypedValue<MGP::TensorSpecType>> tensorSpecOr =
      opLoweringState.createTensorSpec(reshapeOp.getType());
  if (failed(tensorSpecOr))
    return failure();
  opLoweringState.createOutputTensor(reshapeOp.getResult(),
                                     loweringState.getExecuteChain(), buffer,
                                     *tensorSpecOr);
  return success();
}

LogicalResult MOGraphLowering::lowerReshape(MO::ReshapeOp reshapeOp) {
  // Note the lowering is currently identical to Static::ReshapeOp.
  // TODO(#12718): Validate user-annotated shape.
  MGP::MOOpLoweringState opLoweringState(loweringState, reshapeOp);

  // Get the bufferized input.
  MGP::BufferizedTensor bufferizedInput =
      opLoweringState.getBufferizedInput(reshapeOp.getInput());
  Value buffer = bufferizedInput.buffer;

  FailureOr<TypedValue<MGP::TensorSpecType>> tensorSpecOr =
      opLoweringState.createTensorSpec(reshapeOp.getType());
  if (failed(tensorSpecOr))
    return failure();
  opLoweringState.createOutputTensor(reshapeOp.getResult(),
                                     loweringState.getExecuteChain(), buffer,
                                     *tensorSpecOr);
  return success();
}

LogicalResult
MOGraphLowering::lowerDebugTensorPrint(MO::DebugTensorPrintOp printOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, printOp);

  // Get the bufferized input.
  MGP::BufferizedTensor bufferizedInput =
      opLoweringState.getBufferizedInput(printOp.getInput());

  Value outChain = opLoweringState.debugTensorPrint(
      loweringState.getCurrentExecuteChain(), bufferizedInput.buffer,
      bufferizedInput.tensorSpec, printOp.getLabelAttr(),
      printOp.getInput().getType());
  loweringState.setCurrentExecuteChain(
      cast<TypedValue<MEF::ChainType>>(outChain));
  return success();
}

LogicalResult MOGraphLowering::lowerTransfer(MO::TransferOp transferOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, transferOp);

  DeviceRefAttr srcDevice = getDeviceOrDefault(loweringState.getGraph(),
                                               transferOp.getInput().getType());
  DeviceRefAttr dstDevice = transferOp.getDestDeviceAttr();
  DeviceRefAttr hostDevice = loweringState.getHostDevice();

  MGP::BufferizedTensor bufferizedInput =
      opLoweringState.getBufferizedInput(transferOp.getInput());
  FailureOr<SmallVector<MGP::MOOutputStorage>> loweredOutputsStorageOr =
      opLoweringState.createOutputStorage(loweringState.getExecuteChain(),
                                          {bufferizedInput.tensorSpec});
  if (failed(loweredOutputsStorageOr))
    return failure();
  auto &loweredOutputsStorage = *loweredOutputsStorageOr;

  Value outChain;
  if (srcDevice == hostDevice) {
    outChain = opLoweringState.bufferTransferFromHost(
        loweringState.getExecuteChain(),
        cast<TypedValue<MGP::DeviceBufferType>>(bufferizedInput.buffer),
        loweredOutputsStorage[0].buffer, transferOp.getResult().getType());
  } else if (dstDevice == hostDevice) {
    outChain = opLoweringState.bufferTransferToHost(
        loweringState.getExecuteChain(), bufferizedInput.buffer,
        cast<TypedValue<MGP::DeviceBufferType>>(
            loweredOutputsStorage[0].buffer),
        transferOp.getInput().getType());
  } else if (srcDevice.getLabel() == dstDevice.getLabel()) {
    outChain = opLoweringState.bufferTransferDevice(
        loweringState.getExecuteChain(), bufferizedInput.buffer,
        loweredOutputsStorage[0].buffer, transferOp.getInput().getType(),
        transferOp.getResult().getType());
  } else {
    return transferOp->emitOpError("mo.transfer from device '")
           << srcDevice << "' to device '" << dstDevice
           << "' is not supported.";
  }

  opLoweringState.createOutputs(outChain, loweredOutputsStorage);
  return success();
}

LogicalResult MOGraphLowering::lowerListAppend(MO::ListAppendOp listAppendOp) {
  ASSERT_STREAM(MO::isOnHost(listAppendOp),
                << "mo.list.append is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, listAppendOp);

  // Collect lowered inputs.
  Value mgpInputList =
      loweringState.lookupExecuteLoweredValue(listAppendOp.getInput());
  Value mgpElementTensor =
      loweringState.lookupExecuteLoweredValue(listAppendOp.getElement());

  // Construct output mgp list type
  MGP::ListType mgpOutputListTy = lowerListType(listAppendOp.getType());

  // Construct the corresponding mgp.list.insert op
  Value mgpOutputTensor = opLoweringState.create<MGP::ListAppendOp>(
      mgpOutputListTy, mgpInputList, mgpElementTensor);
  loweringState.emplaceExecuteLoweredValue(listAppendOp, mgpOutputTensor);

  return success();
}

LogicalResult MOGraphLowering::lowerListConcat(MO::ListConcatOp listConcatOp) {
  ASSERT_STREAM(MO::isOnHost(listConcatOp),
                << "mo.list.concat is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, listConcatOp);

  Value mgpLhsList =
      loweringState.lookupExecuteLoweredValue(listConcatOp.getLhs());
  Value mgpRhsList =
      loweringState.lookupExecuteLoweredValue(listConcatOp.getRhs());

  MGP::ListType mgpOutputListTy = lowerListType(listConcatOp.getType());

  Value mgpOutputList = opLoweringState.create<MGP::ListConcatOp>(
      mgpOutputListTy, mgpLhsList, mgpRhsList);

  loweringState.emplaceExecuteLoweredValue(listConcatOp, mgpOutputList);

  return mlir::success();
}

LogicalResult MOGraphLowering::lowerListCreate(MO::ListCreateOp listCreateOp) {
  ASSERT_STREAM(MO::isOnHost(listCreateOp),
                << "mo.list.create is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, listCreateOp);

  // Collect lowered input tensors
  SmallVector<Value> mgpInputTensors;
  for (Value input : listCreateOp.getInputs()) {
    mgpInputTensors.push_back(loweringState.lookupExecuteLoweredValue(input));
  }

  // Construct output mgp list type
  MGP::ListType mgpOutputListTy = lowerListType(listCreateOp.getType());

  // Construct the corresponding mgp.list.create op
  Value mgpOutputList = opLoweringState.create<MGP::ListCreateOp>(
      mgpOutputListTy, mgpInputTensors);
  loweringState.emplaceExecuteLoweredValue(listCreateOp, mgpOutputList);

  return success();
}

LogicalResult MOGraphLowering::lowerListGet(MO::ListGetOp listGetOp) {
  ASSERT_STREAM(MO::isOnHost(listGetOp), << "mo.list.get is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, listGetOp);

  // Collect lowered inputs.
  Value mgpInputList =
      loweringState.lookupExecuteLoweredValue(listGetOp.getInput());
  Value mgpIndexTensor =
      loweringState.lookupExecuteLoweredValue(listGetOp.getIndex());

  // Construct the corresponding mgp.list.create op
  Value mgpOutputTensor =
      opLoweringState.create<MGP::ListGetOp>(mgpInputList, mgpIndexTensor);
  loweringState.emplaceExecuteLoweredValue(listGetOp, mgpOutputTensor);

  return success();
}

LogicalResult MOGraphLowering::lowerListInsert(MO::ListInsertOp listInsertOp) {
  ASSERT_STREAM(MO::isOnHost(listInsertOp),
                << "mo.list.insert is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, listInsertOp);

  // Collect lowered inputs.
  Value mgpInputList =
      loweringState.lookupExecuteLoweredValue(listInsertOp.getInput());
  Value mgpElementTensor =
      loweringState.lookupExecuteLoweredValue(listInsertOp.getElement());
  Value mgpIndexTensor =
      loweringState.lookupExecuteLoweredValue(listInsertOp.getIndex());

  // Construct output mgp list type
  MGP::ListType mgpOutputListTy = lowerListType(listInsertOp.getType());

  // Construct the corresponding mgp.list.insert op
  Value mgpOutputTensor = opLoweringState.create<MGP::ListInsertOp>(
      mgpOutputListTy, mgpInputList, mgpElementTensor, mgpIndexTensor);
  loweringState.emplaceExecuteLoweredValue(listInsertOp, mgpOutputTensor);

  return success();
}

LogicalResult MOGraphLowering::lowerListSlice(MO::ListSliceOp listSliceOp) {
  ASSERT_STREAM(MO::isOnHost(listSliceOp), << "mo.list.slice is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, listSliceOp);

  Value mgpInputList =
      loweringState.lookupExecuteLoweredValue(listSliceOp.getInput());
  Value mgpStartTensor =
      loweringState.lookupExecuteLoweredValue(listSliceOp.getStart());
  Value mgpEndTensor =
      loweringState.lookupExecuteLoweredValue(listSliceOp.getEnd());
  Value mgpStepTensor =
      loweringState.lookupExecuteLoweredValue(listSliceOp.getStep());

  // Construct output mgp list type
  MGP::ListType mgpOutputListTy = lowerListType(listSliceOp.getType());

  Value mgpOutputTensor = opLoweringState.create<MGP::ListSliceOp>(
      mgpOutputListTy, mgpInputList, mgpStartTensor, mgpEndTensor,
      mgpStepTensor);
  loweringState.emplaceExecuteLoweredValue(listSliceOp, mgpOutputTensor);

  return success();
}

LogicalResult MOGraphLowering::lowerGuard(MO::GuardOp guardOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, guardOp);

  Value loweredChain =
      loweringState.lookupExecuteLoweredValue(guardOp.getChain());

  SmallVector<Value> loweredInputs;

  for (Value val : guardOp.getInputs()) {
    Value loweredInput = loweringState.lookupExecuteLoweredValue(val);
    loweredInputs.push_back(loweredInput);
  }

  auto mgpGuardOp =
      opLoweringState.create<MGP::GuardOp>(loweredChain, loweredInputs);

  for (auto &&[origRes, loweredRes] :
       llvm::zip(guardOp.getResults(), mgpGuardOp.getResults())) {
    loweringState.emplaceExecuteLoweredValue(origRes, loweredRes);
  }

  return success();
}

LogicalResult
MOGraphLowering::lowerChainCreate(MO::ChainCreateOp chainCreateOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, chainCreateOp);

  SmallVector<Value> loweredOperands;

  for (Value val : chainCreateOp.getOperands()) {
    Value loweredOperand = loweringState.lookupExecuteLoweredValue(val);
    loweredOperands.push_back(loweredOperand);
  }

  auto grtChainCreateOp =
      opLoweringState.create<GRT::ChainCreateOp>(loweredOperands);

  loweringState.emplaceExecuteLoweredValue(chainCreateOp.getResult(),
                                           grtChainCreateOp.getResult());

  return success();
}

LogicalResult MOGraphLowering::lowerIf(MO::IfOp ifOp) {
  ASSERT_STREAM(MO::isOnHost(ifOp), << "mo.if is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, ifOp);

  // We'll need to thread the current 'execute' chain through the if
  // 'then' and 'else' blocks.
  Value currentChain = loweringState.getCurrentExecuteChain();

  Value cond = ifOp.getCond();

  bool isScalarCond = isa<MO::ScalarType>(cond.getType());

  Value condBool;
  if (isScalarCond) {
    condBool = loweringState.lookupExecuteLoweredValue(cond);
  } else {
    // Lower the condition.
    MGP::BufferizedTensor bufferizedCond =
        opLoweringState.getBufferizedInput(cond);
    condBool = opLoweringState.bufferToBool(
        loweringState.getExecuteChain(), bufferizedCond.buffer, cond.getType());
  }

  SmallVector<Value> operands;
  operands.push_back(condBool);

  SmallVector<Type> resultTypes;
  // Always return the current 'execute' chain
  resultTypes.emplace_back(currentChain.getType());
  for (Type type : ifOp.getResultTypes())
    resultTypes.emplace_back(
        loweringState.getTypeConverter().convertType(type));

  // Create grt.if op
  auto outputIf = opLoweringState.create<GRT::IfOp>(resultTypes, operands);

  auto lower_block = [&](Region &inputRegion,
                         Region &outputRegion) -> LogicalResult {
    // Enter the lowered region's unique block.
    Block &newBlock = outputRegion.emplaceBlock();
    opLoweringState.builder.setInsertionPointToEnd(&newBlock);

    if (failed(lowerBlock(&inputRegion.getBlocks().front())))
      return failure();
    return success();
  };

  // Lower Then Block
  if (failed(lower_block(ifOp.getThenRegion(), outputIf.getThenRegion())))
    return failure();

  // Reset the main chain between blocks
  loweringState.setCurrentExecuteChain(currentChain);

  // Lower Else Block
  if (failed(lower_block(ifOp.getElseRegion(), outputIf.getElseRegion())))
    return failure();

  // Set the outputs
  loweringState.setCurrentExecuteChain(outputIf.getResult(0));
  for (auto [origResult, loweredResult] :
       llvm::zip(ifOp.getResults(), llvm::drop_begin(outputIf.getResults())))
    loweringState.emplaceExecuteLoweredValue(origResult, loweredResult);

  // Since we moved the insertion points for the then/else, reset to the main
  // block.
  opLoweringState.builder.setInsertionPointAfter(outputIf);

  return success();
}

LogicalResult MOGraphLowering::lowerWhile(MO::WhileOp whileOp) {
  ASSERT_STREAM(MO::isOnHost(whileOp), << "mo.while is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, whileOp);

  // We'll need to thread the current 'execute' chain through the while loop's
  // 'cond' and 'body' blocks.
  Value rootChain = loweringState.getExecuteChain();
  Value currentChain = loweringState.getCurrentExecuteChain();
  bool currentIsRoot = currentChain == rootChain;

  // Collect the lowered form of all the outer SSA values which (may) appear
  // in the final grt.while 'cond' and 'body' blocks.
  SmallVector<Value> loweredFreeValues;
  loweredFreeValues.reserve(1 + whileOp.getInputs().size());
  loweredFreeValues.emplace_back(currentChain);
  for (Value value : whileOp.getInputs())
    loweredFreeValues.emplace_back(
        loweringState.lookupExecuteLoweredValue(value));

  // We now know the grt.while op's result types and signature.
  SmallVector<Value> operands;
  operands.reserve(loweredFreeValues.size());
  SmallVector<Type> resultTypes;
  resultTypes.reserve(loweredFreeValues.size());
  for (Value loweredFreeValue : loweredFreeValues) {
    operands.emplace_back(loweredFreeValue);
    resultTypes.emplace_back(loweredFreeValue.getType());
  }
  auto loweredWhileOp =
      opLoweringState.create<GRT::WhileOp>(resultTypes, operands);

  auto lower_block = [&](Region &inputRegion,
                         Region &outputRegion) -> LogicalResult {
    // Enter the lowered region's unique block.
    Block &newBlock = outputRegion.emplaceBlock();
    opLoweringState.builder.setInsertionPointToEnd(&newBlock);

    // Establish the block's arguments. The current 'execute' chain comes first.
    auto loweredChainArg =
        outputRegion.addArgument(currentChain.getType(), whileOp.getLoc());
    loweringState.setCurrentExecuteChain(loweredChainArg);
    if (currentIsRoot)
      loweringState.setRootExecuteChain(loweredChainArg);
    // else: if the block needs to reference the root chain then an
    //       additional block argument will be added by fixupIsolatedFromAbove.

    // Then an arg for each of the original region args.
    // We'll use the loweredFreeValues since they have the lowered types
    // already, though we skip the chain.
    for (auto [loweredFreeValue, arg] : llvm::zip(
             llvm::drop_begin(loweredFreeValues), inputRegion.getArguments())) {
      Value loweredArg =
          outputRegion.addArgument(loweredFreeValue.getType(), arg.getLoc());
      loweringState.emplaceExecuteLoweredValue(arg, loweredArg);
    }

    // Lower the block, including its terminating mo.yield.
    if (failed(lowerBlock(&inputRegion.getBlocks().front())))
      return failure();

    return success();
  };

  // Lower the 'cond' block. We'll later need to make 2 adjustments:
  // 1) The mo.while 'cond' block will mo.yield only the condition result. That
  //    mo.yield will be lowered to a mef.output with the 'execute' chain and
  //    lowered condition results. However, the grt.while requires the 'cond'
  //    block to return a result for all signature arguments plus the condition,
  //    so we'll need to include those.
  // 2) The lowered condition will be a mgp.tensor<bool>, which we need to
  //    convert to an i1.
  if (failed(
          lower_block(whileOp.getCondRegion(), loweredWhileOp.getCondRegion())))
    return failure();

  // Lower the 'body' block.
  if (failed(
          lower_block(whileOp.getBodyRegion(), loweredWhileOp.getBodyRegion())))
    return failure();

  // Now repair the lowered 'cond' block's mef.output to include all the
  // additional variables between the output chain and the condition,
  // and fix the condition to be an i1.
  {
    auto origYieldOp =
        cast<MO::YieldOp>(whileOp.getCondRegion().front().getTerminator());
    assert(origYieldOp->getNumOperands() == 1); // condition
    Type condMoTensorType = origYieldOp.getOperand(0).getType();

    auto loweredCondOutputOp = cast<MEF::OutputOp>(
        loweredWhileOp.getCondRegion().front().getTerminator());
    assert(loweredCondOutputOp->getNumOperands() == 2); // chain, condition
    Value loweredOutputChain = loweredCondOutputOp->getOperand(0);
    Value loweredCondTensor = loweredCondOutputOp->getOperand(1);

    SmallVector<Value> condOutputOpOperands;
    condOutputOpOperands.reserve(loweredWhileOp->getNumResults() + 1);

    // chain
    condOutputOpOperands.emplace_back(loweredOutputChain);

    // additional non-chain args
    llvm::append_range(
        condOutputOpOperands,
        llvm::drop_begin(loweredWhileOp.getCondRegion().getArguments()));

    // i1 condition
    opLoweringState.builder.setInsertionPoint(loweredCondOutputOp);
    Value condBuffer =
        opLoweringState.tensorExtract(loweredCondTensor, condMoTensorType)
            .first;
    Value loweredInputChain = loweredWhileOp.getCondRegion().getArgument(0);
    Value condBool = opLoweringState.bufferToBool(loweredInputChain, condBuffer,
                                                  condMoTensorType);
    condOutputOpOperands.emplace_back(condBool);

    loweredCondOutputOp.getOperandsMutable().clear();
    loweredCondOutputOp.getOperandsMutable().append(condOutputOpOperands);
  }

  // Capture the lowered outputs.
  assert(loweredWhileOp.getNumResults() == 1 + whileOp.getNumResults());
  loweringState.setCurrentExecuteChain(loweredWhileOp.getResult(0));
  loweringState.setRootExecuteChain(rootChain);
  for (auto [origResult, loweredResult] : llvm::zip(
           whileOp.getResults(), llvm::drop_begin(loweredWhileOp.getResults())))
    loweringState.emplaceExecuteLoweredValue(origResult, loweredResult);

  // Since we moved the insertion points for the 'cond' and 'body' blocks,
  // reset back to the main block
  opLoweringState.builder.setInsertionPointAfter(loweredWhileOp);

  return success();
}

LogicalResult MOGraphLowering::lowerBlock(Block *block) {
  // Ensure the block is sorted with operands to operations appearing before
  // their use.
  mlir::sortTopologically(block);

  for (Operation &op : *block) {
    if (auto declOp = dyn_cast<KGEN::ParamDeclareOp>(op)) {
      if (failed(lowerParamDeclare(block->getParent(), declOp)))
        return failure();
      continue;
    }

    if (auto paramToValueOp = dyn_cast<MOSH::ParamToValueOp>(op)) {
      if (failed(lowerParamToValue(block->getParent(), paramToValueOp)))
        return failure();
      continue;
    }

    if (auto toTensorOp = dyn_cast<MO::IndexToTensorOp>(op)) {
      if (failed(lowerIndexToTensorOp(toTensorOp)))
        return failure();
      continue;
    }

    if (auto toTensorOp = dyn_cast<MO::Shape::ToTensorOp>(op)) {
      if (failed(lowerShapeToTensorOp(toTensorOp)))
        return failure();
      continue;
    }

    if (auto rebindOp = dyn_cast<MO::RebindOp>(op)) {
      if (failed(lowerRebind(rebindOp)))
        return failure();
      continue;
    }

    if (auto constOp = dyn_cast<MO::ConstantOp>(op)) {
      if (failed(lowerConstant(constOp)))
        return failure();
      continue;
    }

    if (auto constOp = dyn_cast<MO::ConstantScalarOp>(op)) {
      if (failed(lowerConstantScalar(constOp)))
        return failure();
      continue;
    }

    if (auto reshapeOp = dyn_cast<MO::ReshapeOp>(op)) {
      if (failed(lowerReshape(reshapeOp)))
        return failure();
      continue;
    }

    if (auto reshapeOp = dyn_cast<MO::Static::ReshapeOp>(op)) {
      if (failed(lowerStaticReshape(reshapeOp)))
        return failure();
      continue;
    }

    if (auto debugTensorPrintOp = dyn_cast<MO::DebugTensorPrintOp>(op)) {
      if (failed(lowerDebugTensorPrint(debugTensorPrintOp)))
        return failure();
      continue;
    }

    if (auto transferOp = dyn_cast<MO::TransferOp>(op)) {
      if (failed(lowerTransfer(transferOp)))
        return failure();
      continue;
    }

    if (auto listAppendOp = dyn_cast<MO::ListAppendOp>(op)) {
      if (failed(lowerListAppend(listAppendOp)))
        return failure();
      continue;
    }

    if (auto listConcatOp = dyn_cast<MO::ListConcatOp>(op)) {
      if (failed(lowerListConcat(listConcatOp)))
        return failure();
      continue;
    }

    if (auto listCreateOp = dyn_cast<MO::ListCreateOp>(op)) {
      if (failed(lowerListCreate(listCreateOp)))
        return failure();
      continue;
    }

    if (auto listGetOp = dyn_cast<MO::ListGetOp>(op)) {
      if (failed(lowerListGet(listGetOp)))
        return failure();
      continue;
    }

    if (auto listInsertOp = dyn_cast<MO::ListInsertOp>(op)) {
      if (failed(lowerListInsert(listInsertOp)))
        return failure();
      continue;
    }

    if (auto listSliceOp = dyn_cast<MO::ListSliceOp>(op)) {
      if (failed(lowerListSlice(listSliceOp)))
        return failure();
      continue;
    }

    // Lower mo.output to mef.output trivially.
    if (auto output = dyn_cast<MO::OutputOp>(op)) {
      if (failed(lowerOutput(output)))
        return failure();
      continue;
    }

    // Lower mo.if to grf.if, recursively lowering the body of
    // the 'then' and 'else' blocks.
    if (auto ifOp = dyn_cast<MO::IfOp>(op)) {
      if (failed(lowerIf(ifOp)))
        return failure();
      continue;
    }

    // Lower mo.while to grf.while, recursively lowering the body of
    // the 'cond' and 'body' blocks.
    if (auto whileOp = dyn_cast<MO::WhileOp>(op)) {
      if (failed(lowerWhile(whileOp)))
        return failure();
      continue;
    }

    // Lower mo.yield to mef.output trivially.
    if (auto output = dyn_cast<MO::YieldOp>(op)) {
      if (failed(lowerYield(output)))
        return failure();
      continue;
    }

    // Lowers mo.guard to
    if (auto guardOp = dyn_cast<MO::GuardOp>(op)) {
      if (failed(lowerGuard(guardOp)))
        return failure();
      continue;
    }

    // Lowers mo.guard to
    if (auto chainCreateOp = dyn_cast<MO::ChainCreateOp>(op)) {
      if (failed(lowerChainCreate(chainCreateOp)))
        return failure();
      continue;
    }

    // Implementers of the lowering interface define their own logic.
    if (auto lowerable = dyn_cast<MGP::LowerableToMGPInterface>(op)) {
      if (failed(lowerable.lower(loweringState)))
        return failure();
      continue;
    }

    return op.emitOpError("currently not supported");
  }
  return success();
}

LogicalResult MOGraphLowering::run() {
  MO::GraphOp &graph = loweringState.getGraph();

  // Create the mgp.model and the regions for the init/execute phases.
  if (failed(createModelOp()))
    return failure();

  // Enter the main body of the graph
  for (auto [graphArg, exeArg] :
       llvm::zip(graph.getBody().front().getArguments(),
                 loweringState.getModelOp().getExecuteInputs())) {
    auto hasParamsOr =
        loweringState.lowerArg(&graph.getRegion(), graphArg, exeArg);
    if (hasParamsOr.isError())
      return graph->emitOpError(hasParamsOr.getError());

    // If the argument has no uses (and no parameters) then just ignore it.
    if (!hasParamsOr.get() && graphArg.use_empty())
      continue;

    loweringState.emplaceExecuteLoweredValue(graphArg, exeArg);
  }

  // Lower all of the operations in the graph.
  if (failed(lowerBlock(graph.getBodyBlock())))
    return failure();

  OpBuilder b(graph);
  MGP::ModelOp model = loweringState.getModelOp();
  // Bind any free variables in grt.if and grt.while control flow ops.
  // Adds and returns the new model to the module.
  (void)MGP::fixupIsolatedFromAbove(b, model);
  // We've fixed up the MGP model, so remove the old one.
  model.erase();
  // We're also done with the original mo.graph.
  graph.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// MOToPrimitivesPass
//===----------------------------------------------------------------------===//

namespace {
struct MOToPrimitivesPass
    : public MGP::impl::MOToPrimitivesBase<MOToPrimitivesPass> {
  using MOToPrimitivesBase::MOToPrimitivesBase;

  void runOnOperation() override;

  void getDependentDialects(::mlir::DialectRegistry &registry) const override;
};
} // namespace

void MOToPrimitivesPass::getDependentDialects(DialectRegistry &registry) const {
  MOToPrimitivesBase::getDependentDialects(registry);
  // Bring in all the device-specific dialects.
  MO::registerAllDeviceDialects(registry);
}

void MOToPrimitivesPass::runOnOperation() {
  ModuleOp theModule = getOperation();

  // Convert each mo.graph into a mgp.model.
  for (auto graphOp :
       llvm::make_early_inc_range(theModule.getOps<MO::GraphOp>())) {
    if (failed(MOGraphLowering(graphOp, compiledFrameworkLabel.getValue(),
                               minimumCpuAlignment.getValue())
                   .run()))
      return signalPassFailure();
  }
}
