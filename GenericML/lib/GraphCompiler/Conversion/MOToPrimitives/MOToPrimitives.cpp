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
#include "Support/Compiler/OperationUtils.h"
#include "Support/MArchTarget/Host.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/ML/CommonAttributes.h"
#include "Support/STLExtras.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
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

  // Emit code into the 'init' block to check the runtime environment
  // satisfies all the compile-time device specs. This requires an
  // !mgp.context slot to hold the calculated device spec map.
  LogicalResult setupCheckRuntime(Location loc);

  // Establish type builders for all the (virtual) devices needed by the
  // model. These will be available via the lowering state's type converter.
  LogicalResult setupTypeBuildersForAllDevices();

  // Establisnh a buffer builder for the 'host' (ie CPU) device.
  LogicalResult makeCPUBufferBuilder(DeviceSpecAttr deviceSpec);

  // Establish buffer builders for all the (virtual) devices needed by the
  // model. This may require emitting device-specific setup code.
  LogicalResult setupBufferBuildersForAllDevices(Location loc);

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
  LogicalResult lowerParamDeclare(KGEN::ParamDeclareOp declOp);

  /// Lower a parameter constant op.
  LogicalResult lowerParamConstant(KGEN::ParamConstantOp paramConstOp);

  /// Lower an index -> scalar-like tensor cast op.
  LogicalResult lowerIndexToTensorOp(MO::IndexToTensorOp toTensorOp);

  /// Lower a shape -> shape-like tensor cast op.
  LogicalResult lowerShapeToTensorOp(MO::Shape::ToTensorOp toTensorOp);

  /// Lower a tensor rebind.
  LogicalResult lowerRebind(MO::RebindOp rebindOp);

  /// Lowers mo.constant op.
  LogicalResult lowerConstant(MO::ConstantOp constOp);

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

  /// Lowers mo.list.create op.
  LogicalResult lowerListCreate(MO::ListCreateOp listCreateOp);

  /// Lowers mo.list.get op.
  LogicalResult lowerListGet(MO::ListGetOp listGetOp);

  /// Lowers mo.list.insert op.
  LogicalResult lowerListInsert(MO::ListInsertOp listInsertOp);

private:
  /// Lowers a Block
  LogicalResult lowerBlock(mlir::Block *block);

  MGP::MOGraphLoweringState loweringState;
};

} // namespace

//===----------------------------------------------------------------------===//
// Creation of the init/execute invokables
//===----------------------------------------------------------------------===//

LogicalResult MOGraphLowering::setupCheckRuntime(Location loc) {
  FailureOr<DeviceSpecCollectionAttr> deviceSpecsOr =
      loweringState.getDeviceSpecs();
  if (failed(deviceSpecsOr))
    return failure();

  uint64_t slot = loweringState.getInitMgpContext().getNumSlots();
  auto checkRuntimeOp =
      loweringState.getInitOpBuilder().create<MGP::CheckRuntimeOp>(
          loc,
          /*inChain=*/loweringState.getInitChain(),
          /*theContext=*/loweringState.getInitMgpContext(),
          /*aSlot=*/slot,
          /*bDevices=*/*deviceSpecsOr);
  loweringState.setCheckRuntimeOp(checkRuntimeOp);
  loweringState.getInitMgpContext().setNumSlots(slot + 1);
  loweringState.setInitChain(checkRuntimeOp.getOutChain());
  return success();
}

LogicalResult MOGraphLowering::setupTypeBuildersForAllDevices() {
  FailureOr<DeviceSpecCollectionAttr> deviceSpecsOr =
      loweringState.getDeviceSpecs();
  if (failed(deviceSpecsOr))
    return failure();

  for (DeviceSpecAttr deviceSpec : deviceSpecsOr->getDevices()) {
    if (deviceSpec.getRef().getLabel() == kCPULabel) {
      // We know how to work with MGP tensors for the CPU.
      loweringState.registerTypeBuilder(
          deviceSpec.getRef(), std::make_unique<MGP::CpuTypeBuilder>());
    } else {
      // Rely on the registered 'makeTypeBuilder' function for this device.
      if (failed(MO::makeTypeBuilder(deviceSpec, loweringState)))
        return failure();
    }
  }
  return success();
}

LogicalResult MOGraphLowering::makeCPUBufferBuilder(DeviceSpecAttr deviceSpec) {
  // Setup the compiled kernel context for all CPU kernels.
  uint64_t slot = loweringState.getInitMgpContext().getNumSlots();
  auto createContextOp =
      loweringState.getInitOpBuilder().create<MGP::CpuKernelCreateContextOp>(
          loweringState.getGraph().getLoc(),
          /*inChain=*/loweringState.getInitChain(),
          /*theContext=*/loweringState.getInitMgpContext(),
          /*slot=*/slot);
  loweringState.getInitMgpContext().setNumSlots(slot + 1);
  loweringState.setInitChain(createContextOp.getOutChain());

  // Register the builder for CPU.
  loweringState.registerBufferBuilder(
      deviceSpec.getRef(), std::make_unique<MGP::CpuBufferBuilder>(slot));

  return success();
}

LogicalResult MOGraphLowering::setupBufferBuildersForAllDevices(Location loc) {
  FailureOr<DeviceSpecCollectionAttr> deviceSpecsOr =
      loweringState.getDeviceSpecs();
  if (failed(deviceSpecsOr))
    return failure();

  // We must always establish the buffer builder for the 'host' first,
  // since it may be needed by other device-specific builders.
  DeviceSpecAttr hostDeviceSpec = deviceSpecsOr->getHostDeviceSpec();
  assert(hostDeviceSpec.getRef() == getHostDevice(loweringState.getGraph()) &&
         "The 'host' device must be a CPU");
  // We always know how to work with the host CPU device.
  if (failed(makeCPUBufferBuilder(hostDeviceSpec)))
    return failure();

  // Now we can establish buffer builders for all the other devices, if any.
  for (DeviceSpecAttr deviceSpec : deviceSpecsOr->getDevices()) {
    if (deviceSpec.getRef() == hostDeviceSpec.getRef())
      continue;

    // Rely on the registered 'makeBufferBuilder' function for this device.
    // It may inject setup code into the 'init' block as required, and
    // then register a BufferBuilder for the device spec's device reference.
    // It may also grab the 'host' buffer builder to handle ops which execute
    // via the host.
    if (failed(MO::makeBufferBuilder(hostDeviceSpec.getRef(), deviceSpec,
                                     loweringState)))
      return failure();
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
          graph.getArgumentTypes(), inputTypes)) ||
      failed(loweringState.getTypeConverter().convertTypes(
          graph.getResultTypes(), outputTypes))) {
    return graph.emitOpError("failed to convert mo graph input/output types");
  }

  // Preserve the MO::GraphOp argument locations in the MGP::ModelOp 'execute'
  // block argument locations.
  SmallVector<Location> inputLocs;
  for (BlockArgument arg : graph.getArguments())
    inputLocs.push_back(arg.getLoc());

  // Collect additional attributes to include with the MGP::ModelOp to guide
  // the MEF executor.
  SmallVector<NamedAttribute> extraAttrs;

  ArrayAttr argNamesAttr =
      graph->getAttrOfType<ArrayAttr>(kMgpModelArgumentNames);
  ArrayAttr resultNamesAttr =
      graph->getAttrOfType<ArrayAttr>(kMgpModelResultNames);
  assert((argNamesAttr == nullptr) == (resultNamesAttr == nullptr) &&
         "expected either both or neither of the `argument_names` and "
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
  auto modelOp = b.create<MGP::ModelOp>(
      graph.getLoc(),
      /*sym_name=*/graph.getNameAttr(),
      /*frameworkName=*/loweringState.getCompiledFrameworkLabel(),
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

  // Check the runtime environment.
  if (failed(setupCheckRuntime(graph.getLoc())))
    return failure();

  // Initialize the execute state.
  Block *executeBlock = loweringState.getModelOp().getExecuteBlock();
  loweringState.getExecuteOpBuilder().setInsertionPointToEnd(executeBlock);

  // Extract chain and MGP context from arguments
  loweringState.setExecuteChain(executeBlock->getArgument(0));
  loweringState.setExecuteMgpContext(
      cast<TypedValue<MGP::ContextType>>(executeBlock->getArgument(1)));

  // Establish buffer builders for all the devices. This may require some
  // extra device-specific setup in the 'init' block.
  // CAUTION: Must be done after the 'init' and 'execute' mgp.contexts have
  // been established, but before we lower any device-specific ops.
  if (failed(setupBufferBuildersForAllDevices(graph.getLoc())))
    return failure();

  return success();
}

MGP::ListType MOGraphLowering::lowerListType(MO::ListType listTy) {
  auto outputTensorTy = dyn_cast<MO::TensorType>(listTy.getElementType());
  // TODO support other element type
  assert(outputTensorTy && "only support lowering list of tensors");

  MLIRContext *ctx = listTy.getContext();
  MGP::CpuTensorSpecType outputTensorSpecTy =
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
  operands.push_back(loweringState.getExecuteChain());

  for (Value operand : output.getOperands())
    operands.push_back(loweringState.lookupExecuteLoweredValue(operand));

  // Finally, we can create the mef.output op.
  loweringState.getExecuteOpBuilder().create<MEF::OutputOp>(output.getLoc(),
                                                            operands);
  return success();
}

LogicalResult MOGraphLowering::lowerYield(MO::YieldOp yield) {
  // Execute phase lowering.

  // Figure out what values to return. We always implicitly return the main
  // chain.
  SmallVector<Value> operands;
  operands.push_back(loweringState.getExecuteChain());

  // If we are yielding from a while then it will expect the context as well.
  if (isa<MO::WhileOp>(yield->getParentOp()))
    operands.push_back(loweringState.getExecuteMgpContext());

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

LogicalResult MOGraphLowering::lowerParamDeclare(KGEN::ParamDeclareOp declOp) {
  // We first lower the expression.
  auto valueOr =
      loweringState.lowerExpression(declOp->getLoc(), declOp.getValue());
  if (valueOr.isError())
    return declOp->emitOpError(valueOr.getError());

  Value value = *valueOr;
  // Then map the parameter declaration to it.
  if (auto err = loweringState.lowerParamDecl(declOp.getParamDecl(), value))
    return declOp->emitOpError(err.getError());

  return success();
}

LogicalResult
MOGraphLowering::lowerParamConstant(KGEN::ParamConstantOp paramConstOp) {
  auto valueOr = loweringState.lowerExpression(paramConstOp.getLoc(),
                                               paramConstOp.getValue());
  if (valueOr.isError())
    return paramConstOp->emitOpError(valueOr.getError());

  loweringState.emplaceExecuteLoweredValue(paramConstOp.getResult(), *valueOr);
  return success();
}

LogicalResult
MOGraphLowering::lowerIndexToTensorOp(MO::IndexToTensorOp toTensorOp) {
  // NOTE: This op is for CPU only.
  MGP::MOOpLoweringState opLoweringState(loweringState, toTensorOp,
                                         loweringState.getHostDevice());

  // Setup output buffer.
  FailureOr<SmallVector<MGP::BufferizedTensor>> bufferizedOutputsOr =
      opLoweringState.createOutputBuffers(loweringState.getExecuteChain());
  if (failed(bufferizedOutputsOr))
    return failure();
  auto &bufferizedOutput = (*bufferizedOutputsOr)[0];
  assert(isa<MGP::CpuBufferType>(bufferizedOutput.buffer.getType()) &&
         "must be on CPU");

  // Fill the buffer with the index value.
  Value loweredIndexVal =
      loweringState.lookupExecuteLoweredValue(toTensorOp.getInput());
  auto bufferSetOp = opLoweringState.create<MGP::CpuBufferSetWithIndexOp>(
      /*buffer=*/bufferizedOutput.buffer, /*value=*/loweredIndexVal);

  // Setup output tensor.
  opLoweringState.createOutputTensors(bufferSetOp.getOutChain(),
                                      *bufferizedOutputsOr);
  return success();
}

LogicalResult
MOGraphLowering::lowerShapeToTensorOp(MO::Shape::ToTensorOp toTensorOp) {
  // NOTE: This op is for CPU only.
  MGP::MOOpLoweringState opLoweringState(loweringState, toTensorOp,
                                         loweringState.getHostDevice());

  // Setup output buffer.
  FailureOr<SmallVector<MGP::BufferizedTensor>> bufferizedOutputsOr =
      opLoweringState.createOutputBuffers(loweringState.getExecuteChain());
  if (failed(bufferizedOutputsOr))
    return failure();
  auto &bufferizedOutput = (*bufferizedOutputsOr)[0];
  assert(isa<MGP::CpuBufferType>(bufferizedOutput.buffer.getType()) &&
         "must be on CPU");

  // Fill the buffer with the runtime dimensions.
  auto loweredShape =
      loweringState.lookupExecuteLoweredValue(toTensorOp.getInput());
  auto loweredTensorSpec =
      cast<TypedValue<MGP::CpuTensorSpecType>>(loweredShape);
  // Note loweredTensorSpec is the encoding for the shape we are trying to
  // reify as a tensor, whereas bufferizedOutput.tensorSpec just describes
  // the statically known shape of our output tensor.
  auto dynDimsOp = opLoweringState.create<MGP::CpuTensorSpecDimsDynamicOp>(
      /*spec=*/loweredTensorSpec,
      /*rawDims=*/loweredTensorSpec.getType().getRawDims());
  auto bufferSetOp = opLoweringState.create<MGP::CpuBufferSetWithIndexOp>(
      /*buffer=*/bufferizedOutput.buffer,
      /*values=*/dynDimsOp.getResults());

  // Setup output tensor.
  opLoweringState.createOutputTensors(bufferSetOp.getOutChain(),
                                      *bufferizedOutputsOr);
  return success();
}

/// Returns the lowered tensor that the rebind's result should map to.
static Value getRebindResTensor(MO::RebindOp rebindOp,
                                MGP::MOGraphLoweringState &loweringState) {
  MGP::MOOpLoweringState opLoweringState(loweringState, rebindOp,
                                         getDeviceOrDefault(rebindOp));

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
  Value buffer = opLoweringState.tensorExtract(loweredInput).first;
  TypedValue<MGP::CpuTensorSpecType> tensorSpec =
      opLoweringState.create<MGP::CpuTensorSpecCreateOp>(outSpecTy,
                                                         runtimeDims);
  return opLoweringState.tensorCreate(loweringState.getExecuteChain(), buffer,
                                      tensorSpec);
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

      MGP::CpuTensorSpecType specTy = dynDims.getSpec().getType();
      Value rtVal = dynDims->getResult(specTy.getDynamicDimIndex(idx));
      if (auto err = loweringState.lowerParamDecl(decl, rtVal))
        return rebindOp->emitOpError(err.getError());
    }
  }

  return success();
}

LogicalResult MOGraphLowering::lowerConstant(MO::ConstantOp constOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, constOp,
                                         getDeviceOrDefault(constOp));

  ErrorOr<ElementsAttr> elementsOr = lowerElementsAttr(constOp.getValue());
  if (elementsOr)
    return constOp->emitOpError(elementsOr.getError());

  Value buffer = opLoweringState.bufferConstant(*elementsOr);
  FailureOr<TypedValue<MGP::CpuTensorSpecType>> tensorSpecOr =
      opLoweringState.createTensorSpec(constOp.getType());
  if (failed(tensorSpecOr))
    return failure();
  opLoweringState.createOutputTensor(constOp.getResult(),
                                     loweringState.getExecuteChain(), buffer,
                                     *tensorSpecOr);
  return success();
}

LogicalResult
MOGraphLowering::lowerStaticReshape(MO::Static::ReshapeOp reshapeOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, reshapeOp,
                                         getDeviceOrDefault(reshapeOp));

  // Get the bufferized input.
  MGP::BufferizedTensor bufferizedInput =
      opLoweringState.getBufferizedInput(reshapeOp.getInput());
  Value buffer = bufferizedInput.buffer;

  // Create a tensor result with spec matching the result type.
  FailureOr<TypedValue<MGP::CpuTensorSpecType>> tensorSpecOr =
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
  MGP::MOOpLoweringState opLoweringState(loweringState, reshapeOp,
                                         getDeviceOrDefault(reshapeOp));

  // Get the bufferized input.
  MGP::BufferizedTensor bufferizedInput =
      opLoweringState.getBufferizedInput(reshapeOp.getInput());
  Value buffer = bufferizedInput.buffer;

  FailureOr<TypedValue<MGP::CpuTensorSpecType>> tensorSpecOr =
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
  MGP::MOOpLoweringState opLoweringState(loweringState, printOp,
                                         getDeviceOrDefault(printOp));
  auto loweredTensor =
      loweringState.lookupExecuteLoweredValue(printOp.getInput());
  Value outChain = opLoweringState.debugTensorPrint(
      loweringState.getExecuteChain(), loweredTensor, printOp.getLabelAttr());
  loweringState.setExecuteChain(outChain);
  return success();
}

LogicalResult MOGraphLowering::lowerTransfer(MO::TransferOp transferOp) {
  DeviceRefAttr srcDevice = getDeviceOrDefault(loweringState.getGraph(),
                                               transferOp.getInput().getType());
  if (!srcDevice) {
    return transferOp->emitOpError(
        "requires operand tensor type to be annotated with the source device.");
  }
  DeviceRefAttr dstDevice = transferOp.getDestDeviceAttr();
  DeviceRefAttr hostDevice = loweringState.getHostDevice();
  if (srcDevice == dstDevice) {
    return transferOp->emitOpError("transfer between the same device ")
           << srcDevice << " is not supported.";
  }

  MGP::MOOpLoweringState opSrcLoweringState(loweringState, transferOp,
                                            srcDevice);
  MGP::BufferizedTensor bufferizedInput =
      opSrcLoweringState.getBufferizedInput(transferOp.getInput());

  MGP::MOOpLoweringState opDstLoweringState(loweringState, transferOp,
                                            dstDevice);
  FailureOr<SmallVector<MGP::BufferizedTensor>> bufferizedOutputsOr =
      opDstLoweringState.createOutputBuffers(loweringState.getExecuteChain());
  if (failed(bufferizedOutputsOr))
    return failure();
  SmallVector<MGP::BufferizedTensor> &bufferizedOutputs = *bufferizedOutputsOr;

  Value outChain;
  if (srcDevice == hostDevice) {
    outChain = opDstLoweringState.bufferTransferFromHost(
        loweringState.getExecuteChain(),
        cast<TypedValue<MGP::CpuBufferType>>(bufferizedInput.buffer),
        bufferizedOutputs[0].buffer);
  } else if (dstDevice == hostDevice) {
    outChain = opSrcLoweringState.bufferTransferToHost(
        loweringState.getExecuteChain(), bufferizedInput.buffer,
        cast<TypedValue<MGP::CpuBufferType>>(bufferizedOutputs[0].buffer));
  } else {
    return transferOp->emitOpError("transfer from ")
           << srcDevice << " to " << dstDevice
           << " is not currently supported: either the source or destination "
              "device must be the host device "
           << hostDevice << ".";
  }

  opDstLoweringState.createOutputTensors(outChain, bufferizedOutputs);
  return success();
}

LogicalResult MOGraphLowering::lowerListAppend(MO::ListAppendOp listAppendOp) {
  // Collect lowered input tensors
  Value mgpInputList =
      loweringState.lookupExecuteLoweredValue(listAppendOp.getInput());
  Value mgpElementTensor =
      loweringState.lookupExecuteLoweredValue(listAppendOp.getElement());

  // Construct output mgp list type
  MGP::ListType mgpOutputListTy = lowerListType(listAppendOp.getType());

  // Construct the corresponding mgp.list.insert op
  MGP::MOOpLoweringState opLoweringState(loweringState, listAppendOp,
                                         getDeviceOrDefault(listAppendOp));
  Value mgpOutputTensor = opLoweringState.create<MGP::ListAppendOp>(
      mgpOutputListTy, mgpInputList, mgpElementTensor);
  loweringState.emplaceExecuteLoweredValue(listAppendOp, mgpOutputTensor);

  return success();
}

LogicalResult MOGraphLowering::lowerListCreate(MO::ListCreateOp listCreateOp) {
  // Collect lowered input tensors
  SmallVector<Value> mgpInputTensors;
  for (Value input : listCreateOp.getInputs()) {
    mgpInputTensors.push_back(loweringState.lookupExecuteLoweredValue(input));
  }

  // Construct output mgp list type
  MGP::ListType mgpOutputListTy = lowerListType(listCreateOp.getType());

  // Construct the corresponding mgp.list.create op
  MGP::MOOpLoweringState opLoweringState(loweringState, listCreateOp,
                                         getDeviceOrDefault(listCreateOp));
  Value mgpOutputList = opLoweringState.create<MGP::ListCreateOp>(
      mgpOutputListTy, mgpInputTensors);
  loweringState.emplaceExecuteLoweredValue(listCreateOp, mgpOutputList);

  return success();
}

LogicalResult MOGraphLowering::lowerListGet(MO::ListGetOp listGetOp) {
  // Collect lowered input tensors
  Value mgpInputList =
      loweringState.lookupExecuteLoweredValue(listGetOp.getInput());
  Value mgpIndexTensor =
      loweringState.lookupExecuteLoweredValue(listGetOp.getIndex());

  // Construct the corresponding mgp.list.create op
  MGP::MOOpLoweringState opLoweringState(loweringState, listGetOp,
                                         getDeviceOrDefault(listGetOp));
  Value mgpOutputTensor =
      opLoweringState.create<MGP::ListGetOp>(mgpInputList, mgpIndexTensor);
  loweringState.emplaceExecuteLoweredValue(listGetOp, mgpOutputTensor);

  return success();
}

LogicalResult MOGraphLowering::lowerListInsert(MO::ListInsertOp listInsertOp) {
  // Collect lowered input tensors
  Value mgpInputList =
      loweringState.lookupExecuteLoweredValue(listInsertOp.getInput());
  Value mgpElementTensor =
      loweringState.lookupExecuteLoweredValue(listInsertOp.getElement());
  Value mgpIndexTensor =
      loweringState.lookupExecuteLoweredValue(listInsertOp.getIndex());

  // Construct output mgp list type
  MGP::ListType mgpOutputListTy = lowerListType(listInsertOp.getType());

  // Construct the corresponding mgp.list.insert op
  MGP::MOOpLoweringState opLoweringState(loweringState, listInsertOp,
                                         getDeviceOrDefault(listInsertOp));
  Value mgpOutputTensor = opLoweringState.create<MGP::ListInsertOp>(
      mgpOutputListTy, mgpInputList, mgpElementTensor, mgpIndexTensor);
  loweringState.emplaceExecuteLoweredValue(listInsertOp, mgpOutputTensor);

  return success();
}

LogicalResult MOGraphLowering::lowerIf(MO::IfOp ifOp) {
  if (getDeviceOrDefault(ifOp) != loweringState.getHostDevice()) {
    ifOp->emitOpError("op must be assigned to the host");
    return failure();
  }

  MGP::MOOpLoweringState opLoweringState(loweringState, ifOp,
                                         getDeviceOrDefault(ifOp));
  Value mainChain = loweringState.getExecuteChain();

  // Lower the condition.
  MGP::BufferizedTensor bufferizedCond =
      opLoweringState.getBufferizedInput(ifOp.getCond());
  Value condBool = opLoweringState.create<MGP::ToGRTBoolOp>(
      mainChain, bufferizedCond.buffer);

  SmallVector<Value> operands;
  operands.push_back(condBool);

  SmallVector<Type> resultTypes;
  // Always return the main chain
  resultTypes.emplace_back(mainChain.getType());
  for (Type type : ifOp.getResultTypes())
    resultTypes.emplace_back(
        loweringState.getTypeConverter().convertType(type));

  // Create grt.if op
  auto outputIf = opLoweringState.create<GRT::IfOp>(resultTypes, operands);

  auto lower_block = [&](Region &inputRegion,
                         Region &outputRegion) -> LogicalResult {
    // Enter a new Block
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
  loweringState.setExecuteChain(mainChain);

  // Lower Else Block
  if (failed(lower_block(ifOp.getElseRegion(), outputIf.getElseRegion())))
    return failure();

  // This function finds all of the SSA values implicitly captured by
  // the GRT::IfOp and returns them for argument inlining.
  // Note: The function does not de-duplicate if the Values are
  // used repeatedly, need to handle that below
  llvm::SetVector<Value> uniqueCaptures;
  operationIsIsolatedFromAbove(outputIf, &uniqueCaptures,
                               /* allowIsolated */ true);
  uniqueCaptures.erase(llvm::find(uniqueCaptures, condBool));

  // Pass captures as inputs
  outputIf.getInputsMutable().append(llvm::to_vector(uniqueCaptures));

  auto isolate_region = [&](Region &region) -> LogicalResult {
    for (Value capture : uniqueCaptures) {
      auto arg = region.addArgument(capture.getType(), ifOp.getLoc());
      capture.replaceUsesWithIf(arg, [&](OpOperand &use) {
        return region.findAncestorOpInRegion(*use.getOwner());
      });
    }
    return success();
  };

  // Isolate Then Block
  if (failed(isolate_region(outputIf.getThenRegion())))
    return failure();

  // Isolate Else Block
  if (failed(isolate_region(outputIf.getElseRegion())))
    return failure();

  // Set the outputs
  loweringState.setExecuteChain(outputIf.getResult(0));
  for (size_t i = 1, e = resultTypes.size(); i < e; i++) {
    loweringState.emplaceExecuteLoweredValue(ifOp.getResult(i - 1),
                                             outputIf.getResult(i));
  }

  // since we moved the insertion points for the then/else, reset to the main
  // block
  opLoweringState.builder.setInsertionPointAfter(outputIf);

  return success();
}

LogicalResult MOGraphLowering::lowerWhile(MO::WhileOp whileOp) {
  if (getDeviceOrDefault(whileOp) != loweringState.getHostDevice()) {
    whileOp->emitOpError("op must be assigned to the host");
    return failure();
  }

  MGP::MOOpLoweringState opLoweringState(loweringState, whileOp,
                                         getDeviceOrDefault(whileOp));

  // We'll need to thread the 'execute' chain through the while loop's 'cond'
  // and 'body' blocks.
  Value mainChain = loweringState.getExecuteChain();

  // Collect the lowered form of all the outer SSA values which (may) appear
  // in the final grt.while 'cond' and 'body' blocks.
  SmallVector<Value> loweredFreeValues;
  loweredFreeValues.emplace_back(mainChain);
  loweredFreeValues.emplace_back(loweringState.getExecuteMgpContext());
  for (Value value : whileOp.getInputs())
    loweredFreeValues.emplace_back(
        loweringState.lookupExecuteLoweredValue(value));

  // It would be nice to collect SSA values from the outer scope which are used
  // by the 'cond' or 'body' blocks without being mentioned by the signature,
  // and automatically include those in the grt.while signature. That would
  // avoid a lot of tedious boilerplate in MO for naming and returning SAA
  // values which are only read and never 'updated' by the 'body' block. However
  // MLIR does not allow 'partially isolated' regions.

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

    // Establish the block's arguments. The chain comes first.
    auto loweredChainArg =
        outputRegion.addArgument(mainChain.getType(), whileOp.getLoc());
    loweringState.setExecuteChain(loweredChainArg);

    auto contextArg = outputRegion.addArgument(
        loweringState.getExecuteMgpContext().getType(), whileOp.getLoc());

    // Then an arg for each of the original region args.
    // We'll use the loweredFreeValues since they have the lowered types
    // already. Skip the chain and the context.
    for (size_t i = 0; i < loweredFreeValues.size() - 2; ++i) {
      Value loweredFreeValue = loweredFreeValues[2 + i];
      Value arg = inputRegion.getArguments()[i];

      Value loweredArg =
          outputRegion.addArgument(loweredFreeValue.getType(), arg.getLoc());
      loweringState.emplaceExecuteLoweredValue(arg, loweredArg);
    }

    // Lower the block, including its terminating mo.yield.
    if (failed(lowerBlock(&inputRegion.getBlocks().front())))
      return failure();

    // Remap any uses of the top level mgp context in the while body with the
    // one in the region argument.
    outputRegion.getBlocks().front().walk([&](Operation *op) {
      for (size_t i = 0, e = op->getNumOperands(); i < e; ++i) {
        if (op->getOperand(i) == loweringState.getExecuteMgpContext()) {
          op->setOperand(i, contextArg);
        }
      }
    });
    return success();
  };

  // Lower the 'cond' block. We'll later need to make 2 adjustments:
  // 1) The mo.while 'cond' block will mo.yield only the condition result. That
  //    mo.yield will be lowered to a mef.output with the 'execute' chain and
  //    lowered condition results. However, the grt.while requires the 'cond'
  //    block to return a result for all signature arguments plus the condition,
  //    so we'll need to include those.
  // 2) The lowered condition will be a mgp.cpu.tensor<bool>, which we need to
  //    convert to an i1.
  if (failed(
          lower_block(whileOp.getCondRegion(), loweredWhileOp.getCondRegion())))
    return failure();

  // Lower the 'body' block.
  if (failed(
          lower_block(whileOp.getBodyRegion(), loweredWhileOp.getBodyRegion())))
    return failure();

  // Now repair the lowered 'cond' block's mef.output to include all the
  // additional variables between the output chain, context and, the condition,
  // and fix the condition to be an i1.
  {
    auto condOutputOp = cast<MEF::OutputOp>(
        loweredWhileOp.getCondRegion().front().getTerminator());
    assert(condOutputOp->getNumOperands() ==
           3); // chain, context, and condition
    Value loweredOutputChain = condOutputOp->getOperand(0);
    Value loweredCondTensor = condOutputOp->getOperand(2);

    SmallVector<Value> condOutputOpOperands;
    condOutputOpOperands.reserve(loweredWhileOp->getNumResults() + 1);

    // chain
    condOutputOpOperands.emplace_back(loweredOutputChain);

    // additional non-chain args
    llvm::append_range(
        condOutputOpOperands,
        llvm::drop_begin(loweredWhileOp.getCondRegion().getArguments()));

    // i1 condition
    opLoweringState.builder.setInsertionPoint(condOutputOp);
    Value condBuffer = opLoweringState.tensorExtract(loweredCondTensor).first;
    Value loweredInputChain = loweredWhileOp.getCondRegion().getArgument(0);
    Value condBool =
        opLoweringState.create<MGP::ToGRTBoolOp>(loweredInputChain, condBuffer);
    condOutputOpOperands.emplace_back(condBool);

    condOutputOp.getOperandsMutable().clear();
    condOutputOp.getOperandsMutable().append(condOutputOpOperands);
  }

  // Capture the lowered outputs.
  assert(loweredWhileOp.getNumResults() == whileOp.getNumResults() + 2);
  loweringState.setExecuteChain(loweredWhileOp.getResult(0));
  for (size_t i = 0, e = whileOp.getNumResults(); i < e; i++) {
    loweringState.emplaceExecuteLoweredValue(whileOp.getResult(i),
                                             loweredWhileOp.getResult(i + 2));
  }

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
      if (failed(lowerParamDeclare(declOp)))
        return failure();
      continue;
    }

    if (auto paramConstOp = dyn_cast<KGEN::ParamConstantOp>(op)) {
      if (failed(lowerParamConstant(paramConstOp)))
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
    // the 'cond' and 'body' blocks..
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
       llvm::zip(graph.getArguments(), loweringState.getModelOp()
                                           .getExecuteBlock()
                                           ->getArguments()
                                           .drop_front(2))) {
    auto hasParamsOr = loweringState.lowerArg(graphArg, exeArg);
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
  MGP::fixupIsolatedFromAbove(b, model);
  // Ok, we're done with the original mo.graph, so remove it.
  graph.erase();
  // We've fixed up the MGP model, so remove the old one.
  model.erase();
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
