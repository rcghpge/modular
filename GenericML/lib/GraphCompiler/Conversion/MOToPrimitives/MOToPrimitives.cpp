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
#include "GenericML/GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/MGPTransforms/FixupIsolatedFromAbove.h"
#include "GenericML/Support/HostDeviceSpecs.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "GraphRT/MEFDialect/MEFOps.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Host.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/ML/CommonAttributes.h"
#include "Support/STLExtras.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include <cassert>
#include <numeric>

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
  MOGraphLowering(MO::GraphOp graph, TypeConverter &typeConverter,
                  llvm::StringRef compiledFrameworkLabel,
                  size_t minimumCpuAlignment,
                  DeviceSpecCollectionAttr deviceSpecCollectionAttr)
      : loweringState(graph, typeConverter, compiledFrameworkLabel,
                      minimumCpuAlignment, deviceSpecCollectionAttr) {}

  /// Lower the mo.graph to mgp.model.
  LogicalResult run();

  // Emit code into the 'init' block to setup the compiled kernels context,
  // which will hold function pointers for the loaded kernels. This requires
  // an !mgp.context slot.
  LogicalResult setupCompiledKernelsContext(Location loc);

  // Emit code into the 'init' block to check the runtime environment
  // satisfies all the compile-time device specs. This requires an
  // !mgp.context slot to hold the calculated device spec map.
  LogicalResult setupCheckRuntime(Location loc);

  LogicalResult createModelOp();

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

  /// Lowers mo.if op.
  LogicalResult lowerIf(MO::IfOp ifOp);

  /// Lowers mo.while op.
  LogicalResult lowerWhile(MO::WhileOp whileOp);

private:
  /// Lowers a Block
  LogicalResult lowerBlock(mlir::Block *block);

  MGP::MOGraphLoweringState loweringState;
};

} // namespace

//===----------------------------------------------------------------------===//
// Creation of the init/execute invokables
//===----------------------------------------------------------------------===//

LogicalResult MOGraphLowering::setupCompiledKernelsContext(Location loc) {
  uint64_t slot = loweringState.getInitMgpContext().getNumSlots();
  auto createContextOp =
      loweringState.getInitOpBuilder().create<MGP::CpuKernelCreateContextOp>(
          loc,
          /*inChain=*/loweringState.getInitChain(),
          /*theContext=*/loweringState.getInitMgpContext(),
          /*slot=*/slot);
  loweringState.setCompiledKernelsContext(createContextOp);
  loweringState.getInitMgpContext().setNumSlots(slot + 1);
  loweringState.setInitChain(createContextOp.getOutChain());
  return success();
}

LogicalResult MOGraphLowering::setupCheckRuntime(Location loc) {
  if (loweringState.getDeviceSpecCollection()) {
    uint64_t slot = loweringState.getInitMgpContext().getNumSlots();
    auto checkRuntimeOp =
        loweringState.getInitOpBuilder().create<MGP::CheckRuntimeOp>(
            loc,
            /*inChain=*/loweringState.getInitChain(),
            /*theContext=*/loweringState.getInitMgpContext(),
            /*aSlot=*/slot,
            /*bDevices=*/loweringState.getDeviceSpecCollection());
    loweringState.getInitMgpContext().setNumSlots(slot + 1);
    loweringState.setInitChain(checkRuntimeOp.getOutChain());
  }
  return success();
}

// Create the mgp.model and the regions for the init/execute phases.
LogicalResult MOGraphLowering::createModelOp() {
  MO::GraphOp &graph = loweringState.getGraph();
  OpBuilder b(graph);

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

  // Setup the compiled kernels context.
  if (failed(setupCompiledKernelsContext(graph.getLoc())))
    return failure();

  // Check the runtime environment.
  if (failed(setupCheckRuntime(graph.getLoc())))
    return failure();

  // Initialize the execute state.
  Block *executeBlock = loweringState.getModelOp().getExecuteBlock();
  loweringState.getExecuteOpBuilder().setInsertionPointToEnd(executeBlock);

  // Extract chain and MGP context from arguments
  loweringState.setExecuteChain(executeBlock->getArgument(0));
  loweringState.setExecuteMgpContext(executeBlock->getArgument(1));

  return success();
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
  MGP::MOOpLoweringState opLoweringState(loweringState, toTensorOp);

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
  MGP::MOOpLoweringState opLoweringState(loweringState, toTensorOp);

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
static Value getRebindResMgpTensor(MO::RebindOp rebindOp,
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
      dynDimsOp = loweringState.getDynamicDims(loc, loweredInput);

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

  Value loweredOutput = getRebindResMgpTensor(rebindOp, loweringState);
  loweringState.emplaceExecuteLoweredValue(resTensor, loweredOutput);

  // Establish the lowered form of the rebind's result parameters.
  // TODO: Note this may end up extracting the dynamic dimensions from the
  // result tensor even though the same dimension values were just passed
  // as operands to a tensor_spec.create op.
  ArrayRef<KGEN::ParamDeclAttr> decls = rebindOp.getOutputParamDecls();
  if (!decls.empty()) {
    MGP::CpuTensorSpecDimsDynamicOp dynDims =
        loweringState.getDynamicDims(rebindOp.getLoc(), loweredOutput);

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
  MGP::MOOpLoweringState opLoweringState(loweringState, constOp);

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
  MGP::MOOpLoweringState opLoweringState(loweringState, reshapeOp);

  // Get the bufferized input.
  SmallVector<MGP::BufferizedTensor> bufferizedInputs =
      opLoweringState.getBufferizedInputs();
  Value buffer = bufferizedInputs[0].buffer;

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
  MGP::MOOpLoweringState opLoweringState(loweringState, reshapeOp);

  // Get the bufferized input.
  SmallVector<MGP::BufferizedTensor> bufferizedInputs =
      opLoweringState.getBufferizedInputs();
  Value buffer = bufferizedInputs[0].buffer;

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
  // TODO: Generalize from CPU
  MGP::MOOpLoweringState opLoweringState(loweringState, printOp);
  auto loweredTensor =
      loweringState.lookupExecuteLoweredValue(printOp.getInput());
  Value chain = opLoweringState
                    .create<MGP::DebugCpuTensorPrintOp>(
                        loweringState.getExecuteChain(), loweredTensor,
                        printOp.getLabelAttr())
                    .getResult();
  loweringState.setExecuteChain(chain);
  return success();
}

LogicalResult MOGraphLowering::lowerIf(MO::IfOp ifOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, ifOp);
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
  MGP::MOOpLoweringState opLoweringState(loweringState, whileOp);

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
};
} // namespace

void MOToPrimitivesPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  auto &ctx = getContext();

  TypeConverter converter;
  converter.addConversion([&ctx](MO::TensorType t) -> std::optional<Type> {
    // TODO: this may fail for some element types.
    return MGP::CpuTensorType::get(&ctx, MGP::createCpuTensorSpecType(t));
  });

  // The mojo kernels will already be composed into novel jit kernels at this
  // point so there is no lowering for the MojoRegisterOp which is just a helper
  // to link between MOGG and mojo.
  for (MOGG::MojoRegisterOp reg :
       llvm::make_early_inc_range(theModule.getOps<MOGG::MojoRegisterOp>()))
    reg.erase();

  // TEMPORARY: Until mo.graphs convey a DeviceSpecCollectionAttr we create a
  // default one representing the current CPU host. In this way any assumptions
  // baked into our code due to the use of -march=native can be checked
  // before the model is executed.
  ErrorOr<std::unique_ptr<DeviceSpecCollection>> defaultHostDeviceSpecsOr =
      GML::getHostCPUDeviceSpecs();
  if (defaultHostDeviceSpecsOr) {
    theModule->emitOpError(defaultHostDeviceSpecsOr.getError());
    return signalPassFailure();
  }
  DeviceSpecCollectionAttr defaultDeviceSpecCollectionAttr =
      fromRuntimeDeviceSpecs(&ctx, **defaultHostDeviceSpecsOr);

  // Convert each mo.graph into a mgp.model.
  for (auto graphOp :
       llvm::make_early_inc_range(theModule.getOps<MO::GraphOp>())) {
    if (failed(MOGraphLowering(graphOp, converter,
                               compiledFrameworkLabel.getValue(),
                               minimumCpuAlignment.getValue(),
                               defaultDeviceSpecCollectionAttr)
                   .run()))
      return signalPassFailure();
  }
}
