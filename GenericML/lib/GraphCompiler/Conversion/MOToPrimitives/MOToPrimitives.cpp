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
#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "GenericML/GraphCompiler/Conversion/DeviceBufferizationRegistry.h"
#include "GenericML/GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/MGPDialect/MGPTypes.h"
#include "GenericML/MGPTransforms/FixupIsolatedFromAbove.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "GraphRT/MEFDialect/MEFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/TransformUtils/SCCUtils.h"
#include "Support/AssertStream.h"
#include "Support/ML/CommonAttributes.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"

#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/StringMap.h"

#include <deque>

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_MOTOPRIMITIVES
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

//===----------------------------------------------------------------------===//
// MOGraphLowering implementation
//===----------------------------------------------------------------------===//

namespace {

struct IndexLowering {
  MGP::MOOpLoweringState &opLoweringState;

  // Arithmetic helpers
  auto constant(int64_t val) -> Value {
    return opLoweringState.create<mlir::index::ConstantOp>(val);
  }

  auto mul(Value a, Value b) -> Value {
    auto indexTy = opLoweringState.builder.getIndexType();
    return opLoweringState.create<mlir::index::MulOp>(indexTy, a, b);
  }

  auto add(Value a, Value b) -> Value {
    auto indexTy = opLoweringState.builder.getIndexType();
    return opLoweringState.create<mlir::index::AddOp>(indexTy, a, b);
  }

  auto div(Value a, Value b) -> Value {
    auto indexTy = opLoweringState.builder.getIndexType();
    return opLoweringState.create<mlir::index::DivSOp>(indexTy, a, b);
  }

  auto ifPositive(Value check, Value yes, Value no) -> Value {
    auto indexTy = opLoweringState.builder.getIndexType();
    auto cond = opLoweringState.create<mlir::index::CmpOp>(
        mlir::index::IndexCmpPredicate::SLT, check, constant(0));
    return opLoweringState.create<KGEN::POP::SelectOp>(indexTy, cond, no, yes);
  }

  auto ifZero(Value check, Value yes, Value no) -> Value {
    auto indexTy = opLoweringState.builder.getIndexType();
    auto cond = opLoweringState.create<mlir::index::CmpOp>(
        mlir::index::IndexCmpPredicate::EQ, check, constant(0));
    return opLoweringState.create<KGEN::POP::SelectOp>(indexTy, cond, yes, no);
  }
};

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
  LogicalResult setupTypeBuildersForAllDevices(MO::GraphOp graph);

  // Establish buffer builders for all the (virtual) devices needed by the
  // model. Each device will have its 'runtime' created in the 'init' block
  // and held in the !mgp.context.
  CompileDiagnosticOrSuccess setupBufferBuildersForAllDevices();

  LogicalResult createModelOp();

  /// Lowers a MO::ListType to a MGP::ListType.
  MGP::ListType lowerListType(MO::ListType listTy);

  /// Clones a trivial operations that does not need transformation. E.g.,
  /// `index.constant`
  LogicalResult cloneTrivialOp(Operation *trivialOp);

  /// Lowers the mo.output op to the mef.output for the final results.
  LogicalResult lowerOutput(MO::OutputOp output);

  /// Lowers the mo.yield to the mef.output for the if block results.
  LogicalResult lowerYield(MO::YieldOp output);

  /// Lowers the mo.while.conddition to the mef.output for the if block results.
  LogicalResult lowerWhileCondition(MO::WhileConditionOp yield);

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
  ErrorOr<ElementsAttr> lowerElementsAttr(ElementsAttr elementsAttr) {
    return loweringState.lowerElementsAttr(elementsAttr);
  }

  /// Lower a parameter declaration op.
  LogicalResult lowerParamDeclare(Region *parentRegion,
                                  KGEN::ParamDeclareOp declOp);

  /// Lower a MOSH::ParamToValueOp op.
  LogicalResult lowerParamToValue(Region *parentRegion,
                                  MOSH::ParamToValueOp paramToValueOp);

  /// Lower a MOSH::ParamFromValueOp op.
  LogicalResult lowerParamFromValue(Region *parentRegion,
                                    MOSH::ParamFromValueOp paramFromValueOp);

  /// Lower an index -> scalar-like tensor cast op.
  LogicalResult lowerIndexToTensorOp(MO::IndexToTensorOp toTensorOp);

  /// Lower a shape -> shape-like tensor cast op.
  LogicalResult lowerShapeToTensorOp(MO::Shape::ToTensorOp toTensorOp);

  /// Lower a shape-like -> shape tensor cast op.
  LogicalResult lowerShapeFromTensorOp(MO::Shape::FromTensorOp fromTensorOp);

  /// Lower a tensor rebind.
  LogicalResult lowerRebind(MO::RebindOp rebindOp);

  /// Lowers mo.constant op.
  LogicalResult lowerConstant(MO::ConstantOp constOp);

  /// Lowers mo.constant.external op.
  LogicalResult lowerConstantExternal(MO::ConstantExternalOp constExternalOp);

  /// Lowers mo.constant.scalar op.
  LogicalResult lowerConstantScalar(MO::ConstantScalarOp constOp);

  /// Lowers mo.debug.print op.
  LogicalResult lowerDebugPrint(MO::DebugPrintOp printOp);

  /// Lowers mo.debug.tensor.print op.
  LogicalResult lowerDebugTensorPrint(MO::DebugTensorPrintOp printOp);

  /// Lowers mo.debug.tensor.unsafe.print op.
  LogicalResult
  lowerDebugTensorUnsafePrint(MO::DebugTensorUnsafePrintOp unsafePrintOp);

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

  /// Lowers mo.mutable.store op
  LogicalResult lowerMutableStore(MO::MutableStoreOp);

  /// Lowers mo.mutable.load op
  LogicalResult lowerMutableLoad(MO::MutableLoadOp);

  /// Lowers mo.buffer.create op
  LogicalResult lowerBufferCreate(MO::BufferCreateOp);

  /// Lowers mo.buffer.transfer op
  LogicalResult lowerBufferTransfer(MO::BufferTransferOp bufferTransferOp);

  /// Lowers mo.call op
  LogicalResult lowerCallGraph(MO::CallOp callOp);

private:
  /// Lowers a Block
  LogicalResult lowerBlock(mlir::Block *block);

  MGP::MOGraphLoweringState loweringState;
};

} // namespace

static SmallVector<DeviceRefAttr> getAllDevicesUsedInGraph(MO::GraphOp graph) {

  // This is not a list of devices that are actually available on a given
  // machine, but rather devices that are used in the graph. Each device is
  // in the list exactly once.
  llvm::DenseSet<DeviceRefAttr> devices;

  // Always add the host device, it may not be explicitly in the graph.
  DeviceRefAttr hostDevice = DeviceRefAttr::getDefault(graph->getContext());
  devices.insert(hostDevice);

  mlir::AttrTypeWalker walker;
  walker.addWalk([&](DeviceRefAttr attr) { devices.insert(attr); });
  graph->walk([&](Operation *op) {
    walker.walk(op->getAttrDictionary());
    for (Type type : op->getOperands().getTypes())
      walker.walk(type);
    for (Type type : op->getResultTypes())
      walker.walk(type);
    for (Region &region : op->getRegions()) {
      for (Type type : region.getArgumentTypes())
        walker.walk(type);
    }
  });

  // Sort the results so that the host device is always listed first. The sort
  // other devices in alphabetical order by their string representation. This
  // allows the device attributes on the graph to have a consistent sorting.
  SmallVector<DeviceRefAttr> sorted = llvm::to_vector(devices);
  llvm::sort(sorted.begin(), sorted.end(),
             [hostDevice](DeviceRefAttr left, DeviceRefAttr right) {
               if (left == hostDevice)
                 return true;
               if (right == hostDevice)
                 return false;
               return left.toString() < right.toString();
             });
  return sorted;
}

LogicalResult
MOGraphLowering::setupTypeBuildersForAllDevices(MO::GraphOp graph) {
  for (DeviceRefAttr device : getAllDevicesUsedInGraph(graph)) {
    if (failed(MO::makeTypeBuilder(device, loweringState)))
      return failure();
  }
  return success();
}

CompileDiagnosticOrSuccess MOGraphLowering::setupBufferBuildersForAllDevices() {
  DeviceRefAttr hostDevice =
      DeviceRefAttr::getDefault(loweringState.getModelOp()->getContext());

  // Make the host buffer builder first.
  if (auto err = MO::makeBufferBuilder(hostDevice, hostDevice, loweringState);
      err.isError())
    return err;

  // Add the buffer builders for other supported devices.
  for (auto device : getAllDevicesUsedInGraph(loweringState.getGraph())) {
    if (device != hostDevice) {
      if (auto err = MO::makeBufferBuilder(hostDevice, device, loweringState);
          err.isError())
        return err;
    }
  }

  return success();
}

// Create the mgp.model and the regions for the init/execute phases.
LogicalResult MOGraphLowering::createModelOp() {
  MO::GraphOp &graph = loweringState.getGraph();
  OpBuilder b(graph);

  // Establish type builders for all the devices.
  // CAUTION: Must be done before any types are lowered, eg the next
  // paragraph.
  if (failed(setupTypeBuildersForAllDevices(graph)))
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

  // Get all the devices
  ErrorOrSuccess err(success());
  auto devices = llvm::map_to_vector(getAllDevicesUsedInGraph(graph),
                                     [&err](DeviceRefAttr attr) {
                                       auto refOr = toRuntimeDeviceRef(attr);
                                       if (refOr.isError()) {
                                         err = refOr.takeError();
                                         return DeviceRef();
                                       } else {
                                         return refOr.takeValue();
                                       }
                                     });
  if (err.isError())
    return err;

  // Go through Input Output types and their device and add to extraAttrs.
  auto ctx = graph->getContext();
  SmallDenseMap<DeviceRefAttr, IntegerAttr> deviceIndexMap;
  auto idxType = IntegerType::get(ctx, 32, IntegerType::Unsigned);
  for (const auto &[idx, deviceRef] :
       llvm::enumerate(getAllDevicesUsedInGraph(graph)))
    deviceIndexMap[deviceRef] = IntegerAttr::get(idxType, idx);
  auto populateDeviceIndicesExtraAttr = [&](const char *label,
                                            SmallVector<Type> types) {
    llvm::SmallVector<Attribute> deviceIndices;
    for (const auto &type : types) {
      if (auto hasDeviceType = dyn_cast<MGP::MGPHasDeviceTypeInterface>(type))
        deviceIndices.push_back(deviceIndexMap[hasDeviceType.getDevice()]);
    }
    // Set the attribute.
    if (!deviceIndices.empty()) {
      ArrayAttr deviceIndicesAttr = ArrayAttr::get(ctx, deviceIndices);
      extraAttrs.emplace_back(b.getStringAttr(label), deviceIndicesAttr);
    }
  };
  // Set the `argument_device_indices` attribute.
  populateDeviceIndicesExtraAttr(kMgpModelArgumentDeviceIndices, inputTypes);
  // Set the `result_device_indices` attribute.
  populateDeviceIndicesExtraAttr(kMgpModelResultDeviceIndices, outputTypes);

  // Create the mgp.model and the regions for the init/execute phases.
  b.setInsertionPoint(graph);
  auto modelOp = MGP::ModelOp::create(
      b, graph.getLoc(),
      /*name=*/graph.getNameAttr(),
      /*frameworkName=*/loweringState.getCompiledFrameworkLabel(),
      /*devices=*/devices,
      /*inputTypes=*/inputTypes,
      /*outputTypes=*/outputTypes,
      /*inputArgLocs=*/inputLocs,
      /*isSubgraph=*/graph.getIsSubgraph(),
      /*extraAttrs=*/extraAttrs);

  loweringState.setModelOp(modelOp);

  // Initialize the init state.
  loweringState.getInitOpBuilder().setInsertionPointToEnd(
      loweringState.getModelOp().getInitBlock());
  // Get the chain and context values for the body.
  // This field tracks the current output chain as we lower into the init
  // body.
  loweringState.setInitChain(GRT::ChainCreateOp::create(
      loweringState.getInitOpBuilder(), graph.getLoc()));

  // Create the !mgp.context. Its numSlots attribute will be incremented
  // as needed.
  loweringState.setInitMgpContext(MGP::ContextCreateOp::create(
      loweringState.getInitOpBuilder(), graph.getLoc(), /*numSlots=*/0));

  // Initialize the execute state.
  Block *executeBlock = loweringState.getModelOp().getExecuteBlock();
  loweringState.getExecuteOpBuilder().setInsertionPointToEnd(executeBlock);

  // Extract chain and MGP context from arguments
  loweringState.setRootExecuteChain(executeBlock->getArgument(0));
  loweringState.setCurrentExecuteChain(executeBlock->getArgument(0));
  loweringState.setExecuteMgpContext(executeBlock->getArgument(1));

  for (auto [devRef, devArg] :
       llvm::zip(modelOp.getDevices(), modelOp.getInitDevices())) {
    loweringState.setInitDevice(
        dyn_cast<DeviceRefAttr>(devRef),
        dyn_cast<TypedValue<MGP::DeviceContextType>>(devArg));
  }

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
      MGP::createTensorSpecType(outputTensorTy);
  DeviceRefAttr tensorDeviceRef = MO::getDeviceOrDefault(outputTensorTy);
  MGP::TypeBuilder *typeBuilder = loweringState.getTypeBuilder(tensorDeviceRef);
  auto mgpOutputTensorTy = typeBuilder->tensorType(ctx, outputTensorSpecTy);
  return MGP::ListType::get(ctx, mgpOutputTensorTy);
}

LogicalResult MOGraphLowering::cloneTrivialOp(Operation *trivialOp) {
  Operation *cloned = loweringState.getExecuteOpBuilder().clone(*trivialOp);
  for (auto [from, to] :
       llvm::zip_equal(trivialOp->getResults(), cloned->getResults()))
    loweringState.emplaceExecuteLoweredValue(from, to);

  return success();
}

LogicalResult MOGraphLowering::lowerOutput(MO::OutputOp output) {
  // Init phase lowering.

  // Return the final chain.
  SmallVector<Value> outputs;
  outputs.push_back(loweringState.getInitChain());
  outputs.push_back(loweringState.getInitMgpContext());
  outputs.push_back(loweringState.getModelOp().getInitWeightsRegistry());
  for (auto dev : loweringState.getModelOp().getDevices())
    outputs.push_back(
        loweringState.getInitDevice(dyn_cast<DeviceRefAttr>(dev)));
  MEF::OutputOp::create(loweringState.getInitOpBuilder(), output.getLoc(),
                        outputs);

  // Execute phase lowering.

  // Figure out what values to return.  We always return a chain.
  SmallVector<Value> operands;
  operands.push_back(loweringState.getCurrentExecuteChain());

  for (Value operand : output.getOperands()) {
    Value loweredValue = loweringState.lookupExecuteLoweredValue(operand);

    // In emit-mojo path, mo.opaque types need to be converted to mgp.opaque
    // for graph outputs. The model's result types expect mgp.opaque.
    // ToMGPFromOpaqueOp is essentially a no-op/rebind that converts the type.
    // This op should be folded away when the input type gets converted to
    // mgp.opaque later in the pipeline (e.g., after OutlineKernels).
    if (auto moOpaqueType = dyn_cast<MO::OpaqueType>(loweredValue.getType())) {
      // Preserve type parameters when converting mo.opaque to mgp.opaque.
      Type mgpOpaqueType =
          MGP::OpaqueType::get(output->getContext(), moOpaqueType.getSymbol(),
                               moOpaqueType.getOptionalParameters());
      loweredValue = MOGG::ToMGPFromOpaqueOp::create(
          loweringState.getExecuteOpBuilder(), operand.getLoc(),
          TypeRange{mgpOpaqueType}, loweredValue);
    }

    operands.push_back(loweredValue);
  }

  // Finally, we can create the mef.output op.
  MEF::OutputOp::create(loweringState.getExecuteOpBuilder(), output.getLoc(),
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

  for (TypedAttr param : yield.getParameters()) {
    auto valueOr = loweringState.lowerExpression(yield->getParentRegion(),
                                                 yield.getLoc(), param);
    if (valueOr.isError())
      return yield->emitOpError(valueOr.getError());
    operands.push_back(*valueOr);
  }

  // Finally, we can create the mef.output op.
  MEF::OutputOp::create(loweringState.getExecuteOpBuilder(), yield.getLoc(),
                        operands);
  return success();
}

LogicalResult
MOGraphLowering::lowerWhileCondition(MO::WhileConditionOp condition) {
  // Execute phase lowering.

  // Figure out what values to return. We always implicitly return the current
  // 'execute' block chain.
  SmallVector<Value> operands;
  operands.push_back(loweringState.getCurrentExecuteChain());

  for (Value operand : condition.getOperands())
    operands.push_back(loweringState.lookupExecuteLoweredValue(operand));

  // Finally, we can create the mef.output op.
  MEF::OutputOp::create(loweringState.getExecuteOpBuilder(), condition.getLoc(),
                        operands);
  return success();
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
MOGraphLowering::lowerParamFromValue(Region *parentRegion,
                                     MOSH::ParamFromValueOp paramFromValueOp) {
  auto input =
      loweringState.lookupExecuteLoweredValue(paramFromValueOp.getValue());

  if (auto err = loweringState.lowerParamDecl(
          parentRegion, paramFromValueOp.getParamDecl(), input))
    return paramFromValueOp->emitOpError(err.getError());

  return success();
}

LogicalResult
MOGraphLowering::lowerIndexToTensorOp(MO::IndexToTensorOp toTensorOp) {
  ASSERT_STREAM(MO::isOnHost(toTensorOp),
                << "mo.index.to_tensor is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, toTensorOp);

  // Set up output buffer.
  FailureOr<SmallVector<MGP::MOOutputStorage>> loweredOutputStorageOr =
      opLoweringState.createOutputStorage(loweringState.getExecuteChain());
  if (failed(loweredOutputStorageOr))
    return failure();
  auto &outputStorage = (*loweredOutputStorageOr)[0];
  [[maybe_unused]] auto bufType =
      dyn_cast<MGP::DeviceBufferType>(outputStorage.buffer.getType());
  ASSERT_STREAM(bufType && bufType.getDevice().getLabel() == kCPULabel,
                << "expecting !mgp.buffer on host but it is on "
                << bufType.getDevice());

  // Fill the buffer with the index value.
  Value loweredIndexVal =
      loweringState.lookupExecuteLoweredValue(toTensorOp.getInput());

  // The input to mo.index.to_tensor is now an si64.
  // Cast to index to match the expectation of mgp.set_with_index.
  auto &builder = loweringState.getExecuteOpBuilder();
  loweredIndexVal = mlir::index::CastSOp::create(
      builder, toTensorOp.getLoc(), builder.getIndexType(), loweredIndexVal);

  auto outChain = opLoweringState.bufferSetWithIndex(
      outputStorage.buffer, {loweredIndexVal},
      toTensorOp.getResult().getType());

  // Set up output tensor.
  opLoweringState.createOutputs(outChain, *loweredOutputStorageOr);
  return success();
}

LogicalResult
MOGraphLowering::lowerShapeToTensorOp(MO::Shape::ToTensorOp toTensorOp) {
  ASSERT_STREAM(MO::isOnHost(toTensorOp),
                << "mo.shape.to_tensor is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, toTensorOp);

  // Set up output buffer.
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
  SmallVector<Value> values(loweredTensorSpec.getType().getRank());
  for (size_t i = 0; i < values.size(); ++i)
    values[i] =
        opLoweringState.create<MGP::TensorSpecGetDimOp>(loweredTensorSpec, i);

  auto outChain = opLoweringState.bufferSetWithIndex(
      outputStorage.buffer, values, toTensorOp.getResult().getType());

  // Set up output tensor.
  opLoweringState.createOutputs(outChain, *loweredOutputsStorageOr);
  return success();
}

LogicalResult
MOGraphLowering::lowerShapeFromTensorOp(MO::Shape::FromTensorOp fromTensorOp) {
  ASSERT_STREAM(MO::isOnHost(fromTensorOp),
                << "mo.shape.to_tensor is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, fromTensorOp);

  MO::TensorType tensorType = fromTensorOp.getInput().getType();
  std::optional<int64_t> rank = tensorType.getNumElements();

  ASSERT_STREAM(rank, << "unable to determine rank of result shape");

  MGP::BufferizedTensor inputBuffer =
      opLoweringState.getBufferizedInput(fromTensorOp.getInput());

  Value inChain =
      opLoweringState.getGuardChain(loweringState.getExecuteChain());

  auto sizeInBytes = opLoweringState.create<mlir::index::ConstantOp>(
      tensorType.getDtype().getSizeInBytes());

  SmallVector<Value> indices;
  for (auto dim : llvm::seq(*rank)) {
    IndexLowering indexLowering{opLoweringState};

    Value offset = indexLowering.constant(dim);
    Value offsetInBytes = indexLowering.mul(offset, sizeInBytes);

    // Slice the input tensor to the correct size.
    Value sliced = opLoweringState.bufferSlice(
        inputBuffer.buffer, offsetInBytes, sizeInBytes, tensorType);

    // Convert the tensor to an index value.
    Value indexValue =
        opLoweringState.create<MGP::BufferToIndexOp>(inChain, sliced);

    indices.push_back(indexValue);
  }

  SmallVector<int64_t> shape(*rank, mlir::ShapedType::kDynamic);
  auto tensorSpecType =
      MGP::TensorSpecType::get(shape, tensorType.getElementType());

  auto result =
      opLoweringState.create<MGP::TensorSpecCreateOp>(tensorSpecType, indices);

  loweringState.emplaceExecuteLoweredValue(fromTensorOp.getResult(), result);

  return success();
}

/// Returns the lowered tensor that the rebind's result should map to.
static Value getRebindResTensor(MO::RebindOp rebindOp,
                                MGP::MOGraphLoweringState &loweringState) {
  MGP::MOOpLoweringState opLoweringState(loweringState, rebindOp);

  MO::TensorValue inputTensor = rebindOp.getInput();
  MO::TensorType inputTy = inputTensor.getType();
  MO::TensorType resTy = rebindOp.getResult().getType();

  // If the input and output specs are the same, we can just map to the
  // already lowered tensor.
  auto outSpecTy = MGP::createTensorSpecType(resTy);
  auto inSpecTy = MGP::createTensorSpecType(inputTy);
  Value loweredInput = loweringState.lookupExecuteLoweredValue(inputTensor);
  if (outSpecTy == inSpecTy)
    return loweredInput;

  // We will collect the runtime dimensions for the result spec.
  SmallVector<Value> runtimeDims;
  Location loc = opLoweringState.loc;
  for (auto [idx, inDim, resDim] :
       llvm::enumerate(inputTy.getDims(), resTy.getDims())) {

    // If the result dimension is static, nothing to do.
    if (isa<IntegerAttr>(resDim))
      continue;

    // If the input dimension is static, and the result is dynamic, we create
    // a constant runtime dimension.
    if (auto intAttr = dyn_cast<IntegerAttr>(inDim)) {
      auto constOp =
          opLoweringState.create<mlir::index::ConstantOp>(intAttr.getSInt());
      runtimeDims.emplace_back(constOp.getResult());
      continue;
    }

    // At this point we are dealing with dynamic input and result dimensions.
    // We extract the dynamic dimensions only once.
    auto getDimOp = loweringState.getDim(loc, loweredInput, idx,
                                         getDeviceOrDefault(inputTy));

    // Forward the appropriate dynamic dimension from the input.
    runtimeDims.emplace_back(getDimOp);
  }

  // Create the output tensor.
  Value buffer = opLoweringState.tensorExtract(loweredInput, inputTy).first;
  TypedValue<MGP::TensorSpecType> tensorSpec =
      opLoweringState.create<MGP::TensorSpecCreateOp>(outSpecTy, runtimeDims);
  return opLoweringState.tensorCreate(loweringState.getExecuteChain(), buffer,
                                      tensorSpec, inputTy);
}

LogicalResult MOGraphLowering::lowerRebind(MO::RebindOp rebindOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, rebindOp);

  MO::TensorValue inputTensor = rebindOp.getInput();
  MO::TensorValue resTensor = rebindOp.getResult();

  MO::TensorType inputTy = inputTensor.getType();
  MO::TensorType resTy = resTensor.getType();
  if (inputTy.hasRank() != resTy.hasRank())
    return rebindOp.emitOpError(
        "rebind lowering doesn't support unknown shapes");

  // The input chain is propagated to also guard the users of the reshaped
  // buffer.
  Value inChain =
      opLoweringState.getGuardChain(loweringState.getExecuteChain());

  Value loweredOutput = getRebindResTensor(rebindOp, loweringState);
  loweringState.emplaceExecuteLoweredBufferValue(resTensor, loweredOutput,
                                                 inChain);

  // Establish the lowered form of the rebind's result parameters.
  // TODO: Note this may end up extracting the dynamic dimensions from the
  // result tensor even though the same dimension values were just passed
  // as operands to a tensor_spec.create op.
  ArrayRef<KGEN::ParamDeclAttr> decls = rebindOp.getOutputParamDecls();
  if (!decls.empty()) {
    MOSH::ShapeAttr resShape = resTy.getTypedShapeAttr();
    for (KGEN::ParamDeclAttr decl : decls) {
      int64_t idx = findDeclIdx(resShape, decl);
      if (idx == -1) {
        return rebindOp.emitOpError(
            "rebind declares parameter not appearing in its result");
      }

      auto getDimOp = loweringState.getDim(rebindOp.getLoc(), loweredOutput,
                                           idx, getDeviceOrDefault(resTy));

      auto asDimension = loweringState.indexToDimension(
          loweringState.getExecuteOpBuilder(), rebindOp.getLoc(), getDimOp);

      if (auto err = loweringState.lowerParamDecl(rebindOp->getParentRegion(),
                                                  decl, asDimension))
        return rebindOp->emitOpError(err.getError());
    }
  }

  return success();
}

/// Takes a constant op's result buffer on host and:
/// - allocates device memory,
/// - copies from host to device, then,
/// - creates the output tensor.
/// If the target device is host, then just creates the output tensor.
static LogicalResult
constHostBufferToTarget(MGP::MOOpLoweringState &opLoweringState,
                        MGP::MOGraphLoweringState &graphLoweringState,
                        Value hostBuffer, TypedValue<MEF::ChainType> chain,
                        TypedValue<MO::TensorType> opResult,
                        std::optional<uint64_t> maybeAlignment = std::nullopt) {
  MO::TensorType moTensorType = opResult.getType();
  FailureOr<TypedValue<MGP::TensorSpecType>> tensorSpecOr =
      opLoweringState.createTensorSpec(moTensorType);
  if (failed(tensorSpecOr))
    return failure();
  TypedValue<MGP::TensorSpecType> tensorSpec = *tensorSpecOr;

  DeviceRefAttr device = MO::getDeviceOrDefault(moTensorType);
  DeviceRefAttr hostDevice = graphLoweringState.getHostDevice();

  // If targetDevice is on host, then we just create output tensor and return.
  // Otherwise, we allocate a device buffer and explicitly transfer HtoD.
  if (device == hostDevice) {
    opLoweringState.createOutputTensor(opResult, chain, hostBuffer, tensorSpec);
    return success();
  }

  // Default alignment to the preferred alignment for device if not passed.
  uint64_t alignment = maybeAlignment.value_or(
      AsyncRT::getPreferredAlignment(device.getLabel().str()));

  uint64_t sizeInBytes = *tensorSpec.getType().getSizeInBytes();
  Value bufferSize =
      opLoweringState.create<mlir::index::ConstantOp>(sizeInBytes);
  Value deviceBuffer =
      opLoweringState.bufferAlloc(chain, bufferSize, alignment, moTensorType);
  // Use bufferTransferFromHost to also take care of the stream sync.
  // Note that this is needlessly conservative because the constant external's
  // lifetime is controlled outside the graph.
  Value outChain = opLoweringState.bufferTransferFromHost(
      chain, cast<TypedValue<MGP::DeviceBufferType>>(hostBuffer), deviceBuffer,
      moTensorType);

  opLoweringState.createOutputTensor(opResult, outChain, deviceBuffer,
                                     tensorSpec);
  return success();
}

LogicalResult MOGraphLowering::lowerConstant(MO::ConstantOp constOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, constOp);

  ErrorOr<ElementsAttr> elementsOr = lowerElementsAttr(constOp.getValue());
  if (elementsOr)
    return constOp->emitOpError(elementsOr.getError());

  TypedValue<MO::TensorType> opResult = constOp.getResult();
  TypedValue<MEF::ChainType> chain = loweringState.getExecuteChain();

  // Create the mgp.buffer.constant op.
  Value hostBuffer = opLoweringState.hostBufferConstant(chain, *elementsOr,
                                                        opResult.getType());
  return constHostBufferToTarget(opLoweringState, loweringState, hostBuffer,
                                 chain, opResult);
}

LogicalResult
MOGraphLowering::lowerConstantExternal(MO::ConstantExternalOp constExternalOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, constExternalOp);

  TypedValue<MEF::ChainType> chain = loweringState.getExecuteChain();
  TypedValue<MO::TensorType> opResult = constExternalOp.getResult();
  uint64_t sizeInBytes = opResult.getType().getDtype().getSizeInBytes() *
                         (*opResult.getType().getNumElements());
  uint64_t alignment = constExternalOp.getAlign();
  bool hasAlias = constExternalOp.getHasAlias();
  StringRef name = constExternalOp.getName();
  bool isPlaceholder = constExternalOp.getIsPlaceholder();

  // Lower mo.constant.external depending on its device attribute.
  DeviceRefAttr opDevice = constExternalOp.getExecutionDevice();
  // If not present, use the default host device.
  opDevice =
      opDevice ? opDevice : DeviceRefAttr::getDefault(chain.getContext());
  if (opDevice.getLabel() == kCPULabel) {
    // If the external constant is on CPU, lower similarly to mo.constant.

    // Create the mgp.buffer.constant.external op.
    Value hostBuffer = opLoweringState.hostBufferConstantExternal(
        chain, name, sizeInBytes, alignment, hasAlias, isPlaceholder);

    return constHostBufferToTarget(opLoweringState, loweringState, hostBuffer,
                                   chain, opResult, alignment);
  }

  // If the external constant is already on device, it should stay on device.
  DeviceRefAttr resultDevice = MO::getDeviceOrDefault(opResult.getType());
  if (resultDevice != opDevice)
    return constExternalOp->emitOpError(
        Twine("expected op device ") + opDevice.getLabel() +
        " and result device " + resultDevice.getLabel() + " to match");

  // Create the MGP device buffer without copying or allocations.
  Value deviceBuffer = opLoweringState.deviceBufferConstantExternal(
      opDevice, chain, name, sizeInBytes, alignment, hasAlias, isPlaceholder);

  MO::TensorType moTensorType = opResult.getType();
  FailureOr<TypedValue<MGP::TensorSpecType>> tensorSpecOr =
      opLoweringState.createTensorSpec(moTensorType);
  if (failed(tensorSpecOr))
    return failure();
  TypedValue<MGP::TensorSpecType> tensorSpec = *tensorSpecOr;

  opLoweringState.createOutputTensor(opResult, chain, deviceBuffer, tensorSpec);

  return success();
}

LogicalResult
MOGraphLowering::lowerConstantScalar(MO::ConstantScalarOp constOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, constOp);

  // currently only si32, si64, and i1 constants are supported
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
    } else if (intAttr.getType().isSignedInteger(64)) {
      auto mipConstOp =
          opLoweringState.create<mlir::index::ConstantOp>(intAttr.getSInt());
      loweringState.emplaceExecuteLoweredValue(constOp, mipConstOp);
    }
  } else {
    return constOp->emitOpError(
        "Only si32, si64 and i1 constant scalar are currently lowerable");
  }

  return success();
}

static LogicalResult lowerReshapeImpl(MGP::MOOpLoweringState &opLoweringState,
                                      Value input, Value output, Value chain,
                                      MO::TensorType outputType) {
  // Get the bufferized input.
  MGP::BufferizedTensor bufferizedInput =
      opLoweringState.getBufferizedInput(input);
  Value buffer = bufferizedInput.buffer;

  // Create a tensor result with spec matching the result type.
  FailureOr<TypedValue<MGP::TensorSpecType>> tensorSpecOr =
      opLoweringState.createTensorSpec(outputType);
  if (failed(tensorSpecOr))
    return failure();

  // The input chain is propagated to also guard the users of the reshaped
  // buffer.
  Value inChain = opLoweringState.getGuardChain(chain);

  opLoweringState.createOutputTensor(output, inChain, buffer, *tensorSpecOr);

  return success();
}

LogicalResult MOGraphLowering::lowerDebugPrint(MO::DebugPrintOp printOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, printOp);

  OpBuilder b(printOp);

  DeviceRefAttr targetDevice = printOp.getExecutionDevice();
  DeviceRefAttr hostDevice = loweringState.getHostDevice();

  auto chain = opLoweringState.getGuardChain(
      loweringState.lookupExecuteLoweredValue(printOp.getInChain()));

  auto outChain =
      cast<TypedValue<MEF::ChainType>>(opLoweringState.hostDebugPrint(
          chain, printOp.getValueAttr(), printOp.getLabelAttr()));

  chain = opLoweringState.sync(chain);

  if (targetDevice != hostDevice)
    chain = opLoweringState.sync(chain);

  loweringState.setCurrentExecuteChain(
      cast<TypedValue<MEF::ChainType>>(outChain));

  loweringState.emplaceExecuteLoweredValue(printOp.getOutChain(), outChain);
  return success();
}

LogicalResult
MOGraphLowering::lowerDebugTensorPrint(MO::DebugTensorPrintOp printOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, printOp);

  OpBuilder b(printOp);
  auto moTensorType = printOp.getInput().getType();
  DeviceRefAttr targetDevice = MO::getDeviceOrDefault(moTensorType);
  DeviceRefAttr hostDevice = loweringState.getHostDevice();

  // Get the bufferized input.
  MGP::BufferizedTensor bufferizedInput =
      opLoweringState.getBufferizedInput(printOp.getInput());

  auto &buffer = bufferizedInput.buffer;
  auto &tensorSpec = bufferizedInput.tensorSpec;
  auto chain = opLoweringState.getGuardChain(
      loweringState.lookupExecuteLoweredValue(printOp.getInChain()));

  if (printOp.getLabelAttr().empty() || targetDevice == hostDevice) {
    auto outChain = opLoweringState.debugTensorPrint(
        chain, buffer, tensorSpec, printOp.getLabelAttr(),
        printOp.getInput().getType());
    loweringState.setCurrentExecuteChain(
        cast<TypedValue<MEF::ChainType>>(outChain));

    loweringState.emplaceExecuteLoweredValue(printOp.getOutChain(), outChain);
    return success();
  }

  // If the device we are trying to print from is not the host, insert an
  // explicit DtoH + sync for the buffer before printing using the host.
  // This is a safe assumption since the is what happens in the device
  // version of the primitive already
  auto maybeTensorSizeInBytes = tensorSpec.getType().getSizeInBytes();
  Value bufferSize;
  if (maybeTensorSizeInBytes.has_value())
    bufferSize = opLoweringState.create<mlir::index::ConstantOp>(
        *maybeTensorSizeInBytes);
  else
    bufferSize = opLoweringState.bufferSizeBytes(tensorSpec, moTensorType);
  Value hostBuffer = opLoweringState.hostBufferAlloc(
      cast<TypedValue<MEF::ChainType>>(chain), bufferSize,
      AsyncRT::getPreferredAlignment(hostDevice.getLabel().str()));
  auto transferChain = opLoweringState.bufferTransferToHost(
      chain, buffer, cast<TypedValue<MGP::DeviceBufferType>>(hostBuffer),
      moTensorType);
  auto outChain = opLoweringState.hostDebugTensorPrint(
      transferChain, hostBuffer, tensorSpec, printOp.getLabelAttr());
  loweringState.setCurrentExecuteChain(
      cast<TypedValue<MEF::ChainType>>(outChain));
  loweringState.emplaceExecuteLoweredValue(printOp.getOutChain(), outChain);
  return success();
}

LogicalResult MOGraphLowering::lowerDebugTensorUnsafePrint(
    MO::DebugTensorUnsafePrintOp unsafePrintOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, unsafePrintOp);

  OpBuilder b(unsafePrintOp);
  auto moTensorType = unsafePrintOp.getInput().getType();
  DeviceRefAttr targetDevice = MO::getDeviceOrDefault(moTensorType);
  DeviceRefAttr hostDevice = loweringState.getHostDevice();

  // Get the bufferized input.
  MGP::BufferizedTensor bufferizedInput =
      opLoweringState.getBufferizedInput(unsafePrintOp.getInput());

  auto &buffer = bufferizedInput.buffer;
  auto &tensorSpec = bufferizedInput.tensorSpec;
  auto chain =
      opLoweringState.getGuardChain(loweringState.getCurrentExecuteChain());

  if (unsafePrintOp.getLabelAttr().empty() || targetDevice == hostDevice) {
    auto outChain = opLoweringState.debugTensorPrint(
        chain, buffer, tensorSpec, unsafePrintOp.getLabelAttr(),
        unsafePrintOp.getInput().getType());
    loweringState.setCurrentExecuteChain(
        cast<TypedValue<MEF::ChainType>>(outChain));
    return success();
  }

  // If the device we are trying to print from is not the host, insert an
  // explicit DtoH + sync for the buffer before printing using the host.
  // This is a safe assumption since the is what happens in the device
  // version of the primitive already
  uint64_t sizeInBytes = *tensorSpec.getType().getSizeInBytes();
  Value bufferSize =
      opLoweringState.create<mlir::index::ConstantOp>(sizeInBytes);
  Value hostBuffer = opLoweringState.hostBufferAlloc(
      cast<TypedValue<MEF::ChainType>>(chain), bufferSize,
      AsyncRT::getPreferredAlignment(hostDevice.getLabel().str()));
  auto transferChain = opLoweringState.bufferTransferToHost(
      chain, buffer, cast<TypedValue<MGP::DeviceBufferType>>(hostBuffer),
      moTensorType);
  auto outChain = opLoweringState.hostDebugTensorPrint(
      transferChain, hostBuffer, tensorSpec, unsafePrintOp.getLabelAttr());
  loweringState.setCurrentExecuteChain(
      cast<TypedValue<MEF::ChainType>>(outChain));
  return success();
}

// Common implementation for buffer transfers between devices
static Value lowerTransferImpl(MGP::MOOpLoweringState &opLoweringState,
                               DeviceRefAttr hostDevice,
                               MGP::BufferizedTensor &srcBuffer,
                               Value dstBuffer, MO::TensorType srcType,
                               MO::TensorType dstType, Value inChain) {
  DeviceRefAttr srcDevice = srcType.getDeviceRef();
  DeviceRefAttr dstDevice = dstType.getDeviceRef();

  if (srcDevice == hostDevice) {
    return opLoweringState.bufferTransferFromHost(
        inChain, cast<TypedValue<MGP::DeviceBufferType>>(srcBuffer.buffer),
        dstBuffer, dstType);
  }

  if (dstDevice == hostDevice) {
    return opLoweringState.bufferTransferToHost(
        inChain, srcBuffer.buffer,
        cast<TypedValue<MGP::DeviceBufferType>>(dstBuffer), srcType);
  }

  if (srcDevice.getLabel() == dstDevice.getLabel()) {
    return opLoweringState.bufferTransferDevice(inChain, srcBuffer.buffer,
                                                dstBuffer, srcType, dstType);
  }

  return nullptr; // Indicates unsupported device transfer
}

LogicalResult MOGraphLowering::lowerTransfer(MO::TransferOp transferOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, transferOp);

  auto srcType = transferOp.getInput().getType();
  auto dstType = transferOp.getResult().getType();
  if (!srcType.getDeviceRef() || !dstType.getDeviceRef())
    return transferOp->emitOpError(
        "should have explicit devices before lowering");

  MGP::BufferizedTensor bufferizedInput =
      opLoweringState.getBufferizedInput(transferOp.getInput());

  // Respect explicit chain threading for transfer.
  Value inChain =
      loweringState.lookupExecuteLoweredValue(transferOp.getInChain());
  inChain = opLoweringState.getGuardChain(inChain);

  FailureOr<SmallVector<MGP::MOOutputStorage>> loweredOutputsStorageOr =
      opLoweringState.createOutputStorage(inChain,
                                          {bufferizedInput.tensorSpec});
  if (failed(loweredOutputsStorageOr))
    return failure();
  auto &loweredOutputsStorage = *loweredOutputsStorageOr;

  Value outChain = lowerTransferImpl(
      opLoweringState, loweringState.getHostDevice(), bufferizedInput,
      loweredOutputsStorage[0].buffer, srcType, dstType, inChain);

  if (!outChain)
    return transferOp->emitOpError("mo.transfer from device '")
           << srcType.getDeviceRef() << "' to device '"
           << dstType.getDeviceRef() << "' is not supported.";

  opLoweringState.createOutputs(outChain, loweredOutputsStorage);

  // Map explicit outChain result.
  loweringState.emplaceExecuteLoweredValue(transferOp.getOutChain(), outChain);

  return success();
}

LogicalResult
MOGraphLowering::lowerBufferTransfer(MO::BufferTransferOp transferOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, transferOp);

  auto srcType = MO::castFromTensorLike(transferOp.getSrc().getType());
  auto dstType = MO::castFromTensorLike(transferOp.getDst().getType());
  if (!srcType.getDeviceRef() || !dstType.getDeviceRef())
    return transferOp->emitOpError(
        "should have explicit devices before lowering");

  MGP::BufferizedTensor bufferizedInput =
      opLoweringState.getBufferizedInput(transferOp.getSrc());
  MGP::BufferizedTensor bufferizedOutput =
      opLoweringState.getBufferizedInput(transferOp.getDst());

  auto inChain =
      loweringState.lookupExecuteLoweredValue(transferOp.getInChain());
  inChain = opLoweringState.getGuardChain(inChain);

  Value outChain = lowerTransferImpl(
      opLoweringState, loweringState.getHostDevice(), bufferizedInput,
      bufferizedOutput.buffer, srcType, dstType, inChain);

  if (!outChain)
    return transferOp->emitOpError("mo.buffer.transfer from device '")
           << srcType.getDeviceRef() << "' to device '"
           << dstType.getDeviceRef() << "' is not supported.";

  loweringState.emplaceExecuteLoweredValue(transferOp.getOutChain(), outChain);
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

LogicalResult
MOGraphLowering::lowerMutableStore(MO::MutableStoreOp mutableStoreOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, mutableStoreOp);

  MO::TensorType srcType = mutableStoreOp.getInTensor().getType();
  MO::TensorType dstType =
      MO::castFromTensorLike(mutableStoreOp.getInBuffer().getType());

  auto srcBufferized =
      opLoweringState.getBufferizedInput(mutableStoreOp.getInTensor());
  auto dstBufferized =
      opLoweringState.getBufferizedInput(mutableStoreOp.getInBuffer());

  auto inChain =
      loweringState.lookupExecuteLoweredValue(mutableStoreOp.getInChain());
  inChain = opLoweringState.getGuardChain(inChain);

  auto loweredChain = opLoweringState.bufferTransferDevice(
      inChain, srcBufferized.buffer, dstBufferized.buffer, srcType, dstType);

  loweringState.emplaceExecuteLoweredValue(mutableStoreOp.getResult(),
                                           loweredChain);

  return success();
}

LogicalResult
MOGraphLowering::lowerBufferCreate(MO::BufferCreateOp bufferCreateOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, bufferCreateOp);

  Type resultType = bufferCreateOp.getResult().getType();
  auto resTensorType = MO::castFromTensorLike(resultType);

  Value inChain = loweringState.getCurrentExecuteChain();
  inChain = opLoweringState.getGuardChain(inChain);

  auto tensorSpecOr = opLoweringState.createTensorSpec(resTensorType);
  if (failed(tensorSpecOr))
    return failure();

  auto bufferOr =
      opLoweringState.createBuffer(inChain, *tensorSpecOr, resultType);
  if (failed(bufferOr))
    return failure();

  opLoweringState.createOutputTensor(bufferCreateOp.getResult(), inChain,
                                     bufferOr->buffer, bufferOr->tensorSpec);

  return success();
}

LogicalResult
MOGraphLowering::lowerMutableLoad(MO::MutableLoadOp mutableLoadOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, mutableLoadOp);

  auto inChain =
      loweringState.lookupExecuteLoweredValue(mutableLoadOp.getInChain());

  auto bufferizedInput =
      opLoweringState.getBufferizedInput(mutableLoadOp.getInBuffer());

  inChain = opLoweringState.getGuardChain(inChain);
  auto bufOr =
      opLoweringState.createBuffer(inChain, bufferizedInput.tensorSpec,
                                   mutableLoadOp.getOutTensor().getType());
  if (failed(bufOr))
    return failure();

  auto &outputStorage = *bufOr;

  auto loweredChain = opLoweringState.bufferTransferDevice(
      inChain, bufferizedInput.buffer, outputStorage.buffer,
      MO::castFromTensorLike(mutableLoadOp.getInBuffer().getType()),
      mutableLoadOp.getOutTensor().getType());

  opLoweringState.createOutputTensor(mutableLoadOp.getOutTensor(), loweredChain,
                                     outputStorage.buffer,
                                     outputStorage.tensorSpec);

  loweringState.emplaceExecuteLoweredValue(mutableLoadOp.getOutChain(),
                                           loweredChain);

  return success();
}

LogicalResult MOGraphLowering::lowerIf(MO::IfOp ifOp) {
  ASSERT_STREAM(MO::isOnHost(ifOp), << "mo.if is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, ifOp);

  // We'll need to thread the current 'execute' chain through the if
  // 'then' and 'else' blocks.
  Value currentChain = loweringState.getCurrentExecuteChain();
  currentChain = opLoweringState.getGuardChain(currentChain);

  Value cond = ifOp.getCond();

  bool isScalarCond = isa<MO::ScalarType>(cond.getType());

  Value condBool;
  if (isScalarCond) {
    condBool = loweringState.lookupExecuteLoweredValue(cond);
  } else {
    // Lower the condition.
    MGP::BufferizedTensor bufferizedCond =
        opLoweringState.getBufferizedInput(cond);
    condBool = opLoweringState.bufferToBool(currentChain, bufferizedCond.buffer,
                                            cond.getType());
  }

  SmallVector<Value> operands;
  operands.push_back(condBool);

  SmallVector<Type> resultTypes;
  // Always return the current 'execute' chain
  resultTypes.emplace_back(currentChain.getType());

  for (Type type : ifOp.getResultTypes())
    resultTypes.emplace_back(
        loweringState.getTypeConverter().convertType(type));

  for (ParamDeclAttr decl : ifOp.getOutputParamDecls()) {
    Type declType = decl.getType();
    Type converted = loweringState.getTypeConverter().convertType(declType);
    resultTypes.emplace_back(converted ? converted : declType);
  }

  // Create grt.if op
  auto outputIf = opLoweringState.create<GRT::IfOp>(resultTypes, operands);

  // Lower optional output parameters
  size_t paramStartIdx = 1 + ifOp.getNumResults(); // After chain + tensors
  for (auto [decl, result] :
       llvm::zip(ifOp.getOutputParamDecls(),
                 outputIf.getResults().drop_front(paramStartIdx))) {
    if (auto err =
            loweringState.lowerParamDecl(ifOp->getParentRegion(), decl, result))
      return ifOp->emitOpError(err.getError());
  }

  auto enterLowerBlock = [&](Region &inputRegion,
                             Region &outputRegion) -> LogicalResult {
    // Enter the lowered region's unique block.
    Block &newBlock = outputRegion.emplaceBlock();
    opLoweringState.builder.setInsertionPointToEnd(&newBlock);

    if (failed(lowerBlock(&inputRegion.getBlocks().front())))
      return failure();
    return success();
  };

  // Lower Then Block
  if (failed(enterLowerBlock(ifOp.getThenRegion(), outputIf.getThenRegion())))
    return failure();

  // Reset the main chain between blocks
  loweringState.setCurrentExecuteChain(currentChain);

  // Lower Else Block
  if (failed(enterLowerBlock(ifOp.getElseRegion(), outputIf.getElseRegion())))
    return failure();

  Value outChain = outputIf.getResult(0);

  // Set the outputs
  loweringState.setCurrentExecuteChain(outChain);

  for (auto [origResult, loweredResult] :
       llvm::zip(ifOp.getResults(), llvm::drop_begin(outputIf.getResults()))) {

    if (isa<MGP::DeviceTensorType>(loweredResult.getType())) {
      loweringState.emplaceExecuteLoweredBufferValue(origResult, loweredResult,
                                                     outChain);
    } else {
      loweringState.emplaceExecuteLoweredValue(origResult, loweredResult);
    }
  }

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
  currentChain = opLoweringState.getGuardChain(currentChain);

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

  auto enterLowerBlock = [&](Region &inputRegion,
                             Region &outputRegion) -> LogicalResult {
    // Enter the lowered region's unique block.
    Block &newBlock = outputRegion.emplaceBlock();
    opLoweringState.builder.setInsertionPointToEnd(&newBlock);

    // Establish the block's arguments. The current 'execute' chain comes
    // first.
    auto loweredChainArg =
        outputRegion.addArgument(currentChain.getType(), whileOp.getLoc());
    loweringState.setCurrentExecuteChain(loweredChainArg);
    if (currentIsRoot)
      loweringState.setRootExecuteChain(loweredChainArg);
    // else: if the block needs to reference the root chain then an
    //       additional block argument will be added by
    //       fixupIsolatedFromAbove.

    // Then an arg for each of the original region args.
    // We'll use the loweredFreeValues since they have the lowered types
    // already, though we skip the chain.
    for (auto [loweredFreeValue, arg] : llvm::zip(
             llvm::drop_begin(loweredFreeValues), inputRegion.getArguments())) {
      Value loweredArg =
          outputRegion.addArgument(loweredFreeValue.getType(), arg.getLoc());
      if (isa<MGP::DeviceTensorType>(loweredArg.getType())) {
        loweringState.emplaceExecuteLoweredBufferValue(arg, loweredArg,
                                                       loweredChainArg);
      } else {
        loweringState.emplaceExecuteLoweredValue(arg, loweredArg);
      }
    }

    // Lower the block, including its terminating mo.yield.
    if (failed(lowerBlock(&inputRegion.getBlocks().front())))
      return failure();

    return success();
  };

  // Lower the 'cond' block. We'll later need to make 2 adjustments:
  // 1) The mo.while 'cond' block will mo.yield only the condition result.
  // That
  //    mo.yield will be lowered to a mef.output with the 'execute' chain and
  //    lowered condition results. However, the grt.while requires the 'cond'
  //    block to return a result for all signature arguments plus the
  //    condition, so we'll need to include those.
  // 2) The lowered condition will be a mgp.tensor<bool>, which we need to
  //    convert to an i1.
  if (failed(enterLowerBlock(whileOp.getCondRegion(),
                             loweredWhileOp.getCondRegion())))
    return failure();

  // Lower the 'body' block.
  if (failed(enterLowerBlock(whileOp.getBodyRegion(),
                             loweredWhileOp.getBodyRegion())))
    return failure();

  // Now repair the lowered 'cond' block's mef.output to include all the
  // additional variables between the output chain and the condition,
  // and fix the condition to be an i1.
  {
    MO::WhileConditionOp origYieldOp = whileOp.getCondTerminator();

    Value origTensor = origYieldOp.getCondition();
    Type condMoTensorType = origTensor.getType();

    MEF::OutputOp loweredCondOutputOp = loweredWhileOp.getCondTerminator();

    // At this point, the lowered OutputOp operands are the form:
    // 1. the execute chain
    // 2. the lowered condition tensor
    // 3. the lowered arguments of the mo.yield
    Value loweredOutputChain = loweredCondOutputOp->getOperand(0);
    Value loweredCondTensor = loweredCondOutputOp->getOperand(1);
    ValueRange loweredArgs = loweredCondOutputOp->getOperands().drop_front(2);

    SmallVector<Value> condOutputOpOperands;
    condOutputOpOperands.reserve(loweredWhileOp->getNumResults() + 1);

    // chain
    condOutputOpOperands.push_back(loweredOutputChain);

    // Pass along any additional values, substituting out the chain values for
    // the corresponding chain arguments from the mo.yield.
    llvm::append_range(condOutputOpOperands, loweredArgs);

    // i1 condition
    opLoweringState.builder.setInsertionPoint(loweredCondOutputOp);
    Value condBuffer =
        opLoweringState.tensorExtract(loweredCondTensor, condMoTensorType)
            .first;

    Value chain = loweredOutputChain;
    if (Value guardChain =
            loweringState.getChainGuardingValueOrNull(origTensor)) {
      chain =
          GRT::ChainCreateOp::create(opLoweringState.builder, whileOp.getLoc(),
                                     ValueRange{guardChain, chain})
              .getResult();
    }
    Value condBool =
        opLoweringState.bufferToBool(chain, condBuffer, condMoTensorType);
    condOutputOpOperands.push_back(condBool);

    loweredCondOutputOp.getOperandsMutable().clear();
    loweredCondOutputOp.getOperandsMutable().append(condOutputOpOperands);
  }

  // Capture the lowered outputs.
  assert(loweredWhileOp.getNumResults() == 1 + whileOp.getNumResults());

  Value outChain = loweredWhileOp.getResult(0);

  loweringState.setCurrentExecuteChain(outChain);
  loweringState.setRootExecuteChain(rootChain);

  for (auto [origResult, loweredResult] :
       llvm::zip(whileOp.getResults(),
                 llvm::drop_begin(loweredWhileOp.getResults()))) {
    if (isa<MGP::DeviceTensorType>(loweredResult.getType())) {
      loweringState.emplaceExecuteLoweredBufferValue(origResult, loweredResult,
                                                     outChain);
    } else {
      loweringState.emplaceExecuteLoweredValue(origResult, loweredResult);
    }
  }

  // Since we moved the insertion points for the 'cond' and 'body' blocks,
  // reset back to the main block
  opLoweringState.builder.setInsertionPointAfter(loweredWhileOp);

  return success();
}

// level-wise topological sort
static LogicalResult levelwiseTopologicalSort(Block *block) {
  llvm::iterator_range<Block::iterator> ops = *block;
  // If the block has a terminator, we need to exclude it from the sort.
  if (block->back().hasTrait<OpTrait::IsTerminator>())
    ops = block->without_terminator();

  if (ops.empty())
    return success();

  llvm::SmallVector<Operation *> allOps;
  llvm::DenseMap<Operation *, unsigned> inDegree;
  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> dependents;

  // Collect all operations and find their dependents
  for (Operation &op : ops) {
    allOps.push_back(&op);
    inDegree[&op] = 0;
  }

  for (Operation *op : allOps) {
    llvm::SetVector<Operation *> dependencies;

    // Walk through operands of all nested operations to find dependencies
    op->walk</*Order=*/mlir::WalkOrder::PreOrder>([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
        Operation *parent = operand.getDefiningOp();
        if (!parent)
          continue; // Block argument

        while (parent->getParentOp() &&
               parent->getParentOp() != block->getParentOp()) {
          parent = parent->getParentOp();
        }

        // Only consider dependencies within the same block
        if (parent->getBlock() == block && parent != op) {
          dependencies.insert(parent);
        }
      }

      // If the operation IsIsolatedFromAbove, then we can skip any child
      // regions since we know they cannot refer to values defined outside
      // their bodies.
      if (nestedOp->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
        return WalkResult::skip();
      return WalkResult::advance();
    });

    // Update in-degrees and dependent lists
    for (Operation *dep : dependencies) {
      dependents[dep].push_back(op);
      inDegree[op]++;
    }
  }

  Block::iterator nextScheduledOp = block->begin();
  [[maybe_unused]] unsigned sortedOps = 0;

  // Initialize queue with operations that have no dependencies
  llvm::SmallVector<Operation *> readyQueue;
  for (Operation *op : allOps) {
    if (inDegree[op] == 0) {
      readyQueue.push_back(op);
    }
  }

  // Process operations level by level
  while (!readyQueue.empty()) {
    llvm::SmallVector<Operation *> currentLevel = std::move(readyQueue);
    readyQueue.clear();

    // Schedule all operations at current level
    for (Operation *op : currentLevel) {
      ++sortedOps;
      op->moveBefore(block, nextScheduledOp);

      if (op == &*nextScheduledOp) {
        ++nextScheduledOp;
      }

      // Update in-degrees of dependent operations
      for (Operation *dependent : dependents[op]) {
        inDegree[dependent]--;
        if (inDegree[dependent] == 0) {
          readyQueue.push_back(dependent);
        }
      }
    }
  }

  // Check if all operations were scheduled
  ASSERT_STREAM(sortedOps == allOps.size(),
                << "Failed to sort the MO operations topologically.");

  return success();
}

LogicalResult MOGraphLowering::lowerBlock(Block *block) {
  // Ensure the block is sorted with operands to operations appearing before
  // their use.
  if (failed(levelwiseTopologicalSort(block)))
    return failure();

  for (Operation &op : *block) {
    if (llvm::isa_and_nonnull<mlir::index::IndexDialect>(op.getDialect())) {
      if (failed(cloneTrivialOp(&op)))
        return failure();
      continue;
    }

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

    if (auto paramFromValueOp = dyn_cast<MOSH::ParamFromValueOp>(op)) {
      if (failed(lowerParamFromValue(block->getParent(), paramFromValueOp)))
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

    if (auto fromTensorOp = dyn_cast<MO::Shape::FromTensorOp>(op)) {
      if (failed(lowerShapeFromTensorOp(fromTensorOp)))
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

    if (auto constExternalOp = dyn_cast<MO::ConstantExternalOp>(op)) {
      if (failed(lowerConstantExternal(constExternalOp)))
        return failure();
      continue;
    }

    if (auto constOp = dyn_cast<MO::ConstantScalarOp>(op)) {
      if (failed(lowerConstantScalar(constOp)))
        return failure();
      continue;
    }

    if (auto reshapeOp = dyn_cast<MO::Static::ReshapeOp>(op)) {
      MGP::MOOpLoweringState opLoweringState(loweringState, reshapeOp);

      if (failed(lowerReshapeImpl(
              opLoweringState, reshapeOp.getInput(), reshapeOp.getResult(),
              loweringState.getExecuteChain(), reshapeOp.getType())))
        return failure();
      continue;
    }

    if (auto reshapeOp = dyn_cast<MO::AddSingletonDimOp>(op)) {
      MGP::MOOpLoweringState opLoweringState(loweringState, reshapeOp);

      if (failed(lowerReshapeImpl(
              opLoweringState, reshapeOp.getInput(), reshapeOp.getResult(),
              loweringState.getExecuteChain(), reshapeOp.getType())))
        return failure();
      continue;
    }

    if (auto reshapeOp = dyn_cast<MO::SplitDimOp>(op)) {
      MGP::MOOpLoweringState opLoweringState(loweringState, reshapeOp);

      if (failed(lowerReshapeImpl(
              opLoweringState, reshapeOp.getInput(), reshapeOp.getResult(),
              loweringState.getExecuteChain(), reshapeOp.getType())))
        return failure();
      continue;
    }

    if (auto reshapeOp = dyn_cast<MO::MergeDimOp>(op)) {
      MGP::MOOpLoweringState opLoweringState(loweringState, reshapeOp);

      if (failed(lowerReshapeImpl(
              opLoweringState, reshapeOp.getInput(), reshapeOp.getResult(),
              loweringState.getExecuteChain(), reshapeOp.getType())))
        return failure();
      continue;
    }

    if (auto debugPrintOp = dyn_cast<MO::DebugPrintOp>(op)) {
      if (failed(lowerDebugPrint(debugPrintOp)))
        return failure();
      continue;
    }

    if (auto debugTensorPrintOp = dyn_cast<MO::DebugTensorPrintOp>(op)) {
      if (failed(lowerDebugTensorPrint(debugTensorPrintOp)))
        return failure();
      continue;
    }

    if (auto debugTensorUnsafePrintOp =
            dyn_cast<MO::DebugTensorUnsafePrintOp>(op)) {
      if (failed(lowerDebugTensorUnsafePrint(debugTensorUnsafePrintOp)))
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

    // Lower mo.output to mef.output, accounting for implicit init/exec args.
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

    // Lower mo.while.condition to mef.output trivially.
    if (auto output = dyn_cast<MO::WhileConditionOp>(op)) {
      if (failed(lowerWhileCondition(output)))
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

    // Lowers mo.mutable.store to a mgp.buffer.device_to_device
    if (auto mop = dyn_cast<MO::MutableStoreOp>(op)) {
      if (failed(lowerMutableStore(mop)))
        return failure();
      continue;
    }

    // Lowers mo.mutable.load to a mgp.buffer.device_to_device
    if (auto mop = dyn_cast<MO::MutableLoadOp>(op)) {
      if (failed(lowerMutableLoad(mop)))
        return failure();
      continue;
    }

    // Lowers mo.buffer.create
    if (auto bufferCreateOp = dyn_cast<MO::BufferCreateOp>(op)) {
      if (failed(lowerBufferCreate(bufferCreateOp)))
        return failure();
      continue;
    }

    if (auto bufferTransferOp = dyn_cast<MO::BufferTransferOp>(op)) {
      if (failed(lowerBufferTransfer(bufferTransferOp)))
        return failure();
      continue;
    }

    if (auto callOp = dyn_cast<MO::CallOp>(op)) {
      if (failed(lowerCallGraph(callOp)))
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
    auto loweringResult =
        loweringState.lowerArg(&graph.getRegion(), graphArg, exeArg);
    if (loweringResult.isError())
      return graph->emitOpError(loweringResult.getError());

    // If the argument has no uses (and no parameters) then just ignore it.
    if (!loweringResult->hasParameters && graphArg.use_empty())
      continue;

    loweringState.emplaceExecuteLoweredValue(graphArg,
                                             loweringResult->loweredArg);
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

LogicalResult MOGraphLowering::lowerCallGraph(MO::CallOp callOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, callOp);
  OpBuilder b(callOp);

  // Get the target model name
  StringRef targetModelName = callOp.getCallee();

  // Lookup the target model in the module
  auto moduleOp = callOp->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return callOp.emitError() << "Failed to find parent ModuleOp";

  auto targetModel = moduleOp.lookupSymbol<MGP::ModelOp>(targetModelName);
  if (!targetModel)
    return callOp.emitError()
           << "Failed to find target model with name: " << targetModelName;

  // The MGP call needs all the implicit arguments plus the explicit ones
  // The implicit args are: chain, context, weights registry, and devices
  SmallVector<Value> callOperands;

  // First, we need the implicit arguments
  // Get chain, context, weights registry from the current model's execute block
  callOperands.push_back(loweringState.getCurrentExecuteChain());
  callOperands.push_back(loweringState.getExecuteMgpContext());
  callOperands.push_back(loweringState.getExecuteWeightsRegistry());

  // Add device contexts
  for (auto deviceAttr : targetModel.getDevices()) {
    DeviceRefAttr deviceRef = cast<DeviceRefAttr>(deviceAttr);
    callOperands.push_back(loweringState.getExecuteDevice(deviceRef));
  }

  // Now add the explicit operands from the mo.call op
  // These need to be mapped to their lowered MGP equivalents
  for (auto operand : callOp.getOperands()) {
    // Get the lowered value (must have been lowered earlier)
    Value loweredValue = loweringState.lookupExecuteLoweredValue(operand);
    if (!loweredValue)
      return callOp.emitError() << "Operand not lowered yet: " << operand;

    callOperands.push_back(loweredValue);
  }

  // Determine result types for the call
  SmallVector<Type> resultTypes;

  // First result is always chain
  resultTypes.push_back(b.getType<MEF::ChainType>());

  // Add the explicit result types
  for (auto result : callOp.getResults()) {
    Type loweredType =
        loweringState.getTypeConverter().convertType(result.getType());
    if (!loweredType) {
      return callOp.emitError()
             << "Failed to lower result type: " << result.getType();
    }
    resultTypes.push_back(loweredType);
  }

  // Create the mgp.call operation
  auto targetSymbol = SymbolRefAttr::get(targetModel.getSymNameAttr());
  auto mgpCallOp = opLoweringState.create<MGP::CallOp>(
      resultTypes, targetSymbol, callOp.getPrefixAttr(), callOperands);
  Value callOutChain = mgpCallOp.getResult(0);

  // Map the results back
  // Skip the first result (chain) for mapping back to original results
  for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
    loweringState.emplaceExecuteLoweredValue(callOp.getResult(i),
                                             mgpCallOp.getResult(i + 1));

    // Guard every result with the chain
    loweringState.guardValueWithChain(mgpCallOp.getResult(i + 1), callOutChain);
  }

  loweringState.setCurrentExecuteChain(callOutChain);

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

  std::deque<MO::GraphOp> worklist;

  DenseSet<StringAttr> completedGraphs;
  DenseMap<StringAttr, DenseSet<StringAttr>> graphDependencies;

  for (auto graphOp : theModule.getOps<MO::GraphOp>()) {
    worklist.push_back(graphOp);

    auto &dependencies = graphDependencies[graphOp.getNameAttr()];
    graphOp.walk([&](MO::CallOp callOp) {
      auto calleeName = callOp.getCalleeAttr().getAttr();
      dependencies.insert(calleeName);
    });
  }

  size_t lastProcessed = 0;
  // Process graphs in dependency order until all graphs are processed.
  while (!worklist.empty()) {
    // If no graphs have been processed after one full iteration through
    // the worklist, then there must be a cycle.
    if (lastProcessed > worklist.size()) {
      theModule.emitError("cycle detected in graph ops");
      return signalPassFailure();
    }

    ++lastProcessed;

    MO::GraphOp graph = worklist.back();
    worklist.pop_back();

    // Check if all our dependencies have been processed.
    auto &dependencies = graphDependencies[graph.getNameAttr()];

    if (!llvm::set_is_subset(dependencies, completedGraphs)) {
      // There are still unprocessed dependencies, so push the graph onto the
      // front of the worklist and move on.
      worklist.push_front(graph);
      continue;
    }

    // Mark the graph as processed.
    completedGraphs.insert(graph.getNameAttr());

    // All dependencies are processed, so we can convert this graph
    // Convert each mo.graph into a mgp.model using MOGraphLowering
    if (failed(MOGraphLowering(graph, compiledFrameworkLabel.getValue(),
                               minimumCpuAlignment.getValue())
                   .run()))
      return signalPassFailure();

    lastProcessed = 0;
  }
}

// Instead of storing `KernelStubOp` directly, we can store it with its
// precomputed hash. This speeds up multiple insertions of the same stub op
// because the part of the code that inserts the op also has access to the hash.
struct KernelStubHashEntry {
  MOGG::KernelStubOp op;
  unsigned hash;

  // Will be called in isEqual only.
  bool operator==(const KernelStubHashEntry &other) const {
    return other.op == op;
  }
};

struct KernelStubOpEquivalenceInfo
    : public llvm::DenseMapInfo<KernelStubHashEntry> {

  // This scheme breaks all of LLVM/MLIR if a virtual address has a value of 1
  // or 2.
  static inline KernelStubHashEntry getEmptyKey() {
    return {MOGG::KernelStubOp::getFromOpaquePointer(
                DenseMapInfo<void *>::getEmptyKey()),
            0};
  }
  static inline KernelStubHashEntry getTombstoneKey() {
    return {MOGG::KernelStubOp::getFromOpaquePointer(
                DenseMapInfo<void *>::getTombstoneKey()),
            0};
  }

  // We first compute the hash of a node and store it with the entry in the
  // dense set. This is an optimization where subsequent insertions of the same
  // op will not rehash. We will be comparing two entries when the hash collides
  // though but this is less frequent than rehashing every time we try to insert
  // something.
  static unsigned getHashValue(const KernelStubHashEntry cStubOp) {
    return cStubOp.hash;
  }
  static bool isEqual(KernelStubHashEntry lhs, KernelStubHashEntry rhs) {
    if (lhs.op == rhs.op)
      return true;

    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;

    auto lhsProperties = lhs.op.getProperties();
    auto rhsProperties = rhs.op.getProperties();

    // Check if properties are equivalent modulo sym_name and summary
    // attributes.
    lhsProperties.sym_name = rhsProperties.sym_name = {};
    lhsProperties.summary = rhsProperties.summary = {};

    if (lhsProperties != rhsProperties)
      return false;

    DictionaryAttr lhsDictAttr = lhs.op->getRawDictionaryAttrs();
    DictionaryAttr rhsDictAttr = rhs.op->getRawDictionaryAttrs();
    if (lhsDictAttr.size() != rhsDictAttr.size())
      return false;

    // Check if discardable attributes are equivalent.
    for (auto [lhsAttr, rhsAttr] : llvm::zip_equal(lhsDictAttr, rhsDictAttr)) {
      if (lhsAttr.getName() != rhsAttr.getName())
        return false;

      if (lhsAttr.getValue() == rhsAttr.getValue())
        continue;

      auto attrName = lhsAttr.getName();
      if (attrName != MO::kOperationDeviceRefAttrName)
        return false;

      // Do not compare the device ID, just the label.
      if (cast<DeviceRefAttr>(lhsAttr.getValue()).getLabel() !=
          cast<DeviceRefAttr>(rhsAttr.getValue()).getLabel())
        return false;
    }

    return mlir::OperationEquivalence::isRegionEquivalentTo(
        &lhs.op.getBodyRegion(), &rhs.op.getBodyRegion(),
        /*flags=*/mlir::OperationEquivalence::Flags::IgnoreLocations);
  }
};

static unsigned getHashValue(const MOGG::KernelStubOp cStubOp) {
  llvm::hash_code hash = {};
  MOGG::KernelStubOp stubOp = const_cast<MOGG::KernelStubOp &>(cStubOp);

  auto properties = stubOp.getProperties();
  properties.sym_name = {};
  properties.summary = {};

  stubOp->getName().hashOpProperties(mlir::OpaqueProperties(&properties));

  for (NamedAttribute namedAttr : cStubOp->getDiscardableAttrDictionary()) {
    StringAttr attrName = namedAttr.getName();

    // Do not hash the device ID, just the label.
    if (attrName == MO::kOperationDeviceRefAttrName) {
      hash = llvm::hash_combine(
          hash, cast<DeviceRefAttr>(namedAttr.getValue()).getLabel());
      continue;
    }

    hash = llvm::hash_combine(hash, namedAttr);
  }

  stubOp.getBodyBlock().walk([&](Operation *op) {
    hash = llvm::hash_combine(
        hash,
        mlir::OperationEquivalence::computeHash(
            op, /*hashOperands=*/mlir::OperationEquivalence::ignoreHashValue,
            /*hashResults=*/mlir::OperationEquivalence::ignoreHashValue,
            /*flags=*/mlir::OperationEquivalence::Flags::IgnoreLocations));
  });

  return hash;
}

struct CallGraphNode : public KGEN::SCCNode<CallGraphNode, MOGG::KernelStubOp,
                                            MOGG::CallStubOp> {
  using Base =
      KGEN::SCCNode<CallGraphNode, MOGG::KernelStubOp, MOGG::CallStubOp>;
  using Base::Base;

  unsigned hash;
};

struct CallGraph : public KGEN::SCCGraph<CallGraph, CallGraphNode,
                                         /*BottomUpRewrite=*/true> {
  bool doAnalysis(CallGraphNode *node) { return false; }
  // For all callsites inside the current node, rewrite them if you aren't
  // able to insert them in the hash table.
  void doRewrite(CallGraphNode *node);

  DenseSet<KernelStubHashEntry, KernelStubOpEquivalenceInfo> uniqueFuncSet;

  // A mapping that shows the renaming of kernels that aren't called by any
  // other kernel (top-level). This is necessary because the CallGraph will only
  // replace the ones that get called. At the very end in runOnOperation, we do
  // the replacement for the top-level kernels using this mapping.
  llvm::DenseMap<StringAttr, StringAttr> topLevelReplacements;

  // The mutex to protect uniqueFuncSet accesses.
  std::mutex setAccessMutex;
};

void CallGraph::doRewrite(CallGraphNode *node) {
  for (auto [caller, callee] : node->callsites) {
    MOGG::KernelStubOp replacementCallee = nullptr;
    {
      std::lock_guard<std::mutex> lock(setAccessMutex);
      replacementCallee =
          uniqueFuncSet.insert(KernelStubHashEntry{callee->func, callee->hash})
              .first->op;
    }

    if (replacementCallee != callee->func) {
      caller.setNameAttr(FlatSymbolRefAttr::get(
          caller->getContext(), replacementCallee.getSymName()));
    }
  }

  node->hash = getHashValue(node->func);
  // Try to insert yourself if no CallGraphNode is calling you. If another top
  // level stub (that has no callers) hashes to the same value, store the
  // replacement in another map that will be used after the analysis to replace
  // the external calls (see runOnOperation below).
  if (node->callers.empty()) {
    MOGG::KernelStubOp replacementCallee = nullptr;
    std::lock_guard<std::mutex> lock(setAccessMutex);
    replacementCallee =
        uniqueFuncSet.insert(KernelStubHashEntry{node->func, node->hash})
            .first->op;
    if (replacementCallee != node->func)
      topLevelReplacements[node->func.getSymNameAttr()] =
          replacementCallee.getSymNameAttr();
  }
}

void MGP::mergeIdenticalKernelStubs(ModuleOp module) {
  SymbolTable symTable(module);
  CallGraph cg;
  cg.build(module, symTable);
  cg.run(*loadContext(module.getContext())->get<AsyncRT::Runtime>());

  // ModelOp contain calls to stubs too but those do not come from a
  // KernelStubOp. They are ExecuteCompiledOp. In order to merge high level
  // stubs, we have to do the attribute replacement here.
  for (auto modelOp : module.getOps<MGP::ModelOp>()) {
    modelOp.walk([&](MOGG::ExecuteCompiledOp callOp) {
      auto it =
          cg.topLevelReplacements.find(callOp.getCalleeSymbolAttr().getAttr());
      if (it != cg.topLevelReplacements.end())
        callOp.setCalleeSymbolAttr(mlir::FlatSymbolRefAttr::get(it->second));
    });
  }
}
