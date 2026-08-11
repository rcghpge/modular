//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Converts MO ops to MGP and MEF ops.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/MOGGToMGP/MOGGToMGP.h"
#include "AsyncRT/CompilerSupport/Context.h"
#include "GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GraphCompiler/Interfaces/MOGGKernelOpLoweringState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/MGPDialect/Transforms/FixupIsolatedFromAbove.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/TransformUtils/SCCUtils.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "Support/AssertStream.h"
#include "Support/ML/CommonAttributes.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/StringMap.h"

#include <deque>
#include <numeric>

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_MOGGTOMGP
#include "GraphCompiler/Conversion/MOGGToMGP/MOGGToMGP.h.inc"

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

  // `pop.select` takes a `!kgen.scalar<bool>` condition, but `index.cmp`
  // produces an `i1`. Convert the builtin `i1` to `!kgen.scalar<bool>`.
  auto asScalarBool(Value i1Cond) -> Value {
    auto boolTy = KGEN::SIMDType::get(opLoweringState.builder.getContext(), 1,
                                      KGEN::KGENDType::kBool);
    return opLoweringState.create<KGEN::POP::CastFromBuiltinOp>(boolTy, i1Cond);
  }

  auto ifPositive(Value check, Value yes, Value no) -> Value {
    auto indexTy = opLoweringState.builder.getIndexType();
    auto cond = opLoweringState.create<mlir::index::CmpOp>(
        mlir::index::IndexCmpPredicate::SLT, check, constant(0));
    return opLoweringState.create<KGEN::POP::SelectOp>(
        indexTy, asScalarBool(cond), no, yes);
  }

  auto ifZero(Value check, Value yes, Value no) -> Value {
    auto indexTy = opLoweringState.builder.getIndexType();
    auto cond = opLoweringState.create<mlir::index::CmpOp>(
        mlir::index::IndexCmpPredicate::EQ, check, constant(0));
    return opLoweringState.create<KGEN::POP::SelectOp>(
        indexTy, asScalarBool(cond), yes, no);
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
                  size_t minimumCpuAlignment,
                  mlir::SymbolTable &moduleSymbolTable)
      : loweringState(graph, compiledFrameworkLabel, minimumCpuAlignment,
                      moduleSymbolTable) {}

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

  /// Per-op lowering overloads. Each concrete MO/MOGG/KGEN op type has a
  /// corresponding `lower()` overload; `lowerOp()` dispatches to these via
  /// TypeSwitch.
  LogicalResult lower(MO::OutputOp op);
  LogicalResult lower(MO::YieldOp op);
  LogicalResult lower(MO::WhileConditionOp op);
  LogicalResult lower(MO::IndexToTensorOp op);
  LogicalResult lower(MO::Shape::ToTensorOp op);
  LogicalResult lower(MO::Shape::FromTensorOp op);
  LogicalResult lower(MO::RebindOp op);
  LogicalResult lower(MO::ConstantOp op);
  LogicalResult lower(MO::ConstantExternalOp op);
  LogicalResult lower(MO::ConstantScalarOp op);
  LogicalResult lower(MO::DebugPrintOp op);
  LogicalResult lower(MO::DebugTensorPrintOp op);
  LogicalResult lower(MO::TransferOp op);
  LogicalResult lower(MO::IfOp op);
  LogicalResult lower(MO::WhileOp op);
  LogicalResult lower(MO::GuardOp op);
  LogicalResult lower(MO::ChainCreateOp op);
  LogicalResult lower(MO::MutableStoreOp op);
  LogicalResult lower(MO::MutableLoadOp op);
  LogicalResult lower(MO::BufferCreateOp op);
  LogicalResult lower(MO::BufferTransferOp op);
  LogicalResult lower(MO::CallOp op);
  LogicalResult lower(MO::ParallelOp op);
  LogicalResult lower(MO::SequenceOp op);
  LogicalResult lower(MO::TensorBundleOp op);
  LogicalResult lower(MO::TensorUnbundleOp op);
  LogicalResult lower(MOGG::KernelRegionOp op);

  /// Param ops need the enclosing region to resolve parameter expressions.
  LogicalResult lowerParam(Region *parentRegion, KGEN::ParamDeclareOp op);
  LogicalResult lowerParam(Region *parentRegion, MOSH::ParamToValueOp op);
  LogicalResult lowerParam(Region *parentRegion, MOSH::ParamFromValueOp op);

  /// Interface-based lowering for externally-defined ops.
  LogicalResult lowerInterface(MGP::LowerableToMGPInterface op);

private:
  /// Lowers a single Operation via type-based dispatch.
  LogicalResult lowerOp(Operation &op, mlir::Region *parentRegion);

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
  for (DeviceRefAttr device : getAllDevicesUsedInGraph(graph))
    loweringState.registerTypeBuilder(device);
  return success();
}

CompileDiagnosticOrSuccess MOGraphLowering::setupBufferBuildersForAllDevices() {
  DeviceRefAttr hostDevice =
      DeviceRefAttr::getDefault(loweringState.getModelOp()->getContext());

  auto makeBufferBuilder =
      [&](DeviceRefAttr targetDevice) -> CompileDiagnosticOrSuccess {
    MGP::MGPBufferBuilder *hostBufferBuilder;
    auto bufferBuilder = std::make_unique<MGP::MGPBufferBuilder>(targetDevice);
    if (hostDevice == targetDevice) {
      hostBufferBuilder = bufferBuilder.get();
    } else {
      hostBufferBuilder = static_cast<MGP::MGPBufferBuilder *>(
          loweringState.getBufferBuilder(hostDevice));
      ASSERT_STREAM(
          hostBufferBuilder,
          << "no buffer builder has been created for the host device '"
          << hostDevice << "'");
    }
    auto inChain =
        cast<TypedValue<MEF::ChainType>>(loweringState.getInitChain());
    auto chain = bufferBuilder->setupForModel(
        loweringState.getModelOp(), hostBufferBuilder,
        loweringState.getInitOpBuilder(), loweringState.getGraph()->getLoc(),
        inChain);
    loweringState.setInitChain(chain);
    loweringState.registerBufferBuilder(targetDevice, std::move(bufferBuilder));
    return success();
  };

  // Make the host buffer builder first.
  if (auto err = makeBufferBuilder(hostDevice); err.isError())
    return err;

  // Add the buffer builders for other supported devices.
  for (auto device : getAllDevicesUsedInGraph(loweringState.getGraph())) {
    if (device != hostDevice) {
      if (auto err = makeBufferBuilder(device); err.isError())
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
      /*isDeviceGraph=*/graph.getIsDeviceGraph(),
      /*extraAttrs=*/extraAttrs);

  loweringState.setModelOp(modelOp);

  // Maintain the cached module symbol table: the new `mgp.model` supersedes the
  // `mo.graph` under the same symbol name. Drop the graph's now-stale entry
  // before registering the model so the name stays free; the graph itself stays
  // in the IR (its body is lowered into the model below) until it is erased at
  // the end of `run()`. Without this, the cached `SymbolTableAnalysis` -- never
  // invalidated by the pass manager -- would keep a dangling entry for the
  // erased graph and omit the model.
  mlir::SymbolTable &moduleSymbolTable = loweringState.getModuleSymbolTable();
  moduleSymbolTable.remove(graph);
  moduleSymbolTable.insert(modelOp);

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
  auto mgpOutputTensorTy = loweringState.getTypeBuilder(tensorDeviceRef)
                               .tensorType(outputTensorSpecTy);
  return MGP::ListType::get(ctx, mgpOutputTensorTy);
}

LogicalResult MOGraphLowering::cloneTrivialOp(Operation *trivialOp) {
  Operation *cloned = loweringState.getExecuteOpBuilder().clone(*trivialOp);
  for (auto [from, to] :
       llvm::zip_equal(trivialOp->getResults(), cloned->getResults()))
    loweringState.emplaceExecuteLoweredValue(from, to);

  return success();
}

LogicalResult MOGraphLowering::lower(MO::OutputOp output) {
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

LogicalResult MOGraphLowering::lower(MO::YieldOp yield) {
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

LogicalResult MOGraphLowering::lower(MO::WhileConditionOp condition) {
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

LogicalResult MOGraphLowering::lowerParam(Region *parentRegion,
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

LogicalResult MOGraphLowering::lowerParam(Region *parentRegion,
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
MOGraphLowering::lowerParam(Region *parentRegion,
                            MOSH::ParamFromValueOp paramFromValueOp) {
  auto input =
      loweringState.lookupExecuteLoweredValue(paramFromValueOp.getValue());

  if (auto err = loweringState.lowerParamDecl(
          parentRegion, paramFromValueOp.getParamDecl(), input))
    return paramFromValueOp->emitOpError(err.getError());

  return success();
}

LogicalResult MOGraphLowering::lowerInterface(MGP::LowerableToMGPInterface op) {
  return op.lower(loweringState);
}

LogicalResult MOGraphLowering::lower(MO::IndexToTensorOp toTensorOp) {
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

  // The input to mo.index.to_tensor is an scalar<si64>. Cast to index to match
  // the expectation of mgp.set_with_index.
  auto &builder = loweringState.getExecuteOpBuilder();
  loweredIndexVal =
      MGP::dimensionToIndex(builder, toTensorOp.getLoc(), loweredIndexVal);

  auto outChain = opLoweringState.bufferSetWithIndex(
      outputStorage.buffer, {loweredIndexVal},
      toTensorOp.getResult().getType());

  // Set up output tensor.
  opLoweringState.createOutputs(outChain, *loweredOutputStorageOr);
  return success();
}

LogicalResult MOGraphLowering::lower(MO::Shape::ToTensorOp toTensorOp) {
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

LogicalResult MOGraphLowering::lower(MO::Shape::FromTensorOp fromTensorOp) {
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
    if (MOSH::ShapeAttr::getStaticDim(resDim))
      continue;

    // If the input dimension is static, and the result is dynamic, we create
    // a constant runtime dimension.
    if (std::optional<int64_t> intDim = MOSH::ShapeAttr::getStaticDim(inDim)) {
      auto constOp = opLoweringState.create<mlir::index::ConstantOp>(*intDim);
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

LogicalResult MOGraphLowering::lower(MO::RebindOp rebindOp) {
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

      auto asDimension = MGP::indexToDimension(
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
      M::Driver::getPreferredAlignment(device.getLabel().str()));

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

LogicalResult MOGraphLowering::lower(MO::ConstantOp constOp) {
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

LogicalResult MOGraphLowering::lower(MO::ConstantExternalOp constExternalOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, constExternalOp);
  OpBuilder &builder = opLoweringState.builder;
  Location loc = constExternalOp.getLoc();

  TypedValue<MEF::ChainType> chain = loweringState.getExecuteChain();
  TypedValue<MO::TensorType> opResult = constExternalOp.getResult();
  uint64_t sizeInBytes = opResult.getType().getDtype().getSizeInBytes() *
                         (*opResult.getType().getNumElements());
  uint64_t alignment = constExternalOp.getAlign();
  bool hasAlias = constExternalOp.getHasAlias();
  bool isPlaceholder = constExternalOp.getIsPlaceholder();

  // Create mgp.string.create op for the weight name.
  auto mojoStringType = MGP::OpaqueType::get(
      builder.getContext(), builder.getStringAttr("mojo_string"), {});
  auto name = MGP::StringCreateOp::create(
      builder, loc, mojoStringType,
      builder.getStringAttr(constExternalOp.getName()));

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

LogicalResult MOGraphLowering::lower(MO::ConstantScalarOp constOp) {
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

static LogicalResult lowerMoggSliceImpl(MGP::MOOpLoweringState &opLoweringState,
                                        MOGG::SliceOp sliceOp, Value chain) {
  MO::TensorType outputType = sliceOp.getType();

  // Create the output tensor spec from the output type.
  FailureOr<TypedValue<MGP::TensorSpecType>> outputSpecOr =
      opLoweringState.createTensorSpec(outputType);
  if (failed(outputSpecOr))
    return failure();

  // Get the lowered input tensor and start tensor.
  Value loweredInput = opLoweringState.loweringState.lookupExecuteLoweredValue(
      sliceOp.getInput());
  Value loweredStart = opLoweringState.loweringState.lookupExecuteLoweredValue(
      sliceOp.getStart());

  // Emit a single mgp.tensor.slice op. The Mojo primitive computes the sliced
  // dimension, stride, and start offset at runtime from the operands.
  auto inputDevice =
      cast<MGP::DeviceTensorType>(loweredInput.getType()).getDevice();
  TypedValue<MGP::DeviceContextType> deviceContext =
      opLoweringState.loweringState.getExecuteDevice(inputDevice);
  Value slicedTensor = opLoweringState.create<MGP::TensorSliceOp>(
      loweredInput, *outputSpecOr, loweredStart, deviceContext);

  // Associate the sliced tensor with the MOGG slice result.
  Value inChain = opLoweringState.getGuardChain(chain);
  opLoweringState.loweringState.emplaceExecuteLoweredBufferValue(
      sliceOp.getResult(), slicedTensor, inChain);

  return success();
}

LogicalResult MOGraphLowering::lower(MO::DebugPrintOp printOp) {
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

LogicalResult MOGraphLowering::lower(MO::DebugTensorPrintOp printOp) {
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
      M::Driver::getPreferredAlignment(hostDevice.getLabel().str()));
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

LogicalResult MOGraphLowering::lower(MO::TransferOp transferOp) {
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

LogicalResult MOGraphLowering::lower(MO::BufferTransferOp transferOp) {
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

LogicalResult MOGraphLowering::lower(MO::GuardOp guardOp) {
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

LogicalResult MOGraphLowering::lower(MO::ChainCreateOp chainCreateOp) {
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

LogicalResult MOGraphLowering::lower(MO::MutableStoreOp mutableStoreOp) {
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

/// Computes the little-endian bit pattern of a scalar `initValue` attribute,
/// zero-extended to 64 bits, for use as the `valueBits` of an
/// mgp.buffer.memset. Handles the single-element elements attribute the Python
/// API emits (so any dtype is represented exactly), as well as scalar
/// integer/bool and float attributes. Returns failure for any other attribute
/// kind, which the verifier is expected to have already rejected.
static FailureOr<uint64_t> initValueBits(Attribute initValue) {
  // The Python API encodes the scalar as a single-element elements attribute.
  // `M::ArrayElementsAttr` only supports iteration as `Attribute`, so pull out
  // the element as a scalar attribute and reuse the scalar handling below.
  if (auto elts = dyn_cast<ElementsAttr>(initValue)) {
    if (elts.getNumElements() != 1)
      return failure();
    initValue = *elts.value_begin<Attribute>();
  }
  // A BoolAttr is an IntegerAttr with an i1 type, so the integer branch also
  // covers booleans.
  if (auto intAttr = dyn_cast<IntegerAttr>(initValue))
    return intAttr.getValue().zextOrTrunc(64).getZExtValue();
  if (auto floatAttr = dyn_cast<FloatAttr>(initValue))
    return floatAttr.getValue().bitcastToAPInt().zextOrTrunc(64).getZExtValue();
  return failure();
}

LogicalResult MOGraphLowering::lower(MO::BufferCreateOp bufferCreateOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, bufferCreateOp);

  Type resultType = bufferCreateOp.getResult().getType();
  auto resTensorType = MO::castFromTensorLike(resultType);

  Attribute initValue = bufferCreateOp.getInitValueAttr();
  if (!initValue) {
    // Uninitialized per-call scratch buffer: allocate a fresh buffer in the
    // 'execute' block on every invocation (the original behavior).
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

  // Persistent, once-initialized state buffer. The buffer is allocated and
  // filled with `initValue` exactly once in the 'init' region, cached in the
  // context, then retrieved (as the same allocation) on every 'execute' call so
  // that kernel mutations persist across calls.

  // Persistent buffers must have a statically known byte size so that a single
  // memset can fill them during init.
  std::optional<int64_t> numElements = resTensorType.getNumElements();
  if (!numElements)
    return bufferCreateOp->emitOpError(
        "persistent buffer with an 'initValue' must have a statically known "
        "shape (dynamic dimensions are unsupported)");
  uint64_t sizeOfDtype = resTensorType.getDtype().getSizeInBytes();
  uint64_t sizeInBytes = sizeOfDtype * static_cast<uint64_t>(*numElements);

  FailureOr<uint64_t> valueBitsOr = initValueBits(initValue);
  if (failed(valueBitsOr))
    return bufferCreateOp->emitOpError(
        "unsupported 'initValue' attribute kind");
  uint64_t valueBits = *valueBitsOr;

  DeviceRefAttr device = MO::getDeviceOrDefault(resTensorType);
  MGP::BufferBuilder *bufferBuilder = loweringState.getBufferBuilder(device);

  Location loc = bufferCreateOp.getLoc();

  //===--- Init region: alloc -> memset -> cache. ---===//
  OpBuilder &initBuilder = loweringState.getInitOpBuilder();
  TypedValue<MGP::ContextType> initContext =
      loweringState.getInitMgpContext().getResult();
  TypedValue<MGP::DeviceContextType> initDevCtx =
      loweringState.getInitDevice(device);
  auto initChain =
      cast<TypedValue<MEF::ChainType>>(loweringState.getInitChain());

  // Match the alignment logic in MOOpLoweringState::createBuffer.
  uint64_t alignment =
      M::Driver::getPreferredAlignment(device.getLabel().str());
  if (device.getLabel() == kCPULabel)
    alignment = std::gcd(alignment, opLoweringState.minimumCpuAlignment);

  Value sizeVal = mlir::index::ConstantOp::create(
      initBuilder, loc, static_cast<int64_t>(sizeInBytes));

  Value allocBuffer = bufferBuilder->bufferAlloc(
      initBuilder, loc, initContext, /*inChain=*/initChain, /*size=*/sizeVal,
      /*deviceContext=*/initDevCtx, alignment);

  auto memsetChain = MGP::BufferMemsetOp::create(
                         initBuilder, loc, /*inChain=*/initChain,
                         /*buffer=*/allocBuffer, /*deviceContext=*/initDevCtx,
                         /*valueBits=*/valueBits, /*elemSize=*/sizeOfDtype)
                         .getOutChain();

  auto [cacheChain, slot] = bufferBuilder->cache(
      initBuilder, loc, initContext, /*inChain=*/memsetChain,
      /*contextCreateOp=*/loweringState.getInitMgpContext(),
      /*buffer=*/allocBuffer);
  loweringState.setInitChain(cacheChain);

  //===--- Execute region: retrieve the cached buffer, build the tensor.
  //---===//
  OpBuilder &execBuilder = loweringState.getExecuteOpBuilder();
  Value execChain = loweringState.getCurrentExecuteChain();
  execChain = opLoweringState.getGuardChain(execChain);

  Value cachedBuffer = bufferBuilder->getCached(
      execBuilder, loc, /*context=*/loweringState.getExecuteMgpContext(),
      /*inChain=*/cast<TypedValue<MEF::ChainType>>(execChain), slot,
      /*expectedBufferType=*/allocBuffer.getType());

  auto tensorSpecOr = opLoweringState.createTensorSpec(resTensorType);
  if (failed(tensorSpecOr))
    return failure();

  opLoweringState.createOutputTensor(bufferCreateOp.getResult(), execChain,
                                     cachedBuffer, *tensorSpecOr);

  return success();
}

LogicalResult MOGraphLowering::lower(MO::MutableLoadOp mutableLoadOp) {
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

LogicalResult MOGraphLowering::lower(MO::IfOp ifOp) {
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

LogicalResult MOGraphLowering::lower(MO::WhileOp whileOp) {
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

// Drops the param-decl-ref name from every `!mogg.dim_<name>` block arg of a
// MAP-ABI kernel stub (and its function signature), turning them into anonymous
// `!mogg.dim_`. A symbolic dim name on the stub boundary would block kernel
// deduplication, so `MOGGToMGP` strips it here to match the anonymous casts
// feeding the re-homed `mogg.call.stub`. Changing a block-arg type updates its
// in-body uses automatically. Idempotent, so a stub shared by several regions
// is safe to revisit.
static void stripStubDimNames(MOGG::KernelStubOp stub) {
  MLIRContext *ctx = stub.getContext();
  SmallVector<Type> argTypes;
  bool changed = false;
  for (BlockArgument arg : stub.getBodyBlock().getArguments()) {
    Type type = arg.getType();
    if (auto dim = dyn_cast<MOGG::DimType>(type); dim && dim.getName()) {
      type = MOGG::DimType::get(ctx, /*name=*/{});
      arg.setType(type);
      changed = true;
    }
    argTypes.push_back(type);
  }
  if (!changed)
    return;
  FunctionType funcType = stub.getFunctionType();
  stub.setFunctionTypeAttr(
      TypeAttr::get(FunctionType::get(ctx, argTypes, funcType.getResults())));
}

// Lowers a new-path `mogg._kernel` into a `mogg.generic.compiled.region`. This
// is deliberately NOT the legacy `mogg.kernel` path
// (MOGGKernelOpLoweringState::lowerKernelOp), which is left untouched: here the
// kernel stub is already outlined (OutlineKernelsIntoFunctions), so we only
// materialize the region operands (input buffers, freshly-allocated output
// buffers, dim values, device context), build the region, unpack + cast each
// async block arg to the handle the stub expects, and re-home the existing
// `mogg.call.stub`.
//
// TODO(GEX-3965): buffer allocation / context / dim materialization should move
// to dedicated pre-MGP passes, leaving this a genuinely thin lowering.
LogicalResult MOGraphLowering::lower(MOGG::KernelRegionOp op) {
  MGP::MOOpLoweringState st(loweringState, op);
  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  DeviceRefAttr device = MO::getDeviceOrDefaultAny(op);

  // The body block args are, in order: one `!mogg.pointer_` per input, then per
  // output, then the `!mogg.dim_<name>`s, then one trailing `!mogg.context`.
  Block &kernelBody = op.getBody().front();
  unsigned numInputs = op.getInputs().size();
  unsigned numOutputs = op.getOutputs().size();
  unsigned numArgs = kernelBody.getNumArguments();
  if (numArgs < numInputs + numOutputs + 1)
    return op.emitOpError("MOGGToMGP: malformed mogg._kernel body");

  Value inChain = st.getGuardChain(loweringState.getCurrentExecuteChain());

  // Region operands (plain sync values) + their in/out signature, built in the
  // exact body-block-arg order so operand i lines up with kernel block arg i.
  SmallVector<Value> operands;
  SmallVector<InOutSignatureAttr::InOutSemantics> semantics;

  // Inputs: `!mo.tensor` operands lower to `!mgp.buffer`.
  for (Value buffer : st.getLoweredOperands()) {
    operands.push_back(buffer);
    semantics.push_back(InOutSignatureAttr::kIn);
  }

  // Outputs: freshly-allocated buffers (externally-allocated, DPS).
  FailureOr<SmallVector<MGP::MOOutputStorage>> storageOr =
      st.createOutputStorage(inChain);
  if (failed(storageOr))
    return failure();
  SmallVector<MGP::MOOutputStorage> storage = *storageOr;
  assert(storage.size() == op.getNumResults() &&
         "expected one output storage per mogg._kernel result");
  for (const MGP::MOOutputStorage &out : storage) {
    operands.push_back(out.buffer);
    semantics.push_back(InOutSignatureAttr::kOut);
  }

  // Dims: materialize each trailing `!mogg.dim_<name>` block arg from its
  // param.
  for (unsigned i = numInputs + numOutputs; i + 1 < numArgs; ++i) {
    auto dimType = dyn_cast<MOGG::DimType>(kernelBody.getArgument(i).getType());
    if (!dimType || !dimType.getName())
      return op.emitOpError("MOGGToMGP: mogg._kernel dim arg lacks a name");
    ErrorOr<Value> dimValue = loweringState.lowerExpression(
        op->getParentRegion(), loc, dimType.getName());
    if (dimValue.isError())
      return op.emitOpError("MOGGToMGP: ") << dimValue.getError();
    operands.push_back(dimValue.takeValue());
    semantics.push_back(InOutSignatureAttr::kNone);
  }

  // Context.
  operands.push_back(loweringState.getExecuteDevice(device));
  semantics.push_back(InOutSignatureAttr::kNone);

  // Build the region; result 0 is the output chain.
  OpBuilder &builder = st.builder;
  auto region = MOGG::GenericCompiledRegionOp::create(
      builder, loc, TypeRange{MEF::ChainType::get(ctx)}, operands,
      /*outputParamDecls=*/KGEN::ParamDeclArrayAttr::get(ctx, {}),
      InOutSignatureAttr::get(ctx, semantics),
      /*fusionInfo=*/MOGG::FusionInfoAttr{},
      /*isAsync=*/BoolAttr::get(ctx, false),
      /*constantResources=*/ArrayAttr::get(ctx, {}),
      /*canFuseIntoOther=*/BoolAttr::get(ctx, !MO::isOnHost(op)),
      /*canBeFusedInto=*/BoolAttr::get(ctx, true),
      /*canDuplicate=*/BoolAttr::get(ctx, false),
      /*isStructuredKernel=*/BoolAttr::get(ctx, true));
  region.setDeviceRef(device);

  // Region body: one `!mogg.async<T>` block arg per operand, same order.
  Block *regionBody = &region.getBody().emplaceBlock();
  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  for (Value operand : operands) {
    argTypes.push_back(MOGG::AsyncType::get(operand.getType()));
    argLocs.push_back(loc);
  }
  regionBody->addArguments(argTypes, argLocs);

  // Unpack + cast each async arg to the handle the stub expects, then re-home
  // the already-outlined `mogg.call.stub`.
  builder.setInsertionPointToStart(regionBody);
  SmallVector<Value> stubArgs;
  for (unsigned i = 0; i < numArgs; ++i) {
    Value unpacked =
        MOGG::AsyncUnpackOp::create(builder, loc, regionBody->getArgument(i));
    // `mogg._rebind` reconciles the unpacked MGP handle (`!mgp.buffer` /
    // `!mgp.device_context` / a dim value) with the stub's codegen handle
    // (`!mogg.pointer_` / `!mogg.context` / `!mogg.dim_`). For `!mogg.pointer_`
    // it emits real Mojo -- bridging the unpacked `OwnedByteBuffer` to the raw
    // opaque data pointer the MAP-ABI stub expects; the context/dim casts are
    // pass-throughs (the unpacked handles already have the right Mojo type).
    // debt: drop once `mogg.async.unpack` folds directly to the typed handle.
    // For a dim arg, cast to an ANONYMOUS `!mogg.dim_` (name dropped): the
    // symbolic name blocks kernel dedup, and the referenced stub's dim args are
    // stripped in lockstep below so the re-homed call still type-checks.
    Type handleType = kernelBody.getArgument(i).getType();
    if (isa<MOGG::DimType>(handleType))
      handleType = MOGG::DimType::get(ctx, /*name=*/{});
    Value handle = MOGG::RebindOp::create(builder, loc, handleType, unpacked);
    stubArgs.push_back(handle);
  }
  auto originalCall = cast<MOGG::CallStubOp>(&kernelBody.front());
  MOGG::CallStubOp::create(builder, loc, /*results=*/TypeRange{},
                           originalCall.getNameAttr(), stubArgs,
                           /*affinity_device_id=*/IntegerAttr{});
  // Strip the dim-arg names from the referenced stub so its signature matches
  // the anonymous casts above (see stripStubDimNames).
  if (MOGG::KernelStubOp stub = originalCall.getCallee())
    stripStubDimNames(stub);
  Value readyChain = MOGG::AsyncReadyOp::create(builder, loc);
  MOGG::OutputOp::create(builder, loc, ValueRange{readyChain});

  // Register the outputs against the graph, guarded by the region's out chain.
  builder.setInsertionPointAfter(region);
  st.createOutputs(region->getResult(0), storage);
  return success();
}

LogicalResult MOGraphLowering::lowerOp(Operation &op, Region *parentRegion) {
  if (llvm::isa_and_nonnull<mlir::index::IndexDialect>(op.getDialect()))
    return cloneTrivialOp(&op);

  // clang-format off
  return llvm::TypeSwitch<Operation *, LogicalResult>(&op)
      // Param ops need the enclosing region.
      .Case<KGEN::ParamDeclareOp, MOSH::ParamToValueOp,
            MOSH::ParamFromValueOp>(
          [this, parentRegion](auto op) { return lowerParam(parentRegion, op); })
      // Reshape/slice ops need a local MOOpLoweringState.
      .Case<MO::Static::ReshapeOp, MO::AddSingletonDimOp, MO::SplitDimOp,
            MO::MergeDimOp>([this](auto op) {
        MGP::MOOpLoweringState opLoweringState(loweringState, op);
        return lowerReshapeImpl(opLoweringState, op.getInput(), op.getResult(),
                                loweringState.getExecuteChain(), op.getType());
      })
      .Case<MOGG::SliceOp>([this](auto op) {
        MGP::MOOpLoweringState opLoweringState(loweringState, op);
        return lowerMoggSliceImpl(opLoweringState, op,
                                  loweringState.getExecuteChain());
      })
      // All other concrete op types dispatch to lower() overloads.
      .Case<MO::IndexToTensorOp, MO::Shape::ToTensorOp,
            MO::Shape::FromTensorOp, MO::RebindOp, MO::ConstantOp,
            MO::ConstantExternalOp, MO::ConstantScalarOp, MO::DebugPrintOp,
            MO::DebugTensorPrintOp, MO::TransferOp, MO::OutputOp, MO::IfOp,
            MO::WhileOp, MO::TensorBundleOp, MO::TensorUnbundleOp,
            MO::ParallelOp, MO::SequenceOp, MO::YieldOp, MO::WhileConditionOp,
            MO::GuardOp, MO::ChainCreateOp, MO::MutableStoreOp,
            MO::MutableLoadOp, MO::BufferCreateOp, MO::BufferTransferOp,
            MO::CallOp>([this](auto op) { return lower(op); })
      // Advanced-fusion new-path kernel (gated by MAX_GC_USE_ADV_FUSION).
      .Case<MOGG::KernelRegionOp>([this](auto op) { return lower(op); })
      // Interface-based lowering for externally-defined ops.
      .Case<MGP::LowerableToMGPInterface>(
          [this](auto op) { return lowerInterface(op); })
      .Default([](Operation *op) -> LogicalResult {
        return op->emitOpError("currently not supported");
      });
  // clang-format on
}

LogicalResult MOGraphLowering::lowerBlock(Block *block) {
  // Ensure the block is sorted with operands to operations appearing before
  // their use.
  if (failed(levelwiseTopologicalSort(block)))
    return failure();

  for (Operation &op : *block) {
    if (failed(lowerOp(op, block->getParent())))
      return failure();
  }
  return success();
}

LogicalResult MOGraphLowering::lower(MO::SequenceOp seq) {
  // Lower mo.sequence by inlining its body into the execute block, bound to a
  // per-device side-stream view of the device context. The region was a fusion
  // boundary (enforced at the MOGG stage) and a marker for which ops get the
  // side stream. Cross-stream ordering is expressed purely through buffer/
  // tensor guard chains -- never the running execute chain: the fork gates on
  // the guard chains of the body's inputs *and* its captured values, and the
  // body's outputs are re-guarded on the join chain so downstream consumers
  // wait for the side stream.
  OpBuilder &b = loweringState.getExecuteOpBuilder();
  Location loc = seq.getLoc();
  Block &body = seq.getBodyRegion().front();
  Value execContext = loweringState.getExecuteMgpContext();
  Value rootChain = loweringState.getExecuteChain();

  // Combine guard chains (dropping nulls); the root chain stands in for "ready
  // immediately" when there are no producers (e.g. model inputs).
  auto combineChains = [&](ArrayRef<Value> chains) -> Value {
    SmallVector<Value> nonNull;
    for (Value c : chains)
      if (c)
        nonNull.push_back(c);
    if (nonNull.empty())
      return rootChain;
    if (nonNull.size() == 1)
      return nonNull.front();
    return GRT::ChainCreateOp::create(b, loc, nonNull).getResult();
  };
  // Device of a value, or null for non-tensor (e.g. chain) values: those carry
  // no device and must not fabricate a default one.
  auto deviceOf = [](Value v) -> DeviceRefAttr {
    if (auto devTy = dyn_cast<MO::MOHasDeviceTypeInterface>(v.getType()))
      return MO::getDeviceOrDefault(devTy);
    return {};
  };
  // Guard-chain lookup mirroring the canonical pattern (see
  // lower(MO::TensorUnbundleOp)): a value may be guarded under its MO key or
  // its lowered key. Try the MO value first, then its lowered value.
  auto guardOf = [&](Value moVal) -> Value {
    if (Value g = loweringState.getChainGuardingValueOrNull(moVal))
      return g;
    if (Value lowered = loweringState.lookupExecuteLoweredValue(moVal))
      return loweringState.getChainGuardingValueOrNull(lowered);
    return {};
  };

  // Collect the unique accelerator devices the body touches. A replicated body
  // (e.g. the EP shared expert) spans every device, one side-stream view each.
  // streamId 0 (default stream) or a host-only body adds no binding.
  uint64_t streamId = seq.getStreamId();
  SmallVector<DeviceRefAttr> devices;
  auto addDevice = [&](Value v) {
    DeviceRefAttr dev = deviceOf(v);
    if (dev && !MO::isOnHost(v.getType()) && !llvm::is_contained(devices, dev))
      devices.push_back(dev);
  };
  for (Value in : seq.getInputs())
    addDevice(in);
  for (Value res : seq.getResults())
    addDevice(res);
  const bool bindStream = streamId != 0 && !devices.empty();

  // Values the body reads from the enclosing scope (everything that is NOT a
  // block argument). mo.sequence is intentionally not IsolatedFromAbove -- the
  // body may capture immutable weights/constants directly -- so to keep the
  // cross-stream sync complete we also gate the fork on the producer chains of
  // these captures. A read-only constant/weight has no guard chain (no-op); a
  // capture produced on the main stream contributes its chain, so the side
  // stream correctly waits for it.
  llvm::SetVector<Value> captured;
  mlir::getUsedValuesDefinedAbove(seq.getBodyRegion(), captured);

  // Fork: per device, create the side-stream view and a device_wait that makes
  // the side stream wait for the base-stream producers of that device's inputs
  // and captures.
  llvm::SmallDenseMap<DeviceRefAttr, TypedValue<MGP::DeviceContextType>>
      viewFor;
  llvm::SmallDenseMap<DeviceRefAttr, Value> forkChainFor;
  if (bindStream) {
    for (DeviceRefAttr dev : devices) {
      TypedValue<MGP::DeviceContextType> base =
          loweringState.getExecuteDevice(dev);
      auto view =
          MGP::DeviceContextSelectStreamOp::create(b, loc, base, streamId);
      viewFor[dev] = view;
      SmallVector<Value> inGuards;
      for (Value in : seq.getInputs())
        if (deviceOf(in) == dev)
          inGuards.push_back(guardOf(in));
      for (Value cap : captured)
        if (deviceOf(cap) == dev)
          inGuards.push_back(guardOf(cap));
      forkChainFor[dev] =
          MGP::DeviceWaitOp::create(b, loc, /*inChain=*/combineChains(inGuards),
                                    execContext, /*waitingDeviceContext=*/view,
                                    /*signalingDeviceContext=*/base,
                                    /*bDevice=*/dev)
              .getResult();
      loweringState.setExecuteDeviceOverride(dev, view);
    }
  }

  // Map each body block argument to its lowered input and guard the (region-
  // local) block argument on the device's fork chain, so the body's ops wait
  // for the fork. We guard the block argument -- not the shared input value --
  // so unrelated main-stream consumers of the same input are not serialized.
  for (auto [arg, operand] : llvm::zip(body.getArguments(), seq.getInputs())) {
    loweringState.emplaceExecuteLoweredValue(
        arg, loweringState.lookupExecuteLoweredValue(operand));
    Value guard =
        bindStream ? forkChainFor.lookup(deviceOf(operand)) : guardOf(operand);
    if (guard)
      loweringState.replaceChainGuard(arg, guard);
  }

  // Lower the body ops (excluding the yield terminator) in place; with the
  // overrides installed, getExecuteDevice routes them onto the side streams.
  if (failed(levelwiseTopologicalSort(&body)))
    return failure();
  for (Operation &op : body.without_terminator())
    if (failed(lowerOp(op, &seq.getBodyRegion())))
      return failure();

  // Join: per device, a device_wait that makes the base stream wait for the
  // side stream, gated on the guard chains of that device's body outputs.
  auto yieldOp = cast<MO::YieldOp>(body.getTerminator());
  llvm::SmallDenseMap<DeviceRefAttr, Value> joinChainFor;
  if (bindStream) {
    for (DeviceRefAttr dev : devices)
      loweringState.clearExecuteDeviceOverride(dev);
    for (DeviceRefAttr dev : devices) {
      TypedValue<MGP::DeviceContextType> base =
          loweringState.getExecuteDevice(dev);
      SmallVector<Value> outGuards;
      for (Value out : yieldOp.getOperands())
        if (deviceOf(out) == dev)
          outGuards.push_back(guardOf(out));
      joinChainFor[dev] = MGP::DeviceWaitOp::create(
                              b, loc, /*inChain=*/combineChains(outGuards),
                              execContext, /*waitingDeviceContext=*/base,
                              /*signalingDeviceContext=*/viewFor[dev],
                              /*bDevice=*/dev)
                              .getResult();
    }
  }

  // Map each result to its lowered yield value and guard the result on the
  // device's join chain, so downstream consumers wait for the side stream.
  for (auto [res, yielded] :
       llvm::zip_equal(seq.getResults(), yieldOp.getOperands())) {
    loweringState.emplaceExecuteLoweredValue(
        res, loweringState.lookupExecuteLoweredValue(yielded));
    Value guard =
        bindStream ? joinChainFor.lookup(deviceOf(res)) : guardOf(yielded);
    if (guard)
      loweringState.replaceChainGuard(res, guard);
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

/// Fuse a set of synchronous compiled regions into one, then convert the
/// synchronous call stubs to async dispatch with a join barrier.
///
/// After this function:
///   - The fused region contains CallStubAsyncOps, an AsyncJoinOp, and a
///     single AsyncReadyOp producing the output chain.
///   - The original regions in `groupRegions` have been erased.
///   - The fused region is inserted after `groupRegions.back()`.
static MOGG::GenericCompiledRegionOp
fuseAndConvertToAsync(mlir::MLIRContext *ctx, Location loc,
                      ArrayRef<Operation *> groupRegions) {
  MOGG::GenericCompiledRegionOp fusedRegion =
      MOGG::fuseAndReplaceCompiledRegions(/*allowDuplication=*/false,
                                          groupRegions);

  // fuseAndReplaceCompiledRegions leaves InOutSignatureAttr empty because it
  // normally runs after ABI lowering. We call it earlier, so downstream passes
  // still need a valid signature.
  SmallVector<InOutSignatureAttr::InOutSemantics> sig(
      fusedRegion->getNumOperands(), InOutSignatureAttr::kNone);
  fusedRegion.setInOutSignatureAttr(InOutSignatureAttr::get(ctx, sig));

  // Replace each synchronous mogg.call.stub with mogg.call.stub.async.
  // Propagate the optional $device attribute so the emitter can choose an
  // affinity-aware create_task form for GPU launches.
  Block &body = fusedRegion.getBody().front();
  SmallVector<Value> taskHandles;
  body.walk([&](MOGG::CallStubOp callStub) {
    // Forward the affinity device ordinal (set on GPU launches only by the
    // parallel orchestrator) straight from CallStubOp to CallStubAsyncOp.
    OpBuilder b(ctx);
    b.setInsertionPoint(callStub);
    MOGG::CallStubAsyncOp asyncCall = MOGG::CallStubAsyncOp::create(
        b, callStub.getLoc(), callStub.getName(), callStub.getInputs(),
        callStub.getAffinityDeviceIdAttr());
    taskHandles.push_back(asyncCall.getTaskHandle());
    callStub.erase();
  });

  // Insert the join barrier before the output terminator.
  Operation *outputOp = body.getTerminator();
  OpBuilder b(ctx);
  b.setInsertionPoint(outputOp);
  MOGG::AsyncJoinOp::create(b, loc, taskHandles);

  // Fusion cloned one AsyncReadyOp per original region. The first one (which
  // fuseAndReplaceCompiledRegions wired into the OutputOp) is the canonical
  // chain; erase the rest and move the survivor after the join.
  MOGG::AsyncReadyOp canonical;
  body.walk([&](MOGG::AsyncReadyOp readyOp) {
    if (!canonical) {
      canonical = readyOp;
      return;
    }
    readyOp.getResult().replaceAllUsesWith(canonical.getResult());
    readyOp.erase();
  });
  ASSERT_STREAM(canonical, << "expected at least one AsyncReadyOp in fused "
                              "parallel region body");
  canonical->moveBefore(outputOp);

  for (Operation *region : groupRegions)
    region->erase();

  return fusedRegion;
}

LogicalResult MOGraphLowering::lower(MO::ParallelOp parallelOp) {
  mlir::MLIRContext *ctx = parallelOp->getContext();
  Location loc = parallelOp.getLoc();

  Block &bodyBlock = parallelOp.getBodyRegion().front();
  size_t numLaunches = parallelOp.getNumLaunches();
  size_t numBundleGroups = parallelOp.getInputs().size();

  // The parallel op threads its chain explicitly: the in-chain enters the body
  // through a trailing block argument, the body's buffer ops thread it, and the
  // result is yielded as the terminator's last operand and becomes the op's
  // out-chain. The lowering is a pass-through: it maps the chain block arg to
  // the lowered in-chain and the out-chain to the lowered yielded chain.

  // Resolve per-launch MO-level values from each bundle input via the
  // lowering state (populated by lower(TensorBundleOp)).  The optional buffer
  // group comes directly from the parallel op's buffers() operand list.
  size_t numGroups = numBundleGroups;
  SmallVector<SmallVector<Value>> bundleElements(numGroups);

  for (size_t group = 0; group < numBundleGroups; ++group) {
    Value bundleVal = parallelOp.getInputs()[group];
    auto moElements = loweringState.lookupBundleElements(bundleVal);
    for (size_t launch = 0; launch < numLaunches; ++launch)
      bundleElements[group].push_back(moElements[launch]);
  }

  // Single body walk: collect kernels and bundled.expand ops together, and
  // reject any other op the lowering can't handle. The verifier accepts a
  // few preserved ops as transient intermediate IR (see ParallelOp::verify),
  // but by lowering time only kernels and bundled.expand should remain --
  // anything else indicates a missing cleanup pass or an unsupported pattern.
  SmallVector<MOGG::KernelOp> kernelOps;
  SmallVector<MO::BundledExpandOp> expandOps;
  for (Operation &op : bodyBlock.without_terminator()) {
    if (auto kernel = dyn_cast<MOGG::KernelOp>(&op))
      kernelOps.push_back(kernel);
    else if (auto expand = dyn_cast<MO::BundledExpandOp>(&op))
      expandOps.push_back(expand);
    else
      return op.emitOpError()
             << "lowering of this op inside mo.parallel is not yet "
                "supported; see GEX-3701";
  }
  if (kernelOps.empty())
    return parallelOp.emitOpError("body must contain at least one kernel");

  // For each kernel in the body, clone it once per launch, lower each clone
  // via the standard lowerKernelOp path, then fuse and convert to async.
  //
  // Fusion is per kernel group (not across groups) to avoid creating dataflow
  // cycles: a later group's buffer allocs may depend on the earlier group's
  // chain output.
  //
  // Bundled kernels (collectives like allreduce) are cloned per launch like
  // regular kernels, but each clone is N-expanded to receive inputs from ALL
  // launches so the collective can see every device's data.
  //
  // perLaunchCloneResults tracks inter-kernel dependencies: for each original
  // kernel result in the parallel body, we record the per-launch clone result
  // so that subsequent kernel clones can resolve their operands via IRMapping.
  // Clone ops are kept alive until after all kernels are processed because
  // lookupExecuteLoweredValue needs the clone result SSA values.
  DenseMap<Value, SmallVector<Value>> perLaunchCloneResults;
  SmallVector<Operation *> allClones;

  // The in-chain enters the body through the trailing chain block argument
  // (present iff the parallel is chained); map it to the lowered in-chain so
  // the body's buffer ops thread the real predecessor.
  Value loweredInChain;
  if (parallelOp.hasChain()) {
    loweredInChain =
        loweringState.lookupExecuteLoweredValue(parallelOp.getInChain());
    ASSERT_STREAM(loweredInChain,
                  << "parallel in-chain not lowered before parallel op");
  }
  // Each kernel group's fused completion chain, keyed by the group's MO-level
  // result values, so the yielded out-chain can be resolved to it.
  DenseMap<Value, Value> resultToFusedChain;

  // Resolve mo.bundled.expand ops: each result maps to the per-launch element
  // of the input's source bundle.  The expand op is a no-op placeholder that
  // makes N-expansion explicit in the IR.
  for (MO::BundledExpandOp expandOp : expandOps) {
    Value input = expandOp.getInput();
    // Find which bundle group this input comes from.
    for (size_t group = 0; group < numBundleGroups; ++group) {
      if (input != bodyBlock.getArgument(group))
        continue;
      for (size_t i = 0; i < expandOp.getNumResults(); ++i) {
        SmallVector<Value> &cloneResults =
            perLaunchCloneResults[expandOp.getResult(i)];
        cloneResults.resize(numLaunches);
        // Expand result i always resolves to launch i's value, regardless
        // of which per-launch clone is being built.
        for (size_t launch = 0; launch < numLaunches; ++launch)
          cloneResults[launch] = bundleElements[group][i];
      }
      break;
    }
  }

  for (MOGG::KernelOp kernelOp : kernelOps) {
    SmallVector<Operation *> kernelClones;
    SmallVector<Operation *> groupRegions;

    for (size_t launch = 0; launch < numLaunches; ++launch) {
      IRMapping mapper;
      for (size_t group = 0; group < numGroups; ++group)
        mapper.map(bodyBlock.getArgument(group), bundleElements[group][launch]);
      // Trailing chain block arg (chained parallels): map it to the MO-level
      // in-chain so the cloned body op carries an MO chain operand (as the old
      // captured-chain form did); lowerKernelOp resolves its lowered value.
      if (parallelOp.hasChain())
        mapper.map(bodyBlock.getArgument(numBundleGroups),
                   parallelOp.getInChain());
      for (auto &[origResult, cloneResults] : perLaunchCloneResults)
        mapper.map(origResult, cloneResults[launch]);

      OpBuilder cloneBuilder(ctx);
      cloneBuilder.setInsertionPointAfter(parallelOp);
      Operation *clonedOp = cloneBuilder.clone(*kernelOp, mapper);
      auto clonedKernel = cast<MOGG::KernelOp>(clonedOp);

      // The parallel body uses the first input's device as a representative.
      // For launches on other devices, update the clone's device attribute
      // and result types so that lowerKernelOp uses the correct device
      // context and allocates output buffers on the right device.
      // Per-launch device comes from the `launchDevices` attribute on
      // `mo.parallel`.
      DeviceRefAttr launchDevice = parallelOp.getLaunchDeviceList()[launch];
      if (launchDevice != MO::getDeviceOrDefaultAny(clonedKernel)) {
        clonedKernel.setDeviceRef(launchDevice);
        clonedKernel.walk([&](MOGG::DeviceContextPlaceholderOp placeholder) {
          placeholder.setDeviceAttr(launchDevice);
        });
      }
      for (OpResult result : clonedKernel->getResults()) {
        if (auto resTy = dyn_cast<MO::TensorType>(result.getType()))
          if (resTy.getDeviceRef() != launchDevice)
            result.setType(resTy.cloneWith(launchDevice));
      }

      MGP::MOGGKernelOpLoweringState perLaunchState(clonedKernel,
                                                    loweringState);
      if (failed(perLaunchState.lowerKernelOp(clonedKernel)))
        return failure();

      // For GPU launches, stamp the integer device ordinal onto every
      // CallStubOp so that fuseAndConvertToAsync can forward it unchanged
      // to the async variant.  The ordinal is stored as a plain I64Attr
      // rather than a DeviceRefAttr because mogg-emit-mojo internally runs
      // stripDeviceIdsAndDynamicShapes (which zeroes DeviceRefAttr IDs)
      // before emitting Mojo text.
      MOGG::GenericCompiledRegionOp launchRegion =
          perLaunchState.getMGPRegion();
      if (launchDevice && launchDevice.getLabel() == M::kGPULabel) {
        auto affinityId =
            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                   static_cast<int64_t>(launchDevice.getId()));
        launchRegion->walk([&](MOGG::CallStubOp callStub) {
          callStub.setAffinityDeviceIdAttr(affinityId);
        });
      }

      groupRegions.push_back(launchRegion);
      kernelClones.push_back(clonedOp);
    }

    for (auto [idx, origResult] : llvm::enumerate(kernelOp->getResults())) {
      SmallVector<Value> &cloneResults = perLaunchCloneResults[origResult];
      for (size_t launch = 0; launch < numLaunches; ++launch)
        cloneResults.push_back(kernelClones[launch]->getResult(idx));
    }
    allClones.append(kernelClones);

    MOGG::GenericCompiledRegionOp fusedRegion =
        fuseAndConvertToAsync(ctx, loc, groupRegions);
    Value fusedChain = fusedRegion.getResult(0);
    // Record this group's completion chain against its MO-level results, so the
    // yielded out-chain (produced by the body's buffer op) resolves to it.
    for (Value res : kernelOp->getResults())
      resultToFusedChain[res] = fusedChain;

    // lowerKernelOp registered chain guards against individual compiled region
    // chains, which fuseAndConvertToAsync just erased. Repoint them to the
    // fused region's chain so that subsequent kernel groups see valid guards.
    for (Operation *clone : kernelClones) {
      for (Value cloneRes : clone->getResults()) {
        loweringState.replaceChainGuard(cloneRes, fusedChain);
        Value lowered = loweringState.lookupExecuteLoweredValue(cloneRes);
        if (lowered)
          loweringState.replaceChainGuard(lowered, fusedChain);
      }
    }
  }

  // Register per-element clone results as the bundle elements for each
  // parallel result so that lower(TensorUnbundleOp) can resolve them.
  auto yieldOp = cast<MO::YieldOp>(bodyBlock.getTerminator());
  size_t resultIdx = 0;
  for (Value yieldOperand : yieldOp.getOperands()) {
    if (!isa<MO::TensorType>(yieldOperand.getType()))
      continue;
    Value bundleResult = parallelOp.getResult(resultIdx++);
    SmallVector<Value> &cloneResults = perLaunchCloneResults[yieldOperand];

    // Register clone results as the bundle elements for this result so that
    // lower(TensorUnbundleOp) can resolve them via lookupBundleElements.
    loweringState.emplaceBundleElements(bundleResult,
                                        SmallVector<Value>(cloneResults));
  }

  // The out-chain is the lowered completion of the body's yielded chain (the
  // buffer ops' threaded chain). The IR states this explicitly, so the lowering
  // is a pass-through: no spine advance, no buffer-op detection.
  if (parallelOp.hasChain()) {
    Value yieldedChain = yieldOp.getOperands().back();
    ASSERT_STREAM(
        isa<MO::ChainType>(yieldedChain.getType()),
        << "chained parallel must yield a trailing chain (enforced by verify)");
    auto it = resultToFusedChain.find(yieldedChain);
    // If the body merely passes the in-chain through (no buffer op produced
    // it), the out-chain is the in-chain itself.
    Value outChain =
        it != resultToFusedChain.end() ? it->second : loweredInChain;

    loweringState.emplaceExecuteLoweredValue(parallelOp.getOutChain(),
                                             outChain);
  }

  // Clone shells must stay alive: their result Values are referenced by
  // loweredValues (and potentially bundleElements in downstream passes).
  // Move them into the parallel op's body region so lowerBlock's
  // graph-level iterator doesn't re-encounter them.  They become dead
  // code destroyed when the MO graph region is replaced by MGP.
  Block &bodyFront = parallelOp.getBodyRegion().front();
  for (Operation *clone : allClones)
    clone->moveBefore(&bodyFront, bodyFront.begin());

  return success();
}

LogicalResult MOGraphLowering::lower(MO::TensorBundleOp bundleOp) {
  SmallVector<Value> moElements = llvm::to_vector(bundleOp.getInputs());
  loweringState.emplaceBundleElements(bundleOp.getResult(),
                                      std::move(moElements));
  return success();
}

LogicalResult MOGraphLowering::lower(MO::TensorUnbundleOp unbundleOp) {
  auto moElements = loweringState.lookupBundleElements(unbundleOp.getInput());
  for (auto [result, moElem] :
       llvm::zip_equal(unbundleOp.getOutputs(), moElements)) {
    Value lowered = loweringState.lookupExecuteLoweredValue(moElem);
    Value chain = loweringState.getChainGuardingValueOrNull(moElem);
    if (!chain)
      chain = loweringState.getChainGuardingValueOrNull(lowered);
    if (chain)
      loweringState.emplaceExecuteLoweredBufferValue(result, lowered, chain);
    else
      loweringState.emplaceExecuteLoweredValue(result, lowered);
  }
  return success();
}

LogicalResult MOGraphLowering::lower(MO::CallOp callOp) {
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
// MOGGToMGPPass
//===----------------------------------------------------------------------===//

namespace {
struct MOGGToMGPPass : public MGP::impl::MOGGToMGPBase<MOGGToMGPPass> {
  using MOGGToMGPBase::MOGGToMGPBase;

  void runOnOperation() override;

  void getDependentDialects(::mlir::DialectRegistry &registry) const override;
};
} // namespace

void MOGGToMGPPass::getDependentDialects(DialectRegistry &registry) const {
  MOGGToMGPBase::getDependentDialects(registry);
}

void MOGGToMGPPass::runOnOperation() {
  ModuleOp theModule = getOperation();

  // Kernel stubs are created at module scope during lowering. Register them
  // through the module's `SymbolTableAnalysis` so the cached table stays
  // consistent for later passes -- the analysis is never invalidated by the
  // pass manager, so symbols added via a raw builder would otherwise be
  // invisible to consumers that read the cached table. We keep it up to date
  // here, hence we can mark it preserved.
  SymbolTable &moduleSymbolTable =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

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
                               minimumCpuAlignment.getValue(),
                               moduleSymbolTable)
                   .run()))
      return signalPassFailure();

    lastProcessed = 0;
  }
}
