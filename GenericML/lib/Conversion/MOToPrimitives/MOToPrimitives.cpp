//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Converts MO ops to MGP and BEF ops.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Conversion/MOToPrimitives.h"
#include "GenericML/Conversion/Support.h"
#include "GenericML/Interfaces/MOLoweringInterface.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MODialect/MOOps.h"
#include "GenericML/MOGGDialect/MOGGDialect.h"
#include "GenericML/MOGGDialect/MOGGOps.h"
#include "GraphRT/BEFDialect/BEFOps.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "Support/AlignedAlloc.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Transforms/DialectConversion.h"

#include <numeric>

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_MOTOPRIMITIVES
#include "GenericML/Conversion/MOToPrimitives.h.inc"

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
  MOGraphLowering(M::MO::GraphOp graph, TypeConverter &typeConverter,
                  llvm::StringRef compiledFrameworkLabel)
      : loweringState(graph, typeConverter, compiledFrameworkLabel) {}

  /// Lower the mo.graph to mgp.model.
  LogicalResult run();

  LogicalResult createModelOp();

  /// Lowers the mo.output op to the bef.output for the final results.
  LogicalResult lowerOutput(MO::OutputOp output);

  /// Lowers mo.constant op.
  LogicalResult lowerConstant(MO::ConstantOp constOp);

private:
  M::MGP::MOGraphLoweringState loweringState;
};

} // namespace

//===----------------------------------------------------------------------===//
// Creation of the init/execute invokables
//===----------------------------------------------------------------------===//

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
    return graph.emitError("failed to convert mo graph input/output types");
  }

  SmallVector<Location> inputLocs;
  for (BlockArgument arg : graph.getArguments())
    inputLocs.push_back(arg.getLoc());

  // Create the mgp.model and the regions for the init/execute phases.
  b.setInsertionPoint(graph);
  loweringState.setCommonModelOp(b.create<MGP::ModelOp>(
      graph.getLoc(),
      /*sym_name=*/graph.getNameAttr(),
      /*frameworkName=*/loweringState.getCompiledFrameworkLabel(),
      /*inputTypes=*/inputTypes,
      /*outputTypes=*/outputTypes,
      /*inputArgLocs=*/inputLocs));

  // Initialize the init state.
  loweringState.getInitOpBuilder().setInsertionPointToEnd(
      loweringState.getCommonModelOp().getInitBlock());
  // Get the chain and context values for the body.
  // This field tracks the current output chain as we lower into the init body.
  loweringState.setInitChain(
      loweringState.getInitOpBuilder().create<GRT::ChainCreateOp>(
          graph.getLoc()));

  loweringState.setInitMgpContext(
      loweringState.getInitOpBuilder().create<MGP::ContextCreateOp>(
          graph.getLoc(), 1));

  // The slot in the mgp context reserved for the JIT context which loads the
  // compiled binaries.
  loweringState.setJitCtxSlot(0);

  // Build up the MLIR state for the global JIT context.
  mlir::OperationState opState{graph.getLoc(), "mop.create_jit_context"};
  opState.addAttribute("slot",
                       loweringState.getInitOpBuilder().getUI32IntegerAttr(
                           loweringState.getJitCtxSlot()));
  opState.addOperands(
      {loweringState.getInitChain(), loweringState.getInitMgpContext()});
  opState.addTypes({loweringState.getInitChain().getType()});

  // Create the context building op.
  mlir::Operation *jit = loweringState.getInitOpBuilder().create(opState);

  loweringState.setJitCtx(jit);
  loweringState.setInitChain(jit->getResult(0));

  // Initialize the execute state.
  Block *executeBlock = loweringState.getCommonModelOp().getExecuteBlock();
  loweringState.getExecuteOpBuilder().setInsertionPointToEnd(executeBlock);

  // Extract chain and MGP context from arguments
  loweringState.setExecuteInvokableInputChain(executeBlock->getArgument(0));
  loweringState.setExecuteMgpContext(executeBlock->getArgument(1));

  // This field tracks the current output chain as we lower into the execute
  // body.
  loweringState.setExecuteChain(loweringState.getExecuteInvokableInputChain());

  return success();
}

LogicalResult MOGraphLowering::lowerOutput(MO::OutputOp output) {
  // Init phase lowering.

  // Return the final chain.
  loweringState.getInitOpBuilder().create<BEF::OutputOp>(
      output.getLoc(), ArrayRef<Value>{loweringState.getInitChain(),
                                       loweringState.getInitMgpContext()});

  // Execute phase lowering.

  // Figure out what values to return.  We always return a chain.
  SmallVector<Value> operands;
  operands.push_back(loweringState.getExecuteChain());

  // TODO: merge chains as needed
  for (Value operand : output.getOperands())
    operands.push_back(loweringState.lookupExecuteLoweredTensor(operand));

  // Finally, we can create the bef.output op.
  loweringState.getExecuteOpBuilder().create<BEF::OutputOp>(output.getLoc(),
                                                            operands);
  return success();
}

LogicalResult MOGraphLowering::lowerConstant(MO::ConstantOp constOp) {
  Location loc = constOp.getLoc();
  MO::TensorType moTensorType = constOp.getType();
  Type convertedType =
      loweringState.getTypeConverter().convertType(moTensorType);
  if (!convertedType)
    return constOp->emitError("unsupported type ") << constOp.getType();

  // Init phase lowering.

  // Execute phase lowering.
  // We'll represent large constant attributes as DenseResourceElementsAttrs,
  // and small as AlignedBytes.
  ElementsAttr value;
  if (auto elementsAttr = dyn_cast<ArrayElementsAttr>(constOp.getValue())) {
    size_t size = elementsAttr.getRawData().size();
    // TODO(#6838): Alignment probably needs to be a pass attribute.
    // TODO(#6912): For now, leave alignment as 1. It appears 'aligned'
    // attributes are not making it into the runtime respecting arbitrary
    // alignment constraints.
    size_t alignment = kPreferredMemoryAlignment;
    Type type = MGP::AlignedBytesType::get(
        loweringState.getExecuteOpBuilder().getContext(),
        static_cast<uint64_t>(size), static_cast<uint64_t>(alignment));
    std::string name =
        "const_" + std::to_string(loweringState.getExecuteNextConstId());
    const char *data =
        reinterpret_cast<const char *>(elementsAttr.getRawData().data());
    auto resourceManager = DenseResourceElementsHandle::getManagerInterface(
        loweringState.getExecuteOpBuilder().getContext());
    value = getAttrForTensorData(type, name, ArrayRef<char>(data, size),
                                 resourceManager, alignment);
  } else if (auto resource =
                 dyn_cast<DenseResourceElementsAttr>(constOp.getValue())) {
    size_t size = resource.size();
    size_t alignment = resource.getRawHandle().getBlob()->getDataAlignment();
    assert(alignment > 0 && "resource blob must have non-zero alignment");
    Type type = MGP::AlignedBytesType::get(
        loweringState.getExecuteOpBuilder().getContext(),
        static_cast<uint64_t>(
            size * resource.getElementType().getIntOrFloatBitWidth() /
            CHAR_BIT),
        static_cast<uint64_t>(alignment));
    value = DenseResourceElementsAttr::get(type, resource.getRawHandle());
  } else {
    return constOp->emitError("unsupported constant attribute type");
  }

  auto bufferOp =
      loweringState.getExecuteOpBuilder().create<MGP::CpuBufferConstantOp>(
          loc, value);

  // We create the tensor spec op.
  auto outSpecType =
      MGP::createCpuTensorSpec(cast<mlir::ShapedType>(convertedType));
  auto specOp =
      loweringState.getExecuteOpBuilder().create<MGP::CpuTensorSpecCreateOp>(
          loc, outSpecType);

  // We create the tensor op itself using the spec and the buffer. The result
  // of this op is what we remember.
  auto tensorType = MGP::CpuTensorType::get(
      loweringState.getExecuteOpBuilder().getContext(), outSpecType);
  auto tensorCreateOp =
      loweringState.getExecuteOpBuilder().create<MGP::CpuTensorCreateOp>(
          loc, tensorType, loweringState.getExecuteChain(), bufferOp, specOp);
  loweringState.emplaceExecuteLoweredTensor(constOp->getResult(0),
                                            tensorCreateOp->getResult(0));
  return success();
}

LogicalResult MOGraphLowering::run() {
  MO::GraphOp &graph = loweringState.getGraph();

  // Create the mgp.model and the regions for the init/execute phases.
  if (failed(createModelOp()))
    return failure();

  for (auto [graphArg, exeArg] :
       llvm::zip(graph.getArguments(), loweringState.getCommonModelOp()
                                           .getExecuteBlock()
                                           ->getArguments()
                                           .drop_front(2))) {
    // If the argument has no uses then just ignore it.
    if (graphArg.use_empty())
      continue;

    loweringState.emplaceExecuteLoweredTensor(graphArg, exeArg);
  }

  // Lower all of the operations in the graph.
  for (Operation &op : *graph.getBodyBlock()) {
    if (auto constOp = dyn_cast<MO::ConstantOp>(op)) {
      if (failed(lowerConstant(constOp)))
        return failure();
      continue;
    }

    // Lower mo.output to bef.output trivially.
    if (auto output = dyn_cast<MO::OutputOp>(op)) {
      if (failed(lowerOutput(output)))
        return failure();
      continue;
    }

    // Implementers of the lowering interface define their own logic.
    if (auto lowerable = dyn_cast<MGP::MOLoweringInterface>(op)) {
      if (failed(lowerable.lower(loweringState)))
        return failure();
      continue;
    }

    return op.emitError("op currently not supported");
  }

  // Ok, we're done with the original mo.graph, so remove it.
  graph.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// MOToPrimitivesPass
//===----------------------------------------------------------------------===//

namespace {
struct MOToPrimitivesPass
    : public M::MGP::impl::MOToPrimitivesBase<MOToPrimitivesPass> {
  using MOToPrimitivesBase::MOToPrimitivesBase;

  void runOnOperation() override;
};
} // namespace

void MOToPrimitivesPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  auto &ctx = getContext();

  TypeConverter converter;
  converter.addConversion([&ctx](MO::TensorType t) -> Optional<Type> {
    // TODO: this may fail for some element types.
    return MGP::CpuTensorType::get(&ctx, MGP::createCpuTensorSpec(t));
  });

  // Convert each mo.graph into a mgp.model.
  for (auto graphOp :
       llvm::make_early_inc_range(theModule.getOps<MO::GraphOp>())) {
    if (failed(MOGraphLowering(graphOp, converter,
                               compiledFrameworkLabel.getValue())
                   .run()))
      return signalPassFailure();
  }
}
