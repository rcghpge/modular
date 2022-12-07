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
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MODialect/MOOps.h"
#include "GraphRT/BEFDialect/BEFOps.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Transforms/DialectConversion.h"

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
  MOGraphLowering(MO::GraphOp graph, TypeConverter &typeConverter)
      : graph(graph), typeConverter(typeConverter), init(graph.getContext()),
        execute(graph.getContext()) {}

  /// Lower the mo.graph to mgp.model.
  LogicalResult run();

private:
  /// This is common state used by both the init and execute phases.
  struct CommonState {
    explicit CommonState() = default;

    /// This is the top-level mgp.model operation we're building.
    MGP::ModelOp modelOp;

    CommonState(const CommonState &) = delete;
    void operator=(const CommonState &) = delete;
  };

  /// This is the state used when building an invokable for the 'init' phase.
  struct InitState {
    InitState(MLIRContext *ctx) : builder(ctx) {}

    /// This is the current chain value being added to.
    Value chain;
    /// This op creates and returns the overall MGP context.
    MGP::ContextCreateOp mgpContext;

    /// This is a builder appending operations to the end of the invokable.
    OpBuilder builder;

    InitState(const InitState &) = delete;
    void operator=(const InitState &) = delete;
  };

  /// This is stated used when building an invokable for the 'execute' phase.
  struct ExecuteState {
    ExecuteState(MLIRContext *ctx) : builder(ctx) {}

    /// This is the original input chain to the invokable, and is ready
    /// when the argument tensors are ready.
    Value invokableInputChain;

    /// This chain is ready when the current global state is ready for the
    /// next possibly mutating op.
    Value chain;

    /// This is the MGP context passed in as the second argument.
    Value mgpContext;

    /// Keeps track of what mo.tensor SSA values are lowered to.
    DenseMap<Value, Value> loweredTensors;

    /// This is a builder appending operations to the end of the invokable.
    OpBuilder builder;

    ExecuteState(const ExecuteState &) = delete;
    void operator=(const ExecuteState &) = delete;
  };

  LogicalResult createModelOp();

  /// Lowers the mo.output op to the bef.output for the final results.
  LogicalResult lowerOutput(MO::OutputOp output);

  /// Lowers mo.constant op.
  LogicalResult lowerConstant(MO::ConstantOp constOp);

  //===-----------------------------------------------------------------===//
  // Lowering state
  //===-----------------------------------------------------------------===//

  /// This is the graph being lowered.
  MO::GraphOp graph;
  /// This helps lower tensor types.
  TypeConverter &typeConverter;

  /// This keeps track of the init and execute invokables that we're building.
  CommonState common;
  InitState init;
  ExecuteState execute;
};

} // namespace

//===----------------------------------------------------------------------===//
// Creation of the init/execute invokables
//===----------------------------------------------------------------------===//

// Create the mgp.model and the regions for the init/execute phases.
LogicalResult MOGraphLowering::createModelOp() {
  OpBuilder b(graph);

  // Convert the tensor types for the model signature over.
  SmallVector<Type> inputTypes, outputTypes;
  if (failed(
          typeConverter.convertTypes(graph.getArgumentTypes(), inputTypes)) ||
      failed(typeConverter.convertTypes(graph.getResultTypes(), outputTypes))) {
    return graph.emitError("failed to convert mo graph input/output types");
  }

  SmallVector<Location> inputLocs;
  for (BlockArgument arg : graph.getArguments())
    inputLocs.push_back(arg.getLoc());

  // Create the mgp.model and the regions for the init/execute phases.
  b.setInsertionPoint(graph);
  StringAttr frameworkName = b.getStringAttr("mgp");
  common.modelOp =
      b.create<MGP::ModelOp>(graph.getLoc(), graph.getNameAttr(), frameworkName,
                             inputTypes, outputTypes, inputLocs);

  // Initialize the init state.
  init.builder.setInsertionPointToEnd(common.modelOp.getInitBlock());
  // Get the chain and context values for the body.
  // This field tracks the current output chain as we lower into the init body.
  init.chain = init.builder.create<GRT::ChainCreateOp>(graph.getLoc());

  // Create the mgp.context.create op without any slots.
  init.mgpContext =
      init.builder.create<MGP::ContextCreateOp>(graph.getLoc(), 0);

  // Initialize the execute state.
  Block *executeBlock = common.modelOp.getExecuteBlock();
  execute.builder.setInsertionPointToEnd(executeBlock);

  // Extract chain and MGP context from arguments
  execute.invokableInputChain = executeBlock->getArgument(0);

  // This field tracks the current output chain as we lower into the execute
  // body.
  execute.chain = execute.invokableInputChain;

  return success();
}

LogicalResult MOGraphLowering::lowerOutput(MO::OutputOp output) {
  // Init phase lowering.

  // Return the final chain.
  init.builder.create<BEF::OutputOp>(
      output.getLoc(), ArrayRef<Value>{init.chain, init.mgpContext});

  // Execute phase lowering.

  // Figure out what values to return.  We always return a chain.
  SmallVector<Value> operands;
  operands.push_back(execute.chain);

  // TODO: merge chains as needed
  for (Value operand : output.getOperands())
    operands.push_back(execute.loweredTensors.lookup(operand));

  // Finally, we can create the bef.output op.
  execute.builder.create<BEF::OutputOp>(output.getLoc(), operands);
  return success();
}

LogicalResult MOGraphLowering::lowerConstant(MO::ConstantOp constOp) {
  Location loc = constOp.getLoc();

  // Init phase lowering.

  // Execute phase lowering.

  MO::TensorType moTensorType = constOp.getType();
  Type convertedType = typeConverter.convertType(moTensorType);
  if (!convertedType)
    return constOp->emitError("unsupported type ") << constOp.getType();

  // MO stores constants either in line as ArrayElementsAttr or as a resource.
  // We canonicalize the value's type to ArrayType and build a buffer op.
  ElementsAttr value;
  if (auto elementsAttr = dyn_cast<ArrayElementsAttr>(constOp.getValue())) {
    auto type =
        ArrayType::get(elementsAttr.getRawData().size(),
                       execute.builder.getIntegerType(8, /*isSigned=*/false));
    value = ArrayElementsAttr::get(elementsAttr.getRawData(), type);
  } else {
    // Value must be a resource.
    auto resource = cast<DenseResourceElementsAttr>(constOp.getValue());
    auto type = ArrayType::get(
        resource.size(), execute.builder.getIntegerType(8, /*isSigned=*/false));
    value = DenseResourceElementsAttr::get(type, resource.getRawHandle());
  }
  auto bufferOp = execute.builder.create<MGP::CpuBufferConstantOp>(loc, value);

  // We create the tensor spec op.
  auto shapedType = cast<mlir::ShapedType>(convertedType);
  auto outSpecType = MGP::CpuTensorSpecType::get(shapedType.getShape(),
                                                 shapedType.getElementType());
  auto specOp =
      execute.builder.create<MGP::CpuTensorSpecCreateOp>(loc, outSpecType);

  // We create the tensor op itself using the spec and the buffer. The result
  // of this op is what we remember.
  auto tensorType =
      MGP::CpuTensorType::get(execute.builder.getContext(), outSpecType);
  auto tensorCreateOp = execute.builder.create<MGP::CpuTensorCreateOp>(
      loc, tensorType, execute.invokableInputChain, specOp, bufferOp);
  execute.loweredTensors.try_emplace(constOp->getResult(0),
                                     tensorCreateOp->getResult(0));
  return success();
}

LogicalResult MOGraphLowering::run() {
  // Create the mgp.model and the regions for the init/execute phases.
  if (failed(createModelOp()))
    return failure();

  for (auto [graphArg, exeArg] : llvm::zip(
           graph.getArguments(),
           common.modelOp.getExecuteBlock()->getArguments().drop_front(2))) {
    // If the argument has no uses then just ignore it.
    if (graphArg.use_empty())
      continue;

    execute.loweredTensors.try_emplace(graphArg, exeArg);
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
    return MGP::CpuTensorType::get(
        &ctx, MGP::CpuTensorSpecType::get(t.getShape(), t.getElementType()));
  });

  // Convert each mo.graph into a mgp.model.
  for (auto graphOp :
       llvm::make_early_inc_range(theModule.getOps<MO::GraphOp>())) {
    if (failed(MOGraphLowering(graphOp, converter).run()))
      return signalPassFailure();
  }
}
