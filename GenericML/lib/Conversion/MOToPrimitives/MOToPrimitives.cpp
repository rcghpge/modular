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
#include "GenericML/Interfaces/MOLoweringInterface.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MODialect/MOOps.h"
#include "GraphRT/BEFDialect/BEFOps.h"
#include "GraphRT/GRTDialect/GRTOps.h"
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
class MOGraphLowering : public MGP::MOGraphLoweringState {
public:
  MOGraphLowering(M::MO::GraphOp graph, TypeConverter &typeConverter,
                  llvm::StringRef compiledFrameworkLabel)
      : MOGraphLoweringState(graph, typeConverter, compiledFrameworkLabel) {}

  /// Lower the mo.graph to mgp.model.
  LogicalResult run();

  LogicalResult createModelOp();

  /// Lowers the mo.output op to the bef.output for the final results.
  LogicalResult lowerOutput(MO::OutputOp output);

  /// Lowers mo.constant op.
  LogicalResult lowerConstant(MO::ConstantOp constOp);

  // Lower a generic operation to a KGEN invoke.
  LogicalResult lowerOperation(mlir::Operation *op);
};

} // namespace

//===----------------------------------------------------------------------===//
// Creation of the init/execute invokables
//===----------------------------------------------------------------------===//
template <typename T>
static MGP::CpuTensorSpecType createCpuTensorSpec(const T &tensor) {
  // TODO: Check for dynamic shape
  ArrayRef<int64_t> shape = tensor.getShape();
  // Convert scalars to rank 1 tensors
  // make a local value for one so it doesn't disappear as an rvalue
  // when creating the ArrayRef
  int64_t one = 1;
  if (tensor.hasRank() && tensor.getRank() == 0)
    shape = ArrayRef<int64_t>(one);
  // Output tensor type.
  return MGP::CpuTensorSpecType::get(shape, tensor.getElementType());
}

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
  common.modelOp =
      b.create<MGP::ModelOp>(graph.getLoc(),
                             /*sym_name=*/graph.getNameAttr(),
                             /*frameworkName=*/common.compiledFrameworkLabel,
                             /*inputTypes=*/inputTypes,
                             /*outputTypes=*/outputTypes,
                             /*inputArgLocs=*/inputLocs);

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
  execute.mgpContext = executeBlock->getArgument(1);

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

LogicalResult MOGraphLowering::lowerOperation(mlir::Operation *op) {
  auto loc = op->getLoc();

  // Operands to pass to the operation. Input tensors are expanded to buffers
  // and specs in IR those are passed in.
  SmallVector<Value> operands;
  for (Value operand : op->getOperands()) {
    // Extract the buffer and spec from the MGP tensor.
    mlir::Value mgpTensor = execute.loweredTensors.lookup(operand);

    auto extract =
        execute.builder.create<MGP::CpuTensorExtractOp>(loc, mgpTensor);

    // Add the expanded buffer and spec as arguments to the operation.
    operands.push_back(extract.getBufferResult());
    operands.push_back(extract.getTensorSpecResult());
  }

  // We can't create the tensors until after the operation as they are
  // immutable.
  SmallVector<
      std::pair<MGP::CpuBufferAllocStaticOp, MGP::CpuTensorSpecCreateOp>>
      outputBufferAndSpec;

  // Allocate the output buffer for each result and add as argument.
  for (Type t : op->getResultTypes()) {
    auto outSpecType = createCpuTensorSpec(t.cast<MO::TensorType>());
    Optional<uint64_t> bufferSize = outSpecType.getSizeInBytes();
    // Fail for dynamic shapes for now.
    if (!bufferSize)
      return failure();

    // Allocate the buffer for the output.
    auto bufferOp = execute.builder.create<MGP::CpuBufferAllocStaticOp>(
        loc, execute.mgpContext, *bufferSize);

    // Create an MGP tensor to match the MO output tensor.
    auto specOp =
        execute.builder.create<MGP::CpuTensorSpecCreateOp>(loc, outSpecType);

    // Add the output buffer and spec as inputs to the operation.
    operands.push_back(bufferOp.getResult());
    operands.push_back(specOp.getResult());

    // Track the buffer and spec so we can create the immutable tensor after the
    // operation.
    outputBufferAndSpec.push_back({bufferOp, specOp});
  }

  const std::string opName = "mop.kgen." + op->getName().stripDialect().str();

  // Create a call to a custom invoke function we can put in the BEF.
  mlir::OperationState opState{loc, opName};

  // Add the execute chain as the first argument.
  operands.insert(operands.begin(), execute.chain);
  opState.addOperands(operands);

  // The return type is just the chain, other outputs are passed as argument by
  // their buffer pointers.
  opState.addTypes({execute.chain.getType()});

  // Add it to the MLIR.
  mlir::Operation *invokeOp = execute.builder.create(opState);
  execute.chain = invokeOp->getResult(0);

  size_t outputIndex = 0;

  // Allocate the output tensors this must be done after as the tensor is read
  // only.
  for (auto &[buffer, spec] : outputBufferAndSpec) {
    // Get the type of the spec and infer the type of the tensor from that.
    auto ty = cast<MGP::CpuTensorSpecType>(spec->getResultTypes()[0]);
    auto tensorType = MGP::CpuTensorType::get(execute.builder.getContext(), ty);

    // Create an immutable tensor
    auto tensorCreateOp = execute.builder.create<MGP::CpuTensorCreateOp>(
        loc, tensorType, execute.chain, buffer, spec);

    // Update the MO tensor to now refer to this mgp tensor.
    execute.loweredTensors.try_emplace(op->getResult(outputIndex),
                                       tensorCreateOp.getResult());
    outputIndex++;
  }

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
    Type type;
    size_t blobAlign = resource.getRawHandle().getBlob()->getDataAlignment();
    if (blobAlign != 0)
      // NOTE: Unclear if blobs can end up with a zero alignment since
      // the parser assert fails with such. So this is probably always true.
      // If the resource has an alignment, we must capture it within the
      // lowered resource attribute type.
      type = MGP::AlignedBytesType::get(execute.builder.getContext(),
                                        resource.size(),
                                        static_cast<uint64_t>(blobAlign));
    else
      // Otherwise the type can just be an array of uint8_ts.
      type =
          ArrayType::get(resource.size(),
                         execute.builder.getIntegerType(8, /*isSigned=*/false));
    value = DenseResourceElementsAttr::get(type, resource.getRawHandle());
  }
  auto bufferOp = execute.builder.create<MGP::CpuBufferConstantOp>(loc, value);

  // We create the tensor spec op.
  auto outSpecType = createCpuTensorSpec(cast<mlir::ShapedType>(convertedType));
  auto specOp =
      execute.builder.create<MGP::CpuTensorSpecCreateOp>(loc, outSpecType);

  // We create the tensor op itself using the spec and the buffer. The result
  // of this op is what we remember.
  auto tensorType =
      MGP::CpuTensorType::get(execute.builder.getContext(), outSpecType);
  auto tensorCreateOp = execute.builder.create<MGP::CpuTensorCreateOp>(
      loc, tensorType, execute.chain, bufferOp, specOp);
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

    // If we still have a MO op just use the generic lowering.
    // TODO-maybe?: Add as a trait.
    if (isa<MO::MODialect>(op.getDialect())) {
      if (failed(lowerOperation(&op)))
        return failure();
      continue;
    }

    // Implementers of the lowering interface define their own logic.
    if (auto lowerable = dyn_cast<MGP::MOLoweringInterface>(op)) {
      if (failed(lowerable.lower(*this)))
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
    return MGP::CpuTensorType::get(&ctx, createCpuTensorSpec(t));
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
