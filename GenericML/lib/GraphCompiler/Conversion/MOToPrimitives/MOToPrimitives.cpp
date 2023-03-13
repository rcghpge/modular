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
#include "GraphRT/GRTDialect/GRTOps.h"
#include "GraphRT/MEFDialect/MEFOps.h"
#include "Support/Compiler/OperationUtils.h"
#include "mlir/IR/DialectResourceBlobManager.h"

#include <numeric>

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_MOTOPRIMITIVES
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

//===----------------------------------------------------------------------===//
// MOGraphLoweringState implementation
//===----------------------------------------------------------------------===//

SmallVector<Value>
MGP::MOGraphLoweringState::lookupExecuteDynDims(MO::ShapeAttr shapeAttr) {
  SmallVector<Value> runtimeDims;
  for (TypedAttr dim : shapeAttr.getValues())
    if (auto dimRef = dyn_cast<KGEN::ParamDeclRefAttr>(dim))
      runtimeDims.emplace_back(lookupExecuteLoweredParam(dimRef.getName()));
  return runtimeDims;
}

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
                  size_t minimumCpuAlignment)
      : loweringState(graph, typeConverter, compiledFrameworkLabel,
                      minimumCpuAlignment) {}

  /// Lower the mo.graph to mgp.model.
  LogicalResult run();

  LogicalResult createModelOp();

  /// Lowers the mo.output op to the mef.output for the final results.
  LogicalResult lowerOutput(MO::OutputOp output);

  /// Lowers the mo.yield to the mef.output for the if block results.
  LogicalResult lowerYield(MO::YieldOp output);

  /// Lowers mo.constant op.
  LogicalResult lowerConstant(MO::ConstantOp constOp);

  /// Lowers mo.reshape op.
  LogicalResult lowerReshape(MO::ReshapeOp reshapeOp);

  /// Lowers mo.static.reshape op.
  LogicalResult lowerStaticReshape(MO::Static::ReshapeOp reshapeOp);

  /// Lowers mo.if op.
  LogicalResult lowerIf(MO::IfOp ifOp);

private:
  /// Lowers a Block
  LogicalResult lowerBlock(mlir::Block *block);

  MGP::MOGraphLoweringState loweringState;
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
  loweringState.setModelOp(b.create<MGP::ModelOp>(
      graph.getLoc(),
      /*sym_name=*/graph.getNameAttr(),
      /*frameworkName=*/loweringState.getCompiledFrameworkLabel(),
      /*inputTypes=*/inputTypes,
      /*outputTypes=*/outputTypes,
      /*inputArgLocs=*/inputLocs));

  // Initialize the init state.
  loweringState.getInitOpBuilder().setInsertionPointToEnd(
      loweringState.getModelOp().getInitBlock());
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
  Block *executeBlock = loweringState.getModelOp().getExecuteBlock();
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
  loweringState.getInitOpBuilder().create<MEF::OutputOp>(
      output.getLoc(), ArrayRef<Value>{loweringState.getInitChain(),
                                       loweringState.getInitMgpContext()});

  // Execute phase lowering.

  // Figure out what values to return.  We always return a chain.
  SmallVector<Value> operands;
  operands.push_back(loweringState.getExecuteInvokableInputChain());

  for (Value operand : output.getOperands())
    operands.push_back(loweringState.lookupExecuteLoweredTensor(operand));

  // Finally, we can create the mef.output op.
  loweringState.getExecuteOpBuilder().create<MEF::OutputOp>(output.getLoc(),
                                                            operands);
  return success();
}

LogicalResult MOGraphLowering::lowerYield(MO::YieldOp output) {
  // Execute phase lowering.

  // Figure out what values to return. We always return the main chain
  SmallVector<Value> operands;
  operands.push_back(loweringState.getExecuteChain());
  for (Value operand : output.getOperands())
    operands.push_back(loweringState.lookupExecuteLoweredTensor(operand));

  // Finally, we can create the mef.output op.
  loweringState.getExecuteOpBuilder().create<MEF::OutputOp>(output.getLoc(),
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
  size_t minAlignment = loweringState.getMinimumCpuAlignment();
  if (auto elementsAttr = dyn_cast<ArrayElementsAttr>(constOp.getValue())) {
    size_t size = elementsAttr.getRawData().size();
    Type type = MGP::AlignedBytesType::get(
        loweringState.getExecuteOpBuilder().getContext(),
        static_cast<uint64_t>(size), static_cast<uint64_t>(minAlignment));
    std::string name =
        "const_" + std::to_string(loweringState.getExecuteNextConstId());
    const char *data =
        reinterpret_cast<const char *>(elementsAttr.getRawData().data());
    auto resourceManager = DenseResourceElementsHandle::getManagerInterface(
        loweringState.getExecuteOpBuilder().getContext());
    value = getAttrForTensorData(
        type, name, ArrayRef<char>(data, size), resourceManager, minAlignment,
        /*forceOutOfLine=*/false, /*mustBeAligned=*/true);
  } else if (auto resource =
                 dyn_cast<DenseResourceElementsAttr>(constOp.getValue())) {
    size_t size = resource.size();
    size_t alignment = resource.getRawHandle().getBlob()->getDataAlignment();
    assert(alignment > 0 && "resource blob must have non-zero alignment");
    if (alignment < minAlignment)
      return constOp->emitError()
             << "dense_resource has alignment " << alignment
             << " however minimum alignment has been set as " << minAlignment
             << ".";
    // The element type might have a bitwidth of 1 - in which case we've turned
    // it into a set of ui8 (basically). If that's the case, use a single byte
    // for this. Because integer division rounds down, N / 8 will result in 0 if
    // N < 8, so we do the max(N/8, 1).
    int64_t elementTypeInBytes =
        std::max(resource.getElementType().getIntOrFloatBitWidth() / CHAR_BIT,
                 (unsigned)1);
    Type type = MGP::AlignedBytesType::get(
        loweringState.getExecuteOpBuilder().getContext(),
        static_cast<uint64_t>(size * elementTypeInBytes),
        static_cast<uint64_t>(alignment));
    value = DenseResourceElementsAttr::get(type, resource.getRawHandle());
  } else if (auto alignedBytes =
                 dyn_cast<AlignedBytesAttr>(constOp.getValue())) {
    Optional<size_t> optAlignment = alignedBytes.getAlign();
    if (optAlignment && *optAlignment < minAlignment)
      return constOp->emitError()
             << "m.aligned_bytes has alignment " << *optAlignment
             << " however minimum alignment has been set as " << minAlignment
             << ".";
    size_t alignment = optAlignment.value_or(minAlignment);
    value =
        AlignedBytesAttr::get(loweringState.getExecuteOpBuilder().getContext(),
                              alignment, alignedBytes.getData());
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
          loc, tensorType, loweringState.getExecuteInvokableInputChain(),
          bufferOp, specOp);
  loweringState.emplaceExecuteLoweredTensor(constOp->getResult(0),
                                            tensorCreateOp->getResult(0));
  return success();
}

LogicalResult
MOGraphLowering::lowerStaticReshape(MO::Static::ReshapeOp reshapeOp) {
  auto loc = reshapeOp.getLoc();
  OpBuilder exeBuilder = loweringState.getExecuteOpBuilder();

  auto mgpTensor =
      loweringState.lookupExecuteLoweredTensor(reshapeOp.getInput());
  auto extract =
      loweringState.getExecuteOpBuilder().create<MGP::CpuTensorExtractOp>(
          loc, mgpTensor);
  auto buffer = extract.getBufferResult();

  MO::TensorType resTy = reshapeOp.getType();
  if (!resTy.hasRank())
    return reshapeOp.emitError("Dynamic output rank not support");

  // Create the new spec from the dimensions.
  auto specTy = MGP::createCpuTensorSpec(resTy);
  SmallVector<Value> runtimeDims =
      loweringState.lookupExecuteDynDims(resTy.getTypedShapeAttr());
  auto specOp =
      exeBuilder.create<MGP::CpuTensorSpecCreateOp>(loc, specTy, runtimeDims);

  // Create an immutable tensor
  auto outTensorTy = exeBuilder.getType<MGP::CpuTensorType>(specTy);
  auto tensorCreateOp = exeBuilder.create<MGP::CpuTensorCreateOp>(
      loc, outTensorTy, loweringState.getExecuteInvokableInputChain(), buffer,
      specOp.getResult());

  // Update the MO tensor to now refer to this mgp tensor.
  loweringState.emplaceExecuteLoweredTensor(reshapeOp.getResult(),
                                            tensorCreateOp.getResult());

  return success();
}

LogicalResult MOGraphLowering::lowerReshape(MO::ReshapeOp reshapeOp) {
  auto loc = reshapeOp.getLoc();
  OpBuilder exeBuilder = loweringState.getExecuteOpBuilder();

  // get input tensor buffer
  auto mgpTensor =
      loweringState.lookupExecuteLoweredTensor(reshapeOp.getInput());
  auto extract = exeBuilder.create<MGP::CpuTensorExtractOp>(loc, mgpTensor);
  auto buffer = extract.getBufferResult();

  // get spec from output type (assume static input and output)
  auto outSpecTy = MGP::createCpuTensorSpec(reshapeOp.getType());
  Value outSpecValue;
  // TODO
  // 1. This isn't unified with `MOGG::ExtOp::lower` because we don't want
  // reshape to allocate extra output buffer. We should refactor this at some
  // point.
  // 2. This shape function is pure hackery, and should be removed as soon as we
  // can lower parametric shapes to the runtime.
  if (outSpecTy.hasStaticShape()) {
    if (!buffer.getType().hasStaticShape())
      return reshapeOp.emitError(
          "Dynamic input shape and static output shape not supported");

    auto specOp = exeBuilder.create<MGP::CpuTensorSpecCreateOp>(loc, outSpecTy);
    outSpecValue = specOp.getResult();
  } else {
    if (buffer.getType().hasStaticShape())
      return reshapeOp.emitError(
          "Static input shape and dynamic output shape not supported");

    OperationState opState{loc, "mop.shape.reshape"};
    Value targetShape =
        loweringState.lookupExecuteLoweredTensor(reshapeOp.getNewShape());
    // inputs are input specs
    opState.addOperands({extract.getTensorSpecResult(), targetShape});
    // The return is the new spec
    opState.addTypes(cast<Type>(outSpecTy));

    // Add it to the MLIR.
    Operation *invokeOp = exeBuilder.create(opState);
    outSpecValue = invokeOp->getResult(0);
  }

  // Create an immutable tensor
  auto outTensorTy = exeBuilder.getType<MGP::CpuTensorType>(outSpecTy);
  auto tensorCreateOp = exeBuilder.create<MGP::CpuTensorCreateOp>(
      loc, outTensorTy, loweringState.getExecuteInvokableInputChain(), buffer,
      outSpecValue);

  // Update the MO tensor to now refer to this mgp tensor.
  loweringState.emplaceExecuteLoweredTensor(reshapeOp.getResult(),
                                            tensorCreateOp.getResult());

  return success();
}

LogicalResult MOGraphLowering::lowerIf(MO::IfOp ifOp) {
  auto loc = ifOp->getLoc();
  OpBuilder &builder = loweringState.getExecuteOpBuilder();
  TypeConverter &converter = loweringState.getTypeConverter();
  Value mainChain = loweringState.getExecuteChain();

  // Lower Cond
  Value condTensor = loweringState.lookupExecuteLoweredTensor(ifOp.getCond());
  auto condExtract = builder.create<MGP::CpuTensorExtractOp>(loc, condTensor);

  auto cond = builder.create<MGP::ToGRTBool>(
      loc, loweringState.getExecuteInvokableInputChain(),
      condExtract.getBufferResult());

  SmallVector<Value> operands;
  operands.push_back(cond.getResult());
  // This is creating a Value semantic If Op
  // TODO: Change it to take buffers and specs? That breaks current lowering
  // assumptions, but we don't really want to hard limit the execution of the
  // body of the if on _all_ inputs being ready, which is what a value semantic
  // input list will do. Possibly better solution: split the tensors here,
  // recreate them inside the if body, let the standard lowering + tensor
  // creation elimination clean it up?

  SmallVector<Type> resultTypes;
  // Always return the main chain
  resultTypes.emplace_back(mainChain.getType());
  for (Type type : ifOp.getResultTypes())
    resultTypes.emplace_back(converter.convertType(type));

  // Create grt.if op
  auto outputIf = builder.create<GRT::IfOp>(loc, resultTypes, operands);

  auto lower_block = [&](Region &inputRegion,
                         Region &outputRegion) -> LogicalResult {
    // Enter a new Block
    Block &newBlock = outputRegion.emplaceBlock();
    builder.setInsertionPointToEnd(&newBlock);

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
  uniqueCaptures.erase(llvm::find(uniqueCaptures, Value(cond)));

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
    loweringState.emplaceExecuteLoweredTensor(ifOp.getResult(i - 1),
                                              outputIf.getResult(i));
  }

  // since we moved the insertion points for the then/else, reset to the main
  // block
  builder.setInsertionPointAfter(outputIf);

  return success();
}

LogicalResult MOGraphLowering::lowerBlock(Block *block) {
  for (Operation &op : *block) {
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

    // Lower mo.output to mef.output trivially.
    if (auto output = dyn_cast<MO::OutputOp>(op)) {
      if (failed(lowerOutput(output)))
        return failure();
      continue;
    }

    // Lower mo.if to grf.if, recursively lowering the body of
    // the then and else branches of if.
    if (auto ifOp = dyn_cast<MO::IfOp>(op)) {
      if (failed(lowerIf(ifOp)))
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

    return op.emitError("op currently not supported");
  }
  return success();
}

LogicalResult MOGraphLowering::run() {
  MO::GraphOp &graph = loweringState.getGraph();
  Location loc = graph->getLoc();

  // Create the mgp.model and the regions for the init/execute phases.
  if (failed(createModelOp()))
    return failure();

  // Enter the main body of the graph
  for (auto [graphArg, exeArg] :
       llvm::zip(graph.getArguments(), loweringState.getModelOp()
                                           .getExecuteBlock()
                                           ->getArguments()
                                           .drop_front(2))) {
    bool hasParams = false;
    if (auto tensorArg = dyn_cast<MO::TensorValue>(graphArg)) {
      MO::TensorType argType = tensorArg.getType();
      // We check if the tensor is parameterized.
      if (!argType.hasUnknownShape() && !argType.hasStaticShape()) {
        if (isa<KGEN::ParamDeclRefAttr>(argType.getShapeAttr()))
          return graph.emitError("Shape parameters are not supported");

        // At this point, we must be dealing with a statically ranked,
        // parameterized argument.
        hasParams = true;
        OpBuilder &builder = loweringState.getExecuteOpBuilder();
        auto extractOp = builder.create<MGP::CpuTensorExtractOp>(loc, exeArg);
        TypedValue<MGP::CpuTensorSpecType> spec =
            extractOp.getTensorSpecResult();
        auto dynDims = builder.create<MGP::CpuTensorSpecDimsDynamicOp>(
            loc, spec, spec.getType().getRawDims());

        // The semantics of CpuTensorSpecDimsDynamicOp are quirky.
        assert(dynDims->getNumResults() == argType.getRank());
        for (auto [dim, dimValue] :
             llvm::zip(argType.getDims(), dynDims->getResults())) {
          if (auto dimRef = dyn_cast<KGEN::ParamDeclRefAttr>(dim)) {
            StringAttr paramName = dimRef.getName();
            if (!loweringState.lookupExecuteLoweredParam(paramName))
              loweringState.emplaceExecuteLoweredParam(paramName, dimValue);
          }
        }
      }
    }

    // If the argument has no uses (and no parameters) then just ignore it.
    if (!hasParams && graphArg.use_empty())
      continue;

    loweringState.emplaceExecuteLoweredTensor(graphArg, exeArg);
  }

  // Lower all of the operations in the graph.
  if (failed(lowerBlock(graph.getBodyBlock())))
    return failure();
  // Ok, we're done with the original mo.graph, so remove it.
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

  // The lit kernels should already be fused so the registry is no longer
  // needed.
  for (MOGG::LitRegisterOp reg :
       llvm::make_early_inc_range(theModule.getOps<MOGG::LitRegisterOp>()))
    reg.erase();

  // Convert each mo.graph into a mgp.model.
  for (auto graphOp :
       llvm::make_early_inc_range(theModule.getOps<MO::GraphOp>())) {
    if (failed(MOGraphLowering(graphOp, converter,
                               compiledFrameworkLabel.getValue(),
                               minimumCpuAlignment.getValue())
                   .run()))
      return signalPassFailure();
  }
}
