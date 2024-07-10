//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GenericML/GraphCompiler/Interfaces/LoweringState.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOGGCompiler.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "GenericML/MGPDialect/MGPUtils.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/AssertStream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace M;
using namespace MOGG;

//===----------------------------------------------------------------------===//
// MOGG ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::lower(MGP::MOGraphLoweringState &loweringState) {
  ASSERT_STREAM(MO::isOnHost(*this), << "mogg.concat is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, *this);

  // Gather the input buffers.
  SmallVector<Value> loweredOperands = opLoweringState.getLoweredOperands();

  // Allocate the output buffer.
  FailureOr<SmallVector<MGP::MOOutputStorage>> loweredOutputsStorageOr =
      opLoweringState.createOutputStorage(loweringState.getExecuteChain());
  if (failed(loweredOutputsStorageOr))
    return failure();
  auto &loweredOutputsStorage = *loweredOutputsStorageOr;
  assert(loweredOutputsStorageOr->size() == 1);

  Value outChain = opLoweringState.bufferConcat(
      loweringState.getExecuteChain(), loweredOutputsStorage[0].buffer,
      loweredOperands, getResult().getType());

  // Create the output tensors, but only once the op has completed.
  opLoweringState.createOutputs(outChain, loweredOutputsStorage);
  return success();
}

//===----------------------------------------------------------------------===//
// MOGG SplitOp
//===----------------------------------------------------------------------===//

LogicalResult SplitOp::lower(MGP::MOGraphLoweringState &loweringState) {
  ASSERT_STREAM(MO::isOnHost(*this), << "mogg.split is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, *this);

  // Gather the input buffer.
  Value inputBuffer = opLoweringState.getBufferizedInput(getInput()).buffer;

  // TODO Enable this.
  if (inputBuffer.getDefiningOp<MGP::BufferConstantOp>())
    return emitOpError("constant input buffer is unsupported");

  // Create specs for the outputs. We don't need any buffers since this is
  // a split.
  FailureOr<SmallVector<TypedValue<MGP::TensorSpecType>>> outputSpecsOr =
      opLoweringState.createOutputTensorSpecs();
  if (failed(outputSpecsOr))
    return failure();
  auto &outputSpecs = *outputSpecsOr;

  // Get the output !mo.tensor types from which we can calculate split sizes.
  SmallVector<MO::TensorType> resultMoTypes =
      MGP::castToMoTensorTypes(getResultTypes());

  // Build/collect inputs to the MGP buffer split op
  SmallVector<Value> runtimeSizes;
  SmallVector<std::optional<uint64_t>> outBufferSizes;
  for (TypedValue<MGP::TensorSpecType> outSpec : outputSpecs) {
    MGP::TensorSpecType outSpecTy = outSpec.getType();
    std::optional<uint64_t> bufferSize = outSpecTy.getSizeInBytes();
    outBufferSizes.emplace_back(bufferSize);
    if (!bufferSize.has_value()) {
      runtimeSizes.emplace_back(
          opLoweringState.create<MGP::CpuTensorSpecSizeOp>(
              /*spec=*/outSpec));
    }
  }

  SmallVector<Value> outputBuffers = opLoweringState.splitBorrow(
      loweringState.getExecuteChain(), inputBuffer, runtimeSizes,
      outBufferSizes, getInput().getType());
  // Create the output tensors. Note there's no new chain to wait on.
  opLoweringState.createOutputTensors(loweringState.getExecuteChain(),
                                      outputBuffers, outputSpecs);
  return success();
}

//===----------------------------------------------------------------------===//
// MOGG LoadCompiledKernelOp
//===----------------------------------------------------------------------===//

LogicalResult
LoadCompiledKernelOp::lower(MGP::MOGraphLoweringState &loweringState) {
  OpBuilder &builder = loweringState.getInitOpBuilder();
  MGP::BufferBuilder *bufferBuilder =
      loweringState.getBufferBuilder(MO::getDeviceOrDefaultAny(*this));
  Value outChain = bufferBuilder->moduleLoad(
      builder, getLoc(), loweringState.getInitMgpContext(),
      // DANGER: init chain is the result of device context creation which is a
      // result of runtime creation. Module load does not actually depend on the
      // device context. And the device context creation is turned into another
      // module load later. But we need to order module loads by when they are
      // created, so we make this module load depend only on the runtime
      // creation.
      /*inChain=*/
      cast<TypedValue<MEF::ChainType>>(
          loweringState.getInitChain()
              .getDefiningOp()
              ->getOperand(0)
              .getDefiningOp<MGP::RuntimeCreateOp>()
              .getOutChain()),
      /*image=*/getValue(),
      /*functionSymbols=*/getSymbols(),
      /*constantSymbols=*/{});
  auto combined = builder.create<GRT::ChainCreateOp>(
      getLoc(), mlir::ValueRange({outChain, loweringState.getInitChain()}));
  loweringState.setInitChain(combined);
  return success();
}

//===----------------------------------------------------------------------===//
// MOGG ExecuteCompiledOp
//===----------------------------------------------------------------------===//

LogicalResult
ExecuteCompiledOp::lower(MGP::MOGraphLoweringState &loweringState) {
  MGP::MOOpLoweringState opLoweringState(loweringState, *this);

  const bool isInplace = llvm::any_of(getOperands(), [](Value val) {
    return isa<MO::BufferType>(val.getType());
  });

  if (isInplace)
    ASSERT_STREAM(getResults().size() == 1 &&
                      isa<MO::ChainType>(getResult(0).getType()),
                  "In-place kernels (those with buffer inputs) are not allowed "
                  "to have any outputs other than a single chain.");

  // Gather the lowered op operands.
  SmallVector<Value> loweredInputOperands =
      opLoweringState.getLoweredOperands();

  SmallVector<Value> chainOperands;

  // Gather all the input and output operands and their in/out semantics.
  SmallVector<Value> operands;
  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics;
  for (auto [origOperand, loweredInputOperand] :
       llvm::zip(getOperands(), loweredInputOperands)) {
    if (isa<MO::ChainType>(origOperand.getType())) {
      chainOperands.push_back(origOperand);
      continue;
    }
    auto sig = isa<MO::BufferType>(origOperand.getType())
                   ? InOutSignatureAttr::kMut
                   : InOutSignatureAttr::kIn;

    operands.push_back(loweredInputOperand);
    operandSemantics.emplace_back(sig);
  }
  FailureOr<SmallVector<MGP::MOOutputStorage>> loweredOutputsStorageOr;
  if (!isInplace) {
    // Allocate for the op's results.
    // At this stage we gate the allocations only on the overall 'execute'
    // block's chain. The DeferAllocsPass will change this to use the chain
    // which signals when all of the operation's input buffers are ready to be
    // read, thus ensuring we allocate buffers just-in-time for their writers.
    loweredOutputsStorageOr =
        opLoweringState.createOutputStorage(loweringState.getExecuteChain());
    if (failed(loweredOutputsStorageOr))
      return failure();
    auto &loweredOutputsStorage = *loweredOutputsStorageOr;
    for (auto &outputStorage : loweredOutputsStorage) {
      operands.push_back(outputStorage.buffer);
      operandSemantics.emplace_back(InOutSignatureAttr::kOut);
    }
  }

  // Gather the input dimensions from the dimension parameters in the
  // !mo.tensor types.
  SmallVector<Value> inDimensions;
  ArrayAttr usedParams = getUsedParamsAttr();
  for (auto param : usedParams.getAsRange<KGEN::ParamDeclRefAttr>()) {
    auto symbolicDimOr = loweringState.lowerExpression(
        opLoweringState.opBeingLowered->getParentRegion(), opLoweringState.loc,
        param);
    if (symbolicDimOr.isError())
      return failure();
    inDimensions.push_back(symbolicDimOr.takeValue());
  }

  ASSERT_STREAM(chainOperands.size() <= 1,
                "mogg.execute_kernel can have at most one chain input");

  auto inChain = [&]() -> Value {
    if (chainOperands.size() == 1)
      return loweringState.lookupExecuteLoweredValue(chainOperands.front());
    else
      return loweringState.getExecuteChain();
  }();

  // Emit the kernel call.
  SmallVector<Value> outDimensions;
  auto outChain = opLoweringState.kernelExecute(
      /*kernelName=*/getKernelName(),
      /*inChain=*/inChain,
      /*args=*/operands,
      /*inOutSemantics=*/operandSemantics,
      /*inDimensions=*/inDimensions,
      /*numOutDimensions=*/static_cast<uint64_t>(getOutputParamDecls().size()),
      /*outDimensions=*/outDimensions,
      /*summary=*/getSummary(),
      /*isAsync*/ getIsAsync());

  // Each of the parameter outputs now maps onto a concrete dimension value.
  for (const auto &[param, value] :
       llvm::zip(getOutputParamDecls(), outDimensions)) {
    if (failed(loweringState.lowerParamDecl(
            opLoweringState.opBeingLowered->getParentRegion(), param, value)))
      return failure();
  }

  if (!isInplace) {
    // Create outputs from their output storage, but only once the op has
    // completed.
    auto &loweredOutputsStorage = *loweredOutputsStorageOr;
    opLoweringState.createOutputs(outChain, loweredOutputsStorage);
  } else {
    loweringState.emplaceExecuteLoweredValue(getResult(0), outChain);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Parameter printing/parsing support in form: <P1, P2, P3>
//===----------------------------------------------------------------------===//

static ParseResult parseParamDecls(OpAsmParser &p,
                                   KGEN::ParamDeclArrayAttr &paramDecls) {
  if (succeeded(p.parseOptionalLess()) && failed(p.parseOptionalGreater())) {
    if (KGEN::parseParamDecls(p, paramDecls) || p.parseGreater())
      return failure();
  } else {
    paramDecls = KGEN::ParamDeclArrayAttr::get(p.getContext(), {});
  }
  return success();
}

static void printParamDecls(OpAsmPrinter &p, Operation *,
                            KGEN::ParamDeclArrayAttr paramDecls) {
  if (!paramDecls.empty()) {
    p << "<";
    printParamDecls(p, paramDecls);
    p << '>';
  }
}

static ParseResult
parseRegionArguments(OpAsmParser &p, SmallVectorImpl<Type> &resTypes,
                     SmallVectorImpl<OpAsmParser::Argument> &args) {
  auto parseArg = [&] {
    return p.parseArgument(args.emplace_back(), /*allowType=*/true);
  };
  if (p.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseArg) ||
      p.parseOptionalArrowTypeList(resTypes))
    return failure();
  return success();
}

static void printRegionArguments(OpAsmPrinter &p, TypeRange resTypes,
                                 ArrayRef<BlockArgument> args) {
  p << '(';
  llvm::interleaveComma(args, p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << ')';
  if (resTypes.empty())
    return;
  p << ' ';
  p.printArrowTypeList(resTypes);
}

static ParseResult parseLambda(OpAsmParser &p, StringAttr &name, TypeAttr &func,
                               Region &region) {
  SmallVector<OpAsmParser::Argument> args;
  llvm::SMLoc loc;
  SmallVector<Type> resTypes;

  llvm::SMLoc sigLoc;
  if (p.parseSymbolName(name) || p.parseEqual() ||
      p.getCurrentLocation(&sigLoc) ||
      parseRegionArguments(p, resTypes, args) || p.getCurrentLocation(&loc) ||
      p.parseRegion(region, args))
    return failure();
  func = TypeAttr::get(p.getBuilder().getFunctionType(
      llvm::map_to_vector(args, [&](auto &arg) { return arg.type; }),
      resTypes));
  return success();
}

static void printLambda(OpAsmPrinter &p, Operation *lambda, StringAttr name,
                        TypeAttr sig, Region &region) {
  p.printSymbolName(name.strref());
  p << " = ";

  auto funcType = cast<FunctionType>(sig.getValue());
  printRegionArguments(p, funcType.getResults(), region.getArguments());
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// MOGG CallOp
//===----------------------------------------------------------------------===//

bool CallOp::setKGENParamToValueIfPresent(StringRef name, Attribute value) {
  // Exit out if we don't have the parameter already (we can't add new ones as
  // the params come from mojo).
  std::optional<mlir::DictionaryAttr> dict = getKgenParams();
  if (!dict)
    return false;

  mlir::NamedAttrList attributes(*dict);
  // Exit if the parameter doesn't exist in the dictionary attr.
  if (!attributes.getNamed(name).has_value())
    return false;

  Attribute oldAttr = attributes.set(name, value);

  // Update the dict if it doesn't already have that value.
  if (oldAttr != value)
    setKgenParamsAttr(attributes.getDictionary(getContext()));

  return true;
}

//===----------------------------------------------------------------------===//
// MOGG KernelOp
//===----------------------------------------------------------------------===//

SmallVector<KGEN::ParamDeclRefAttr>
M::MOGG::getKernelUsedParameters(MOGG::CompilableKernel kernel) {
  DenseSet<KGEN::ParamDeclRefAttr> seenParams;
  SmallVector<KGEN::ParamDeclRefAttr> usedParameters;

  auto params = kernel.getOutputParamDecls();
  auto isInOutputParams = [&](StringRef name) {
    for (KGEN::ParamDeclAttr attr : params) {
      if (attr.getName().str() == name)
        return true;
    }
    return false;
  };

  // A walker allows us to crawl the parameters used within expression
  // attributes.
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](KGEN::ParamDeclRefAttr param) {
    if (seenParams.contains(param))
      return WalkResult::advance();

    if (isInOutputParams(param.getName()))
      return WalkResult::advance();

    seenParams.insert(param);
    usedParameters.push_back(param);
    return WalkResult::advance();
  });

  // Track all unique parameters from a given tensor that we haven't seen
  // already.
  auto addParamsFromTensor = [&](Type t) {
    MOSH::ShapeAttr shape;

    // The type can either be a shape or a tensor.
    MO::TensorType asTensor = dyn_cast<MO::TensorType>(t);
    if (asTensor)
      shape = asTensor.getTypedShapeAttr();
    MOGG::ShapeType asShape = dyn_cast<MOGG::ShapeType>(t);
    if (asShape)
      shape = cast<MOSH::ShapeAttr>(asShape.getShapeAttr());

    if (!shape)
      return;
    walker.walk(shape);
  };

  // Get all parameters used by an operation.
  auto addParamsFromOp = [&](Operation *op) {
    for (Type t : op->getOperandTypes())
      addParamsFromTensor(t);

    for (Type t : op->getResultTypes())
      addParamsFromTensor(t);
  };

  // Start with the inputs / outputs on the kernel. Generally this would be all
  // parameters.
  addParamsFromOp(kernel);

  // However rarely we could create a kernel which internally uses a parameter
  // which isn't known from the inputs. I.E

  // clang-format off
  // %0 = broadcast(%input) : tensor<[N]> -> tensor<N, J>
  // %1 = broadcast(%0) : tensor<[N, J]> -> tensor<N, B>
  // %3 = broadcast(%1) : tensor<[N, B, K]> -> tensor<N, B, K, M>
  // clang-format on

  // We would see `N` on the input and `B` / `K` / `M` on the outputs but the
  // intermediate `J` is not on either.

  // To capture the above legal composition we walk the kernel ops as well.
  kernel.walk(addParamsFromOp);

  return usedParameters;
}

// This function lowers the `mogg.kernel` to `mogg.generic.compiled.region`,
// adding the ability to consider the in-place allocated dynamic tensors from
// the user kernels. We do the storage allocation of mgp buffers only for
// pre-allocated tensors and leave the ones for the in-place allocation. We also
// convert the parameter values to the indexes (materialization) during the
// lowering.
// NOTE: does mogg.kernel --> region lowering
LogicalResult KernelOp::lower(MGP::MOGraphLoweringState &loweringState) {
  MGP::MOOpLoweringState opLoweringState(loweringState, *this);
  mlir::MLIRContext *ctx = getContext();
  auto &builder = opLoweringState.builder;
  auto chain = loweringState.getExecuteChain();

  // Gather the lowered op operands.
  SmallVector<Value> loweredInputOperands =
      opLoweringState.getLoweredOperands();

  FailureOr<SmallVector<MGP::MOOutputStorage>> loweredOutputsStorageOr =
      opLoweringState.createOutputStorage(chain);
  if (failed(loweredOutputsStorageOr))
    return failure();
  auto &loweredOutputsStorage = *loweredOutputsStorageOr;

  DenseSet<int> inplaceAllocIndicesSet;
  SmallVector<int> preAllocIndices;
  SmallVector<int> inplaceAllocIndices;
  auto alloc = getAllocIndices();
  if (alloc.has_value()) {
    for (auto idxAttr : alloc.value().getAsValueRange<mlir::IntegerAttr>()) {
      int64_t idx = idxAttr.getSExtValue();
      inplaceAllocIndicesSet.insert(idx);
    }
  }

  OutputPlaceholdersOp oldOutputs = getOutputPlaceholder();

  DenseSet<int> returnAllocIndicesSet;

  for (unsigned int i = 0; i < getNumResults(); i++) {
    if (i >= oldOutputs->getNumResults())
      returnAllocIndicesSet.insert(i);
    if (!inplaceAllocIndicesSet.contains(i))
      preAllocIndices.push_back(i);
    else
      inplaceAllocIndices.push_back(i);
  }

  size_t numInplaceOutputs = inplaceAllocIndicesSet.size();

  size_t numOperands =
      1 + loweredInputOperands.size() + loweredOutputsStorage.size();

  // Gather all the input and output operands and their in/out semantics.
  SmallVector<Value> operands;
  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics;
  operands.reserve(numOperands);
  operandSemantics.reserve(numOperands);

  operands.push_back(chain);
  operandSemantics.emplace_back(InOutSignatureAttr::kIn);

  for (auto &outputStorage : loweredOutputsStorage) {
    operands.push_back(outputStorage.buffer);
    operandSemantics.emplace_back(InOutSignatureAttr::kOut);
  }

  for (Value loweredInputOperand : loweredInputOperands) {
    operands.push_back(loweredInputOperand);
    operandSemantics.emplace_back(InOutSignatureAttr::kIn);
  }

  SmallVector<KGEN::ParamDeclRefAttr> usedParams =
      M::MOGG::getKernelUsedParameters(*this);
  llvm::StringMap<int> paramDimToIdx;
  for (auto param : usedParams) {
    auto symbolicDimOr = loweringState.lowerExpression(
        opLoweringState.opBeingLowered->getParentRegion(), opLoweringState.loc,
        param);

    ASSERT_STREAM(!symbolicDimOr.isError(),
                  << "can't find the symbolic dimension of the parameter: "
                  << symbolicDimOr.takeError());

    Value symbol = symbolicDimOr.takeValue();
    paramDimToIdx[param.getName().strref()] = operands.size();
    operands.push_back(symbol);
    operandSemantics.emplace_back(InOutSignatureAttr::kIn);
  }

  auto idxType = mlir::IndexType::get(ctx);
  DeviceRefAttr device = MO::getDeviceOrDefaultAny(*this);
  auto mgpBufferType =
      MGP::DeviceBufferType::get(device, std::optional<uint64_t>());

  SmallVector<mlir::Type> inPlaceMGPOutputsTypes(numInplaceOutputs);
  for (auto [idx, inPlaceIdx] : llvm::enumerate(inplaceAllocIndices)) {
    Type oldOutTy = getResult(inPlaceIdx).getType();
    // mgp.bufer is user for mo.tensor, mgp.opaque for mo.opaque.
    Type mgpResultTy = mgpBufferType;
    if (isa<MO::OpaqueType>(oldOutTy))
      mgpResultTy = MGP::OpaqueType::get(ctx, std::nullopt);
    inPlaceMGPOutputsTypes[idx] = mgpResultTy;
  }

  auto outParamNum = getOutputParamDecls().size();
  llvm::StringMap<int> outParamNameToIdx;
  SmallVector<mlir::Type> newKernelResTypes;

  newKernelResTypes.push_back(chain.getType());
  newKernelResTypes.insert(newKernelResTypes.end(),
                           inPlaceMGPOutputsTypes.begin(),
                           inPlaceMGPOutputsTypes.end());
  for (size_t i = 0; i < outParamNum; i++) {
    newKernelResTypes.push_back(idxType);
    StringRef name = getOutputParamDecls()[i].getName();
    outParamNameToIdx[name] = i;
  }

  // TODO: Fusing kernels with inplace allocations is not supported
  // (intermediates allocations unsupported / leaking).
  bool canFuseIntoOther =
      !(getAllocIndices().has_value() && !getAllocIndices()->empty());
  bool hasCuda = device.getLabel() != kCPULabel;
  // Later update the isAsync instead of hard-coding
  auto newKernel = builder.create<MOGG::GenericCompiledRegionOp>(
      getLoc(), TypeRange{newKernelResTypes}, operands,
      getOutputParamDeclsAttr(), InOutSignatureAttr::get(ctx, operandSemantics),
      getIsKGENAttr(),                     // isKGEN
      getSummaryAttr(),                    // summary
      BoolAttr::get(ctx, false),           // isAsync
      BoolAttr::get(ctx, hasCuda),         // hasCuda
      BoolAttr::get(ctx, canFuseIntoOther) // isFuseable
  );

  mlir::Block &kernelBody = newKernel.getBody().emplaceBlock();
  builder.setInsertionPoint(&kernelBody, kernelBody.begin());
  kernelBody.addArgument(chain.getType(), chain.getLoc());
  for (auto val : loweredOutputsStorage)
    kernelBody.addArgument(val.buffer.getType(), val.buffer.getLoc());
  for (auto val : loweredInputOperands)
    kernelBody.addArgument(val.getType(), val.getLoc());
  for (Value operand : newKernel.getOperands()) {
    if (!isa<MGP::DeviceBufferType, MGP::OpaqueType>(operand.getType()) &&
        operand != chain)
      kernelBody.addArgument(operand.getType(), operand.getLoc());
  }

  DenseMap<TypedAttr, Value> loweredParams;
  auto addParams = [&](Location loc, Operation &op,
                       Type t) -> CompileDiagnosticOrSuccess {
    ArrayRef<TypedAttr> dims;
    if (auto asTensor = dyn_cast<MO::TensorType>(t))
      dims = asTensor.getDims();
    else if (auto shape = dyn_cast<MOGG::ShapeType>(t))
      dims = dyn_cast<MOSH::ShapeAttr>(shape.getShapeAttr()).getValues();
    else
      return success();

    for (auto ty : dims) {
      if (loweredParams.contains(ty))
        continue;

      if (auto cst = dyn_cast<IntegerAttr>(ty)) {
        int64_t asInt = cst.getInt();
        KGEN::ParamConstantOp op = builder.create<KGEN::ParamConstantOp>(
            loc, builder.getIndexType(), builder.getIndexAttr(asInt));

        loweredParams[ty] = op.getResult();
      } else if (auto cst = dyn_cast<KGEN::ParamDeclRefAttr>(ty)) {
        loweredParams[ty] =
            kernelBody.getArgument(paramDimToIdx[cst.getName()]);
      } else {
        return op.emitError("Unsupported dim attribute type: ") << ty;
      }
    }
    return success();
  };

  // Add all the seen shape parameters values as concrete KGEN param values.
  for (auto &op : this->getOps()) {
    for (auto type : op.getOperandTypes()) {
      auto err = addParams(op.getLoc(), op, type);
      if (err.isError())
        return err.takeError();
    }
    for (auto type : op.getResultTypes()) {
      auto err = addParams(op.getLoc(), op, type);
      if (err.isError())
        return err.takeError();
    }
  }

  IRMapping mapper;

  // Convert MGP block input to tensor object (!mo.tensor) or opaque.
  auto convertMGPInputToTensorOrOpaque = [&](Value input, Type oldTy) -> Value {
    // Opaque types are already pointer to objects, they don't need any
    // transformation to be passed to the kernel.
    if (isa<MGP::OpaqueType>(input.getType())) {
      return builder.create<MOGG::ToOpaqueFromMGPOp>(input.getLoc(), oldTy,
                                                     input);
    }

    MO::TensorType asTensor = cast<MO::TensorType>(oldTy);

    SmallVector<Value> operands;
    operands.push_back(input);
    for (auto ty : asTensor.getDims())
      operands.push_back(loweredParams[ty]);

    return builder.create<MOGG::ToTensorFromMGPOp>(input.getLoc(), oldTy,
                                                   operands);
  };

  int outputOffset = 1; // 1 for the input chain
  for (auto [idx, preAllocIdx] : llvm::enumerate(preAllocIndices)) {
    Value newBlockArg = kernelBody.getArgument(outputOffset + idx);
    Value oldOut = oldOutputs.getResult(preAllocIdx);
    Value loweredValue =
        convertMGPInputToTensorOrOpaque(newBlockArg, oldOut.getType());
    mapper.map(oldOut, loweredValue);
  }

  int inputOffset = loweredOutputsStorage.size() + 1;
  for (auto [idx, val] : llvm::enumerate(loweredInputOperands)) {
    Value newBlockArg = kernelBody.getArgument(idx + inputOffset);
    Value oldBlockArg = getBody().getArgument(idx);
    Value loweredValue =
        convertMGPInputToTensorOrOpaque(newBlockArg, oldBlockArg.getType());
    mapper.map(oldBlockArg, loweredValue);
  }

  // Allocate stack for in-place tensors
  SmallVector<Type> internalTensorAllocTypes;
  SmallVector<Type> internalOpaqueAllocTypes;
  Operation *stackAllocOp = nullptr;
  for (auto [idx, inPlaceIdx] : llvm::enumerate(inplaceAllocIndices)) {
    // Outputs directly return by the MOGG call don't need a stack allocation.
    if (returnAllocIndicesSet.contains(inPlaceIdx))
      continue;

    Value oldOut = oldOutputs.getResult(inPlaceIdx);
    if (auto asTensor = dyn_cast<MO::TensorType>(oldOut.getType())) {
      SmallVector<int64_t> dynamicShape(asTensor.getIntDims().size(), 1);
      auto dynamicTensorTy =
          MO::TensorType::get(dynamicShape, asTensor.getElementType());
      internalTensorAllocTypes.push_back(dynamicTensorTy);
    } else {
      // Opaque type.
      internalOpaqueAllocTypes.push_back(oldOut.getType());
    }
  }
  if (!internalTensorAllocTypes.empty()) {
    stackAllocOp = builder.create<MOGG::StackAllocateTensorsOp>(
        getLoc(), internalTensorAllocTypes);
    for (auto [oldOut, newAlloc] :
         llvm::zip(oldOutputs->getResults(), stackAllocOp->getResults()))
      mapper.map(oldOut, newAlloc);
  }
  if (!internalOpaqueAllocTypes.empty()) {
    assert(!stackAllocOp &&
           "mix of tensor / opaque output types not supported");
    stackAllocOp = builder.create<MOGG::StackAllocateOpaquesOp>(
        getLoc(), internalOpaqueAllocTypes);
    for (auto [oldOut, newAlloc] :
         llvm::zip(oldOutputs->getResults(), stackAllocOp->getResults()))
      mapper.map(oldOut, newAlloc);
  }

  // Clone the graph
  SmallPtrSet<Operation *, 4> setShapeParamOpSet;
  for (Operation &op : getBody().front()) {
    if (isa<MOGG::OutputPlaceholdersOp, MOGG::OutputOp>(op))
      continue;

    if (isa<MOGG::SetParameterOp>(op)) {
      setShapeParamOpSet.insert(builder.clone(op, mapper));
      continue;
    }
    builder.clone(op, mapper);
  }

  // We must remove all param_shape_to_value ops because parameters don't exist
  // in MGP.
  SmallVector<MOGG::ParamShapeToValueOp> shapes(
      kernelBody.getOps<MOGG::ParamShapeToValueOp>());
  for (auto getShape : shapes) {
    builder.setInsertionPointAfter(getShape);
    SmallVector<Value> dimensions;
    for (auto dim : dyn_cast<MOSH::ShapeAttr>(
                        getShape.getResult().getType().getShapeAttr())
                        .getValues()) {
      auto param = loweredParams.find(dim);
      if (param == loweredParams.end())
        return getShape->emitError("Parameter has not been lowered: ") << dim;
      dimensions.push_back(param->second);
    }

    // NOTE(GRA-531): Right now we assume a rank0 tensor maps to a shape of [1].
    if (dimensions.empty()) {
      auto kgenOne = builder.create<KGEN::ParamConstantOp>(
          getShape->getLoc(), IntegerAttr::get(builder.getIndexType(), 1));
      dimensions.push_back(kgenOne);
    }

    auto arrayCreate =
        builder.create<KGEN::POP::ArrayCreateOp>(getShape.getLoc(), dimensions);
    auto toMOGG = builder.create<MOGG::ToMOGGOp>(
        getShape.getLoc(), TypeRange(getShape.getResult().getType()),
        ValueRange(arrayCreate));
    getShape.replaceAllUsesWith(toMOGG.getResult());
    getShape->erase();
  };

  DenseMap<int, MOGG::ToKGENOp> outIdxToKgenTensorOp;
  for (auto [idx, inPlaceIdx] : llvm::enumerate(inplaceAllocIndices)) {
    // Outputs returned by a MOGG call don't have a stack allocation.
    if (returnAllocIndicesSet.contains(inPlaceIdx))
      continue;
    Value moTensor = stackAllocOp->getResult(idx);
    ToKGENOp toKGEN;
    for (Operation *user : moTensor.getUsers())
      if (auto tmp = dyn_cast<ToKGENOp>(user))
        toKGEN = tmp;

    ASSERT_STREAM(toKGEN, << " in-place allocated tensor must have a "
                             "to_kgen op from an outputplaceholder.");
    outIdxToKgenTensorOp[inPlaceIdx] = toKGEN;
  }

  builder.setInsertionPointToEnd(&kernelBody);
  DenseMap<TypedValue<MOGG::ShapeType>, Operation *> shapeToDimFromShape;
  llvm::StringMap<Value> paramToDim;
  for (Operation *newOp : setShapeParamOpSet) {
    if (auto setParam = dyn_cast<SetParameterOp>(newOp)) {
      auto inputShape = setParam.getShape();
      if (!shapeToDimFromShape.contains(inputShape)) {
        auto shapeAttr =
            cast<MOGG::ShapeType>(inputShape.getType()).getShapeAttr();
        auto shapeMosh = cast<MOSH::ShapeAttr>(shapeAttr);
        size_t rank = shapeMosh.getRank();
        if (rank == 0)
          rank = 1;

        SmallVector<Type> resT;
        for (size_t i = 0; i < rank; i++)
          resT.push_back(idxType);

        Operation *dsh = builder.create<DimFromShapeOp>(
            getLoc(), mlir::TypeRange{resT}, inputShape);
        shapeToDimFromShape[inputShape] = dsh;
      }
      int index = setParam.getIndex().getInt();
      StringRef name = setParam.getParamName().strref();
      paramToDim[name] = shapeToDimFromShape[inputShape]->getResult(index);
    }
  }
  // newKernel generates outputs as follows:
  //   mogg.output %other:numPrependedOutputs %inPlaceOutputs:numInplaceOutputs,
  //   %OutputShapeParams:outParamNum
  constexpr unsigned int numPrependedOutputs = 1;
  SmallVector<Value> outputs(numPrependedOutputs + numInplaceOutputs +
                             outParamNum);
  // numPrependedOutputs: Create placeholder
  outputs[0] = builder.create<MOGG::CreateReadyChainOp>(
      getLoc(), TypeRange{newKernel->getResult(0).getType()});
  // numInplaceOutputs: Add intrinsic to map to mgp for specific types
  for (auto [idx, inPlaceIdx] : llvm::enumerate(inplaceAllocIndices)) {
    // Get the output KGEN value for in-place allocs.
    // This is either the ToKGENOp following the stack alloc.
    // Or the value directly returned by the kernel
    Value kgenVal;
    if (returnAllocIndicesSet.contains(inPlaceIdx))
      kgenVal = mapper.lookup(
          getBody().front().getTerminator()->getOperand(inPlaceIdx));
    else
      kgenVal = outIdxToKgenTensorOp[inPlaceIdx].getResult();

    Type resultTy = inPlaceMGPOutputsTypes[idx];
    if (isa<MGP::OpaqueType>(resultTy)) {
      outputs[idx + numPrependedOutputs] =
          builder.create<MOGG::ToMGPFromOpaqueOp>(getLoc(), TypeRange{resultTy},
                                                  kgenVal);
    } else {
      outputs[idx + numPrependedOutputs] =
          builder.create<MOGG::ToMGPFromTensorOp>(getLoc(), TypeRange{resultTy},
                                                  kgenVal);
    }
  }
  // outParamNum: Add all the params from stringMap paramToDim so do lookup
  for (const auto &[name, idx] : outParamNameToIdx)
    outputs[idx + numPrependedOutputs + numInplaceOutputs] = paramToDim[name];
  // Construct a mogg.output that newKernel spec
  builder.create<MOGG::OutputOp>(getLoc(), ValueRange{outputs});

  // Each of the parameter outputs now maps onto a concrete dimension value.
  for (const auto param : getOutputParamDecls()) {
    if (failed(loweringState.lowerParamDecl(
            opLoweringState.opBeingLowered->getParentRegion(), param,
            newKernel.getResult(outParamNameToIdx[param.getName()] + 1 +
                                numInplaceOutputs))))
      return failure();
  }
  builder.setInsertionPointAfter(newKernel);
  for (unsigned int idx = 0; idx < numInplaceOutputs; ++idx) {
    unsigned int resultIdx = idx + numPrependedOutputs;
    Type resultTy = getResult(idx).getType();
    if (!isa<MO::TensorType>(resultTy)) {
      // Non-Tensor output types don't need to materialize a tensor or tie it
      // to chains/specs
      loweringState.emplaceExecuteLoweredValue(getResult(idx),
                                               newKernel.getResult(resultIdx));
      continue;
    }
    auto tensorTy = cast<MO::TensorType>(resultTy);
    // Materialize the output tensor spec directly. This will automatically
    // pick up any of the output dimensions set by the kernel.
    FailureOr<TypedValue<MGP::TensorSpecType>> tensorSpecOr =
        opLoweringState.createTensorSpec(tensorTy);
    if (failed(tensorSpecOr))
      return failure();
    TypedValue<MGP::TensorSpecType> &spec = *tensorSpecOr;

    opLoweringState.createOutputTensor(getResult(idx), newKernel.getResult(0),
                                       newKernel.getResult(resultIdx), spec);
  }
  opLoweringState.createOutputs(newKernel->getResult(0), loweredOutputsStorage);

  // Clean up useless ops.
  for (Operation *op : setShapeParamOpSet)
    op->erase();

  return success();
}

OutputPlaceholdersOp KernelOp::getOutputPlaceholder() {
  auto placeholder = this->getOps<MOGG::OutputPlaceholdersOp>();
  if (placeholder.empty())
    return nullptr;

  assert(llvm::range_size(placeholder) == 1);

  return *placeholder.begin();
}

StackAllocateTensorsOp KernelOp::getStackAllocateTensor() {
  auto stackAlloc = this->getOps<MOGG::StackAllocateTensorsOp>();
  if (stackAlloc.empty())
    return nullptr;

  assert(llvm::range_size(stackAlloc) == 1);

  return *stackAlloc.begin();
}

CreateEmptyTensorsOp KernelOp::getCreateEmptyTensor() {
  auto createEmpty = this->getOps<MOGG::CreateEmptyTensorsOp>();
  if (createEmpty.empty())
    return nullptr;

  assert(llvm::range_size(createEmpty) == 1);

  return *createEmpty.begin();
}

OutputOp KernelOp::getOutputOp() {
  auto createEmpty = this->getOps<MOGG::OutputOp>();
  if (createEmpty.empty())
    return nullptr;

  assert(llvm::range_size(createEmpty) == 1);

  return *createEmpty.begin();
}

void KernelOp::setInsertionPointToTop(OpBuilder &builder) {
  OutputPlaceholdersOp outputPh = getOutputPlaceholder();

  // Not everything has to have an input placeholder, e.g a shape function.
  if (outputPh)
    builder.setInsertionPointAfter(outputPh);
  else
    builder.setInsertionPointToStart(&getRegion().front());
}

/// Disables future fusion on this input.
void KernelOp::disableInputFusionOnInput(size_t index) {
  std::optional<ArrayAttr> inputLambdas = getInputFusionInterfaceAttr();

  if (inputLambdas.has_value()) {
    SmallVector<Attribute> lambdaNames;

    // Add all the old attributes to the new one but remove the attribute at
    // the disabled index.
    size_t i = 0;
    for (StringAttr attr : inputLambdas->getAsRange<StringAttr>()) {
      if (i == index) {
        lambdaNames.push_back(StringAttr::get(getContext(), ""));
      } else {
        lambdaNames.push_back(attr);
      }

      i++;
    }

    // Update the attribute.
    setInputFusionInterfaceAttr(ArrayAttr::get(getContext(), lambdaNames));
  }
}

void KernelOp::removeTemporaryFusionMarks() {
  SmallVector<Attribute> lambdaNames;
  for (StringAttr attr : getInputFusionInterface()->getAsRange<StringAttr>()) {
    // If this is the temporary string then replace it with null.
    if (attr == KernelOp::getTempLambdaName())
      lambdaNames.push_back(StringAttr::get(getContext(), ""));
    else
      lambdaNames.push_back(attr);
  }
  setInputFusionInterfaceAttr(ArrayAttr::get(getContext(), lambdaNames));
}

/// Removes an operand at the given index and disables input fusion on that
/// position if this kernel otherwise supports it.
void KernelOp::removeOperandAtIndex(size_t index) {
  (*this)->eraseOperand(index);
  getBody().front().eraseArgument(index);

  std::optional<ArrayAttr> inputInterfaces = getInputFusionInterface();

  // Update the input fusion interfaces if we have them to remove fusion on
  // this attr.
  if (inputInterfaces.has_value()) {
    SmallVector<Attribute> lambdaNames;
    size_t i = 0;
    // Add each of the current input hooks to the list, skipping the input we
    // just removed.
    for (StringAttr name : inputInterfaces->getAsRange<StringAttr>()) {
      if (i != index)
        lambdaNames.push_back(name);
      i++;
    }
    setInputFusionInterfaceAttr(ArrayAttr::get(getContext(), lambdaNames));
  }
}

/// Add an operand to the kernel and adds an argument to the kernel region.
/// Fixes up the input fusion interfaces to disallow input fusion on this
/// position. Returns the added kernel argument.
Value KernelOp::addOperand(Value operand) {
  (*this)->insertOperands(getNumOperands(), ValueRange{operand});
  Value blockArg =
      getBody().front().addArgument(operand.getType(), operand.getLoc());

  std::optional<ArrayAttr> inputInterfaces = getInputFusionInterface();

  // Update the input fusion interfaces if we have them to remove fusion on
  // this attr.
  if (inputInterfaces.has_value()) {
    SmallVector<Attribute> lambdaNames;
    for (StringAttr name : inputInterfaces->getAsRange<StringAttr>())
      lambdaNames.push_back(name);
    // Add an empty lambda name for the operand we just added.
    lambdaNames.push_back(StringAttr::get(getContext(), ""));
    setInputFusionInterfaceAttr(ArrayAttr::get(getContext(), lambdaNames));
  }

  return blockArg;
}

void KernelOp::setInputFusionInterface(size_t index, StringRef lambdaName) {
  std::optional<ArrayAttr> inputInterfaces = getInputFusionInterface();

  // Update the input fusion interfaces if we have them to remove fusion on
  // this attr.
  if (inputInterfaces.has_value()) {
    SmallVector<Attribute> lambdaNames;
    for (StringAttr name : inputInterfaces->getAsRange<StringAttr>()) {
      if (lambdaNames.size() == index)
        lambdaNames.push_back(StringAttr::get(getContext(), lambdaName));
      else
        lambdaNames.push_back(name);
    }
    setInputFusionInterfaceAttr(ArrayAttr::get(getContext(), lambdaNames));
  }
}

// Fuse the body of this kernel into another.
void KernelOp::fuseBodyIntoNew(
    OpBuilder &builder, mlir::SymbolTable &symTable, IRMapping &remapper,
    llvm::StringMap<StringRef> &oldLambdaNamesToNew) {
  auto neverSkip = [](Operation *) { return false; };

  fuseBodyIntoNewWithSkip(builder, symTable, remapper, oldLambdaNamesToNew,
                          neverSkip);
}

// Fuse the body of this kernel into another.
void KernelOp::fuseBodyIntoNewWithSkip(
    OpBuilder &builder, mlir::SymbolTable &symTable, IRMapping &remapper,
    llvm::StringMap<StringRef> &oldLambdaNamesToNew,
    std::function<bool(Operation *)> shouldSkip) {

  // Clone the incoming kernel into this kernel.
  for (Operation &op : getBody().front()) {
    if (shouldSkip(&op))
      continue;
    if (isa<MOGG::OutputOp>(op)) {
      // Skip placeholders & output, already remapped.
    } else if (auto lambda = dyn_cast<LambdaOp>(op)) {
      // Add lambdas seperately so we rename them on insertion. (The
      // symbol table does this for us).
      auto newLambda = cast<LambdaOp>(builder.clone(op, remapper));
      symTable.insert(newLambda, builder.getInsertionPoint());
      builder.setInsertionPointAfter(newLambda);
      oldLambdaNamesToNew[lambda.getSymName()] = newLambda.getSymName();
    } else if (auto call = dyn_cast<CallOp>(op)) {
      // Lambdas will have been inlined. The only place we can have
      // a lambda reference is in a call.
      auto newCall = dyn_cast<CallOp>(builder.clone(op, remapper));

      SmallVector<Attribute> newAttrs;

      // Remap any lambda references to point to the new lambda name.
      std::optional<ArrayAttr> params = newCall.getParams();
      if (params) {
        for (SymbolRefAttr sym : params->getAsRange<SymbolRefAttr>()) {
          StringRef name = sym.getLeafReference().strref();
          auto itr = oldLambdaNamesToNew.find(name);
          ASSERT_STREAM(
              itr != oldLambdaNamesToNew.end(),
              << "First use of lambda inserted before the lambda itself");
          newAttrs.push_back(FlatSymbolRefAttr::get(getContext(), itr->second));
        }
        newCall.setParamsAttr(builder.getArrayAttr(newAttrs));
      }
    } else {
      builder.clone(op, remapper);
    }
  }
}

KGEN::ParamDeclareRegionOp KernelOp::findLambda(StringAttr name) {
  for (auto op : getBody().getOps<KGEN::ParamDeclareRegionOp>()) {
    if (op.getParamDecl().getName() == name)
      return op;
  }

  return nullptr;
}

StringAttr KernelOp::getInputFusionInterfaceAt(size_t index) {
  auto ctx = getContext();
  auto inputLambdaNamesOpt = getInputFusionInterface();
  if (!inputLambdaNamesOpt)
    return StringAttr::get(ctx, "");

  auto inputLambdaNames = *inputLambdaNamesOpt;

  return cast<StringAttr>(inputLambdaNames[index]);
}

DenseSet<int> KernelOp::getInplaceAllocIndicesSet() {
  DenseSet<int> inplaceAllocIndicesSet;
  auto alloc = getAllocIndices();
  if (alloc.has_value()) {
    for (auto idxAttr : alloc.value().getAsValueRange<mlir::IntegerAttr>()) {
      int64_t idx = idxAttr.getSExtValue();
      inplaceAllocIndicesSet.insert(idx);
    }
  }
  return inplaceAllocIndicesSet;
}

bool KernelOp::isInPlaceAlloc() {
  auto alloc = getAllocIndices();
  if (alloc.has_value())
    return true;
  return false;
}

StringAttr KernelOp::getOutputFusionInterfaceAt(size_t index) {
  auto ctx = getContext();
  auto inputLambdaNamesOpt = getOutputFusionInterface();
  if (!inputLambdaNamesOpt)
    return StringAttr::get(ctx, "");

  auto inputLambdaNames = *inputLambdaNamesOpt;

  return cast<StringAttr>(inputLambdaNames[index]);
}

// Verify that all the `mogg.set_param` set the symbolic dimensions properly.
static LogicalResult verifyKernelSetParam(KernelOp kernel) {
  llvm::SmallSet<StringRef, 4> symbolicDimSet;

  if (kernel.getOps<MOGG::SetParameterOp>().empty())
    return success();

  // Check that each symbolic dim gets assigned at most once.
  for (auto op : kernel.getOps<MOGG::SetParameterOp>()) {
    if (!symbolicDimSet.insert(op.getParamName()).second)
      return mlir::emitError(op->getLoc(),
                             "Symbolic dimension has been assigned more than "
                             "once");
  }

  // Check that the output (at most one) has all the symbolic dimensions
  // assigned by `mogg.set_param`.
  for (auto outputPlaceholder : kernel.getOps<MOGG::OutputPlaceholdersOp>()) {
    if (!llvm::all_of(outputPlaceholder.getResults(), [&](mlir::OpResult res) {
          auto resType = dyn_cast<MO::TensorType>(res.getType());
          if (!resType)
            return true;

          return llvm::all_of(
              resType.getTypedShapeAttr().getValues(),
              [&](mlir::TypedAttr dim) {
                auto symblicDim = dyn_cast<KGEN::ParamDeclRefAttr>(dim);
                if (!symblicDim)
                  return true;
                return symbolicDimSet.contains(symblicDim.getName());
              });
        }))
      return mlir::emitError(outputPlaceholder->getLoc(),
                             "Not all symbolic dimensions are assigned");
  }

  return success();
}

LogicalResult KernelOp::verify() {
  if (llvm::range_size(getOps<MOGG::OutputPlaceholdersOp>()) > 1)
    return emitError("A KernelOp must have at most one output placeholder op");

  auto inputInterfaces = getInputFusionInterface();
  auto outputInterfaces = getOutputFusionInterface();

  if (inputInterfaces.has_value() &&
      getNumOperands() != inputInterfaces->size()) {
    return emitError("Number of input fusion interfaces does not match number "
                     "of inputs")
           << " (" << inputInterfaces->size() << " vs " << getNumOperands()
           << ")";
  }

  if (outputInterfaces.has_value() &&
      getNumResults() != outputInterfaces->size()) {
    return emitError("Number of output fusion interfaces does not match number "
                     "of results")
           << " (" << outputInterfaces->size() << " vs " << getNumResults()
           << ")";
  }

  DenseSet<StringRef> fusionInterfaces;
  mlir::SymbolTable symTable{*this};

  if (inputInterfaces.has_value()) {
    for (auto interface : inputInterfaces->getAsValueRange<StringAttr>()) {
      if (interface.empty())
        continue;

      if (!fusionInterfaces.insert(interface).second)
        return emitError("Fusion interfaces must be unique");

      Operation *lambda = symTable.lookup(interface);
      if (!lambda)
        return emitError("Could not find lambda op specified in the input "
                         "fusion interface as \"")
               << interface.str() << "\"";
    }
  }

  if (outputInterfaces.has_value()) {
    for (auto interface : outputInterfaces->getAsValueRange<StringAttr>()) {
      if (interface.empty())
        continue;

      if (!fusionInterfaces.insert(interface).second)
        return emitError("Fusion interfaces must be unique");

      Operation *lambda = symTable.lookup(interface);
      if (!lambda)
        return emitError("Could not find lambda op specified in the output "
                         "fusion interface as \"")
               << interface.str() << "\"";
    }
  }

  if (getBody().empty())
    return emitError("Region cannot be empty");

  uint64_t numArguments = llvm::range_size(getBody().getArguments());
  if (getNumOperands() != numArguments) {
    return emitError(
               "Number of operands does not match number of body arguments ")
           << " (" << getNumOperands() << " vs " << numArguments << ")";
  }

  // Verify that symbolic dims are set properly in the kernel.
  auto setParamCheck =
      verifyKernelSetParam(llvm::cast<KernelOp>(getOperation()));
  if (mlir::failed(setParamCheck))
    return setParamCheck;

  return success();
}

static inline bool allNullFusionInterface(ArrayAttr arr) {
  auto strArr = arr.getAsRange<StringAttr>();
  return std::all_of(strArr.begin(), strArr.end(), [](StringAttr strAttr) {
    return strAttr.strref().empty();
  });
}

LogicalResult KernelOp::canonicalize(KernelOp op, PatternRewriter &) {
  if (op.getInputFusionInterface().has_value() &&
      allNullFusionInterface(op.getInputFusionInterfaceAttr())) {
    op.removeInputFusionInterfaceAttr();
  }

  if (op.getOutputFusionInterface().has_value() &&
      allNullFusionInterface(op.getOutputFusionInterfaceAttr())) {
    op.removeOutputFusionInterfaceAttr();
  }

  return success();
}

static ParseResult parseKernel(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &operandTypes, SmallVectorImpl<Type> &resultTypes,
    Region &region, KGEN::ParamDeclArrayAttr &resultParams) {
  SmallVector<OpAsmParser::Argument> args;
  llvm::SMLoc loc;

  llvm::SMLoc sigLoc;
  if (parseParamDecls(p, resultParams) ||
      p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
      p.parseColon() || p.getCurrentLocation(&sigLoc) ||
      parseRegionArguments(p, resultTypes, args) ||
      p.getCurrentLocation(&loc) || p.parseRegion(region, args))
    return failure();
  for (auto &arg : args)
    operandTypes.push_back(arg.type);
  return success();
}

static void printKernel(OpAsmPrinter &p, Operation *kernel, ValueRange operands,
                        TypeRange operandTypes, TypeRange resultTypes,
                        Region &region, KGEN::ParamDeclArrayAttr resultParams) {
  printParamDecls(p, kernel, resultParams);
  p << '(';
  p.printOperands(operands);
  p << ") : ";

  printRegionArguments(p, resultTypes, region.getArguments());
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// GenericCompiledRegionOp
//===----------------------------------------------------------------------===//

InOutSignatureAttr::InOutSemantics
GenericCompiledRegionOp::getInOutSemantics(size_t index) {
  ASSERT_STREAM(
      index < getNumOperands(),
      << "GenericCompiledRegionOp::getInOutSemantics index out of range.");
  return getInOutSignature()[index];
}

//===----------------------------------------------------------------------===//
// ShapeToTensorOp
//===----------------------------------------------------------------------===//

LogicalResult ShapeToTensorOp::verify() {
  MO::TensorType resTy = getType();
  if (!resTy.hasRank() || resTy.getRank() != 1)
    return emitError() << "result type must have rank 1";

  // NOTE this implies we don't support dynamically ranked shape at runtime,
  // since this op converts a shape to a tensor.
  if (!resTy.hasStaticShape())
    return emitError() << "result type must have static shape";

  TypedAttr inputShapeAttr = getShape().getType().getShapeAttr();
  if (auto rankedShapeAttr = dyn_cast<MOSH::ShapeAttr>(inputShapeAttr)) {
    if (rankedShapeAttr.getRank() != resTy.getNumElements()) {
      return emitError() << "rank of input shape must match number of elements "
                            "in result tensor";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ToKGENOp
//===----------------------------------------------------------------------===//

FailureOr<DictionaryAttr>
ToKGENOp::buildOutputTypeMethodsDict(KGEN::GeneratorOp mojoKernel,
                                     llvm::ArrayRef<StringRef> methods) {
  KGEN::ParameterExprArrayAttr conformancesAttr =
      dyn_cast_or_null<KGEN::ParameterExprArrayAttr>(
          mojoKernel->getAttr(KGEN::MOGGPreElab::MOGG_RESULT_CONFORMANCES));
  if (!conformancesAttr) {
    return mojoKernel.emitError()
           << "no MOGG type conformance informations found (maybe kernel is "
              "missing a register decorator like `@mogg_register`)";
  }

  auto findMethod =
      [&](StringRef methodName) -> FailureOr<KGEN::VTableEntryAttr> {
    for (TypedAttr attr : conformancesAttr) {
      auto typeAttr = cast<KGEN::TypeConstantAttr>(attr);
      for (KGEN::VTableEntryAttr entry : typeAttr.getVTable().getEntries()) {
        if (entry.getName() == methodName)
          return entry;
      }
    }
    return mojoKernel.emitError()
           << "kernel output type doesn't implement the method `" << methodName
           << "`. Maybe it's missing a trait implementation.";
  };

  // Build the dictionary of methods
  NamedAttrList methodsDict;
  for (StringRef methName : methods) {
    FailureOr<KGEN::VTableEntryAttr> entryOr = findMethod(methName);
    if (failed(entryOr))
      return failure();
    methodsDict.set(methName, entryOr->getMethod());
  }
  return DictionaryAttr::get(mojoKernel.getContext(), methodsDict);
}

TypedAttr ToKGENOp::getMethodOfMojoStruct(StringRef methodName) {
  if (!getTypeMethodsDict().has_value())
    return nullptr;
  DictionaryAttr methodsDict = *getTypeMethodsDict();
  return dyn_cast_or_null<TypedAttr>(methodsDict.get(methodName));
}

//===----------------------------------------------------------------------===//
// ToMOGGOp
//===----------------------------------------------------------------------===//

TypedAttr ToMOGGOp::getMethodOfMojoStruct(StringRef methodName) {
  if (!getTypeMethodsDict().has_value())
    return nullptr;
  DictionaryAttr methodsDict = *getTypeMethodsDict();
  return dyn_cast_or_null<TypedAttr>(methodsDict.get(methodName));
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

void MOGGDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/GraphCompiler/MOGGDialect/MOGG.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MOGGDialect/MOGG.cpp.inc"
