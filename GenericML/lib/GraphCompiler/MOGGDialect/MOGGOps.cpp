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
#include "GenericML/GraphCompiler/Interfaces/LoweringState.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOGGCompiler.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/TypeManager.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/MOGGPreElab/MOGGPreElabHelpers.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/ParserUtils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace M;
using namespace MOGG;

static void printSetDim(OpAsmPrinter &p, Operation *op, StringAttr dim,
                        Value val) {
  KGEN::printParamName(p, dim);
  p << " = ";
  p << val;
}

static ParseResult parseSetDim(OpAsmParser &p, StringAttr &name,
                               mlir::OpAsmParser::UnresolvedOperand &value) {
  if (KGEN::parseParamName(p, name) || p.parseEqual() || p.parseOperand(value))
    return failure();
  return success();
}

static ParseResult parseDim(OpAsmParser &p, StringAttr &name,
                            Type &resultType) {
  SmallVector<Type> tmpTypes;
  if (p.parseLess() || KGEN::parseParamName(p, name) || p.parseGreater() ||
      p.parseArrowTypeList(tmpTypes))
    return failure();
  resultType = tmpTypes[0];
  return success();
}

static void printDim(OpAsmPrinter &p, Operation *op, StringAttr dim,
                     Type resultTy) {
  p << '<';
  KGEN::printParamName(p, dim);
  p << '>';
  p.printArrowTypeList(TypeRange{resultTy});
}

//===----------------------------------------------------------------------===//
// MOGG ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::lower(MGP::MOGraphLoweringState &loweringState) {
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

  Value executeChain =
      opLoweringState.getGuardChain(loweringState.getCurrentExecuteChain());

  Value outChain = opLoweringState.bufferConcat(
      executeChain, loweredOutputsStorage[0].buffer, loweredOperands,
      getResult().getType());

  // Create the output tensors, but only once the op has completed.
  opLoweringState.createOutputs(outChain, loweredOutputsStorage);
  return success();
}

//===----------------------------------------------------------------------===//
// MOGG SplitOp
//===----------------------------------------------------------------------===//

LogicalResult SplitOp::lower(MGP::MOGraphLoweringState &loweringState) {
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

  // Build/collect inputs to the MGP buffer split op
  SmallVector<Value> outputBuffers;

  Value offset = opLoweringState.create<mlir::index::ConstantOp>(0).getResult();
  Value sliceSize;
  for (TypedValue<MGP::TensorSpecType> outSpec : outputSpecs) {
    // Increment the offset for each iteration after the first.
    if (sliceSize)
      offset = opLoweringState.create<mlir::index::AddOp>(offset, sliceSize);

    MGP::TensorSpecType outSpecTy = outSpec.getType();
    std::optional<uint64_t> bufferSize = outSpecTy.getSizeInBytes();

    if (!bufferSize.has_value()) {
      ASSERT_STREAM(outSpecTy.getOptDType().has_value(),
                    << "SplitOp output spec cannot have unknown dtype.");
      sliceSize =
          opLoweringState.bufferSizeBytes(outSpec, getInput().getType());

    } else {
      sliceSize = opLoweringState.create<mlir::index::ConstantOp>(*bufferSize)
                      .getResult();
    }

    Value outSlice =
        opLoweringState
            .create<MGP::BufferSliceOp>(inputBuffer, offset, sliceSize)
            .getResult();
    outputBuffers.push_back(outSlice);
  }

  // The split can only be accessed once it's safe to access the inputs. Though
  // fine to perfom the actual split.
  Value guardChain =
      opLoweringState.getGuardChain(loweringState.getCurrentExecuteChain());

  // Create the output tensors. Note there's no new chain to wait on.
  opLoweringState.createOutputTensors(guardChain, outputBuffers, outputSpecs);
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
  SmallVector<Value> values = {outChain, loweringState.getInitChain()};
  auto combined = builder.create<GRT::ChainCreateOp>(getLoc(), values);
  loweringState.setInitChain(combined);
  return success();
}

//===----------------------------------------------------------------------===//
// Parameter printing/parsing support in form: <P1, P2, P3>
//===----------------------------------------------------------------------===//

static ParseResult parseParamDecls(OpAsmParser &p,
                                   KGEN::ParamDeclArrayAttr &paramDecls) {
  auto defaultType = MOSH::ShapeAttr::getDimType(p.getContext());

  auto declParser = [&](SmallVectorImpl<ParamDeclAttr> &decls) -> ParseResult {
    StringAttr name;
    Type type;
    if (KGEN::parseParamName(p, name) ||
        KGEN::parseColonTypeOrDefault(p, type, defaultType))
      return failure();
    decls.push_back(ParamDeclAttr::get(name, type));
    return success();
  };

  if (succeeded(p.parseOptionalLess()) && failed(p.parseOptionalGreater())) {
    if (KGEN::parseParamDecls(p, paramDecls, declParser) || p.parseGreater())
      return failure();
  } else {
    paramDecls = KGEN::ParamDeclArrayAttr::get(p.getContext(), {});
  }
  return success();
}

static void printParamDecls(OpAsmPrinter &p, Operation *,
                            KGEN::ParamDeclArrayAttr paramDecls) {
  auto defaultType = MOSH::ShapeAttr::getDimType(paramDecls.getContext());

  auto printElt = [&](ParamDeclAttr decl) {
    KGEN::printParamName(p, decl.getName());
    KGEN::printColonTypeOrDefault(p, decl.getType(), defaultType);
  };

  if (!paramDecls.empty()) {
    p << "<";
    printParamDecls(p, paramDecls, printElt);
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

//===----------------------------------------------------------------------===//
// MOGG CallOp
//===----------------------------------------------------------------------===//

void CallOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {

  bool hasWriteEffect = true;
  bool hasReadEffect = true;
  std::string target = getMojoKernelTargetAttr().str();
  if (target == "mgp.tensor.extract.tensor_spec" ||
      target == "mgp.tensor_spec.get_dim" ||
      target == "builtin.unpack_buffer_ref") {
    hasWriteEffect = false;
    hasReadEffect = false;
  }

  if (hasWriteEffect)
    effects.emplace_back(mlir::MemoryEffects::Write::get());
  if (hasReadEffect)
    effects.emplace_back(mlir::MemoryEffects::Read::get());
}

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
// MOGG BindLambdaOp
//===----------------------------------------------------------------------===//

BlockArgument BindLambdaOp::getIndicesArg() {
  return getLambda().getArguments().back();
}

ArrayRef<BlockArgument> BindLambdaOp::getValueArgs() {
  return getLambda().getArguments().take_front(getNumOperands());
}

bool BindLambdaOp::isIdentityLambda() {
  auto yields = cast<MOGG::OutputOp>(getBody()->getTerminator());

  // If all the yields value simply forwards block argument, this is an
  // identity (unfused) lambda.
  return llvm::all_of(yields->getOperands(), [this](Value operand) {
    auto ba = dyn_cast<BlockArgument>(operand);
    return ba && ba.getOwner() == getBody();
  });
}

LogicalResult BindLambdaOp::verify() {
  if (getFusibleTensors().empty())
    return success();

  if (getLambda().getNumArguments() != getFusibleTensors().size() + 1)
    return emitError("Expecting #operands + 1 arguments for lambda");

  SmallVector<Type> valTps;
  Type indicesTp;

  if (getBindKind() == BindLambdaKind::OutputLambda) {
    auto outOp = cast<MOGG::OutputOp>(getBody()->getTerminator());
    ValueRange outVals = outOp->getOperands();
    if (outVals.size() != 2)
      return emitError("Expecting output lambda to yield 2 values");

    // Bind the output value to the output tensor.
    valTps.push_back(outVals.front().getType());
    indicesTp = outVals.back().getType();
  } else {
    // Bind the input value from input tensors.
    valTps = llvm::map_to_vector(getValueArgs(),
                                 [](BlockArgument ba) { return ba.getType(); });
    indicesTp = getIndicesArg().getType();
  }

  for (auto [elemTp, tensorTp] :
       llvm::zip_equal(valTps, getFusibleTensors().getTypes())) {
    if (elemTp != cast<MO::TensorType>(tensorTp).getElementType())
      return emitError(
          "Mismatch between tensor type and fused tensor element type");
  }

  int64_t rank =
      cast<MO::TensorType>(getFusibleTensors().front().getType()).getRank();
  if (auto idxTp = dyn_cast<KGEN::POP::ArrayType>(indicesTp);
      idxTp == nullptr || idxTp.getResolvedSize().value() != rank)
    return emitError("Mismatch between tensor rank and and indices array size");

  return success();
}

ParseResult BindLambdaOp::parse(OpAsmParser &parser, OperationState &result) {
  Region &lambdaRegion = *result.addRegion();

  SmallVector<OpAsmParser::Argument> argInfo;
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  std::string lambdaKind;

  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseKeywordOrString(&lambdaKind) ||
      parser.parseArgumentList(argInfo, OpAsmParser::Delimiter::Paren, true) ||
      parser.parseRegion(lambdaRegion, argInfo, false) || parser.parseColon()) {
    return failure();
  }
  auto bindKind = MOGG::symbolizeBindLambdaKind(lambdaKind);
  if (!bindKind)
    return failure();

  result.addAttribute(
      BindLambdaOp::getBindKindAttrName(mlir::OperationName(
          BindLambdaOp::getOperationName(), parser.getContext())),
      BindLambdaKindAttr::get(parser.getContext(), *bindKind));

  SmallVector<Type, 4> operandTps;
  if (bindKind == BindLambdaKind::OutputLambda) {
    if (parser.parseTypeList(operandTps))
      return failure();
    // Resolve to the same result types.
    result.types = operandTps;
  } else {
    if (parser.parseLParen() || parser.parseTypeList(operandTps) ||
        parser.parseRParen() || parser.parseArrowTypeList(result.types))
      return failure();
  }
  if (parser.resolveOperands(operands, operandTps, {}, result.operands) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  return success();
}

void BindLambdaOp::print(OpAsmPrinter &p) {
  p << " (";
  p.printOperands(getOperands());
  p << ") ";
  p.printKeywordOrString(stringifyBindLambdaKind(getBindKind()));
  p << " (";
  llvm::interleaveComma(getLambda().getArguments(), p,
                        [&p](BlockArgument ba) { p.printRegionArgument(ba); });
  p << ") ";
  p.printRegion(getLambda(), /*printEntryBlockArgs=*/false);

  p << ": ";
  if (getBindKind() == BindLambdaKind::OutputLambda) {
    // To bind an output lambda, it has no input tensor (the value coming from a
    // data stream during kernel execution).
    p.printType(getResult().getType());
  } else {
    p.printFunctionalType(getOperands().getTypes(), (*this)->getResultTypes());
  }

  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          getBindKindAttrName().strref());
}

void BindLambdaOp::build(OpBuilder &b, OperationState &state, Type outputTp,
                         ValueRange fusibleTensors, BindLambdaKind bindKind) {
  OpBuilder::InsertionGuard guard(b);
  build(b, state, outputTp, bindKind, fusibleTensors);

  // Create an identity lambda.
  Block &block = state.regions.front()->emplaceBlock();

  // For each bounded tensor, we use append a block argument for the
  // corresponding loaded value.
  for (Value v : fusibleTensors)
    block.addArgument(cast<MO::TensorType>(v.getType()).getElementType(),
                      v.getLoc());
  // Appends indices
  //
  // TODO: Technically, we only need indices for output lambda (maybe also
  // elementwise lambda) at the moment, but we append it regardless here to
  // support indices transformation in the future, it also saves us a bunch of
  // if statements by having a canonical form.
  Type indicesTp = KGEN::POP::ArrayType::get(
      cast<MO::TensorType>(outputTp).getRank(), b.getIndexType());
  block.addArgument(indicesTp, state.location);

  // Caller need to populate the lambda body.
  if (bindKind == BindLambdaKind::ElemwiseLambda)
    return;

  b.setInsertionPointToStart(&block);
  b.create<MOGG::OutputOp>(state.location, block.getArguments());
}

//===----------------------------------------------------------------------===//
// MOGG KernelTerminatorOp
//===----------------------------------------------------------------------===//

ValueRange CallForeachOp::getKernelOutputs() {
  return getOperands().take_back();
}
ValueRange CallMaterializeOp::getKernelOutputs() {
  return getOperands().take_back();
}
ValueRange CallExecuteOp::getKernelOutputs() { return getOutputs(); }

//===----------------------------------------------------------------------===//
// MOGG ExpKernelOp
//===----------------------------------------------------------------------===//

KernelTerminator ExpKernelOp::getTerminator() {
  return cast<KernelTerminator>(getExecBody().front().getTerminator());
}

BindLambdaOp ExpKernelOp::getBoundInputLambda(unsigned operandIdx) {
  auto executeOp = dyn_cast<CallExecuteOp>(getTerminator());
  return executeOp
             ? executeOp.getInputs()[operandIdx].getDefiningOp<BindLambdaOp>()
             : nullptr;
}

BindLambdaOp ExpKernelOp::getBoundOutputLambda(unsigned resultIdx) {
  auto executeOp = dyn_cast<CallExecuteOp>(getTerminator());
  return executeOp
             ? executeOp.getOutputs()[resultIdx].getDefiningOp<BindLambdaOp>()
             : nullptr;
}

BindLambdaOp ExpKernelOp::getElemwiseBodyLambda() {
  auto foreachOp = dyn_cast<CallForeachOp>(getTerminator());
  return foreachOp ? foreachOp.getInput().getDefiningOp<BindLambdaOp>()
                   : nullptr;
}

bool ExpKernelOp::isOperandFusible(unsigned operandIdx) {
  auto users = getExecBody().getArgument(operandIdx).getUsers();
  // Returns true if the operand is used by a MOGG::BindLambdaOp
  return llvm::find_if(users, [](Operation *op) {
           return isa<MOGG::BindLambdaOp>(op);
         }) != users.end();
}

LogicalResult M::MOGG::ExpKernelOp::verify() {
  auto term =
      dyn_cast<MOGG::KernelTerminator>(getExecBody().front().getTerminator());
  if (!term)
    return emitError("Expect to terminate with a KernelTerminator");

  term.getKernelOutputs().getTypes();
  if (!llvm::equal(term.getKernelOutputs().getTypes(), getResultTypes()))
    return emitError("Expect to return the same result types as terminator");

  return success();
}

void ExpKernelOp::build(OpBuilder &builder, OperationState &state,
                        TypeRange results, ValueRange operands,
                        StringAttr summary, BoolAttr isAsync,
                        DeviceRefAttr device) {
  state.addOperands(operands);

  if (summary)
    state.addAttribute(getSummaryAttrName(state.name), summary);

  if (isAsync)
    state.addAttribute(getIsAsyncAttrName(state.name), isAsync);

  Region *execBodyRegion = state.addRegion();
  Block &kernelBody = execBodyRegion->emplaceBlock();
  for (auto [i, operand] : llvm::enumerate(operands))
    kernelBody.addArgument(operand.getType(), operand.getLoc());

  state.addTypes(results);

  if (device)
    state.addAttribute(MO::kOperationDeviceRefAttrName, device);
}

ParseResult ExpKernelOp::parse(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::Argument> argInfo;
  if (parseParenOperandListWithShadowing(parser, state, argInfo, true) ||
      parser.parseArrowTypeList(state.types) ||
      parseRegion(parser, argInfo, *state.addRegion(), true) ||
      parser.parseOptionalAttrDict(state.attributes))
    return failure();

  return success();
}

void ExpKernelOp::print(OpAsmPrinter &p) {
  p << ' ';
  printParenOperandListWithShadowing(p, getOperands(),
                                     getExecBody().getArguments(),
                                     /*isIsolatedFromAbove=*/true);

  p.printArrowTypeList(getResultTypes());
  p << ' ';
  printRegion(p, getOperands(), getExecBody(), /*isIsolatedFromAbove=*/true);
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

OutputPlaceholdersOp ExpKernelOp::getOutputPlaceholder() {
  auto placeholder = this->getOps<MOGG::OutputPlaceholdersOp>();
  if (placeholder.empty())
    return nullptr;

  assert(llvm::hasSingleElement(placeholder));

  return *placeholder.begin();
}

StackAllocateTensorsOp ExpKernelOp::getStackAllocateTensor() {
  auto stackAlloc = this->getOps<MOGG::StackAllocateTensorsOp>();
  if (stackAlloc.empty())
    return nullptr;

  assert(llvm::hasSingleElement(stackAlloc));

  return *stackAlloc.begin();
}

CreateEmptyTensorsOp ExpKernelOp::getCreateEmptyTensor() {
  auto createEmpty = this->getOps<MOGG::CreateEmptyTensorsOp>();
  if (createEmpty.empty())
    return nullptr;

  assert(llvm::hasSingleElement(createEmpty));

  return *createEmpty.begin();
}

OutputOp ExpKernelOp::getOutputOp() {
  auto output = this->getOps<MOGG::OutputOp>();
  if (output.empty())
    return nullptr;

  assert(llvm::hasSingleElement(output));

  return *output.begin();
}

//===----------------------------------------------------------------------===//
// MOGG KernelOp
//===----------------------------------------------------------------------===//

void KernelOp::build(
    OpBuilder &builder, OperationState &state, TypeRange results,
    ValueRange operands, ArrayRef<ParamDeclAttr> outputParamDecls,
    ArrayAttr inputFusionInterface, ArrayAttr outputFusionInterface,
    StringAttr fusionBodyLambda, BoolAttr isViewOp, StringAttr summary,
    BoolAttr isAsync, BoolAttr computationallyTrivial, ArrayAttr allocIndices,
    bool hasReadEffect, bool hasWriteEffect, DeviceRefAttr device) {
  state.addOperands(operands);

  if (!outputParamDecls.empty())
    state.addAttribute(getOutputParamDeclsAttrName(state.name),
                       builder.getAttr<ParamDeclArrayAttr>(outputParamDecls));

  if (inputFusionInterface)
    state.addAttribute(getInputFusionInterfaceAttrName(state.name),
                       inputFusionInterface);

  if (outputFusionInterface)
    state.addAttribute(getOutputFusionInterfaceAttrName(state.name),
                       outputFusionInterface);

  if (fusionBodyLambda)
    state.addAttribute(getFusionBodyLambdaAttrName(state.name),
                       fusionBodyLambda);

  if (isViewOp)
    state.addAttribute(getIsViewOpAttrName(state.name), isViewOp);

  if (summary)
    state.addAttribute(getSummaryAttrName(state.name), summary);

  if (isAsync)
    state.addAttribute(getIsAsyncAttrName(state.name), isAsync);

  if (computationallyTrivial)
    state.addAttribute(getComputationallyTrivialAttrName(state.name),
                       computationallyTrivial);

  if (allocIndices)
    state.addAttribute(getAllocIndicesAttrName(state.name), allocIndices);

  state.addAttribute(getHasReadEffectAttrName(state.name),
                     builder.getBoolAttr(hasReadEffect));
  state.addAttribute(getHasWriteEffectAttrName(state.name),
                     builder.getBoolAttr(hasWriteEffect));
  (void)state.addRegion();
  state.addTypes(results);

  if (device)
    state.addAttribute(MO::kOperationDeviceRefAttrName, device);
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

  auto voidStar = KGEN::PointerType::get(KGEN::NoneType::get(ctx));

  auto maybeInChain = maybeGetInChain();
  auto maybeOutChain = maybeGetOutChain();

  Value chain = loweringState.getCurrentExecuteChain();
  if (maybeInChain)
    chain = loweringState.lookupExecuteLoweredValue(*maybeInChain);
  chain = opLoweringState.getGuardChain(chain);

  // Gather the lowered op operands.
  SmallVector<Value> loweredInputOperands = [&]() {
    // Dropping the lowered input chain simplifies subsequent lowering logic in
    // this function.
    if (maybeInChain) {
      return SmallVector<Value>{
          llvm::drop_end(opLoweringState.getLoweredOperands())};
    } else
      return opLoweringState.getLoweredOperands();
  }();

  SmallVector<MGP::MOOutputStorage> loweredOutputsStorage;
  FailureOr<SmallVector<MGP::MOOutputStorage>> loweredOutputsStorageOr =
      opLoweringState.createOutputStorage(chain);
  if (failed(loweredOutputsStorageOr))
    return failure();
  loweredOutputsStorage = *std::move(loweredOutputsStorageOr);

  // Handle scratch output placeholders if they exist
  SmallVector<MGP::MOOutputStorage> scratchOutputsStorage;
  MOGG::ScratchPlaceholdersOp scratchOutputs = getScratchPlaceholder();
  if (scratchOutputs) {
    SmallVector<TypedValue<MGP::TensorSpecType>> scratchSpecs;
    for (Type t : scratchOutputs.getResultTypes()) {
      auto tensorType = cast<MO::TensorType>(t);
      auto specResult = opLoweringState.createTensorSpec(tensorType);
      if (failed(specResult))
        return failure();
      scratchSpecs.push_back(*specResult);
    }

    scratchOutputsStorage.reserve(scratchSpecs.size());
    for (auto [spec, type] :
         llvm::zip(scratchSpecs, scratchOutputs.getResultTypes())) {
      auto bufOr = opLoweringState.createBuffer(chain, spec, type);
      if (failed(bufOr))
        return failure();
      scratchOutputsStorage.emplace_back(*bufOr);
    }
  }

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
    if (!oldOutputs || i >= oldOutputs->getNumResults())
      returnAllocIndicesSet.insert(i);
    if (!inplaceAllocIndicesSet.contains(i)) {
      if (!isa<MO::ChainType>(getResult(i).getType()))
        preAllocIndices.push_back(i);
    } else
      inplaceAllocIndices.push_back(i);
  }

  size_t numInplaceOutputs = inplaceAllocIndicesSet.size();

  size_t numOperands = 1 + loweredInputOperands.size() +
                       loweredOutputsStorage.size() +
                       scratchOutputsStorage.size();

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

  for (auto &&[origOperand, loweredInputOperand] :
       llvm::zip(getOperands(), loweredInputOperands)) {

    auto sig = isa<MO::BufferType>(origOperand.getType())
                   ? InOutSignatureAttr::kMut
                   : InOutSignatureAttr::kIn;

    operands.push_back(loweredInputOperand);
    operandSemantics.emplace_back(sig);
  }

  // Add scratch output buffers to operands (after input operands)
  if (scratchOutputs) {
    for (auto &scratchStorage : scratchOutputsStorage) {
      operands.push_back(scratchStorage.buffer);
      operandSemantics.emplace_back(InOutSignatureAttr::kNone);
    }
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

  DeviceRefAttr device = MO::getDeviceOrDefaultAny(*this);
  auto mgpBufferType = MGP::DeviceBufferType::get(device);

  SmallVector<mlir::Type> inPlaceMGPOutputsTypes(numInplaceOutputs);
  for (auto [idx, inPlaceIdx] : llvm::enumerate(inplaceAllocIndices)) {
    Type oldOutTy = getResult(inPlaceIdx).getType();
    // mgp.buffer is user for mo.tensor, mgp.opaque for mo.opaque.
    Type mgpResultTy = mgpBufferType;
    if (auto moOpaqueType = dyn_cast<MO::OpaqueType>(oldOutTy))
      mgpResultTy =
          MGP::OpaqueType::get(ctx, moOpaqueType.getSymbol(), std::nullopt);
    ASSERT_STREAM(!isa<MO::ListType>(oldOutTy), "in place list not supported");
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
    newKernelResTypes.push_back(getOutputParamDecls()[i].getType());
    StringRef name = getOutputParamDecls()[i].getName();
    outParamNameToIdx[name] = i;
  }

  // TODO: Fusing kernels with inplace allocations is not supported
  // (intermediates allocations unsupported / leaking).
  bool hasInternalAllocs =
      !(getAllocIndices().has_value() && !getAllocIndices()->empty());

  // Allow all kernels to be fused into other kernels.
  bool canFuseIntoOther = !hasInternalAllocs;

  // Later update the isAsync instead of hard-coding
  auto newKernel = builder.create<MOGG::GenericCompiledRegionOp>(
      getLoc(), TypeRange{newKernelResTypes}, operands,
      getOutputParamDeclsAttr(), InOutSignatureAttr::get(ctx, operandSemantics),
      getSummaryAttr(),                                 // summary
      BoolAttr::get(ctx, getIsAsync().value_or(false)), // isAsync
      ArrayAttr::get(ctx, {}), // KernelOp has no constant resources
      BoolAttr::get(ctx, canFuseIntoOther), // canFuseIntoOther
      BoolAttr::get(ctx, true),             // canBeFusedInto
      BoolAttr::get(ctx, false)             // canDuplicate
  );
  newKernel.setDeviceRef(device);

  mlir::Block &kernelBody = newKernel.getBody().emplaceBlock();
  builder.setInsertionPoint(&kernelBody, kernelBody.begin());
  SmallVector<Location> locs =
      llvm::map_to_vector(operands, [](Value v) { return v.getLoc(); });
  kernelBody.addArguments(ValueRange(operands).getTypes(), locs);

  DenseMap<TypedAttr, Value> loweredParams;
  auto addParams = [&](Location loc, Type t) -> CompileDiagnosticOrSuccess {
    ArrayRef<TypedAttr> dims;
    if (auto asTensor = MO::dynCastFromTensorLike(t))
      dims = asTensor.getDims();
    else if (auto shape = dyn_cast<MOGG::ShapeType>(t))
      dims = dyn_cast<MOSH::ShapeAttr>(shape.getShapeAttr()).getValues();
    else
      return success();

    for (auto dim : dims) {
      if (loweredParams.contains(dim))
        continue;

      if (auto cst = dyn_cast<IntegerAttr>(dim)) {
        KGEN::ParamConstantOp op =
            builder.create<KGEN::ParamConstantOp>(loc, cst);
        loweredParams[dim] = op.getResult();
      } else if (auto ref = dyn_cast<KGEN::ParamDeclRefAttr>(dim)) {
        loweredParams[dim] =
            kernelBody.getArgument(paramDimToIdx[ref.getName()]);
        loweredParams[ref.getName()] =
            kernelBody.getArgument(paramDimToIdx[ref.getName()]);
      } else {
        return mlir::emitError(loc, "Unsupported dim attribute type: ") << dim;
      }
    }
    return success();
  };

  // Look at all the GetDims to pick up any parameters which are only referenced
  // by get dim.
  this->walk([&](MOGG::GetDimOp getDim) {
    StringAttr name = getDim.getDimension();
    auto itr = loweredParams.find(name);
    if (itr == loweredParams.end()) {
      loweredParams[name] = kernelBody.getArgument(paramDimToIdx[name]);
    }
  });

  // Add all the seen shape parameters values as concrete KGEN param values.
  for (auto arg : this->getBodyRegion().getArguments()) {
    auto err = addParams(arg.getLoc(), arg.getType());
    if (err.isError())
      return err.takeError();
  }
  for (auto &op : this->getOps()) {
    for (auto type : op.getOperandTypes()) {
      auto err = addParams(op.getLoc(), type);
      if (err.isError())
        return err.takeError();
    }
    for (auto type : op.getResultTypes()) {
      auto err = addParams(op.getLoc(), type);
      if (err.isError())
        return err.takeError();
    }
  }

  // Convert MGP block input to tensor object (!mo.tensor) or opaque.
  auto convertMGPInputToTensorOrOpaque = [&](Value input, Type oldTy) -> Value {
    // Opaque types are already pointer to objects, they don't need any
    // transformation to be passed to the kernel.
    if (isa<MGP::OpaqueType, MGP::DeviceContextType, mlir::IndexType>(
            input.getType())) {
      return input;
    } else if (isa<MO::TensorType, MO::BufferType>(oldTy)) {
      MO::TensorType asTensor = MO::castFromTensorLike(oldTy);

      Value dataPtr = builder.create<MOGG::DataPointerFromBufferOp>(
          input.getLoc(), voidStar, input);

      SmallVector<Value> operands;
      operands.push_back(dataPtr);
      for (auto dim : asTensor.getDims()) {
        auto loweredParam = loweredParams.at(dim);
        if (!loweredParam.getType().isIndex()) {
          loweredParam = builder.create<mlir::index::CastSOp>(
              input.getLoc(), builder.getIndexType(), loweredParam);
        }

        operands.push_back(loweredParam);
      }

      return builder.create<MOGG::ToTensorFromMGPOp>(input.getLoc(), asTensor,
                                                     operands);
    } else if (auto asListTy = dyn_cast<MO::ListType>(oldTy)) {
      // Assume the new extensibility API uses KGEN types.
      return builder.create<MOGG::ToListFromMGPOp>(input.getLoc(), oldTy,
                                                   input);
    } else {
      ASSERT_STREAM(false, << "Unknown type " << oldTy << " with value "
                           << input);
      llvm_unreachable("[convertMGPInputToMOValue] Unknown type.");
      return {};
    }
  };

  IRMapping mapper;

  int outputOffset = 1; // 1 for the input chain
  for (auto [idx, preAllocIdx] : llvm::enumerate(preAllocIndices)) {
    Value newBlockArg = kernelBody.getArgument(outputOffset + idx);

    // Order of pre-allocated result memory matches that of original outputs.
    Value oldOut = getOutputOp()->getOperand(preAllocIdx);

    // TODO(GEX-1183): Another hack from inconsistency in opaque lowering.
    // Sometimes we return the KGEN type directly, sometimes not depending
    // on whether we are in the old and new pipelines.
    if (isa<KGEN::StructType>(oldOut.getType())) {
      while (!isa<OutputPlaceholdersOp>(oldOut.getDefiningOp())) {
        Operation *definingOp = oldOut.getDefiningOp();

        // Only expect ops like "convert_to_ndbuffer" or "mogg.to_kgen"
        // which all have a single operand and just transform types.
        ASSERT_STREAM(definingOp->getNumOperands() == 1,
                      << "Expect one operand.");
        oldOut = definingOp->getOperand(0);
      }
    }

    ASSERT_STREAM(
        isa<OutputPlaceholdersOp>(oldOut.getDefiningOp()),
        << "Expect all pre-allocated outputs to be in `OutputPlaceholdersOp`.");

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

  // Map scratch outputs if they exist
  if (scratchOutputs) {
    int scratchOffset = inputOffset + loweredInputOperands.size();
    for (auto [idx, result] : llvm::enumerate(scratchOutputs.getResults())) {
      Value newBlockArg = kernelBody.getArgument(scratchOffset + idx);
      Value oldOut = result;
      Value loweredValue =
          convertMGPInputToTensorOrOpaque(newBlockArg, oldOut.getType());
      mapper.map(oldOut, loweredValue);
    }
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
      SmallVector<int64_t> dynamicShape(asTensor.getRank(), 1);
      auto dynamicTensorTy =
          MO::TensorType::get(dynamicShape, asTensor.getElementType());
      internalTensorAllocTypes.push_back(dynamicTensorTy);
    } else if (isa<MO::OpaqueType>(oldOut.getType())) {
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

  auto si64ScalarTy = KGEN::POP::SIMDType::get(ctx, 1, DType::si64);

  // Clone the graph
  for (Operation &op : getBody().front()) {
    if (isa<MOGG::OutputPlaceholdersOp, MOGG::OutputOp,
            MOGG::ScratchPlaceholdersOp>(op))
      continue;

    if (auto getDim = dyn_cast<MOGG::GetDimOp>(op)) {
      auto dim = loweredParams.lookup(getDim.getDimension());
      Value v = builder.create<KGEN::POP::CastFromBuiltinOp>(op.getLoc(),
                                                             si64ScalarTy, dim);
      mapper.map(getDim.getResult(), v);
      continue;
    }

    if (auto createEmptyOp = dyn_cast<MOGG::CreateEmptyTensorsOp>(op)) {
      for (Value val : createEmptyOp->getResults()) {
        auto tensorTy = cast<MO::TensorType>(val.getType());

        SmallVector<Value> operands;
        // We will allocate a stack of 1 element in place of a nullptr. This
        // should never be read and the tensor considered 'empty'. I.E just a
        // stride, type and size with a nullptr attached.
        auto voidStar =
            KGEN::PointerType::get(KGEN::NoneType::get(builder.getContext()));
        Value dataPtr =
            builder.create<KGEN::POP::StackAllocationOp>(op.getLoc(), voidStar);
        operands.push_back(dataPtr);
        for (auto dim : tensorTy.getDims()) {
          auto loweredParam = loweredParams.at(dim);
          if (!loweredParam.getType().isIndex()) {
            loweredParam = builder.create<mlir::index::CastSOp>(
                op.getLoc(), builder.getIndexType(), loweredParam);
          }
          operands.push_back(loweredParam);
        }

        Value newTensor = builder.create<MOGG::ToTensorFromMGPOp>(
            op.getLoc(), tensorTy, operands);
        mapper.map(val, newTensor);
      }
      continue;
    }

    // Make sure to replace get dim ops in regions / lambdas.
    // Create an association between GetDim ops in the old kernelOp and their
    // lowered form in the output kernelOp.
    SmallVector<Value> loweredGetDimsNewOp;
    if (auto regionDeclarationOldOp =
            dyn_cast<KGEN::ParamDeclareRegionOp>(op)) {
      for (auto op : regionDeclarationOldOp.getOps<MOGG::GetDimOp>()) {
        auto loweredParam = loweredParams.lookup(op.getDimension());
        Value v = builder.create<KGEN::POP::CastFromBuiltinOp>(
            op.getLoc(), si64ScalarTy, loweredParam);
        loweredGetDimsNewOp.push_back(v);
      }
    }

    // Replace all usages of GetDim ops in the new kernelOp using our map.
    auto newOp = builder.clone(op, mapper);
    if (auto regionDeclarationNewOp =
            dyn_cast<KGEN::ParamDeclareRegionOp>(newOp)) {
      for (auto [i, op] :
           llvm::enumerate(regionDeclarationNewOp.getOps<MOGG::GetDimOp>())) {
        op.replaceAllUsesWith(loweredGetDimsNewOp[i]);
      }
    }
  }

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
      // TODO(GEX-1183): Stop hack which is needed only in new extensibility
      // path Essentially kgenVal isn't the kgen val but instead the initial
      // stack alloc for memory-types.
      //
      // But this requires fixing up this 500 line function.
      if (isa<MO::OpaqueType>(kgenVal.getType())) {
        auto users = kgenVal.getUsers();
        for (auto op : users) {
          if (auto asKGEN = dyn_cast<MOGG::ToKGENOp>(op)) {
            kgenVal = asKGEN.getResult();
            break;
          }
        }
      }

      outputs[idx + numPrependedOutputs] =
          builder.create<MOGG::ToMGPFromOpaqueOp>(getLoc(), TypeRange{resultTy},
                                                  kgenVal);
    } else {
      outputs[idx + numPrependedOutputs] =
          builder.create<MOGG::ToMGPFromTensorOp>(getLoc(), TypeRange{resultTy},
                                                  kgenVal);
    }
  }

  // Elide all set dims so they simply return the value.
  llvm::StringMap<Value> paramToDim;
  SmallVector<MOGG::SetParameterOp> toRemove;
  for (auto setParam : newKernel.getOps<MOGG::SetParameterOp>()) {
    paramToDim[setParam.getParamName()] = setParam.getDim();
    toRemove.push_back(setParam);
  }

  // outParamNum: Add all the params from stringMap paramToDim so do lookup
  llvm::BitVector needCast(outputs.size());
  for (const auto &[name, idx] : outParamNameToIdx) {
    size_t outIdx = idx + numPrependedOutputs + numInplaceOutputs;
    Type retTp = newKernel->getResultTypes()[outIdx];
    Value outVal = paramToDim[name];
    // Output for parameters might have been inferred to a different types when
    // creating the new kernel. Here, we try to fix the type mismatch between
    // scalar<builtin> to builtin.
    //
    // NOTE: we only record the indices of the value that needs to be casted
    // here and insert the cast operation later to make sure that the inserted
    // pop.cast_to_builtin appears in a deterministic order (i.e., in a index
    // increasing order). The current loop order depends on the the storage
    // order of the hashmap and an inconsistent order might result in caches
    // missing during JitCompilerKernel pass, leading to worse compilation time.
    if (auto scalarOut = dyn_cast<KGEN::POP::SIMDType>(outVal.getType());
        scalarOut && retTp != scalarOut)
      needCast.set(outIdx);

    outputs[outIdx] = outVal;
  }

  for (auto idx : needCast.set_bits()) {
    Value &toCast = outputs[idx];
    toCast = builder.create<KGEN::POP::CastToBuiltinOp>(
        toCast.getLoc(), newKernel->getResultTypes()[idx], toCast);
  }

  for (MOGG::SetParameterOp p : toRemove)
    p.erase();

  // Construct a mogg.output that newKernel spec
  builder.create<MOGG::OutputOp>(getLoc(), outputs);

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

  if (maybeOutChain) {
    loweringState.emplaceExecuteLoweredValue(*maybeOutChain,
                                             newKernel.getResult(0));
  }

  // Replace any device context placeholder with the actual context.
  newKernel.walk([&](MOGG::DeviceContextPlaceholderOp placeholdCtx) {
    OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPoint(placeholdCtx);

    Value old = placeholdCtx.getResult();
    Value actualCtx = loweringState.getExecuteDevice(placeholdCtx.getDevice());
    newKernel->insertOperands(newKernel.getNumOperands(), {actualCtx});

    Value asArg =
        kernelBody.addArgument(actualCtx.getType(), actualCtx.getLoc());

    builder.setInsertionPoint(placeholdCtx);
    Value toKGEN =
        builder.create<ToKGENOp>(actualCtx.getLoc(), old.getType(), asArg);
    placeholdCtx.replaceAllUsesWith(toKGEN);
    placeholdCtx.erase();

    // Update the kernel InOut semantics if needed.
    if (newKernel.getInOutSignature().size() != 0) {
      SmallVector<InOutSignatureAttr::InOutSemantics> newSemantics(
          1, InOutSignatureAttr::InOutSemantics::kNone);
      InOutSignatureAttr oldSemantics = newKernel.getInOutSignature();
      newKernel.setInOutSignatureAttr(oldSemantics.append(newSemantics));
    }
  });

  return success();
}

OutputPlaceholdersOp KernelOp::getOutputPlaceholder() {
  auto placeholder = this->getOps<MOGG::OutputPlaceholdersOp>();
  if (placeholder.empty())
    return nullptr;

  assert(llvm::hasSingleElement(placeholder));

  return *placeholder.begin();
}

ScratchPlaceholdersOp KernelOp::getScratchPlaceholder() {
  auto scratch = this->getOps<MOGG::ScratchPlaceholdersOp>();

  if (scratch.empty())
    return nullptr;

  assert(llvm::hasSingleElement(scratch));

  return *scratch.begin();
}

StackAllocateTensorsOp KernelOp::getStackAllocateTensor() {
  auto stackAlloc = this->getOps<MOGG::StackAllocateTensorsOp>();
  if (stackAlloc.empty())
    return nullptr;

  assert(llvm::hasSingleElement(stackAlloc));

  return *stackAlloc.begin();
}

CreateEmptyTensorsOp KernelOp::getCreateEmptyTensor() {
  auto createEmpty = this->getOps<MOGG::CreateEmptyTensorsOp>();
  if (createEmpty.empty())
    return nullptr;

  assert(llvm::hasSingleElement(createEmpty));

  return *createEmpty.begin();
}

OutputOp KernelOp::getOutputOp() {
  auto output = this->getOps<MOGG::OutputOp>();
  if (output.empty())
    return nullptr;

  assert(llvm::hasSingleElement(output));

  return *output.begin();
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

/// Add an operand to the kernel at the idx specified by insertIdx and adds an
/// argument to the kernel region. Fixes up the input fusion interfaces to
/// disallow input fusion on this position. Returns the added kernel argument.
Value KernelOp::addOperandAtIndex(Value operand, size_t insertIdx) {
  SmallVector<Value> values = {operand};
  (*this)->insertOperands(insertIdx, values);

  Value blockArg = getBody().front().insertArgument(
      insertIdx, operand.getType(), operand.getLoc());

  std::optional<ArrayAttr> inputInterfaces = getInputFusionInterface();

  // Update the input fusion interfaces if we have them to remove fusion on
  // this attr.
  if (inputInterfaces.has_value()) {
    SmallVector<Attribute> lambdaNames;
    for (StringAttr name : inputInterfaces->getAsRange<StringAttr>())
      lambdaNames.push_back(name);

    // Add an empty lambda name for the operand we just added.
    lambdaNames.insert(lambdaNames.begin() + insertIdx,
                       StringAttr::get(getContext(), ""));
    setInputFusionInterfaceAttr(ArrayAttr::get(getContext(), lambdaNames));
  }

  return blockArg;
}

/// Add an operand to the kernel and adds an argument to the kernel region.
/// Fixes up the input fusion interfaces to disallow input fusion on this
/// position. Returns the added kernel argument.
Value KernelOp::addOperand(Value operand) {
  // TODO(GEX-868): Make mogg.kernel lowering work with an input chains that
  // aren't the final operand.
  //
  // For now, keep the chain as last if it exists.
  if (maybeGetInChain().has_value())
    return addOperandAtIndex(operand, getNumOperands() - 1);

  return addOperandAtIndex(operand, getNumOperands());
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
    } else if (auto call = dyn_cast<CallOp>(op)) {
      // Helper lambda to translate remap the lambda name.
      auto translateLambdaSym = [&](SymbolRefAttr sym) {
        StringRef name = sym.getLeafReference().strref();
        auto itr = oldLambdaNamesToNew.find(name);
        ASSERT_STREAM(
            itr != oldLambdaNamesToNew.end(),
            << "First use of lambda inserted before the lambda itself");

        return FlatSymbolRefAttr::get(getContext(), itr->second);
      };

      // Lambdas will have been inlined. The only place we can have
      // a lambda reference is in a call.
      auto newCall = dyn_cast<CallOp>(builder.clone(op, remapper));
      SmallVector<Attribute> newAttrs;

      // Remap any lambda references to point to the new lambda name.
      std::optional<ArrayAttr> params = newCall.getParams();
      if (params) {
        for (auto symOrArray : params->getAsRange<Attribute>()) {
          if (auto sym = dyn_cast<SymbolRefAttr>(symOrArray)) {
            newAttrs.push_back(translateLambdaSym(sym));
          } else {
            // Must be a tuple of lambda for tuple inputs. (e.g., concat).
            SmallVector<Attribute> newTuple;
            auto arraySym = cast<ArrayAttr>(symOrArray);
            for (auto sym : arraySym.getAsRange<SymbolRefAttr>())
              newTuple.push_back(translateLambdaSym(sym));
            // Repack the lambda tuple.
            newAttrs.push_back(builder.getArrayAttr(newTuple));
          }
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
  if (alloc.has_value() && !alloc->getValue().empty())
    return true;
  return false;
}

bool KernelOp::isInPlace() {
  return maybeGetInChain().has_value() && maybeGetOutChain().has_value();
}

MO::ChainValue KernelOp::getInChain() {
  auto inChain = maybeGetInChain();
  ASSERT_STREAM(inChain.has_value(), "Expected to find an input chain.");

  return inChain.value();
}

MO::ChainValue KernelOp::getOutChain() {
  auto outChain = maybeGetOutChain();
  ASSERT_STREAM(outChain.has_value(), "Expected to find an output chain.");

  return outChain.value();
}

OpOperand &KernelOp::getInChainMutable() {
  OpOperand &finalOpOperand = getOperation()->getOpOperands().back();

  ASSERT_STREAM(isa<MO::ChainValue>(finalOpOperand.get()),
                "Expected to find an input chain as the final operand.");

  return finalOpOperand;
}

std::optional<MO::ChainValue> KernelOp::maybeGetInChain() {
  if (getNumOperands() == 0)
    return std::nullopt;

  auto finalVal = getOperands().back();

  if (auto chain = dyn_cast<MO::ChainValue>(finalVal))
    return chain;
  else
    return std::nullopt;
}

std::optional<MO::ChainValue> KernelOp::maybeGetOutChain() {
  if (getNumResults() == 0)
    return std::nullopt;

  auto finalVal = getResults().back();

  if (auto chain = dyn_cast<MO::ChainValue>(finalVal))
    return chain;
  else
    return std::nullopt;
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

  if (llvm::range_size(getOps<MOGG::ScratchPlaceholdersOp>()) > 1)
    return emitError("A KernelOp must have at most one scratch placeholder op");

  auto inputInterfaces = getInputFusionInterface();
  auto outputInterfaces = getOutputFusionInterface();

  if (getOutputOp()->getOperandTypes() != getResultTypes()) {
    return emitError("Output types do not match what is returned by body. Body "
                     "returns: ")
           << getOutputOp()->getOperandTypes() << " but got "
           << getResultTypes();
  }

  OutputOp outputOp = getOutputOp();

  for (auto output : outputOp.getOperands()) {
    if (auto scratchOp = output.getDefiningOp<MOGG::ScratchPlaceholdersOp>()) {
      return emitError(
          "Scratch output tensors cannot be used as outputs of the kernel");
    }
  }

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

  auto isChain = [](Value val) { return isa<MO::ChainType>(val.getType()); };

  auto chainIsLastIfPresent = [&isChain](auto &&rng) {
    auto chainIter = llvm::find_if(rng, isChain);

    if (chainIter == rng.end())
      return true;

    return std::next(chainIter) == rng.end();
  };

  if (!chainIsLastIfPresent(getOperands()))
    return emitError("Found a chain operand that is not in the final position");

  if (!chainIsLastIfPresent(getResults()))
    return emitError("Found an output chain that is not in the final position");

  bool hasMemoryEffects = getHasReadEffect() || getHasWriteEffect();
  if (hasMemoryEffects) {
    if (llvm::count_if(getOperands(), isChain) != 1)
      return emitError(
          "Kernels with memory effects must have exactly one input chain");

    if (llvm::count_if(getResults(), isChain) != 1)
      return emitError(
          "Kernels with MemoryEffects must have exactly one output chain");

    auto nMutOperands = llvm::count_if(getOperands(), MO::isMutableOperand);
    if (nMutOperands == 0) {
      return emitError("Kernel has no mutable operands but either "
                       "hasReadEffect or hasWriteEffect is set to true\n");
    }
  }

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

void KernelOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {

  bool hasWriteEffect = false;
  bool hasReadEffect = false;

  if (getHasReadEffect())
    hasReadEffect |= true;

  if (getHasWriteEffect())
    hasWriteEffect |= true;

  if (!getOutputParamDecls().empty())
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());

  // Right now this will apply read/write effects to ALL mutable operands if the
  // hasReadEffect/hasWriteEffect flag is true.
  // TODO(GEX-1516): Establish fine-grained tracking of mem-effects for
  // mogg.kernel operands and maintain it through transforms.
  for (auto &&opOperand : getOperation()->getOpOperands()) {
    if (!MO::isMutableOperand(opOperand.get()))
      continue;
    if (hasWriteEffect)
      effects.emplace_back(mlir::MemoryEffects::Write::get(), &opOperand);
    if (hasReadEffect)
      effects.emplace_back(mlir::MemoryEffects::Read::get(), &opOperand);
  }
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

LogicalResult GenericCompiledRegionOp::verify() {
  Region &body = getBody();

  if (getOperands().size() != body.getNumArguments())
    return emitError("number of operands and body arguments must match");

  // There are a large number of unittests which use an empty
  // InOutSignatureAttr. For now, allow them to pass verification.
  // TODO: Update unittests and remove this guard.
  if (getInOutSignature().size() != 0) {
    if (getOperands().size() != getInOutSignature().size()) {
      return emitError() << "inOutSignature size ("
                         << getInOutSignature().size()
                         << ") does not match number of operands ("
                         << getOperands().size() << ")";
    }
  }

  if (!llvm::equal(getOperandTypes(), body.getArgumentTypes()))
    return emitError("operand types must match region arguments");

  if (!getDeviceRef())
    return emitError("Compiled regions must have a device");

  return success();
}

LogicalResult GenericCompiledRegionOp::canonicalize(GenericCompiledRegionOp op,
                                                    PatternRewriter &rewriter) {
  // Canonicalize sequences of grt.chain.create + mogg.generic.compiled.region
  //
  //     %ch = grt.chain.create(%0, %1)
  //     mogg.generic.compiled.region(%ch)
  //
  // to
  //
  //     mogg.generic.compiled.region(%0, %1)
  Region &body = op.getBody();

  llvm::BitVector blockArgsToRemove(op.getNumOperands());
  SmallVector<Value> operandsToKeep;
  SmallVector<Value> newOperands;

  // Collect all operands from grt.chain.create and their corresponding
  // block arguments when the block argument is unused.
  for (OpOperand &operand : op.getOperandsMutable()) {
    BlockArgument bodyArgument = body.getArgument(operand.getOperandNumber());

    auto chainCreateOp = operand.get().getDefiningOp<GRT::ChainCreateOp>();
    // Only remove grt.chain.create operands whose corresponding body
    // argument is unused.
    if (!chainCreateOp || !bodyArgument.getUses().empty()) {
      operandsToKeep.push_back(operand.get());
      continue;
    }

    blockArgsToRemove.set(bodyArgument.getArgNumber());
    newOperands.append(chainCreateOp.getOperands().begin(),
                       chainCreateOp.getOperands().end());
  }

  if (blockArgsToRemove.none())
    return failure();

  SmallVector<Type> newTypes =
      llvm::map_to_vector(newOperands, [](Value v) { return v.getType(); });

  SmallVector<Location> newLocs =
      llvm::map_to_vector(newOperands, [](Value v) { return v.getLoc(); });

  rewriter.modifyOpInPlace(op, [&]() {
    // Update the operands to [operandsToKeep; newOperands]
    op.getOperandsMutable().clear();
    op.getOperandsMutable().append(operandsToKeep);
    op.getOperandsMutable().append(newOperands);

    // Update the body block arguments
    // Erase block that correspond to the removed chain arguments
    Block &entryBlock = *op.getBody().begin();
    entryBlock.eraseArguments(blockArgsToRemove);
    entryBlock.addArguments(newTypes, newLocs);

    // Only update the inout signature if we actually have an InOutSignature.
    if (op.getInOutSignature().size() != 0) {
      // Update the InOutSignatureAttr by removing the entries corresponding
      // to removed arguments and appending a `kNone` for each new argument.
      SmallVector<InOutSignatureAttr::InOutSemantics> newSemantics(
          newOperands.size(), InOutSignatureAttr::InOutSemantics::kNone);

      InOutSignatureAttr oldSemantics = op.getInOutSignature();
      op.setInOutSignatureAttr(
          oldSemantics.remove(blockArgsToRemove).append(newSemantics));
    }
  });

  return success();
}

//===----------------------------------------------------------------------===//
// ToTensorFromMGPOp
//===----------------------------------------------------------------------===//

LogicalResult ToTensorFromMGPOp::verify() {
  auto devBuf = dyn_cast<MGP::DeviceBufferType>(getDataPointer().getType());
  if (!devBuf)
    return success();

  // If have access to device buffer, do some sanity checks
  MO::TensorValue result = getResult();
  DeviceRefAttr tensorDev = result.getType().getDeviceRef();

  // TODO(GEX-1212): Make sure tensor type has device assignment at this
  // stage.
  if (tensorDev && tensorDev != devBuf.getDevice())
    return emitError() << "Device ref of buffer and tensor do not match";

  return success();
}

//===----------------------------------------------------------------------===//
// AlignmentPlaceholderOp
//===----------------------------------------------------------------------===//

void AlignmentPlaceholderOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  walkDef(getParamDecl(), {});
}

//===----------------------------------------------------------------------===//
// GetResourcePtrOp
//===----------------------------------------------------------------------===//

LogicalResult GetResourcePtrOp::verify() {
  auto parent = cast<GenericCompiledRegionOp>(getParentOp());
  if (getResourceSlot().getZExtValue() >= parent.getConstantResources().size())
    return emitError("Request resources slot out of bound.");
  return success();
}

//===----------------------------------------------------------------------===//
// ToKGENOp
//===----------------------------------------------------------------------===//

LogicalResult ToKGENOp::verify() {
  if (auto attr = getTensorIOTypeAttr()) {
    auto ref = attr.strref();

    bool valid = KGEN::MOGGPreElab::toIOSpec(ref).has_value();

    if (!valid) {
      return emitError() << "Invalid tensorIOType string attr: " << ref
                         << " -- expected one of: { "
                         << KGEN::MOGGPreElab::kInputTensor << ", "
                         << KGEN::MOGGPreElab::kOutputTensor << ", "
                         << KGEN::MOGGPreElab::kFusedInputTensor << ", "
                         << KGEN::MOGGPreElab::kFusedOutputTensor << ", "
                         << KGEN::MOGGPreElab::kMutableInputTensor << " }";
    }
  }

  return success();
}

KGEN::ParamDeclRefAttr ToKGENOp::getStaticTensorRefAttr() {
  StringAttr name = getStaticTensorSpecNameAttr();
  ASSERT_STREAM(name != nullptr, << "staticTensorSpecName attr is null");

  // ToKGEN should always be in the outermost block of the kernel
  // And same for the ParamDeclareOp.
  for (auto declareOp :
       getOperation()->getBlock()->getOps<KGEN::ParamDeclareOp>()) {
    if (declareOp.getParamDecl().getName() == name)
      return KGEN::ParamDeclRefAttr::get(declareOp.getParamDecl());
  }

  ASSERT_STREAM(false, << "KGEN::ParamDeclareOp not found in kernel body for "
                          "staticTensorSpec named "
                       << name);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// StackAllocateTensorsOp
//===----------------------------------------------------------------------===//

LogicalResult StackAllocateTensorsOp::verify() {
  auto isCpuTensor = [](Type t) {
    auto tensor = cast<MO::TensorType>(t);
    return !tensor.getDeviceRef() ||
           tensor.getDeviceRef().getLabel() == kCPULabel;
  };

  if (!llvm::all_of(getResultTypes(), isCpuTensor))
    return emitError() << "stack allocated tensors must be on the CPU";

  return success();
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

using PointerType = KGEN::PointerType;
#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MOGGDialect/MOGG.cpp.inc"
