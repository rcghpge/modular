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
#include "GenericML/GraphCompiler/Interfaces/EmitMojoState.h"
#include "GenericML/GraphCompiler/Interfaces/LoweringState.h"
#include "GenericML/GraphCompiler/Interfaces/MOGGKernelOpLoweringState.h"
#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/TypeManager.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/MOGGPreElab/MOGGPreElabHelpers.h"
#include "KGEN/MOGGPreElab/MojoKernelAdaptor.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Support/AssertStream.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringMap.h"
#include <future>

using namespace M;
using namespace MOGG;

template <typename T>
static auto stringifyMLIRObject(T entity) {
  std::string str;
  llvm::raw_string_ostream stream(str);
  entity.print(stream);

  return str;
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
// MOGG ToMOGG
//===----------------------------------------------------------------------===//

LogicalResult ToMOGGOp::canonicalize(ToMOGGOp toMOGG,
                                     mlir::PatternRewriter &rewriter) {
  // simply fold %b = mogg.to_mogg(mogg.to_kgen(%a)) to %a, this ensures that
  // output_placeholder will be directly returned by mogg.output, which is an
  // implicit pattern exploits by many passes.
  //
  // TODO: we should probably make those passes more robust by analyzing data
  // flow more carefully.
  if (auto toKGEN = toMOGG.getInput().getDefiningOp<MOGG::ToKGENOp>()) {
    rewriter.replaceOp(toMOGG, toKGEN.getInput());
    return success();
  }

  return failure();
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
  // fine to perform the actual split.
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
  auto combined = GRT::ChainCreateOp::create(builder, getLoc(), values);
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

MojoValue callFunctionNestedInStruct(KGEN::LIT::FnOp function,
                                     EmitMojoState::InferedParameters params,
                                     ArrayRef<MojoValue> operands,
                                     EmitMojoState &state) {
  auto prefixSymbolName = state.getOriginalMojoNamePrefixFor(function);
  MojoValue executeFunc(state.getOriginalMojoNameSuffixFor(function));
  MojoValue structType = prefixSymbolName[params.structParameters];
  MojoValue fullExecuteFunc = structType.get(executeFunc);

  return fullExecuteFunc(params.functionParameters, operands);
}

LogicalResult CallOp::emitMojo(EmitMojoState &state) {
  KGEN::LIT::FnOp function =
      state.analysis.getRegisteredLITMojoKernelFor(getMojoKernelTarget());
  auto symbolName = state.getOriginalMojoNameFor(function);

  SmallVector<MojoValue> mojoOperands = state.getMojoValuesFor(getOperands());

  // Infer the kernel parameters.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    return state.inferDefaultKernelParamValue(*this, paramName,
                                              getKgenParamsAttr());
  };
  auto maybeParams = EmitMojoState::inferKernelParameters(function, inferParam);
  if (maybeParams.isError())
    return emitError("Failed to infer all the kernel parameters");

  auto results = getResults();
  if (results.empty()) {
    state.builder.call(symbolName, maybeParams->functionParameters,
                       mojoOperands);
  } else if (results.size() == 1) {
    state.setMojoValueFor(results.front(),
                          state.builder.var(symbolName(
                              maybeParams->functionParameters, mojoOperands)));
  } else {
    return emitError("CallOp with multiple outputs not supported");
  }

  return success();
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

bool BindLambdaOp::hasAffineViewOperation() {
  // Only output lambdas can have affine operations.
  if (getBindKind() != BindLambdaKind::OutputLambda)
    return false;

  return walk([](mlir::Operation *op) {
           // MOGG::CallOp is only used inside a lambda to represent an affine
           // function call
           if (isa<MOGG::CallOp>(op))
             return WalkResult::interrupt();
           return WalkResult::advance();
         })
      .wasInterrupted();
}

LogicalResult BindLambdaOp::verify() {
  if (getFusibleTensors().empty())
    return success();

  if (getLambda().getNumArguments() != getFusibleTensors().size() + 1)
    return emitError("Expecting #operands + 1 arguments for lambda");

  auto verifyRank = [this](Value val, Type indicesTp) -> LogicalResult {
    auto shapedType = cast<MO::MOShapedTypeInterface>(val.getType());
    size_t rank = shapedType.getRank();

    auto idxTp = dyn_cast<KGEN::POP::ArrayType>(indicesTp);
    if (idxTp == nullptr ||
        idxTp.getResolvedSize().value() != static_cast<int64_t>(rank))
      return emitError(
          "Mismatch between tensor rank and and indices array size");

    return success();
  };

  auto verifyDType = [this](Value val, Value elemVal) -> LogicalResult {
    auto shapedType = cast<MO::MOShapedTypeInterface>(val.getType());
    DType valDType = shapedType.getDtype();

    if (valDType != MO::getDTypeOrInvalid(elemVal.getType()))
      return emitError(
          "Mismatch between tensor dtype and dtype used by lambda");

    return success();
  };

  auto outOp = cast<MOGG::OutputOp>(getBody()->getTerminator());
  ValueRange outVals = outOp->getOperands();

  Value castedVal = outVals.front();
  if (getBindKind() == BindLambdaKind::OutputLambda) {
    if (outVals.size() != 2)
      return emitError("Expecting output lambda to yield 2 values");

    // Bind the output value to the output tensor.
    Type indicesTp = outVals.back().getType();
    // lambdaArg <-> virtual tensor (the returned one).
    // castedVal <-> physical tensor (the bound one).
    if (failed(verifyRank(getFusibleTensors().front(), indicesTp)) ||
        failed(verifyDType(getResult(), getValueArgs().front())) ||
        failed(verifyDType(getFusibleTensors().front(), castedVal)))
      return failure();

  } else {
    // Bind the input value from input tensors.
    // Symmetrical to output lambda:
    // castedVal <-> virtual tensor (the returned one).
    // lambdaArg <-> physical tensor (the bound one).
    if (failed(verifyDType(getResult(), castedVal)))
      return failure();

    Type indicesTp = getIndicesArg().getType();
    for (auto [t, v] : llvm::zip_equal(getFusibleTensors(), getValueArgs())) {
      if (failed(verifyRank(t, indicesTp)) || failed(verifyDType(t, v)))
        return failure();
    }
  }

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
      parser.parseRegion(lambdaRegion, argInfo, false)) {
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

  FunctionType ftp;
  if (parser.parseColonType(ftp))
    return failure();

  operandTps.assign(ftp.getInputs().begin(), ftp.getInputs().end());
  result.types.assign(ftp.getResults().begin(), ftp.getResults().end());

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
  p.printFunctionalType(getOperands().getTypes(), (*this)->getResultTypes());
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
  MO::TensorType outputTpAsTensor = nullptr;
  if (auto bufferTy = dyn_cast<MO::BufferType>(outputTp))
    outputTpAsTensor = bufferTy.toTensorType();
  else
    outputTpAsTensor = cast<MO::TensorType>(outputTp);

  Type indicesTp =
      KGEN::POP::ArrayType::get(outputTpAsTensor.getRank(), b.getIndexType());
  block.addArgument(indicesTp, state.location);

  // Caller need to populate the lambda body.
  if (bindKind == BindLambdaKind::ElemwiseLambda)
    return;

  b.setInsertionPointToStart(&block);
  MOGG::OutputOp::create(b, state.location, block.getArguments());
}

LogicalResult BindLambdaOp::emitMojo(EmitMojoState &state) {

  bool isOutputLambda = getBindKind() == BindLambdaKind::OutputLambda;
  bool generateStoreForOutput = isOutputLambda;
  if (isOutputLambda) {
    // We need to check if we have a Compute Lambda (in that case we return the
    // output instead of doing a store).
    if (!getResult().hasOneUse())
      return emitError("BindLambdaOp should have a unique use");
    auto fusedOp =
        dyn_cast<MOGG::TensorFusedOp>(*getResult().getUsers().begin());
    if (!fusedOp)
      return emitError("BindLamdbdaOp user should be a FusedTensorOp");
    auto outputLambdaTypeOr = getKernelOutputFusionLambdaType(
        state.analysis, fusedOp, state.enableComputeLambdas);
    if (outputLambdaTypeOr.isError())
      return emitError(outputLambdaTypeOr.getError());

    generateStoreForOutput =
        *outputLambdaTypeOr != KernelOutputFusionLambdaType::Compute;
  }

  // Build the corresponding lambda.
  MO::TensorType outputTy = MO::castFromTensorLike(getOutput().getType());

  auto widthParamName = state.builder.getIdentifier("width");

  Block *lambdaBody = getBody();

  // Check if we're on a GPU target. For GPU kernels, we need to use the
  // GPU-safe load function that takes raw pointers instead of
  // ManagedTensorSlice. This is because ManagedTensorSlice captures in GPU
  // kernel closures don't work correctly - the pointer values become garbage on
  // the device.
  auto kernelStubOp = this->getOperation()->getParentOfType<KernelStubOp>();
  bool isGpuKernel =
      kernelStubOp &&
      MO::getDeviceOrDefaultAny(kernelStubOp).getLabel() == "gpu";

  // Prepare the inputs
  SmallVector<MojoValue> fusedInputs;
  // For GPU kernels, we extract ptr and strides before the lambda to avoid
  // capturing ManagedTensorSlice objects which don't work correctly on GPU.
  SmallVector<MojoValue> inputPtrs;
  SmallVector<MojoValue> inputShapes;
  SmallVector<MojoValue> inputStrides;
  SmallVector<MojoValue> inputStaticSpecs;
  // Store tensor types for GPU kernel parameter passing
  SmallVector<MO::TensorType> inputTensorTypes;

  auto fusibleTensors = getFusibleTensors();
  for (auto [idx, input] :
       llvm::enumerate(state.getMojoValuesFor(fusibleTensors))) {
    fusedInputs.push_back(input);

    // The Compiler doesn't know the body lambda is using the tensor, and can't
    // keep it alive.
    state.markAsKeepAlive(input);

    if (isGpuKernel) {
      // Extract raw pointer and strides before the lambda.
      // These are trivial types that can be safely captured by GPU kernels.
      MojoValue ptr = state.builder.var(input.get("unsafe_ptr")());
      MojoValue shape = state.builder.var(input.get("shape")());
      MojoValue strides = state.builder.var(input.get("strides")());
      inputPtrs.push_back(ptr);
      inputShapes.push_back(shape);
      inputStrides.push_back(strides);
      inputStaticSpecs.push_back(input.get("static_spec"));
      // Store tensor type for dtype/rank parameters
      inputTensorTypes.push_back(
          MO::castFromTensorLike(fusibleTensors[idx].getType()));
    }
  }

  std::optional<MojoValue> returnType = std::nullopt;
  if (!generateStoreForOutput) {
    returnType = EmitMojoState::getSIMDType(
        outputTy.getDtype(), state.builder.newVar(widthParamName));
  }

  bool errOccurred = false;
  auto lambdaName = [&]() -> StringLiteral {
    if (getBindKind() == BindLambdaKind::ElemwiseLambda)
      return "body_lambda";
    else if (getBindKind() == BindLambdaKind::InputLambda)
      return "input_lambda";
    else
      return "output_lambda";
  }();

  SmallVector<NamedMojoDef> lambdaArguments = {NamedMojoDef(
      "indices",
      MojoValue(HardcodedMojoSymbols::struct_IndexList)[static_cast<int64_t>(
          outputTy.getRankNonZero())])};
  if (isOutputLambda) {
    // Output lambda has an extra argument which is the kernel output.
    lambdaArguments.push_back(NamedMojoDef(
        "input",
        EmitMojoState::getSIMDType(outputTy.getDtype(),
                                   state.builder.newVar(widthParamName))));
  }

  // Build the list of variables to copy-capture for GPU kernels.
  // This is needed because @__copy_capture must explicitly list all variables
  // that will be used inside GPU kernels through nested closures.
  SmallVector<MojoValue> copyCaptures;
  if (isGpuKernel) {
    copyCaptures.append(inputPtrs.begin(), inputPtrs.end());
    copyCaptures.append(inputShapes.begin(), inputShapes.end());
    copyCaptures.append(inputStrides.begin(), inputStrides.end());
    for (auto &&loadOp : lambdaBody->getOps<MOGG::LoadTensorOp>())
      copyCaptures.push_back(state.getMojoValueFor(loadOp.getTensor()));
  }

  // The body lambda has with / alignment parameters and indices arguments.
  // For GPU kernels, use lambdaWithCopyCapture to explicitly copy-capture
  // the ptr/strides variables so they're available inside the GPU kernel.
  auto lambdaBodyFunc = [&](const MOGG::FunctionBody &functionBody) {
    state.currentFunctionBody = functionBody;
    const MojoValue &widthParam = functionBody.parameters[0];
    const MojoValue &alignmentParam = functionBody.parameters[1];
    const MojoValue &indicesInput = functionBody.arguments[0];
    state.addDebugPrintLambdaParam(getBindKind(), widthParam, alignmentParam,
                                   indicesInput);

    // Do all load for each inputs.
    unsigned inputIdx = 0;
    for (unsigned i = 0; i < fusedInputs.size(); ++i) {
      Value blockArg = lambdaBody->getArgument(i);
      if (i == 0 && isOutputLambda) {
        // For the output lambda, just map the argument to the last operand.
        state.setMojoValueFor(blockArg, functionBody.arguments.back());
      } else {
        // Do a load from the input tensor
        const auto &mojoInput = fusedInputs[i];
        state.addDebugPrintTensorSpecs(mojoInput, indicesInput,
                                       /*isInput=*/true, inputIdx++);
        MOGG::MojoValue loadedSimd;
        if (isGpuKernel) {
          // Use GPU-safe load that takes raw pointer and strides.
          // This avoids capturing ManagedTensorSlice which doesn't work
          // correctly in GPU kernel closures.
          MO::TensorType tensorTy = inputTensorTypes[i];
          loadedSimd = MojoValue(
              HardcodedMojoSymbols::func_simd_load_from_tensor_pointer)(
              {MojoValue("dtype").equal(
                   state.mlirDTypeToMojoType(tensorTy.getDtype())),
               MojoValue("rank").equal(
                   MojoValue(static_cast<int64_t>(tensorTy.getRankNonZero()))),
               MojoValue("static_spec").equal(inputStaticSpecs[i]),
               MojoValue("simd_width").equal(widthParam),
               MojoValue("element_alignment").equal(alignmentParam)},
              {inputPtrs[i], inputShapes[i], inputStrides[i], indicesInput});
        } else {
          loadedSimd = MojoValue(
              HardcodedMojoSymbols::func_simd_load_from_managed_tensor_slice)(
              {MojoValue("simd_width").equal(widthParam),
               MojoValue("element_alignment").equal(alignmentParam)},
              {mojoInput, indicesInput});
        }
        // Map the argument
        state.setMojoValueFor(blockArg, state.builder.var(loadedSimd));
      }
    }

    // The last argument is the indices
    state.setMojoValueFor(lambdaBody->getArguments().back(), indicesInput);

    // Lower block
    errOccurred |=
        llvm::any_of(lambdaBody->getOperations(), [&](mlir::Operation &op) {
          return state.lowerOpToMojo(&op).failed();
        });

    if (errOccurred)
      return;

    auto outputOp = cast<MOGG::OutputOp>(lambdaBody->getTerminator());
    const MojoValue &resultValue =
        state.getMojoValueFor(outputOp.getOperand(0));
    // The lambda might also compute and return the indices.
    const MojoValue &resultIndices =
        outputOp->getNumOperands() > 1
            ? state.getMojoValueFor(outputOp.getOperand(1))
            : indicesInput;

    if (generateStoreForOutput) {
      // Store the output value.
      state.addDebugPrintTensorSpecs(fusedInputs.front(), resultIndices,
                                     /*isInput=*/false, /*operandIndex=*/0);
      if (isGpuKernel) {
        state.builder.call(
            HardcodedMojoSymbols::func_simd_store_into_tensor_pointer,
            {MojoValue("static_spec").equal(inputStaticSpecs[0]),
             MojoValue("element_alignment").equal(alignmentParam)},
            {inputPtrs[0], inputShapes[0], inputStrides[0], resultIndices,
             resultValue});
      } else {
        state.builder.call(
            HardcodedMojoSymbols::func_simd_store_into_managed_tensor_slice,
            ArrayRef<MojoValue>{
                MojoValue("element_alignment").equal(alignmentParam)},
            {fusedInputs.front(), resultIndices, resultValue});
      }
    } else {
      // Add the return statement.
      state.builder.returnsValue(resultValue);
    }
  };

  SmallVector<NamedMojoDef> lambdaParams = {
      NamedMojoDef(widthParamName, HardcodedMojoSymbols::struct_Int),
      NamedMojoDef("alignment", HardcodedMojoSymbols::struct_Int)};

  MojoValue lambdaParam;
  if (isGpuKernel) {
    // Use lambdaWithCopyCapture to explicitly copy-capture ptr/strides
    // variables for GPU kernels.
    lambdaParam = state.builder.lambdaWithCopyCapture(
        lambdaName, copyCaptures, lambdaParams, lambdaArguments, returnType,
        /*isInline=*/true, /*isParameter=*/true, std::move(lambdaBodyFunc));
  } else {
    lambdaParam =
        state.builder.lambda(lambdaName, lambdaParams, lambdaArguments,
                             returnType, /*isInline=*/true,
                             /*isParameter=*/true, std::move(lambdaBodyFunc));
  }

  if (errOccurred)
    return failure();

  // Bind the output to the lambda.
  state.setMojoValueFor(getOutput(), lambdaParam);
  return success();
}

//===----------------------------------------------------------------------===//
// MOGG KernelTerminatorOp
//===----------------------------------------------------------------------===//

LogicalResult CallForeachOp::verify() {
  if (getContext().size() > 1)
    return emitError("Can use at most one context or tuple of contexts");

  return success();
}

Value CallForeachOp::getDeviceContext() {
  if (getContext().empty())
    return nullptr;
  return getContext().front();
}
SmallVector<Value> CallForeachOp::getKernelInputs() {
  // foreach does not have input, tensors are captured and loads are fused into
  // a static lambda.
  return {};
}
SmallVector<Value> CallForeachOp::getKernelDPSOutputs() {
  return {getOutput()};
}
Type CallForeachOp::getKernelReturnOutput() { return nullptr; }

bool CallForeachOp::isThrows() { return true; }

LogicalResult CallForeachOp::emitMojo(EmitMojoState &state) {
  auto function = state.analysis.getLITForeachKernel();
  const MojoValue &bodyLambdaParam = state.getMojoValueFor(getInput());

  // Get the kernel operands.
  auto contextInputs = getContext();
  if (contextInputs.size() != 1)
    return emitError(
        "Op must have exactly one context operand to be lowered to mojo");
  const MojoValue &contextValue = state.getMojoValueFor(contextInputs.front());
  MojoValue mojoOutput = state.getMojoValueFor(getOutput());

  // Infer the kernel parameters.

  // Figure out the simd_width of the foreach kernel (depending on the tensor
  // dtypes). First collect all dtypes used in the kernel.
  llvm::SmallSetVector<DType, 4> allTensorDtypes;
  auto lambdaOp = cast<MOGG::BindLambdaOp>(getInput().getDefiningOp());
  lambdaOp.getBody()->walk([&](Operation *op) {
    for (Type t :
         llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
      ErrorOr<DType> dTypeOr = M::MO::fromMLIRType(t);
      if (!dTypeOr.isError())
        allTensorDtypes.insert(*dTypeOr);
    }
  });

  // Now call mojo primitives to figure out the width from the dtypes.
  llvm::SmallVector<MojoValue> allDtypeMojoParams;
  for (DType dtype : allTensorDtypes)
    allDtypeMojoParams.push_back(state.mlirDTypeToMojoType(dtype));
  MojoValue paramsDtypesTuple = MojoValue(
      HardcodedMojoSymbols::struct_StaticTuple)[{
      HardcodedMojoSymbols::struct_DType,
      MojoValue(int64_t(allDtypeMojoParams.size()))}]({}, allDtypeMojoParams);

  // Get the target device from the kernel stub that contains this operation.
  // This ensures we use the correct device (gpu/cpu) for simd_width
  // calculation.
  auto kernelStubOp = this->getOperation()->getParentOfType<KernelStubOp>();
  MojoValue targetDeviceLabel =
      MojoValue(kernelStubOp
                    ? MO::getDeviceOrDefaultAny(kernelStubOp).getLabel()
                    : MO::getDeviceOrDefaultAny(*this).getLabel())
          .quote();

  static const StringLiteral foreachBodyLambdaParamName = "func";
  static const StringLiteral foreachSimdWidthParamName = "simd_width";
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    if (paramName == foreachBodyLambdaParamName)
      return bodyLambdaParam;
    if (paramName == foreachSimdWidthParamName)
      return MojoValue(HardcodedMojoSymbols::func_get_simd_width_for_dtypes)(
          {paramsDtypesTuple, targetDeviceLabel});
    auto inferredParam = state.inferDefaultKernelParamValue(*this, paramName);
    return inferredParam;
  };
  auto maybeParams = EmitMojoState::inferKernelParameters(function, inferParam);
  if (maybeParams.isError())
    return emitError("Failed to infer all the kernel parameters");

  state.builder.verbatim(callFunctionNestedInStruct(
      function, *maybeParams, {mojoOutput, contextValue}, state));
  state.genAllKeepAliveStatements();

  return success();
}

//===----------------------------------------------------------------------===//
// CallMaterializeOp
//===----------------------------------------------------------------------===//

LogicalResult CallMaterializeOp::verify() {
  if (getContext().size() > 1)
    return emitError("Can use at most one context or tuple of contexts");

  return success();
}

Value CallMaterializeOp::getDeviceContext() {
  if (getContext().empty())
    return nullptr;
  return getContext().front();
}
SmallVector<Value> CallMaterializeOp::getKernelInputs() { return {getInput()}; }

SmallVector<Value> CallMaterializeOp::getKernelDPSOutputs() {
  return {getOutput()};
}

Type CallMaterializeOp::getKernelReturnOutput() { return nullptr; }

bool CallMaterializeOp::isThrows() { return true; }

LogicalResult CallMaterializeOp::emitMojo(EmitMojoState &state) {
  auto function = state.analysis.getLITViewMaterializeKernel();

  // Get the kernel operands.
  SmallVector<MojoValue> mojoOperands = {state.getMojoValuesFor(getOperands())};
  std::swap(mojoOperands[0], mojoOperands[1]);

  // Infer the kernel parameters.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    return state.inferDefaultKernelParamValue(*this, paramName);
  };
  auto maybeParams = EmitMojoState::inferKernelParameters(function, inferParam);
  if (maybeParams.isError())
    return emitError("Failed to infer all the kernel parameters");

  auto mojoSymbolName = state.getOriginalMojoNameFor(function);

  state.builder.call(mojoSymbolName, maybeParams->functionParameters,
                     mojoOperands);

  state.genAllKeepAliveStatements();

  return success();
}

LogicalResult CallExecuteOp::verify() {
  uint64_t numberOfContexts = llvm::count_if(getOperandTypes(), [](auto ty) {
    if (isa<MOGG::ContextType>(ty))
      return true;

    if (auto tupleTy = dyn_cast<mlir::TupleType>(ty)) {
      if (isa<MOGG::ContextType>(tupleTy.getTypes().front()))
        return true;
    }

    return false;
  });

  if (numberOfContexts > 1)
    return emitError("Can use at most one context or tuple of contexts");

  return success();
}

Value CallExecuteOp::getDeviceContext() {
  if (getInputs().empty())
    return nullptr;

  auto lastOperand = getInputs().back();
  if (isa<MOGG::ContextType>(lastOperand.getType()))
    return lastOperand;

  if (!isa<TupleType>(lastOperand.getType()))
    return nullptr;

  if (isa<MOGG::ContextType>(
          lastOperand.getDefiningOp<AsTupleOp>().getOperand(0).getType())) {
    return lastOperand;
  }

  return nullptr;
}
SmallVector<Value> CallExecuteOp::getKernelInputs() {
  return getInputs().drop_back(getDeviceContext() != nullptr);
}
SmallVector<Value> CallExecuteOp::getKernelDPSOutputs() { return getOutputs(); }
Type CallExecuteOp::getKernelReturnOutput() {
  return getRegRetTypeAttr() ? getRegRetTypeAttr().getValue() : nullptr;
}

bool CallExecuteOp::isThrows() { return getIsThrows(); }

namespace {
/// Check if input value is a mutable tensor
bool isMutableInputTensor(Value input) {
  auto tensorType = dyn_cast<MO::TensorType>(input.getType());
  if (!tensorType)
    return false;
  auto ioKindAttr = tensorType.getIOKindAttr();
  auto isMutableAttr = tensorType.getIsMutableAttr();
  return ioKindAttr && isMutableAttr && isMutableAttr.getValue();
}

/// Generate mogg_tensor_init to transform MutableInputTensor to InputTensor
MojoValue generateMutableTensorInit(Value input, EmitMojoState &state) {
  auto tensorType = cast<MO::TensorType>(input.getType());
  MojoValue inputValue = state.getMojoValueFor(input);

  // Extract dtype
  ErrorOr<DType> dTypeOr = M::MO::fromMLIRType(tensorType.getElementType());
  ASSERT_STREAM(!dTypeOr.isError(),
                "Failed to extract dtype from mutable input tensor");
  MojoValue dtype = state.mlirDTypeToMojoType(*dTypeOr);

  // Extract rank
  ASSERT_STREAM(tensorType.hasRank(),
                "Mutable input tensor must have statically known rank");
  MojoValue rank(static_cast<int64_t>(tensorType.getRankNonZero()));

  // Get static_spec fields
  MojoValue staticSpec = inputValue.get("static_spec");

  // Build template parameters
  SmallVector<MojoValue> templateParams = {dtype,
                                           rank,
                                           MojoValue(false), // False constant
                                           MojoValue("io_spec.IO.Input"),
                                           staticSpec.get("shape"),
                                           staticSpec.get("strides"),
                                           staticSpec.get("alignment"),
                                           staticSpec.get("exclusive")};

  // Build runtime arguments
  SmallVector<MojoValue> runtimeArgs = {
      inputValue.get("_ptr").get("bitcast")[MojoValue("NoneType")](),
      inputValue.get("shape")()};

  // Generate the mogg_tensor_init call
  MojoValue moggTensorInit("MOGGPrimitives.mogg_tensor_init");
  return state.builder.var(moggTensorInit(templateParams, runtimeArgs));
}
} // namespace

LogicalResult CallExecuteOp::emitMojo(EmitMojoState &state) {
  auto function =
      state.analysis.getRegisteredLITMojoExecuteFunctionFor(getKernelName());

  // Build kernel adaptor to check what types the kernel expects
  using namespace KGEN::MOGGPreElab;
  MojoKernelFunctionAdaptor<KGEN::LIT::FnOp> kernelAdaptor(function);

  // Collect all kernel I/Os
  SmallVector<MojoValue> kernelOperands;
  for (Value val : getKernelDPSOutputs())
    kernelOperands.push_back(state.getMojoValueFor(val));

  auto inputs = getKernelInputs();
  for (unsigned i = 0; i < inputs.size(); ++i) {
    Value input = inputs[i];
    MojoValue inputValue = state.getMojoValueFor(input);

    // Transform mutable inputs if kernel expects InputTensor
    if (isMutableInputTensor(input) &&
        i < kernelAdaptor.inputArguments.size()) {
      auto *tensorInfo = std::get_if<TensorOperandAdaptor>(
          &kernelAdaptor.inputArguments[i].underlyingType);
      ASSERT_STREAM(tensorInfo, "Expected tensor operand for kernel input #"
                                    << i
                                    << " when "
                                       "matching "
                                       "mutable "
                                       "MO "
                                       "operand");
      if (tensorInfo->ioSpec == IOSpec::InputTensor)
        inputValue = generateMutableTensorInit(input, state);
    }

    kernelOperands.push_back(inputValue);
  }

  // And the optional context.
  if (auto maybeContext = getDeviceContext())
    kernelOperands.push_back(state.getMojoValueFor(maybeContext));

  // Infer the kernel parameters.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    return state.inferDefaultKernelParamValue(*this, paramName);
  };
  auto maybeParams = EmitMojoState::inferKernelParameters(function, inferParam);
  if (maybeParams.isError())
    return emitError("Failed to infer all the kernel parameters");

  auto prefixSymbolName = state.getOriginalMojoNamePrefixFor(function);
  MojoValue executeFunc("execute");
  MojoValue structType = prefixSymbolName[maybeParams->structParameters];
  MojoValue fullExecuteFunc = structType.get(executeFunc);

  // If the kernel has a return type, capture and return the value.
  if (Type returnType = getKernelReturnOutput()) {
    MojoValue result = state.builder.var(
        fullExecuteFunc(maybeParams->functionParameters, kernelOperands));
    state.genAllKeepAliveStatements();
    // Use transfer for opaque types to handle move-only types correctly.
    if (isa<MO::OpaqueType>(returnType))
      result = result.transfer();
    state.builder.returnsValue(result);
  } else {
    state.builder.call(fullExecuteFunc, maybeParams->functionParameters,
                       kernelOperands);
    state.genAllKeepAliveStatements();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// KernelStubOp
//===----------------------------------------------------------------------===//

LogicalResult KernelStubOp::verify() {
  FunctionType funcType = getFunctionType();
  if (getAbi() == ABIKind::MOGG && funcType.getNumResults() > 1) {
    return emitError(
        "Only at most one result is currently supported for MOGG ABI");
  }

  // No need to verify arguments because the FunctionOpInterface already does
  // it.

  return success();
}

bool KernelStubOp::isThrows() {
  return llvm::any_of(getBodyBlock().getOps<MOGG::EmitsMojoInterface>(),
                      [](auto emitMojoOp) { return emitMojoOp.isThrows(); });
}

LogicalResult KernelStubOp::emitMojo(EmitMojoState &state) {
  // If the ABI is MOGG, this was originally a MOGG kernel. Otherwise, it was a
  // GenericCompiledRegionOp which has that implicit void** ABI.
  if (getAbi() == ABIKind::MOGG) {
    // Helper to convert type to Mojo, using proper opaque type representation
    // for parameterized opaque types.
    auto typeToMojo = [&](Type type) -> MojoValue {
      if (auto opaqueType = dyn_cast<MO::OpaqueType>(type)) {
        MojoValue baseTy = state.mlirOpaqueTypeToMojoType(opaqueType);
        auto params = EmitMojoState::getOpaqueTypeParameters(opaqueType);
        if (!params.empty())
          return baseTy[params];
        return baseTy;
      } else if (auto mgpBufferType = dyn_cast<MGP::DeviceBufferType>(type)) {
        // mgp.buffer should always be treated as NDBuffer of uint8_t.
        return HardcodedMojoSymbols::comptime_NDBuffer_In_MGP;
      }
      return state.mlirTypeToMojoType(type);
    };

    SmallVector<MOGG::NamedMojoDef> inputTypes;
    for (auto arg : getBody().getArguments())
      inputTypes.push_back(
          MOGG::NamedMojoDef("arg", typeToMojo(arg.getType())));

    // Determine the Mojo return type from the stub's return type
    std::optional<MOGG::MojoValue> mojoReturnType = std::nullopt;
    FunctionType functionType = getFunctionType();
    if (functionType.getNumResults())
      mojoReturnType = typeToMojo(functionType.getResult(0));

    bool errOccurred = false;
    state.builder.fn(
        getSymNameAttr().str(), {}, inputTypes,
        /*returnType=*/mojoReturnType, /*exported=*/false,
        /*raises=*/isThrows(), [&](const MOGG::FunctionBody &functionBody) {
          state.currentFunctionBody = functionBody;
          for (auto &&[mlirArg, mojoArg] :
               llvm::zip(getBody().getArguments(), functionBody.arguments)) {
            state.setMojoValueFor(mlirArg, mojoArg);
          }

          errOccurred = llvm::any_of(getOps(), [&](mlir::Operation &op) {
            return state.lowerOpToMojo(&op).failed();
          });

          if (!errOccurred)
            state.genAllKeepAliveStatements();
        });

    if (errOccurred)
      return failure();
  } else {
    // Clear any previous kernel's return values from the state
    state.mojoReturnValues.clear();
    state.mojoConstantResources.clear();

    bool errOccurred = false;
    static const MOGG::MojoValue vMutAnyOrigin =
        MOGG::MojoValue(MOGG::HardcodedMojoSymbols::comptime_MutAnyOrigin);
    static const MOGG::MojoValue vUnsafePointer =
        MOGG::HardcodedMojoSymbols::struct_UnsafePointer;
    static const MOGG::MojoValue vOpaquePointer =
        MOGG::HardcodedMojoSymbols::comptime_OpaquePointer;
    state.builder.fn(
        getSymNameAttr().strref(), {},
        {MOGG::NamedMojoDef("arg",
                            vUnsafePointer[{vOpaquePointer, vMutAnyOrigin}]),
         MOGG::NamedMojoDef("arg", vOpaquePointer)},
        std::nullopt, true, false, [&](const MOGG::FunctionBody &functionBody) {
          state.currentFunctionBody = functionBody;
          state.builder.comment(
              "This is the entry point of the kernel. The first argument is a "
              "pointer to a list of inputs and outputs. The second is a "
              "pointer "
              "to a runtime context.");
          auto runtimeArgument = functionBody.arguments.front();
          auto &kernelBody = getBody().front();
          // The order of arguments in runtimeArgument:
          // 1. Inputs and DPS outputs
          // 2. Return values (chain and non DPS outputs)
          for (auto argument : kernelBody.getArguments()) {
            state.setMojoValueFor(
                argument,
                runtimeArgument[std::to_string(argument.getArgNumber())]);
          }
          // Return values, although in DPS form from the runtime's perspective,
          // are still in value semantics inside of the kernel. Therefore, there
          // is no value to associate them with. Instead, we have a mapping that
          // mogg.return and mogg.async.pack will use.
          uint64_t numArguments = getBodyBlock().getNumArguments();
          for (uint64_t i = 0; i < getNumResults(); ++i) {
            state.mojoReturnValues.emplace_back(
                runtimeArgument[std::to_string(i + numArguments)]);
          }

          // After inputs and outputs, there are constant resources appended to
          // the void** array. These come in pairs of ptr + size which is
          // handled here.
          for (uint64_t i = 0; i < getConstantResources().size(); ++i) {
            uint64_t flatIndex = i * 2 + numArguments + getNumResults();
            MOGG::MojoValue ptr = runtimeArgument[flatIndex];
            // The ABI is that the bytecount is directly stored as an
            // OpaquePointer.
            MOGG::MojoValue byteCount =
                runtimeArgument.get("bitcast[Int]()")[flatIndex + 1];
            state.mojoConstantResources.emplace_back(
                std::tuple<MOGG::MojoValue, MOGG::MojoValue>{ptr, byteCount});
          }

          errOccurred = llvm::any_of(getOps(), [&](mlir::Operation &op) {
            return state.lowerOpToMojo(&op).failed();
          });

          if (!errOccurred)
            state.genAllKeepAliveStatements();
        });

    if (errOccurred)
      return failure();
  }

  return success();
}

// These FuncOp attributes are disallowed while parsing since they can
// be inferred. Likewise while printing we ignore them.
static StringRef kernelStubOpdisallowedAttrNames[] = {
    "name", "returnType", "sym_name", "function_type"};

ParseResult KernelStubOp::parse(OpAsmParser &parser, OperationState &state) {
  StringAttr name;
  if (failed(parser.parseSymbolName(name)))
    return failure();
  state.addAttribute(getSymNameAttrName(state.name), name);

  SmallVector<Type> resTypes;
  SmallVector<OpAsmParser::Argument> args;
  if (failed(parseRegionArguments(parser, resTypes, args)))
    return failure();

  SmallVector<Type> argTypes;
  for (auto &&arg : args)
    argTypes.push_back(arg.type);

  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(FunctionType::get(parser.getContext(),
                                                     argTypes, resTypes)));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  llvm::SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed : kernelStubOpdisallowedAttrNames) {
    if (parsedAttributes.get(disallowed))
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
  }
  state.attributes.append(parsedAttributes);

  if (failed(parser.parseRegion(*state.addRegion(), args)))
    return failure();

  return success();
}

void KernelStubOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printSymbolName(getSymNameAttr());
  p << " ";
  printRegionArguments(p, getFunctionType().getResults(),
                       getBody().getArguments());
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     kernelStubOpdisallowedAttrNames);
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// MOGG KernelOp
//===----------------------------------------------------------------------===//

LogicalResult KernelOp::lower(MGP::MOGraphLoweringState &loweringState) {
  MGP::MOGGKernelOpLoweringState kernelLoweringState(getOperation(),
                                                     loweringState);
  return kernelLoweringState.lowerKernelOp(*this);
}

void KernelOp::insertInputsAt(unsigned idx, ValueRange vals) {
  SmallVector<Value> ops = getInputs();
  ops.insert(ops.begin() + idx, vals.begin(), vals.end());
  getInputsMutable().assign(ops);

  for (auto v : llvm::reverse(vals))
    getExecBody().insertArgument(idx, v.getType(), v.getLoc());
}

void KernelOp::eraseInputsAt(unsigned idx, size_t len) {
  getInputsMutable().erase(idx, len);

  for (unsigned i = 0; i < len; ++i)
    getExecBody().eraseArgument(idx);
}

void KernelOp::setChain(Value chain) {
  if (chain)
    getChainMutable().assign(chain);
}
void KernelOp::eraseChain() { getChainMutable().erase(0); }

std::optional<MO::ChainValue> KernelOp::maybeGetInChain() {
  auto chainOrNull = getChain();
  return chainOrNull ? std::optional(chainOrNull) : std::nullopt;
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

KernelTerminator KernelOp::getTerminator() {
  return cast<KernelTerminator>(getExecBody().front().getTerminator());
}

BindLambdaOp KernelOp::getBoundInputLambda(uint64_t operandIdx) {
  auto users = getExecBody().getArgument(operandIdx).getUsers();
  if (llvm::range_size(users) != 1)
    return nullptr;

  auto lambda = dyn_cast<BindLambdaOp>(*users.begin());
  if (!lambda)
    return nullptr;

  if (lambda.getBindKind() != BindLambdaKind::InputLambda)
    return nullptr;

  return lambda;
}

BindLambdaOp KernelOp::getBoundOutputLambda(uint64_t resultIdx) {
  // NOTE: dyn_cast<CallExecuteOp>(getTerminator()) won't work as expected
  // (it always returns non-nullptr) because the deduced return type would be
  // `const CallExecuteOp &`.
  auto executeOp = dyn_cast<CallExecuteOp>(getTerminator().getOperation());
  if (!executeOp)
    return nullptr;

  SmallVector<Value> flattenedOutputs;
  for (auto outputValue : executeOp.getKernelDPSOutputs()) {
    if (auto asTupleOp = outputValue.getDefiningOp<MOGG::AsTupleOp>()) {
      for (auto outputValueInTuple : asTupleOp.getInputs())
        flattenedOutputs.push_back(outputValueInTuple);
    } else {
      flattenedOutputs.push_back(outputValue);
    }
  }

  if (resultIdx >= flattenedOutputs.size())
    return nullptr;

  auto lambdaOp = flattenedOutputs[resultIdx].getDefiningOp<BindLambdaOp>();
  if (!lambdaOp)
    return nullptr;

  if (lambdaOp.getBindKind() != MOGG::BindLambdaKind::OutputLambda)
    return nullptr;

  return lambdaOp;
}

BindLambdaOp KernelOp::getElemwiseBodyLambda() {
  Operation *t = getTerminator();
  auto foreachOp = mlir::dyn_cast<CallForeachOp>(t);
  return foreachOp ? foreachOp.getInput().getDefiningOp<BindLambdaOp>()
                   : nullptr;
}

bool KernelOp::isOperandFusible(uint64_t operandIdx, bool lookThroughView) {
  auto users = getExecBody().getArgument(operandIdx).getUsers();
  SmallVector<Operation *> workList(users.begin(), users.end());
  while (!workList.empty()) {
    auto curUser = workList.back();
    workList.pop_back();

    if (isa<MOGG::BindLambdaOp>(curUser))
      return true;

    if (lookThroughView && isa<MOGG::CallViewOp>(curUser))
      workList.append(curUser->user_begin(), curUser->user_end());
  }

  return false;
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

LogicalResult KernelOp::verify() {
  auto term =
      dyn_cast<MOGG::KernelTerminator>(getExecBody().front().getTerminator());
  if (!term)
    return emitError("Expect to terminate with a KernelTerminator");

  ValueRange dpsResult = getResults();
  // Chain does not occur on kernel (but propagated by graph compiler).
  if (!dpsResult.empty() && isa<MO::ChainType>(dpsResult.back().getType()))
    dpsResult = dpsResult.drop_back();

  for (auto [inner, outer] :
       llvm::zip_first(getExecBody().getArgumentTypes(), getOperandTypes())) {
    if (outer != inner)
      return emitError("Except same operands/block argument type");
  }

  if (llvm::any_of(getExecBody().getArgumentTypes(),
                   [](auto argType) { return isa<MO::ChainType>(argType); })) {
    return emitError("Kernel arguments can't be chains");
  }

  if (auto regOutput = term.getKernelReturnOutput()) {
    // Shape functions are somewhat weird because their outputs are really KGEN
    // parameters (corresponds to symbolic dimensions) no SSA values are
    // actually used by dependent kernel ops.
    if (!isa<MOGG::ShapeFuncResultType>(regOutput) &&
        regOutput == dpsResult.back().getType()) {
      dpsResult = dpsResult.drop_back();
    }
  }

  SmallVector<Type> dpsOutputTypesWithoutTuples;
  for (Value result : term.getKernelDPSOutputs()) {
    if (isa<TupleType>(result.getType())) {
      for (auto element : result.getDefiningOp<MOGG::AsTupleOp>().getInputs())
        dpsOutputTypesWithoutTuples.push_back(element.getType());
    } else {
      dpsOutputTypesWithoutTuples.push_back(result.getType());
    }
  }

  if (getNumOperands() &&
      llvm::any_of(getOperands().drop_back(1), [](auto operand) {
        return isa<MO::ChainType>(operand.getType());
      })) {
    return emitError("Chain must be the last operand of a kernel op");
  }

  if (getNumResults() &&
      llvm::any_of(getResults().drop_back(1), [](auto result) {
        return isa<MO::ChainType>(result.getType());
      })) {
    return emitError("Chain must be the last result of a kernel op");
  }

  return success();
}

ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::Argument> argInfo;
  OpAsmParser::UnresolvedOperand chain;

  auto parseOptionalChain = [&]() -> ParseResult {
    if (succeeded(parser.parseOptionalKeyword("chain"))) {
      if (parser.parseLParen() || parser.parseOperand(chain) ||
          parser.parseRParen() ||
          parser.resolveOperand(chain, MO::ChainType::get(parser.getContext()),
                                state.operands)) {
        return failure();
      }
    }
    return success();
  };

  if (parseParenOperandListWithShadowing(parser, state, argInfo, true) ||
      parseOptionalChain() || parser.parseArrowTypeList(state.types) ||
      parseRegion(parser, argInfo, *state.addRegion(), true) ||
      parser.parseOptionalAttrDict(state.attributes))
    return failure();

  state.addAttribute(
      "operandSegmentSizes",
      mlir::DenseI32ArrayAttr::get(
          parser.getContext(),
          {static_cast<int32_t>(argInfo.size()),
           static_cast<int32_t>(state.operands.size() - argInfo.size())}));

  return success();
}

void KernelOp::print(OpAsmPrinter &p) {
  printParenOperandListWithShadowing(p, getInputs(),
                                     getExecBody().getArguments(),
                                     /*isIsolatedFromAbove=*/true);
  if (auto chain = getChain()) {
    p << " chain";
    p << "(";
    p.printOperand(chain);
    p << ")";
  }

  p.printArrowTypeList(getResultTypes());
  p << ' ';
  printRegion(p, getInputs(), getExecBody(), /*isIsolatedFromAbove=*/true);
  p.printOptionalAttrDict(getOperation()->getAttrs(), {"operandSegmentSizes"});
}

void KernelOp::build(OpBuilder &builder, OperationState &state,
                     TypeRange results, ValueRange operands, StringAttr summary,
                     BoolAttr isAsync, DeviceRefAttr device,
                     ArrayRef<ParamDeclAttr> outputParamDecls,
                     ArrayAttr allocIndices, bool hasReadEffect,
                     bool hasWriteEffect) {
  state.addOperands(operands);
  if (summary)
    state.addAttribute(getSummaryAttrName(state.name), summary);

  if (isAsync)
    state.addAttribute(getIsAsyncAttrName(state.name), isAsync);

  if (!outputParamDecls.empty())
    state.addAttribute(getOutputParamDeclsAttrName(state.name),
                       builder.getAttr<ParamDeclArrayAttr>(outputParamDecls));

  if (allocIndices)
    state.addAttribute(getAllocIndicesAttrName(state.name), allocIndices);

  state.addAttribute(getHasReadEffectAttrName(state.name),
                     builder.getBoolAttr(hasReadEffect));
  state.addAttribute(getHasWriteEffectAttrName(state.name),
                     builder.getBoolAttr(hasWriteEffect));

  Region *execBodyRegion = state.addRegion();
  Block &body = execBodyRegion->emplaceBlock();
  for (auto [i, operand] : llvm::enumerate(operands)) {
    if (!isa<MO::ChainType>(operand.getType()))
      body.addArgument(operand.getType(), operand.getLoc());
  }

  state.addTypes(results);

  if (device)
    state.addAttribute(MO::kOperationDeviceRefAttrName, device);

  state.addAttribute(
      "operandSegmentSizes",
      mlir::DenseI32ArrayAttr::get(
          state.getContext(),
          {static_cast<int32_t>(body.getNumArguments()),
           static_cast<int32_t>(operands.size() - body.getNumArguments())}));
}

OutputPlaceholdersOp KernelOp::getOutputPlaceholder() {
  auto placeholder = this->getOps<MOGG::OutputPlaceholdersOp>();
  if (placeholder.empty())
    return nullptr;

  assert(llvm::hasSingleElement(placeholder));

  return *placeholder.begin();
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

ScratchPlaceholdersOp KernelOp::getScratchPlaceholder() {
  auto scratch = this->getOps<MOGG::ScratchPlaceholdersOp>();

  if (scratch.empty())
    return nullptr;

  assert(llvm::hasSingleElement(scratch));

  return *scratch.begin();
}

OutputOp KernelOp::getOutputOp() {
  auto output = this->getOps<MOGG::OutputOp>();
  if (output.empty())
    return nullptr;

  assert(llvm::hasSingleElement(output));

  return *output.begin();
}

//===----------------------------------------------------------------------===//
// AsScalarOp
//===----------------------------------------------------------------------===//

LogicalResult AsScalarOp::verify() {
  MO::TensorType type;
  if (auto tensorType = dyn_cast<MO::TensorType>(getInput().getType()))
    type = tensorType;
  else if (auto bufferType = dyn_cast<MO::BufferType>(getInput().getType()))
    type = bufferType.toTensorType();
  else
    emitError("Expected tensor or buffer input");
  if (type.getElementType() != getResult().getType().getDType())
    return emitError("Expected input and result to have the same data type");

  return success();
}

LogicalResult AsScalarOp::emitMojo(EmitMojoState &state) {
  const MojoValue &input = state.getMojoValueFor(getInput());

  auto moggAsScalar = state.getOriginalMojoNameFor(
      state.analysis.getRegisteredLITMojoKernelFor(this->getOperationName()));
  MOGG::MojoValue scalarValue = moggAsScalar(ArrayRef<MojoValue>{}, {input});
  state.setMojoValueFor(getResult(), state.builder.var(scalarValue));
  return success();
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

  if (!getExecutionDevice())
    return emitError("Compiled regions must have a device");

  // Ensure that output ops are fed by only a certain set of operations.
  auto outputOp = body.front().getTerminator();
  for (auto &&operand : outputOp->getOperands()) {
    if (!isa<KernelTerminatorProvider>(operand.getDefiningOp())) {
      return operand.getDefiningOp()->emitError(
          "This operation is feeding an output op but it is not marked as a "
          "KernelTerminatorProvider");
    }
  }

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
    if (!chainCreateOp || !chainCreateOp.getResult().hasOneUse() ||
        !bodyArgument.getUses().empty()) {
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
// ToMGPFromOpaqueOp
//===----------------------------------------------------------------------===//

LogicalResult ToMGPFromOpaqueOp::emitMojo(EmitMojoState &state) {
  // This operation is a no-op type conversion from mo.opaque to mgp.opaque.
  // In emit-mojo path, both types represent the same underlying value, so we
  // just pass through the Mojo value from the input to the output.
  state.setMojoValueFor(getDataPointer(), state.getMojoValueFor(getInput()));
  return success();
}

OpFoldResult ToMGPFromOpaqueOp::fold(FoldAdaptor adaptor) {
  // Fold away when input and output types are the same.
  // This can happen if the input's type was converted upstream.
  if (getInput().getType() == getDataPointer().getType())
    return getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// AlignmentPlaceholderOp
//===----------------------------------------------------------------------===//

void AlignmentPlaceholderOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  walkDef(getParamDecl(), {});
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
                         << KGEN::MOGGPreElab::kMutableInputTensor << ", "
                         << KGEN::MOGGPreElab::kFusedComputeOutputTensor
                         << " }";
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
// AsTupleOp
//===----------------------------------------------------------------------===//

LogicalResult AsTupleOp::verify() {
  if (getNumOperands() != getResult().getType().getTypes().size()) {
    return emitError()
           << "Number of operands does not match number of types in tuple";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AsyncUnpackOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncUnpackOp::canonicalize(AsyncUnpackOp unpackOp,
                                          mlir::PatternRewriter &rewriter) {
  if (auto packOp = unpackOp.getInput().getDefiningOp<AsyncPackOp>()) {
    rewriter.replaceOp(unpackOp, packOp.getInput());
    return success();
  }

  if (auto packOp = unpackOp.getInput().getDefiningOp<AsyncPackBorrowV2Op>()) {
    rewriter.replaceOp(unpackOp, packOp.getInput());
    return success();
  }

  return failure();
}

LogicalResult AsyncUnpackOp::emitMojo(EmitMojoState &state) {
  MojoValue unpackedValue;
  mlir::TypedValue<mlir::Type> input = getInput();
  mlir::Type inputType = input.getType();

  if (auto tensorSpec = dyn_cast<MGP::TensorSpecType>(inputType)) {

    MojoValue asyncUnpackFunc{HardcodedMojoSymbols::func_unpack_tensor_spec};
    unpackedValue = asyncUnpackFunc({int(tensorSpec.getRank())},
                                    {state.getMojoValueFor(input)});

  } else if (auto tensorType = dyn_cast<MGP::DeviceTensorType>(inputType)) {

    MojoValue asyncUnpackFunc{HardcodedMojoSymbols::func_unpack_tensor};
    MGP::TensorSpecType tensorSpec = tensorType.getTensorSpec();
    if (!tensorSpec.hasRank())
      return emitError() << "Expected Tensor operand to have known rank";

    auto tensorRank = tensorSpec.getRank();
    auto bufferRank = tensorRank == 0 ? 1 : tensorRank;
    std::optional<DType> maybeDType = tensorSpec.getOptDType();
    if (!maybeDType)
      return emitError() << "Expected Tensor operand to have DType";

    unpackedValue = asyncUnpackFunc(
        {bufferRank, tensorRank, state.mlirDTypeToMojoType(*maybeDType)},
        state.getMojoValueFor(input));

  } else if (isa<MGP::DeviceBufferType>(inputType)) {

    // TODO: (GEX-2732) Move away from NDBuffer in MOGGPrimitives.mojo
    MojoValue asyncUnpackFunc{HardcodedMojoSymbols::func_unpack_buffer_ref};
    unpackedValue = asyncUnpackFunc({}, {state.getMojoValueFor(input)});

  } else if (isa<MGP::DeviceContextType>(inputType)) {

    MojoValue asyncUnpackFunc{HardcodedMojoSymbols::func_unpack_device_ctx};
    unpackedValue = asyncUnpackFunc({}, {state.getMojoValueFor((input))});

  } else {

    auto moggAsyncUnpackFunc = state.getOriginalMojoNameFor(
        state.analysis.getRegisteredLITMojoKernelFor(this->getOperationName()));
    unpackedValue = moggAsyncUnpackFunc({state.mlirTypeToMojoType(getType())},
                                        state.getMojoValueFor(input));
  }

  MojoValue finalValue = state.builder.var(unpackedValue);

  // For opaque types, we need to bitcast and dereference from
  // LegacyOpaquePointer to the actual parameterized type.
  // Store the pointer in a variable, then dereference at use site.
  // This generates: var_XX = value.bitcast[Type]()
  //                 ... later: var_XX[]
  if (auto opaqueType = dyn_cast<MO::OpaqueType>(getType())) {
    MojoValue baseTy = state.mlirOpaqueTypeToMojoType(opaqueType);
    auto params = EmitMojoState::getOpaqueTypeParameters(opaqueType);
    MojoValue opaqueTypeStr = params.empty() ? baseTy : baseTy[params];
    MojoValue bitcastCall = finalValue.get("bitcast")({opaqueTypeStr}, {});
    // Store the pointer in a variable
    MojoValue ptrVar = state.builder.var(bitcastCall);
    // The value to use is the dereference expression: var_XX[]
    finalValue = state.builder.var(ptrVar.get("take_pointee")());
  }

  state.setMojoValueFor(getResult(), finalValue);
  return success();
}

//===----------------------------------------------------------------------===//
// AsyncPackOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncPackOp::emitMojo(EmitMojoState &state) {
  auto resultIndexOr = getOutputIndex();
  if (!resultIndexOr)
    return failure();
  MojoValue asyncPackHelperStruct =
      HardcodedMojoSymbols::struct_MoggAsyncPackHelper;
  MojoValue moggAsyncPackFunc = state.getOriginalMojoNameFor(
      state.analysis.getRegisteredLITMojoKernelFor(this->getOperationName()));
  SmallVector<MojoValue> operands;
  for (auto [idx, operand] : llvm::enumerate(getOperands())) {
    MojoValue mojoVal = state.getMojoValueFor(operand);
    // The first operand (input) is consumed, so use transfer for opaque types
    // to handle move-only types correctly.
    if (idx == 0 && isa<MO::OpaqueType, MGP::OpaqueType>(operand.getType()))
      mojoVal = mojoVal.transfer();
    operands.push_back(mojoVal);
  }
  operands.push_back(state.mojoReturnValues[*resultIndexOr]);
  state.builder.call(moggAsyncPackFunc, {},
                     asyncPackHelperStruct({}, operands));
  return success();
}

LogicalResult AsyncPackOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitError(
        "async.pack requires the same type for the result and operand 0");
  return success();
}

//===----------------------------------------------------------------------===//
// AsyncDelOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncDelOp::emitMojo(EmitMojoState &state) {
  auto moggAsyncDelFunc = state.getOriginalMojoNameFor(
      state.analysis.getRegisteredLITMojoKernelFor(this->getOperationName()));
  /// TODO: Add assertion that this operations resides in a stub with ABI ==
  /// MGP.
  state.builder.call(
      moggAsyncDelFunc, {},
      {state.currentFunctionBody.arguments.front(), getInputs().size()});
  return success();
}

//===----------------------------------------------------------------------===//
// TensorInitOp
//===----------------------------------------------------------------------===//

LogicalResult TensorInitOp::emitMojo(EmitMojoState &state) {
  auto tensorTy = MO::dynCastFromTensorLike(getType());
  MojoValue dTypeParam = state.mlirDTypeToMojoType(tensorTy.getDtype());
  auto mutParam = getMut();
  auto exclusiveParam = getExclusive();
  auto ioParam = MojoValue(HardcodedMojoSymbols::struct_IO)
                     .get(MO::stringifyIOKind(getIo()));
  MojoValue rankParam =
      MojoValue(static_cast<uint64_t>(tensorTy.getRankNonZero()));
  auto shapeParam =
      state.builder.comptime(state.getStaticShapeAsMojoValue(tensorTy));
  auto stridesParam =
      state.builder.comptime(state.getStaticStridesAsMojoValue(tensorTy));
  MojoValue alignmentParam(
      static_cast<uint64_t>(M::Driver::getPreferredAlignment(
          MO::getDeviceOrDefaultAny(tensorTy).getLabel().str())));
  auto shapeVar = state.builder.var(state.getShapeAsIndexListMojoValue(
      tensorTy.getTypedShapeAttr(), getDims()));
  auto moggTensorInit = state.getOriginalMojoNameFor(
      state.analysis.getRegisteredLITMojoKernelFor(this->getOperationName()));
  state.setMojoValueFor(
      getResult(), state.builder.var(moggTensorInit(
                       {dTypeParam, rankParam, mutParam, ioParam, shapeParam,
                        stridesParam, alignmentParam, exclusiveParam},
                       {state.getMojoValueFor(getBuffer())
                            .get("data")
                            .get("bitcast")({"NoneType"}),
                        shapeVar})));

  return success();
}

//===----------------------------------------------------------------------===//
// CallStubOp
//===----------------------------------------------------------------------===//

KernelStubOp CallStubOp::getCallee() {
  mlir::SymbolTable symTable(getOperation()->getParentOfType<ModuleOp>());
  return dyn_cast_or_null<KernelStubOp>(symTable.lookup(getName()));
}

LogicalResult CallStubOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto calleeFunc = symbolTable.lookupNearestSymbolFrom<KernelStubOp>(
      *this, getCalleeSymbol());
  if (!calleeFunc) {
    return emitError(
        llvm::formatv("There is no kernel stub named '{}'", getName()));
  }

  auto calleeArgumentTypes = calleeFunc.getArgumentTypes();
  auto callerOperandTypes = getOperandTypes();

  if (calleeArgumentTypes.size() != callerOperandTypes.size()) {
    return emitError(llvm::formatv("Callee expected {} arguments but got {}",
                                   calleeArgumentTypes.size(),
                                   callerOperandTypes.size()));
  }

  for (uint64_t i = 0; i < calleeArgumentTypes.size(); ++i) {
    if (calleeArgumentTypes[i] == callerOperandTypes[i])
      continue;

    if (isa<MOGG::ContextType>(calleeArgumentTypes[i]) &&
        isa<MGP::DeviceContextType>(callerOperandTypes[i])) {
      continue;
    }

    return emitError(llvm::formatv(
        "Callee argument at position {} expected to be {} but got {}", i,
        stringifyMLIRObject(calleeArgumentTypes[i]),
        stringifyMLIRObject(callerOperandTypes[i])));
  }

  if (calleeFunc.getAbi() != ABIKind::MOGG)
    return emitError("Call site should be in the MOGG ABI");

  return success();
}

bool CallStubOp::isThrows() { return getCallee().isThrows(); }

LogicalResult CallStubOp::emitMojo(EmitMojoState &state) {
  state.builder.newline();
  MojoValue func(getName());
  if (getNumResults()) {
    state.setMojoValueFor(
        getResult(0),
        state.builder.var(func({}, state.getMojoValuesFor(getOperands()))));
  } else {
    state.builder.call(func, {}, state.getMojoValuesFor(getOperands()));
  }
  state.builder.newline();

  return success();
}

//===----------------------------------------------------------------------===//
// AsyncReadyOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncReadyOp::emitMojo(EmitMojoState &state) {
  // If none of the users of this op is an output op, this is probably because
  // we sequence fused and then we remove the output chain because nothing was
  // using it from the outside. The result is an AsyncReadyOp with no output op
  // as a user. There is nothing to emit and this op can be treated as a no-op.
  /// TODO: Canonicalize this op into a placeholder for a chain instead of doing
  /// this context sensitive logic.
  if (llvm::none_of(this->getResult().getUsers(), [](auto user) {
        return isa<MOGG::OutputOp, MOGG::YieldOp>(user);
      })) {
    return success();
  }
  auto resultIndexOr = getOutputIndex();
  if (!resultIndexOr)
    return failure();
  auto moggAsyncReadyFunc = state.getOriginalMojoNameFor(
      state.analysis.getRegisteredLITMojoKernelFor(this->getOperationName()));
  state.builder.call(moggAsyncReadyFunc, {},
                     state.mojoReturnValues[*resultIndexOr]);

  return success();
}

//===----------------------------------------------------------------------===//
// ChainDummyOp
//===----------------------------------------------------------------------===//

LogicalResult ChainDummyOp::emitMojo(EmitMojoState &) {
  // Do nothing because the consuming op should never be using the chain.
  return success();
}

//===----------------------------------------------------------------------===//
// ChainSinkOp
//===----------------------------------------------------------------------===//

LogicalResult ChainSinkOp::emitMojo(EmitMojoState &) {
  // Do nothing because the producing op should never be returning a chain
  // anyway.
  return success();
}

LogicalResult ChainSinkOp::canonicalize(ChainSinkOp sinkOp,
                                        mlir::PatternRewriter &rewriter) {
  if (auto dummyOp = sinkOp.getChain().getDefiningOp<ChainDummyOp>()) {
    rewriter.eraseOp(sinkOp);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// AsyncPackBorrowV2Op
//===----------------------------------------------------------------------===//

LogicalResult AsyncPackBorrowV2Op::emitMojo(EmitMojoState &state) {
  auto resultIndexOr = getOutputIndex();
  if (!resultIndexOr)
    return failure();
  MojoValue moggAsyncPackBorrowFunc{
      HardcodedMojoSymbols::func_mogg_async_pack_borrow_v2};
  MojoValue specRankParam;
  MojoValue isTensorParam;
  if (isa<MGP::DeviceBufferType>(getInput().getType())) {
    specRankParam = 1;
    isTensorParam = false;
  } else if (auto tensorType =
                 dyn_cast<MGP::DeviceTensorType>(getInput().getType())) {
    specRankParam = tensorType.getRank();
    isTensorParam = true;
  }

  state.builder.call(moggAsyncPackBorrowFunc, {specRankParam, isTensorParam},
                     {/* borrower */ state.mojoReturnValues[*resultIndexOr],
                      /* borrowee */ state.getMojoValueFor(getInput()),
                      state.getMojoValueFor(getMem())});

  return success();
}

//===----------------------------------------------------------------------===//
// AsyncUnpackAnyAsyncValueRefOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncUnpackAnyAsyncValueRefOp::canonicalize(
    AsyncUnpackAnyAsyncValueRefOp unpackOp, mlir::PatternRewriter &rewriter) {
  ASSERT_STREAM(!unpackOp.getValue().getDefiningOp<AsyncPackOp>(),
                << "WE do not support unpacking async memory refs from packed "
                   "operations. This should not happen.");
  if (auto packOp = unpackOp.getValue().getDefiningOp<AsyncPackBorrowV2Op>()) {
    rewriter.replaceOp(unpackOp, packOp.getMem());
    return success();
  }

  return failure();
}

enum class MGPRT_BorroweeType : size_t {
  kHandle = 0,
  kBuffer = 1,
  kTensor = 2,
};

LogicalResult AsyncUnpackAnyAsyncValueRefOp::emitMojo(EmitMojoState &state) {
  MojoValue getBufferMemStorageHandleFn =
      HardcodedMojoSymbols::func_get_buffer_mem_storage_handle;
  MojoValue inputVar = state.getMojoValueFor(getValue());
  MojoValue typeVar = isa<MGP::DeviceBufferType>(getValue().getType())
                          ? (size_t)MGPRT_BorroweeType::kBuffer
                          : (size_t)MGPRT_BorroweeType::kTensor;
  auto opaquePtr = state.builder.var(
      MojoValue{HardcodedMojoSymbols::comptime_OpaquePointer}());
  auto opaquePtrPtr = state.builder.var(
      MojoValue{HardcodedMojoSymbols::struct_UnsafePointer}(
          {}, MojoValue("to").equal(opaquePtr))
          .get(HardcodedMojoSymbols::func_UnsafePointer_bitcast)({"NoneType"}));

  state.builder.call(getBufferMemStorageHandleFn, {},
                     {inputVar, typeVar, opaquePtrPtr});
  state.setMojoValueFor(getResult(), opaquePtr);
  return success();
}

//===----------------------------------------------------------------------===//
// CallViewOp
//===----------------------------------------------------------------------===//

LogicalResult CallViewOp::emitMojo(EmitMojoState &state) {
  // Get the kernel inputs.
  SmallVector<MojoValue> inputMojoValues;
  for (auto input : getInputs()) {
    MojoValue inputValue = state.getMojoValueFor(input);
    // Transform mutable input tensors - view ops always expect InputTensor
    if (isMutableInputTensor(input)) {
      inputValue = generateMutableTensorInit(input, state);
    }
    inputMojoValues.push_back(inputValue);
  }

  KGEN::LIT::FnOp function =
      state.analysis.getRegisteredLITMojoUpdateViewFunctionFor(getKernelName());

  // Infer the kernel parameters.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    constexpr StringLiteral staticOutShapeParamName = "output_static_shape";
    if (paramName == staticOutShapeParamName) {
      return state.builder.comptime(
          state.getStaticShapeAsMojoValue(getResult().getType()));
    }
    return state.inferDefaultKernelParamValue(*this, paramName);
  };
  auto maybeParams = EmitMojoState::inferKernelParameters(function, inferParam);
  if (maybeParams.isError())
    return emitError("Failed to infer all the kernel parameters");

  state.setMojoValueFor(getResult(),
                        state.builder.var(callFunctionNestedInStruct(
                            function, *maybeParams, inputMojoValues, state)));
  return success();
}

//===----------------------------------------------------------------------===//
// CallElementwiseOp
//===----------------------------------------------------------------------===//

LogicalResult CallElementwiseOp::emitMojo(EmitMojoState &state) {
  // Get the lambda inputs
  SmallVector<MojoValue> inputs = state.getMojoValuesFor(getInputs());

  auto function = state.analysis.getRegisteredLITMojoElementwiseFunctionFor(
      getKernelName());

  auto outputs = getOutputs();
  if (outputs.size() != 1)
    return emitError(
        "elementwise with outputs size != 1 not supported for EmitMojo");
  Value outputVal = outputs.front();

  // Get the lambda parameters if needed.
  // For ops like mo.cast, where the output dtype differs, we need to explicitly
  // specify the output dtype.
  // TODO(GEX-2535): Avoid hardcoding the parameter name.
  SmallVector<MojoValue> parameters;
  constexpr StringLiteral simdDtypeParameterName = "dtype";
  if (auto outDtypeParamName = state.getOriginalResultParamNameFor(
          function, simdDtypeParameterName)) {
    ErrorOr<DType> resultTyOr = M::MO::fromMLIRType(outputVal.getType());
    if (resultTyOr.isError())
      return emitError("invalid output type can't be converted to mojo dtype");
    parameters.push_back(
        outDtypeParamName->equal(state.mlirDTypeToMojoType(*resultTyOr)));
  }

  auto mojoSymbolName = state.getOriginalMojoNameFor(function);

  state.setMojoValueFor(outputVal,
                        state.builder.var(mojoSymbolName(parameters, inputs)));
  return success();
}

//===----------------------------------------------------------------------===//
// InlineConstantBufferOp
//===----------------------------------------------------------------------===//

LogicalResult InlineConstantBufferOp::emitMojo(EmitMojoState &state) {
  // Get the raw value.
  ArrayRef<uint8_t> rawValue;
  if (auto arrayAttr = dyn_cast<M::ArrayElementsAttr>(getValue())) {
    rawValue = arrayAttr.getRawData();
  } else if (auto resourceAttr =
                 dyn_cast<DenseResourceElementsAttr>(getValue())) {
    DenseResourceElementsHandle handle = resourceAttr.getRawHandle();
    mlir::AsmResourceBlob *blob = handle.getBlob();
    rawValue = blob->getDataAs<uint8_t>();
  } else {
    return failure();
  }

  // We use pop.global_constant to emit inline constant data (as was used in the
  // older KGEN path)
  //
  // Ideally we'd have this wrapped up in a nice function that we could just
  // invoke on the Mojo side, but I wasn't able to figure out how to
  // parameterize the raw mlir we're generating with the array size.
  //
  // Instead directly invoke pop.global_constant op via the __mlir_op machinery
  // of Mojo.
  //
  static constexpr StringLiteral popArrayStr = "#pop.array<";

  std::string arrayAttrStr;
  // Multiply by 5 as the worst case since we can have at most 3 digits and the
  // string ', '
  arrayAttrStr.reserve(rawValue.size() * 5 + popArrayStr.size());
  llvm::raw_string_ostream arrayAttrStream(arrayAttrStr);

  // Build up the pop.array constant attribute
  arrayAttrStream << popArrayStr;
  llvm::interleave(
      rawValue, arrayAttrStream,
      // Cast to 'unsigned' as printing the uint8_t values directly prints them
      // as chars.
      [&](uint8_t value) { arrayAttrStream << static_cast<unsigned>(value); },
      ", ");
  arrayAttrStream << ">";

  std::string popArrayTypeStr =
      llvm::formatv("!pop.array<{0}, ui8>", rawValue.size()).str();
  std::string kgenPtrTypeStr =
      llvm::formatv("!kgen.pointer<array<{0}, ui8>>", rawValue.size()).str();

  // Emit the __mlir_op.pop.global_constant operation
  MOGG::MojoValue arrPtr = state.builder.var(MOGG::MojoValue(
      llvm::formatv(
          "__mlir_op.`pop.global_constant`[value = __mlir_attr[`{0}`, `: "
          "{1}`], _type = __mlir_type[`{2}`]]()",
          arrayAttrStream.str(), popArrayTypeStr, kgenPtrTypeStr)
          .str()));

  // Create UnsafePointer from the array pointer so we can easily bitcast to
  // the expected pointer of NoneType.
  MOGG::MojoValue altPtr = state.builder.var(MOGG::MojoValue(
      "LegacyUnsafePointer")({}, {MOGG::MojoValue("value").equal(arrPtr)}));

  // Bitcast to NoneType (expected dtype for the buffers).
  MOGG::MojoValue opaquePtrVar = state.builder.var(
      altPtr.get("bitcast")({MOGG::MojoValue("NoneType")}, {}));
  MOGG::MojoValue mgpBufferConstantFn{
      MOGG::HardcodedMojoSymbols::func_buffer_constant};
  state.setMojoValueFor(
      getResult(), mgpBufferConstantFn({}, {opaquePtrVar, rawValue.size()}));
  return success();
}

//===----------------------------------------------------------------------===//
// LoadTensorOp
//===----------------------------------------------------------------------===//

LogicalResult LoadTensorOp::emitMojo(EmitMojoState &state) {
  const MojoValue &tensor = state.getMojoValueFor(getTensor());
  const MojoValue &indices = state.getMojoValueFor(getIndices());
  const MojoValue &widthParam = state.currentFunctionBody.parameters[0];
  const MojoValue &alignmentParam = state.currentFunctionBody.parameters[1];

  // MOGGLoadOp is only used inside an output lambda
  // Mark the tensor as keep alive to ensure it doesn't get deallocated.
  state.markAsKeepAlive(tensor);

  // TODO: Infer the correct operand index.
  state.addDebugPrintTensorSpecs(tensor, indices, /*isInput=*/true, 0);
  MOGG::MojoValue loadedSimd =
      MojoValue(HardcodedMojoSymbols::func_simd_load_from_managed_tensor_slice)(
          {MojoValue("simd_width").equal(widthParam),
           MojoValue("element_alignment").equal(alignmentParam)},
          {tensor, indices});
  state.setMojoValueFor(getOutputs(), state.builder.var(loadedSimd));
  return success();
}

//===----------------------------------------------------------------------===//
// EmptyBufferOp
//===----------------------------------------------------------------------===//

LogicalResult EmptyBufferOp::emitMojo(EmitMojoState &state) {
  // Represent the empty buffer with a NULL pointer.
  state.setMojoValueFor(
      getResult(),
      state.builder.var(MojoValue(MOGG::HardcodedMojoSymbols::struct_NDBuffer)(
          {MojoValue(HardcodedMojoSymbols::struct_DType).get("uint8"), 1,
           HardcodedMojoSymbols::comptime_MutAnyOrigin})));
  return success();
}

//===----------------------------------------------------------------------===//
// TensorFusedOp
//===----------------------------------------------------------------------===//

LogicalResult TensorFusedOp::emitMojo(EmitMojoState &state) {

  BindLambdaOp lambdaOp = getBindLambdaOp();
  MojoValue lambdaParam = state.getMojoValueFor(getLambda());
  MojoValue tensorValue = state.getMojoValueFor(getTensor());

  // Figure out if we need to build a fused input / output / compute tensor
  // object.
  auto bindMethodNameOr = [&]() -> ErrorOr<StringLiteral> {
    if (lambdaOp.getBindKind() == BindLambdaKind::InputLambda)
      return HardcodedMojoSymbols::func_ManagedTensorSlice_bind_to_fused_input;

    // For the output lambdas, we need to look at the kernel signature to figure
    // out if we need to build a FusedOutputTensor or a FusedComputeOutputTensor
    // object.
    auto outputLambdaTypeOr = getKernelOutputFusionLambdaType(
        state.analysis, *this, state.enableComputeLambdas);
    if (outputLambdaTypeOr.isError())
      return outputLambdaTypeOr.takeError();

    return *outputLambdaTypeOr == KernelOutputFusionLambdaType::Output
               ? HardcodedMojoSymbols::
                     func_ManagedTensorSlice_bind_to_fused_output
               : HardcodedMojoSymbols::
                     func_ManagedTensorSlice_bind_to_fused_compute_output;
  }();
  if (bindMethodNameOr.isError())
    return emitError(bindMethodNameOr.getError());

  MojoValue fusedTensor = tensorValue.get(*bindMethodNameOr)(lambdaParam);
  state.setMojoValueFor(getResult(), state.builder.var(fusedTensor));
  return success();
}

LogicalResult TensorFusedOp::verify() {
  auto lambdaOp = getLambda().getDefiningOp<BindLambdaOp>();
  if (!lambdaOp)
    return emitError("`lambda` operand must be a BindLambdaOp");
  if (lambdaOp.getBindKind() != BindLambdaKind::InputLambda &&
      lambdaOp.getBindKind() != BindLambdaKind::OutputLambda)
    return emitError("`lambda` operand must be an input or output lambda");

  auto lambdaTy = MO::dynCastFromTensorLike(getLambda().getType());
  auto tensorTy = MO::dynCastFromTensorLike(getTensor().getType());
  auto resultTy = MO::dynCastFromTensorLike(getResult().getType());
  if (!lambdaTy || !tensorTy || !resultTy)
    return emitError("All operation types must be tensor / buffer");

  if (resultTy.getDtype() != lambdaTy.getDtype() ||
      resultTy.getDtype() != tensorTy.getDtype())
    return emitError("All operation types must have the same dtype");

  if (resultTy.getShapeAttr() != lambdaTy.getShapeAttr() ||
      resultTy.getShapeAttr() != tensorTy.getShapeAttr())
    return emitError("All operation types must have the same shape");

  return success();
}

//===----------------------------------------------------------------------===//
// AsTupleOp
//===----------------------------------------------------------------------===//

LogicalResult AsTupleOp::emitMojo(EmitMojoState &state) {
  SmallVector<MojoValue> inputValues = state.getMojoValuesFor(getInputs());
  if (isa<MOGG::ContextType>(getInputs().front().getType())) {
    MojoValue tupleConstructor =
        MojoValue(HardcodedMojoSymbols::struct_StaticTuple)[{
            MojoValue("size").equal(std::to_string(inputValues.size()))}];
    MojoValue deviceContextPtrListConstructor =
        MojoValue(HardcodedMojoSymbols::struct_DeviceContextPtrList);
    MojoValue inputTupleValue =
        deviceContextPtrListConstructor({}, tupleConstructor({}, inputValues));
    state.setMojoValueFor(getResult(), state.builder.var(inputTupleValue));
    return success();
  }

  SmallVector<MojoValue> runtimePointers = llvm::map_to_vector(
      inputValues, [&](const MojoValue &val) { return val.get("_ptr"); });

  SmallVector<MojoValue> runtimeShapes = llvm::map_to_vector(
      inputValues, [&](const MojoValue &val) { return val.get("shape")(); });

  MojoValue inputTupleConstructor =
      MojoValue(HardcodedMojoSymbols::struct_StaticTuple)[{
          MojoValue("size").equal(std::to_string(inputValues.size()))}];

  MojoValue runtimePointersTuple =
      state.builder.var(inputTupleConstructor({}, runtimePointers));
  MojoValue runtimeShapesTuple =
      state.builder.var(inputTupleConstructor({}, runtimeShapes));

  // Create the static specs tuple
  // comptime inputSpecs = StaticTuple[size=N](
  //     var1.static_spec, var2.static_spec, ..., varN.static_spec
  // )
  SmallVector<MojoValue> staticSpecs;
  for (const auto &reboundInput : inputValues) {
    staticSpecs.push_back(reboundInput.get("static_spec"));
  }
  MojoValue staticSpecsConstructor = MojoValue("StaticTuple")[{
      MojoValue("size").equal(std::to_string(staticSpecs.size()))}];
  MojoValue staticSpecsValue = staticSpecsConstructor({}, staticSpecs);
  MojoValue inputSpecsComptime = state.builder.comptime(staticSpecsValue);

  MojoValue variadicTensorsConstructor = MojoValue("VariadicTensors")[{
      MojoValue("io_spec").equal(inputValues[0].get("io_spec")),
      MojoValue("static_specs").equal(inputSpecsComptime)}];

  MojoValue variadicTensorsValue = variadicTensorsConstructor(
      {}, {runtimePointersTuple, runtimeShapesTuple});

  state.setMojoValueFor(getResult(), state.builder.var(variadicTensorsValue));
  return success();
}

//===----------------------------------------------------------------------===//
// CallShapeOp
//===----------------------------------------------------------------------===//

// MOGG_KernelCallInterface methods
Value CallShapeOp::getDeviceContext() {
  // Shape functions typically don't need device contexts
  return nullptr;
}

SmallVector<Value> CallShapeOp::getKernelInputs() { return getOperands(); }

SmallVector<Value> CallShapeOp::getKernelDPSOutputs() {
  // Shape functions don't have DPS outputs
  return {};
}

Type CallShapeOp::getKernelReturnOutput() {
  return getRetTypeAttr().getValue();
}

bool CallShapeOp::isThrows() { return getIsThrows(); }

LogicalResult CallShapeOp::emitMojo(EmitMojoState &state) {
  // Get the shape function from the analysis
  KGEN::LIT::FnOp function = state.analysis.getRegisteredLITMojoFunctionFor(
      getShapeFuncName(), KGEN::MOGGPreElab::kMOGGShapeFunctionLabel);

  if (!function) {
    return emitError("Shape function ") << getShapeFuncName() << " not found";
  }

  // Rebind all inputs to ensure specs are deduced (tensors -> InputTensor,
  // etc.)
  SmallVector<MojoValue> kernelOperands;
  for (Value operand : getOperands())
    kernelOperands.push_back(state.getMojoValueFor(operand));

  // Infer any kernel parameters required by the shape function.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    return state.inferDefaultKernelParamValue(*this, paramName);
  };
  auto maybeParams = EmitMojoState::inferKernelParameters(function, inferParam);
  if (maybeParams.isError())
    return emitError("Failed to infer all the kernel parameters");

  // Call the shape function and return its value via the kernel terminator.
  MojoValue result = state.builder.var(callFunctionNestedInStruct(
      function, *maybeParams, kernelOperands, state));
  state.genAllKeepAliveStatements();
  state.builder.returnsValue(result);

  return success();
}

//===----------------------------------------------------------------------===//
// ShapeResultGetItemOp
//===----------------------------------------------------------------------===//

LogicalResult ShapeResultGetItemOp::verify() {
  auto shapeType = dyn_cast<MOGG::ShapeFuncResultType>(getShape().getType());
  if (!shapeType) {
    return emitOpError("shape operand must be of ShapeFuncResultType");
  }

  int64_t index = getIndex().getValue().getSExtValue();
  int64_t rank = shapeType.getRank();

  if (index < 0 || index >= rank) {
    return emitOpError("index ")
           << index << " is out of bounds for shape result with rank " << rank;
  }

  return success();
}

LogicalResult ShapeResultGetItemOp::emitMojo(EmitMojoState &state) {
  const MojoValue &shapeValue = state.getMojoValueFor(getShape());
  int64_t index = getIndex().getValue().getSExtValue();

  // Access the shape result element by index
  MojoValue indexValue = MojoValue(std::to_string(index));
  state.setMojoValueFor(getResult(), shapeValue[{indexValue}]);

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantScalarOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantScalarOp::emitMojo(EmitMojoState &state) {
  if (auto intAttr = dyn_cast<IntegerAttr>(getValue())) {
    state.setMojoValueFor(getResult(), MojoValue(intAttr.getAPSInt()));
    return success();
  } else {
    return emitError("TODO: unsupported scalar type");
  }
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

LogicalResult OutputOp::emitMojo(EmitMojoState &) { return success(); }

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::emitMojo(EmitMojoState &) { return success(); }

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

LogicalResult TryOp::emitMojo(EmitMojoState &state) {
  bool errOccurred = false;
  state.builder
      .try_([&]() {
        // Lower try block.
        errOccurred |=
            llvm::any_of(getTryRegion().getOps(), [&](mlir::Operation &op) {
              return state.lowerOpToMojo(&op).failed();
            });
      })
      .except([&](const MojoValue &e) {
        // Lower except block.
        state.setMojoValueFor(getExceptRegion().getArgument(0), e);
        errOccurred |=
            llvm::any_of(getExceptRegion().getOps(), [&](mlir::Operation &op) {
              return state.lowerOpToMojo(&op).failed();
            });
      });

  return failure(errOccurred);
}

//===----------------------------------------------------------------------===//
// AsyncErrorOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncErrorOp::emitMojo(EmitMojoState &state) {
  auto moggAsyncErrorFunc = state.getOriginalMojoNameFor(
      state.analysis.getRegisteredLITMojoKernelFor(this->getOperationName()));

  // Each AsyncErrorOp corresponds to a specific output position in the try op.
  // Find which output index this AsyncErrorOp is for by checking which yield
  // operand uses this result.
  auto tryOp = cast<MOGG::TryOp>(getOperation()->getParentOp());
  auto exceptYield =
      cast<MOGG::YieldOp>(tryOp.getExceptRegion().front().getTerminator());

  // Find the index of this AsyncErrorOp's result in the yield operands
  int outputIndex = -1;
  for (auto [idx, operand] : llvm::enumerate(exceptYield.getOperands())) {
    if (operand == getResult()) {
      outputIndex = idx;
      break;
    }
  }

  if (outputIndex < 0 ||
      static_cast<size_t>(outputIndex) >= state.mojoReturnValues.size()) {
    return emitOpError("Could not find corresponding output index for "
                       "AsyncErrorOp. Found index ")
           << outputIndex << " but have " << state.mojoReturnValues.size()
           << " return values";
  }

  // Emit only one call for this specific output pointer
  state.builder.call(
      moggAsyncErrorFunc, {},
      {state.mojoReturnValues[outputIndex], state.getMojoValueFor(getError())});

  return success();
}

//===----------------------------------------------------------------------===//
// TensorReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult TensorReshapeOp::emitMojo(EmitMojoState &state) {
  // Call "tmp_reshape_contiguous_buffer" primitive with the output shape /
  // strides.
  MO::TensorType toType = MO::castFromTensorLike(getResult().getType());
  MojoValue tensorValue = state.getMojoValueFor(getTensor());
  MojoValue shapeValue = state.getShapeAsIndexListMojoValue(
      toType.getTypedShapeAttr(), getDynamicDims());
  MojoValue shapeParam = state.getStaticShapeAsMojoValue(toType);
  MojoValue stridesParam = state.getStaticStridesAsMojoValue(toType);
  state.setMojoValueFor(
      getResult(), state.builder.var(MojoValue(
                       HardcodedMojoSymbols::func_reshape_contiguous_buffer)(
                       {MojoValue("static_shape").equal(shapeParam),
                        MojoValue("static_stride").equal(stridesParam)},
                       {tensorValue, shapeValue})));
  return success();
}

//===----------------------------------------------------------------------===//
// AsyncPlaceholderAnyAsyncValueRefOp
//===----------------------------------------------------------------------===//
LogicalResult
AsyncPlaceholderAnyAsyncValueRefOp::emitMojo(EmitMojoState &state) {
  MojoValue result = state.builder.var(
      MojoValue{HardcodedMojoSymbols::comptime_OpaquePointer}());
  state.setMojoValueFor(getResult(), result);
  return success();
}

//===----------------------------------------------------------------------===//
// ShapeInitOp
//===----------------------------------------------------------------------===//

LogicalResult ShapeInitOp::emitMojo(EmitMojoState &state) {
  MojoValue shapeValue =
      state.getShapeAsIndexListMojoValue(getShape(), getDynamicDims());
  state.setMojoValueFor(getResult(), shapeValue);
  return success();
}

//===----------------------------------------------------------------------===//
// GenericCompiledRegionOp Parsing/Printing
//===----------------------------------------------------------------------===//

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

  // For GenericCompiledRegionOp, if the kernel takes in a chain, there will be
  // 1 fewer argument than operands.
  operandTypes.resize(operands.size(), MO::ChainType::get(p.getContext()));

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
