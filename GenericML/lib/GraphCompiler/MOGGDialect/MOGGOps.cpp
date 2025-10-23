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
#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "GenericML/GraphCompiler/Interfaces/EmitMojoState.h"
#include "GenericML/GraphCompiler/Interfaces/LoweringState.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOGGCompiler.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/TypeManager.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTypes.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/MOGGPreElab/MOGGPreElabHelpers.h"
#include "KGEN/MOGGPreElab/MojoKernelAdaptor.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/AssertStream.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringMap.h"
#include <future>

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
  ErrorOr<llvm::SmallVector<MojoValue>> maybeParams =
      EmitMojoState::inferKernelParameters(function, inferParam);
  if (maybeParams.isError())
    return emitError("Failed to infer all the kernel parameters");

  auto results = getResults();
  if (results.empty()) {
    state.builder.call(symbolName, *maybeParams, mojoOperands);
  } else if (results.size() == 1) {
    state.setMojoValueFor(results.front(), state.builder.var(symbolName(
                                               *maybeParams, mojoOperands)));
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
  b.create<MOGG::OutputOp>(state.location, block.getArguments());
}

LogicalResult BindLambdaOp::emitMojo(EmitMojoState &state) {

  bool generateStoreForOutput = getBindKind() == BindLambdaKind::OutputLambda;

  // Build the corresponding lambda.
  MO::TensorType outputTy = MO::castFromTensorLike(getOutput().getType());

  auto widthParamName = state.builder.getIdentifier("width");

  Block *lambdaBody = getBody();

  // Prepare the inputs
  SmallVector<MojoValue> fusedInputs;
  for (MojoValue input : state.getMojoValuesFor(getFusibleTensors())) {
    fusedInputs.push_back(input);

    // The Compiler doesn't know the body lambda is using the tensor, and can't
    // keep it alive.
    state.markAsKeepAlive(input);
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
  if (generateStoreForOutput) {
    // Output lambda has an extra argument which is the kernel output.
    lambdaArguments.push_back(NamedMojoDef(
        "input",
        EmitMojoState::getSIMDType(outputTy.getDtype(),
                                   state.builder.newVar(widthParamName))));
  }

  // The body lambda has with / alignment parameters and indices arguments.
  MojoValue lambdaParam = state.builder.lambda(
      lambdaName,
      {NamedMojoDef(widthParamName, HardcodedMojoSymbols::struct_Int),
       NamedMojoDef("alignment", HardcodedMojoSymbols::struct_Int)},
      lambdaArguments, returnType,
      /*isInline=*/true,
      /*isParameter=*/true, [&](const MOGG::FunctionBody &functionBody) {
        state.currentFunctionBody = functionBody;
        const MojoValue &widthParam = functionBody.parameters[0];
        const MojoValue &alignmentParam = functionBody.parameters[1];
        const MojoValue &indicesInput = functionBody.arguments[0];

        // Do all load for each inputs.
        for (unsigned i = 0; i < fusedInputs.size(); ++i) {
          Value blockArg = lambdaBody->getArgument(i);
          if (i == 0 && generateStoreForOutput) {
            // For the output lambda, just map the argument to the last operand.
            state.setMojoValueFor(blockArg, functionBody.arguments.back());
          } else {
            // Do a load from the input tensor
            const auto &mojoInput = fusedInputs[i];
            MOGG::MojoValue loadedSimd = MojoValue(
                HardcodedMojoSymbols::func_simd_load_from_managed_tensor_slice)(
                {MojoValue("simd_width").equal(widthParam),
                 MojoValue("element_alignment").equal(alignmentParam)},
                {mojoInput, indicesInput});
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
          state.builder.call(
              HardcodedMojoSymbols::func_simd_store_into_managed_tensor_slice,
              ArrayRef<MojoValue>{
                  MojoValue("element_alignment").equal(alignmentParam)},
              {fusedInputs.front(), resultIndices, resultValue});
        } else {
          // Add the return statement.
          state.builder.returnsValue(resultValue);
        }
      });

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
  MojoValue targetMojoParam =
      MojoValue(MO::getDeviceOrDefaultAny(*this).getLabel()).quote();
  MojoValue widthMojoParam =
      MojoValue(HardcodedMojoSymbols::func_get_simd_width_for_dtypes)(
          {paramsDtypesTuple, targetMojoParam});

  static const StringLiteral foreachBodyLambdaParamName = "func";
  static const StringLiteral foreachSimdWidthParamName = "simd_width";
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    if (paramName == foreachBodyLambdaParamName)
      return bodyLambdaParam;
    if (paramName == foreachSimdWidthParamName)
      return widthMojoParam;
    return state.inferDefaultKernelParamValue(*this, paramName);
  };
  ErrorOr<llvm::SmallVector<MojoValue>> maybeParams =
      EmitMojoState::inferKernelParameters(function, inferParam);
  if (maybeParams.isError())
    return emitError("Failed to infer all the kernel parameters");

  auto mojoSymbolName = state.getOriginalMojoNameFor(function);

  state.builder.call(mojoSymbolName, *maybeParams, {mojoOutput, contextValue});
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

LogicalResult CallMaterializeOp::emitMojo(EmitMojoState &state) {
  auto function = state.analysis.getLITViewMaterializeKernel();

  // Get the kernel operands.
  SmallVector<MojoValue> mojoOperands = {state.getMojoValuesFor(getOperands())};
  std::swap(mojoOperands[0], mojoOperands[1]);

  // Infer the kernel parameters.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    return state.inferDefaultKernelParamValue(*this, paramName);
  };
  ErrorOr<llvm::SmallVector<MojoValue>> maybeParams =
      EmitMojoState::inferKernelParameters(function, inferParam);
  if (maybeParams.isError())
    return emitError("Failed to infer all the kernel parameters");

  auto mojoSymbolName = state.getOriginalMojoNameFor(function);

  state.builder.call(mojoSymbolName, *maybeParams, mojoOperands);
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

LogicalResult CallExecuteOp::emitMojo(EmitMojoState &state) {
  auto function =
      state.analysis.getRegisteredLITMojoExecuteFunctionFor(getKernelName());

  // Collect all kernel I/Os.
  SmallVector<MojoValue> kernelOperands;
  for (Value val : getKernelDPSOutputs())
    kernelOperands.push_back(state.getMojoValueFor(val));
  for (Value val : getKernelInputs())
    kernelOperands.push_back(state.getMojoValueFor(val));

  // And the optional context.
  if (auto maybeContext = getDeviceContext())
    kernelOperands.push_back(state.getMojoValueFor(maybeContext));

  // Infer the kernel parameters.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    return state.inferDefaultKernelParamValue(*this, paramName);
  };
  ErrorOr<llvm::SmallVector<MojoValue>> maybeParams =
      EmitMojoState::inferKernelParameters(function, inferParam);
  if (maybeParams.isError())
    return emitError("Failed to infer all the kernel parameters");

  auto mojoSymbolName = state.getOriginalMojoNameFor(function);

  state.builder.call(mojoSymbolName, *maybeParams, kernelOperands);
  state.genAllKeepAliveStatements();

  return success();
}

// These FuncOp attributes are disallowed while parsing since they can
// be inferred. Likewise while printing we ignore them.
static StringRef kernelStubOpdisallowedAttrNames[] = {"name", "returnType"};

ParseResult KernelStubOp::parse(OpAsmParser &parser, OperationState &state) {
  StringAttr name;
  if (failed(parser.parseSymbolName(name)))
    return failure();
  state.addAttribute("name", name);

  SmallVector<Type> resTypes;
  SmallVector<OpAsmParser::Argument> args;
  if (failed(parseRegionArguments(parser, resTypes, args)))
    return failure();

  if (!resTypes.empty())
    state.addAttribute("returnType", TypeAttr::get(resTypes.front()));

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
  p.printSymbolName(getName());
  p << " ";
  printRegionArguments(p, getReturnType() ? *getReturnType() : TypeRange{},
                       getBody().getArguments());
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     kernelStubOpdisallowedAttrNames);
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// MOGG ExpKernelOp
//===----------------------------------------------------------------------===//

LogicalResult ExpKernelOp::lower(MGP::MOGraphLoweringState &loweringState) {
  MGP::MOOpLoweringState opLoweringState(loweringState, *this);
  auto &builder = opLoweringState.builder;
  mlir::MLIRContext *ctx = getContext();
  auto loc = getLoc();

  // Outline the kernel into a stub by creating the op and taking the body of
  // the kernel op we are lowering. Output and scratch placeholders are part of
  // the MGP memory management analysis which means they will come from the
  // outside.
  Block *kernelBody = &getExecBody().front();
  auto outputPlaceholdersOp = getOutputPlaceholder();
  if (outputPlaceholdersOp) {
    for (auto value : outputPlaceholdersOp.getResults())
      value.replaceAllUsesWith(kernelBody->addArgument(value.getType(), loc));
  }

  auto scratchPlaceholdersOp = getScratchPlaceholder();
  if (scratchPlaceholdersOp) {
    for (auto value : scratchPlaceholdersOp.getResults())
      value.replaceAllUsesWith(kernelBody->addArgument(value.getType(), loc));
  }

  // If the kernel returns opaque types, they must be part of the stub's
  // signature.
  uint64_t resultTypeIndex;
  SmallVector<MO::OpaqueType> resultTypes;
  for (auto [i, resultType] : llvm::enumerate(getResultTypes())) {
    if (auto opaqueType = dyn_cast<MO::OpaqueType>(resultType)) {
      resultTypeIndex = i;
      resultTypes.push_back(opaqueType);
    }
  }

  if (resultTypes.size() > 1) {
    this->emitError(
        "MGP lowering does not support multiple returned opaque types");
    return failure();
  }

  bool isThrows = llvm::any_of(getExecBody().getOps(), [](Operation &op) {
    // ForEach / Materialize kernels always throw.
    if (isa<CallForeachOp, CallMaterializeOp>(op))
      return true;

    if (auto execOp = dyn_cast<CallExecuteOp>(op)) {
      // Some kernels throw.
      return execOp.getIsThrows();
    }

    // We assume all other ops don't throw.
    return false;
  });

  // Create the stub op at the top of the module.
  /// TODO: Find a better way of naming stubs.
  static int stubCounter = 0;
  auto stubName =
      StringAttr::get(ctx, llvm::formatv("stub_{}", stubCounter++).str());
  auto kernelStub = builder.create<MOGG::KernelStubOp>(
      loc, stubName,
      resultTypes.empty() ? nullptr : TypeAttr::get(resultTypes.front()),
      getSummaryAttr(), isThrows);
  kernelStub.getBody().takeBody(getExecBody());

  // This is a list of input operands in MGP but without the chain.
  auto loweredInputOperands = opLoweringState.getLoweredOperands();
  Value chain = loweringState.getCurrentExecuteChain();
  if (maybeGetInChain()) {
    chain = loweringState.lookupExecuteLoweredValue(*maybeGetInChain());
    loweredInputOperands.pop_back();
  }
  chain = opLoweringState.getGuardChain(chain);

  // This is a list of all the outputs of the kernel but lowered to MGP.
  SmallVector<MGP::MOOutputStorage> loweredOutputsStorage;
  auto loweredOutputsStorageOr = opLoweringState.createOutputStorage(chain);
  if (failed(loweredOutputsStorageOr))
    return failure();
  loweredOutputsStorage = *std::move(loweredOutputsStorageOr);

  // Scratch output placeholders turn into DPS kernel inputs. This makes it
  // possible to do memory scheduling on temporary storage. Lower the output of
  // the scratch placeholders into MGP types.
  SmallVector<MGP::MOOutputStorage> scratchOutputsStorage;
  if (scratchPlaceholdersOp) {
    for (Type t : scratchPlaceholdersOp.getResultTypes()) {
      auto specResult =
          opLoweringState.createTensorSpec(cast<MO::TensorType>(t));
      if (failed(specResult))
        return failure();

      auto bufOr = opLoweringState.createBuffer(chain, *specResult, t);
      if (failed(bufOr))
        return failure();
      scratchOutputsStorage.emplace_back(*bufOr);
    }
  }

  // Staging all the necessary inputs to create the mgp.generic.compiled.region
  // op.
  SmallVector<Value> operands;
  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics;
  SmallVector<mlir::Type> newKernelResTypes;

  // This maps an outside SSA value into its position in the operand list of the
  // mogg.generic.compiled.region.
  DenseMap<Value, uint64_t> outsideValueToOperandIndex;
  auto addOperandAndMapToIndex = [&](Value operand,
                                     InOutSignatureAttr::InOutSemantics sig) {
    outsideValueToOperandIndex[operand] = operands.size();
    operands.emplace_back(operand);
    operandSemantics.emplace_back(sig);
  };

  // 1. Add the chain with the input signature.
  addOperandAndMapToIndex(chain, InOutSignatureAttr::kIn);

  // 2. Add the inputs with the input or mutable input signature.
  // Use zip_first because we removed the original chain from the list.
  for (auto &&[origOperand, loweredInputOperand] :
       llvm::zip_first(getOperands(), loweredInputOperands)) {
    auto sig = isa<MO::BufferType>(origOperand.getType())
                   ? InOutSignatureAttr::kMut
                   : InOutSignatureAttr::kIn;
    addOperandAndMapToIndex(loweredInputOperand, sig);
  }

  // 3. Add the outputs with the output signature.
  for (auto &&outputStorage : loweredOutputsStorage)
    addOperandAndMapToIndex(outputStorage.buffer, InOutSignatureAttr::kOut);

  // 4. Add scratch buffers with the none signature because it is only used
  // internally.
  for (auto &&scratchStorage : scratchOutputsStorage)
    addOperandAndMapToIndex(scratchStorage.buffer, InOutSignatureAttr::kNone);

  // 5. In MGP, parameters turn into values. Add the values of input parameters
  // with the input signature. The following maps between a parameter to a
  // lowered SSA value. We use TypedAttr instead of ParamDeclRefAttr to make it
  // possible to search integers (they won't be present obviously).
  DenseMap<TypedAttr, Value> parameterToOutsideSSAValue;
  for (auto param : getKernelUsedParameters(*this)) {
    auto symbolicDimOr = loweringState.lowerExpression(
        opLoweringState.opBeingLowered->getParentRegion(), opLoweringState.loc,
        param);
    Value symbol = symbolicDimOr.takeValue();
    addOperandAndMapToIndex(symbol, InOutSignatureAttr::kIn);
    parameterToOutsideSSAValue[param] = symbol;
  }

  // 6. Add a chain to the result types.
  newKernelResTypes.push_back(chain.getType());

  // 7. The original kernel could also return an opaque type which is allocated
  // inside of its own body. If that's the case, add it to the output of the
  // mogg.generic.compiled.region op since it is not in DPS form. The current
  // contract is that if a kernel op returns an opaque type, it must have been
  // allocated internally. Tensors can't be returned in a non-DPS form.
  DeviceRefAttr device = MO::getDeviceOrDefaultAny(*this);
  if (!resultTypes.empty()) {
    newKernelResTypes.push_back(MGP::OpaqueType::get(
        ctx, resultTypes.front().getSymbol(), std::nullopt));
  }

  // 8. The original kernel may be setting some parameters (shape functions do
  // just that). These will returned from the mogg.generic.compiled.region.
  for (size_t i = 0; i < getOutputParamDecls().size(); i++)
    newKernelResTypes.push_back(getOutputParamDecls()[i].getType());

  auto newKernel = builder.create<MOGG::GenericCompiledRegionOp>(
      getLoc(), TypeRange{newKernelResTypes}, operands,
      getOutputParamDeclsAttr(), InOutSignatureAttr::get(ctx, operandSemantics),
      getSummaryAttr(),        // summary
      getIsAsyncAttr(),        // isAsync
      ArrayAttr::get(ctx, {}), // KernelOp has no constant resources
      BoolAttr::get(ctx, getAllocIndices().empty() &&
                             !getIsAsync()), // canFuseIntoOther
      BoolAttr::get(ctx, true),              // canBeFusedInto
      BoolAttr::get(ctx, false),             // canDuplicate
      BoolAttr::get(ctx, true)               // emitsMojo
  );
  newKernel.setDeviceRef(device);

  // Add a body to the created operation and mirror its operands with block
  // arguments.
  mlir::Block &regionBody = newKernel.getBody().emplaceBlock();
  builder.setInsertionPoint(&regionBody, regionBody.begin());
  SmallVector<Location> locs =
      llvm::map_to_vector(operands, [](Value v) { return v.getLoc(); });
  regionBody.addArguments(ValueRange(operands).getTypes(), locs);

  // This mapping is similar to `outsideValueToOperandIndex`.
  // But it covers the parameters needed inside the region stub.
  // Usually these are fused parameters of fused-intermediate tensors.
  // They are created inside the stub directly.
  DenseMap<Value, uint64_t> outsideValueToStubOperandIndex;

  auto getTensorDynamicDimensions =
      [&](MO::TensorType asTensor,
          bool isInsideStub) -> ErrorOr<SmallVector<Value>> {
    SmallVector<Value> dynamicDimensions;
    for (auto dim : asTensor.getDims()) {
      if (isa<IntegerAttr>(dim))
        continue;

      auto outsideSSAIt = parameterToOutsideSSAValue.find(dim);
      if (outsideSSAIt == parameterToOutsideSSAValue.end())
        return Error(
            llvm::formatv("Failed to find the runtime value of the dynamic "
                          "tensor dimension #{} for tensor type {}",
                          dim, asTensor));
      Value outsideSSAValue = outsideSSAIt->second;

      if (isInsideStub) {
        if (!outsideValueToStubOperandIndex.contains(outsideSSAValue))
          return Error(
              llvm::formatv("Failed to find the stub argument of the dynamic "
                            "tensor dimension #{} for tensor type {}",
                            dim, asTensor));
        dynamicDimensions.push_back(kernelStub.getBody().getArgument(
            outsideValueToStubOperandIndex[outsideSSAValue]));
      } else {
        if (!outsideValueToOperandIndex.contains(outsideSSAValue))
          return Error(
              llvm::formatv("Failed to find the kernel argument of the dynamic "
                            "tensor dimension #{} for tensor type {}",
                            dim, asTensor));
        dynamicDimensions.push_back(regionBody.getArgument(
            outsideValueToOperandIndex[outsideSSAValue]));
      }
    }

    return dynamicDimensions;
  };

  // Call the previously created kernel stub by creating ops that bridge the gap
  // between MGP and MO/MOGG. We can safely zip_first iterate through the stub's
  // arguments and the regions operands (minus the chain that is always first)
  // because the stub and the region have the same arguments (except for the
  // added dimensions). Remember that we added them in the same order: inputs,
  // outputs then scratch.
  const uint64_t endOfInputArguments = getNumOperands();
  uint64_t idx = 0;
  SmallVector<Value> stubCallOperands;
  builder.setInsertionPointToStart(&regionBody);
  for (auto &&[stubArgument, regionArgument] :
       llvm::zip_first(kernelStub.getBody().front().getArguments(),
                       regionBody.getArguments().drop_front())) {
    if (auto asTensor = MO::dynCastFromTensorLike(stubArgument.getType())) {
      auto dynamicDimensionsOr =
          getTensorDynamicDimensions(asTensor, /*isInsideStub=*/false);
      if (dynamicDimensionsOr.isError())
        return emitError(dynamicDimensionsOr.getError());

      /// TODO: Support fusion enums
      MO::IOKind io =
          (idx < endOfInputArguments) ? MO::IOKind::Input : MO::IOKind::Output;
      bool mut = idx >= endOfInputArguments ||
                 isa<MO::BufferType>(stubArgument.getType());
      // Exclusive is True unless it's used by a view kernel.
      bool isExclusive =
          llvm::none_of(stubArgument.getUsers(), [](Operation *use) {
            return isa<MOGG::CallViewOp>(use);
          });
      auto ioAttr = MO::IOKindAttr::get(ctx, io);
      stubArgument.setType(asTensor.cloneWith(ioAttr));
      auto tensorInitOp = builder.create<MOGG::TensorInitOp>(
          loc, stubArgument.getType(), regionArgument, *dynamicDimensionsOr,
          BoolAttr::get(ctx, mut), ioAttr, BoolAttr::get(ctx, isExclusive));
      stubCallOperands.emplace_back(tensorInitOp);
    } else {
      emitError("Type not yet support in the lowering");
      return failure();
    }
    ++idx;
  }

  // Add extra arguments for the dynamic dimensions of fused tensors
  kernelStub.walk([&](MOGG::BindLambdaOp bindOp) {
    // We only need to get the dynamic dimensions when materializing an
    // intermediate tensor for the fused kernel operand. This happens only for
    // Input and AffineView fusion.
    if (bindOp.getBindKind() != BindLambdaKind::InputLambda &&
        !bindOp.hasAffineViewOperation())
      return;

    Value fusedOutput = bindOp.getOutput();
    auto asTensor = MO::castFromTensorLike(fusedOutput.getType());
    for (auto dim : asTensor.getDims()) {
      if (auto outsideSSAValue = parameterToOutsideSSAValue.lookup(dim)) {
        // Check if the argument isn't already defined.
        if (outsideValueToStubOperandIndex.contains(outsideSSAValue))
          continue;

        // Pass the dynamic dimension as an argument of the stub.
        Value regionArg =
            regionBody.getArgument(outsideValueToOperandIndex[outsideSSAValue]);
        kernelStub.getBody().addArgument(regionArg.getType(), bindOp->getLoc());
        outsideValueToStubOperandIndex[outsideSSAValue] =
            stubCallOperands.size();
        stubCallOperands.emplace_back(regionArg);
      }
    }
  });

  assert(stubCallOperands.size() ==
         (kernelStub.getBody().front().getNumArguments()));

  // Create the call to the outlined kernel stub and use whatever return value
  // (if any) in the output op of the new kernel.
  auto callStubOp = builder.create<MOGG::CallStubOp>(
      loc, resultTypes.empty() ? TypeRange{} : resultTypes.front(),
      SymbolRefAttr::get(stubName), stubCallOperands);
  auto readyChain = builder.create<MOGG::AsyncReadyOp>(getLoc());
  auto outputs = llvm::map_to_vector(callStubOp.getResults(),
                                     [](Value val) { return val; });
  outputs.push_back(readyChain);
  builder.create<MOGG::OutputOp>(getLoc(), outputs);

  auto mgpBufferType = MGP::DeviceBufferType::get(device);

  // Some operations needs to be lowered to work with EmitMojo.

  auto lowerDeviceContextPlaceholder =
      [&](MOGG::DeviceContextPlaceholderOp moggContext) {
        Value moggContextArg = kernelStub.getBody().front().addArgument(
            moggContext.getType(), moggContext.getLoc());
        moggContext.replaceAllUsesWith(moggContextArg);

        Value mgpContext =
            loweringState.getExecuteDevice(moggContext.getDevice());
        newKernel->insertOperands(newKernel.getNumOperands(), {mgpContext});
        Value mgpContextArg =
            regionBody.addArgument(mgpContext.getType(), mgpContext.getLoc());
        // The MGP context to MOGG context conversion happens at the stub (no
        // need for a conversion op since it's 1:1).
        callStubOp->insertOperands(callStubOp.getNumOperands(), mgpContextArg);

        // Update the kernel InOut semantics if needed.
        if (newKernel.getInOutSignature().size() != 0) {
          SmallVector<InOutSignatureAttr::InOutSemantics> newSemantics(
              1, InOutSignatureAttr::InOutSemantics::kNone);
          InOutSignatureAttr oldSemantics = newKernel.getInOutSignature();
          newKernel.setInOutSignatureAttr(oldSemantics.append(newSemantics));
        }

        moggContext.erase();
      };

  auto lowerBindLambdaOp = [&](MOGG::BindLambdaOp bindOp) -> LogicalResult {
    // Only replace I/O lambdas.
    if (bindOp.getBindKind() != BindLambdaKind::InputLambda &&
        bindOp.getBindKind() != BindLambdaKind::OutputLambda)
      return success();

    Value tensorVal = bindOp.getFusibleTensors().front();
    Value lambdaVal = bindOp.getOutput();

    // We always pass an empty tensor to the kernels for fused input tensors or
    // output lambdas with a cast.
    bool lambdaHasCast =
        MO::castFromTensorLike(tensorVal.getType()).getDtype() !=
        MO::castFromTensorLike(lambdaVal.getType()).getDtype();
    bool substituteWithEmptyTensor =
        bindOp.getBindKind() == BindLambdaKind::InputLambda || lambdaHasCast;

    builder.setInsertionPointAfter(bindOp);

    // Replace the output of the lambda with an empty tensor.
    if (substituteWithEmptyTensor) {
      Value emptyBuffer =
          builder.create<MOGG::EmptyBufferOp>(bindOp->getLoc(), mgpBufferType);
      auto dynamicDimensionsOr = getTensorDynamicDimensions(
          MO::castFromTensorLike(lambdaVal.getType()),
          /*isInsideStub=*/true);
      if (dynamicDimensionsOr.isError())
        return emitError(dynamicDimensionsOr.getError());
      tensorVal = builder.create<MOGG::TensorInitOp>(
          loc, lambdaVal.getType(), emptyBuffer, *dynamicDimensionsOr,
          /*mut=*/BoolAttr::get(ctx, false),
          /*io=*/MO::IOKindAttr::get(ctx, MO::IOKind::Input),
          /*exclusive=*/BoolAttr::get(ctx, true));
    }

    else if (bindOp.hasAffineViewOperation()) {
      // The tensor needs to be reshaped to match the expected kernel operand
      // shape.
      auto dynamicDimensionsOr = getTensorDynamicDimensions(
          MO::castFromTensorLike(lambdaVal.getType()),
          /*isInsideStub=*/true);
      if (dynamicDimensionsOr.isError())
        return emitError(dynamicDimensionsOr.getError());
      tensorVal = builder.create<MOGG::TensorReshapeOp>(
          loc, lambdaVal.getType(), tensorVal, *dynamicDimensionsOr);
    }

    // Build the Fused tensor object.
    auto fusedTensorOp = builder.create<MOGG::TensorFusedOp>(
        bindOp->getLoc(), lambdaVal.getType(), lambdaVal, tensorVal);

    // And finally replace it for the users of the lambda.
    lambdaVal.replaceAllUsesExcept(fusedTensorOp.getResult(), fusedTensorOp);
    return success();
  };

  auto lowerConstantOp = [&](MO::ConstantOp constantOp) {
    // Exclusive is True unless it's used by a view kernel.
    bool isExclusive =
        llvm::none_of(constantOp.getResult().getUsers(), [](Operation *use) {
          return isa<MOGG::CallViewOp>(use);
        });

    builder.setInsertionPoint(constantOp);
    Value bufferValue =
        builder
            .create<MOGG::InlineConstantBufferOp>(
                constantOp->getLoc(), mgpBufferType, constantOp.getValue())
            .getResult();
    SmallVector<Value> dynamicDimensions;
    Value constantValue = builder.create<MOGG::TensorInitOp>(
        loc, constantOp.getResult().getType(), bufferValue,
        /*dynamicDimensions=*/ArrayRef<Value>{},
        /*mut=*/BoolAttr::get(ctx, false),
        /*io=*/MO::IOKindAttr::get(ctx, MO::IOKind::Input),
        BoolAttr::get(ctx, isExclusive));
    constantOp.getResult().replaceAllUsesWith(constantValue);
    constantOp->erase();
  };

  auto walkResult = kernelStub.walk([&](mlir::Operation *op) {
    if (auto moggContext = dyn_cast<MOGG::DeviceContextPlaceholderOp>(op))
      lowerDeviceContextPlaceholder(moggContext);
    else if (auto bindOp = dyn_cast<MOGG::BindLambdaOp>(op)) {
      if (failed(lowerBindLambdaOp(bindOp)))
        return WalkResult::interrupt();
    } else if (auto constantOp = dyn_cast<MO::ConstantOp>(op))
      lowerConstantOp(constantOp);

    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  // The following is to inform the lowering state of the values we have
  // created.
  builder.setInsertionPointAfter(newKernel);
  if (!resultTypes.empty()) {
    Value oldResult = getResult(resultTypeIndex);
    Value newResult = newKernel.getResult(1);
    loweringState.emplaceExecuteLoweredValue(oldResult, newResult);
  }

  opLoweringState.createOutputs(newKernel->getResult(0), loweredOutputsStorage);
  if (auto chain = maybeGetOutChain()) {
    loweringState.emplaceExecuteLoweredValue(std::move(*chain),
                                             newKernel.getResult(0));
  }

  // Remove the placeholder ops that we replaced with DPS operands.
  outputPlaceholdersOp.erase();
  if (scratchPlaceholdersOp)
    scratchPlaceholdersOp.erase();

  mlir::BitVector unpackItAll(regionBody.getNumArguments(), true);
  mlir::BitVector packItAll(newKernel.getNumResults(), true);
  loweringState.lowerMOGGABI(newKernel, unpackItAll, packItAll,
                             /*callsThrowingMojoStub=*/isThrows);

  return success();
}

void ExpKernelOp::insertInputsAt(unsigned idx, ValueRange vals) {
  SmallVector<Value> ops = getInputs();
  ops.insert(ops.begin() + idx, vals.begin(), vals.end());
  getInputsMutable().assign(ops);

  for (auto v : llvm::reverse(vals))
    getExecBody().insertArgument(idx, v.getType(), v.getLoc());
}

void ExpKernelOp::eraseInputsAt(unsigned idx, size_t len) {
  getInputsMutable().erase(idx, len);

  for (unsigned i = 0; i < len; i++)
    getExecBody().eraseArgument(idx);
}

void ExpKernelOp::setChain(Value chain) {
  if (chain)
    getChainMutable().assign(chain);
}
void ExpKernelOp::eraseChain() { getChainMutable().erase(0); }

std::optional<MO::ChainValue> ExpKernelOp::maybeGetInChain() {
  auto chainOrNull = getChain();
  return chainOrNull ? std::optional(chainOrNull) : std::nullopt;
}

std::optional<MO::ChainValue> ExpKernelOp::maybeGetOutChain() {
  if (getNumResults() == 0)
    return std::nullopt;

  auto finalVal = getResults().back();

  if (auto chain = dyn_cast<MO::ChainValue>(finalVal))
    return chain;
  else
    return std::nullopt;
}

KernelTerminator ExpKernelOp::getTerminator() {
  return cast<KernelTerminator>(getExecBody().front().getTerminator());
}

BindLambdaOp ExpKernelOp::getBoundInputLambda(uint64_t operandIdx) {
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

BindLambdaOp ExpKernelOp::getBoundOutputLambda(uint64_t resultIdx) {
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

BindLambdaOp ExpKernelOp::getElemwiseBodyLambda() {
  Operation *t = getTerminator();
  auto foreachOp = mlir::dyn_cast<CallForeachOp>(t);
  return foreachOp ? foreachOp.getInput().getDefiningOp<BindLambdaOp>()
                   : nullptr;
}

bool ExpKernelOp::isOperandFusible(uint64_t operandIdx, bool lookThroughView) {
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

void ExpKernelOp::getEffects(
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

LogicalResult ExpKernelOp::verify() {
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
    if (regOutput != dpsResult.back().getType())
      return emitError("Expect to return the same result types as terminator");

    dpsResult = dpsResult.drop_back();
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

ParseResult ExpKernelOp::parse(OpAsmParser &parser, OperationState &state) {
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

void ExpKernelOp::print(OpAsmPrinter &p) {
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

void ExpKernelOp::build(OpBuilder &builder, OperationState &state,
                        TypeRange results, ValueRange operands,
                        StringAttr summary, BoolAttr isAsync,
                        DeviceRefAttr device,
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

ScratchPlaceholdersOp ExpKernelOp::getScratchPlaceholder() {
  auto scratch = this->getOps<MOGG::ScratchPlaceholdersOp>();

  if (scratch.empty())
    return nullptr;

  assert(llvm::hasSingleElement(scratch));

  return *scratch.begin();
}

OutputOp ExpKernelOp::getOutputOp() {
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
  if (getInput().getType().getElementType() != getResult().getType().getDType())
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

    // new path will insert a to_mogg to link the ssa, simply strip it.
    if (auto simpleToMogg = oldOut.getDefiningOp<ToMOGGOp>())
      oldOut = simpleToMogg->getOperand(0);

    // TODO(GEX-1183): Another hack from inconsistency in opaque lowering.
    // Sometimes we return the KGEN type directly, sometimes not depending
    // on whether we are in the old and new pipelines.
    if (isa<KGEN::StructType>(oldOut.getType())) {
      while (!isa<OutputPlaceholdersOp>(oldOut.getDefiningOp())) {
        Operation *definingOp = oldOut.getDefiningOp();

        // Only expect ops like "convert_to_ndbuffer" or "mogg.to_kgen"
        // which all have a single operand and just transform types.
        ASSERT_STREAM(definingOp->getNumOperands() == 1,
                      << "Expect one operand." << *definingOp);
        oldOut = definingOp->getOperand(0);
      }
    }

    ASSERT_STREAM(isa<OutputPlaceholdersOp>(oldOut.getDefiningOp()),
                  << "Expect all pre-allocated outputs to be in "
                     "`OutputPlaceholdersOp`. found:\n"
                  << *oldOut.getDefiningOp());

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
                auto symbolicDim = dyn_cast<KGEN::ParamDeclRefAttr>(dim);
                if (!symbolicDim)
                  return true;
                return symbolicDimSet.contains(symbolicDim.getName());
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

LogicalResult KernelOp::canonicalize(KernelOp op, PatternRewriter &rewriter) {
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

  // For MOGG::ExpKernelOp, if the kernel takes in a chain, there will be 1
  // fewer argument than operands.
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

  if (!getExecutionDevice())
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

LogicalResult AsyncUnpackOp::emitMojo(EmitMojoState &state) {
  MojoValue unpackedValue;
  if (auto tensorSpec = dyn_cast<MGP::TensorSpecType>(getInput().getType())) {
    MOGG::MojoValue asyncUnpackFunc{
        HardcodedMojoSymbols::func_unpack_tensor_spec};
    unpackedValue = asyncUnpackFunc({int(tensorSpec.getRank())},
                                    {state.getMojoValueFor(getInput())});
  } else {
    auto moggAsyncUnpackFunc = state.getOriginalMojoNameFor(
        state.analysis.getRegisteredLITMojoKernelFor(this->getOperationName()));
    unpackedValue = moggAsyncUnpackFunc({state.mlirTypeToMojoType(getType())},
                                        state.getMojoValueFor(getInput()));
  }
  state.setMojoValueFor(getResult(), state.builder.var(unpackedValue));
  return success();
}

//===----------------------------------------------------------------------===//
// AsyncPackOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncPackOp::emitMojo(EmitMojoState &state) {
  ASSERT_STREAM(
      llvm::any_of(getOutput().getUsers(),
                   [](auto user) { return isa<MOGG::OutputOp>(user); }) == 1,
      << "Expected at least one of the users to be an output op.");
  uint64_t resultIndex;
  for (auto &&use : getOutput().getUses()) {
    if (isa<MOGG::OutputOp>(use.getOwner())) {
      resultIndex = use.getOperandNumber();
      break;
    }
  }
  MojoValue asyncPackHelperStruct =
      HardcodedMojoSymbols::struct_MoggAsyncPackHelper;
  MojoValue moggAsyncPackFunc = state.getOriginalMojoNameFor(
      state.analysis.getRegisteredLITMojoKernelFor(this->getOperationName()));
  state.builder.call(
      moggAsyncPackFunc, {},
      asyncPackHelperStruct({}, {state.getMojoValueFor(getInput()),
                                 state.mojoReturnValues[resultIndex]}));
  return success();
}

//===----------------------------------------------------------------------===//
// AsyncDelOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncDelOp::emitMojo(EmitMojoState &state) {
  auto moggAsyncDelFunc = state.getOriginalMojoNameFor(
      state.analysis.getRegisteredLITMojoKernelFor(this->getOperationName()));
  state.builder.call(moggAsyncDelFunc, {}, state.getMojoValueFor(getInput()));
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
      state.builder.alias(state.getStaticShapeAsMojoValue(tensorTy));
  auto stridesParam =
      state.builder.alias(state.getStaticStridesAsMojoValue(tensorTy));
  MojoValue alignmentParam(static_cast<uint64_t>(AsyncRT::getPreferredAlignment(
      MO::getDeviceOrDefaultAny(tensorTy).getLabel().str())));
  auto shapeVar = state.builder.var(
      state.getShapeAsIndexListMojoValue(tensorTy, getDims()));
  auto moggTensorInit = state.getOriginalMojoNameFor(
      state.analysis.getRegisteredLITMojoKernelFor(this->getOperationName()));
  state.setMojoValueFor(
      getResult(), state.builder.var(moggTensorInit(
                       {dTypeParam, rankParam, mutParam, ioParam, shapeParam,
                        stridesParam, alignmentParam, exclusiveParam},
                       {state.getMojoValueFor(getBuffer()), shapeVar})));

  return success();
}

//===----------------------------------------------------------------------===//
// CallStubOp
//===----------------------------------------------------------------------===//

LogicalResult CallStubOp::emitMojo(EmitMojoState &state) {
  state.builder.newline();
  MojoValue func(getName().getLeafReference().str());
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
  auto moggAsyncReadFunc = state.getOriginalMojoNameFor(
      state.analysis.getRegisteredLITMojoKernelFor(this->getOperationName()));
  state.builder.call(moggAsyncReadFunc, {}, state.mojoReturnValues[0]);

  return success();
}

//===----------------------------------------------------------------------===//
// AsyncPackBorrowOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncPackBorrowOp::emitMojo(EmitMojoState &state) {
  /// AsyncPackBorrowOp should feed into a OutputOp at all times. If it does
  /// not, we assume it is about to be DCE-ed so a canonicalizer pass should be
  /// run before emit mojo.
  ASSERT_STREAM(
      llvm::any_of(getOutput().getUsers(),
                   [](auto user) { return isa<MOGG::OutputOp>(user); }) == 1,
      << "Expected at least one of the users to be an output op. Missing a "
         "canonicalizer pass before emit Mojo?");
  uint64_t resultIndex;
  for (auto &&use : getOutput().getUses()) {
    if (isa<MOGG::OutputOp>(use.getOwner())) {
      resultIndex = use.getOperandNumber();
      break;
    }
  }
  auto moggAsyncPackBorrowFunc = state.getOriginalMojoNameFor(
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName()));
  state.builder.call(moggAsyncPackBorrowFunc, {},
                     {/* borrower */ state.mojoReturnValues[resultIndex],
                      /* borrowee */ state.getMojoValueFor(getInput())});

  return success();
}

LogicalResult AsyncPackBorrowOp::canonicalize(AsyncPackBorrowOp borrowOp,
                                              mlir::PatternRewriter &rewriter) {
  rewriter.replaceOpUsesWithIf(
      borrowOp, borrowOp.getInput(),
      [](OpOperand &use) { return !isa<MOGG::OutputOp>(use.getOwner()); });

  return failure();
}

//===----------------------------------------------------------------------===//
// TensorGetShapeOp
//===----------------------------------------------------------------------===//

LogicalResult TensorGetShapeOp::emitMojo(EmitMojoState &state) {
  /// TODO: Look it up in the MojoLibraryAnalysis instead.
  auto stubOp = getOperation()->getParentOfType<KernelStubOp>();
  auto tensorTy = MO::castFromTensorLike(getTensor());
  for (auto arg : stubOp.getBody().getArguments()) {
    auto argTy = MO::dynCastFromTensorLike(arg.getType());
    if (argTy && argTy.getShapeAttr() == tensorTy.getShapeAttr()) {
      auto tensorMojoValue = state.getMojoValueFor(arg);
      state.setMojoValueFor(
          getResult(),
          tensorMojoValue.get(
              HardcodedMojoSymbols::func_ManagedTensorslice_shape)());

      return success();
    }
  }

  emitError("Could not find a tensor to get the shape");

  return failure();
}

//===----------------------------------------------------------------------===//
// CallViewOp
//===----------------------------------------------------------------------===//

LogicalResult CallViewOp::emitMojo(EmitMojoState &state) {
  // Get the kernel inputs.
  SmallVector<MojoValue> inputMojoValues = state.getMojoValuesFor(getInputs());

  KGEN::LIT::FnOp function =
      state.analysis.getRegisteredLITMojoUpdateViewFunctionFor(getKernelName());

  // Infer the kernel parameters.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    constexpr StringLiteral staticOutShapeParamName = "output_static_shape";
    if (paramName == staticOutShapeParamName) {
      return state.builder.alias(
          state.getStaticShapeAsMojoValue(getResult().getType()));
    }
    return state.inferDefaultKernelParamValue(*this, paramName);
  };
  ErrorOr<llvm::SmallVector<MojoValue>> maybeParams =
      EmitMojoState::inferKernelParameters(function, inferParam);
  if (maybeParams.isError())
    return emitError("Failed to infer all the kernel parameters");

  auto mojoSymbolName = state.getOriginalMojoNameFor(function);

  state.setMojoValueFor(getResult(), state.builder.var(mojoSymbolName(
                                         *maybeParams, inputMojoValues)));
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
  // invoke on the Mojo side, but I wasn't able to figure out how to paramterize
  // the raw mlir we're generating with the array size.
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
  MOGG::MojoValue altPtr = state.builder.var(MOGG::MojoValue("UnsafePointer")(
      {}, {MOGG::MojoValue("value").equal(arrPtr)}));

  // Bitcast to NoneType (expected dtype for the buffers).
  state.setMojoValueFor(getResult(), state.builder.var(altPtr.get("bitcast")(
                                         {MOGG::MojoValue("NoneType")}, {})));
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
      state.builder.var(MojoValue(
          MOGG::HardcodedMojoSymbols::struct_UnsafePointer)({"NoneType"}, {})));
  return success();
}

//===----------------------------------------------------------------------===//
// TensorFusedOp
//===----------------------------------------------------------------------===//

LogicalResult TensorFusedOp::emitMojo(EmitMojoState &state) {

  BindLambdaOp lambdaOp = getBindLambdaOp();
  MojoValue lambdaParam = state.getMojoValueFor(getLambda());
  MojoValue tensorValue = state.getMojoValueFor(getTensor());

  // Check if the called kernel takes a compute lambda.
  auto kernelTakesAComputeLambda = [&]() -> ErrorOr<bool> {
    Value fusedTensor = getResult();
    // Find the kernel call op.
    if (!fusedTensor.hasOneUse())
      return Error("MaterializeFusedTensor should have a unique user");
    auto executeOp = dyn_cast<CallExecuteOp>(*fusedTensor.getUsers().begin());
    if (!executeOp)
      return Error("TensorFusedOp user must be a kernel call "
                   "(CallExecuteOp)");

    // Find the corresponding output index.
    std::optional<unsigned> outputIdx = std::nullopt;
    for (auto [idx, outputVal] : llvm::enumerate(executeOp.getOutputs())) {
      if (outputVal == fusedTensor) {
        outputIdx = idx;
        break;
      }
    }

    if (!outputIdx.has_value())
      return Error("TensorFusedOp result must be a kernel call output");

    // Check if the output operand corresponds to an output lambda.
    auto function = state.analysis.getRegisteredLITMojoExecuteFunctionFor(
        executeOp.getKernelName());
    KGEN::MOGGPreElab::MojoKernelFunctionAdaptor<KGEN::LIT::FnOp>
        functionAdaptor(function);
    return functionAdaptor.outputArguments[*outputIdx].isTensorWithIOSpec(
        KGEN::MOGGPreElab::IOSpec::FusedComputeOutputTensor);
  };

  if (lambdaOp.getBindKind() == BindLambdaKind::InputLambda) {
    // Build the FusedTensor with the new spec.
    // TODO(MOCO-2595): Use helper function instead of relying on rebind.
    MojoValue newSpec = MojoValue(
        HardcodedMojoSymbols::
            func_rebuild_static_tensor_specs_with_input_lambda)(
        {}, ArrayRef<MojoValue>({tensorValue.get("static_spec"), lambdaParam}));
    MojoValue fusedTensor = tensorValue.rebind(MojoValue(
        HardcodedMojoSymbols::alias_FusedInputTensor)[MojoValue("static_spec")
                                                          .equal(newSpec)]);

    state.setMojoValueFor(getResult(), state.builder.var(fusedTensor));
  } else {

    // Check if we need a compute or an output lambda.
    auto requiresComputeLambdaOr = kernelTakesAComputeLambda();
    if (requiresComputeLambdaOr.isError())
      return emitError(requiresComputeLambdaOr.getError());
    auto spec = *requiresComputeLambdaOr
                    ? HardcodedMojoSymbols::alias_FusedComputeOutputTensor
                    : HardcodedMojoSymbols::alias_FusedOutputTensor;

    // Build the FusedTensor with the new spec.
    // TODO(MOCO-2595): Use helper function instead of relying on rebind.
    MojoValue newSpec = MojoValue(
        HardcodedMojoSymbols::
            func_rebuild_static_tensor_specs_with_output_lambda)(
        {}, ArrayRef<MojoValue>({tensorValue.get("static_spec"), lambdaParam}));
    MojoValue fusedTensor = tensorValue.rebind(
        MojoValue(spec)[MojoValue("static_spec").equal(newSpec)]);

    state.setMojoValueFor(getResult(), state.builder.var(fusedTensor));
  }

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

  // Create DynamicTensor variables for each input
  SmallVector<MojoValue> dynamicTensorVars;
  for (const auto &inputVal : inputValues) {
    // Create a DynamicTensor from each rebound input (which is already rebuilt)
    // var aN = DynamicTensor[dtype = varN.dtype, rank = varN.rank](
    //     varN._ptr, varN.shape()
    // )
    MojoValue dynamicTensorConstructor = MojoValue(
        "DynamicTensor")[{MojoValue("dtype").equal(inputVal.get("dtype")),
                          MojoValue("rank").equal(inputVal.get("rank"))}];
    MojoValue dynamicTensorValue = dynamicTensorConstructor(
        {}, {inputVal.get("_ptr"), inputVal.get("shape")()});
    dynamicTensorVars.push_back(state.builder.var(dynamicTensorValue));
  }

  // Create StaticTuple with the DynamicTensor variables
  // var inputTup = StaticTuple[size=N](a1, a2, ..., aN)
  MojoValue inputTupleConstructor = MojoValue("StaticTuple")[{
      MojoValue("size").equal(std::to_string(dynamicTensorVars.size()))}];
  MojoValue inputTupleValue = inputTupleConstructor({}, dynamicTensorVars);
  MojoValue inputTupVar = state.builder.var(inputTupleValue);

  // Create the static specs tuple
  // alias inputSpecs = StaticTuple[size=N](
  //     var1.static_spec, var2.static_spec, ..., varN.static_spec
  // )
  SmallVector<MojoValue> staticSpecs;
  for (const auto &reboundInput : inputValues) {
    staticSpecs.push_back(reboundInput.get("static_spec"));
  }
  MojoValue staticSpecsConstructor = MojoValue("StaticTuple")[{
      MojoValue("size").equal(std::to_string(staticSpecs.size()))}];
  MojoValue staticSpecsValue = staticSpecsConstructor({}, staticSpecs);
  MojoValue inputSpecsAlias = state.builder.alias(staticSpecsValue);

  // Create the VariadicTensors
  // var varInput = VariadicTensors[
  //     io_spec = var1.io_spec, static_specs=inputSpecs
  // ](inputTup)
  // Note: Using the first rebound input's io_spec as they should all be the
  // same
  MojoValue variadicTensorsConstructor = MojoValue("VariadicTensors")[{
      MojoValue("io_spec").equal(inputValues[0].get("io_spec")),
      MojoValue("static_specs").equal(inputSpecsAlias)}];
  MojoValue variadicTensorsValue =
      variadicTensorsConstructor({}, {inputTupVar});

  state.setMojoValueFor(getResult(), state.builder.var(variadicTensorsValue));
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
  state.builder.call(
      moggAsyncErrorFunc, {},
      {state.mojoReturnValues[0], state.getMojoValueFor(getError())});

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
  MojoValue shapeValue =
      state.getShapeAsIndexListMojoValue(toType, getDynamicDims());
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
