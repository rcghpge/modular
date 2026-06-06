//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect operations.
//
//===----------------------------------------------------------------------===//
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/GraphCompiler/Interfaces/LoweringState.h"
#include "GraphCompiler/GraphCompiler/Interfaces/MOGGKernelOpLoweringState.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "GraphCompiler/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
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
#include "llvm/Support/FormatVariadic.h"
#include <cstdlib>
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

/// Build a Mojo expression that loads from decomposed tensor components on the
/// fusion struct, using the struct's StaticTensorSpec parameter.
static std::string buildStructLoadExpr(StringRef captureIdx,
                                       StringRef indicesExpr) {
  return llvm::formatv("{2}[static_spec = Self.spec_{0}, "
                       "simd_width = simd_width, "
                       "element_alignment = element_alignment]"
                       "(self.ptr_{0}, self.shape_{0}, self.strides_{0}, {1})",
                       captureIdx, indicesExpr, kFnMsSimdLoadFromTensorPointer)
      .str();
}

/// Execute-scope locals backing one tensor capture in the register_passable
/// elementwise path. `ptrVar`/`shapeVar`/`stridesVar` go in the def's capture
/// clause; `staticSpec` is comptime and referenced directly (not captured).
struct TensorCaptureVars {
  MojoValue ptrVar;
  MojoValue shapeVar;
  MojoValue stridesVar;
  MojoValue staticSpec;
};

/// Load expression for a register_passable elementwise body's local-var-backed
/// tensor captures. Counterpart of `buildStructLoadExpr`.
static std::string buildDefLoadExpr(const TensorCaptureVars &v,
                                    StringRef simdWidthName,
                                    StringRef elementAlignmentName,
                                    StringRef indicesExpr) {
  return llvm::formatv("{0}[static_spec = {1}, simd_width = {2}, "
                       "element_alignment = {3}]"
                       "({4}, {5}, {6}, {7})",
                       kFnMsSimdLoadFromTensorPointer, v.staticSpec.text,
                       simdWidthName, elementAlignmentName, v.ptrVar.text,
                       v.shapeVar.text, v.stridesVar.text, indicesExpr)
      .str();
}

namespace {
/// The two pieces that differ between the fusion-struct and register_passable
/// paths: how a tensor capture is loaded, and the SIMD-width parameter name.
/// `emitElementwiseLambdaBodyOps` and its shells provide the shared dispatch.
struct ElementwiseBodyRenderer {
  virtual ~ElementwiseBodyRenderer() = default;
  /// Load expression for the capture at `captureIdx`, indexed by `indicesExpr`
  /// (already wrapped in an IndexList rebind).
  virtual std::string buildLoadExpr(unsigned captureIdx,
                                    StringRef indicesExpr) const = 0;
  /// SIMD-width parameter name for this body, spliced into
  /// `rebind[SIMD[<dtype>, <name>]](…)` wrappers.
  virtual StringRef simdWidthParamName() const = 0;
};

/// Legacy fusion-struct body: loads through `self.{ptr,shape,strides}_<i>` on
/// the struct's `simd_width` method parameter.
struct FusionStructRenderer final : ElementwiseBodyRenderer {
  std::string buildLoadExpr(unsigned captureIdx,
                            StringRef indicesExpr) const override {
    return buildStructLoadExpr(std::to_string(captureIdx), indicesExpr);
  }
  StringRef simdWidthParamName() const override { return "simd_width"; }
};

/// Register_passable def body: loads through the execute-scope locals in the
/// def's `{var …}` capture clause and its own SIMD-width/element-alignment
/// parameters. Holds captured names by value to avoid dangling into
/// stack-locals.
struct RegisterPassableRenderer final : ElementwiseBodyRenderer {
  RegisterPassableRenderer(SmallVector<TensorCaptureVars> tensorVars,
                           std::string simdWidthName,
                           std::string elementAlignmentName)
      : tensorVars(std::move(tensorVars)),
        simdWidthName(std::move(simdWidthName)),
        elementAlignmentName(std::move(elementAlignmentName)) {}
  std::string buildLoadExpr(unsigned captureIdx,
                            StringRef indicesExpr) const override {
    return buildDefLoadExpr(tensorVars[captureIdx], simdWidthName,
                            elementAlignmentName, indicesExpr);
  }
  StringRef simdWidthParamName() const override { return simdWidthName; }

private:
  SmallVector<TensorCaptureVars> tensorVars;
  std::string simdWidthName;
  std::string elementAlignmentName;
};
} // namespace

/// Add a compile-time StaticTensorSpec parameter for a tensor capture.
static void addTensorCaptureParam(
    SmallVectorImpl<std::pair<MojoValue, MojoValue>> &parameters,
    unsigned captureIdx, StringRef dtypeText, int64_t rank) {
  auto ci = std::to_string(captureIdx);
  parameters.push_back(
      {MojoValue("spec_" + ci),
       MojoValue(llvm::formatv("{0}[{1}, {2}, ...]", kStructStaticTensorSpec,
                               dtypeText, rank)
                     .str())});
}

/// Add ptr, shape, and strides fields for a tensor capture.
static void
addTensorCaptureField(SmallVectorImpl<std::pair<MojoValue, MojoValue>> &fields,
                      unsigned captureIdx, StringRef dtypeText, int64_t rank) {
  auto ci = std::to_string(captureIdx);
  fields.push_back({MojoValue("ptr_" + ci),
                    MojoValue(kStructUnsafePointer)[{
                        MojoValue(kStructScalar)[{MojoValue(dtypeText)}],
                        MojoValue(kComptimeMutAnyOrigin)}]});
  fields.push_back(
      {MojoValue("shape_" + ci), MojoValue(kStructIndexList)[rank]});
  fields.push_back(
      {MojoValue("strides_" + ci), MojoValue(kStructIndexList)[rank]});
}

/// Resolve a Value from the body's value map, applying
/// `rebind[SIMD[dtype, <renderer-simd-width>]]` wrapping for tensor-typed
/// values. Scalar captures are returned without wrapping.
static MojoValue
resolveStructOperand(Value operand, const DenseMap<Value, MojoValue> &valueMap,
                     const DenseSet<Value> &scalarValues, EmitMojoState &state,
                     const ElementwiseBodyRenderer &renderer) {
  auto it = valueMap.find(operand);
  ASSERT_STREAM(it != valueMap.end(), << "missing value in body map");
  if (scalarValues.contains(operand))
    return it->second;
  auto inputTy = M::MO::fromMLIRType(operand.getType());
  if (!inputTy.isError()) {
    return MojoValue(llvm::formatv("rebind[{0}[{1}, {3}]]({2})", kStructSIMD,
                                   state.mlirDTypeToMojoType(*inputTy).text,
                                   it->second.text,
                                   renderer.simdWidthParamName()));
  }
  return it->second;
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

// Returns the fully-qualified Mojo source name for a PrimitiveDeclOp.
// sym_name is set to the fully-qualified Mojo source name by ImportPackages,
// so it doubles as the identifier to emit in Mojo source.
static llvm::StringRef primitiveMojoName(PrimitiveDeclOp decl) {
  return decl.getSymNameAttr().strref();
}

MojoValue callFunctionNestedInStruct(const KernelDeclFunctionAdaptor &fnAdaptor,
                                     EmitMojoState::InferredParameters params,
                                     ArrayRef<MojoValue> operands,
                                     EmitMojoState &) {
  auto [prefixSymbolName, fnName] =
      EmitMojoState::splitFunctionName(fnAdaptor.fullyResolvedName.strref());
  MojoValue structType = prefixSymbolName[params.structParameters];
  MojoValue fullFn = structType.get(fnName);
  return fullFn(params.functionParameters, operands);
}

/// Common verifySymbolUses logic for ops implementing KernelCallInterface.
/// Looks up the KernelDeclOp and delegates to `check` for the op-specific
/// function presence check.
static LogicalResult verifySymbolUsesForKernelDeclCall(
    KernelCallInterface op, SymbolTableCollection &symbolTable,
    llvm::function_ref<LogicalResult(KernelDeclOp)> check) {
  auto symAttr = op.getKernelSymNameAttr();
  auto decl = symbolTable.lookupNearestSymbolFrom<KernelDeclOp>(op, symAttr);
  if (!decl)
    return op->emitOpError("'")
           << symAttr.getValue() << "' does not refer to a kernel.decl op";
  return check(decl);
}

static LogicalResult
verifySymbolUsesForPrimitiveDeclCall(KernelCallInterface op,
                                     SymbolTableCollection &symbolTable) {
  auto symAttr = op.getKernelSymNameAttr();
  auto decl = symbolTable.lookupNearestSymbolFrom<PrimitiveDeclOp>(op, symAttr);
  if (!decl)
    return op->emitOpError("'")
           << symAttr.getValue() << "' does not refer to a primitive.decl op";
  return success();
}

LogicalResult
CallAffineOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifySymbolUsesForPrimitiveDeclCall(*this, symbolTable);
}

LogicalResult CallAffineOp::emitMojo(EmitMojoState &state) {
  auto primDecl = state.registry.lookupPrimitiveBySymName(getKernelSymName());
  MojoValue symbolName(getKernelSymName());

  SmallVector<MojoValue> mojoOperands = state.getMojoValuesFor(getOperands());

  // Infer the kernel parameters.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    return state.inferDefaultKernelParamValue(*this, paramName,
                                              getParamsAttr());
  };
  auto maybeParams = EmitMojoState::inferKernelParameters(primDecl, inferParam);
  if (maybeParams.isError())
    return emitError() << "Failed to infer all the kernel parameters: "
                       << maybeParams.takeError();

  // Call the function and map its output.
  state.setMojoValueFor(
      getResult(), state.builder.var(symbolName(maybeParams->functionParameters,
                                                mojoOperands)));
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
           // MOGG::CallAffineOp is only used inside a lambda to represent an
           // affine function call
           if (isa<MOGG::CallAffineOp>(op))
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

//===----------------------------------------------------------------------===//
// Fusion struct emission helpers for BindLambdaOp
//===----------------------------------------------------------------------===//

/// Collect struct compile-time parameters and fields from the lambda body.
/// Each tensor capture becomes a StaticTensorSpec parameter and a
/// ManagedTensorSlice field. Non-tensor captures (scalars) are appended
/// after all tensor fields.
static void collectFusionCaptures(
    EmitMojoState &state, Block *lambdaBody, OperandRange fusibleTensors,
    bool isOutputLambda,
    SmallVectorImpl<std::pair<MojoValue, MojoValue>> &parameters,
    SmallVectorImpl<std::pair<MojoValue, MojoValue>> &fields,
    SmallVectorImpl<MO::TensorType> &captureTensorTypes,
    SmallVectorImpl<std::pair<Value, unsigned>> &extraTensorCaptures,
    SmallVectorImpl<std::tuple<Value, std::string, std::string>>
        &scalarCaptures,
    SmallVectorImpl<std::pair<Value, MojoValue>> &inlineConstants) {
  // Primary tensor captures from fusibleTensors.
  for (auto [i, input] : llvm::enumerate(fusibleTensors)) {
    auto tensorTy = MO::castFromTensorLike(input.getType());
    captureTensorTypes.push_back(tensorTy);
    auto dtype = state.mlirDTypeToMojoType(tensorTy.getDtype()).text;
    auto rank = tensorTy.getRank() == 0 ? 1 : tensorTy.getRank();
    addTensorCaptureParam(parameters, i, dtype, rank);
    addTensorCaptureField(fields, i, dtype, rank);
  }

  // Find values used inside the body but defined outside (external captures).
  DenseSet<Value> bodyDefinedValues;
  for (auto arg : lambdaBody->getArguments())
    bodyDefinedValues.insert(arg);
  for (auto &op : lambdaBody->getOperations())
    for (auto result : op.getResults())
      bodyDefinedValues.insert(result);

  DenseSet<Value> alreadyCaptured;
  for (auto input : fusibleTensors)
    alreadyCaptured.insert(input);

  for (auto &op : lambdaBody->getOperations()) {
    for (auto operand : op.getOperands()) {
      if (bodyDefinedValues.contains(operand) ||
          alreadyCaptured.contains(operand))
        continue;
      alreadyCaptured.insert(operand);

      if (auto tensorTy = MO::dynCastFromTensorLike(operand.getType())) {
        unsigned captureIdx = captureTensorTypes.size();
        captureTensorTypes.push_back(tensorTy);
        auto dtype = state.mlirDTypeToMojoType(tensorTy.getDtype()).text;
        auto rank = tensorTy.getRank() == 0 ? 1 : tensorTy.getRank();
        addTensorCaptureParam(parameters, captureIdx, dtype, rank);
        addTensorCaptureField(fields, captureIdx, dtype, rank);
        extraTensorCaptures.push_back({operand, captureIdx});
      } else {
        // If the scalar comes from a constant.scalar op defined outside the
        // lambda, inline its literal value directly instead of capturing it as
        // a runtime field. This lets LLVM emit magic-multiply sequences for
        // non-power-of-2 divisors (e.g. split_dim with new_shape_dim=208)
        // rather than hardware integer division (~40 cycles).
        if (auto constOp = dyn_cast_or_null<MOGG::ConstantScalarOp>(
                operand.getDefiningOp())) {
          if (auto intScalar = MO::tryGetConstantInt(constOp.getValue())) {
            inlineConstants.push_back({operand, MojoValue(*intScalar)});
            continue;
          }
        }
        auto mojoType = EmitMojoState::mlirTypeToMojoType(operand.getType());
        auto fieldName = "scalar_" + std::to_string(scalarCaptures.size());
        scalarCaptures.push_back({operand, fieldName, mojoType.text});
      }
    }
  }

  for (auto &[val, fieldName, mojoType] : scalarCaptures)
    fields.push_back({MojoValue(fieldName), MojoValue(mojoType)});
}

/// Seed `valueMap`/`scalarCaptureValues` with inline constants (mapped to
/// literal text) and runtime scalar captures (mapped via
/// `scalarCaptureNameFor`: `self.<field>` for the fusion-struct path,
/// `state.getMojoValueFor(val)` for the register_passable path).
static void seedScalarValueMapEntries(
    ArrayRef<std::pair<Value, MojoValue>> inlineConstants,
    ArrayRef<std::tuple<Value, std::string, std::string>> scalarCaptures,
    llvm::function_ref<MojoValue(Value, StringRef)> scalarCaptureNameFor,
    DenseMap<Value, MojoValue> &valueMap,
    DenseSet<Value> &scalarCaptureValues) {
  // Map inline constants to literal text so downstream ops (e.g.
  // split_dim_indices) get a compile-time constant and LLVM uses
  // magic-multiply instead of idiv.
  for (auto &[val, mojoVal] : inlineConstants) {
    valueMap[val] = mojoVal;
    scalarCaptureValues.insert(val);
  }
  for (auto &[val, fieldName, mojoType] : scalarCaptures) {
    valueMap[val] = scalarCaptureNameFor(val, fieldName);
    scalarCaptureValues.insert(val);
  }
}

/// Verify every op in a bind lambda body is one `emitElementwiseLambdaBodyOps`
/// can lower. Same supported set across all `BindLambdaKind`s and both
/// emission paths, so the check lives here.
static LogicalResult verifyBindLambdaBodyOpsForMojoEmission(BindLambdaOp bindOp,
                                                            Block *lambdaBody) {
  for (auto &op : lambdaBody->getOperations()) {
    if (!isa<MOGG::CallElementwiseOp, MOGG::OutputOp, MOGG::LoadTensorOp,
             MOGG::CallAffineOp>(&op))
      return bindOp.emitError(
                 "Bind lambda body has op not yet supported for Mojo "
                 "emission: ")
             << op.getName();
  }
  return success();
}

/// Walk an elementwise lambda body, emitting one Mojo statement per op and
/// updating `valueMap`. Shared by the fusion-struct and register_passable
/// paths; `renderer` supplies the load expression and SIMD-width name.
static void emitElementwiseLambdaBodyOps(
    EmitMojoState &state, MojoBuilder &sdb, Block *lambdaBody,
    ArrayRef<MO::TensorType> captureTensorTypes,
    ArrayRef<std::pair<Value, unsigned>> extraTensorCaptures,
    DenseMap<Value, MojoValue> &valueMap,
    const DenseSet<Value> &scalarCaptureValues,
    const ElementwiseBodyRenderer &renderer,
    KernelOutputFusionLambdaType lambdaType =
        KernelOutputFusionLambdaType::Compute) {

  for (auto &op : lambdaBody->getOperations()) {
    if (isa<MOGG::OutputOp>(&op))
      continue;

    if (auto loadOp = dyn_cast<MOGG::LoadTensorOp>(&op)) {
      bool foundCapture = false;
      unsigned captureIdx = 0;
      for (auto &[val, idx] : extraTensorCaptures) {
        if (val == loadOp.getTensor()) {
          captureIdx = idx;
          foundCapture = true;
          break;
        }
      }
      ASSERT_STREAM(foundCapture,
                    << "LoadTensorOp tensor not found in extraTensorCaptures");
      auto tensorTy = captureTensorTypes[captureIdx];
      auto loadRank = tensorTy.getRank() == 0 ? 1 : tensorTy.getRank();

      auto indicesIt = valueMap.find(loadOp.getIndices());
      std::string indicesExpr;
      if (indicesIt != valueMap.end()) {
        indicesExpr = llvm::formatv("rebind[{0}[{1}]]({2})", kStructIndexList,
                                    loadRank, indicesIt->second.text)
                          .str();
      } else {
        indicesExpr =
            llvm::formatv("rebind[{0}[{1}]](idx)", kStructIndexList, loadRank)
                .str();
      }

      if (lambdaType == KernelOutputFusionLambdaType::ComputeTile) {
        auto captureN = std::to_string(captureIdx);
        TileTensorRef val("val");

        // Local storage for the destination tile.
        auto storageVar = sdb.var(TileTensorRef::makeStorage());

        // Dest TileTensor backed by local storage with val's layout.
        auto dstVar =
            sdb.var(TileTensorRef::fromMutStorage(storageVar, val.layout()));

        // Tile-offset pointer into the captured tensor.
        MojoValue tileOffset("0");
        for (unsigned d = 0; d < loadRank; ++d) {
          MojoValue coord = MojoValue(kParameterTileCoords)[d];
          MojoValue stride =
              MojoValue(std::string("self.strides_") + captureN)[d];
          MojoValue term = coord * val.dim(d) * stride;
          tileOffset = (d == 0) ? term : tileOffset + term;
        }
        auto tileOffsetVar = sdb.var(tileOffset);

        // Src TileTensor for captured data in global memory at the tile
        // position.
        MojoValue capturePtr =
            MojoValue(std::string("self.ptr_") + captureN) + tileOffsetVar;
        auto srcVar =
            sdb.var(TileTensorRef::fromExternalPtr(capturePtr, val.layout()));

        // Copy captured tile from global memory into local dst.
        sdb.verbatim(
            MojoValue(kParameterCopier).get("copy")({}, {dstVar, srcVar}));
        valueMap[loadOp.getOutputs()] = dstVar;
      } else {
        auto loadExpr = renderer.buildLoadExpr(captureIdx, indicesExpr);
        valueMap[loadOp.getOutputs()] = sdb.var(MojoValue(loadExpr));
      }
      continue;
    }

    if (auto elemOp = dyn_cast<MOGG::CallElementwiseOp>(&op)) {
      KernelDeclOp elemKernelDecl =
          state.registry.lookupKernelBySymName(elemOp.getKernelSymName());
      KernelDeclAdaptor elemAdaptor(elemKernelDecl);

      bool useTileElementwise =
          lambdaType == KernelOutputFusionLambdaType::ComputeTile &&
          elemAdaptor.isTileElementwise();

      SmallVector<MojoValue> inputs;
      for (auto input : elemOp.getInputs()) {
        if (useTileElementwise) {
          // Tile path: pass TileTensor values directly without SIMD rebind.
          auto it = valueMap.find(input);
          ASSERT_STREAM(it != valueMap.end(),
                        << "missing tile value in body map");
          inputs.push_back(it->second);
        } else {
          inputs.push_back(resolveStructOperand(
              input, valueMap, scalarCaptureValues, state, renderer));
        }
      }
      const KernelDeclFunctionAdaptor &elemFnAdaptor =
          useTileElementwise            ? *elemAdaptor.tileElementwiseFunction
          : elemAdaptor.isElementwise() ? *elemAdaptor.elementwiseFunction
                                        : *elemAdaptor.executeFunction;
      MojoValue mojoSymbolName(elemFnAdaptor.fullyResolvedName.strref());

      Value outputVal = elemOp.getOutputs().front();
      std::optional<StringRef> maybeDtypeParamName =
          EmitMojoState::getResultParamNameFor(elemFnAdaptor, kParameterDType);
      auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
        if (maybeDtypeParamName.has_value() &&
            paramName == *maybeDtypeParamName) {
          ErrorOr<DType> resultTyOr = M::MO::fromMLIRType(outputVal.getType());
          if (resultTyOr.isError())
            return Error(
                "invalid output type can't be converted to mojo dtype");
          return state.mlirDTypeToMojoType(*resultTyOr);
        }
        return state.inferDefaultKernelParamValue(elemOp, paramName);
      };
      llvm::StringRef elemFuncLabel =
          useTileElementwise
              ? llvm::StringRef(MOGG::kMoggTileElementwiseFunctionLabel)
              : (elemAdaptor.isElementwise()
                     ? llvm::StringRef(MOGG::kMoggElementwiseFunctionLabel)
                     : llvm::StringRef(MOGG::kMoggExecuteFunctionLabel));
      auto maybeParams = EmitMojoState::inferKernelParameters(
          elemAdaptor, elemFuncLabel, inferParam);
      if (maybeParams.isError()) {
        lambdaBody->getParentOp()->emitError()
            << "Failed to infer kernel parameters in elementwise lambda "
               "body: "
            << maybeParams.takeError();
        return;
      }

      MojoValue expr = mojoSymbolName(maybeParams->functionParameters, inputs);
      valueMap[elemOp.getOutputs().front()] = sdb.var(expr);
      continue;
    }

    if (auto callOp = dyn_cast<MOGG::CallAffineOp>(&op)) {
      auto affinePrimDecl =
          state.registry.lookupPrimitiveBySymName(callOp.getKernelSymName());
      MojoValue symbolName(primitiveMojoName(affinePrimDecl));

      SmallVector<MojoValue> inputs;
      for (auto operand : callOp.getOperands()) {
        auto it = valueMap.find(operand);
        if (it != valueMap.end()) {
          inputs.push_back(resolveStructOperand(
              operand, valueMap, scalarCaptureValues, state, renderer));
        } else {
          bool foundCapture = false;
          for (auto &[val, captIdx] : extraTensorCaptures) {
            if (val == operand) {
              auto cTy = captureTensorTypes[captIdx];
              auto cDtype = state.mlirDTypeToMojoType(cTy.getDtype());
              auto cRank = cTy.getRank() == 0 ? 1 : cTy.getRank();
              auto loadExpr = renderer.buildLoadExpr(
                  captIdx, llvm::formatv("rebind[{0}[{1}]](idx)",
                                         kStructIndexList, cRank)
                               .str());
              MojoValue varName = sdb.var(MojoValue(loadExpr));
              inputs.push_back(MojoValue(llvm::formatv(
                  "rebind[{0}[{1}, {3}]]({2})", kStructSIMD, cDtype.text,
                  varName.text, renderer.simdWidthParamName())));
              foundCapture = true;
              break;
            }
          }
          if (!foundCapture)
            inputs.push_back(state.getMojoValueFor(operand));
        }
      }

      auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
        return state.inferDefaultKernelParamValue(callOp, paramName,
                                                  callOp.getParamsAttr());
      };
      auto maybeParams =
          EmitMojoState::inferKernelParameters(affinePrimDecl, inferParam);

      SmallVector<MojoValue> params;
      if (!maybeParams.isError())
        params = maybeParams->functionParameters;

      MojoValue expr = symbolName(params, inputs);
      valueMap[callOp.getResult()] = sdb.var(expr);
      continue;
    }
  }
}

/// Emit the final return value or store expression for the fusion struct
/// method.
static void
emitFusionStructReturnOrStore(EmitMojoState &state, MojoBuilder &sdb,
                              Block *lambdaBody,
                              ArrayRef<MO::TensorType> captureTensorTypes,
                              const DenseMap<Value, MojoValue> &valueMap,
                              KernelOutputFusionLambdaType lambdaType) {
  bool generateStoreForOutput =
      lambdaType == KernelOutputFusionLambdaType::Output ||
      lambdaType == KernelOutputFusionLambdaType::OutputLegacyForCompute;

  auto outputOp = cast<MOGG::OutputOp>(lambdaBody->getTerminator());
  auto resultIt = valueMap.find(outputOp.getOperand(0));
  ASSERT_STREAM(resultIt != valueMap.end(),
                << "missing output value in struct body map");
  MojoValue resultValue = resultIt->second;

  // Tile-based compute fusion: return the result TileTensor directly.
  if (lambdaType == KernelOutputFusionLambdaType::ComputeTile) {
    sdb.returnsValue(resultValue);
    return;
  }

  if (generateStoreForOutput) {
    auto tensorTy = captureTensorTypes[0];
    auto dtype = state.mlirDTypeToMojoType(tensorTy.getDtype());
    auto rank = tensorTy.getRank() == 0 ? 1 : tensorTy.getRank();

    std::string storeIndicesExpr;
    if (outputOp->getNumOperands() > 1) {
      auto indicesVal = outputOp.getOperand(1);
      auto indicesIt = valueMap.find(indicesVal);
      if (indicesIt != valueMap.end()) {
        storeIndicesExpr =
            llvm::formatv("rebind[{0}[{1}]]({2})", kStructIndexList, rank,
                          indicesIt->second.text)
                .str();
      } else {
        storeIndicesExpr =
            llvm::formatv("rebind[{0}[{1}]](idx)", kStructIndexList, rank)
                .str();
      }
    } else {
      storeIndicesExpr =
          llvm::formatv("rebind[{0}[{1}]](idx)", kStructIndexList, rank).str();
    }

    auto storeExpr =
        llvm::formatv("{0}[static_spec = Self.spec_0, "
                      "simd_width = simd_width, "
                      "element_alignment = element_alignment]"
                      "(self.ptr_0, self.shape_0, self.strides_0, "
                      "{1}, "
                      "rebind[{2}[{3}, simd_width]]({4}))",
                      kFnMsSimdStoreIntoTensorPointer, storeIndicesExpr,
                      kStructSIMD, dtype.text, resultValue.text);
    sdb.verbatim(MojoValue(storeExpr.str()));
  } else {
    sdb.returnsValue(
        MojoValue(llvm::formatv("rebind[{0}[dtype, simd_width]]({1})",
                                kStructSIMD, resultValue.text)
                      .str()));
  }
}

/// Emit the load/store/compute method body for a fusion struct.
/// Phase 1: Initialize value maps and emit loads from tensor captures.
/// Phase 2: Emit body operations (elementwise, LoadTensor, Call).
/// Phase 3: Emit the final return or store.
static void emitFusionStructMethodBody(
    EmitMojoState &state, MojoBuilder &sdb, Block *lambdaBody,
    ArrayRef<MojoValue> fusedInputs,
    ArrayRef<MO::TensorType> captureTensorTypes,
    ArrayRef<std::pair<Value, unsigned>> extraTensorCaptures,
    ArrayRef<std::tuple<Value, std::string, std::string>> scalarCaptures,
    ArrayRef<std::pair<Value, MojoValue>> inlineConstants, bool isOutputLambda,
    KernelOutputFusionLambdaType lambdaType) {

  DenseMap<Value, MojoValue> valueMap;
  DenseSet<Value> scalarCaptureValues;
  FusionStructRenderer renderer;

  // Phase 1: Initialize value maps and emit loads from tensor captures.
  seedScalarValueMapEntries(
      inlineConstants, scalarCaptures,
      [](Value, StringRef fieldName) {
        return MojoValue("self." + fieldName.str());
      },
      valueMap, scalarCaptureValues);

  // For tile compute fusion, the position parameter is tile_coords; for
  // SIMD compute/output fusion it is idx.
  bool isTile = lambdaType == KernelOutputFusionLambdaType::ComputeTile;
  valueMap[lambdaBody->getArguments().back()] =
      MojoValue(isTile ? kParameterTileCoords : kParameterIdx);

  for (unsigned i = 0; i < fusedInputs.size(); ++i) {
    Value blockArg = lambdaBody->getArgument(i);
    if (i == 0 && isOutputLambda) {
      valueMap[blockArg] = MojoValue("val");
      continue;
    }

    if (isTile) {
      // Tile captures are resolved lazily by the LoadTensorOp handler in
      // emitElementwiseLambdaBodyOps, which looks up each capture via
      // extraTensorCaptures and emits copier.copy into a local dst TileTensor.
      continue;
    }

    auto tensorTy = captureTensorTypes[i];
    auto rank = tensorTy.getRank() == 0 ? 1 : tensorTy.getRank();
    auto loadExpr = renderer.buildLoadExpr(
        i,
        llvm::formatv("rebind[{0}[{1}]](idx)", kStructIndexList, rank).str());
    valueMap[blockArg] = sdb.var(MojoValue(loadExpr));
  }

  // Phase 2: Emit body operations.
  emitElementwiseLambdaBodyOps(state, sdb, lambdaBody, captureTensorTypes,
                               extraTensorCaptures, valueMap,
                               scalarCaptureValues, renderer, lambdaType);

  // Phase 3: Emit the final return or store.
  emitFusionStructReturnOrStore(state, sdb, lambdaBody, captureTensorTypes,
                                valueMap, lambdaType);
}

/// Emit a trait-conforming fusion struct definition and register it with the
/// EmitMojoState for later use by TensorFusedOp. The struct captures tensor
/// components (ptr, shape, strides) and scalar values, and implements a
/// load/store/compute method that replays the lambda body's computation.
static LogicalResult
emitFusionStruct(BindLambdaOp bindOp, EmitMojoState &state, Block *lambdaBody,
                 OperandRange fusibleTensors, ArrayRef<MojoValue> fusedInputs,
                 bool isOutputLambda, KernelOutputFusionLambdaType lambdaType) {
  if (failed(verifyBindLambdaBodyOpsForMojoEmission(bindOp, lambdaBody)))
    return failure();

  // Determine the trait to conform to and method name.
  StringRef traitName;
  StringRef methodName;
  bool methodReturnsValue = false;
  if (bindOp.getBindKind() == BindLambdaKind::ElemwiseLambda) {
    traitName = kTraitElementwiseFusion;
    methodName = "compute";
    methodReturnsValue = true;
  } else if (bindOp.getBindKind() == BindLambdaKind::InputLambda) {
    traitName = kTraitInputFusion;
    methodName = "load";
    methodReturnsValue = true;
  } else if (lambdaType == KernelOutputFusionLambdaType::Output ||
             lambdaType ==
                 KernelOutputFusionLambdaType::OutputLegacyForCompute) {
    traitName = kTraitOutputFusion;
    methodName = "store";
  } else if (lambdaType == KernelOutputFusionLambdaType::ComputeTile) {
    traitName = kTraitComputeOutputFusionTile;
    methodName = "compute";
    methodReturnsValue = true;
  } else {
    traitName = kTraitComputeOutputFusion;
    methodName = "compute";
    methodReturnsValue = true;
  }

  // Collect struct compile-time parameters, fields, and external captures.
  SmallVector<std::pair<MojoValue, MojoValue>> parameters;
  SmallVector<std::pair<MojoValue, MojoValue>> fields;
  SmallVector<MO::TensorType> captureTensorTypes;
  SmallVector<std::pair<Value, unsigned>> extraTensorCaptures;
  SmallVector<std::tuple<Value, std::string, std::string>> scalarCaptures;
  SmallVector<std::pair<Value, MojoValue>> inlineConstants;
  collectFusionCaptures(state, lambdaBody, fusibleTensors, isOutputLambda,
                        parameters, fields, captureTensorTypes,
                        extraTensorCaptures, scalarCaptures, inlineConstants);

  // Emit struct definition with trait-conforming method.
  auto structName = state.structDefsBuilder.getIdentifier("FusionStruct");
  MojoValue structNameVal = state.structDefsBuilder.newVar(structName);
  auto &sdb = state.structDefsBuilder;
  const auto simdWidthType = isOutputLambda ? kStructSIMDSize : kStructInt;
  sdb.structDecl(
      structNameVal, parameters, {MojoValue(traitName)}, fields, [&]() {
        // Trait-conforming method: load/store/compute. Parameter and argument
        // names must match the trait signature exactly so the @always_inline
        // body can reference `self.*`, `simd_width`, `dtype`, etc. without
        // counter mangling — hence `NamedMojoDef::asVerbatim`.
        MojoValue idxType = MojoValue(kStructIndexList)[kParameterRank];
        SmallVector<NamedMojoDef> methodParameters;
        SmallVector<NamedMojoDef> methodArgs;
        std::optional<MojoValue> returnType;

        if (lambdaType == KernelOutputFusionLambdaType::ComputeTile) {
          MojoValue tileTensorType = MojoValue(kStructTileTensor)[{
              kParameterDType, kParameterLayoutType, kComptimeMutAnyOrigin}];
          methodParameters = {
              NamedMojoDef::asVerbatim(kParameterDType, kStructDType),
              NamedMojoDef::asVerbatim(kParameterRank, kStructInt),
              NamedMojoDef::asVerbatim(kParameterLayoutType,
                                       kStructTensorLayout),
              NamedMojoDef::asVerbatim(kParameterCopier, kStructTileCopier),
          };
          methodArgs.push_back(NamedMojoDef::asVerbatim(kParameterSelf));
          methodArgs.push_back(
              NamedMojoDef::asVerbatim(kParameterTileCoords, idxType));
          methodArgs.push_back(
              NamedMojoDef::asVerbatim(kParameterCopier, kParameterCopier));
          methodArgs.push_back(
              NamedMojoDef::asVerbatim(kParameterVal, tileTensorType));
          returnType = tileTensorType;
        } else {
          // SIMD-based fusion paths.
          MojoValue simdType =
              MojoValue(kStructSIMD)[{kParameterDType, kParameterSimdWidth}];
          methodParameters = {
              NamedMojoDef::asVerbatim(kParameterDType, kStructDType),
              NamedMojoDef::asVerbatim(kParameterRank, kStructInt),
              NamedMojoDef::asVerbatim(kParameterSimdWidth, simdWidthType),
              NamedMojoDef::asVerbatim(kParameterElementAlignment, kStructInt,
                                       MojoValue(1)),
          };
          methodArgs.push_back(NamedMojoDef::asVerbatim(kParameterSelf));
          methodArgs.push_back(
              NamedMojoDef::asVerbatim(kParameterIdx, idxType));
          if (isOutputLambda)
            methodArgs.push_back(
                NamedMojoDef::asVerbatim(kParameterVal, simdType));
          if (methodReturnsValue)
            returnType = simdType;
        }

        sdb.def(methodName, methodParameters, methodArgs, returnType,
                /*exported=*/false, /*raises=*/false,
                /*isAlwaysInline=*/true, [&](const MOGG::FunctionBody &) {
                  emitFusionStructMethodBody(
                      state, sdb, lambdaBody, fusedInputs, captureTensorTypes,
                      extraTensorCaptures, scalarCaptures, inlineConstants,
                      isOutputLambda, lambdaType);
                });
      });

  // Debug: dump the shape/strides of each captured tensor.
  // This runs in the kernel function body (once per call) and accesses
  // the fused input's _static_shape/_static_strides.
  if (state.isDebugDumpFusedTensorSpecsEnabled()) {
    unsigned inputIdx = 0;
    for (unsigned i = 0; i < fusedInputs.size(); ++i) {
      if (i == 0 && isOutputLambda)
        continue;
      ManagedTensorSliceRef mts(fusedInputs[i]);
      state.builder.call(
          kFnPrint, {},
          {MojoValue(isOutputLambda ? kParameterOutput : kParameterInput)
               .quote(),
           MojoValue("#" + std::to_string(inputIdx++) + ":").quote(),
           MojoValue("shape =").quote(), mts.staticShapeTuple(),
           MojoValue("strides =").quote(), mts.staticStridesTuple(),
           MojoValue("alignment =").quote(), mts.alignment()});
    }
  }

  // Emit struct instantiation: decompose each tensor into (ptr, shape, strides)
  // and pass static_spec as compile-time parameters.
  SmallVector<MojoValue> comptimeArgs;
  SmallVector<MojoValue> instanceArgs;
  for (unsigned i = 0; i < fusedInputs.size(); ++i) {
    ManagedTensorSliceRef mts(fusedInputs[i]);
    comptimeArgs.push_back(mts.staticSpec());
    instanceArgs.push_back(state.builder.var(mts.unsafePtr()));
    instanceArgs.push_back(state.builder.var(mts.shape()));
    instanceArgs.push_back(state.builder.var(mts.strides()));
  }
  for (auto &[tensorVal, captureIdx] : extraTensorCaptures) {
    ManagedTensorSliceRef mts(state.getMojoValueFor(tensorVal));
    comptimeArgs.push_back(mts.staticSpec());
    instanceArgs.push_back(state.builder.var(mts.unsafePtr()));
    instanceArgs.push_back(state.builder.var(mts.shape()));
    instanceArgs.push_back(state.builder.var(mts.strides()));
  }
  for (auto &[val, fieldName, mojoType] : scalarCaptures)
    instanceArgs.push_back(state.getMojoValueFor(val));
  MojoValue fusionInstance =
      state.builder.var(structNameVal(comptimeArgs, instanceArgs));

  // Determine the bind method name for TensorFusedOp.
  // Elementwise lambdas don't bind to a tensor -- the struct is passed
  // directly to foreach -- so we use an empty bind method name.
  auto bindMethod = [&]() -> StringLiteral {
    if (bindOp.getBindKind() == BindLambdaKind::ElemwiseLambda)
      return "";

    if (bindOp.getBindKind() == BindLambdaKind::InputLambda)
      return ManagedTensorSliceRef::kBindToFusedInput;

    auto fusedOp =
        dyn_cast<MOGG::TensorFusedOp>(*bindOp.getResult().getUsers().begin());
    auto outputLambdaType = getKernelOutputFusionLambdaType(
        state.registry, fusedOp, state.enableComputeLambdas);
    if (!outputLambdaType.isError() &&
        *outputLambdaType == KernelOutputFusionLambdaType::Output) {
      return ManagedTensorSliceRef::kBindToFusedOutput;
    }
    if (!outputLambdaType.isError() &&
        *outputLambdaType == KernelOutputFusionLambdaType::ComputeTile) {
      return ManagedTensorSliceRef::kBindToFusedComputeOutputTile;
    }
    // Both SIMD compute-output cases use the same Mojo method name;
    // overload resolution distinguishes by trait type.
    return ManagedTensorSliceRef::kBindToFusedComputeOutput;
  }();
  // Build the concrete parameterized type name for use by the variadic path
  // (e.g., FusionStruct_0[arg__0.static_spec, arg__1.static_spec]).
  MojoValue concreteTypeName = structNameVal[comptimeArgs];
  state.setFusionStructLoweringFor(bindOp.getOutput(), fusionInstance,
                                   bindMethod, concreteTypeName);

  return success();
}

/// Emit a `var` local per tensor capture holding ptr/shape/strides.
static SmallVector<TensorCaptureVars> materializeTensorCaptureVars(
    EmitMojoState &state, ArrayRef<MojoValue> fusedInputs,
    ArrayRef<std::pair<Value, unsigned>> extraTensorCaptures) {
  SmallVector<TensorCaptureVars> tensorVars;
  tensorVars.reserve(fusedInputs.size() + extraTensorCaptures.size());
  auto emitVarsForTensor = [&](const MojoValue &t) {
    TensorCaptureVars v;
    v.ptrVar = state.builder.var(t.get("_ptr"));
    v.shapeVar = state.builder.var(t.get("shape")());
    v.stridesVar = state.builder.var(t.get("strides")());
    v.staticSpec = t.get("static_spec");
    tensorVars.push_back(std::move(v));
  };
  for (auto &t : fusedInputs)
    emitVarsForTensor(t);
  for (auto &[val, _] : extraTensorCaptures)
    emitVarsForTensor(state.getMojoValueFor(val));
  return tensorVars;
}

/// Pre-resolved identifiers and signature pieces for a register_passable
/// elementwise def. Resolving the param/arg ids in the outer scope lets the
/// return type splice in the SIMD-width spelling before the body runs (`newVar`
/// on an already-resolved id is a no-op, so the spellings flow through
/// unchanged).
struct ElementwiseDefSignature {
  SmallVector<NamedMojoDef> templateParams;
  SmallVector<NamedMojoDef> arguments;
  MojoValue returnType;
  MojoValue wName;
  MojoValue eaName;
  MojoValue idxName;
};

/// Build the def signature for an elementwise register_passable body.
static ElementwiseDefSignature
buildElementwiseDefSignature(MojoBuilder &builder, int64_t rank,
                             const MojoValue &bodyDtype) {
  auto wParamId = builder.getIdentifier("w");
  auto eaParamId = builder.getIdentifier("ea");
  auto idxArgId = builder.getIdentifier("idx");
  ElementwiseDefSignature sig;
  sig.wName = builder.newVar(wParamId);
  sig.eaName = builder.newVar(eaParamId);
  sig.idxName = builder.newVar(idxArgId);
  sig.templateParams = {
      NamedMojoDef(wParamId, MojoValue("Int")),
      NamedMojoDef(eaParamId, MojoValue("Int"),
                   /*defaultValue=*/MojoValue("1")),
  };
  sig.arguments = {
      NamedMojoDef(
          idxArgId,
          MojoValue(llvm::formatv("{0}[{1}]", kStructIndexList, rank).str())),
  };
  sig.returnType = MojoValue(llvm::formatv("{0}[{1}, {2}]", kStructSIMD,
                                           bodyDtype.text, sig.wName.text)
                                 .str());
  return sig;
}

/// Emit `print(...)` calls dumping each fused input's comptime
/// shape/strides/alignment, when debug dumping is enabled. Mirrors the
/// fusion-struct path's debug emission.
static void
maybeEmitDebugDumpFusedTensorSpecs(EmitMojoState &state,
                                   ArrayRef<MojoValue> fusedInputs) {
  if (!state.isDebugDumpFusedTensorSpecsEnabled())
    return;
  unsigned inputIdx = 0;
  for (auto &t : fusedInputs) {
    state.builder.call(
        "print", {},
        {MojoValue("input").quote(),
         MojoValue("#" + std::to_string(inputIdx++) + ":").quote(),
         MojoValue("shape =").quote(), t.get("_static_shape_tuple"),
         MojoValue("strides =").quote(), t.get("_static_strides_tuple"),
         MojoValue("alignment =").quote(), t.get("alignment")});
  }
}

/// Emit a register_passable elementwise body at execute-function scope and
/// register its name for the downstream `CallForeachOp`.
///
/// Unlike the fusion-struct path, whose implicit `FusionStruct` captures of
/// host stack addresses don't survive `crossDeviceCaptures` on every target,
/// this flattens each fused input's ptr/shape/strides into three execute-scope
/// `var` locals listed explicitly in the body's `{var ...}` clause, loading via
/// `simd_load_from_tensor_pointer`. Leaves one patchable layer (UnsafePointer +
/// IndexList).
static LogicalResult emitElementwiseRegisterPassableBody(
    BindLambdaOp bindOp, EmitMojoState &state, Block *lambdaBody,
    OperandRange fusibleTensors, ArrayRef<MojoValue> fusedInputs) {
  if (failed(verifyBindLambdaBodyOpsForMojoEmission(bindOp, lambdaBody)))
    return failure();

  // Reuse the legacy capture-collection: we want the same set of fields and
  // extra-tensor / scalar bookkeeping, just consumed differently.
  SmallVector<std::pair<MojoValue, MojoValue>> parameters;
  SmallVector<std::pair<MojoValue, MojoValue>> fields;
  SmallVector<MO::TensorType> captureTensorTypes;
  SmallVector<std::pair<Value, unsigned>> extraTensorCaptures;
  SmallVector<std::tuple<Value, std::string, std::string>> scalarCaptures;
  SmallVector<std::pair<Value, MojoValue>> inlineConstants;
  collectFusionCaptures(state, lambdaBody, fusibleTensors,
                        /*isOutputLambda=*/false, parameters, fields,
                        captureTensorTypes, extraTensorCaptures, scalarCaptures,
                        inlineConstants);

  // Derive the body's output dtype from the lambda terminator.
  auto outputOp = cast<MOGG::OutputOp>(lambdaBody->getTerminator());
  ErrorOr<DType> bodyDtypeOr =
      M::MO::fromMLIRType(outputOp.getOperand(0).getType());
  if (bodyDtypeOr.isError())
    return bindOp.emitError(
        "Failed to determine elementwise body output dtype");
  MojoValue bodyDtype = state.mlirDTypeToMojoType(*bodyDtypeOr);

  // `idx: IndexList[rank]` must match the rank `foreach` iterates with. Use the
  // indices arg's array size — the value BindLambdaOp::verify checks inputs
  // against, well-defined even for index-only bodies. `captureTensorTypes`
  // would mislead with no fusibles (falls back to 1, or an unrelated capture).
  int64_t rank = 1;
  if (auto idxArrayTy =
          dyn_cast<KGEN::POP::ArrayType>(bindOp.getIndicesArg().getType())) {
    if (auto resolvedSize = idxArrayTy.getResolvedSize())
      rank = std::max<int64_t>(*resolvedSize, 1);
  }

  maybeEmitDebugDumpFusedTensorSpecs(state, fusedInputs);

  // Emit execute-scope locals for each tensor capture's ptr/shape/strides.
  SmallVector<TensorCaptureVars> tensorVars =
      materializeTensorCaptureVars(state, fusedInputs, extraTensorCaptures);

  // The `{var ...}` clause must list each captured local explicitly;
  // `static_spec` is comptime and referenced without capture.
  SmallVector<MojoValue> capturedVarNames;
  capturedVarNames.reserve(3 * tensorVars.size() + scalarCaptures.size());
  for (auto &v : tensorVars) {
    capturedVarNames.push_back(v.ptrVar);
    capturedVarNames.push_back(v.shapeVar);
    capturedVarNames.push_back(v.stridesVar);
  }
  for (auto &[val, _, __] : scalarCaptures)
    capturedVarNames.push_back(state.getMojoValueFor(val));

  ElementwiseDefSignature sig =
      buildElementwiseDefSignature(state.builder, rank, bodyDtype);

  MojoValue funcName = state.builder.registerPassableElementwiseDef(
      "func", sig.templateParams, sig.arguments, capturedVarNames,
      sig.returnType, [&](const FunctionBody &) {
        // Seed the value map: inline constants -> literal text, scalar captures
        // -> their state mojo values, index arg -> renamed `idx`, each fused
        // input -> a load from its captured triple.
        DenseMap<Value, MojoValue> valueMap;
        DenseSet<Value> scalarCaptureValues;
        RegisterPassableRenderer renderer(tensorVars, sig.wName.text,
                                          sig.eaName.text);
        seedScalarValueMapEntries(
            inlineConstants, scalarCaptures,
            [&](Value val, StringRef) { return state.getMojoValueFor(val); },
            valueMap, scalarCaptureValues);
        valueMap[lambdaBody->getArguments().back()] = sig.idxName;
        for (unsigned i = 0; i < fusedInputs.size(); ++i) {
          Value blockArg = lambdaBody->getArgument(i);
          auto tensorTy = captureTensorTypes[i];
          int64_t loadRank = tensorTy.getRank() == 0 ? 1 : tensorTy.getRank();
          auto loadExpr = renderer.buildLoadExpr(
              i, llvm::formatv("rebind[{0}[{1}]]({2})", kStructIndexList,
                               loadRank, sig.idxName.text)
                     .str());
          valueMap[blockArg] = state.builder.var(MojoValue(loadExpr));
        }

        emitElementwiseLambdaBodyOps(state, state.builder, lambdaBody,
                                     captureTensorTypes, extraTensorCaptures,
                                     valueMap, scalarCaptureValues, renderer);

        // Final return value, rebound to the declared return type.
        auto resultIt = valueMap.find(outputOp.getOperand(0));
        ASSERT_STREAM(resultIt != valueMap.end(),
                      << "missing output value in register_passable "
                         "elementwise body map");
        state.builder.returnsValue(MojoValue(
            llvm::formatv("rebind[{0}[{1}, {2}]]({3})", kStructSIMD,
                          bodyDtype.text, sig.wName.text, resultIt->second.text)
                .str()));
      });

  state.setElementwiseFuncLoweringFor(bindOp.getOutput(), funcName);
  return success();
}

LogicalResult BindLambdaOp::emitMojo(EmitMojoState &state) {

  bool isOutputLambda = getBindKind() == BindLambdaKind::OutputLambda;
  // Default to Compute (return-based) so that non-output lambdas (InputLambda)
  // emit a return rather than a store in emitFusionStructReturnOrStore.
  auto lambdaType = KernelOutputFusionLambdaType::Compute;
  if (isOutputLambda) {
    // Determine whether this is a store, compute, or tile-compute lambda.
    if (!getResult().hasOneUse())
      return emitError("BindLambdaOp should have a unique use");
    auto fusedOp =
        dyn_cast<MOGG::TensorFusedOp>(*getResult().getUsers().begin());
    if (!fusedOp)
      return emitError("BindLamdbdaOp user should be a FusedTensorOp");
    auto outputLambdaTypeOr = getKernelOutputFusionLambdaType(
        state.registry, fusedOp, state.enableComputeLambdas);
    if (outputLambdaTypeOr.isError())
      return emitError(outputLambdaTypeOr.getError());
    lambdaType = *outputLambdaTypeOr;
  }

  Block *lambdaBody = getBody();

  SmallVector<MojoValue> fusedInputs;
  auto fusibleTensors = getFusibleTensors();
  for (auto input : state.getMojoValuesFor(fusibleTensors)) {
    fusedInputs.push_back(input);
    state.markAsKeepAlive(input);
  }

  // Graph-API lowering goes through `foreach[E: ElementwiseFusion]`, whose
  // capturing shim closes over the FusionStruct without the elaborator
  // flattening those captures into the outer register_passable wrapper.
  // ElemwiseLambda routes through this path instead so each fused input's
  // ptr/shape/strides triple is listed explicitly in the `{var ...}` clause.
  // Input/Output/ComputeOutput keep the FusionStruct path — consumed via
  // `bind_to_fused_*` helpers, never entering `crossDeviceCaptures`.
  if (getBindKind() == BindLambdaKind::ElemwiseLambda)
    return emitElementwiseRegisterPassableBody(*this, state, lambdaBody,
                                               fusibleTensors, fusedInputs);

  return emitFusionStruct(*this, state, lambdaBody, fusibleTensors, fusedInputs,
                          isOutputLambda, lambdaType);
}

//===----------------------------------------------------------------------===//
// MOGG KernelTerminatorOp
//===----------------------------------------------------------------------===//

LogicalResult CallForeachOp::verify() {
  if (getContext().size() > 1)
    return emitError("Can use at most one context or tuple of contexts");

  return success();
}

LogicalResult
CallForeachOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifySymbolUsesForPrimitiveDeclCall(*this, symbolTable);
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
  // Look up the lambda lowering emitted by BindLambdaOp.
  auto loweringIt = state.lambdaLowerings.find(getInput());
  if (loweringIt == state.lambdaLowerings.end())
    return emitError("No lambda lowering emitted for elementwise BindLambdaOp");
  auto &lowering = loweringIt->second;

  // Get the kernel operands.
  auto contextInputs = getContext();
  if (contextInputs.size() != 1)
    return emitError(
        "Op must have exactly one context operand to be lowered to mojo");
  const MojoValue &contextValue = state.getMojoValueFor(contextInputs.front());
  MojoValue mojoOutput = state.getMojoValueFor(getOutput());

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
  MojoValue paramsDtypesTuple = MojoValue(kStructStaticTuple)[{
      kStructDType, MojoValue(int64_t(allDtypeMojoParams.size()))}](
      {}, allDtypeMojoParams);

  // Get the target device from the kernel stub that contains this operation.
  auto kernelStubOp = this->getOperation()->getParentOfType<KernelStubOp>();
  MojoValue targetDeviceLabel =
      MojoValue(kernelStubOp
                    ? MO::getDeviceOrDefaultAny(kernelStubOp).getLabel()
                    : MO::getDeviceOrDefaultAny(*this).getLabel())
          .quote();

  MojoValue simdWidthExpr = MojoValue(kFnGetSimdWidthForDtypes)(
      {paramsDtypesTuple, targetDeviceLabel});

  // Get the trace name from the fusion info tree.
  auto traceNameOr =
      state.inferDefaultKernelParamValue(*this, kMoggTraceNameParameterName);
  MojoValue traceNameExpr = traceNameOr.isError()
                                ? MojoValue(getOperationName()).quote()
                                : *traceNameOr;

  // ElementwiseFuncInfo -> register_passable `foreach` (`func`/`tensor`/`ctx`;
  // the `func` kwarg disambiguates the same-named parametric-closure overload).
  // FusionStructInfo -> `foreach_fusion` (`tensor`/`elem`/`ctx`), named by this
  // op's `getKernelSymName()`.
  MojoValue foreachFunc;
  llvm::SmallVector<MojoValue, 3> bodyKwargs;
  if (llvm::isa<ElementwiseFuncInfo>(lowering)) {
    const auto &efInfo = llvm::cast<ElementwiseFuncInfo>(lowering);
    foreachFunc = MojoValue(kFnMoggForeach);
    bodyKwargs = {
        MojoValue(kParameterForeachFunc).equal(efInfo.funcName),
        MojoValue(kParameterForeachTensor).equal(mojoOutput),
        MojoValue(kParameterForeachCtx).equal(contextValue),
    };
  } else {
    const auto &fsInfo = llvm::cast<FusionStructInfo>(lowering);
    foreachFunc = MojoValue(getKernelSymName());
    bodyKwargs = {
        MojoValue(kParameterForeachTensor).equal(mojoOutput),
        MojoValue(kParameterForeachElem).equal(fsInfo.instance),
        MojoValue(kParameterForeachCtx).equal(contextValue),
    };
  }
  state.builder.verbatim(
      foreachFunc({MojoValue(kMoggTargetParameterName).equal(targetDeviceLabel),
                   MojoValue(kParameterSimdWidth).equal(simdWidthExpr),
                   MojoValue(kMoggTraceNameParameterName).equal(traceNameExpr)},
                  bodyKwargs));
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

LogicalResult
CallMaterializeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifySymbolUsesForPrimitiveDeclCall(*this, symbolTable);
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
  auto viewMatPrimDecl =
      state.registry.lookupPrimitiveBySymName(getKernelSymName());
  // Get the kernel operands.
  SmallVector<MojoValue> mojoOperands = {state.getMojoValuesFor(getOperands())};
  std::swap(mojoOperands[0], mojoOperands[1]);

  // Infer the kernel parameters.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    return state.inferDefaultKernelParamValue(*this, paramName);
  };
  auto maybeParams =
      EmitMojoState::inferKernelParameters(viewMatPrimDecl, inferParam);
  if (maybeParams.isError())
    return emitError() << "Failed to infer all the kernel parameters: "
                       << maybeParams.takeError();

  MojoValue mojoSymbolName(primitiveMojoName(viewMatPrimDecl));
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

LogicalResult
CallExecuteOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifySymbolUsesForKernelDeclCall(
      *this, symbolTable, [this](KernelDeclOp decl) -> LogicalResult {
        if (!decl.getExecute())
          return emitOpError("kernel '")
                 << getKernelSymName() << "' has no 'execute' function";
        return success();
      });
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
  ManagedTensorSliceRef mts(inputValue);
  StaticTensorSpecRef staticSpec(mts.staticSpec());

  // Build template parameters. The static layout is conveyed through the
  // layout argument's type (below), so no separate layout param.
  SmallVector<MojoValue> templateParams = {dtype, rank,
                                           MojoValue(false), // False constant
                                           MojoValue(kEnumIoInput),
                                           staticSpec.alignment()};

  // Build runtime arguments: the slice already carries its layout (shape and
  // strides as static-aware Coords), so forward it directly.
  SmallVector<MojoValue> runtimeArgs = {
      OpaquePointerRef(mts.rawPtr())
          .bitcast(MojoValue(kComptimeDTypeNoneType))(),
      inputValue.get("runtime_layout")()};

  // Generate the mogg_tensor_init call
  auto tensorInitPrimDecl =
      state.registry.lookupPrimitive(TensorInitOp::getOperationName());
  MojoValue moggTensorInit(primitiveMojoName(tensorInitPrimDecl));
  return state.builder.var(moggTensorInit(templateParams, runtimeArgs));
}
} // namespace

LogicalResult CallExecuteOp::emitMojo(EmitMojoState &state) {
  KernelDeclOp execKernelDecl =
      state.registry.lookupKernelBySymName(getKernelSymName());
  KernelDeclAdaptor execAdaptor(execKernelDecl);
  const KernelDeclFunctionAdaptor &execFnAdaptor = *execAdaptor.executeFunction;

  // Collect all kernel I/Os
  SmallVector<MojoValue> kernelOperands;
  for (Value val : getKernelDPSOutputs())
    kernelOperands.push_back(state.getMojoValueFor(val));

  auto inputs = getKernelInputs();
  for (unsigned i = 0; i < inputs.size(); ++i) {
    Value input = inputs[i];
    MojoValue inputValue = state.getMojoValueFor(input);

    // Transform mutable inputs if kernel expects InputTensor.
    // Skip for variadic kernel args (e.g., bundled collectives with
    // N-expanded inputs) where the adaptor entry is a
    // KernelDeclVariadicTensorAdaptor, not a KernelDeclTensorAdaptor.
    if (isMutableInputTensor(input) &&
        i < execFnAdaptor.inputArguments.size()) {
      const auto *tensorInfo = std::get_if<KernelDeclTensorAdaptor>(
          &execFnAdaptor.inputArguments[i].underlyingType);
      if (tensorInfo && tensorInfo->ioSpec == MOGG::IOSpec::InputTensor)
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
  auto maybeParams = EmitMojoState::inferKernelParameters(
      execAdaptor, MOGG::kMoggExecuteFunctionLabel, inferParam);
  if (maybeParams.isError())
    return emitError() << "Failed to infer all the kernel parameters: "
                       << maybeParams.takeError();

  auto [prefixSymbolName, executeFunc] = EmitMojoState::splitFunctionName(
      execFnAdaptor.fullyResolvedName.strref());
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
// Error context helpers (used by KernelStubOp and AsyncErrorOp)
//===----------------------------------------------------------------------===//

/// Collect source notes (Python stack traces) from a location tree. Source
/// notes are StringAttr metadata on FusedLoc nodes, set by
/// AttachPythonStackTraces.
static SmallVector<StringRef, 4> collectSourceNotes(Location rootLoc) {
  SmallVector<StringRef, 4> notes;
  llvm::SmallSet<StringRef, 4> seen;
  SmallVector<Location, 8> worklist = {rootLoc};
  while (!worklist.empty()) {
    Location loc = worklist.pop_back_val();
    auto fusedLoc = dyn_cast<mlir::FusedLoc>(loc);
    if (!fusedLoc)
      continue;
    if (auto metadata = dyn_cast_or_null<StringAttr>(fusedLoc.getMetadata())) {
      StringRef val = metadata.getValue();
      if (!val.empty() && seen.insert(val).second)
        notes.push_back(val);
    }
    for (Location subLoc : fusedLoc.getLocations())
      worklist.push_back(subLoc);
  }
  return notes;
}

/// Parse a single Python traceback frame line of the form:
///   File "path/to/file.py", line 42, in func_name
/// into a compact "last/components.py:42:func_name" representation, keeping
/// only the trailing `numComponents` path segments.
static std::string compactFrame(StringRef frameLine,
                                unsigned numComponents = 3) {
  // Expected format: `  File "<path>", line <N>, in <func>`
  auto filePos = frameLine.find("File \"");
  if (filePos == StringRef::npos)
    return "";
  StringRef afterFile = frameLine.substr(filePos + 6); // skip `File "`
  auto quoteEnd = afterFile.find('"');
  if (quoteEnd == StringRef::npos)
    return "";
  StringRef fullPath = afterFile.substr(0, quoteEnd);
  StringRef rest = afterFile.substr(quoteEnd + 1); // after closing quote

  // Extract line number: `, line <N>`
  auto linePos = rest.find("line ");
  StringRef lineNo;
  if (linePos != StringRef::npos) {
    StringRef afterLine = rest.substr(linePos + 5);
    lineNo = afterLine.take_while([](char c) { return c >= '0' && c <= '9'; });
  }

  // Extract function name: `, in <func>`
  StringRef func;
  auto inPos = rest.find(", in ");
  if (inPos != StringRef::npos) {
    func = rest.substr(inPos + 5).trim();
  }

  // Keep only the last N path components.
  SmallVector<StringRef, 8> components;
  fullPath.split(components, '/');
  unsigned start =
      components.size() > numComponents ? components.size() - numComponents : 0;
  std::string shortPath;
  llvm::raw_string_ostream pathOs(shortPath);
  for (unsigned i = start; i < components.size(); ++i) {
    if (i > start)
      pathOs << '/';
    pathOs << components[i];
  }

  std::string result;
  llvm::raw_string_ostream os(result);
  os << shortPath;
  if (!lineNo.empty())
    os << ':' << lineNo;
  if (!func.empty())
    os << ':' << func;
  return result;
}

/// Return the maximum number of Python stack frames to include in NVTX trace
/// detail strings. Configurable via MODULAR_TRACE_MAX_FRAMES (default 10).
static unsigned getTraceMaxFrames() {
  static unsigned cached = [] {
    if (const char *env = std::getenv("MODULAR_TRACE_MAX_FRAMES")) {
      int val = std::atoi(env);
      if (val > 0)
        return static_cast<unsigned>(val);
    }
    return 10u;
  }();
  return cached;
}

/// Collect compact source notes for NVTX trace detail strings. Delegates to
/// collectSourceNotes for the FusedLoc tree walk, then reformats each trace
/// to show only the bottom N frames in compact `path:line:func` form.
/// N is controlled by MODULAR_TRACE_MAX_FRAMES (default 10). Source detail is
/// non-empty only when the `max-debug.source-tracebacks` config key was
/// enabled at graph construction time (for example, via
/// `Graph.debug.source_tracebacks = True` or
/// `MODULAR_DEBUG=source-tracebacks`).
static std::string collectCompactSourceNotes(Location rootLoc) {
  unsigned maxFrames = getTraceMaxFrames();
  auto notes = collectSourceNotes(rootLoc);
  if (notes.empty())
    return "";

  std::string result;
  llvm::raw_string_ostream os(result);

  for (auto note : notes) {
    // Parse all frames from the traceback, then take the bottom N.
    SmallVector<std::string, 8> frames;
    StringRef remaining = note;
    while (!remaining.empty()) {
      auto [line, rest] = remaining.split('\n');
      if (line.contains("File \""))
        if (auto compact = compactFrame(line); !compact.empty())
          frames.push_back(std::move(compact));
      remaining = rest;
    }

    unsigned start = frames.size() > maxFrames ? frames.size() - maxFrames : 0;
    for (unsigned i = start; i < frames.size(); ++i) {
      if (i > start)
        os << '\n';
      os << frames[i];
    }
    if (!frames.empty())
      os << '\n';
  }

  if (notes.size() > 1)
    os << "[fused from " << notes.size() << " ops]";

  return result;
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
  // MOGG ABI kernels always use `with Trace[...]` which can raise, so they
  // unconditionally return true. This also prevents deep recursion through
  // CallStubOp chains in sequence-fused kernels.
  if (getAbi() == ABIKind::MOGG)
    return true;
  return llvm::any_of(getBodyBlock().getOps<MOGG::EmitsMojoInterface>(),
                      [](auto emitMojoOp) { return emitMojoOp.isThrows(); });
}

LogicalResult KernelStubOp::emitMojo(EmitMojoState &state) {
  // Emit the fusion info tree as a top-level comment before the function.
  state.builder.comment("FusionInfo:");
  std::string fusionTree = getFusionInfoAttr().toPrettyTree();
  for (auto line : llvm::split(fusionTree, '\n'))
    state.builder.comment(("  " + line).str());

  // If the ABI is MOGG, this was originally a MOGG kernel. Otherwise, it was a
  // GenericCompiledRegionOp which has that implicit void** ABI.
  if (getAbi() == ABIKind::MOGG)
    return emitMojoMOGGABI(state);
  return emitMojoMGPABI(state);
}

LogicalResult KernelStubOp::emitMojoMOGGABI(EmitMojoState &state) {
  // Helper to convert type to Mojo, using proper opaque type representation
  // for parameterized opaque types.
  auto typeToMojo = [&](Type type) -> MojoValue {
    if (auto opaqueType = dyn_cast<MO::OpaqueType>(type)) {
      MojoValue baseTy = state.mlirOpaqueTypeToMojoType(opaqueType);
      auto params = EmitMojoState::getOpaqueTypeParameters(opaqueType);
      if (!params.empty())
        return baseTy[params];
      return baseTy;
    } else if (dyn_cast<MGP::DeviceBufferType>(type)) {
      // mgp.buffer is represented as MutByteBuffer (DynamicTensor[DType.int8,
      // 1]).
      return kComptimeMutByteBuffer;
    }
    return state.mlirTypeToMojoType(type);
  };

  SmallVector<MOGG::NamedMojoDef> inputTypes;
  for (auto arg : getBody().getArguments())
    inputTypes.push_back(MOGG::NamedMojoDef("arg", typeToMojo(arg.getType())));

  // Determine the Mojo return type from the stub's return type.
  std::optional<MOGG::MojoValue> mojoReturnType = std::nullopt;
  FunctionType functionType = getFunctionType();
  if (functionType.getNumResults())
    mojoReturnType = typeToMojo(functionType.getResult(0));

  bool errOccurred = false;
  state.builder.def(
      getSymNameAttr().str(), {}, inputTypes,
      /*returnType=*/mojoReturnType, /*exported=*/false,
      /*raises=*/isThrows(), /*isAlwaysInline=*/false,
      [&](const MOGG::FunctionBody &functionBody) {
        state.currentFunctionBody = functionBody;

        for (auto &&[mlirArg, mojoArg] :
             llvm::zip(getBody().getArguments(), functionBody.arguments))
          state.setMojoValueFor(mlirArg, mojoArg);

        // Emit body ops: TryOp (wrapping setup, WithTraceOp, RaiseOp) and
        // OutputOp. Each op's emitMojo handles the details.
        errOccurred = llvm::any_of(getOps(), [&](mlir::Operation &op) {
          return state.lowerOpToMojo(&op).failed();
        });
      });

  return failure(errOccurred);
}

LogicalResult KernelStubOp::emitMojoMGPABI(EmitMojoState &state) {
  // Clear any previous kernel's return values from the state.
  state.mojoReturnValues.clear();
  state.mojoConstantResources.clear();

  bool errOccurred = false;
  static const MOGG::MojoValue vMutAnyOrigin =
      MOGG::MojoValue(MOGG::kComptimeMutAnyOrigin);
  static const MOGG::MojoValue vUnsafePointer = MOGG::kStructUnsafePointer;
  static const MOGG::MojoValue vOpaquePointer = MOGG::kComptimeOpaquePointer;

  state.builder.def(
      getSymNameAttr().strref(), {},
      {MOGG::NamedMojoDef("arg",
                          vUnsafePointer[{vOpaquePointer, vMutAnyOrigin}])},
      std::nullopt, /*exported=*/true, /*raises=*/false,
      /*isAlwaysInline=*/false, [&](const MOGG::FunctionBody &functionBody) {
        state.currentFunctionBody = functionBody;
        state.builder.comment(
            "This is the entry point of the kernel. The argument is a "
            "pointer to a list of inputs and outputs.");

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
              OpaquePointerRef(runtimeArgument)
                  .bitcast(MojoValue(kStructInt))()[flatIndex + 1];
          state.mojoConstantResources.emplace_back(
              std::tuple<MOGG::MojoValue, MOGG::MojoValue>{ptr, byteCount});
        }

        // void** ABI entry-point functions are never `raises`, so we cannot
        // use `with Trace[...]` here (Trace.__enter__ is raises). Tracing
        // is handled in the MOGG ABI function this entry point calls.
        errOccurred = llvm::any_of(getOps(), [&](mlir::Operation &op) {
          return state.lowerOpToMojo(&op).failed();
        });
        if (!errOccurred)
          state.genAllKeepAliveStatements();
      });

  if (errOccurred)
    return failure();
  return success();
}

// These FuncOp attributes are disallowed while parsing since they can
// be inferred. Likewise while printing we ignore them.
static StringRef kernelStubOpDisallowedAttrNames[] = {
    "name", "returnType", "sym_name", "function_type", "fusionInfo"};

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

  // Parse required fusionInfo before attr-dict.
  FusionInfoAttr fusionInfo;
  if (parseFusionInfo(parser, fusionInfo))
    return failure();
  if (!fusionInfo)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected 'fusionInfo' attribute");
  state.addAttribute("fusionInfo", fusionInfo);

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  llvm::SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed : kernelStubOpDisallowedAttrNames) {
    if (parsedAttributes.get(disallowed)) {
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
    }
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
  printFusionInfo(p, *this, getFusionInfoAttr());
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     kernelStubOpDisallowedAttrNames);
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
  bool addedEffect = false;
  for (auto &&opOperand : getOperation()->getOpOperands()) {
    if (!MO::isMutableOperand(opOperand.get()))
      continue;
    if (hasWriteEffect)
      effects.emplace_back(mlir::MemoryEffects::Write::get(), &opOperand);
    if (hasReadEffect)
      effects.emplace_back(mlir::MemoryEffects::Read::get(), &opOperand);
    addedEffect = true;
  }

  // Kernels with observable side effects (e.g. nan_check_raise) but no
  // mutable operands still need a generic write effect to prevent DCE.
  // This is only triggered when hasWriteEffect is explicitly set to true
  // during lowering (e.g. by NanCheckPass for the raise kernel).
  if (!addedEffect && (hasReadEffect || hasWriteEffect))
    effects.emplace_back(mlir::MemoryEffects::Write::get());
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
      parseRegion(parser, argInfo, *state.addRegion(), true))
    return failure();

  // Parse required fusionInfo before attr-dict.
  FusionInfoAttr fusionInfo;
  if (parseFusionInfo(parser, fusionInfo))
    return failure();
  if (!fusionInfo) {
    return parser.emitError(parser.getCurrentLocation(),
                            "'mogg.kernel' requires a 'fusionInfo' attribute");
  }
  state.addAttribute("fusionInfo", fusionInfo);

  if (parser.parseOptionalAttrDict(state.attributes))
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
  printFusionInfo(p, *this, getFusionInfoAttr());
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          {"operandSegmentSizes", "fusionInfo"});
}

void KernelOp::build(OpBuilder &builder, OperationState &state,
                     TypeRange results, ValueRange operands,
                     FusionInfoAttr fusionInfo, BoolAttr isAsync,
                     DeviceRefAttr device,
                     ArrayRef<ParamDeclAttr> outputParamDecls,
                     ArrayAttr allocIndices, bool hasReadEffect,
                     bool hasWriteEffect) {
  state.addOperands(operands);
  state.addAttribute(getFusionInfoAttrName(state.name), fusionInfo);

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

  auto asScalarPrimDecl =
      state.registry.lookupPrimitive(this->getOperationName());
  ASSERT_STREAM(asScalarPrimDecl,
                "Primitive not found for " << getOperationName().str());
  MojoValue moggAsScalar(primitiveMojoName(asScalarPrimDecl));
  MOGG::MojoValue scalarValue = moggAsScalar(ArrayRef<MojoValue>{}, {input});
  state.setMojoValueFor(getResult(), state.builder.var(scalarValue));
  return success();
}

//===----------------------------------------------------------------------===//
// GenericCompiledRegionOp
//===----------------------------------------------------------------------===//

OutputOp GenericCompiledRegionOp::getBodyTerminator() {
  return cast<OutputOp>(getBodyBlock().getTerminator());
}

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

  // Each body block argument is a `!mogg.async<T>` wrapping the corresponding
  // operand type `T`. The region boundary wraps operands into async values on
  // entry; the body unpacks them via `mogg.async.unpack`. Arguments that are
  // deliberately consumed as the raw async slot (e.g. a context) may instead be
  // retyped to the plain operand type `T` during ABI lowering, so both forms
  // are accepted.
  for (auto [operandType, argType] :
       llvm::zip(getOperandTypes(), body.getArgumentTypes())) {
    if (argType == operandType)
      continue;
    auto asyncType = dyn_cast<AsyncType>(argType);
    if (!asyncType || asyncType.getElementType() != operandType)
      return emitError("each region body argument must be the corresponding "
                       "operand type or a mogg.async of it");
  }

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

// Combine two in/out semantics by unioning their read and write capabilities.
// kNone='.', kIn='i', kOut='o', kMut='m' (see Support/.../MAttrs.td). When the
// same value is passed as multiple operands, the surviving operand must reflect
// every access the body performs through any of the duplicates.
static InOutSignatureAttr::InOutSemantics
unionInOutSemantics(InOutSignatureAttr::InOutSemantics a,
                    InOutSignatureAttr::InOutSemantics b) {
  using Sem = InOutSignatureAttr::InOutSemantics;
  auto reads = [](Sem s) { return s == Sem::kIn || s == Sem::kMut; };
  auto writes = [](Sem s) { return s == Sem::kOut || s == Sem::kMut; };
  bool r = reads(a) || reads(b);
  bool w = writes(a) || writes(b);
  if (r && w)
    return Sem::kMut;
  if (r)
    return Sem::kIn;
  if (w)
    return Sem::kOut;
  return Sem::kNone;
}

// Canonicalize duplicated operands of a generic compiled region:
//
//     mogg.generic.compiled.region(%a, %b, %a) : (%x, %y, %z) { ...use %z... }
//
// to
//
//     mogg.generic.compiled.region(%a, %b) : (%x, %y) { ...use %x... }
//
// The block argument for each duplicate is redirected to the first
// occurrence's argument, the duplicate operand/argument are erased, and the
// surviving operand's in/out semantics absorb the union of the duplicates'.
static LogicalResult deduplicateOperands(GenericCompiledRegionOp op,
                                         PatternRewriter &rewriter) {
  Block &entry = op.getBody().front();

  // Map each distinct operand value to the index of its first occurrence and
  // mark later duplicates for removal.
  llvm::SmallDenseMap<Value, unsigned> firstIndex;
  llvm::BitVector argsToRemove(op.getNumOperands());
  for (OpOperand &operand : op.getOperandsMutable()) {
    unsigned idx = operand.getOperandNumber();
    if (!firstIndex.try_emplace(operand.get(), idx).second)
      argsToRemove.set(idx);
  }

  if (argsToRemove.none())
    return rewriter.notifyMatchFailure(op, "no duplicate operands");

  // Redirect each duplicate block argument to its first occurrence so the body
  // reads/writes the single surviving argument.
  for (unsigned idx : argsToRemove.set_bits()) {
    unsigned keptIdx = firstIndex[op.getOperand(idx)];
    rewriter.replaceAllUsesWith(entry.getArgument(idx),
                                entry.getArgument(keptIdx));
  }

  // Fold the removed entries' semantics into their first occurrence, then drop
  // the duplicate entries. Empty signatures (used by some unittests) are left
  // untouched.
  InOutSignatureAttr oldSemantics = op.getInOutSignature();
  InOutSignatureAttr newSemantics;
  if (oldSemantics.size() != 0) {
    SmallVector<InOutSignatureAttr::InOutSemantics> merged(oldSemantics.size());
    for (size_t i = 0, e = oldSemantics.size(); i < e; ++i)
      merged[i] = oldSemantics[i];
    for (unsigned idx : argsToRemove.set_bits()) {
      unsigned keptIdx = firstIndex[op.getOperand(idx)];
      merged[keptIdx] = unionInOutSemantics(merged[keptIdx], oldSemantics[idx]);
    }
    SmallVector<InOutSignatureAttr::InOutSemantics> kept;
    for (size_t i = 0, e = oldSemantics.size(); i < e; ++i)
      if (!argsToRemove.test(i))
        kept.push_back(merged[i]);
    newSemantics = InOutSignatureAttr::get(op.getContext(), kept);
  }

  rewriter.modifyOpInPlace(op, [&]() {
    op->eraseOperands(argsToRemove);
    entry.eraseArguments(argsToRemove);
    if (newSemantics)
      op.setInOutSignatureAttr(newSemantics);
  });

  return success();
}

// Canonicalize sequences of grt.chain.create + mogg.generic.compiled.region
//
//     %ch = grt.chain.create(%0, %1)
//     mogg.generic.compiled.region(%ch)
//
// to
//
//     mogg.generic.compiled.region(%0, %1)
static LogicalResult sinkChainCreateOperands(GenericCompiledRegionOp op,
                                             PatternRewriter &rewriter) {
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

  // Body block arguments are `!mogg.async<T>` wrappers around the operand
  // types.
  SmallVector<Type> newTypes = llvm::map_to_vector(
      newOperands, [](Value v) -> Type { return AsyncType::get(v.getType()); });

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

// Canonicalize redundant outputs of a generic compiled region by rebuilding it
// with a collapsed result list:
//   - drop non-chain results that have no uses, and
//   - collapse duplicate `!mef.chain` outputs into the last chain output, which
//     is always retained so error/side-effect propagation and async-completion
//     sequencing are preserved.
// Uses of removed chain results are rewired to the surviving chain.
static LogicalResult removeRedundantResults(GenericCompiledRegionOp op,
                                            PatternRewriter &rewriter) {
  OutputOp outputOp = op.getBodyTerminator();

  // Candidates for removal: every chain output (duplicate chains are collapsed
  // into one), plus any non-chain output whose corresponding region result is
  // unused.
  llvm::BitVector toRemoveBits(outputOp->getNumOperands());

  for (OpOperand &operand : outputOp->getOpOperands()) {
    unsigned idx = operand.getOperandNumber();
    bool isChain = isa<MEF::ChainType>(operand.get().getType());
    if (!isChain && !op->getResult(idx).use_empty())
      continue;
    toRemoveBits.set(idx);
  }

  if (toRemoveBits.none())
    return rewriter.notifyMatchFailure(op, "no removable outputs");

  // Always keep the last chain output (collapsing any duplicates into it) so
  // error/side-effect propagation and async-completion sequencing are
  // preserved. Uses of the removed chains are rewired to this one.
  std::optional<unsigned> keptChainNewIdx;
  for (OpOperand &operand : llvm::reverse(outputOp->getOpOperands())) {
    if (isa<MEF::ChainType>(operand.get().getType())) {
      unsigned origIdx = operand.getOperandNumber();
      toRemoveBits.reset(origIdx);
      // Position of the kept chain within the collapsed result list.
      unsigned newIdx = 0;
      for (unsigned i = 0; i < origIdx; ++i)
        if (!toRemoveBits.test(i))
          ++newIdx;
      keptChainNewIdx = newIdx;
      break;
    }
  }

  // Nothing to remove, or removing everything would leave no result and no
  // side-effect chain: leave the op unchanged rather than asserting.
  if (toRemoveBits.none() || toRemoveBits.all())
    return rewriter.notifyMatchFailure(op, "no redundant outputs to remove");

  // Collapse the terminator operand list, then rebuild the op with the
  // corresponding (unwrapped) result types.
  rewriter.modifyOpInPlace(outputOp,
                           [&]() { outputOp->eraseOperands(toRemoveBits); });

  SmallVector<Type> newResultTypes =
      llvm::map_to_vector(outputOp->getOperandTypes(), [](Type type) -> Type {
        if (auto asyncType = dyn_cast<AsyncType>(type))
          return asyncType.getElementType();
        return type;
      });

  // Rebuild the op with the collapsed result types, carrying over all of its
  // attributes (inherent + discardable) so nothing relied on later in the
  // pipeline (e.g. isStructuredKernel, emitsMojo) is dropped.
  rewriter.setInsertionPoint(op);
  auto newKernel = GenericCompiledRegionOp::create(
      rewriter, op->getLoc(), newResultTypes, op.getOperands(),
      op.getOutputParamDeclsAttr(), op.getInOutSignatureAttr(),
      op.getFusionInfoAttr(),        // fusionInfo
      op.getIsAsyncAttr(),           // isAsync
      op.getConstantResourcesAttr(), // constantResources
      op.getCanFuseIntoOtherAttr(),  // canFuseIntoOther
      op.getCanBeFusedIntoAttr(),    // canBeFusedInto
      op.getCanDuplicateAttr(),      // canDuplicate
      op.getIsStructuredKernelAttr() // isStructuredKernel
  );

  newKernel.setDeviceRef(op.getExecutionDevice());
  newKernel->setDiscardableAttrs(op->getDiscardableAttrDictionary());
  rewriter.inlineRegionBefore(op.getBody(), newKernel.getBody(),
                              newKernel.getBody().end());

  // Rewire uses: kept results map positionally; a removed chain redirects to
  // the surviving chain; removed unused non-chain results have no uses.
  unsigned curIdx = 0;
  for (OpResult result : op->getOpResults()) {
    if (toRemoveBits.test(result.getResultNumber())) {
      if (keptChainNewIdx && isa<MEF::ChainType>(result.getType()))
        rewriter.replaceAllUsesWith(result,
                                    newKernel->getResult(*keptChainNewIdx));
    } else {
      rewriter.replaceAllUsesWith(result, newKernel.getResult(curIdx));
      ++curIdx;
    }
  }
  assert(curIdx == newKernel->getNumResults());
  rewriter.eraseOp(op);
  return success();
}

LogicalResult GenericCompiledRegionOp::canonicalize(GenericCompiledRegionOp op,
                                                    PatternRewriter &rewriter) {
  if (succeeded(deduplicateOperands(op, rewriter)))
    return success();
  if (succeeded(sinkChainCreateOperands(op, rewriter)))
    return success();
  return removeRedundantResults(op, rewriter);
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

LogicalResult AsyncUnpackOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto asyncType = dyn_cast<AsyncType>(adaptor.getInput().getType());
  if (!asyncType)
    return emitOptionalError(location,
                             "async.unpack input must be a mogg.async type");
  inferredReturnTypes.push_back(asyncType.getElementType());
  return success();
}

LogicalResult AsyncUnpackOp::canonicalize(AsyncUnpackOp unpackOp,
                                          mlir::PatternRewriter &rewriter) {
  if (auto packOp = unpackOp.getInput().getDefiningOp<AsyncPackOp>()) {
    rewriter.replaceOp(unpackOp, packOp.getInput());
    return success();
  }

  if (auto packOp = unpackOp.getInput().getDefiningOp<AsyncPackBorrowOp>()) {
    rewriter.replaceOp(unpackOp, packOp.getInput());
    return success();
  }

  return failure();
}

LogicalResult AsyncUnpackOp::emitMojo(EmitMojoState &state) {
  MojoValue unpackedValue;
  mlir::TypedValue<mlir::Type> input = getInput();
  // The input is a `!mogg.async<T>`; switch on the wrapped element type `T`.
  mlir::Type inputType = cast<AsyncType>(input.getType()).getElementType();

  if (auto tensorSpec = dyn_cast<MGP::TensorSpecType>(inputType)) {

    MojoValue asyncUnpackFunc{kFnUnpackTensorSpec};
    unpackedValue = asyncUnpackFunc({int(tensorSpec.getRank())},
                                    {state.getMojoValueFor(input)});

  } else if (auto tensorType = dyn_cast<MGP::DeviceTensorType>(inputType)) {

    MojoValue asyncUnpackFunc{kFnUnpackTensor};
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

    // TODO: (GEX-2732) Move away from NDBuffer in primitives.mojo
    MojoValue asyncUnpackFunc{kFnUnpackBufferRef};
    unpackedValue = asyncUnpackFunc({}, {state.getMojoValueFor(input)});

  } else if (isa<MGP::DeviceContextType>(inputType)) {

    MojoValue asyncUnpackFunc{kFnUnpackDeviceCtx};
    unpackedValue = asyncUnpackFunc({}, {state.getMojoValueFor((input))});

  } else {

    auto unpackPrimDecl =
        state.registry.lookupPrimitive(this->getOperationName());
    ASSERT_STREAM(unpackPrimDecl,
                  "Primitive not found for " << getOperationName().str());
    MojoValue moggAsyncUnpackFunc(primitiveMojoName(unpackPrimDecl));
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
    OpaquePointerRef opaquePtr(finalValue);
    // Store the pointer in a variable
    MojoValue ptrVar = state.builder.var(opaquePtr.bitcast(opaqueTypeStr)());
    // The value to use is the dereference expression: var_XX[]
    finalValue = state.builder.var(OpaquePointerRef(ptrVar).takePointee());
  }

  state.setMojoValueFor(getResult(), finalValue);
  return success();
}

//===----------------------------------------------------------------------===//
// AsyncPackOp
//===----------------------------------------------------------------------===//

LogicalResult
AsyncPackOp::inferReturnTypes(MLIRContext *context,
                              std::optional<Location> location, Adaptor adaptor,
                              SmallVectorImpl<Type> &inferredReturnTypes) {
  // The packed value is the `input` operand; the optional `deviceCtx` operand
  // does not affect the result. The result is an `!mogg.async<T>` of the input.
  inferredReturnTypes.push_back(
      AsyncType::get(context, adaptor.getInput().getType()));
  return success();
}

LogicalResult AsyncPackOp::emitMojo(EmitMojoState &state) {
  auto resultIndexOr = getOutputIndex();
  if (!resultIndexOr)
    return failure();
  MojoValue asyncPackHelperStruct = kStructMoggAsyncPackHelper;
  auto packPrimDecl = state.registry.lookupPrimitive(this->getOperationName());
  ASSERT_STREAM(packPrimDecl,
                "Primitive not found for " << getOperationName().str());
  MojoValue moggAsyncPackFunc(primitiveMojoName(packPrimDecl));
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
  auto asyncType = dyn_cast<AsyncType>(getResult().getType());
  if (!asyncType || asyncType.getElementType() != getInput().getType())
    return emitError(
        "async.pack result must be a mogg.async of the input type");
  return success();
}

//===----------------------------------------------------------------------===//
// AsyncDelOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncDelOp::emitMojo(EmitMojoState &state) {
  auto delPrimDecl = state.registry.lookupPrimitive(this->getOperationName());
  ASSERT_STREAM(delPrimDecl,
                "Primitive not found for " << getOperationName().str());
  MojoValue moggAsyncDelFunc(primitiveMojoName(delPrimDecl));
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
  auto ioParam = MojoValue(kStructIO).get(MO::stringifyIOKind(getIo()));
  MojoValue rankParam =
      MojoValue(static_cast<uint64_t>(tensorTy.getRankNonZero()));
  MojoValue alignmentParam(
      static_cast<uint64_t>(M::Driver::getPreferredAlignment(
          MO::getDeviceOrDefaultAny(tensorTy).getLabel().str())));
  // Encode the shape as a `Coord` (static dims as `Idx[N]`, dynamic dims as
  // `Scalar[DType.int](<dim>)`), then build a row-major `Layout` from it. The
  // layout conveys the static/dynamic structure to `mogg_tensor_init` through
  // its Coord element types instead of a separate static-layout parameter.
  auto shapeVar = state.builder.var(
      state.getShapeAsCoordMojoValue(tensorTy.getTypedShapeAttr(), getDims()));
  auto layoutVar = state.builder.var(MojoValue(kFnRowMajor)({}, {shapeVar}));
  auto tensorInitPrimDecl2 =
      state.registry.lookupPrimitive(this->getOperationName());
  ASSERT_STREAM(tensorInitPrimDecl2,
                "Primitive not found for " << getOperationName().str());
  MojoValue moggTensorInit(primitiveMojoName(tensorInitPrimDecl2));
  state.setMojoValueFor(
      getResult(),
      state.builder.var(moggTensorInit(
          {dTypeParam, rankParam, mutParam, ioParam, alignmentParam},
          {OpaquePointerRef(
               ManagedTensorSliceRef(state.getMojoValueFor(getBuffer()))
                   .unsafePtr())
               .bitcast(MojoValue(kComptimeDTypeNoneType))(),
           layoutVar})));

  return success();
}

//===----------------------------------------------------------------------===//
// Shared verification for CallStubOp / CallStubAsyncOp
//===----------------------------------------------------------------------===//

static FailureOr<KernelStubOp>
verifyKernelStubCall(Operation *op, SymbolTableCollection &symbolTable,
                     FlatSymbolRefAttr calleeSymbol,
                     TypeRange callerOperandTypes) {
  auto calleeFunc =
      symbolTable.lookupNearestSymbolFrom<KernelStubOp>(op, calleeSymbol);
  if (!calleeFunc) {
    return op->emitError(llvm::formatv("There is no kernel stub named '{}'",
                                       calleeSymbol.getValue()));
  }

  auto calleeArgumentTypes = calleeFunc.getArgumentTypes();

  if (callerOperandTypes.size() != calleeArgumentTypes.size()) {
    return op->emitError(
        llvm::formatv("Kernel stub '{0}' expects {1} arguments, but got {2}",
                      calleeSymbol.getValue(), calleeArgumentTypes.size(),
                      callerOperandTypes.size()));
  }

  for (auto [i, calleeArgType, callerOperandType] :
       llvm::enumerate(calleeArgumentTypes, callerOperandTypes)) {
    if (calleeArgType == callerOperandType)
      continue;

    if (isa<MOGG::ContextType>(calleeArgType) &&
        isa<MGP::DeviceContextType>(callerOperandType))
      continue;

    return op->emitError(llvm::formatv(
        "Kernel stub '{0}' expects argument {1} to have type {2}, but got "
        "{3}",
        calleeSymbol.getValue(), i, calleeArgType, callerOperandType));
  }

  return calleeFunc;
}

//===----------------------------------------------------------------------===//
// CallStubOp
//===----------------------------------------------------------------------===//

KernelStubOp CallStubOp::getCallee() {
  mlir::SymbolTable symTable(getOperation()->getParentOfType<ModuleOp>());
  return dyn_cast_or_null<KernelStubOp>(symTable.lookup(getName()));
}

LogicalResult CallStubOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto calleeFunc = verifyKernelStubCall(*this, symbolTable, getCalleeSymbol(),
                                         getOperandTypes());
  if (failed(calleeFunc))
    return failure();

  if (calleeFunc->getAbi() != ABIKind::MOGG)
    return emitError("Call site should be in the MOGG ABI");

  return success();
}

bool CallStubOp::isThrows() {
  auto callee = getCallee();
  return callee && callee.isThrows();
}

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
// CallStubAsyncOp
//===----------------------------------------------------------------------===//

KernelStubOp CallStubAsyncOp::getCallee() {
  mlir::SymbolTable symTable(getOperation()->getParentOfType<ModuleOp>());
  return dyn_cast_or_null<KernelStubOp>(symTable.lookup(getName()));
}

LogicalResult
CallStubAsyncOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyKernelStubCall(*this, symbolTable, getCalleeSymbol(),
                              getOperandTypes());
}

LogicalResult CallStubAsyncOp::emitMojo(EmitMojoState &state) {
  state.builder.newline();

  // 1. Create error slot: var error_N = Optional[Error]()
  MojoValue errorVar = state.builder.var(MojoValue(kComptimeOptionalErrorEmpty),
                                         /*cache=*/false);

  // 2. Create async wrapper closure that calls the stub
  SmallVector<MojoValue> stubArgs = state.getMojoValuesFor(getOperands());
  MojoValue stubFunc(getCallee().getSymNameAttr().str());

  MojoValue wrapper = state.builder.lambda(
      "wrapper", /*parameters=*/{}, /*arguments=*/{},
      /*returnType=*/MojoValue(kComptimeNone),
      /*isInline=*/true, /*isParameter=*/true, /*isAsync=*/true,
      [&](const FunctionBody &) {
        state.builder
            .try_([&]() {
              state.builder.call(stubFunc, /*parameters=*/{}, stubArgs);
            })
            .except([&](const MojoValue &errVar) {
              state.builder.assign(errorVar, MojoValue(errVar.text + "^"));
            });
      });

  // 3. Launch the task.
  //
  // For GPU launches, pin the task to the CPU worker with NUMA affinity for
  // that device by using the affinity-aware _create_task variant.  For all
  // other cases (no affinity hint, CPU/host device) fall back to the standard
  // create_task which passes desired_worker_id=-1 to the AsyncRT scheduler.
  //
  // The device ID is stored as $affinity_device_id (I64Attr) rather than
  // being read from $device (DeviceRefAttr), because mogg-emit-mojo internally
  // runs stripDeviceIdsAndDynamicShapes before emission — that pass zeroes all
  // DeviceRefAttr IDs, destroying the per-launch device ordinal.  A plain
  // integer attribute is not affected by that transformation.
  MojoValue taskVar;
  auto affinityId = getAffinityDeviceIdAttr();
  if (affinityId) {
    int64_t deviceId = affinityId.getInt();
    // var worker_id_N = std.runtime.asyncrt.task_id_for_device(<deviceId>)
    MojoValue workerIdVar =
        state.builder.var(MojoValue(kFnTaskIdForDevice)(
                              /*params=*/{}, /*args=*/{MojoValue(deviceId)}),
                          /*cache=*/false);
    // var task_N = std.runtime.asyncrt._create_task(
    //     wrapper_N(), desired_worker_id=worker_id_N)
    // MojoBuilder has no built-in keyword-argument support, so we construct
    // the call text directly.
    taskVar = state.builder.var(
        MojoValue(llvm::formatv("{}({}, desired_worker_id={})",
                                kFnCreateTaskWithAffinity, wrapper({}, {}).text,
                                workerIdVar.text)
                      .str()),
        /*cache=*/false);
  } else {
    // var task_N = std.runtime.asyncrt.create_task(wrapper_N())
    taskVar = state.builder.var(MojoValue(kFnCreateTask)({}, {wrapper({}, {})}),
                                /*cache=*/false);
  }

  // 4. Record task handle info for AsyncJoinOp to consume
  state.taskHandleInfos[getTaskHandle()] = {taskVar, errorVar};

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
  auto readyPrimDecl = state.registry.lookupPrimitive(this->getOperationName());
  ASSERT_STREAM(readyPrimDecl,
                "Primitive not found for " << getOperationName().str());
  MojoValue moggAsyncReadyFunc(primitiveMojoName(readyPrimDecl));
  state.builder.call(moggAsyncReadyFunc, {},
                     state.mojoReturnValues[*resultIndexOr]);

  return success();
}

//===----------------------------------------------------------------------===//
// AsyncJoinOp
//===----------------------------------------------------------------------===//

bool AsyncJoinOp::isThrows() { return true; }

LogicalResult AsyncJoinOp::emitMojo(EmitMojoState &state) {
  // Wait for all tasks to complete before checking any errors.
  for (Value taskHandle : getTaskHandles()) {
    auto it = state.taskHandleInfos.find(taskHandle);
    if (it == state.taskHandleInfos.end())
      return emitOpError("missing TaskHandleInfo for operand");
    state.builder.keepAlive(MojoTaskRef(it->second.task).wait());
  }

  auto checkTaskErrPrimDecl =
      state.registry.lookupPrimitive(getOperationName());
  MojoValue checkErrorFunc(primitiveMojoName(checkTaskErrPrimDecl));

  for (Value taskHandle : getTaskHandles()) {
    auto &info = state.taskHandleInfos[taskHandle];
    state.builder.call(checkErrorFunc, {}, {info.error});
  }

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
// AsyncPackBorrowOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncPackBorrowOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Like async.pack, the borrowed result is an `!mogg.async<T>` of the input.
  inferredReturnTypes.push_back(
      AsyncType::get(context, adaptor.getInput().getType()));
  return success();
}

LogicalResult AsyncPackBorrowOp::emitMojo(EmitMojoState &state) {
  auto resultIndexOr = getOutputIndex();
  if (!resultIndexOr)
    return failure();
  MojoValue moggAsyncPackBorrowFunc{kFnMoggAsyncPackBorrow};
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
  if (auto packOp = unpackOp.getValue().getDefiningOp<AsyncPackBorrowOp>()) {
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
  MojoValue getBufferMemStorageHandleFn = kFnGetBufferMemStorageHandle;
  MojoValue inputVar = state.getMojoValueFor(getValue());
  // The value is a `!mogg.async<T>` (a runtime slot); dispatch on the wrapped
  // element type.
  Type valueType = getValue().getType().getElementType();
  MojoValue typeVar = isa<MGP::DeviceBufferType>(valueType)
                          ? (size_t)MGPRT_BorroweeType::kBuffer
                          : (size_t)MGPRT_BorroweeType::kTensor;
  auto opaquePtr = state.builder.var(MojoValue{kComptimeOpaquePointer}.get(
      OpaquePointerRef::kUnsafeDangling)());
  auto opaquePtrPtr = state.builder.var(
      OpaquePointerRef(MojoValue{kStructUnsafePointer}(
                           {}, MojoValue(kParameterTo).equal(opaquePtr)))
          .bitcast(MojoValue(kComptimeDTypeNoneType))());

  state.builder.call(getBufferMemStorageHandleFn, {},
                     {inputVar, typeVar, opaquePtrPtr});
  state.setMojoValueFor(getResult(), opaquePtr);
  return success();
}

//===----------------------------------------------------------------------===//
// CallViewOp
//===----------------------------------------------------------------------===//

LogicalResult CallViewOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifySymbolUsesForKernelDeclCall(
      *this, symbolTable, [this](KernelDeclOp decl) -> LogicalResult {
        if (!decl.getUpdateView())
          return emitOpError("kernel '")
                 << getKernelSymName() << "' has no 'update_view' function";
        return success();
      });
}

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

  KernelDeclOp uvKernelDecl =
      state.registry.lookupKernelBySymName(getKernelSymName());
  KernelDeclAdaptor uvAdaptor(uvKernelDecl);
  const KernelDeclFunctionAdaptor &uvFnAdaptor = *uvAdaptor.updateViewFunction;

  // Infer the kernel parameters.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    constexpr StringLiteral staticOutShapeParamName = "output_static_shape";
    if (paramName == staticOutShapeParamName) {
      return state.builder.comptime(EmitMojoState::getAsIntTupleMojoValue(
          getResult().getType().getIntDims()));
    }
    return state.inferDefaultKernelParamValue(*this, paramName);
  };
  auto maybeParams = EmitMojoState::inferKernelParameters(
      uvAdaptor, MOGG::kMoggUpdateViewFunctionLabel, inferParam);
  if (maybeParams.isError())
    return emitError() << "Failed to infer all the kernel parameters: "
                       << maybeParams.takeError();

  state.setMojoValueFor(
      getResult(), state.builder.var(callFunctionNestedInStruct(
                       uvFnAdaptor, *maybeParams, inputMojoValues, state)));
  return success();
}

//===----------------------------------------------------------------------===//
// CallElementwiseOp
//===----------------------------------------------------------------------===//

LogicalResult
CallElementwiseOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifySymbolUsesForKernelDeclCall(
      *this, symbolTable, [this](KernelDeclOp decl) -> LogicalResult {
        if (!decl.getElementwise())
          return emitOpError("kernel '")
                 << getKernelSymName() << "' has no 'elementwise' function";
        return success();
      });
}

LogicalResult CallElementwiseOp::emitMojo(EmitMojoState &state) {
  // Get the lambda inputs
  SmallVector<MojoValue> inputs = state.getMojoValuesFor(getInputs());

  KernelDeclOp ewKernelDecl =
      state.registry.lookupKernelBySymName(getKernelSymName());
  KernelDeclAdaptor ewAdaptor(ewKernelDecl);
  const KernelDeclFunctionAdaptor &ewFnAdaptor =
      ewAdaptor.isElementwise() ? *ewAdaptor.elementwiseFunction
                                : *ewAdaptor.executeFunction;

  auto outputs = getOutputs();
  if (outputs.size() != 1)
    return emitError(
        "elementwise with outputs size != 1 not supported for EmitMojo");
  Value outputVal = outputs.front();

  // Infer the kernel parameters.
  // For ops like mo.cast, where the output dtype differs, we need to explicitly
  // specify the output dtype.
  // TODO(GEX-2535): Avoid hardcoding the parameter name.
  std::optional<StringRef> maybeDtypeParamName =
      EmitMojoState::getResultParamNameFor(ewFnAdaptor, kParameterDType);
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    if (maybeDtypeParamName.has_value() && paramName == *maybeDtypeParamName) {
      ErrorOr<DType> resultTyOr = M::MO::fromMLIRType(outputVal.getType());
      if (resultTyOr.isError())
        return Error("invalid output type can't be converted to mojo dtype");
      return state.mlirDTypeToMojoType(*resultTyOr);
    }
    return state.inferDefaultKernelParamValue(*this, paramName);
  };
  llvm::StringRef ewFuncLabel = ewAdaptor.isElementwise()
                                    ? MOGG::kMoggElementwiseFunctionLabel
                                    : MOGG::kMoggExecuteFunctionLabel;
  auto maybeParams =
      EmitMojoState::inferKernelParameters(ewAdaptor, ewFuncLabel, inferParam);
  if (maybeParams.isError())
    return emitError() << "Failed to infer all the kernel parameters: "
                       << maybeParams.takeError();

  MojoValue mojoSymbolName(ewFnAdaptor.fullyResolvedName.strref());
  state.setMojoValueFor(
      outputVal, state.builder.var(
                     mojoSymbolName(maybeParams->functionParameters, inputs)));
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
  MOGG::MojoValue loadedSimd = MojoValue(kFnSimdLoadFromManagedTensorSlice)(
      {MojoValue(kParameterSimdWidth).equal(widthParam),
       MojoValue(kParameterElementAlignment).equal(alignmentParam)},
      {tensor, indices});
  state.setMojoValueFor(getOutputs(), state.builder.var(loadedSimd));
  return success();
}

//===----------------------------------------------------------------------===//
// EmptyBufferOp
//===----------------------------------------------------------------------===//

LogicalResult EmptyBufferOp::emitMojo(EmitMojoState &state) {
  // Represent the empty buffer with a dangling pointer.
  MojoValue danglingPointer =
      MojoValue(kStructUnsafePointer)
          [{MojoValue(
                kStructScalar)[{MojoValue(kStructDType).get(kEnumDTypeInt8)}],
            MojoValue(kComptimeMutAnyOrigin)}]
              .get(OpaquePointerRef::kUnsafeDangling)();
  MojoValue zeroShape =
      MojoValue(kStructIndexList)({MojoValue("1")}, {MojoValue("0")});
  state.setMojoValueFor(getResult(),
                        state.builder.var(MojoValue(kComptimeMutByteBuffer)(
                            {}, {danglingPointer, zeroShape})));
  return success();
}

//===----------------------------------------------------------------------===//
// TensorFusedOp
//===----------------------------------------------------------------------===//

LogicalResult TensorFusedOp::emitMojo(EmitMojoState &state) {

  MojoValue tensorValue = state.getMojoValueFor(getTensor());

  auto loweringIt = state.lambdaLowerings.find(getLambda());
  if (loweringIt == state.lambdaLowerings.end())
    return emitError("No lambda lowering emitted for BindLambdaOp -- "
                     "lambda-based fusion has been removed");
  // The verifier restricts `lambda` to Input/OutputLambda, which always lower
  // through the FusionStruct path; ElemwiseLambda never reaches here.
  const auto &fusionInfo = llvm::cast<FusionStructInfo>(loweringIt->second);

  // Check if the result feeds into an AsTupleOp (variadic path).
  // If so, defer binding -- AsTupleOp will construct a
  // FusedInputVariadicTensors with the fusion struct in a _FusionPack.
  bool isVariadicPath =
      llvm::any_of(getResult().getUsers(),
                   [](Operation *user) { return isa<MOGG::AsTupleOp>(user); });

  if (isVariadicPath) {
    // Pass through the UNFUSED tensor. Store the fusion struct info
    // keyed by this TensorFusedOp's result so AsTupleOp can find it.
    state.setMojoValueFor(getResult(), tensorValue);
    state.setFusionStructLoweringFor(getResult(), fusionInfo.instance,
                                     fusionInfo.bindMethodName,
                                     fusionInfo.typeName);
    return success();
  }

  // Non-variadic path: bind fusion directly to the tensor.
  MojoValue fusedTensor =
      tensorValue.get(fusionInfo.bindMethodName)({}, {fusionInfo.instance});
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
    // Build a `DeviceContextList[N]` directly from the per-device
    // contexts via its variadic constructor.
    MojoValue deviceContextListConstructor = MojoValue(
        kStructDeviceContextList)[{
        MojoValue(kParameterSize).equal(std::to_string(inputValues.size()))}];
    MojoValue inputTupleValue = deviceContextListConstructor({}, inputValues);
    state.setMojoValueFor(getResult(), state.builder.var(inputTupleValue));
    return success();
  }

  SmallVector<MojoValue> runtimePointers =
      llvm::map_to_vector(inputValues, [&](const MojoValue &val) {
        return ManagedTensorSliceRef(val).rawPtr();
      });

  SmallVector<MojoValue> runtimeShapes =
      llvm::map_to_vector(inputValues, [&](const MojoValue &val) {
        return ManagedTensorSliceRef(val).shape();
      });

  MojoValue inputTupleConstructor = MojoValue(kStructStaticTuple)[{
      MojoValue(kParameterSize).equal(std::to_string(inputValues.size()))}];

  MojoValue runtimePointersTuple =
      state.builder.var(inputTupleConstructor({}, runtimePointers));
  MojoValue runtimeShapesTuple =
      state.builder.var(inputTupleConstructor({}, runtimeShapes));

  // Create the StaticTensorSpecList from per-element internals, shapes,
  // and strides. Each spec is split into its homogeneous runtime part
  // (StaticTensorSpecInternal) and its compile-time shape/strides so that
  // heterogeneous specs can be stored in a single list.
  MojoValue rank = StaticTensorSpecRef(
                       ManagedTensorSliceRef(inputValues.front()).staticSpec())
                       .rank();
  SmallVector<MojoValue> staticSpecInternals;
  SmallVector<MojoValue> staticSpecShapes;
  SmallVector<MojoValue> staticSpecStrides;
  for (const auto &reboundInput : inputValues) {
    StaticTensorSpecRef spec(ManagedTensorSliceRef(reboundInput).staticSpec());
    staticSpecInternals.push_back(spec.getInternals());
    // `shape_tuple` / `strides_tuple` are comptime members of the
    // `StaticTensorSpec` itself, distinct from its `get_internals()` result.
    staticSpecShapes.push_back(
        MojoValue(kFnToIndexList)({rank}, {spec.shapeTuple()}));
    staticSpecStrides.push_back(
        MojoValue(kFnToIndexList)({rank}, {spec.stridesTuple()}));
  }
  MojoValue inputSpecsInternalsComptime =
      state.builder.comptime(ParameterListRef::of(staticSpecInternals));
  MojoValue inputSpecsShapesComptime =
      state.builder.comptime(ParameterListRef::of(staticSpecShapes));
  MojoValue inputSpecsStridesComptime =
      state.builder.comptime(ParameterListRef::of(staticSpecStrides));
  MojoValue inputSpecsComptime =
      state.builder.comptime(MojoValue(kStructStaticTensorSpecList)[{
          inputSpecsInternalsComptime, inputSpecsShapesComptime,
          inputSpecsStridesComptime}]());

  // Check whether inputs have fusion structs. Fused variadics require every
  // element to carry a real fusion struct — partial fusion (some elements
  // fused, some not) is not supported.
  bool anyFused = false;
  bool allFused = true;
  bool hasInputFusion = false;
  for (auto input : getInputs()) {
    auto it = state.lambdaLowerings.find(input);
    if (it != state.lambdaLowerings.end()) {
      const auto &fusionInfo = llvm::cast<FusionStructInfo>(it->second);
      anyFused = true;
      if (fusionInfo.bindMethodName == ManagedTensorSliceRef::kBindToFusedInput)
        hasInputFusion = true;
    } else {
      allFused = false;
    }
  }
  ASSERT_STREAM(!anyFused || allFused,
                << "partial fusion on variadic tensors is not supported: "
                   "either all elements must have fusion structs or none");

  if (anyFused) {
    SmallVector<MojoValue> fusionValues;
    SmallVector<MojoValue> typeParams;
    for (auto input : getInputs()) {
      auto fusIt = state.lambdaLowerings.find(input);
      const auto &fusionInfo = llvm::cast<FusionStructInfo>(fusIt->second);
      fusionValues.push_back(fusionInfo.instance);
      typeParams.push_back(fusionInfo.typeName);
    }

    MojoValue fusionsTuple =
        state.builder.var(MojoValue(kStructMsFusionPack)({}, fusionValues));

    StringRef structName = hasInputFusion ? kStructMsFusedInputVariadicTensors
                                          : kStructMsFusedOutputVariadicTensors;

    // Pass static_specs as keyword-only (after *FusionTypes in the struct).
    // dtype, rank, and size are inferred from constructor args and
    // static_specs. Fusion types are passed positionally (consumed by
    // the *FusionTypes variadic param).
    typeParams.push_back(
        MojoValue(kParameterStaticSpecs).equal(inputSpecsComptime));

    MojoValue variadicTensorsConstructor = MojoValue(structName)[typeParams];

    MojoValue variadicTensorsValue = variadicTensorsConstructor(
        {}, {runtimePointersTuple, runtimeShapesTuple, fusionsTuple});

    state.setMojoValueFor(getResult(), state.builder.var(variadicTensorsValue));
    return success();
  }

  MojoValue variadicTensorsConstructor = MojoValue(kStructVariadicTensors)[{
      MojoValue(kParameterIOSpec)
          .equal(ManagedTensorSliceRef(inputValues[0]).ioSpec()),
      MojoValue(kParameterStaticSpecs).equal(inputSpecsComptime)}];

  MojoValue variadicTensorsValue = variadicTensorsConstructor(
      {}, {runtimePointersTuple, runtimeShapesTuple});

  state.setMojoValueFor(getResult(), state.builder.var(variadicTensorsValue));
  return success();
}

//===----------------------------------------------------------------------===//
// CallShapeOp
//===----------------------------------------------------------------------===//

LogicalResult
CallShapeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifySymbolUsesForKernelDeclCall(
      *this, symbolTable, [this](KernelDeclOp decl) -> LogicalResult {
        if (!decl.getShape())
          return emitOpError("kernel '")
                 << getKernelSymName() << "' has no 'shape' function";
        return success();
      });
}

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
  KernelDeclOp shapeKernelDecl =
      state.registry.lookupKernelBySymName(getKernelSymName());
  KernelDeclAdaptor shapeAdaptor(shapeKernelDecl);
  const KernelDeclFunctionAdaptor *shapeFnAdaptor =
      shapeAdaptor.shapeFunction ? &*shapeAdaptor.shapeFunction : nullptr;
  if (!shapeFnAdaptor) {
    return emitError("Shape function not present for ") << getKernelSymName();
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
  auto maybeParams = EmitMojoState::inferKernelParameters(
      shapeAdaptor, MOGG::kMoggShapeFunctionLabel, inferParam);
  if (maybeParams.isError())
    return emitError() << "Failed to infer all the kernel parameters: "
                       << maybeParams.takeError();

  // Call the shape function and return its value via the kernel terminator.
  MojoValue result = state.builder.var(callFunctionNestedInStruct(
      *shapeFnAdaptor, *maybeParams, kernelOperands, state));
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
  if (auto intScalar = MO::tryGetConstantInt(getValue())) {
    state.setMojoValueFor(getResult(), MojoValue(*intScalar));
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
// WithTraceOp
//===----------------------------------------------------------------------===//

bool WithTraceOp::isThrows() { return true; }

LogicalResult WithTraceOp::emitMojo(EmitMojoState &state) {
  // Find the enclosing KernelStubOp to get the device and fusion info.
  auto kernelStubOp = getOperation()->getParentOfType<MOGG::KernelStubOp>();
  if (!kernelStubOp)
    return emitOpError("WithTraceOp must be nested in a KernelStubOp");

  std::string deviceLabel =
      MO::getDeviceOrDefaultAny(kernelStubOp).getLabel().str();
  std::string traceName =
      kernelStubOp.getFusionInfoAttr().toString(/*includeLineNumbers=*/false,
                                                /*quote=*/false,
                                                /*stripDialectPrefixes=*/true);

  // Build compact source traceback from FusedLoc metadata set by
  // AttachPythonStackTraces, to surface Python call sites in NVTX markers.
  std::string sourceDetail = collectCompactSourceNotes(kernelStubOp.getLoc());

  // Find the context argument (if present) to include task_id in Trace.
  MojoValue contextMojoArg;
  for (auto mlirArg : kernelStubOp.getBody().getArguments()) {
    if (isa<MOGG::ContextType>(mlirArg.getType())) {
      contextMojoArg = state.getMojoValueFor(mlirArg);
      break;
    }
  }

  // Build: Trace[TraceLevel.OP, target=StaticString("device")]("name"[,
  // detail=..., task_id=...])
  SmallVector<MojoValue> traceParams = {
      MojoValue(kEnumTraceLevelOp),
      MojoValue(kMoggTargetParameterName)
          .equal(MojoValue(kStructStaticString)(
              {}, {MojoValue(deviceLabel).quote()}))};
  SmallVector<MojoValue> traceOperands = {MojoValue(traceName).quote()};
  if (!sourceDetail.empty())
    traceOperands.push_back(MojoValue("\n" + sourceDetail).quote());
  if (!contextMojoArg.text.empty())
    traceOperands.push_back(
        MojoValue(kParameterTaskId)
            .equal(MojoValue(kFnGetSafeTaskId)({}, {contextMojoArg})));
  MojoValue traceExpr = MojoValue(kStructTrace)(traceParams, traceOperands);

  bool errOccurred = false;
  state.builder.with_(traceExpr, [&]() {
    for (auto &op : getBody().getOps()) {
      if (state.lowerOpToMojo(&op).failed()) {
        errOccurred = true;
        return;
      }
    }
  });

  if (!errOccurred)
    state.genAllKeepAliveStatements();

  return failure(errOccurred);
}

//===----------------------------------------------------------------------===//
// RaiseOp
//===----------------------------------------------------------------------===//

bool RaiseOp::isThrows() { return true; }

LogicalResult RaiseOp::emitMojo(EmitMojoState &state) {
  // Find the enclosing KernelStubOp for kernel context (name, fusion info,
  // source tracebacks).
  auto kernelStubOp = getOperation()->getParentOfType<MOGG::KernelStubOp>();
  if (!kernelStubOp)
    return emitOpError("RaiseOp must be nested in a KernelStubOp");

  std::string stubName = kernelStubOp.getSymNameAttr().str();

  std::string fusionInfoContent =
      kernelStubOp.getFusionInfoAttr().toPrettyTree({}, /*indent=*/true);

  // Build source traceback content from FusedLoc metadata set by
  // AttachPythonStackTraces.
  auto notesList = collectSourceNotes(kernelStubOp.getLoc());
  std::string tracebackContent;
  if (!notesList.empty()) {
    llvm::raw_string_ostream os(tracebackContent);
    constexpr size_t kMaxInlineTraces = 32;
    size_t numShown = std::min(notesList.size(), kMaxInlineTraces);
    for (size_t i = 0; i < numShown; ++i)
      os << notesList[i] << '\n';
    if (notesList.size() > 1) {
      os << "\nNote: " << notesList.size()
         << " distinct Python source locations contributed ops to this fused"
            " kernel. ";
      if (numShown == notesList.size()) {
        os << "The source traceback for each location is above.\n";
      } else {
        os << "Only the first " << kMaxInlineTraces
           << " source tracebacks are shown above.\n";
      }
    }
  }

  auto fmtErrPrimDecl =
      state.registry.lookupPrimitive(this->getOperationName());
  ASSERT_STREAM(fmtErrPrimDecl,
                "Primitive not found for " << getOperationName().str());
  MojoValue formatKernelErrorFunc(primitiveMojoName(fmtErrPrimDecl));
  MojoValue errorExpr = state.getMojoValueFor(getError());
  SmallVector<MojoValue> args = {
      MojoValue(stubName).quote(),
      errorExpr,
      MojoValue(fusionInfoContent).escape(),
      MojoValue(tracebackContent).escape(),
  };
  auto formattedErr = state.builder.var(formatKernelErrorFunc({}, args));
  state.builder.raise(formattedErr.transfer());

  return success();
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

LogicalResult TryOp::emitMojo(EmitMojoState &state) {
  // MOGG ABI stubs catch errors, format them with kernel context (name,
  // fusion info, source traceback), and re-raise. The outer TryOp here
  // just emits the try/except structure — no caller-side bookkeeping needed.
  // AsyncErrorOp::emitMojo wraps the formatted error with the region name.
  bool errOccurred = false;
  state.builder
      .try_([&]() {
        for (auto &op : getTryRegion().getOps()) {
          if (state.lowerOpToMojo(&op).failed()) {
            errOccurred = true;
            return;
          }
        }
      })
      .except([&](const MojoValue &e) {
        state.setMojoValueFor(getExceptRegion().getArgument(0), e);
        errOccurred |=
            llvm::any_of(getExceptRegion().getOps(), [&](mlir::Operation &op) {
              return state.lowerOpToMojo(&op).failed();
            });
      });

  return failure(errOccurred);
}

bool TryOp::isThrows() {
  auto regionThrows = [](Region &region) {
    return llvm::any_of(region.front().getOps<MOGG::EmitsMojoInterface>(),
                        [](auto op) { return op.isThrows(); });
  };
  return regionThrows(getTryRegion()) && regionThrows(getExceptRegion());
}

namespace {
// Pattern to remove unused results from TryOp and update yield operations.
struct RemoveUnusedTryResults : public OpRewritePattern<TryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TryOp tryOp,
                                PatternRewriter &rewriter) const override {

    // Collect which results are actually used
    SmallVector<unsigned> usedResultIndices;
    SmallVector<Type> usedResultTypes;
    for (auto [idx, result] : llvm::enumerate(tryOp.getResults())) {
      if (!result.use_empty()) {
        usedResultIndices.push_back(idx);
        usedResultTypes.push_back(result.getType());
      }
    }

    // If all results are used, no change needed
    if (usedResultIndices.size() == tryOp.getNumResults())
      return failure();

    // Create a new try op with only the used result types
    auto newTryOp =
        TryOp::create(rewriter, tryOp.getLoc(), TypeRange(usedResultTypes));

    // Clone the try region
    rewriter.cloneRegionBefore(tryOp.getTryRegion(), newTryOp.getTryRegion(),
                               newTryOp.getTryRegion().begin());

    // Clone the except region
    rewriter.cloneRegionBefore(tryOp.getExceptRegion(),
                               newTryOp.getExceptRegion(),
                               newTryOp.getExceptRegion().begin());

    // Update the yield operation in the try region
    auto tryYield =
        cast<YieldOp>(newTryOp.getTryRegion().front().getTerminator());
    SmallVector<Value> newTryYieldOperands;
    for (unsigned idx : usedResultIndices) {
      newTryYieldOperands.push_back(tryYield.getOperand(idx));
    }
    rewriter.setInsertionPoint(tryYield);
    rewriter.replaceOpWithNewOp<YieldOp>(tryYield, newTryYieldOperands);

    // Update the yield operation in the except region
    auto exceptYield =
        cast<YieldOp>(newTryOp.getExceptRegion().front().getTerminator());
    SmallVector<Value> newExceptYieldOperands;
    SmallVector<Value> usedExceptValues;
    for (unsigned idx : usedResultIndices) {
      Value operand = exceptYield.getOperand(idx);
      newExceptYieldOperands.push_back(operand);
      usedExceptValues.push_back(operand);
    }
    rewriter.setInsertionPoint(exceptYield);
    rewriter.replaceOpWithNewOp<YieldOp>(exceptYield, newExceptYieldOperands);

    // Erase unused AsyncErrorOp operations from the except block to prevent CSE
    // from merging them. Each AsyncErrorOp must remain distinct because they
    // correspond to specific output indices for emit-mojo code generation.
    SmallVector<Operation *> opsToErase;
    for (Operation &op : newTryOp.getExceptRegion().front()) {
      if (auto asyncError = dyn_cast<AsyncErrorOp>(&op)) {
        if (!llvm::is_contained(usedExceptValues, asyncError.getResult())) {
          opsToErase.push_back(&op);
        }
      }
    }
    for (Operation *op : opsToErase) {
      rewriter.eraseOp(op);
    }

    // Replace uses of the old try op with the new one
    // Map old results to new results for used indices
    SmallVector<Value> replacements(tryOp.getNumResults());
    for (auto [newIdx, oldIdx] : llvm::enumerate(usedResultIndices)) {
      replacements[oldIdx] = newTryOp.getResult(newIdx);
    }

    rewriter.replaceOp(tryOp, replacements);
    return success();
  }
};
} // namespace

void TryOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<RemoveUnusedTryResults>(context);
}

//===----------------------------------------------------------------------===//
// AsyncErrorOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncErrorOp::emitMojo(EmitMojoState &state) {
  auto asyncErrPrimDecl =
      state.registry.lookupPrimitive(this->getOperationName());
  ASSERT_STREAM(asyncErrPrimDecl,
                "Primitive not found for " << getOperationName().str());
  MojoValue moggAsyncErrorFunc(primitiveMojoName(asyncErrPrimDecl));

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

  MojoValue errorExpr = state.getMojoValueFor(getError());

  // Get the parent MGP region name for the error header.
  std::string regionName;
  if (auto parentKernel = tryOp->getParentOfType<MOGG::KernelStubOp>())
    regionName = parentKernel.getSymNameAttr().str();

  // Format the error with the region name prefix and call mogg_async_error.
  // This primitive is a helper, not a 1:1 mapping with `AsyncErrorOp`, so we
  // can't derive its registration name from `getOperationName()`.
  auto fmtRegionErrPrimDecl =
      state.registry.lookupPrimitive(kMoggPrimitiveFormatRegionError);
  ASSERT_STREAM(fmtRegionErrPrimDecl,
                "format_region_error primitive not found");
  MojoValue formatRegionErrorFunc(primitiveMojoName(fmtRegionErrPrimDecl));
  MojoValue formattedErr = state.builder.var(
      formatRegionErrorFunc({}, {MojoValue(regionName).quote(), errorExpr}));
  state.builder.call(
      moggAsyncErrorFunc, {},
      {state.mojoReturnValues[outputIndex], formattedErr.transfer()});

  return success();
}

//===----------------------------------------------------------------------===//
// TensorReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult TensorReshapeOp::emitMojo(EmitMojoState &state) {
  // Call "reshape_contiguous_buffer" primitive with the output shape /
  // strides.
  MO::TensorType toType = MO::castFromTensorLike(getResult().getType());
  MojoValue tensorValue = state.getMojoValueFor(getTensor());
  MojoValue shapeValue = state.getShapeAsIndexListMojoValue(
      toType.getTypedShapeAttr(), getDynamicDims());
  MojoValue layoutParam =
      state.getStaticTileLayoutFromIndexListsMojoValue(toType);
  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  ASSERT_STREAM(primDecl,
                "Primitive not found for " << getOperationName().str());
  MojoValue reshapeFunc(primitiveMojoName(primDecl));
  state.setMojoValueFor(
      getResult(),
      state.builder.var(reshapeFunc(
          {MojoValue(StaticTensorSpecRef::kStaticLayout).equal(layoutParam)},
          {tensorValue, shapeValue})));
  return success();
}

//===----------------------------------------------------------------------===//
// AsyncPlaceholderAnyAsyncValueRefOp
//===----------------------------------------------------------------------===//
LogicalResult
AsyncPlaceholderAnyAsyncValueRefOp::emitMojo(EmitMojoState &state) {
  MojoValue result = state.builder.var(
      MojoValue{kStructOptional}[{MojoValue{kComptimeOpaquePointer}}]());
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
      parseRegionArguments(p, resultTypes, args))
    return failure();

  // The signature is written in terms of the operand (element) types `T`; the
  // operands are `T` while the region body arguments are `!mogg.async<T>`.
  for (auto &arg : args)
    operandTypes.push_back(arg.type);

  // For GenericCompiledRegionOp, if the kernel takes in a chain, there will be
  // 1 fewer argument than operands.
  operandTypes.resize(operands.size(), MO::ChainType::get(p.getContext()));

  // Wrap the parsed (element) types into `!mogg.async<T>` for the body block
  // arguments before creating the region.
  for (auto &arg : args)
    arg.type = AsyncType::get(arg.type);

  if (p.getCurrentLocation(&loc) || p.parseRegion(region, args))
    return failure();

  return success();
}

static void printKernel(OpAsmPrinter &p, Operation *kernel, ValueRange operands,
                        TypeRange operandTypes, TypeRange resultTypes,
                        Region &region, KGEN::ParamDeclArrayAttr resultParams) {
  printParamDecls(p, kernel, resultParams);
  p << '(';
  p.printOperands(operands);
  p << ") : ";

  // The body block arguments are `!mogg.async<T>`; print the unwrapped element
  // type `T` so the signature reads in terms of the operand types.
  p << '(';
  llvm::interleaveComma(region.getArguments(), p, [&](BlockArgument arg) {
    p.printRegionArgument(arg, /*argAttrs=*/{}, /*omitType=*/true);
    p << ": ";
    Type type = arg.getType();
    if (auto asyncType = dyn_cast<AsyncType>(type))
      type = asyncType.getElementType();
    p.printType(type);
  });
  p << ')';
  if (!resultTypes.empty()) {
    p << ' ';
    p.printArrowTypeList(resultTypes);
  }
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

void MOGGDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGG.cpp.inc"
      >();
}

using PointerType = KGEN::PointerType;
#define GET_OP_CLASSES
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGG.cpp.inc"
