//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect operations.
//
//===----------------------------------------------------------------------===//
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/Conversion/ParameterInference.h"
#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/Interfaces/LoweringState.h"
#include "GraphCompiler/Interfaces/MOGGKernelOpLoweringState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "GraphCompiler/MOGGDialect/Support/MojoTraits.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/Support/NameMangling.h"
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
#include <cmath>
#include <cstdlib>
#include <future>

using namespace M;
using namespace MOGG;

StringAttr M::MOGG::getKernelStubName(FusionInfoAttr fusionInfo, ABIKind abi) {
  ASSERT_STREAM(fusionInfo, << "cannot name a kernel stub without fusion info");
  MLIRContext *ctx = fusionInfo.getContext();
  // Flatten to the same human-readable form used for trace labels (e.g.
  // `Elementwise(relu, add)`), then sanitize into a symbol.
  auto displayName = StringAttr::get(
      ctx, fusionInfo.toString(/*includeLineNumbers=*/false, /*quote=*/false,
                               /*stripDialectPrefixes=*/true));
  auto sanitizedName = KGEN::sanitizeSymbolToUnderscores(displayName);

  // Add the tag after sanitizing, which truncates long names and any tag.
  return StringAttr::get(ctx, llvm::formatv("{0}_{1}", sanitizedName.getValue(),
                                            stringifyABIKind(abi).lower())
                                  .str());
}

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

/// Trait-lane fusion-functor body (Phase 5): loads through a `TileTensor` built
/// from the captured `(ptr, layout)` of capture <i> via the dense
/// `tile.load(idx)` API -- NO `ManagedTensorSlice` /
/// `simd_load_from_tensor_pointer`. Matches the mockup
/// `AddTwoInputLoadFusion.__call__`. The functor `__call__`'s width param is
/// `width` and its index `idx` is a `Coord`; loads are at the iteration index
/// (`indicesExpr` is ignored -- the only form simple elementwise prologue
/// fusion produces).
struct DenseFusionRenderer final : ElementwiseBodyRenderer {
  std::string buildLoadExpr(unsigned captureIdx,
                            StringRef /*indicesExpr*/) const override {
    // The functor captures each source as a bare pointer + layout (NOT a
    // `TileTensor` -- a `TileTensor` field would not translate to a device
    // address, since the encoder bit-copies it; a bare `UnsafePointer`
    // translates through its own `_to_device_type`). Rebuild the dense tile
    // here and load from it, so the same functor body serves CPU and GPU.
    auto ci = std::to_string(captureIdx);
    return llvm::formatv(
               "TileTensor[dtype=Self.dtype_{0}, LayoutType=Self.LayoutType_{0}"
               ", origin=UntrackedOrigin[mut=Self.mut_{0}]](ptr=self.ptr_{0}, "
               "layout=self.layout_{0}).load[width=width, "
               "alignment=element_alignment]({1})",
               ci, kParameterIdx)
        .str();
  }
  StringRef simdWidthParamName() const override { return "width"; }
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

/// Add the runtime field(s) for a tensor capture. The tile path stores a single
/// storage-agnostic `TileTensor`; the SIMD path stores decomposed
/// `(ptr, shape, strides)`.
static void
addTensorCaptureField(SmallVectorImpl<std::pair<MojoValue, MojoValue>> &fields,
                      unsigned captureIdx, StringRef dtypeText, int64_t rank,
                      bool isTile) {
  auto ci = std::to_string(captureIdx);
  if (isTile) {
    // Capture the tensor as one `TileTensor` (built from `mts.to_tile_tensor()`
    // at instantiation) instead of a raw `UnsafePointer` + shape/strides.
    fields.push_back(
        {MojoValue("tile_" + ci),
         TileTensorRef::captureFieldType(dtypeText, "Self.spec_" + ci)});
    return;
  }
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

/// Resolve a scalar-capture operand (e.g. an inlined broadcast constant scalar)
/// of a fused elementwise call. A `!mo.scalar` is splatted to a SIMD vector:
/// Mojo infers the vector `width` from SIMD operands, not from a bare
/// scalar/literal, so a scalar in the `lhs` position would otherwise fail width
/// inference; splatting makes it position-independent. Non-`!mo.scalar`
/// captures are returned unwrapped, matching `resolveStructOperand`'s scalar
/// handling.
static MojoValue resolveScalarCaptureInput(
    Value input, const DenseMap<Value, MojoValue> &valueMap,
    EmitMojoState &state, const ElementwiseBodyRenderer &renderer) {
  auto it = valueMap.find(input);
  ASSERT_STREAM(it != valueMap.end(), << "missing value in body map");
  auto scalarTy = dyn_cast<MO::ScalarType>(input.getType());
  if (!scalarTy || scalarTy.getDtype().isInvalid())
    return it->second;
  return MojoValue(
      llvm::formatv("{0}[{1}, {2}]({3})", kStructSIMD,
                    state.mlirDTypeToMojoType(scalarTy.getDtype()).text,
                    renderer.simdWidthParamName(), it->second.text));
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

  // The slices stay on the input buffer's device.
  auto sliceDeviceRef =
      cast<MGP::DeviceBufferType>(inputBuffer.getType()).getDevice();
  TypedValue<MGP::DeviceContextType> sliceDeviceContext =
      loweringState.getExecuteDevice(sliceDeviceRef);

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

    Value outSlice = opLoweringState
                         .create<MGP::BufferSliceOp>(
                             inputBuffer, offset, sliceSize, sliceDeviceContext)
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
                                     InferredParameters params,
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
    return state.inferDefaultKernelParamValue(*this, paramName);
  };
  auto maybeParams = inferKernelParameters(primDecl, inferParam);
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

/// Convert a `mogg.constant.scalar` value attribute (an IntegerAttr or
/// FloatAttr) to a Mojo literal. Integer and float scalar constants can be
/// emitted directly into a lambda body, letting an elementwise call splat them
/// (e.g. `Add.elementwise[dtype = ...](lhs_simd, 2.0)`). Returns std::nullopt
/// for unsupported attribute kinds.
static std::optional<MojoValue>
scalarConstantAttrToMojoLiteral(TypedAttr attr) {
  if (auto intScalar = MO::tryGetConstantInt(attr))
    return MojoValue(*intScalar);
  // `tryGetConstantFloat` returning a value guarantees `attr` is a FloatAttr.
  auto floatScalar = MO::tryGetConstantFloat(attr);
  if (!floatScalar)
    return std::nullopt;
  auto floatAttr = cast<mlir::FloatAttr>(attr);

  // inf/nan have no Mojo float-literal spelling, so construct them via the
  // numerics helpers parameterized on the scalar's dtype.
  if (std::isinf(*floatScalar) || std::isnan(*floatScalar)) {
    ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(floatAttr.getType());
    if (dtypeOr.isError())
      return std::nullopt;
    MojoValue dtype = EmitMojoState::mlirDTypeToMojoType(*dtypeOr);
    if (std::isnan(*floatScalar))
      return MojoValue(kFnNumericsNan)[{dtype}]();
    MojoValue inf = MojoValue(kFnNumericsInf)[{dtype}]();
    if (std::signbit(*floatScalar))
      return MojoValue("-" + inf.text);
    return inf;
  }

  // Render the finite value as a Mojo float literal (formatting lives in the
  // MojoValue APFloat constructor).
  return MojoValue(floatAttr.getValue());
}

/// Collect struct compile-time parameters and fields from the lambda body.
/// Each tensor capture becomes a StaticTensorSpec parameter and a
/// ManagedTensorSlice field. Non-tensor captures (scalars) are appended
/// after all tensor fields.
static void collectFusionCaptures(
    EmitMojoState &state, Block *lambdaBody, OperandRange fusibleTensors,
    bool isOutputLambda, bool isTile,
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
    addTensorCaptureField(fields, i, dtype, rank, isTile);
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
        addTensorCaptureField(fields, captureIdx, dtype, rank, isTile);
        extraTensorCaptures.push_back({operand, captureIdx});
      } else {
        // If the scalar comes from a constant.scalar op defined outside the
        // lambda, inline its literal value directly instead of capturing it as
        // a runtime field. This lets LLVM emit magic-multiply sequences for
        // non-power-of-2 divisors (e.g. split_dim with new_shape_dim=208)
        // rather than hardware integer division (~40 cycles).
        if (auto constOp = dyn_cast_or_null<MOGG::ConstantScalarOp>(
                operand.getDefiningOp())) {
          if (auto literal =
                  scalarConstantAttrToMojoLiteral(constOp.getValue())) {
            inlineConstants.push_back({operand, *literal});
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

/// Emit a `copier.copy` of the captured tensor `captureIdx`'s tile (at the
/// current `tile_coords`) into a freshly stack-allocated,
/// destination-layout-shaped local `TileTensor`, and return that local tile.
///
/// `layoutSourceName` is the method argument that carries the destination
/// fragment layout: `val` for the compute-output (epilogue) path,
/// `dst` for the pure elementwise path. Used both by the epilogue
/// `LoadTensorOp` handler (extra captures) and by the standalone elementwise
/// block-argument seeding (which has no `LoadTensorOp`s -- the fusible inputs
/// are the block args used directly).
static MojoValue emitTileCopierLoad(MojoBuilder &sdb, unsigned captureIdx,
                                    unsigned loadRank,
                                    llvm::StringRef layoutSourceName) {
  auto captureN = std::to_string(captureIdx);
  TileTensorRef val(MojoValue(layoutSourceName.str()));

  // Local storage for the destination tile. `cache=false`: each captured input
  // needs its own distinct stack allocation. The allocation (and the tile that
  // wraps it) render to identical text across inputs, so the builder's CSE
  // cache would otherwise alias every input onto a single staging tile.
  auto storageVar = sdb.var(TileTensorRef::makeStorage(), /*cache=*/false);

  // Dest TileTensor backed by local storage with the destination layout.
  auto dstVar = sdb.var(TileTensorRef::fromMutStorage(storageVar, val.layout()),
                        /*cache=*/false);

  // Extract the `tile_shape`-sized tile at `tile_coords` from the captured
  // `TileTensor`. This mirrors the driver's own output path
  // (`self.tensor.to_tile_tensor().tile[...](tc)`): the load copier
  // (`GenericToLocalTileCopier[thread_layout=tile_shape]`) `distribute`s this
  // *full* tile across the thread block so each thread stages its own fragment
  // into the per-thread local `dst`. Building a per-thread
  // (`dst.layout`-shaped) src instead leaves the copier's `copy_from` element
  // counts mismatched.
  //
  // The tile shape must equal the driver's `thread_layout`, i.e. the
  // `get_kernel_tile_shape[dtype, "gpu"]()` default the emitter relies on for
  // the `foreach_fusion_tile` call. The tile path is GPU-only (the driver
  // asserts as much), so the target is fixed to "gpu" here.
  MojoValue tileShape = sdb.comptime(MojoValue(kFnGetKernelTileShape)[{
      MojoValue(kParameterDType), MojoValue("gpu").quote()}]());

  // `Coord(Int(tile_coords[0]), ...)` for the tile position.
  SmallVector<MojoValue> coordArgs;
  for (unsigned d = 0; d < loadRank; ++d)
    coordArgs.push_back(
        MojoValue(kStructInt)({}, {MojoValue(kParameterTileCoords)[d]}));
  MojoValue tileCoord = MojoValue(kStructCoord)({}, coordArgs);

  // `self.tile_N.tile[tile_shape[0], ...](tc)` — the captured tensor is stored
  // as a `TileTensor` field, so no `(ptr, shape, strides)` reconstruction is
  // needed.
  SmallVector<MojoValue> tileDims;
  for (unsigned d = 0; d < loadRank; ++d)
    tileDims.push_back(tileShape[d]);
  MojoValue fullTile(std::string("self.tile_") + captureN);
  auto srcVar =
      sdb.var(fullTile.get("tile")[tileDims]({}, {tileCoord}), /*cache=*/false);

  // Copy captured tile from global memory into local dst. The staging `dstVar`
  // is already allocated in `Copier.dst_address_space` (via `makeStorage`), so
  // it matches `Copier.copy`'s `dst` parameter directly; the global-memory src
  // tile must be cast into `Copier.src_address_space` to satisfy the abstract
  // `Copier.copy` signature.
  MojoValue srcCast = TileTensorRef(srcVar).castToSrcAddressSpace();
  sdb.call(MojoValue(kParameterCopier).get("copy"), {}, {dstVar, srcCast});

  // The copier stages into `Copier.dst_address_space`, but the tile
  // `*.elementwise` overload consumes GENERIC / `MutAnyOrigin` tiles. Return a
  // GENERIC view aliasing the *same* staging storage (mirroring the Phase 2
  // reference's `rhs` / `rhs_g` split), so downstream `*.elementwise` operands
  // read the copied data with the layout's default `linear_idx_type`.
  return sdb.var(TileTensorRef::fromGenericStorage(storageVar, val.layout()),
                 /*cache=*/false);
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

      if (lambdaType == KernelOutputFusionLambdaType::ComputeTile ||
          lambdaType == KernelOutputFusionLambdaType::OutputTile ||
          lambdaType == KernelOutputFusionLambdaType::ElementwiseTile) {
        // The layout/dim source is the method's destination-tile argument:
        // `val` for the compute-output / store (epilogue) paths, `dst` for the
        // pure elementwise path (which has no primary `val` output operand).
        llvm::StringRef layoutSrc =
            lambdaType == KernelOutputFusionLambdaType::ElementwiseTile
                ? kParameterDst
                : kParameterVal;
        valueMap[loadOp.getOutputs()] =
            emitTileCopierLoad(sdb, captureIdx, loadRank, layoutSrc);
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
          (lambdaType == KernelOutputFusionLambdaType::ComputeTile ||
           lambdaType == KernelOutputFusionLambdaType::OutputTile ||
           lambdaType == KernelOutputFusionLambdaType::ElementwiseTile) &&
          elemAdaptor.isTileElementwise();

      SmallVector<MojoValue> inputs;
      for (auto input : elemOp.getInputs()) {
        if (useTileElementwise) {
          // Tile path: pass TileTensor values directly without SIMD rebind.
          // Staged input tiles were seeded via `emitTileCopierLoad` as GENERIC
          // views over the copier's staging storage, so they already match the
          // tile `*.elementwise(TileTensor[dtype, LayoutType, MutAnyOrigin])`
          // overload; intermediate `*.elementwise` results are likewise
          // GENERIC.
          auto it = valueMap.find(input);
          ASSERT_STREAM(it != valueMap.end(),
                        << "missing tile value in body map");
          inputs.push_back(it->second);
        } else if (scalarCaptureValues.contains(input)) {
          inputs.push_back(
              resolveScalarCaptureInput(input, valueMap, state, renderer));
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
      auto maybeParams =
          inferKernelParameters(elemAdaptor, elemFuncLabel, inferParam);
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
        return state.inferDefaultKernelParamValue(callOp, paramName);
      };
      auto maybeParams = inferKernelParameters(affinePrimDecl, inferParam);

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

  // Tile-based compute / elementwise fusion: return the result TileTensor
  // directly.
  if (lambdaType == KernelOutputFusionLambdaType::ComputeTile ||
      lambdaType == KernelOutputFusionLambdaType::ElementwiseTile) {
    sdb.returnsValue(resultValue);
    return;
  }

  // Tile-based store fusion (OutputTile): the fusion owns the store.
  if (lambdaType == KernelOutputFusionLambdaType::OutputTile) {
    auto tensorTy = captureTensorTypes[0];
    auto rank = tensorTy.getRank() == 0 ? 1 : tensorTy.getRank();

    MojoValue tileShape = sdb.comptime(MojoValue(kFnGetKernelTileShape)[{
        MojoValue(kParameterDType), MojoValue("gpu").quote()}]());

    SmallVector<MojoValue> coordArgs;
    SmallVector<MojoValue> tileDims;
    for (unsigned d = 0; d < rank; ++d) {
      coordArgs.push_back(
          MojoValue(kStructInt)({}, {MojoValue(kParameterTileCoords)[d]}));
      tileDims.push_back(tileShape[d]);
    }
    MojoValue tileCoord = MojoValue(kStructCoord)({}, coordArgs);

    // The output tensor is stored as a `TileTensor` field (`self.tile_0`);
    // extract the destination tile at `tile_coords`.
    MojoValue outTile =
        MojoValue("self.tile_0").get("tile")[tileDims]({}, {tileCoord});

    MojoValue dstSpace = MojoValue(kParameterStoreCopierType)
                             .get(TileTensorRef::kDstAddressSpace);
    MojoValue srcSpace = MojoValue(kParameterStoreCopierType)
                             .get(TileTensorRef::kSrcAddressSpace);
    MojoValue dstCast =
        outTile.get(TileTensorRef::kAddressSpaceCast)[dstSpace]();
    MojoValue srcCast =
        resultValue.get(TileTensorRef::kAddressSpaceCast)[srcSpace]();

    sdb.call(MojoValue(kParameterStoreCopier).get("copy"), {},
             {dstCast, srcCast});
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

  // For tile compute/elementwise fusion, the position parameter is tile_coords;
  // for SIMD compute/output fusion it is idx.
  bool isTile = lambdaType == KernelOutputFusionLambdaType::ComputeTile ||
                lambdaType == KernelOutputFusionLambdaType::OutputTile ||
                lambdaType == KernelOutputFusionLambdaType::ElementwiseTile;
  valueMap[lambdaBody->getArguments().back()] =
      MojoValue(isTile ? kParameterTileCoords : kParameterIdx);

  for (unsigned i = 0; i < fusedInputs.size(); ++i) {
    Value blockArg = lambdaBody->getArgument(i);
    if (i == 0 && isOutputLambda) {
      valueMap[blockArg] = MojoValue("val");
      continue;
    }

    if (isTile) {
      // Standalone tile elementwise (ElementwiseTile) has no LoadTensorOps --
      // the fusible inputs are the block args used directly by the elementwise
      // op. Seed each with a copier.copy into a `dst.layout`-shaped local tile.
      // The compute-output tile (ComputeTile) path instead resolves its extra
      // captures lazily via the LoadTensorOp handler in
      // emitElementwiseLambdaBodyOps.
      if (lambdaType == KernelOutputFusionLambdaType::ElementwiseTile) {
        auto tensorTy = captureTensorTypes[i];
        auto rank = tensorTy.getRankNonZero();
        valueMap[blockArg] = emitTileCopierLoad(sdb, i, rank, kParameterDst);
      }
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
    traitName = lambdaType == KernelOutputFusionLambdaType::ElementwiseTile
                    ? kTraitElementwiseFusionTile
                    : kTraitElementwiseFusion;
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
  } else if (lambdaType == KernelOutputFusionLambdaType::OutputTile) {
    traitName = kTraitOutputFusionTile;
    methodName = "store";
  } else {
    traitName = kTraitComputeOutputFusion;
    methodName = "compute";
    methodReturnsValue = true;
  }

  // Tile-based fusion structs capture each tensor as one `TileTensor` field
  // instead of decomposed `(ptr, shape, strides)`.
  bool isTile = lambdaType == KernelOutputFusionLambdaType::ComputeTile ||
                lambdaType == KernelOutputFusionLambdaType::OutputTile ||
                lambdaType == KernelOutputFusionLambdaType::ElementwiseTile;

  // Collect struct compile-time parameters, fields, and external captures.
  SmallVector<std::pair<MojoValue, MojoValue>> parameters;
  SmallVector<std::pair<MojoValue, MojoValue>> fields;
  SmallVector<MO::TensorType> captureTensorTypes;
  SmallVector<std::pair<Value, unsigned>> extraTensorCaptures;
  SmallVector<std::tuple<Value, std::string, std::string>> scalarCaptures;
  SmallVector<std::pair<Value, MojoValue>> inlineConstants;
  collectFusionCaptures(state, lambdaBody, fusibleTensors, isOutputLambda,
                        isTile, parameters, fields, captureTensorTypes,
                        extraTensorCaptures, scalarCaptures, inlineConstants);

  // Emit struct definition with trait-conforming method.
  auto structName = state.structDefsBuilder.getIdentifier("FusionStruct");
  MojoValue structNameVal = state.structDefsBuilder.newVar(structName);
  auto &sdb = state.structDefsBuilder;
  const auto simdWidthType = isOutputLambda ? kStructSIMDLength : kStructInt;
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

        if (lambdaType == KernelOutputFusionLambdaType::ComputeTile ||
            lambdaType == KernelOutputFusionLambdaType::OutputTile ||
            lambdaType == KernelOutputFusionLambdaType::ElementwiseTile) {
          MojoValue tileTensorType = MojoValue(kStructTileTensor)[{
              kParameterDType, kParameterLayoutType, kComptimeMutAnyOrigin}];
          // The tile-signature is identical across the compute-output, store
          // (epilogue) and pure elementwise paths; only the third method
          // argument differs. The compute-output and store (epilogue) paths
          // bind the primary output tile as `val`; the pure elementwise path
          // receives the destination fragment as `dst` (all inputs flow
          // through the copier).
          StringRef thirdArgName =
              lambdaType == KernelOutputFusionLambdaType::ElementwiseTile
                  ? kParameterDst
                  : kParameterVal;
          bool isStoreTile =
              lambdaType == KernelOutputFusionLambdaType::OutputTile;
          methodParameters = {
              NamedMojoDef::asVerbatim(kParameterDType, kStructDType),
              NamedMojoDef::asVerbatim(kParameterRank, kStructInt),
              NamedMojoDef::asVerbatim(kParameterLayoutType,
                                       kStructTensorLayout),
              // Comptime copier *type* param named `Copier`; the runtime arg
              // below is `copier: Copier`. The two names must differ or the
              // emitted method redefines `copier`.
              NamedMojoDef::asVerbatim(kParameterCopierType, kStructTileCopier),
          };
          // The store path takes a second, copier that writes the output tile.
          if (isStoreTile)
            methodParameters.push_back(NamedMojoDef::asVerbatim(
                kParameterStoreCopierType, kStructTileCopier));
          methodArgs.push_back(NamedMojoDef::asVerbatim(kParameterSelf));
          methodArgs.push_back(
              NamedMojoDef::asVerbatim(kParameterTileCoords, idxType));
          methodArgs.push_back(
              NamedMojoDef::asVerbatim(kParameterCopier, kParameterCopierType));
          if (isStoreTile)
            methodArgs.push_back(NamedMojoDef::asVerbatim(
                kParameterStoreCopier, kParameterStoreCopierType));
          methodArgs.push_back(
              NamedMojoDef::asVerbatim(thirdArgName, tileTensorType));
          if (methodReturnsValue)
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

  // Emit struct instantiation: pass static_spec as a compile-time parameter and
  // the tensor as either one `TileTensor` (tile path) or decomposed
  // (ptr, shape, strides) (SIMD path).
  SmallVector<MojoValue> comptimeArgs;
  SmallVector<MojoValue> instanceArgs;
  auto pushCaptureInstance = [&](ManagedTensorSliceRef mts) {
    comptimeArgs.push_back(mts.staticSpec());
    if (isTile) {
      instanceArgs.push_back(state.builder.var(mts.toTileTensor()));
      return;
    }
    instanceArgs.push_back(state.builder.var(mts.unsafePtr()));
    instanceArgs.push_back(state.builder.var(mts.shape()));
    instanceArgs.push_back(state.builder.var(mts.strides()));
  };
  for (unsigned i = 0; i < fusedInputs.size(); ++i)
    pushCaptureInstance(ManagedTensorSliceRef(fusedInputs[i]));
  for (auto &[tensorVal, captureIdx] : extraTensorCaptures)
    pushCaptureInstance(
        ManagedTensorSliceRef(state.getMojoValueFor(tensorVal)));
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
        state.registry, fusedOp, state.enableComputeLambdas, state.moduleOp);
    if (!outputLambdaType.isError() &&
        *outputLambdaType == KernelOutputFusionLambdaType::Output) {
      return ManagedTensorSliceRef::kBindToFusedOutput;
    }
    if (!outputLambdaType.isError() &&
        *outputLambdaType == KernelOutputFusionLambdaType::ComputeTile) {
      return ManagedTensorSliceRef::kBindToFusedComputeOutputTile;
    }
    if (!outputLambdaType.isError() &&
        *outputLambdaType == KernelOutputFusionLambdaType::OutputTile) {
      return ManagedTensorSliceRef::kBindToFusedOutputTile;
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

/// Read a trait contract for a fused operand by navigating to the kernel decl
/// (the source of truth; robust to prologue-fusion rewriting the bind op).
static std::optional<TensorArgContract>
inputLoadFusionContract(EmitMojoState &state, MOGG::TensorFusedOp fusedOp) {
  auto bindOp = fusedOp.getBindLambdaOp();
  if (!bindOp || bindOp.getBindKind() != BindLambdaKind::InputLambda)
    return std::nullopt;
  Value fusedTensor = fusedOp.getResult();
  if (!fusedTensor.hasOneUse())
    return std::nullopt;
  auto executeOp =
      dyn_cast<MOGG::CallExecuteOp>(*fusedTensor.getUsers().begin());
  if (!executeOp)
    return std::nullopt;
  std::optional<unsigned> inputIdx;
  for (auto [idx, inVal] : llvm::enumerate(executeOp.getInputs())) {
    if (inVal == fusedTensor) {
      inputIdx = idx;
      break;
    }
  }
  if (!inputIdx)
    return std::nullopt;
  KernelDeclAdaptor adaptor(
      state.registry.lookupKernelBySymName(executeOp.getKernelSymName()));
  if (!adaptor.executeFunction.has_value())
    return std::nullopt;
  const auto &inArgs = adaptor.executeFunction->inputArguments;
  if (*inputIdx >= inArgs.size())
    return std::nullopt;
  return inArgs[*inputIdx].getTraitContract();
}

/// Comptime `mut` for a trait fusion capture. `_MoggDenseTensorArg` exposes
/// `.mut`; `_MoggTensorBufferCapture` does not -- derive mutability from the
/// MO tensor type stamped on the stub block argument during lowering.
static MojoValue traitCaptureMutComptimeExpr(Value captureVal,
                                             StringRef mojoCapName) {
  if (auto tensorTy = MO::dynCastFromTensorLike(captureVal.getType())) {
    if (auto mutAttr = tensorTy.getIsMutableAttr())
      return MojoValue(mutAttr.getValue());
    if (auto ioAttr = tensorTy.getIOKindAttr())
      return MojoValue(ioAttr.getValue() == MO::IOKind::Output);
  }
  return MojoValue(llvm::formatv("type_of({0}).mut", mojoCapName).str());
}

/// Emit a trait-lane input-load fusion FUNCTOR for a trait `mogg.bind` input
/// lambda (Phase 5). Modeled on the mockup `AddTwoInputLoadFusion`: the functor
/// conforms to the inline `InputLoadFusion` trait, captures each producer
/// source tensor as `(ptr, layout)`, and its `__call__` replays the producer
/// body via dense `TileTensor(...).load(idx)` (the `DenseFusionRenderer`) -- NO
/// `ManagedTensorSlice`, `simd_load_from_tensor_pointer`, or
/// `_bind_to_fused_*`. The instance is registered via
/// `setFusionStructLoweringFor` so `TensorFusedOp::emitMojo` wraps it in
/// `_MoggInputLoadFusionTensor`.
static LogicalResult
emitInputLoadFusionFunctor(BindLambdaOp bindOp, EmitMojoState &state,
                           Block *lambdaBody, OperandRange fusibleTensors,
                           ArrayRef<MojoValue> fusedInputs) {
  if (failed(verifyBindLambdaBodyOpsForMojoEmission(bindOp, lambdaBody)))
    return failure();

  // Emit the inline trait + wrapper defs once per module.
  if (!state.emittedInputLoadFusionArgDefs) {
    state.structDefsBuilder.verbatim(MojoValue(kInputLoadFusionDefs.str()));
    state.emittedInputLoadFusionArgDefs = true;
  }

  // Reuse the legacy collector for *which* values are captured; discard its
  // MTS-shaped params/fields and build dense `(dtype_i, LayoutType_i)` params +
  // `(ptr_i, layout_i)` fields instead.
  SmallVector<std::pair<MojoValue, MojoValue>> mtsParams, mtsFields;
  SmallVector<MO::TensorType> captureTensorTypes;
  SmallVector<std::pair<Value, unsigned>> extraTensorCaptures;
  SmallVector<std::tuple<Value, std::string, std::string>> scalarCaptures;
  SmallVector<std::pair<Value, MojoValue>> inlineConstants;
  collectFusionCaptures(state, lambdaBody, fusibleTensors,
                        /*isOutputLambda=*/false, /*isTile=*/false, mtsParams,
                        mtsFields, captureTensorTypes, extraTensorCaptures,
                        scalarCaptures, inlineConstants);

  // Each tensor capture is the `TileTensor` projection of its (dense) source --
  // `TileTensor` is `TrivialRegisterPassable`, so the functor is too. Captured
  // purely through the trait surface: no raw ptr / `.handle`. The projection
  // origin's `mut` is a struct param (the generic stub arg's `.mut` is symbolic
  // and won't fold to a literal, so it cannot be hardcoded in the field type).
  SmallVector<std::pair<MojoValue, MojoValue>> parameters;
  SmallVector<std::pair<MojoValue, MojoValue>> fields;
  for (unsigned i = 0; i < captureTensorTypes.size(); ++i) {
    auto ci = std::to_string(i);
    parameters.push_back({MojoValue("dtype_" + ci), MojoValue(kStructDType)});
    parameters.push_back(
        {MojoValue("LayoutType_" + ci), MojoValue(kStructTensorLayout)});
    parameters.push_back({MojoValue("mut_" + ci), MojoValue(kStructBool)});
    // Capture the source as a bare pointer + layout (NOT a `TileTensor`): a
    // `TileTensor` field would not translate to a device address (the encoder
    // bit-copies it), whereas a bare `UnsafePointer` translates through its own
    // `_to_device_type`. The functor rebuilds the dense tile in `__call__`.
    fields.push_back(
        {MojoValue("ptr_" + ci),
         MojoValue(llvm::formatv("UnsafePointer[Scalar[Self.dtype_{0}], "
                                 "UntrackedOrigin[mut=Self.mut_{0}]]",
                                 ci)
                       .str())});
    fields.push_back(
        {MojoValue("layout_" + ci),
         MojoValue(llvm::formatv("Self.LayoutType_{0}", ci).str())});
  }
  for (auto &[val, fieldName, mojoType] : scalarCaptures)
    fields.push_back({MojoValue(fieldName), MojoValue(mojoType)});

  auto structName = state.structDefsBuilder.getIdentifier("InputLoadFunctor");
  MojoValue structNameVal = state.structDefsBuilder.newVar(structName);
  auto &sdb = state.structDefsBuilder;
  DenseFusionRenderer renderer;
  sdb.structDecl(
      structNameVal, parameters,
      {MojoValue("InputLoadFusion"), MojoValue("TrivialRegisterPassable")},
      fields, [&]() {
        MojoValue simdType = MojoValue(
            kStructSIMD)[{MojoValue(kParameterDType), MojoValue("width")}];
        SmallVector<NamedMojoDef> methodParameters = {
            NamedMojoDef::asVerbatim(kParameterDType, kStructDType),
            NamedMojoDef::asVerbatim(kParameterRank, kStructInt),
            NamedMojoDef::asVerbatim("width", kStructInt),
            NamedMojoDef::asVerbatim(kParameterElementAlignment, kStructInt,
                                     MojoValue(1)),
        };
        SmallVector<NamedMojoDef> methodArgs = {
            NamedMojoDef::asVerbatim(kParameterSelf),
            NamedMojoDef::asVerbatim(kParameterIdx, MojoValue(kStructCoord)),
        };
        sdb.def("__call__", methodParameters, methodArgs, simdType,
                /*exported=*/false, /*raises=*/false,
                /*isAlwaysInline=*/true, [&](const MOGG::FunctionBody &) {
                  DenseMap<Value, MojoValue> valueMap;
                  DenseSet<Value> scalarCaptureValues;
                  seedScalarValueMapEntries(
                      inlineConstants, scalarCaptures,
                      [](Value, StringRef fieldName) {
                        return MojoValue("self." + fieldName.str());
                      },
                      valueMap, scalarCaptureValues);
                  valueMap[lambdaBody->getArguments().back()] =
                      MojoValue(kParameterIdx);
                  for (unsigned i = 0; i < fusedInputs.size(); ++i)
                    valueMap[lambdaBody->getArgument(i)] =
                        sdb.var(MojoValue(renderer.buildLoadExpr(i, "")));
                  emitElementwiseLambdaBodyOps(
                      state, sdb, lambdaBody, captureTensorTypes,
                      extraTensorCaptures, valueMap, scalarCaptureValues,
                      renderer, KernelOutputFusionLambdaType::Compute);
                  auto outputOp =
                      cast<MOGG::OutputOp>(lambdaBody->getTerminator());
                  auto resultIt = valueMap.find(outputOp.getOperand(0));
                  ASSERT_STREAM(resultIt != valueMap.end(),
                                << "missing output value in functor body map");
                  sdb.returnsValue(MojoValue(
                      llvm::formatv("rebind[{0}[{1}, width]]({2})", kStructSIMD,
                                    kParameterDType, resultIt->second.text)
                          .str()));
                });
      });

  // Instantiate the functor: per capture, parameters `(dtype, LayoutType)` and
  // fields `(ptr, layout)` read off the dense source arg (`.handle` /
  // `.layout()` on `_MoggDenseTensorArg` -- no MTS decomposition). Order
  // matches the params: primary captures (`fusedInputs`), then extra captures,
  // then scalars.
  SmallVector<MojoValue> comptimeArgs;
  SmallVector<MojoValue> instanceArgs;
  auto pushTensorCapture = [&](StringRef cap, Value mlirCapture) {
    comptimeArgs.push_back(
        MojoValue(llvm::formatv("type_of({0}).dtype", cap).str()));
    comptimeArgs.push_back(
        MojoValue(llvm::formatv("type_of({0}).LayoutType", cap).str()));
    comptimeArgs.push_back(traitCaptureMutComptimeExpr(mlirCapture, cap));
    // Capture the source as a bare pointer + layout off the dense trait arg's
    // stored handle (the device representation that translates), mirroring the
    // dense raw-pointer path. The functor rebuilds the dense tile in
    // `__call__`.
    instanceArgs.push_back(
        state.builder.var(MojoValue(llvm::formatv("{0}.handle", cap).str())));
    instanceArgs.push_back(
        state.builder.var(MojoValue(llvm::formatv("{0}.layout()", cap).str())));
  };
  for (auto [i, cap] : llvm::enumerate(fusedInputs))
    pushTensorCapture(cap.text, fusibleTensors[i]);
  for (auto &[tensorVal, captureIdx] : extraTensorCaptures)
    pushTensorCapture(state.getMojoValueFor(tensorVal).text, tensorVal);
  for (auto &[val, fieldName, mojoType] : scalarCaptures)
    instanceArgs.push_back(state.getMojoValueFor(val));

  MojoValue functorInstance =
      state.builder.var(structNameVal(comptimeArgs, instanceArgs));
  MojoValue concreteTypeName = structNameVal[comptimeArgs];
  // Empty bind method: TensorFusedOp wraps the functor in
  // `_MoggInputLoadFusionTensor` rather than calling a method on a base slice.
  state.setFusionStructLoweringFor(bindOp.getOutput(), functorInstance,
                                   /*bindMethod=*/"", concreteTypeName);
  return success();
}

/// If `fusedOp`'s output lambda feeds a trait kernel's fused-STORE operand,
/// return that operand's distilled contract. Navigates `TensorFusedOp ->
/// CallExecuteOp -> output operand index -> kernel-decl output arg` (mirrors
/// `getKernelOutputFusionLambdaType`); the kernel decl is the source of truth.
static std::optional<TensorArgContract>
outputFusionContract(EmitMojoState &state, MOGG::TensorFusedOp fusedOp) {
  auto bindOp = fusedOp.getBindLambdaOp();
  if (!bindOp || bindOp.getBindKind() != BindLambdaKind::OutputLambda)
    return std::nullopt;
  Value fusedTensor = fusedOp.getResult();
  if (!fusedTensor.hasOneUse())
    return std::nullopt;
  auto executeOp =
      dyn_cast<MOGG::CallExecuteOp>(*fusedTensor.getUsers().begin());
  if (!executeOp)
    return std::nullopt;
  std::optional<unsigned> outputIdx;
  for (auto [idx, outVal] : llvm::enumerate(executeOp.getOutputs())) {
    if (outVal == fusedTensor) {
      outputIdx = idx;
      break;
    }
  }
  if (!outputIdx)
    return std::nullopt;
  KernelDeclAdaptor adaptor(
      state.registry.lookupKernelBySymName(executeOp.getKernelSymName()));
  if (!adaptor.executeFunction.has_value())
    return std::nullopt;
  const auto &outArgs = adaptor.executeFunction->outputArguments;
  if (*outputIdx >= outArgs.size())
    return std::nullopt;
  return outArgs[*outputIdx].getTraitContract();
}

/// Emit a trait-lane output-store fusion FUNCTOR for a trait `mogg.bind` output
/// lambda (Phase 5). Epilogue counterpart of `emitInputLoadFusionFunctor`: the
/// functor conforms to `OutputStoreFusion`, captures the output's MUTABLE
/// `TileTensor` (capture 0 = the store target) plus any extra input tiles, and
/// its `__call__(idx, value)` runs the epilogue then writes via
/// `tile.store(idx, ...)` -- no MTS / `simd_store_into_tensor_pointer`.
static LogicalResult
emitOutputStoreFusionFunctor(BindLambdaOp bindOp, EmitMojoState &state,
                             Block *lambdaBody, OperandRange fusibleTensors,
                             ArrayRef<MojoValue> fusedInputs) {
  if (failed(verifyBindLambdaBodyOpsForMojoEmission(bindOp, lambdaBody)))
    return failure();

  state.emitOutputStoreFusionDefsOnce();

  SmallVector<std::pair<MojoValue, MojoValue>> mtsParams, mtsFields;
  SmallVector<MO::TensorType> captureTensorTypes;
  SmallVector<std::pair<Value, unsigned>> extraTensorCaptures;
  SmallVector<std::tuple<Value, std::string, std::string>> scalarCaptures;
  SmallVector<std::pair<Value, MojoValue>> inlineConstants;
  collectFusionCaptures(state, lambdaBody, fusibleTensors,
                        /*isOutputLambda=*/true, /*isTile=*/false, mtsParams,
                        mtsFields, captureTensorTypes, extraTensorCaptures,
                        scalarCaptures, inlineConstants);

  // Capture 0 is the output (the store target); the rest are input tiles. All
  // captured as `TileTensor` projections via the trait surface. The projection
  // origin's `mut` is a struct param per capture (capture 0 resolves to mutable
  // for an Output source, the inputs to immutable) -- the generic stub arg's
  // `.mut` is symbolic and won't fold to a literal in the field type.
  SmallVector<std::pair<MojoValue, MojoValue>> parameters;
  SmallVector<std::pair<MojoValue, MojoValue>> fields;
  for (unsigned i = 0; i < captureTensorTypes.size(); ++i) {
    auto ci = std::to_string(i);
    parameters.push_back({MojoValue("dtype_" + ci), MojoValue(kStructDType)});
    parameters.push_back(
        {MojoValue("LayoutType_" + ci), MojoValue(kStructTensorLayout)});
    parameters.push_back({MojoValue("mut_" + ci), MojoValue(kStructBool)});
    // Capture the source as a bare pointer + layout (NOT a `TileTensor`): a
    // `TileTensor` field would not translate to a device address (the encoder
    // bit-copies it), whereas a bare `UnsafePointer` translates through its own
    // `_to_device_type`. The functor rebuilds the dense tile in `__call__`.
    fields.push_back(
        {MojoValue("ptr_" + ci),
         MojoValue(llvm::formatv("UnsafePointer[Scalar[Self.dtype_{0}], "
                                 "UntrackedOrigin[mut=Self.mut_{0}]]",
                                 ci)
                       .str())});
    fields.push_back(
        {MojoValue("layout_" + ci),
         MojoValue(llvm::formatv("Self.LayoutType_{0}", ci).str())});
  }
  for (auto &[val, fieldName, mojoType] : scalarCaptures)
    fields.push_back({MojoValue(fieldName), MojoValue(mojoType)});

  auto structName = state.structDefsBuilder.getIdentifier("OutputStoreFunctor");
  MojoValue structNameVal = state.structDefsBuilder.newVar(structName);
  auto &sdb = state.structDefsBuilder;
  DenseFusionRenderer renderer;
  sdb.structDecl(
      structNameVal, parameters,
      {MojoValue("OutputStoreFusion"), MojoValue("TrivialRegisterPassable")},
      fields, [&]() {
        MojoValue simdType = MojoValue(
            kStructSIMD)[{MojoValue(kParameterDType), MojoValue("width")}];
        SmallVector<NamedMojoDef> methodParameters = {
            NamedMojoDef::asVerbatim(kParameterDType, kStructDType),
            NamedMojoDef::asVerbatim(kParameterRank, kStructInt),
            NamedMojoDef::asVerbatim("width", kStructInt),
            NamedMojoDef::asVerbatim(kParameterElementAlignment, kStructInt,
                                     MojoValue(1)),
        };
        SmallVector<NamedMojoDef> methodArgs = {
            NamedMojoDef::asVerbatim(kParameterSelf),
            NamedMojoDef::asVerbatim(kParameterIdx, MojoValue(kStructCoord)),
            NamedMojoDef::asVerbatim("value", simdType),
        };
        sdb.def(
            "__call__", methodParameters, methodArgs,
            /*returnType=*/std::optional<MojoValue>{},
            /*exported=*/false, /*raises=*/false,
            /*isAlwaysInline=*/true, [&](const MOGG::FunctionBody &) {
              DenseMap<Value, MojoValue> valueMap;
              DenseSet<Value> scalarCaptureValues;
              seedScalarValueMapEntries(
                  inlineConstants, scalarCaptures,
                  [](Value, StringRef fieldName) {
                    return MojoValue("self." + fieldName.str());
                  },
                  valueMap, scalarCaptureValues);
              valueMap[lambdaBody->getArguments().back()] =
                  MojoValue(kParameterIdx);
              // Block arg 0 is the value to store (the kernel's computed
              // element); captures i>0 are loaded input tiles.
              valueMap[lambdaBody->getArgument(0)] = MojoValue("value");
              for (unsigned i = 1; i < fusedInputs.size(); ++i)
                valueMap[lambdaBody->getArgument(i)] =
                    sdb.var(MojoValue(renderer.buildLoadExpr(i, "")));
              emitElementwiseLambdaBodyOps(
                  state, sdb, lambdaBody, captureTensorTypes,
                  extraTensorCaptures, valueMap, scalarCaptureValues, renderer,
                  KernelOutputFusionLambdaType::Compute);
              auto outputOp = cast<MOGG::OutputOp>(lambdaBody->getTerminator());
              auto resultIt = valueMap.find(outputOp.getOperand(0));
              ASSERT_STREAM(resultIt != valueMap.end(),
                            << "missing store value in functor body map");
              // `TileTensor.store` needs evidence that the projection is
              // mutable; capture 0 is the Output, so `mut_0` is True. The
              // assert refines the symbolic param so the store type-checks
              // (same pattern as the dense `raw_store` path).
              sdb.verbatim(MojoValue(
                  "comptime assert Self.mut_0, \"fused store target must be "
                  "mutable\""));
              // Rebuild the output tile from the captured bare pointer + layout
              // (capture 0) and store through it.
              sdb.verbatim(MojoValue(
                  llvm::formatv(
                      "TileTensor[dtype=Self.dtype_0, "
                      "LayoutType=Self.LayoutType_0, "
                      "origin=UntrackedOrigin[mut=Self.mut_0]](ptr=self.ptr_0, "
                      "layout=self.layout_0).store[width=width, "
                      "alignment=element_alignment]({0}, "
                      "rebind[{1}[Self.dtype_0, width]]({2}))",
                      kParameterIdx, kStructSIMD, resultIt->second.text)
                      .str()));
            });
      });

  SmallVector<MojoValue> comptimeArgs;
  SmallVector<MojoValue> instanceArgs;
  auto pushTensorCapture = [&](StringRef cap, Value mlirCapture) {
    comptimeArgs.push_back(
        MojoValue(llvm::formatv("type_of({0}).dtype", cap).str()));
    comptimeArgs.push_back(
        MojoValue(llvm::formatv("type_of({0}).LayoutType", cap).str()));
    comptimeArgs.push_back(traitCaptureMutComptimeExpr(mlirCapture, cap));
    // Bare pointer + layout off the source's stored handle (the device
    // representation that translates), mirroring the dense raw-pointer path.
    instanceArgs.push_back(
        state.builder.var(MojoValue(llvm::formatv("{0}.handle", cap).str())));
    instanceArgs.push_back(
        state.builder.var(MojoValue(llvm::formatv("{0}.layout()", cap).str())));
  };
  for (auto [i, cap] : llvm::enumerate(fusedInputs))
    pushTensorCapture(cap.text, fusibleTensors[i]);
  for (auto &[tensorVal, captureIdx] : extraTensorCaptures)
    pushTensorCapture(state.getMojoValueFor(tensorVal).text, tensorVal);
  for (auto &[val, fieldName, mojoType] : scalarCaptures)
    instanceArgs.push_back(state.getMojoValueFor(val));

  MojoValue functorInstance =
      state.builder.var(structNameVal(comptimeArgs, instanceArgs));
  MojoValue concreteTypeName = structNameVal[comptimeArgs];
  state.setFusionStructLoweringFor(bindOp.getOutput(), functorInstance,
                                   /*bindMethod=*/"", concreteTypeName);
  return success();
}

/// Emit a trait-lane output-TRANSFORM (compute-output) fusion FUNCTOR for a
/// trait `mogg.bind` output lambda (Phase 5). Unlike store, the transform
/// functor is pure value->value: its `__call__(idx, value)` runs the epilogue
/// and RETURNS the transformed value (the parent kernel performs the store via
/// the wrapper's `to_tile_tensor`). It does NOT capture the output (capture 0)
/// -- only the epilogue's extra input tiles.
static LogicalResult
emitTransformFusionFunctor(BindLambdaOp bindOp, EmitMojoState &state,
                           Block *lambdaBody, OperandRange fusibleTensors,
                           ArrayRef<MojoValue> fusedInputs) {
  if (failed(verifyBindLambdaBodyOpsForMojoEmission(bindOp, lambdaBody)))
    return failure();

  if (!state.emittedTransformFusionArgDefs) {
    state.structDefsBuilder.verbatim(MojoValue(kTransformFusionDefs.str()));
    state.emittedTransformFusionArgDefs = true;
  }

  SmallVector<std::pair<MojoValue, MojoValue>> mtsParams, mtsFields;
  SmallVector<MO::TensorType> captureTensorTypes;
  SmallVector<std::pair<Value, unsigned>> extraTensorCaptures;
  SmallVector<std::tuple<Value, std::string, std::string>> scalarCaptures;
  SmallVector<std::pair<Value, MojoValue>> inlineConstants;
  collectFusionCaptures(state, lambdaBody, fusibleTensors,
                        /*isOutputLambda=*/true, /*isTile=*/false, mtsParams,
                        mtsFields, captureTensorTypes, extraTensorCaptures,
                        scalarCaptures, inlineConstants);

  // Capture 0 is the output -- NOT captured here (the wrapper holds the output
  // buffer; the functor only transforms). Captures i>0 are the epilogue's input
  // tiles (origin mut is a per-capture struct param, as the generic stub arg's
  // `.mut` is symbolic).
  SmallVector<std::pair<MojoValue, MojoValue>> parameters;
  SmallVector<std::pair<MojoValue, MojoValue>> fields;
  for (unsigned i = 1; i < captureTensorTypes.size(); ++i) {
    auto ci = std::to_string(i);
    parameters.push_back({MojoValue("dtype_" + ci), MojoValue(kStructDType)});
    parameters.push_back(
        {MojoValue("LayoutType_" + ci), MojoValue(kStructTensorLayout)});
    parameters.push_back({MojoValue("mut_" + ci), MojoValue(kStructBool)});
    // Capture the source as a bare pointer + layout (NOT a `TileTensor`): a
    // `TileTensor` field would not translate to a device address (the encoder
    // bit-copies it), whereas a bare `UnsafePointer` translates through its own
    // `_to_device_type`. The functor rebuilds the dense tile in `__call__`.
    fields.push_back(
        {MojoValue("ptr_" + ci),
         MojoValue(llvm::formatv("UnsafePointer[Scalar[Self.dtype_{0}], "
                                 "UntrackedOrigin[mut=Self.mut_{0}]]",
                                 ci)
                       .str())});
    fields.push_back(
        {MojoValue("layout_" + ci),
         MojoValue(llvm::formatv("Self.LayoutType_{0}", ci).str())});
  }
  for (auto &[val, fieldName, mojoType] : scalarCaptures)
    fields.push_back({MojoValue(fieldName), MojoValue(mojoType)});

  auto structName = state.structDefsBuilder.getIdentifier("TransformFunctor");
  MojoValue structNameVal = state.structDefsBuilder.newVar(structName);
  auto &sdb = state.structDefsBuilder;
  DenseFusionRenderer renderer;
  sdb.structDecl(
      structNameVal, parameters,
      {MojoValue("TransformFusion"), MojoValue("TrivialRegisterPassable")},
      fields, [&]() {
        MojoValue simdType = MojoValue(
            kStructSIMD)[{MojoValue(kParameterDType), MojoValue("width")}];
        SmallVector<NamedMojoDef> methodParameters = {
            NamedMojoDef::asVerbatim(kParameterDType, kStructDType),
            NamedMojoDef::asVerbatim(kParameterRank, kStructInt),
            NamedMojoDef::asVerbatim("width", kStructInt),
            NamedMojoDef::asVerbatim(kParameterElementAlignment, kStructInt,
                                     MojoValue(1)),
        };
        SmallVector<NamedMojoDef> methodArgs = {
            NamedMojoDef::asVerbatim(kParameterSelf),
            NamedMojoDef::asVerbatim(kParameterIdx, MojoValue(kStructCoord)),
            NamedMojoDef::asVerbatim("value", simdType),
        };
        sdb.def(
            "__call__", methodParameters, methodArgs, simdType,
            /*exported=*/false, /*raises=*/false,
            /*isAlwaysInline=*/true, [&](const MOGG::FunctionBody &) {
              DenseMap<Value, MojoValue> valueMap;
              DenseSet<Value> scalarCaptureValues;
              seedScalarValueMapEntries(
                  inlineConstants, scalarCaptures,
                  [](Value, StringRef fieldName) {
                    return MojoValue("self." + fieldName.str());
                  },
                  valueMap, scalarCaptureValues);
              valueMap[lambdaBody->getArguments().back()] =
                  MojoValue(kParameterIdx);
              // Block arg 0 is the value to transform; captures i>0 are
              // loaded epilogue input tiles.
              valueMap[lambdaBody->getArgument(0)] = MojoValue("value");
              for (unsigned i = 1; i < fusedInputs.size(); ++i)
                valueMap[lambdaBody->getArgument(i)] =
                    sdb.var(MojoValue(renderer.buildLoadExpr(i, "")));
              emitElementwiseLambdaBodyOps(
                  state, sdb, lambdaBody, captureTensorTypes,
                  extraTensorCaptures, valueMap, scalarCaptureValues, renderer,
                  KernelOutputFusionLambdaType::Compute);
              auto outputOp = cast<MOGG::OutputOp>(lambdaBody->getTerminator());
              auto resultIt = valueMap.find(outputOp.getOperand(0));
              ASSERT_STREAM(resultIt != valueMap.end(),
                            << "missing transform value in functor body map");
              // Pure value -> value: return the transformed value; the
              // parent stores it through the wrapper's `to_tile_tensor`.
              sdb.returnsValue(MojoValue(
                  llvm::formatv("rebind[{0}[{1}, width]]({2})", kStructSIMD,
                                kParameterDType, resultIt->second.text)
                      .str()));
            });
      });

  // Instantiate: skip the output (capture 0); push the epilogue's input tiles.
  SmallVector<MojoValue> comptimeArgs;
  SmallVector<MojoValue> instanceArgs;
  auto pushTensorCapture = [&](StringRef cap, Value mlirCapture) {
    comptimeArgs.push_back(
        MojoValue(llvm::formatv("type_of({0}).dtype", cap).str()));
    comptimeArgs.push_back(
        MojoValue(llvm::formatv("type_of({0}).LayoutType", cap).str()));
    comptimeArgs.push_back(traitCaptureMutComptimeExpr(mlirCapture, cap));
    // Bare pointer + layout off the source's stored handle (the device
    // representation that translates), mirroring the dense raw-pointer path.
    instanceArgs.push_back(
        state.builder.var(MojoValue(llvm::formatv("{0}.handle", cap).str())));
    instanceArgs.push_back(
        state.builder.var(MojoValue(llvm::formatv("{0}.layout()", cap).str())));
  };
  for (unsigned i = 1; i < fusedInputs.size(); ++i)
    pushTensorCapture(fusedInputs[i].text, fusibleTensors[i]);
  for (auto &[tensorVal, captureIdx] : extraTensorCaptures)
    pushTensorCapture(state.getMojoValueFor(tensorVal).text, tensorVal);
  for (auto &[val, fieldName, mojoType] : scalarCaptures)
    instanceArgs.push_back(state.getMojoValueFor(val));

  MojoValue functorInstance =
      state.builder.var(structNameVal(comptimeArgs, instanceArgs));
  MojoValue concreteTypeName = structNameVal[comptimeArgs];
  state.setFusionStructLoweringFor(bindOp.getOutput(), functorInstance,
                                   /*bindMethod=*/"", concreteTypeName);
  return success();
}

LogicalResult BindLambdaOp::emitMojo(EmitMojoState &state) {

  bool isOutputLambda = getBindKind() == BindLambdaKind::OutputLambda;
  // Default to Compute (return-based) so that non-output lambdas (InputLambda)
  // emit a return rather than a store in emitFusionStructReturnOrStore.
  auto lambdaType = KernelOutputFusionLambdaType::Compute;
  // TODO(GEX-3912): the tile and SIMD lowering paths are interleaved here (and
  // in emitFusionStruct / emitElementwiseLambdaBodyOps), which makes the
  // lowering hard to follow. Unifying the tile compute interface (GEX-3912)
  // should let the tile and SIMD paths converge.
  //
  // When the op's device is configured for tile codegen, a standalone
  // elementwise lambda lowers to the tile-based `ElementwiseFusionTile` path
  // (manufactures its result via the `dst` destination tile and is driven by
  // `foreach_fusion_tile`).
  if (getBindKind() == BindLambdaKind::ElemwiseLambda &&
      isTileCodegenForOp(getOperation(), state.moduleOp))
    lambdaType = KernelOutputFusionLambdaType::ElementwiseTile;
  if (isOutputLambda) {
    // Determine whether this is a store, compute, or tile-compute lambda.
    if (!getResult().hasOneUse())
      return emitError("BindLambdaOp should have a unique use");
    auto fusedOp =
        dyn_cast<MOGG::TensorFusedOp>(*getResult().getUsers().begin());
    if (!fusedOp)
      return emitError("BindLamdbdaOp user should be a FusedTensorOp");
    auto outputLambdaTypeOr = getKernelOutputFusionLambdaType(
        state.registry, fusedOp, state.enableComputeLambdas, state.moduleOp);
    if (outputLambdaTypeOr.isError())
      return emitError(outputLambdaTypeOr.getError());
    lambdaType = *outputLambdaTypeOr;
  }

  Block *lambdaBody = getBody();
  auto fusibleTensors = getFusibleTensors();

  // Trait lane (Phase 5): if this input bind feeds a trait kernel's fused-LOAD
  // operand, emit the MTS-free input-load fusion functor (mockup shape) instead
  // of the legacy MTS `FusionStruct`. (Store / transform contracts are later
  // slices and fall through to the legacy path.) These functors capture bare
  // `.handle` / `.layout()` off a dense trait arg, so skip `markAsKeepAlive`
  // (the legacy MTS path keeps the full tile projection alive).
  {
    MOGG::TensorFusedOp fusedOp;
    for (Operation *user : getOutput().getUsers())
      if (auto f = dyn_cast<MOGG::TensorFusedOp>(user)) {
        fusedOp = f;
        break;
      }
    if (fusedOp) {
      SmallVector<MojoValue> traitFusedInputs =
          state.getMojoValuesFor(fusibleTensors);
      if (!isOutputLambda) {
        std::optional<TensorArgContract> contract =
            inputLoadFusionContract(state, fusedOp);
        if (contract && contract->fused &&
            contract->hasAccess(TensorArgAccess::load))
          return emitInputLoadFusionFunctor(*this, state, lambdaBody,
                                            fusibleTensors, traitFusedInputs);
      } else {
        std::optional<TensorArgContract> contract =
            outputFusionContract(state, fusedOp);
        if (contract && contract->fused &&
            contract->hasAccess(TensorArgAccess::store))
          return emitOutputStoreFusionFunctor(*this, state, lambdaBody,
                                              fusibleTensors, traitFusedInputs);
        if (contract && contract->fused &&
            contract->hasAccess(TensorArgAccess::transform))
          return emitTransformFusionFunctor(*this, state, lambdaBody,
                                            fusibleTensors, traitFusedInputs);
      }
    }
  }

  SmallVector<MojoValue> fusedInputs;
  for (auto input : state.getMojoValuesFor(fusibleTensors)) {
    fusedInputs.push_back(input);
    state.markAsKeepAlive(input);
  }

  // Every BindLambda kind emits a `FusionStruct`; for `ElemwiseLambda` it
  // conforms to `ElementwiseFusion` and `foreach_fusion` captures it by value
  // so it survives `crossDeviceCaptures`.
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

  // Get the target device from the kernel stub that contains this operation.
  auto kernelStubOp = this->getOperation()->getParentOfType<KernelStubOp>();
  MojoValue targetDeviceLabel =
      MojoValue(kernelStubOp
                    ? MO::getDeviceOrDefaultAny(kernelStubOp).getLabel()
                    : MO::getDeviceOrDefaultAny(*this).getLabel())
          .quote();

  // Get the trace name from the fusion info tree.
  auto traceNameOr =
      state.inferDefaultKernelParamValue(*this, kMoggTraceNameParameterName);
  MojoValue traceNameExpr = traceNameOr.isError()
                                ? MojoValue(getOperationName()).quote()
                                : *traceNameOr;

  // When the op's device is configured for tile codegen, drive the fusion with
  // the tile driver
  // `foreach_fusion_tile` (kwargs `tensor`/`elem`/`ctx`, no `simd_width` -- it
  // defaults `tile_shape` internally) instead of the SIMD `foreach_fusion`.
  MojoValue elemValue = lowering.instance;
  if (isTileCodegenForOp(getOperation(), state.moduleOp)) {
    // Resolve the tile driver symbol. Prefer the registered
    // `mogg.call.foreach_tile` primitive when its Mojo is present; otherwise
    // derive the name from the SIMD foreach symbol (`...foreach_fusion` ->
    // `...foreach_fusion_tile`) so text emission is self-contained on branches
    // where the driver Mojo isn't loaded.
    MojoValue tileDriverName =
        state.registry.containsPrimitive(kMoggForeachTileRegistration)
            ? MojoValue(
                  state.registry.lookupPrimitive(kMoggForeachTileRegistration)
                      .getSymNameAttr()
                      .strref())
            : MojoValue((getKernelSymName() + "_tile").str());
    state.builder.call(
        tileDriverName,
        {MojoValue(kMoggTargetParameterName).equal(targetDeviceLabel),
         MojoValue(kMoggTraceNameParameterName).equal(traceNameExpr)},
        {MojoValue(kParameterForeachTensor).equal(mojoOutput),
         MojoValue(kParameterForeachElem).equal(elemValue),
         MojoValue(kParameterForeachCtx).equal(contextValue)});
    state.genAllKeepAliveStatements();
    return success();
  }

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

  MojoValue simdWidthExpr = MojoValue(kFnGetSimdWidthForDtypes)(
      {paramsDtypesTuple, targetDeviceLabel});

  // `foreach_fusion` (`getKernelSymName()`, kwargs `tensor`/`elem`/`ctx`) takes
  // the `ElementwiseFusion` fusion-struct instance as `elem`.
  state.builder.verbatim(MojoValue(getKernelSymName())(
      {MojoValue(kMoggTargetParameterName).equal(targetDeviceLabel),
       MojoValue(kParameterSimdWidth).equal(simdWidthExpr),
       MojoValue(kMoggTraceNameParameterName).equal(traceNameExpr)},
      {MojoValue(kParameterForeachTensor).equal(mojoOutput),
       MojoValue(kParameterForeachElem).equal(elemValue),
       MojoValue(kParameterForeachCtx).equal(contextValue)}));
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
  auto maybeParams = inferKernelParameters(viewMatPrimDecl, inferParam);
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

ParseResult KernelRegionOp::parse(OpAsmParser &parser, OperationState &result) {
  // Op operands: the `!mo.tensor` inputs, printed bare as `(%a, %b : types)`
  // (no `inputs` keyword).
  llvm::SMLoc operandsLoc = parser.getCurrentLocation();
  SmallVector<OpAsmParser::UnresolvedOperand> inputOperands;
  SmallVector<Type> inputTypes;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseOperandList(inputOperands) || parser.parseColon() ||
        parser.parseTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }

  // Results: an optional `-> types`.
  SmallVector<Type> resultTypes;
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return failure();

  if (parser.resolveOperands(inputOperands, inputTypes, operandsLoc,
                             result.operands))
    return failure();
  result.addTypes(resultTypes);

  // Body entry-block arguments given as labeled groups. The display order is
  // inputs, outputs, context, dims; the in-memory order the verifier and
  // downstream passes rely on is [inputs, outputs, dims, context].
  SmallVector<OpAsmParser::Argument> inputArgs, outputArgs, ctxArgs, dimArgs;
  auto parseGroup =
      [&](StringRef keyword,
          SmallVectorImpl<OpAsmParser::Argument> &into) -> ParseResult {
    if (succeeded(parser.parseOptionalKeyword(keyword)))
      return parser.parseArgumentList(into, OpAsmParser::Delimiter::Paren,
                                      /*allowType=*/true);
    return success();
  };
  if (parseGroup("inputs", inputArgs) || parseGroup("outputs", outputArgs) ||
      parseGroup("context", ctxArgs) || parseGroup("dims", dimArgs))
    return failure();

  SmallVector<OpAsmParser::Argument> bodyArgs;
  bodyArgs.append(inputArgs.begin(), inputArgs.end());
  bodyArgs.append(outputArgs.begin(), outputArgs.end());
  bodyArgs.append(dimArgs.begin(), dimArgs.end());
  bodyArgs.append(ctxArgs.begin(), ctxArgs.end());

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  return parser.parseRegion(*result.addRegion(), bodyArgs);
}

void KernelRegionOp::print(OpAsmPrinter &p) {
  // Op operands: the `!mo.tensor` inputs, printed bare.
  p << " (";
  p.printOperands(getInputs());
  if (!getInputs().empty()) {
    p << " : ";
    llvm::interleaveComma(getInputs().getTypes(), p);
  }
  p << ")";

  // Results: `-> types` (omitted when there are none).
  if (!getOutputs().empty()) {
    SmallVector<Type> outTypes(getOutputs().getTypes());
    p.printArrowTypeList(outTypes);
  }

  // Body entry-block arguments, as labeled groups. The in-memory order is
  // [inputs, outputs, dims, context]; the display shows context before dims.
  auto args = getBody().front().getArguments();
  size_t numInputs = getInputs().size();
  size_t numOutputs = getOutputs().size();
  size_t numTensors = std::min(numInputs + numOutputs, args.size());

  auto inputArgs = args.take_front(std::min(numInputs, args.size()));
  auto outputArgs = args.drop_front(inputArgs.size()).take_front(numOutputs);
  auto tail = args.drop_front(numTensors);
  // `tail` is `[dims..., context]`; the context is the trailing argument.
  BlockArgument ctxArg = tail.empty() ? BlockArgument() : tail.back();
  auto dimArgs = tail.empty() ? tail : tail.drop_back(1);

  p.increaseIndent();
  auto printGroup = [&](StringRef label, auto group) {
    if (group.empty())
      return;
    p.printNewline();
    p << label << " (";
    llvm::interleaveComma(group, p,
                          [&](BlockArgument ba) { p.printRegionArgument(ba); });
    p << ")";
  };
  printGroup("inputs", inputArgs);
  printGroup("outputs", outputArgs);
  if (ctxArg) {
    p.printNewline();
    p << "context (";
    p.printRegionArgument(ctxArg);
    p << ")";
  }
  printGroup("dims", dimArgs);
  p.decreaseIndent();

  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult KernelRegionOp::verify() {
  // Inputs are `!mo.tensor` operands and outputs are `!mo.tensor` results; the
  // body block arguments are the codegen handles: one !mogg.pointer_ per input
  // then output tensor, then zero or more !mogg.dim_, then exactly one trailing
  // !mogg.context.
  const size_t numTensors = getInputs().size() + getOutputs().size();
  auto args = getBody().front().getArguments();

  if (args.size() < numTensors + 1)
    return emitOpError()
           << "expected at least " << (numTensors + 1)
           << " body block arguments (one !mogg.pointer_ per input "
              "and output tensor plus a trailing !mogg.context), "
              "got "
           << args.size();

  for (size_t i = 0; i < numTensors; ++i)
    if (!isa<PointerType>(args[i].getType()))
      return emitOpError() << "body block argument #" << i
                           << " must be a !mogg.pointer_ (one per input then "
                              "output tensor)";

  size_t i = numTensors;
  while (i < args.size() && isa<DimType>(args[i].getType()))
    ++i;
  if (i != args.size() - 1 || !isa<ContextType>(args.back().getType()))
    return emitOpError() << "body block arguments after the pointers must be "
                            "zero or more !mogg.dim_ followed by exactly one "
                            "!mogg.context";

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

static MojoValue
maybeAssembleDenseTraitArgFromCapture(EmitMojoState &state, Value val,
                                      const TensorArgContract &contract,
                                      KernelStubOp stub);

LogicalResult CallExecuteOp::emitMojo(EmitMojoState &state) {
  KernelDeclOp execKernelDecl =
      state.registry.lookupKernelBySymName(getKernelSymName());
  KernelDeclAdaptor execAdaptor(execKernelDecl);
  const KernelDeclFunctionAdaptor &execFnAdaptor = *execAdaptor.executeFunction;

  auto stub = (*this)->getParentOfType<MOGG::KernelStubOp>();

  // Collect all kernel I/Os
  SmallVector<MojoValue> kernelOperands;
  for (auto [i, val] : llvm::enumerate(getKernelDPSOutputs())) {
    MojoValue outValue = state.getMojoValueFor(val);
    if (stub && i < execFnAdaptor.outputArguments.size()) {
      if (std::optional<TensorArgContract> contract =
              execFnAdaptor.outputArguments[i].getTraitContract())
        outValue =
            maybeAssembleDenseTraitArgFromCapture(state, val, *contract, stub);
    }
    kernelOperands.push_back(outValue);
  }

  auto inputs = getKernelInputs();
  for (unsigned i = 0; i < inputs.size(); ++i) {
    Value input = inputs[i];
    MojoValue inputValue = state.getMojoValueFor(input);

    if (stub && i < execFnAdaptor.inputArguments.size()) {
      if (std::optional<TensorArgContract> contract =
              execFnAdaptor.inputArguments[i].getTraitContract())
        inputValue = maybeAssembleDenseTraitArgFromCapture(state, input,
                                                           *contract, stub);
    }

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
  auto maybeParams = inferKernelParameters(
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
// CallBuildGraphOp
//===----------------------------------------------------------------------===//

LogicalResult CallBuildGraphOp::verify() {
  // The launch argument is the dedicated `$builder` operand, so a context must
  // not appear among the kernel inputs/outputs (that is the `call.execute`
  // shape this op replaces).
  auto isContext = [](Value v) { return isa<MOGG::ContextType>(v.getType()); };
  if (llvm::any_of(getInputs(), isContext) ||
      llvm::any_of(getOutputs(), isContext))
    return emitError("mogg.call.build_graph takes a device-graph builder as "
                     "its launch argument, not a context");
  return success();
}

LogicalResult
CallBuildGraphOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifySymbolUsesForKernelDeclCall(
      *this, symbolTable, [this](KernelDeclOp decl) -> LogicalResult {
        if (!decl.getBuildGraph())
          return emitOpError("kernel '")
                 << getKernelSymName() << "' has no 'build_graph' function";
        return success();
      });
}

bool CallBuildGraphOp::isThrows() { return getIsThrows(); }

LogicalResult CallBuildGraphOp::emitMojo(EmitMojoState &state) {
  // Mirrors CallExecuteOp::emitMojo, but selects the kernel's `build_graph`
  // function, appends the device-graph builder as the launch argument (in place
  // of the device context), and captures the recorded node handle as the op's
  // result.
  KernelDeclOp buildGraphKernelDecl =
      state.registry.lookupKernelBySymName(getKernelSymName());

  KernelDeclAdaptor buildGraphAdaptor(buildGraphKernelDecl);
  if (!buildGraphAdaptor.buildGraphFunction)
    return emitOpError("kernel '")
           << getKernelSymName() << "' has no 'build_graph' function";
  const KernelDeclFunctionAdaptor &buildGraphFnAdaptor =
      *buildGraphAdaptor.buildGraphFunction;

  // Collect kernel I/Os: DPS outputs, then inputs (mirroring `execute`).
  auto kernelOperands = llvm::map_to_vector(
      getOutputs(), [&](Value val) { return state.getMojoValueFor(val); });

  auto inputs = getInputs();
  for (auto [i, input] : llvm::enumerate(inputs)) {
    MojoValue inputValue = state.getMojoValueFor(input);

    // Transform mutable inputs if the kernel expects an InputTensor.
    if (isMutableInputTensor(input) &&
        i < buildGraphFnAdaptor.inputArguments.size()) {
      const auto *tensorInfo = std::get_if<KernelDeclTensorAdaptor>(
          &buildGraphFnAdaptor.inputArguments[i].underlyingType);
      if (tensorInfo && tensorInfo->ioSpec == MOGG::IOSpec::InputTensor)
        inputValue = generateMutableTensorInit(input, state);
    }

    kernelOperands.push_back(inputValue);
  }

  // The launch argument is the device-graph builder (not a context).
  kernelOperands.push_back(state.getMojoValueFor(getGraphBuilder()));

  // Infer the kernel parameters against the `build_graph` function.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    return state.inferDefaultKernelParamValue(*this, paramName);
  };
  auto maybeParams = inferKernelParameters(
      buildGraphAdaptor, MOGG::kMoggBuildGraphFunctionLabel, inferParam);
  if (maybeParams.isError())
    return emitError() << "Failed to infer all the kernel parameters: "
                       << maybeParams.takeError();

  auto [prefixSymbolName, buildGraphFunc] = EmitMojoState::splitFunctionName(
      buildGraphFnAdaptor.fullyResolvedName.strref());
  MojoValue structType = prefixSymbolName[maybeParams->structParameters];
  MojoValue fullBuildGraphFunc = structType.get(buildGraphFunc);

  // The `build_graph` method records the dispatch into the builder. The op is
  // node-free -- ordering is expressed by the enclosing
  // `mgp.device_graph.dependencies` op's `region` call -- so this is a plain
  // void call with no result to capture.
  state.builder.call(fullBuildGraphFunc, maybeParams->functionParameters,
                     kernelOperands);
  state.genAllKeepAliveStatements();
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

OutputOp KernelStubOp::getBodyTerminator() {
  return cast<OutputOp>(getBodyBlock().getTerminator());
}

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

  // If the ABI is MOGG, this was originally a MOGG kernel. If MAP, this is an
  // advanced-fusion stub outlined by OutlineKernelsIntoFunctions. Otherwise it
  // was a GenericCompiledRegionOp which has that implicit void** ABI.
  if (getAbi() == ABIKind::MOGG)
    return emitMojoMOGGABI(state);
  if (getAbi() == ABIKind::MAP)
    return emitMojoMAPABI(state);
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

  // Trait lane: trait tensor stub arguments are `_MoggTensorBufferCapture` at
  // the region→stub boundary; execute-facing trait args are assembled inside
  // the stub body. Shape-function no-role args keep generic `DenseTensor`
  // bounds via `mogg.trait_stub_arg_bounds`.
  ArrayAttr traitBounds = getTraitStubArgBoundsAttr();
  ArrayAttr traitCaptures = getTraitStubBufferCaptureAttr();
  auto traitBoundFor = [&](size_t i) -> StringRef {
    if (!traitBounds || i >= traitBounds.size())
      return {};
    return cast<StringAttr>(traitBounds[i]).getValue();
  };
  auto traitCaptureDictFor = [&](size_t i) -> DictionaryAttr {
    if (!traitCaptures || i >= traitCaptures.size())
      return {};
    // A `StringAttr` entry is always the "no capture" placeholder, regardless
    // of content.
    if (isa<StringAttr>(traitCaptures[i]))
      return {};
    return dyn_cast<DictionaryAttr>(traitCaptures[i]);
  };

  if (traitBounds) {
    for (Attribute boundAttr : traitBounds) {
      if (!cast<StringAttr>(boundAttr).getValue().empty()) {
        if (!state.emittedTraitMarkerImports) {
          state.builder.verbatim(MojoValue(R"mojo(
from extensibility import DenseTensor
)mojo"));
          state.emittedTraitMarkerImports = true;
        }
        break;
      }
    }
  }
  if (traitCaptures) {
    for (Attribute captureAttr : traitCaptures) {
      if (isa<DictionaryAttr>(captureAttr)) {
        if (!state.emittedTensorBufferCaptureDefs) {
          state.structDefsBuilder.verbatim(
              MojoValue(kTensorBufferCaptureDefs.str()));
          state.emittedTensorBufferCaptureDefs = true;
        }
        break;
      }
    }
  }

  SmallVector<MOGG::NamedMojoDef> stubParams;
  SmallVector<MOGG::NamedMojoDef> inputTypes;
  for (auto [i, arg] : llvm::enumerate(getBody().getArguments())) {
    if (DictionaryAttr captureDict = traitCaptureDictFor(i)) {
      StringRef dtypeExpr =
          cast<StringAttr>(captureDict.get("dtype")).getValue();
      auto tensorTy = MO::dynCastFromTensorLike(arg.getType());
      if (!tensorTy)
        return emitOpError("buffer-capture stub arg must be tensor-like");
      MojoValue shapeCoord =
          state.getShapeAsCoordMojoValue(tensorTy.getTypedShapeAttr());
      MojoValue layoutExpr = MojoValue(kFnRowMajor)({}, {shapeCoord});
      MojoValue layoutType = MojoValue("type_of")({}, {layoutExpr});
      MojoValue captureType("_MoggTensorBufferCapture");
      inputTypes.push_back(MOGG::NamedMojoDef(
          "arg", captureType[{MojoValue(dtypeExpr), layoutType}]));
    } else {
      StringRef bound = traitBoundFor(i);
      if (!bound.empty()) {
        std::string paramName = llvm::formatv("__trait_arg_type_{}", i).str();
        stubParams.push_back(
            MOGG::NamedMojoDef::asVerbatim(paramName, MojoValue(bound)));
        inputTypes.push_back(MOGG::NamedMojoDef("arg", MojoValue(paramName)));
      } else {
        inputTypes.push_back(
            MOGG::NamedMojoDef("arg", typeToMojo(arg.getType())));
      }
    }
  }

  // Determine the Mojo return type from the stub's return type. A trait
  // shape-function stub keeps the concrete `IndexList[N]` return type that
  // `shape_func_result<N>` renders to (N = the op output rank); the forwarded
  // shape-fn call's generic `IndexList[T.rank]` result is reconciled to it by a
  // rebind in `CallShapeOp::emitMojo`.
  std::optional<MOGG::MojoValue> mojoReturnType = std::nullopt;
  FunctionType functionType = getFunctionType();
  if (functionType.getNumResults())
    mojoReturnType = typeToMojo(functionType.getResult(0));

  bool errOccurred = false;
  state.builder.def(
      getSymNameAttr().str(), stubParams, inputTypes,
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

LogicalResult KernelStubOp::emitMojoMAPABI(EmitMojoState &state) {
  // The MAP-ABI stub takes raw runtime handles as arguments -- one
  // `!mogg.pointer_` per input then per output tensor, then the dynamic
  // `!mogg.dim_`s, then a `!mogg.context` -- and reconstructs the codegen
  // tensors internally (`mogg._shape.create` / `mogg._tensor.create`) before
  // invoking the opaque kernel's execute function (`mogg._call.opaque`). Unlike
  // the MOGG ABI there is no trait lane, no async wrapping, and no `void**`
  // marshalling: the body ops emit straight-line Mojo.
  SmallVector<MOGG::NamedMojoDef> inputTypes;
  for (BlockArgument arg : getBody().getArguments())
    inputTypes.push_back(
        MOGG::NamedMojoDef("arg", state.mlirTypeToMojoType(arg.getType())));

  bool errOccurred = false;
  state.builder.def(
      getSymNameAttr().str(), /*parameters=*/{}, inputTypes,
      /*returnType=*/std::nullopt, /*exported=*/false,
      /*raises=*/isThrows(), /*isAlwaysInline=*/false,
      [&](const MOGG::FunctionBody &functionBody) {
        state.currentFunctionBody = functionBody;

        for (auto &&[mlirArg, mojoArg] :
             llvm::zip(getBody().getArguments(), functionBody.arguments))
          state.setMojoValueFor(mlirArg, mojoArg);

        errOccurred = llvm::any_of(getOps(), [&](mlir::Operation &op) {
          return state.lowerOpToMojo(&op).failed();
        });
      });

  return failure(errOccurred);
}

//===----------------------------------------------------------------------===//
// Advanced-fusion codegen op emitters (MAP ABI)
//===----------------------------------------------------------------------===//

LogicalResult ShapeCreateOp::emitMojo(EmitMojoState &state) {
  // Encode the shape as a `Coord` -- static dims as compile-time `Idx[N]`,
  // dynamic dims as runtime `Scalar[DType.int](<dim>)` sourced from the
  // `!mogg.dim_` operands. `tensor.create` builds a row-major `Layout` from it.
  MojoValue coord =
      state.getShapeAsCoordMojoValue(getShape(), getDynamicDims());
  state.setMojoValueFor(getResult(), state.builder.var(coord));
  return success();
}

LogicalResult TensorCreateOp::emitMojo(EmitMojoState &state) {
  // The rank comes from the static shape attr on the feeding `shape.create`;
  // the opaque `!mogg.shape` handle carries no attributes of its own.
  auto shapeCreate = getShape().getDefiningOp<ShapeCreateOp>();
  ASSERT_STREAM(shapeCreate,
                "tensor.create shape operand must come from a shape.create");
  MOSH::ShapeAttr shapeAttr = shapeCreate.getShape();

  ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(getElementType());
  ASSERT_STREAM(!dtypeOr.isError(),
                "Failed to extract dtype from tensor.create element type");
  MojoValue dTypeParam = state.mlirDTypeToMojoType(*dtypeOr);
  MojoValue rankParam(static_cast<uint64_t>(
      shapeAttr.getRank() == 0 ? 1 : shapeAttr.getRank()));
  auto mutParam = getMut();
  auto ioParam = MojoValue(kStructIO).get(MO::stringifyIOKind(getIo()));
  // Device kind (cpu vs gpu) is on the op's own `$device` attr, so the
  // preferred alignment is picked locally without reaching out to the stub.
  MojoValue alignmentParam(static_cast<uint64_t>(
      M::Driver::getPreferredAlignment(getDevice().getLabel().str())));

  // Build a row-major `Layout` from the shape Coord produced by `shape.create`.
  auto layoutVar = state.builder.var(
      MojoValue(kFnRowMajor)({}, {state.getMojoValueFor(getShape())}));

  // Reuse the legacy `mogg.tensor.__init__` primitive: the two ops build the
  // same `ManagedTensorSlice`, differing only in that the pointer arrives as a
  // raw `!mogg.pointer_` here instead of being unwrapped from a slice.
  auto tensorInitPrimDecl =
      state.registry.lookupPrimitive(TensorInitOp::getOperationName());
  ASSERT_STREAM(tensorInitPrimDecl,
                "Primitive not found for " << TensorInitOp::getOperationName());
  MojoValue moggTensorInit(primitiveMojoName(tensorInitPrimDecl));
  state.setMojoValueFor(
      getTensor(),
      state.builder.var(moggTensorInit(
          {dTypeParam, rankParam, mutParam, ioParam, alignmentParam},
          {OpaquePointerRef(state.getMojoValueFor(getPtr()))
               .bitcast(MojoValue(kComptimeDTypeNoneType))(),
           layoutVar})));

  return success();
}

bool CallOpaqueOp::isThrows() {
  // Whether the stub raises depends on the callee kernel, exactly like
  // `CallExecuteOp::isThrows` reads the flag baked from its execute signature.
  // Resolve the callee `mogg.kernel.decl` and read its execute function's
  // signature `raises` bit.
  auto decl = mlir::SymbolTable::lookupNearestSymbolFrom<KernelDeclOp>(
      *this, getCalleeAttr());
  if (!decl)
    return false;
  KernelDeclAdaptor adaptor(decl);
  if (!adaptor.executeFunction || !adaptor.executeFunction->attr)
    return false;
  // `isThrows` is non-const on the signature; copy it to a local to call it.
  auto sig = adaptor.executeFunction->attr.getSignature();
  return sig.isThrows();
}

LogicalResult CallOpaqueOp::emitMojo(EmitMojoState &state) {
  KernelDeclOp execKernelDecl =
      state.registry.lookupKernelBySymName(getCallee());
  KernelDeclAdaptor execAdaptor(execKernelDecl);
  const KernelDeclFunctionAdaptor &execFnAdaptor = *execAdaptor.executeFunction;

  // Destination-passing-style: outputs first, then inputs, then the device
  // context. The context is the op's own `$device_context` operand (wired by
  // `MAPToMOGG`); append it last to match the execute signature's trailing
  // `ctx: DeviceContext`.
  SmallVector<MojoValue> kernelOperands;
  for (Value out : getOutputs())
    kernelOperands.push_back(state.getMojoValueFor(out));
  for (Value in : getInputs())
    kernelOperands.push_back(state.getMojoValueFor(in));
  if (Value ctx = getDeviceContext())
    kernelOperands.push_back(state.getMojoValueFor(ctx));

  // A fused output arrives as a `tensor.create.epilogue` result (already
  // lowered to a `_MoggOutputStoreFusionTensor`). Unlike the MOGG-ABI path
  // there is no `mogg.bind` for `getHasEpilogueFusionValue` to detect, so force
  // the kernel's `has_epilogue_fusion` parameter True here to match the
  // execute-call wiring.
  bool hasFusedOutput = llvm::any_of(getOutputs(), [](Value out) {
    return out.getDefiningOp<TensorCreateEpilogueOp>() != nullptr;
  });
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    if (hasFusedOutput &&
        paramName == MOGG::kMoggHasEpilogueFusionParameterName)
      return MojoValue(true);
    return state.inferDefaultKernelParamValue(*this, paramName,
                                              getKernelParamAttr());
  };
  auto maybeParams = inferKernelParameters(
      execAdaptor, MOGG::kMoggExecuteFunctionLabel, inferParam);
  if (maybeParams.isError())
    return emitError() << "Failed to infer all the kernel parameters: "
                       << maybeParams.takeError();

  auto [prefixSymbolName, executeFunc] = EmitMojoState::splitFunctionName(
      execFnAdaptor.fullyResolvedName.strref());
  MojoValue structType = prefixSymbolName[maybeParams->structParameters];
  MojoValue fullExecuteFunc = structType.get(executeFunc);

  state.builder.call(fullExecuteFunc, maybeParams->functionParameters,
                     kernelOperands);
  state.genAllKeepAliveStatements();
  return success();
}

LogicalResult ScalarCreateOp::emitMojo(EmitMojoState &state) {
  const MojoValue &input = state.getMojoValueFor(getTensor());
  // Reuse the legacy `mogg.as_scalar` primitive: both pack a single-element
  // tensor operand into the scalar the kernel expects.
  auto asScalarPrimDecl =
      state.registry.lookupPrimitive(AsScalarOp::getOperationName());
  ASSERT_STREAM(asScalarPrimDecl,
                "Primitive not found for " << AsScalarOp::getOperationName());
  MojoValue moggAsScalar(primitiveMojoName(asScalarPrimDecl));
  MojoValue scalarValue = moggAsScalar(ArrayRef<MojoValue>{}, {input});
  state.setMojoValueFor(getScalar(), state.builder.var(scalarValue));
  return success();
}

LogicalResult TupleCreateOp::emitMojo(EmitMojoState &state) {
  // Non-fused counterpart of `AsTupleOp`: pack the run of tensor operands into
  // a `VariadicTensors` (runtime pointers + shapes, plus a comptime
  // static-spec list). The fused-variadic path is not yet wired (slice 2).
  SmallVector<MojoValue> inputValues = state.getMojoValuesFor(getElements());

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

  MojoValue rank = StaticTensorSpecRef(
                       ManagedTensorSliceRef(inputValues.front()).staticSpec())
                       .rank();
  SmallVector<MojoValue> staticSpecInternals;
  SmallVector<MojoValue> staticSpecShapes;
  SmallVector<MojoValue> staticSpecStrides;
  for (const auto &reboundInput : inputValues) {
    StaticTensorSpecRef spec(ManagedTensorSliceRef(reboundInput).staticSpec());
    staticSpecInternals.push_back(spec.getInternals());
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

  MojoValue variadicTensorsConstructor = MojoValue(kStructVariadicTensors)[{
      MojoValue(kParameterIOSpec)
          .equal(ManagedTensorSliceRef(inputValues[0]).ioSpec()),
      MojoValue(kParameterStaticSpecs).equal(inputSpecsComptime)}];
  MojoValue variadicTensorsValue = variadicTensorsConstructor(
      {}, {runtimePointersTuple, runtimeShapesTuple});
  state.setMojoValueFor(getTuple(), state.builder.var(variadicTensorsValue));
  return success();
}

LogicalResult TensorStoreOp::emitMojo(EmitMojoState &state) {
  // Only ever reached recursively from `TensorCreateEpilogueOp::emitMojo`,
  // while lowering the `OutputStoreFusion` functor's `__call__` method, so it
  // always writes into `structDefsBuilder` rather than the enclosing kernel
  // stub's builder.
  const MojoValue &tile = state.getMojoValueFor(getTensor());
  const MojoValue &idx = state.getMojoValueFor(getIndex());
  const MojoValue &value = state.getMojoValueFor(getValue());
  const MojoValue &widthParam = state.currentFunctionBody.parameters[2];
  const MojoValue &alignmentParam = state.currentFunctionBody.parameters[3];

  // `__call__`'s generic `dtype` comptime parameter isn't statically known to
  // equal the store target's own field-parameter dtype, so the stored value
  // needs an explicit rebind. The target type is derived from the tile's own
  // reflected `.dtype` rather than threading the struct's field-parameter name
  // through new state.
  MojoValue tileDType = MojoValue("type_of")({}, {tile}).get(kParameterDType);
  MojoValue rebound =
      value.rebind(MojoValue(kStructSIMD)[{tileDType, widthParam}]);
  state.structDefsBuilder.call(tile.get("store"),
                               {MojoValue(kParameterWidth).equal(widthParam),
                                MojoValue("alignment").equal(alignmentParam)},
                               {idx, rebound});
  return success();
}

LogicalResult TensorCreateEpilogueOp::emitMojo(EmitMojoState &state) {
  // Emit the MTS-free output-store fusion FUNCTOR +
  // `_MoggOutputStoreFusionTensor` wrapper for a MAP-ABI fused output. This is
  // the advanced-fusion counterpart of the MOGG-ABI
  // `emitOutputStoreFusionFunctor` / `TensorFusedOp::emitMojo` pair, except the
  // functor's `__call__` body is built by recursively lowering this op's own
  // epilogue body ops (mirroring `emitMojoMAPABI`'s stub-body walk) rather than
  // hand-rendered from a fixed template.
  ASSERT_STREAM(!getCaptures().empty(),
                "tensor.create.epilogue requires at least one (store-target) "
                "capture");

  // Emit the shared trait + wrapper definitions (`OutputStoreFusion` /
  // `_MoggOutputStoreFusionTensor`) once, exactly as the MOGG-ABI store path.
  state.emitOutputStoreFusionDefsOnce();

  // Output dtype / mutability from the epilogue op's own attributes.
  ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(getElementType());
  ASSERT_STREAM(!dtypeOr.isError(),
                "Failed to extract dtype from epilogue element type");
  MojoValue dtypeParam = state.mlirDTypeToMojoType(*dtypeOr);
  MojoValue mutParam(getMut());

  // Layout: `row_major` over the epilogue op's own `$shape` operand (mirrors
  // `tensor.create`); the shape is an operand of the op, not recovered from the
  // capture.
  MojoValue layoutVar = state.builder.var(
      MojoValue(kFnRowMajor)({}, {state.getMojoValueFor(getShape())}));
  MojoValue layoutTypeParam(
      llvm::formatv("type_of({0})", layoutVar.text).str());

  // The functor conforms to `OutputStoreFusion`: one (dtype, LayoutType, mut)
  // param triple and one (ptr, layout) field pair PER CAPTURE -- today always
  // one (the store target), but a fused view or binary elementwise op could
  // add more, each with its own load. Unlike the old fixed-template functor,
  // the struct is built via the scoped `structDecl` overload and `__call__`'s
  // body recurses into the epilogue's own body ops instead of being
  // hand-rendered here.
  //
  // `!mogg.tensor_` is an opaque handle (see MOGGTypes.td): shape / dtype /
  // mutability live on the op that CREATES a capture, not on the SSA value
  // itself. `MAPToMOGG` today only ever captures the store target, whose
  // dtype/layout/mut are exactly this op's own attributes above, so every
  // capture reuses `dtypeParam`/`layoutTypeParam`/`layoutVar`/`mutParam`. A
  // hypothetical future capture with its own dtype (e.g. a second input tile
  // fused into a binary elementwise epilogue) would need that threaded from
  // its defining op; until then, this is the correct value for every capture
  // in practice, while the struct/field/argument-list construction below
  // loops over N generically.
  unsigned numCaptures = getCaptures().size();
  MojoValue structNameVal = state.structDefsBuilder.newVar(
      state.structDefsBuilder.getIdentifier("OutputStoreFunctor"));
  SmallVector<NamedMojoDef> structParams;
  for (unsigned j = 0; j < numCaptures; ++j) {
    structParams.push_back(NamedMojoDef("dtype", kStructDType));
    structParams.push_back(NamedMojoDef("LayoutType", kStructTensorLayout));
    structParams.push_back(NamedMojoDef("mut", kStructBool));
  }

  // Per-capture field names plus their `Self.`-qualified comptime param
  // expressions, populated while declaring fields below and reused when
  // rebuilding each capture's `TileTensor` in `__call__`.
  struct CaptureFieldInfo {
    MojoValue ptrField;
    MojoValue layoutField;
    MojoValue selfDType;
    MojoValue selfLayoutType;
    MojoValue selfMut;
  };

  bool errOccurred = false;
  state.structDefsBuilder.structDecl(
      structNameVal, structParams,
      {MojoValue("OutputStoreFusion"), MojoValue("TrivialRegisterPassable")},
      [&](const MojoBuilder::StructBody &sb) {
        SmallVector<CaptureFieldInfo> captureFields;
        for (unsigned j = 0; j < numCaptures; ++j) {
          MojoValue dtypeName = sb.parameters[3 * j];
          MojoValue layoutTypeName = sb.parameters[3 * j + 1];
          MojoValue mutName = sb.parameters[3 * j + 2];
          MojoValue selfDType = MojoValue("Self").get(dtypeName);
          MojoValue selfMut = MojoValue("Self").get(mutName);
          MojoValue scalarType = MojoValue(kStructScalar)[{selfDType}];
          MojoValue originType = MojoValue(kStructUntrackedOrigin)[{
              MojoValue(kParameterMut).equal(selfMut)}];
          MojoValue ptrField = state.structDefsBuilder.varDecl(
              MojoValue(kStructUnsafePointer)[{scalarType, originType}]);
          MojoValue selfLayoutType = MojoValue("Self").get(layoutTypeName);
          MojoValue layoutField =
              state.structDefsBuilder.varDecl(selfLayoutType);
          captureFields.push_back(
              {ptrField, layoutField, selfDType, selfLayoutType, selfMut});
        }

        MojoValue simdType = MojoValue(kStructSIMD)[{
            MojoValue(kParameterDType), MojoValue(kParameterWidth)}];
        SmallVector<NamedMojoDef> methodParameters = {
            NamedMojoDef::asVerbatim(kParameterDType, kStructDType),
            NamedMojoDef::asVerbatim(kParameterRank, kStructInt),
            NamedMojoDef::asVerbatim(kParameterWidth, kStructInt),
            NamedMojoDef::asVerbatim(kParameterElementAlignment, kStructInt,
                                     MojoValue(1)),
        };
        SmallVector<NamedMojoDef> methodArgs = {
            NamedMojoDef::asVerbatim(kParameterSelf),
            NamedMojoDef::asVerbatim(kParameterIdx, MojoValue(kStructCoord)),
            NamedMojoDef::asVerbatim("value", simdType),
        };
        state.structDefsBuilder.def(
            "__call__", methodParameters, methodArgs,
            /*returnType=*/std::optional<MojoValue>{},
            /*exported=*/false, /*raises=*/false,
            /*isAlwaysInline=*/true, [&](const FunctionBody &fb) {
              EmitMojoState::FunctionBodyGuard functionBodyGuard(state, fb);

              // `fb.arguments` is [self, idx, value]; the epilogue body's
              // leading block args are [value, index] (`MAPToMOGG`'s
              // `buildEpilogueTensor` convention).
              state.setMojoValueFor(getBody().getArgument(0), fb.arguments[2]);
              state.setMojoValueFor(getBody().getArgument(1), fb.arguments[1]);
              for (unsigned j = 0; j < numCaptures; ++j) {
                // Each capture gets its own fresh block arg (not an alias to
                // the outer `capturing(...)` operand), rebuilt here as a
                // `TileTensor` projection over its own captured field.
                const CaptureFieldInfo &info = captureFields[j];
                MojoValue tile = TileTensorRef::fromExternalPtrExplicit(
                    MojoValue(kParameterSelf).get(info.ptrField),
                    MojoValue(kParameterSelf).get(info.layoutField),
                    info.selfDType, info.selfLayoutType, info.selfMut);
                state.setMojoValueFor(getBody().getArgument(2 + j),
                                      state.structDefsBuilder.var(tile));
              }

              errOccurred =
                  llvm::any_of(getBody().getOps(), [&](Operation &op) {
                    return state.lowerOpToMojo(&op).failed();
                  });
            });
      });

  if (errOccurred)
    return failure();

  // Instantiate the functor: per capture, a bare pointer off its stored
  // handle plus the row-major layout (the device representation that
  // translates), in the same (dtype, LayoutType, mut) / (ptr, layout) order
  // the struct was declared with above.
  SmallVector<MojoValue> comptimeArgs;
  SmallVector<MojoValue> instanceArgs;
  for (unsigned j = 0; j < numCaptures; ++j) {
    comptimeArgs.push_back(dtypeParam);
    comptimeArgs.push_back(layoutTypeParam);
    comptimeArgs.push_back(mutParam);
    MojoValue rawPtr =
        ManagedTensorSliceRef(state.getMojoValueFor(getCaptures()[j])).rawPtr();
    instanceArgs.push_back(rawPtr);
    instanceArgs.push_back(layoutVar);
  }
  MojoValue functorInstance =
      state.builder.var(structNameVal(comptimeArgs, instanceArgs));

  // Wrap the functor in `_MoggOutputStoreFusionTensor` (carrying the graph-
  // visible `StoreAccess` marker) and pass it straight to `execute`, mirroring
  // the MOGG-ABI `TensorFusedOp::emitMojo` store branch. The device kind /
  // alignment come from the op's own `$device` attr rather than the stub.
  MojoValue alignmentParam(static_cast<uint64_t>(
      M::Driver::getPreferredAlignment(getDevice().getLabel().str())));
  MojoValue functorType = MojoValue("type_of")({}, {functorInstance});
  // Mark the wrapper device-passable unconditionally: the wrapper's
  // `_access_enabled` param (kept True) gates the callable store surface, so a
  // uniform device-passable value is free and avoids querying the stub.
  MojoValue wrapped = MojoValue(kStructMoggOutputStoreFusionTensor)[{
      dtypeParam, layoutTypeParam, alignmentParam, functorType, MojoValue(true),
      MojoValue(true)}]({}, {functorInstance, layoutVar});
  state.setMojoValueFor(getTensor(), state.builder.var(wrapped));
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

  } else if (isa<MGP::ContextType>(inputType)) {

    MojoValue asyncUnpackFunc{kFnUnpackStateCtx};
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
  // This generates: var_XX = value.unsafe_bitcast[Type]()
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
// RebindOp
//===----------------------------------------------------------------------===//

LogicalResult RebindOp::emitMojo(EmitMojoState &state) {
  // Bridges an unpacked async handle (`mogg.async.unpack` result) to the
  // codegen handle a MAP-ABI stub argument expects.
  Type resultType = getOutput().getType();

  // Context / dim casts are code-generation no-ops: `mogg.async.unpack` already
  // produced a value of the Mojo type the stub expects (a `DeviceContext` for
  // `!mogg.context`, an `Int` for `!mogg.dim_`), so forward it unchanged.
  if (isa<MOGG::ContextType, MOGG::DimType>(resultType)) {
    state.setMojoValueFor(getResult(), state.getMojoValueFor(getInput()));
    return success();
  }

  // Pointer cast: the input is the `OwnedByteBuffer` an `!mgp.buffer` unpacks
  // to, but the MAP-ABI stub takes a raw `OpaquePointer[MutAnyOrigin]` (it does
  // `ptr.bitcast[NoneType]()` in `tensor.create`). Take the buffer's data
  // pointer and bitcast it to the opaque pointer type.
  if (isa<MOGG::PointerType>(resultType)) {
    MojoValue rawPtr =
        OwnedByteBufferRef(state.getMojoValueFor(getInput())).unsafePtr();
    MojoValue opaquePtr =
        OpaquePointerRef(rawPtr).bitcast(MojoValue(kComptimeDTypeNoneType))();
    state.setMojoValueFor(getResult(), state.builder.var(opaquePtr));
    return success();
  }

  return emitOpError(
             "mogg._rebind: unsupported result type for Mojo emission: ")
         << resultType;
}

//===----------------------------------------------------------------------===//
// AsyncPackOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncPackOp::emitMojo(EmitMojoState &state) {
  auto resultIndexOr = getOutputIndex();
  if (!resultIndexOr)
    return failure();

  // A `!mgp.tensor` value is a parametric `OwnedTensor[dtype, rank]`; its
  // MoggAsyncPackHelper constructor would lose overload resolution to the
  // generic `Some[Movable & ImplicitlyDeletable]` ctor (mis-packing it as an
  // opaque Mojo value). Call the dedicated pack function directly instead,
  // passing the true spec rank so a scalar (rank 0) is not packed as the
  // promoted rank-1 buffer view.
  if (auto tensorType = dyn_cast<MGP::DeviceTensorType>(getInput().getType())) {
    MGP::TensorSpecType tensorSpec = tensorType.getTensorSpec();
    if (!tensorSpec.hasRank())
      return emitError() << "Expected Tensor output to have known rank";
    MojoValue specRankParam(static_cast<uint64_t>(tensorSpec.getRank()));
    MojoValue tensorPackFunc(kFnMoggAsyncPackOwnedTensor);

    state.builder.call(tensorPackFunc, {specRankParam},
                       {state.getMojoValueFor(getInput()),
                        state.mojoReturnValues[*resultIndexOr]});
    return success();
  }

  MojoValue asyncPackHelperStruct = kStructMoggAsyncPackHelper;
  auto packPrimDecl = state.registry.lookupPrimitive(this->getOperationName());
  ASSERT_STREAM(packPrimDecl,
                "Primitive not found for " << getOperationName().str());
  MojoValue moggAsyncPackFunc(primitiveMojoName(packPrimDecl));
  SmallVector<MojoValue> operands;
  for (auto [idx, operand] : llvm::enumerate(getOperands())) {
    MojoValue mojoVal = state.getMojoValueFor(operand);
    // The first operand (input) is consumed for move-only opaque types, so
    // transfer (^ move) those. Buffer/tensor composites are NOT transferred:
    // their pack copies (retains) the storage handle, so they may be borrowed
    // (e.g. an Array element from bulk_slice, which cannot be moved out).
    if (idx == 0 && isa<MO::OpaqueType, MGP::OpaqueType>(operand.getType()))
      mojoVal = mojoVal.transfer();
    operands.push_back(mojoVal);
  }
  operands.push_back(state.mojoReturnValues[*resultIndexOr]);
  state.builder.call(moggAsyncPackFunc, {},
                     asyncPackHelperStruct({}, operands));
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
// TensorArgInitOp
//===----------------------------------------------------------------------===//

// Inline Mojo definitions for the dense trait tensor-argument struct.
//
// We do not land a committed library type yet; instead we emit a single struct
// definition directly into the generated Mojo module (D9 "DenseTensorArg only
// inside the emitted mojo"). ONE struct serves both CPU and GPU (mirroring the
// shared `EmitMojoDenseTensor` in tensor-trait-mockups/dense_tensor_args.mojo):
// the storage is always a pointer (`tile_tensorable`), and
// `_is_device_passable` selects whether the value also crosses to the device.
// CPU passes `device_passable = False` and runs the host `TileTensorable`
// projection; GPU passes `device_passable = True`, and `_to_device_type`
// reinterprets the same pointer into a device-side `TileTensor` (valid on
// unified memory -- the macOS pass-1 path; a real `DevicePointer` flavor is a
// later slice). Emitted once per module, guarded by
// `EmitMojoState::emittedDenseTensorArgDefs`.
static constexpr llvm::StringLiteral kDenseTensorArgDefs = R"mojo(
from layout import Coord, TensorLayout, TileTensor
from extensibility import (
    DenseTensor,
    Input,
    MutableInput,
    Output,
    TileTensorable,
)
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


@fieldwise_init("implicit")
struct _MoggTensorIO(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    var value: Int

    comptime INPUT = _MoggTensorIO(0)
    comptime OUTPUT = _MoggTensorIO(1)
    comptime MUTABLE_INPUT = _MoggTensorIO(2)

    @always_inline("builtin")
    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    @always_inline("builtin")
    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    @always_inline("builtin")
    def is_input(self) -> Bool:
        return self == _MoggTensorIO.INPUT or self == _MoggTensorIO.MUTABLE_INPUT

    @always_inline("builtin")
    def is_mut(self) -> Bool:
        return self == _MoggTensorIO.OUTPUT or self == _MoggTensorIO.MUTABLE_INPUT


struct _MoggDenseTensorArg[
    _dtype: DType,
    _LayoutType: TensorLayout,
    _alignment: Int,
    _io: _MoggTensorIO,
    _is_device_passable: Bool,
](
    DenseTensor,
    DevicePassable where _is_device_passable,
    Input where _io == _MoggTensorIO.INPUT,
    MutableInput where _io == _MoggTensorIO.MUTABLE_INPUT,
    Output where _io == _MoggTensorIO.OUTPUT,
    # Host-only projection: a device-passable arg must NOT expose the host
    # `to_tile_tensor()` -- it would build a tile over the untranslated host
    # handle, valid only on unified memory. On the device path the kernel
    # consumes `device_type` (the translated device `TileTensor`) instead.
    TileTensorable where not _is_device_passable,
):
    comptime mut = Self._io.is_mut()
    comptime dtype = Self._dtype
    comptime LayoutType = Self._LayoutType
    comptime ShapeType = Coord[*Self.LayoutType._shape_types]
    comptime rank = Self.LayoutType.rank
    comptime alignment = Self._alignment
    var handle: UnsafePointer[Scalar[Self.dtype], UntrackedOrigin[mut = Self.mut]]
    var _layout: Self.LayoutType

    @always_inline
    def __init__(
        out self,
        ptr: UnsafePointer[NoneType, MutAnyOrigin],
        layout: Self.LayoutType,
    ):
        self.handle = ptr.bitcast[Scalar[Self.dtype]]().mut_cast[
            Self.mut
        ]().unsafe_origin_cast[UntrackedOrigin[mut = Self.mut]]()
        self._layout = layout

    # The device representation is a `TileTensor` (mirrors the mockup
    # `EmitMojoDenseTensor`): the device kernel receives a real `TileTensor`
    # (Coord-indexed load/store), not a bare pointer. `_to_device_type`
    # TRANSLATES the buffer pointer to its device address first (a bit-copy of a
    # `TileTensor` over the host handle would NOT translate -- it works only on
    # unified memory), then builds the device-side `TileTensor` from the
    # translated pointer and encodes it. The graph must transfer the buffer to
    # the device (see the GPU e2e tests' `mo.transfer`) so this address is valid.
    comptime device_type: AnyType = TileTensor[
        dtype = Self.dtype,
        LayoutType = Self.LayoutType,
        origin = UntrackedOrigin[mut = Self.mut],
    ]

    @always_inline
    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ) where Self._is_device_passable:
        var device_ptr = UnsafePointer[
            Scalar[Self.dtype], UntrackedOrigin[mut = Self.mut]
        ].unsafe_dangling()
        self.handle._to_device_type(
            encoder, UnsafePointer(to=device_ptr).bitcast[NoneType]()
        )
        var device_value = TileTensor[
            dtype = Self.dtype,
            LayoutType = Self.LayoutType,
            origin = UntrackedOrigin[mut = Self.mut],
        ](ptr=device_ptr, layout=self.layout())
        encoder.encode(device_value, target)

    @staticmethod
    def get_type_name() -> String where Self._is_device_passable:
        return String("_MoggDenseTensorArg")

    @always_inline
    def shape(self) -> Self.ShapeType:
        return rebind[Self.ShapeType](self.layout().shape_coord())

    @always_inline
    def layout(self) -> Self.LayoutType:
        return self._layout

    @always_inline
    def to_tile_tensor(
        self,
    ) -> TileTensor[
        dtype = Self.dtype,
        LayoutType = Self.LayoutType,
        origin = UntrackedOrigin[mut = Self.mut],
    ] where not Self._is_device_passable:
        return TileTensor[
            dtype = Self.dtype,
            LayoutType = Self.LayoutType,
            origin = UntrackedOrigin[mut = Self.mut],
        ](ptr=self.handle, layout=self.layout())
)mojo";

/// Assemble an execute-facing `_MoggDenseTensorArg` from a stub-bound buffer
/// capture and the distilled contract role.
static MojoValue assembleDenseTensorArgFromCapture(EmitMojoState &state,
                                                   MojoValue capture,
                                                   TensorArgRole role,
                                                   MO::TensorType tensorTy,
                                                   bool devicePassable) {
  if (!state.emittedDenseTensorArgDefs) {
    state.structDefsBuilder.verbatim(MojoValue(kDenseTensorArgDefs.str()));
    state.emittedDenseTensorArgDefs = true;
  }
  auto capName = capture.text;
  MojoValue dTypeParam(llvm::formatv("type_of({0}).dtype", capName).str());
  MojoValue layoutTypeParam(
      llvm::formatv("type_of({0}).LayoutType", capName).str());
  MojoValue alignmentParam(
      static_cast<uint64_t>(M::Driver::getPreferredAlignment(
          MO::getDeviceOrDefaultAny(tensorTy).getLabel().str())));
  MojoValue ioParam = [&] {
    MojoValue ioStruct("_MoggTensorIO");
    switch (role) {
    case TensorArgRole::input:
      return ioStruct.get("INPUT");
    case TensorArgRole::output:
      return ioStruct.get("OUTPUT");
    case TensorArgRole::mutableInput:
      return ioStruct.get("MUTABLE_INPUT");
    case TensorArgRole::none:
      llvm_unreachable("shape-function args are not assembled from capture");
    }
    llvm_unreachable("unhandled TensorArgRole");
  }();
  MojoValue devicePassableParam(devicePassable);
  MojoValue ptrArg = state.builder.var(
      llvm::formatv("{0}.handle.bitcast[NoneType]().as_unsafe_any_origin()",
                    capName)
          .str());
  MojoValue layoutArg =
      state.builder.var(llvm::formatv("{0}.layout()", capName).str());
  MojoValue structName("_MoggDenseTensorArg");
  return state.builder.var(
      structName({dTypeParam, layoutTypeParam, alignmentParam, ioParam,
                  devicePassableParam},
                 {ptrArg, layoutArg}));
}

/// If `val` is a trait stub buffer-capture block argument, assemble the dense
/// execute-facing arg implied by `contract`; otherwise return the existing mojo
/// value unchanged.
static MojoValue
maybeAssembleDenseTraitArgFromCapture(EmitMojoState &state, Value val,
                                      const TensorArgContract &contract,
                                      KernelStubOp stub) {
  if (contract.fused || contract.role == TensorArgRole::none)
    return state.getMojoValueFor(val);
  auto blockArg = dyn_cast<BlockArgument>(val);
  if (!blockArg || blockArg.getOwner() != &stub.getBodyBlock())
    return state.getMojoValueFor(val);
  auto tensorTy = MO::dynCastFromTensorLike(blockArg.getType());
  if (!tensorTy)
    return state.getMojoValueFor(val);
  bool devicePassable = !MO::isOnHost(stub.getOperation());
  return assembleDenseTensorArgFromCapture(state, state.getMojoValueFor(val),
                                           contract.role, tensorTy,
                                           devicePassable);
}

LogicalResult TensorArgInitOp::emitMojo(EmitMojoState &state) {
  auto tensorTy = MO::dynCastFromTensorLike(getType());
  if (!tensorTy)
    return emitOpError("expected a tensor-like result type");

  MojoValue dTypeParam = state.mlirDTypeToMojoType(tensorTy.getDtype());
  auto shapeVar = state.builder.var(
      state.getShapeAsCoordMojoValue(tensorTy.getTypedShapeAttr(), getDims()));
  auto layoutVar = state.builder.var(MojoValue(kFnRowMajor)({}, {shapeVar}));
  MojoValue layoutType = MojoValue("type_of")({}, {layoutVar});

  // Shape-function no-role args keep the legacy `_MoggDenseTensorArg` path: the
  // stub parameter stays generic `DenseTensor` and no buffer-capture assembly
  // is needed before `call.shape`.
  if (getContract().getRole() == MOGG::TensorArgRoleKind::none) {
    bool devicePassable = false;
    if (auto stub = (*this)->getParentOfType<MOGG::KernelStubOp>())
      devicePassable = !MO::isOnHost(stub.getOperation());
    if (!state.emittedDenseTensorArgDefs) {
      state.structDefsBuilder.verbatim(MojoValue(kDenseTensorArgDefs.str()));
      state.emittedDenseTensorArgDefs = true;
    }
    MojoValue alignmentParam(
        static_cast<uint64_t>(M::Driver::getPreferredAlignment(
            MO::getDeviceOrDefaultAny(tensorTy).getLabel().str())));
    MojoValue ioParam = MojoValue("_MoggTensorIO").get("INPUT");
    MojoValue ptr = OpaquePointerRef(ManagedTensorSliceRef(
                                         state.getMojoValueFor(getBuffer()))
                                         .unsafePtr())
                        .bitcast(MojoValue(kComptimeDTypeNoneType))();
    MojoValue structName("_MoggDenseTensorArg");
    state.setMojoValueFor(
        getResult(),
        state.builder.var(structName({dTypeParam, layoutType, alignmentParam,
                                      ioParam, MojoValue(devicePassable)},
                                     {ptr, layoutVar})));
    return success();
  }

  // Region→stub boundary: emit minimal buffer capture (pointer + layout).
  if (!state.emittedTensorBufferCaptureDefs) {
    state.structDefsBuilder.verbatim(MojoValue(kTensorBufferCaptureDefs.str()));
    state.emittedTensorBufferCaptureDefs = true;
  }
  MojoValue rawUnsafePtr =
      ManagedTensorSliceRef(state.getMojoValueFor(getBuffer())).unsafePtr();
  MojoValue ptr = state.builder.var(
      llvm::formatv("{0}.bitcast[Scalar[{1}]]().mut_cast[True]()"
                    ".unsafe_origin_cast[MutUntrackedOrigin]()",
                    rawUnsafePtr.text, dTypeParam.text)
          .str());
  MojoValue structName("_MoggTensorBufferCapture");
  state.setMojoValueFor(getResult(),
                        state.builder.var(structName({dTypeParam, layoutType},
                                                     {ptr, layoutVar})));

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

  // The MAP ABI is the advanced-fusion outlining path (a stub outlined from a
  // `mogg._kernel`); it reuses `mogg.call.stub` as its call site.
  if (calleeFunc->getAbi() != ABIKind::MOGG &&
      calleeFunc->getAbi() != ABIKind::MAP)
    return emitError("Call site should be in the MOGG or MAP ABI");

  return success();
}

bool CallStubOp::isThrows() {
  auto callee = getCallee();
  return callee && callee.isThrows();
}

// Projects a stub-call operand to the bare view the kernel consumes: a
// `!mgp.buffer` / `!mgp.tensor` value is a storage-carrying composite
// (`OwnedByteBuffer` / `OwnedTensor`), so unwrap its `.view` / `.tensor` before
// passing it to the kernel stub.
static SmallVector<MojoValue>
getStubCallOperands(EmitMojoState &state, Operation::operand_range ops) {
  SmallVector<MojoValue> operands;
  operands.reserve(llvm::range_size(ops));
  for (Value operand : ops) {
    MojoValue mojoVal = state.getMojoValueFor(operand);
    if (isa<MGP::DeviceBufferType>(operand.getType()))
      mojoVal = OwnedByteBufferRef(mojoVal).view();
    else if (isa<MGP::DeviceTensorType>(operand.getType()))
      mojoVal = OwnedTensorRef(mojoVal).tensor();
    operands.push_back(mojoVal);
  }
  return operands;
}

LogicalResult CallStubOp::emitMojo(EmitMojoState &state) {
  state.builder.newline();
  MojoValue func(getName());
  if (getNumResults()) {
    state.setMojoValueFor(
        getResult(0),
        state.builder.var(func({}, getStubCallOperands(state, getOperands()))));
  } else {
    state.builder.call(func, {}, getStubCallOperands(state, getOperands()));
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
  SmallVector<MojoValue> stubArgs = getStubCallOperands(state, getOperands());
  MojoValue stubFunc(getCallee().getSymNameAttr().str());

  MojoValue wrapper = state.builder.lambda(
      "wrapper", /*parameters=*/{}, /*arguments=*/{},
      /*returnType=*/MojoValue(kComptimeNone),
      /*isInline=*/true, /*isParameter=*/true, /*isAsync=*/true,
      /*captureConvention=*/"", [&](const FunctionBody &) {
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
  auto maybeParams = inferKernelParameters(
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
  auto maybeParams = inferKernelParameters(ewAdaptor, ewFuncLabel, inferParam);
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
  const auto &fusionInfo = loweringIt->second;

  // Trait lane (Phase 5): a load/store-contract fusion functor (emitted by
  // `emitInputLoadFusionFunctor` / `emitOutputStoreFusionFunctor` /
  // `emitTransformFusionFunctor`) is wrapped in `_MoggInputLoadFusionTensor` /
  // `_MoggOutputStoreFusionTensor` / `_MoggTransformFusionTensor` -- carrying
  // graph-visible access markers (`LoadAccess` / `StoreAccess` /
  // `TransformAccess`) and enabling callable access only on the appropriate
  // view -- and passed straight to `execute`. NO `_bind_to_fused_*` on a base
  // slice. The transform wrapper also holds the real output buffer tile (so the
  // parent kernel stores the transformed value) -- one operand, not two.
  {
    StringRef wrapperName;
    TensorArgAccess access = TensorArgAccess::none;
    if (std::optional<TensorArgContract> c =
            inputLoadFusionContract(state, *this);
        c && c->fused && c->hasAccess(TensorArgAccess::load)) {
      wrapperName = "_MoggInputLoadFusionTensor";
      access = TensorArgAccess::load;
    } else if (std::optional<TensorArgContract> o =
                   outputFusionContract(state, *this)) {
      // Single walk of `TensorFusedOp -> CallExecuteOp -> output-operand
      // index -> kernel-decl arg`; dispatch store vs. transform off the same
      // cached contract.
      if (o->fused && o->hasAccess(TensorArgAccess::store)) {
        wrapperName = "_MoggOutputStoreFusionTensor";
        access = TensorArgAccess::store;
      } else if (o->fused && o->hasAccess(TensorArgAccess::transform)) {
        wrapperName = "_MoggTransformFusionTensor";
        access = TensorArgAccess::transform;
      }
    }
    if (!wrapperName.empty()) {
      auto tensorTy = MO::castFromTensorLike(getResult().getType());
      MojoValue alignmentParam(
          static_cast<uint64_t>(M::Driver::getPreferredAlignment(
              MO::getDeviceOrDefaultAny(tensorTy).getLabel().str())));
      MojoValue functorType = MojoValue("type_of")({}, {fusionInfo.instance});
      SmallVector<MojoValue> comptimeParams;
      SmallVector<MojoValue> ctorArgs;
      if (access == TensorArgAccess::transform) {
        // The transform wrapper composes the transform with the real output
        // buffer (the parent stores the transformed value through it). All
        // tile-related params (dtype / layout / mut) come from the output base
        // arg so the wrapper's `_tile` field unifies with
        // `base.to_tile_tensor()`
        // -- the base's dtype/layout/mut are symbolic on the generic stub arg,
        // so MLIR-derived params would not match.
        MojoValue base = state.getMojoValueFor(getTensor());
        auto baseName = base.text;
        auto layoutArg = state.builder.var(
            MojoValue(llvm::formatv("{0}.layout()", baseName).str()));
        // Bare pointer off the output base's stored handle (NOT the tile
        // itself -- the wrapper rebuilds the tile in `to_tile_tensor()` so it
        // translates to a device address).
        auto ptrArg = state.builder.var(
            MojoValue(llvm::formatv("{0}.handle", baseName).str()));
        bool devicePassable = false;
        if (auto stub = (*this)->getParentOfType<MOGG::KernelStubOp>())
          devicePassable = !MO::isOnHost(stub.getOperation());
        bool accessEnabled = !devicePassable;
        comptimeParams = {
            MojoValue(llvm::formatv("type_of({0}).dtype", baseName).str()),
            MojoValue(llvm::formatv("type_of({0}).LayoutType", baseName).str()),
            alignmentParam,
            traitCaptureMutComptimeExpr(getTensor(), baseName),
            functorType,
            MojoValue(devicePassable),
            MojoValue(accessEnabled)};
        ctorArgs = {fusionInfo.instance, layoutArg, ptrArg};
      } else {
        // Load / store: layout metadata comes from the fused buffer capture on
        // the stub boundary. For store, `wireTraitFusedCapturesToStubArgs`
        // assigns `getTensor()` to that capture; for load, `getTensor()` stays
        // the empty companion placeholder and the capture lives on the bind's
        // fusible operand(s).
        Value layoutSource = getTensor();
        if (access == TensorArgAccess::load) {
          if (auto bindOp = getBindLambdaOp())
            if (!bindOp.getFusibleTensors().empty())
              layoutSource = bindOp.getFusibleTensors().front();
        }
        MojoValue capture = state.getMojoValueFor(layoutSource);
        auto captureName = capture.text;
        auto layoutArg = state.builder.var(
            MojoValue(llvm::formatv("{0}.layout()", captureName).str()));
        bool devicePassable = false;
        if (auto stub = (*this)->getParentOfType<MOGG::KernelStubOp>())
          devicePassable = !MO::isOnHost(stub.getOperation());
        bool accessEnabled = !devicePassable;
        comptimeParams = {
            MojoValue(llvm::formatv("type_of({0}).dtype", captureName).str()),
            MojoValue(
                llvm::formatv("type_of({0}).LayoutType", captureName).str()),
            alignmentParam,
            functorType,
            MojoValue(devicePassable),
            MojoValue(accessEnabled)};
        ctorArgs = {fusionInfo.instance, layoutArg};
      }
      MojoValue structName(wrapperName.str());
      state.setMojoValueFor(
          getResult(), state.builder.var(structName(comptimeParams, ctorArgs)));
      return success();
    }
  }

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
    // Build a `DeviceContextArray[N]` directly from the per-device
    // contexts via its variadic constructor.
    MojoValue deviceContextArrayConstructor = MojoValue(
        kStructDeviceContextArray)[{
        MojoValue(kParameterLength).equal(std::to_string(inputValues.size()))}];
    MojoValue inputTupleValue = deviceContextArrayConstructor({}, inputValues);
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
      const auto &fusionInfo = it->second;
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
      const auto &fusionInfo = fusIt->second;
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
  auto symAttr = getKernelSymNameAttr();
  auto decl =
      symbolTable.lookupNearestSymbolFrom<ShapeFunctionDeclOp>(*this, symAttr);
  if (!decl)
    return emitOpError("'")
           << symAttr.getValue() << "' does not refer to a shape_func.decl op";
  return success();
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
  // Rebind all inputs to ensure specs are deduced (tensors -> InputTensor,
  // etc.)
  SmallVector<MojoValue> kernelOperands;
  for (Value operand : getOperands())
    kernelOperands.push_back(state.getMojoValueFor(operand));

  // Infer any kernel parameters required by the shape function.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    return state.inferDefaultKernelParamValue(*this, paramName);
  };
  ShapeFunctionDeclOp shapeDecl =
      state.registry.lookupShapeFunctionBySymName(getKernelSymName());

  auto maybeParams = inferKernelParameters(shapeDecl, inferParam);
  if (maybeParams.isError())
    return emitError() << "Failed to infer all the kernel parameters: "
                       << maybeParams.takeError();

  // A trait-typed shape function takes its tensor arguments as the no-role
  // `_MoggDenseTensorArg` (constructed by `mogg.tensor_arg.__init__`) instead
  // of a legacy `InputTensor`. Its operands therefore reach the shape kernel
  // stub as a *generic* trait parameter (`mogg.trait_stub_arg_bounds`), so the
  // call result is `IndexList[T.rank]` with `T` generic -- not statically the
  // op's output rank. The stub's declared result is the concrete
  // `IndexList[output_rank]`, so we rebind to it.
  //
  // The rebind target is the authoritative source: `output_rank` is the op's
  // declared output rank (the `shape_func_result<N>` rank, set from
  // `InvokeShapeFuncOp::getOutputParamDecls().size()` in `lowerToShapeKernel`),
  // and a shape function returns exactly the op's output shape -- so the
  // runtime value always has that many dims, regardless of arity or how the
  // shape fn derives its rank from the inputs. (Deriving the rank from an
  // argument instead would be wrong for multi-input or rank-changing shape
  // functions: the output rank is a property of the op, not of any one
  // argument.) This is the same `N` the legacy lane already trusts when it
  // unpacks `N` dims out of `shape_func_result<N>` via `ShapeResultGetItemOp`.
  //
  // Legacy `InputTensor` shape functions already produce a concretely-ranked
  // `IndexList`, so they skip the rebind (and their emitted text is
  // byte-for-byte unchanged).
  bool isTraitTyped = false;
  if (mlir::ArrayAttr contracts =
          shapeDecl.getFunction().getTensorArgContracts())
    for (mlir::Attribute entry : contracts)
      if (mlir::isa<mlir::DictionaryAttr>(entry)) {
        isTraitTyped = true;
        break;
      }

  MojoValue shapeFn(getKernelSymName());
  MojoValue result = state.builder.var(shapeFn[maybeParams->structParameters](
      maybeParams->functionParameters, kernelOperands));
  if (isTraitTyped) {
    auto shapeTy =
        mlir::dyn_cast<MOGG::ShapeFuncResultType>(getRetTypeAttr().getValue());
    if (!shapeTy)
      return emitOpError("expected a ShapeFuncResultType return type");
    result = state.builder.var(
        MojoValue(llvm::formatv("rebind[{0}[{1}]]({2})", kStructIndexList,
                                shapeTy.getRank(), result.text)
                      .str()));
  }
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
  if (auto literal = scalarConstantAttrToMojoLiteral(getValue())) {
    state.setMojoValueFor(getResult(), *literal);
    return success();
  }
  return emitError("TODO: unsupported scalar type");
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
    p << ' ';
  }
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

void MOGGDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GraphCompiler/MOGGDialect/MOGG.cpp.inc"
      >();
}

using PointerType = KGEN::PointerType;
#define GET_OP_CLASSES
#include "GraphCompiler/MOGGDialect/MOGG.cpp.inc"
