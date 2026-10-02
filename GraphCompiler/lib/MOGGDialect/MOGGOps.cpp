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
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "GraphCompiler/MOGGDialect/Support/MojoTraits.h"
#include "GraphCompiler/MOGGDialect/Support/ProfileScopeLabels.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/Support/NameMangling.h"
#include "Support/AssertStream.h"
#include "Support/DeviceSpecs.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
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

/// Fusion-struct body: loads through `self.{ptr,shape,strides}_<i>` on
/// the struct's `simd_width` method parameter.
struct FusionStructRenderer final : ElementwiseBodyRenderer {
  std::string buildLoadExpr(unsigned captureIdx,
                            StringRef indicesExpr) const override {
    return buildStructLoadExpr(std::to_string(captureIdx), indicesExpr);
  }
  StringRef simdWidthParamName() const override { return "simd_width"; }
};

/// Trait-lane fusion-functor body: loads through a dense `TileTensor` built
/// from capture <i>'s `(ptr, layout)` via `tile.load(idx)`. The functor
/// `__call__`'s width param is `width` and `idx` is a `Coord`; loads are at the
/// iteration index (`indicesExpr` is ignored, the only form simple elementwise
/// prologue fusion produces).
struct DenseFusionRenderer final : ElementwiseBodyRenderer {
  std::string buildLoadExpr(unsigned captureIdx,
                            StringRef /*indicesExpr*/) const override {
    // Captures are a bare pointer + layout, not a `TileTensor`: the encoder
    // bit-copies fields, so only a `Pointer` translates to a device address
    // (via `_to_device_type`). Rebuilding the tile here lets one body serve CPU
    // and GPU.
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

  SmallVector<Value> loweredOperands = opLoweringState.getLoweredOperands();

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

  Value inputBuffer = opLoweringState.getBufferizedInput(getInput()).buffer;

  // TODO Enable this.
  if (inputBuffer.getDefiningOp<MGP::BufferConstantOp>())
    return emitOpError("constant input buffer is unsupported");

  // A split needs output specs only, no buffers.
  FailureOr<SmallVector<TypedValue<MGP::TensorSpecType>>> outputSpecsOr =
      opLoweringState.createOutputTensorSpecs();
  if (failed(outputSpecsOr))
    return failure();
  auto &outputSpecs = *outputSpecsOr;

  SmallVector<Value> outputBuffers;

  // The slices stay on the input buffer's device.
  auto sliceDeviceRef =
      cast<MGP::DeviceBufferType>(inputBuffer.getType()).getDevice();
  TypedValue<MGP::DeviceContextType> sliceDeviceContext =
      loweringState.getExecuteDevice(sliceDeviceRef);

  Value offset = opLoweringState.create<mlir::index::ConstantOp>(0).getResult();
  Value sliceSize;
  for (TypedValue<MGP::TensorSpecType> outSpec : outputSpecs) {
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

  // The slices are only accessible once the inputs are.
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
  MGP::MGPBufferBuilder &bufferBuilder =
      loweringState.getBufferBuilder(MO::getDeviceOrDefaultAny(*this));
  Value outChain = bufferBuilder.moduleLoad(
      builder, getLoc(), loweringState.getInitMgpContext(),
      // DANGER: the init chain comes from device context creation, which module
      // load doesn't depend on (and which later becomes another module load).
      // Module loads must be ordered by creation, so depend only on runtime
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

// Returns the fully-qualified Mojo source name for a PrimitiveDeclOp, which
// ImportPackages stores as its sym_name.
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

//===----------------------------------------------------------------------===//
// CallMaterializeOp
//===----------------------------------------------------------------------===//

ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
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
  if (parser.resolveOperands(inputOperands, inputTypes, operandsLoc,
                             result.operands))
    return failure();
  int32_t numInputOperands = static_cast<int32_t>(result.operands.size());

  // Optional chain (%c) clause -- a plain operand, never a body block
  // argument (mirrors `map.iter.foreach`'s `chain(...)`).
  int32_t numChainOperands = 0;
  if (succeeded(parser.parseOptionalKeyword("chain"))) {
    OpAsmParser::UnresolvedOperand chainOperand;
    if (parser.parseLParen() || parser.parseOperand(chainOperand) ||
        parser.parseRParen())
      return failure();
    if (parser.resolveOperand(chainOperand,
                              MO::ChainType::get(parser.getContext()),
                              result.operands))
      return failure();
    numChainOperands = 1;
  }
  result.addAttribute(KernelOp::getOperandSegmentSizesAttrName(result.name),
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {numInputOperands, numChainOperands}));

  // Results: an optional `-> types`.
  SmallVector<Type> resultTypes;
  if (parser.parseOptionalArrowTypeList(resultTypes))
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

void KernelOp::print(OpAsmPrinter &p) {
  // Op operands: the `!mo.tensor` inputs, printed bare.
  p << " (";
  p.printOperands(getInputs());
  if (!getInputs().empty()) {
    p << " : ";
    llvm::interleaveComma(getInputs().getTypes(), p);
  }
  p << ")";

  // chain (%c) -- optional plain operand, no block arg (mirrors
  // `map.iter.foreach`'s `chain(...)`).
  if (Value chain = getChain())
    p << " chain(" << chain << ")";

  // Results: `-> types` (omitted when there are none).
  if (!getOutputs().empty()) {
    SmallVector<Type> outTypes(getOutputs().getTypes());
    p.printArrowTypeList(outTypes);
  }

  // Body entry-block arguments, as labeled groups. The in-memory order is
  // [inputs, outputs, dims, context]; the display shows context before dims.
  // Only destination-passing outputs get a body block argument: not the
  // trailing chain, and not a result that leaves by value.
  auto args = getBody().front().getArguments();
  size_t numInputs = getInputs().size();
  size_t numOutputs = 0;
  for (Value output : getOutputs())
    if (MO::dynCastFromTensorLike(output.getType()))
      ++numOutputs;
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

  // Elided when empty.
  SmallVector<StringRef> elided{getOperandSegmentSizesAttrName().getValue()};
  if (getOutputParamDecls().empty())
    elided.push_back(getOutputParamDeclsAttrName().getValue());
  // Async is the default; only a sync kernel prints `isAsync = false`.
  if (getIsAsync())
    elided.push_back(getIsAsyncAttrName().getValue());
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(), elided);
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult KernelOp::verify() {
  // A chain result, if present, is last (mirrors `IterForeachOp::verify()` /
  // `KernelOp::maybeGetOutChain()`) and never gets a body block argument.
  mlir::ResultRange outputs = getOutputs();
  for (auto [i, output] : llvm::enumerate(outputs))
    if (isa<MO::ChainType>(output.getType()) && i + 1 != outputs.size())
      return emitOpError() << "only the last output may be a chain";

  // A by-value opaque result is allocated inside the kernel body and returned;
  // `lower` maps a single one onto the region's results.
  if (llvm::count_if(outputs, [](Value output) {
        return isa<MO::OpaqueType>(output.getType());
      }) > 1)
    return emitOpError("returns more than one opaque result");

  // The body block arguments are the codegen handles: one per input then
  // destination-passing output, then the !mogg.dims, then one !mogg.context.
  // A tensor-like entry gets a !mogg.pointer; an opaque input carries its own
  // type across. Results that leave by value get none -- a shape function's
  // `!mogg.shape_func_result`, or an `!mo.opaque`.
  SmallVector<Type> boundaryTypes(getInputs().getTypes());
  for (Value output : outputs)
    if (MO::dynCastFromTensorLike(output.getType()))
      boundaryTypes.push_back(output.getType());
  const size_t numTensors = boundaryTypes.size();
  auto args = getBody().front().getArguments();

  if (args.size() < numTensors + 1)
    return emitOpError() << "expected at least " << (numTensors + 1)
                         << " body block arguments (one per input "
                            "and output plus a trailing !mogg.context), "
                            "got "
                         << args.size();

  for (size_t i = 0; i < numTensors; ++i) {
    if (isa<MO::OpaqueType>(boundaryTypes[i])) {
      if (args[i].getType() != boundaryTypes[i])
        return emitOpError()
               << "body block argument #" << i << " must be "
               << boundaryTypes[i] << " to match its opaque input/output, got "
               << args[i].getType();
      continue;
    }
    if (!isa<PointerType>(args[i].getType()))
      return emitOpError() << "body block argument #" << i
                           << " must be a !mogg.pointer (one per input then "
                              "output tensor)";
  }

  size_t i = numTensors;
  while (i < args.size() && isa<DimType>(args[i].getType()))
    ++i;
  // After the dims come one or more trailing `!mogg.context`: exactly one for
  // an ordinary kernel, one per device for a distributed kernel that takes a
  // `DeviceContextArray`.
  if (i >= args.size())
    return emitOpError() << "body block arguments after the pointers must be "
                            "zero or more !mogg.dim followed by one or more "
                            "!mogg.context";
  for (; i < args.size(); ++i)
    if (!isa<ContextType>(args[i].getType()))
      return emitOpError() << "body block argument #" << i
                           << " must be a !mogg.context (the trailing args "
                              "after the dims are all contexts)";

  // One kernel, one call -- counting the `mogg.call.stub` outlining leaves.
  size_t numCalls = 0;
  for (Operation &bodyOp : getBody().front())
    if (isa<CallOpaqueOp, CallForeachOp, CallElementwiseOp, CallShapeOp,
            CallStubOp>(bodyOp))
      ++numCalls;
  if (numCalls != 1)
    return emitOpError() << "expected exactly one call in the body, got "
                         << numCalls;

  // A shape function returns exactly the dims it declares.
  size_t numShapeReturning = 0;
  uint64_t shapeRank = 0;
  for (Operation &bodyOp : getBody().front())
    for (Value result : bodyOp.getResults())
      if (auto shapeTy = dyn_cast<ShapeFuncResultType>(result.getType())) {
        ++numShapeReturning;
        shapeRank = shapeTy.getRank();
      }
  if (numShapeReturning > 1)
    return emitOpError() << "expected at most one shape-returning call in the "
                            "body, got "
                         << numShapeReturning;
  if (getOutputParamDecls().empty() != (numShapeReturning == 0))
    return emitOpError() << "outputParamDecls and a shape-returning call must "
                            "appear together";
  if (numShapeReturning == 1 && shapeRank != getOutputParamDecls().size())
    return emitOpError() << "shape result rank " << shapeRank
                         << " does not match " << getOutputParamDecls().size()
                         << " declared output parameter(s)";

  return success();
}

Value KernelOp::getOutputChain() {
  mlir::ResultRange outputs = getOutputs();
  if (outputs.empty())
    return nullptr;
  Value last = outputs.back();
  return isa<MO::ChainType>(last.getType()) ? last : Value();
}

// Lowers a `mogg.kernel` into a `mogg.generic.compiled.region` around its
// already-outlined stub: materialize the region operands (input buffers, new
// output buffers, dims, device context), unpack + cast each async block arg to
// the stub's handle, and re-home the `mogg.call.stub`.
//
// TODO(GEX-3965): buffer allocation / context / dim materialization should move
// to dedicated pre-MGP passes, leaving this a genuinely thin lowering.
LogicalResult KernelOp::lower(MGP::MOGraphLoweringState &loweringState) {
  return lowerToCompiledRegion(loweringState) ? success() : failure();
}

MOGG::GenericCompiledRegionOp
KernelOp::lowerToCompiledRegion(MGP::MOGraphLoweringState &loweringState) {
  MGP::MOOpLoweringState st(loweringState, getOperation());
  MLIRContext *ctx = getContext();
  Location loc = getLoc();
  DeviceRefAttr device = MO::getDeviceOrDefaultAny(getOperation());

  // The body block args are, in order: one `!mogg.pointer` per input, then per
  // output, then the `!mogg.dim<name>`s, then one trailing `!mogg.context`. A
  // trailing output chain, like the (dedicated-operand) input chain, never
  // gets one of its own.
  Block &kernelBody = getBody().front();
  unsigned numInputs = getInputs().size();
  // Only tensor-like results are allocated and get an output pointer.
  unsigned numOutputs = 0;
  for (Value output : getOutputs())
    if (MO::dynCastFromTensorLike(output.getType()))
      ++numOutputs;
  unsigned numArgs = kernelBody.getNumArguments();
  if (numArgs < numInputs + numOutputs + 1) {
    emitOpError("MOGGToMGP: malformed mogg.kernel body");
    return nullptr;
  }

  // The op's own `$chain` orders it against the kernels that wrote its
  // buffers, so the region waits on it; the ambient execute chain would drop
  // that edge.
  Value inChain = loweringState.getCurrentExecuteChain();
  if (Value opChain = getChain()) {
    Value lowered = loweringState.lookupExecuteLoweredValue(opChain);
    ASSERT_STREAM(lowered, << "no lowered value for mogg.kernel input chain: "
                           << opChain);
    inChain = lowered;
  }
  inChain = st.getGuardChain(inChain);

  // Region operands (plain sync values) + their in/out signature. The chain
  // leads, so block arg i is operand i + 1.
  SmallVector<Value> operands{inChain};
  SmallVector<InOutSignatureAttr::InOutSemantics> semantics{
      InOutSignatureAttr::kIn};

  // Inputs lower to `!mgp.buffer`. The input chain has no body block argument
  // (it is threaded via `inChain`), so drop it to keep operands aligned with
  // the body. A `!mo.buffer` input is written in place, so it takes the
  // mutable signature (as an input MGP would treat it as read-only and lose
  // the write), read off the operand type since body args are type-erased
  // `!mogg.pointer`s.
  auto inputs = getInputs();
  unsigned inputIdx = 0;
  for (Value buffer : st.getLoweredOperands()) {
    if (isa<MEF::ChainType>(buffer.getType()))
      continue;
    operands.push_back(buffer);
    bool isMutable = inputIdx < inputs.size() &&
                     isa<MO::BufferType>(inputs[inputIdx].getType());
    semantics.push_back(isMutable ? InOutSignatureAttr::kMut
                                  : InOutSignatureAttr::kIn);
    ++inputIdx;
  }

  // Outputs: freshly-allocated buffers (externally-allocated, DPS).
  FailureOr<SmallVector<MGP::MOOutputStorage>> storageOr =
      st.createOutputStorage(inChain);
  if (failed(storageOr))
    return nullptr;
  SmallVector<MGP::MOOutputStorage> storage = *storageOr;
  // `createOutputStorage` skips chain-typed results, so it yields `numOutputs`
  // entries.
  assert(storage.size() == numOutputs &&
         "expected one output storage per non-chain mogg.kernel result");
  for (const MGP::MOOutputStorage &out : storage) {
    operands.push_back(out.buffer);
    semantics.push_back(InOutSignatureAttr::kOut);
  }

  // Dims: materialize each `!mogg.dim<name>` block arg from its param, up to
  // the trailing context(s).
  unsigned argIdx = numInputs + numOutputs;
  for (; argIdx < numArgs &&
         isa<MOGG::DimType>(kernelBody.getArgument(argIdx).getType());
       ++argIdx) {
    auto dimType =
        cast<MOGG::DimType>(kernelBody.getArgument(argIdx).getType());
    if (!dimType.getName()) {
      emitOpError("MOGGToMGP: mogg.kernel dim arg lacks a name");
      return nullptr;
    }
    ErrorOr<Value> dimValue = loweringState.lowerExpression(
        getOperation()->getParentRegion(), loc, dimType.getName());
    if (dimValue.isError()) {
      emitOpError("MOGGToMGP: ") << dimValue.getError();
      return nullptr;
    }
    operands.push_back(dimValue.takeValue());
    semantics.push_back(InOutSignatureAttr::kNone);
  }

  // Context(s): one `!mogg.context` block arg, or one per device for a
  // `DeviceContextArray` kernel in `getUniqueDevicesFromKernel` order (as
  // `IterOpaqueOp::lowerToMOGG` created them). Each is its own region operand,
  // though `ConformCallOperands` packed them into a tuple at the call.
  unsigned numContexts = numArgs - argIdx;
  if (numContexts <= 1) {
    operands.push_back(loweringState.getExecuteDevice(device));
    semantics.push_back(InOutSignatureAttr::kNone);
  } else {
    SmallVector<DeviceRefAttr> devices =
        MO::getUniqueDevicesFromKernel(getOperation());
    if (devices.size() != numContexts) {
      emitOpError("MOGGToMGP: mogg.kernel has ")
          << numContexts << " context args but " << devices.size()
          << " unique devices";
      return nullptr;
    }
    for (DeviceRefAttr dev : devices) {
      operands.push_back(loweringState.getExecuteDevice(dev));
      semantics.push_back(InOutSignatureAttr::kNone);
    }
  }

  // Result 0 is the chain, then one dim per declared parameter, then a
  // by-value `!mo.opaque` if there is one. Allocated inside the kernel, so it
  // leaves as a region result rather than an output buffer.
  ArrayRef<KGEN::ParamDeclAttr> decls = getOutputParamDecls();
  Type dimType = MOSH::ShapeAttr::getDimType(ctx);
  SmallVector<Type> regionResultTypes{MEF::ChainType::get(ctx)};
  regionResultTypes.append(decls.size(), dimType);

  Value opaqueKernelResult;
  for (Value output : getOutputs())
    if (isa<MO::OpaqueType>(output.getType()))
      opaqueKernelResult = output;
  const unsigned opaqueResultIdx = regionResultTypes.size();
  if (opaqueKernelResult)
    regionResultTypes.push_back(opaqueKernelResult.getType());

  OpBuilder &builder = st.builder;
  auto region = MOGG::GenericCompiledRegionOp::create(
      builder, loc, regionResultTypes, operands,
      /*outputParamDecls=*/KGEN::ParamDeclArrayAttr::get(ctx, decls),
      InOutSignatureAttr::get(ctx, semantics),
      /*fusionInfo=*/getFusionInfoAttr(),
      /*isAsync=*/BoolAttr::get(ctx, getIsAsync()),
      /*constantResources=*/ArrayAttr::get(ctx, {}),
      /*canFuseIntoOther=*/BoolAttr::get(ctx, !MO::isOnHost(getOperation())),
      /*canBeFusedInto=*/BoolAttr::get(ctx, true),
      /*canDuplicate=*/BoolAttr::get(ctx, false),
      /*isStructuredKernel=*/BoolAttr::get(ctx, true));
  region.setDeviceRef(device);

  // Region body: one `!mogg.async<T>` block arg per operand, same order.
  Block *regionBody = &region.getBody().emplaceBlock();
  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  for (Value operand : operands) {
    argTypes.push_back(MOGG::AsyncType::get(operand.getType()));
    argLocs.push_back(loc);
  }
  regionBody->addArguments(argTypes, argLocs);

  // Unpack + cast each async arg to the handle the stub expects, then re-home
  // the already-outlined `mogg.call.stub`. The leading chain arg has no stub
  // counterpart, hence the `+ 1`.
  builder.setInsertionPointToStart(regionBody);
  SmallVector<Value> stubArgs;
  for (unsigned i = 0; i < numArgs; ++i) {
    Value unpacked = MOGG::AsyncUnpackOp::create(
        builder, loc, regionBody->getArgument(i + 1));
    // `mogg.rebind` reconciles the unpacked MGP handle with the stub's codegen
    // handle. For `!mogg.pointer` it emits real Mojo (`OwnedByteBuffer` -> raw
    // opaque data pointer); context/dim casts are pass-throughs.
    // debt: drop once `mogg.async.unpack` folds directly to the typed handle.
    // A dim arg is cast to an anonymous `!mogg.dim`: the symbolic name blocks
    // kernel dedup, and the stub's dim args are stripped to match.
    Type handleType = kernelBody.getArgument(i).getType();
    if (isa<MOGG::DimType>(handleType))
      handleType = MOGG::DimType::get(ctx, /*name=*/{});
    Value handle = MOGG::RebindOp::create(builder, loc, handleType, unpacked);
    stubArgs.push_back(handle);
  }

  // Re-home `kernelBody`'s contents (the `mogg.call.stub` plus constants kept
  // at the call site) by cloning with block args mapped to `stubArgs`.
  // `mgp-anonymize-dim-names` strips the stub's dim names once all kernels are
  // lowered.
  IRMapping mapping;
  for (unsigned i = 0; i < numArgs; ++i)
    mapping.map(kernelBody.getArgument(i), stubArgs[i]);
  SmallVector<Attribute> blobs;
  Value shapeResult;
  Value opaqueResult;
  for (Operation &calleeOp : kernelBody) {
    // The MEF alignment is only known here, so swap a `mogg.buffer.constant`
    // for the real `mgp.buffer.constant` in this region: resources reach the
    // runtime through the MGP-ABI `void**`, which the MOGG-ABI stub lacks.
    if (auto cst = dyn_cast<MOGG::BufferConstantOp>(calleeOp)) {
      ErrorOr<ElementsAttr> blob =
          loweringState.lowerElementsAttr(cst.getValue());
      if (blob.isError()) {
        cst.emitOpError("MOGGToMGP: ") << blob.getError();
        return nullptr;
      }
      blobs.push_back(*blob);
      Value buffer =
          MGP::BufferConstantOp::create(builder, cst.getLoc(), *blob);
      mapping.map(cst.getPointer(),
                  MOGG::RebindOp::create(builder, cst.getLoc(),
                                         cst.getPointer().getType(), buffer));
      continue;
    }
    Operation *clone = builder.clone(calleeOp, mapping);
    // By type, so no unrelated result can stand in.
    for (Value result : clone->getResults()) {
      if (isa<MOGG::ShapeFuncResultType>(result.getType()))
        shapeResult = result;
      else if (isa<MO::OpaqueType>(result.getType()))
        opaqueResult = result;
    }
  }
  // Only the region carries the bytes out to the runtime; the stub reads them
  // as ordinary arguments, so its own array stays empty.
  if (!blobs.empty())
    region.setConstantResourcesAttr(builder.getArrayAttr(blobs));
  Value readyChain = MOGG::AsyncReadyOp::create(builder, loc);
  SmallVector<Value> regionOutputs{readyChain};
  if (!decls.empty()) {
    ASSERT_STREAM(shapeResult,
                  << "verifier guarantees a shape-returning call whenever "
                     "outputParamDecls is non-empty");
    // Unpack the `N` dims out of the `shape_func_result<N>`.
    for (size_t i = 0, e = decls.size(); i < e; ++i) {
      auto item = MOGG::ShapeResultGetItemOp::create(
          builder, loc, dimType, shapeResult,
          builder.getIntegerAttr(builder.getIntegerType(64, true), i));
      regionOutputs.push_back(
          MOGG::AsyncPackOp::create(builder, loc, item.getResult()));
    }
  }
  if (opaqueKernelResult) {
    if (!opaqueResult) {
      emitOpError("MOGGToMGP: mogg.kernel returns an opaque but its body "
                  "produces none");
      return nullptr;
    }
    regionOutputs.push_back(
        MOGG::AsyncPackOp::create(builder, loc, opaqueResult));
  }
  MOGG::OutputOp::create(builder, loc, regionOutputs);

  // Register the outputs against the graph, guarded by the region's out chain.
  builder.setInsertionPointAfter(region);

  // Bind each dim to the value the region returns. Order-independent:
  // `lowerParamDecl` back-patches any placeholder already handed out.
  for (auto [idx, decl] : llvm::enumerate(decls)) {
    Value asDimension = MGP::indexToDimension(
        loweringState.getExecuteOpBuilder(), loc, region->getResult(idx + 1));
    if (auto err = loweringState.lowerParamDecl(
            getOperation()->getParentRegion(), decl, asDimension)) {
      emitOpError("MOGGToMGP: failed to bind parameter ")
          << decl.getName() << ": " << err.getError();
      return nullptr;
    }
  }

  // No buffer storage for a by-value opaque, so `createOutputs` skips it: hand
  // the region's result straight to whoever consumes the kernel's.
  if (opaqueKernelResult)
    loweringState.emplaceExecuteLoweredValue(
        opaqueKernelResult, region->getResult(opaqueResultIdx));

  st.createOutputs(region->getResult(0), storage);

  // The region's chain result is the kernel's out chain too. Consumers look it
  // up by the kernel's own result, so without this an `mo.output` forwarding
  // that chain finds nothing lowered.
  if (Value outChain = getOutputChain())
    loweringState.emplaceExecuteLoweredValue(outChain, region->getResult(0));

  return region;
}

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

  ErrorOr<DType> dTypeOr = M::MO::fromMLIRType(tensorType.getElementType());
  ASSERT_STREAM(!dTypeOr.isError(),
                "Failed to extract dtype from mutable input tensor");
  MojoValue dtype = state.mlirDTypeToMojoType(*dTypeOr);

  ASSERT_STREAM(tensorType.hasRank(),
                "Mutable input tensor must have statically known rank");
  MojoValue rank(static_cast<int64_t>(tensorType.getRankNonZero()));

  ManagedTensorSliceRef mts(inputValue);
  StaticTensorSpecRef staticSpec(mts.staticSpec());

  // Build template parameters. The static layout is conveyed through the
  // layout argument's type (below), so no separate layout param.
  SmallVector<MojoValue> templateParams = {dtype, rank, MojoValue(false),
                                           MojoValue(kEnumIoInput),
                                           staticSpec.alignment()};

  // Build runtime arguments: the slice already carries its layout (shape and
  // strides as static-aware Coords), so forward it directly.
  SmallVector<MojoValue> runtimeArgs = {
      OpaquePointerRef(mts.rawPtr())
          .bitcast(MojoValue(kComptimeDTypeNoneType))(),
      inputValue.get("runtime_layout")()};

  auto tensorInitPrimDecl =
      state.registry.lookupPrimitive(TensorInitOp::getOperationName());
  MojoValue moggTensorInit(primitiveMojoName(tensorInitPrimDecl));
  return state.builder.var(moggTensorInit(templateParams, runtimeArgs));
}
} // namespace

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

  // `build_graph` records the dispatch into the builder; ordering comes from
  // the enclosing `mgp.device_graph.dependencies` op, so this is a void call.
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
static SmallVector<StringRef> collectSourceNotes(Location rootLoc) {
  llvm::SetVector<StringRef> notes;
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](mlir::FusedLoc loc) {
    if (auto metadata = dyn_cast_if_present<StringAttr>(loc.getMetadata()))
      if (auto val = metadata.getValue(); !val.empty())
        notes.insert(metadata.getValue());
  });
  walker.walk<mlir::WalkOrder::PreOrder>(rootLoc);
  return notes.takeVector();
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

/// Collect source notes for NVTX trace detail strings, each trace cut to its
/// bottom N frames (MODULAR_TRACE_MAX_FRAMES, default 10) in `path:line:func`
/// form. Non-empty only if `max-debug.source-tracebacks` was enabled at graph
/// construction (e.g. `MODULAR_DEBUG=source-tracebacks`).
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
  // FunctionOpInterface already verifies the arguments.
  return success();
}

bool KernelStubOp::isThrows() {
  // `AddTryExceptBlocks` always gives a MOGG stub the `mogg.raise` that renders
  // its error context. Answering without a body walk also keeps
  // `mogg.call.stub`'s `isThrows` off a stub another thread is rewriting, and
  // prevents deep recursion through CallStubOp chains in sequence-fused
  // kernels.
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

  // If MOGG, this is an advanced-fusion stub outlined by
  // OutlineKernelsIntoFunctions. Otherwise it was a GenericCompiledRegionOp
  // which has that implicit void** ABI.
  if (getAbi() == ABIKind::MOGG)
    return emitMojoMOGGABI(state);
  return emitMojoMGPABI(state);
}

LogicalResult KernelStubOp::emitMojoMGPABI(EmitMojoState &state) {
  state.mojoReturnValues.clear();
  state.mojoConstantResources.clear();

  bool errOccurred = false;
  static const MOGG::MojoValue vMutAnyOrigin =
      MOGG::MojoValue(MOGG::kComptimeMutAnyOrigin);
  static const MOGG::MojoValue vPointer = MOGG::kStructPointer;
  static const MOGG::MojoValue vOpaquePointer = MOGG::kComptimeOpaquePointer;

  state.builder.def(
      getSymNameAttr().strref(), {},
      {MOGG::NamedMojoDef("arg", vPointer[{vOpaquePointer, vMutAnyOrigin}])},
      std::nullopt, /*exported=*/true, /*raises=*/false,
      /*isAlwaysInline=*/false, [&](const MOGG::FunctionBody &functionBody) {
        state.currentFunctionBody = functionBody;
        state.builder.comment(
            "This is the entry point of the kernel. The argument is a "
            "pointer to a list of inputs and outputs.");

        auto runtimeArgument = functionBody.arguments.front();
        auto &kernelBody = getBody().front();
        // runtimeArgument order: inputs and DPS outputs, then return values
        // (chain and non-DPS outputs).
        //
        // The runtime expects the generated Mojo to take ownership of every
        // AsyncValue input slot, even if unused:
        //
        //     var val = AnyAsyncValueRef(take_slot=args[i])
        //
        // Non-async arguments (e.g. buffer constants) remain raw pointer slots.
        for (auto argument : kernelBody.getArguments()) {
          MOGG::MojoValue slotPtr =
              runtimeArgument.pointerOffsetGet(argument.getArgNumber());
          if (isa<MOGG::AsyncType>(argument.getType())) {
            MOGG::MojoValue local = state.builder.var(
                MOGG::MojoValue(llvm::formatv("{0}(take_slot={1})",
                                              MOGG::kStructAnyAsyncValueRef,
                                              slotPtr.text)
                                    .str()),
                /*cache=*/false);
            state.setMojoValueFor(argument, local);

            // The unpacked result is an owning handle to the backing AsyncValue
            // and must be kept alive for the duration of the compiled function.
            state.markAsKeepAlive(local);
          } else {
            state.setMojoValueFor(argument, slotPtr);
          }
        }
        // Return values are DPS to the runtime but value-semantic inside the
        // kernel, so they get a mapping for mogg.return and mogg.async.pack.
        uint64_t numArguments = getBodyBlock().getNumArguments();
        for (uint64_t i = 0; i < getNumResults(); ++i) {
          state.mojoReturnValues.emplace_back(
              runtimeArgument.pointerOffsetGet(i + numArguments));
        }

        // Constant resources follow in the void** array as ptr + size pairs.
        for (uint64_t i = 0; i < getConstantResources().size(); ++i) {
          uint64_t flatIndex = i * 2 + numArguments + getNumResults();
          MOGG::MojoValue ptr = runtimeArgument.pointerOffsetGet(flatIndex);
          // The ABI is that the bytecount is directly stored as an
          // OpaquePointer.
          MOGG::MojoValue byteCount = OpaquePointerRef(runtimeArgument)
                                          .bitcast(MojoValue(kStructInt))()
                                          .pointerOffsetGet(flatIndex + 1);
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

LogicalResult KernelStubOp::emitMojoMOGGABI(EmitMojoState &state) {
  // The MOGG-ABI stub takes raw runtime handles (a `!mogg.pointer` per input
  // then output, the dynamic `!mogg.dim`s, a `!mogg.context`) and emits
  // straight-line Mojo, with no async wrapping or `void**` marshalling.
  // `mlirTypeToMojoType` renders an opaque as a bare `OpaquePointer`, but here
  // it crosses as the kernel's own struct (`SIMDPair[4, 8]`).
  auto typeToMojo = [&](Type type) -> MOGG::MojoValue {
    if (auto opaqueType = dyn_cast<MO::OpaqueType>(type)) {
      MOGG::MojoValue baseTy = state.mlirOpaqueTypeToMojoType(opaqueType);
      auto params = EmitMojoState::getOpaqueTypeParameters(opaqueType);
      return params.empty() ? baseTy : baseTy[params];
    }
    return state.mlirTypeToMojoType(type);
  };

  SmallVector<MOGG::NamedMojoDef> inputTypes;
  for (BlockArgument arg : getBody().getArguments())
    inputTypes.push_back(MOGG::NamedMojoDef("arg", typeToMojo(arg.getType())));

  // A shape function's shape, or a kernel-allocated opaque, hands back a value.
  MOGG::OutputOp bodyOutput = getBodyTerminator();
  ASSERT_STREAM(bodyOutput.getNumOperands() <= 1,
                << "a MOGG-ABI stub returns at most one value, got "
                << bodyOutput.getNumOperands());
  const bool returnsByValue = bodyOutput.getNumOperands() == 1;
  std::optional<MOGG::MojoValue> returnType;
  if (returnsByValue)
    returnType = typeToMojo(bodyOutput.getOperand(0).getType());

  bool errOccurred = false;
  state.builder.def(
      getSymNameAttr().str(), /*parameters=*/{}, inputTypes, returnType,
      /*exported=*/false,
      /*raises=*/isThrows(), /*isAlwaysInline=*/false,
      [&](const MOGG::FunctionBody &functionBody) {
        state.currentFunctionBody = functionBody;

        for (auto &&[mlirArg, mojoArg] :
             llvm::zip(getBody().getArguments(), functionBody.arguments))
          state.setMojoValueFor(mlirArg, mojoArg);

        errOccurred = llvm::any_of(getOps(), [&](mlir::Operation &op) {
          return state.lowerOpToMojo(&op).failed();
        });
        if (!errOccurred && returnsByValue) {
          MOGG::MojoValue value =
              state.getMojoValueFor(bodyOutput.getOperand(0));
          // An opaque struct need not be `ImplicitlyCopyable` (`SIMDPair` is
          // only `RegisterPassable`), so transfer rather than copy it out.
          if (isa<MO::OpaqueType>(bodyOutput.getOperand(0).getType()))
            value = value.transfer();
          state.builder.returnsValue(value);
        }
      });

  return failure(errOccurred);
}

//===----------------------------------------------------------------------===//
// Advanced-fusion codegen op emitters (MOGG ABI)
//===----------------------------------------------------------------------===//

LogicalResult ShapeCreateOp::emitMojo(EmitMojoState &state) {
  // Encode the shape as a `Coord` -- static dims as compile-time `Idx[N]`,
  // dynamic dims as runtime `Scalar[DType.int](<dim>)` sourced from the
  // `!mogg.dim` operands. `tensor.create` builds a row-major `Layout` from it.
  MojoValue coord =
      state.getShapeAsCoordMojoValue(getShape(), getDynamicDims());
  state.setMojoValueFor(getResult(), state.builder.var(coord));
  return success();
}

// Shared verifier backing `MOGG_TensorCreateInterface::verifyTrait` (declared
// in MOGGInterfaces.h). Lives here, not in the generated interface header,
// because it names `!mogg.buffer`. See the interface for why only the
// buffer-implies-mutable direction is enforced.
llvm::LogicalResult
M::MOGG::verifyTensorCreateInterfaceMutability(Operation *op) {
  auto iface = cast<TensorCreateInterface>(op);
  if (isa<BufferType>(op->getResult(0).getType()) && !iface.getMut())
    return op->emitOpError("!mogg.buffer result requires mut = true");
  return success();
}

LogicalResult TensorCreateOp::emitMojo(EmitMojoState &state) {
  // The rank comes from the op's own static shape attr; the opaque
  // `!mogg.shape` handle carries no attributes of its own.
  MOSH::ShapeAttr shapeAttr = getShapeAttr();

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
      M::Driver::getPreferredAlignment(getDevice().str())));

  // Build a row-major `Layout` from the shape Coord produced by `shape.create`.
  auto layoutVar = state.builder.var(
      MojoValue(kFnRowMajor)({}, {state.getMojoValueFor(getShape())}));

  // Reuse `mogg.tensor.__init__`: both build the same `ManagedTensorSlice`,
  // but here the pointer arrives as a raw `!mogg.pointer`.
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

LogicalResult TensorCreateSliceOp::emitMojo(EmitMojoState &state) {
  // Mojo infers dtype/rank/start_type/stop_type/step_type from the argument
  // types. output_static_shape/static_starts/static_steps carry what is
  // statically known about the view, from this op's attributes.
  auto sliceCreatePrimDecl =
      state.registry.lookupPrimitive(TensorCreateSliceOp::getOperationName());
  ASSERT_STREAM(sliceCreatePrimDecl,
                "Primitive not found for "
                    << TensorCreateSliceOp::getOperationName());
  MojoValue primitive(primitiveMojoName(sliceCreatePrimDecl));
  MojoValue outputStaticShape = state.builder.comptime(
      EmitMojoState::getAsIntTupleMojoValue(getOutShape().getIntDims()));
  MojoValue staticStarts = state.builder.comptime(
      EmitMojoState::getAsIntTupleMojoValue(getStaticStart()));
  MojoValue staticSteps = state.builder.comptime(
      EmitMojoState::getAsIntTupleMojoValue(getStaticStep()));
  MojoValue result = state.builder.var(primitive(
      {MojoValue(kParameterOutputStaticShape).equal(outputStaticShape),
       MojoValue(kParameterStaticStarts).equal(staticStarts),
       MojoValue(kParameterStaticSteps).equal(staticSteps)},
      {state.getMojoValueFor(getInput()), state.getMojoValueFor(getStarts()),
       state.getMojoValueFor(getStops()), state.getMojoValueFor(getSteps())}));
  state.setMojoValueFor(getResult(), result);
  return success();
}

LogicalResult TensorCreateBroadcastOp::emitMojo(EmitMojoState &state) {
  // Mojo infers dtype/rank from $input. A broadcast's shape is always static
  // by emission time (see `mo.static.broadcast_to`), so output_static_shape
  // carries all of it.
  auto broadcastCreatePrimDecl = state.registry.lookupPrimitive(
      TensorCreateBroadcastOp::getOperationName());
  ASSERT_STREAM(broadcastCreatePrimDecl,
                "Primitive not found for "
                    << TensorCreateBroadcastOp::getOperationName());
  MojoValue primitive(primitiveMojoName(broadcastCreatePrimDecl));
  MojoValue outputStaticShape = state.builder.comptime(
      EmitMojoState::getAsIntTupleMojoValue(getOutShape().getIntDims()));
  MojoValue result = state.builder.var(primitive(
      {MojoValue(kParameterOutputStaticShape).equal(outputStaticShape)},
      {state.getMojoValueFor(getInput())}));
  state.setMojoValueFor(getResult(), result);
  return success();
}

LogicalResult TensorCreateReshapeOp::emitMojo(EmitMojoState &state) {
  // Mojo infers dtype/rank from $input; output_static_shape keeps the static
  // dims typed as such. Unlike broadcast/transpose (strides from the input),
  // reshape's runtime stride de-linearizes its output shape, so runtime dims
  // arrive as `$shape`, converted to an `IndexList` via `coord_to_index_list`.
  auto reshapeCreatePrimDecl =
      state.registry.lookupPrimitive(TensorCreateReshapeOp::getOperationName());
  ASSERT_STREAM(reshapeCreatePrimDecl,
                "Primitive not found for "
                    << TensorCreateReshapeOp::getOperationName());
  MojoValue primitive(primitiveMojoName(reshapeCreatePrimDecl));
  MojoValue outputStaticShape = state.builder.comptime(
      EmitMojoState::getAsIntTupleMojoValue(getOutShape().getIntDims()));
  MojoValue outputShapeValue = state.builder.var(
      MojoValue(kFnCoordToIndexList)({}, {state.getMojoValueFor(getShape())}));
  MojoValue result = state.builder.var(primitive(
      {MojoValue(kParameterOutputStaticShape).equal(outputStaticShape)},
      {state.getMojoValueFor(getInput()), outputShapeValue}));
  state.setMojoValueFor(getResult(), result);
  return success();
}

LogicalResult IndexReshapeOp::emitMojo(EmitMojoState &state) {
  // Per-index reshape (store side, inside an epilogue functor): convert the
  // `Coord` index to an `IndexList`, remap it to the output shape preserving
  // the row-major offset via `mogg.index.reshape`, and wrap it back into a
  // `Coord` for `mogg.tensor.store`. Static dims are comptime `IntTuple`s;
  // dynamic inner dims' from/to shapes arrive as epilogue shape captures
  // (`$dynamicShapes`), read only where the static shape is dynamic.
  bool isDynamic = !getDynamicShapes().empty();
  auto primDecl =
      state.registry.lookupPrimitive(IndexReshapeOp::getOperationName());
  ASSERT_STREAM(primDecl, "Primitive not found for "
                              << IndexReshapeOp::getOperationName());
  MojoValue primitive(primitiveMojoName(primDecl));

  // Inside the epilogue functor, emit through the lambda's builder, not
  // `state.builder` (the kernel scope), as `TensorLoadOp`/`TensorStoreOp` do.
  const EmitMojoState::MojoLambdaInfos &lambdaInfo = state.currentLambdaInfo;
  ASSERT_STREAM(
      lambdaInfo.lambdaBuilder,
      "IndexReshapeOp::emitMojo reached without currentLambdaInfo set");
  MojoBuilder &builder = *lambdaInfo.lambdaBuilder;

  // A rank-0 shape is emitted as rank-1 extent-1, here as everywhere else, so
  // the ranks must follow: a `to_rank` of 0 contradicts its own static shape.
  MojoValue fromStaticShape = builder.comptime(
      EmitMojoState::getAsIntTupleMojoValue(getFromShape().getIntDims()));
  MojoValue toStaticShape = builder.comptime(
      EmitMojoState::getAsIntTupleMojoValue(getToShape().getIntDims()));

  MojoValue idxList = builder.var(
      MojoValue(kFnCoordToIndexList)({}, {state.getMojoValueFor(getIndex())}));

  SmallVector<MojoValue> primParams = {
      MojoValue("to_rank").equal(
          MojoValue(uint64_t(getToShape().getRankNonZero()))),
      MojoValue("from_static_shape").equal(fromStaticShape),
      MojoValue("to_static_shape").equal(toStaticShape)};
  SmallVector<MojoValue> primArgs = {idxList};
  if (isDynamic) {
    // The index carries the store method's abstract rank param, which can't
    // unify with the concrete-rank runtime `from_shape`, so rebind it to the
    // concrete from-rank first (equal at every instantiation).
    // `$dynamicShapes` is [from-shape, to-shape] (see the MAPToMOGG lowering).
    primArgs[0] =
        builder.var(idxList.rebind(MojoValue(kStructIndexList)[MojoValue(
            uint64_t(getFromShape().getRankNonZero()))]));
    primArgs.push_back(state.getMojoValueFor(getDynamicShapes()[0]));
    primArgs.push_back(state.getMojoValueFor(getDynamicShapes()[1]));
  }

  MojoValue newIdxList = builder.var(primitive(primParams, primArgs));
  MojoValue newCoord = builder.var(MojoValue(kStructCoord)({}, {newIdxList}));
  state.setMojoValueFor(getResult(), newCoord);
  return success();
}

LogicalResult TensorCreateTransposeOp::emitMojo(EmitMojoState &state) {
  // Mojo infers dtype/rank/perm_type from $input/$perm.
  // output_static_shape/static_permutations carry the compile-time-known parts
  // (`?` where runtime-only); the runtime `$perm` supplies the rest, and a
  // fully static perm never reads it.
  auto transposeCreatePrimDecl = state.registry.lookupPrimitive(
      TensorCreateTransposeOp::getOperationName());
  ASSERT_STREAM(transposeCreatePrimDecl,
                "Primitive not found for "
                    << TensorCreateTransposeOp::getOperationName());
  MojoValue primitive(primitiveMojoName(transposeCreatePrimDecl));
  MojoValue outputStaticShape = state.builder.comptime(
      EmitMojoState::getAsIntTupleMojoValue(getOutShape().getIntDims()));
  MojoValue staticPermutations = state.builder.comptime(
      EmitMojoState::getAsIntTupleMojoValue(getStaticPermutations()));
  MojoValue result = state.builder.var(primitive(
      {MojoValue(kParameterOutputStaticShape).equal(outputStaticShape),
       MojoValue(kParameterStaticPermutations).equal(staticPermutations)},
      {state.getMojoValueFor(getInput()),
       state.getMojoValueFor(getPermutations())}));
  state.setMojoValueFor(getResult(), result);
  return success();
}

bool CallOpaqueOp::isThrows() {
  // Raises iff the callee's execute signature does, as in
  // `CallExecuteOp::isThrows`.
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

  // DPS: outputs, then inputs, then `$device_contexts` (wired by `MAPToMOGG`)
  // to match the execute signature's trailing `ctx`. For a distributed kernel
  // `ConformCallOperands` packed it into one tuple that emits as a
  // `DeviceContextArray`.
  SmallVector<MojoValue> kernelOperands;
  for (Value out : getOutputs())
    kernelOperands.push_back(state.getMojoValueFor(out));
  for (Value in : getInputs())
    kernelOperands.push_back(state.getMojoValueFor(in));
  for (Value ctx : getDeviceContexts())
    kernelOperands.push_back(state.getMojoValueFor(ctx));

  // There is no `mogg.bind` for `getHasEpilogueFusionValue` to detect, so
  // force `has_epilogue_fusion` / `has_prologue_fusion` True to match the
  // execute-call wiring.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    if (getHasEpilogueFusion() &&
        paramName == MOGG::kMoggHasEpilogueFusionParameterName)
      return MojoValue(true);
    if (getHasPrologueFusion() &&
        paramName == MOGG::kMoggHasPrologueFusionParameterName)
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

  // A kernel returning an `!mo.opaque` returns it by value, so bind the call's
  // result to a var the enclosing `mogg.output` can forward, instead of
  // emitting the call as a bare statement.
  if (Value opaqueRes = getOpaqueResult()) {
    MojoValue returned = state.builder.var(
        fullExecuteFunc(maybeParams->functionParameters, kernelOperands));
    state.setMojoValueFor(opaqueRes, returned);
  } else {
    state.builder.call(fullExecuteFunc, maybeParams->functionParameters,
                       kernelOperands);
  }
  state.genAllKeepAliveStatements();
  return success();
}

LogicalResult ScalarCreateOp::emitMojo(EmitMojoState &state) {
  const MojoValue &input = state.getMojoValueFor(getTensor());
  // Reuse the `mogg.as_scalar` primitive: both pack a single-element
  // tensor operand into the scalar the kernel expects.
  static constexpr llvm::StringLiteral kAsScalarPrimitive = "mogg.as_scalar";
  auto asScalarPrimDecl = state.registry.lookupPrimitive(kAsScalarPrimitive);
  ASSERT_STREAM(asScalarPrimDecl,
                "Primitive not found for " << kAsScalarPrimitive);
  MojoValue moggAsScalar(primitiveMojoName(asScalarPrimDecl));
  MojoValue scalarValue = moggAsScalar(ArrayRef<MojoValue>{}, {input});
  state.setMojoValueFor(getScalar(), state.builder.var(scalarValue));
  return success();
}

LogicalResult TupleCreateOp::emitMojo(EmitMojoState &state) {
  // `$variadic_abi` (set by `ConformCallOperands`) selects `VariadicTensors`,
  // `_FusedInputVariadicTensors`, or `_FusedOutputVariadicTensors`, so this
  // never inspects the producing ops.
  SmallVector<MojoValue> inputValues = state.getMojoValuesFor(getElements());

  // Contexts pack into a `DeviceContextArray[N]` instead, via its variadic
  // constructor -- the per-device contexts of a distributed kernel. Mirrors the
  // `AsTupleOp::emitMojo` context branch.
  if (isa<MOGG::ContextType>(getElements().front().getType())) {
    MojoValue deviceContextArrayConstructor = MojoValue(
        kStructDeviceContextArray)[{
        MojoValue(kParameterLength).equal(std::to_string(inputValues.size()))}];
    state.setMojoValueFor(
        getTuple(),
        state.builder.var(deviceContextArrayConstructor({}, inputValues)));
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

  // Plain: a `VariadicTensors` over the raw pointers/shapes.
  if (getVariadicAbi() == VariadicABI::Plain) {
    MojoValue ctor = MojoValue(kStructVariadicTensors)[{
        MojoValue(kParameterIOSpec)
            .equal(ManagedTensorSliceRef(inputValues[0]).ioSpec()),
        MojoValue(kParameterStaticSpecs).equal(inputSpecsComptime)}];
    state.setMojoValueFor(
        getTuple(), state.builder.var(
                        ctor({}, {runtimePointersTuple, runtimeShapesTuple})));
    return success();
  }

  // Fused: gather each element's fusion functor and its type into a
  // `_FusionPack` + `*FusionTypes`, then build the matching fused aggregate.
  // The functor lives on the element slice itself (`in_fusion` / `out_fusion`),
  // so nothing here reaches back into the producing op.
  bool isInput = getVariadicAbi() == VariadicABI::FusedInput;
  SmallVector<MojoValue> fusionValues;
  SmallVector<MojoValue> typeParams;
  for (const MojoValue &el : inputValues) {
    ManagedTensorSliceRef slice(el);
    fusionValues.push_back(isInput ? slice.inFusion() : slice.outFusion());
    typeParams.push_back(isInput ? slice.inFusionType()
                                 : slice.outFusionType());
  }
  MojoValue fusionsTuple =
      state.builder.var(MojoValue(kStructMsFusionPack)({}, fusionValues));
  typeParams.push_back(
      MojoValue(kParameterStaticSpecs).equal(inputSpecsComptime));

  StringRef structName = isInput ? kStructMsFusedInputVariadicTensors
                                 : kStructMsFusedOutputVariadicTensors;
  MojoValue fusedValue = MojoValue(structName)[typeParams](
      {}, {runtimePointersTuple, runtimeShapesTuple, fusionsTuple});
  state.setMojoValueFor(getTuple(), state.builder.var(fusedValue));
  return success();
}

LogicalResult TensorLoadOp::emitMojo(EmitMojoState &state) {
  // Several callers set up `currentLambdaInfo`, so builder/width/alignment are
  // read by role. Every fusion lane records its receiver's decomposed
  // `ManagedTensorSlice` (ptr/shape/strides + comptime `StaticTensorSpec`) via
  // `setFusedTensorArgFor`, so `simd_load_from_tensor_pointer` honors the
  // spec's static layout (stride-correct for fused views) and alignment.
  const FusedTensorArg &arg = state.getFusedTensorArgFor(getTensor());
  const MojoValue &idx = state.getMojoValueFor(getIndex());
  const EmitMojoState::MojoLambdaInfos &lambdaInfo = state.currentLambdaInfo;
  ASSERT_STREAM(lambdaInfo.lambdaBuilder,
                "TensorLoadOp::emitMojo reached without currentLambdaInfo set");

  // The loop index arrives as a `Coord`; the pointer-based load wants an
  // `IndexList[rank]`, sized by the capture's own static spec.
  MojoValue indices = MojoValue(kFnCoordToIndexList)({}, {idx}).rebind(
      MojoValue(kStructIndexList)[arg.spec.get(kParameterRank)]);
  MojoValue loaded =
      lambdaInfo.lambdaBuilder->var(MojoValue(kFnMsSimdLoadFromTensorPointer)(
          {MojoValue(kParameterStaticSpec).equal(arg.spec),
           MojoValue(kParameterSimdWidth).equal(lambdaInfo.widthParameter),
           MojoValue(kParameterElementAlignment)
               .equal(lambdaInfo.alignmentParameter)},
          {arg.ptr, arg.shape, arg.strides, indices}));
  state.setMojoValueFor(getResult(), loaded);
  return success();
}

LogicalResult TensorStoreOp::emitMojo(EmitMojoState &state) {
  // Reached recursively from whichever caller sets up `currentLambdaInfo` --
  // `TensorCreateEpilogueOp`'s `__call__` and `CallForeachOp`'s closure
  // each have a different comptime parameter list, hence reading
  // width/alignment/builder by role rather than assuming either shape.
  const FusedTensorArg &arg = state.getFusedTensorArgFor(getTensor());
  const MojoValue &idx = state.getMojoValueFor(getIndex());
  const MojoValue &value = state.getMojoValueFor(getValue());
  const EmitMojoState::MojoLambdaInfos &lambdaInfo = state.currentLambdaInfo;
  ASSERT_STREAM(
      lambdaInfo.lambdaBuilder,
      "TensorStoreOp::emitMojo reached without currentLambdaInfo set");

  // The method's generic `dtype` comptime parameter isn't statically known to
  // equal the store target's own spec dtype, so the stored value needs an
  // explicit rebind to the capture spec's own dtype.
  MojoValue rebound = value.rebind(MojoValue(
      kStructSIMD)[{arg.spec.get(kParameterDType), lambdaInfo.widthParameter}]);
  // The loop index arrives as a `Coord`; the pointer-based store wants an
  // `IndexList[rank]`, sized by the capture's own static spec.
  MojoValue indices = MojoValue(kFnCoordToIndexList)({}, {idx}).rebind(
      MojoValue(kStructIndexList)[arg.spec.get(kParameterRank)]);
  lambdaInfo.lambdaBuilder->call(
      MojoValue(kFnMsSimdStoreIntoTensorPointer),
      {MojoValue(kParameterStaticSpec).equal(arg.spec),
       MojoValue(kParameterSimdWidth).equal(lambdaInfo.widthParameter),
       MojoValue(kParameterElementAlignment)
           .equal(lambdaInfo.alignmentParameter)},
      {arg.ptr, arg.shape, arg.strides, indices, rebound});
  return success();
}

/// One fused-region functor capture, as the enclosing function sees it.
/// `!mogg.tensor` is opaque, so shape / dtype / mutability are read from the
/// op that creates the capture, as in `CallForeachOp::emitMojo`.
struct FusedCaptureInfo {
  /// The capture's `ManagedTensorSlice`, decomposed into device-passable parts:
  /// pointer and runtime shape/strides as enclosing-function locals, and
  /// `spec`, the comptime `StaticTensorSpec`. Load/store go through
  /// `simd_{load_from,store_into}` on these.
  MojoValue ptr;
  MojoValue shape;
  MojoValue strides;
  MojoValue spec;
  /// Literals, baked directly into the functor's field/param types.
  MojoValue dtype;
  MojoValue mut;
};

/// Collects per-capture metadata for a fused-region functor. Baking dtype/mut
/// as literals rather than struct params keeps capture loads statically typed
/// in the method body, so no rebind is needed per load site.
static FailureOr<SmallVector<FusedCaptureInfo>>
collectFusedCaptureInfos(EmitMojoState &state, Operation *op,
                         ValueRange captures) {
  SmallVector<FusedCaptureInfo> infos;
  for (Value cap : captures) {
    // Element type / mutability come from the producing `mogg.tensor.create*`
    // op; everything else from the capture's emitted Mojo value.
    auto create = cap.getDefiningOp<TensorCreateInterface>();
    if (!create)
      return op->emitOpError(
          "capture must be a mogg.tensor.create(.slice|.broadcast|"
          ".transpose|.reshape) result");
    ErrorOr<DType> capDtypeOr = M::MO::fromMLIRType(create.getElementType());
    if (capDtypeOr.isError())
      return op->emitOpError("failed to convert capture dtype to Mojo");

    // Views have already folded their access pattern into the decomposed
    // pointer/shape/strides. `spec` is a comptime alias in the enclosing scope
    // so it is nameable where the functor is instantiated and inside a
    // trivially-capturing foreach closure (where `mts.static_spec` would
    // reference the uncaptured slice).
    ManagedTensorSliceRef capRef(state.getMojoValueFor(cap));
    // `unsafe_ptr()` erases the origin to `MutAnyOrigin`, matching the
    // functor's pointer field and `simd_*_tensor_pointer`'s `ptr` parameter.
    MojoValue ptr = state.builder.var(capRef.unsafePtr());
    MojoValue shape = state.builder.var(capRef.shape());
    MojoValue strides = state.builder.var(capRef.strides());
    MojoValue spec = state.builder.comptime(capRef.staticSpec());
    infos.push_back({ptr, shape, strides, spec,
                     state.mlirDTypeToMojoType(*capDtypeOr),
                     MojoValue(create.getMut())});
  }
  return infos;
}

// Builds a `ManagedTensorSlice` over `ptr` with `layout`/`dtype`/`rank`, via
// the `mogg.tensor.__init__` primitive. `ptr` may be a real base pointer (so an
// accumulating kernel can accumulate through it) or a dangling one (for a
// kernel that only reads shape and never touches the pointer).
static MojoValue emitBaseSliceFromPtr(EmitMojoState &state, MojoValue dtype,
                                      unsigned rank, MojoValue layout,
                                      MojoValue alignment, MojoValue ptr) {
  auto tensorInitPrimDecl =
      state.registry.lookupPrimitive(TensorInitOp::getOperationName());
  ASSERT_STREAM(tensorInitPrimDecl,
                "Primitive not found for " << TensorInitOp::getOperationName());
  return state.builder.var(MojoValue(primitiveMojoName(tensorInitPrimDecl))(
      {dtype, MojoValue(static_cast<uint64_t>(rank)), MojoValue(false),
       MojoValue(kStructIO).get(MO::stringifyIOKind(MO::IOKind::Input)),
       alignment},
      {OpaquePointerRef(ptr).bitcast(MojoValue(kComptimeDTypeNoneType))(),
       layout}));
}

/// A dangling `ManagedTensorSlice` that only carries a type; every access goes
/// through the fusion functor bound onto it. A dtype-changing epilogue needs
/// it: the kernel gets a slice of its compute dtype (f32), while the real
/// output (si32) reaches the functor as capture 0, since
/// `_bind_to_fused_output` preserves its receiver's dtype.
static MojoValue emitDanglingBaseSlice(EmitMojoState &state, MojoValue dtype,
                                       unsigned rank, MojoValue layout,
                                       MojoValue alignment) {
  MojoValue danglingPtr =
      MojoValue(
          kStructPointer)[{MojoValue(kStructScalar)[{
                               MojoValue(kStructDType).get(kEnumDTypeInt8)}],
                           MojoValue(kComptimeMutAnyOrigin)}]
          .get(OpaquePointerRef::kUnsafeDangling)();
  return emitBaseSliceFromPtr(state, dtype, rank, layout, alignment,
                              danglingPtr);
}

LogicalResult TensorGetPtrOp::emitMojo(EmitMojoState &state) {
  // Read the raw base pointer of the input slice. This op lives in the kernel
  // body (outside any fusion lambda), so `state.builder` is the right scope.
  ManagedTensorSliceRef mts(state.getMojoValueFor(getTensor()));
  state.setMojoValueFor(getResult(), state.builder.var(mts.rawPtr()));
  return success();
}

/// The rank a fused-region op's `$shape` operand describes, recovered from the
/// feeding `shape.create` (the opaque `!mogg.shape` handle carries none).
static FailureOr<unsigned> fusedRegionRank(Operation *op, Value shape) {
  auto shapeCreate = shape.getDefiningOp<ShapeCreateOp>();
  if (!shapeCreate)
    return op->emitOpError("shape operand must come from a mogg.shape.create");
  unsigned rank = shapeCreate.getShape().getRank();
  return rank == 0 ? 1 : rank;
}

namespace {

/// One MOGG-ABI fusion-lambda flavor. The load side (`map.iter.prologue`) and
/// the store side (`map.iter.epilogue`) differ only in the trait, the method,
/// and whether it returns a value. Mirrors `emitFusionStruct`'s trait/method
/// selection.
struct FusionLambdaInfo {
  /// Prefix for `getIdentifier`, which mints a unique struct name from it
  /// (`EpilogueStoreFunctor_6`).
  llvm::StringRef structName;
  MojoValue traitName;
  MojoValue methodName;
  /// `OutputFusion` spells the width parameter `SIMDLength`; `InputFusion`
  /// spells it `Int`. The two correlate with `takesValue` across today's
  /// traits, but that is a coincidence of those three, not a rule.
  MojoValue widthParamType;
  /// The store lane is handed the kernel's value; the load lane manufactures
  /// one.
  bool takesValue = false;
  /// The MTS-bind traits (`managed_tensor_slice.InputFusion`/`OutputFusion`)
  /// spell the index `IndexList[rank]`; the trait-wrapper traits
  /// (`InputLoadFusion`/`OutputStoreFusion`/`TransformFusion`) spell it `Coord`
  /// and name the method `__call__`, which then takes `idx: Coord` directly.
  bool coordIndex = false;
};

} // namespace

/// Declares the fusion functor for `region` and returns an instance of it.
/// `captureInfos` is filled in with the functor-side field names as the struct
/// is minted. `valueDtype` is the dtype the incoming `value` is rebound to,
/// and is unused by a lane that takes no value.
static FailureOr<MojoValue>
emitMapFusionFunctor(EmitMojoState &state, Region &region,
                     SmallVectorImpl<FusedCaptureInfo> &captureInfos,
                     const FusionLambdaInfo &lane, MojoValue valueDtype,
                     ArrayRef<unsigned> shapeCaptureRanks = {},
                     ArrayRef<MojoValue> shapeCaptureArgs = {}) {
  unsigned numCaptures = captureInfos.size();
  unsigned numShapeCaptures = shapeCaptureRanks.size();
  MojoValue structNameVal = state.structDefsBuilder.newVar(
      state.structDefsBuilder.getIdentifier(lane.structName.str()));
  // One comptime `StaticTensorSpec` param per capture with a concrete dtype
  // (so `simd_*_tensor_pointer` infers it) but inferred rank/layout (`_`: a
  // view's symbolic rank wouldn't unify with a literal). Read back as
  // `sb.parameters[j]`.
  SmallVector<NamedMojoDef> structParams;
  for (unsigned j = 0; j < numCaptures; ++j)
    structParams.push_back(NamedMojoDef(
        "spec",
        MojoValue(llvm::formatv("{0}[{1}, _, ...]", kStructStaticTensorSpec,
                                captureInfos[j].dtype.text)
                      .str())));

  // Shape captures (dynamic reshape runtime shapes) become trailing
  // `IndexList[rank]` fields; their literal rank needs no struct param.
  SmallVector<MojoValue> shapeCaptureFields(numShapeCaptures);

  // Functor-side names per capture: the (ptr, shape, strides) fields and the
  // `Self.`-qualified spec param, minted in the struct callback below and read
  // back in the method body.
  SmallVector<MojoValue> selfPtrFields(numCaptures);
  SmallVector<MojoValue> selfShapeFields(numCaptures);
  SmallVector<MojoValue> selfStridesFields(numCaptures);
  SmallVector<MojoValue> selfSpecParams(numCaptures);

  bool errOccurred = false;
  state.structDefsBuilder.structDecl(
      structNameVal, structParams,
      {lane.traitName, MojoValue("TrivialRegisterPassable")},
      [&](const MojoBuilder::StructBody &sb) {
        for (unsigned j = 0; j < numCaptures; ++j) {
          FusedCaptureInfo &capture = captureInfos[j];
          selfSpecParams[j] = MojoValue("Self").get(sb.parameters[j]);
          // Shape/strides fields are sized by the spec's own rank, so they
          // unify with the runtime `shape()`/`strides()` instance args (both
          // from the same slice) -- symbolic rank and all.
          MojoValue indexListType = MojoValue(
              kStructIndexList)[selfSpecParams[j].get(kParameterRank)];
          MojoValue scalarType = MojoValue(kStructScalar)[{capture.dtype}];
          // A `MutAnyOrigin` pointer field (mutable-any, matching
          // `simd_*_tensor_pointer`'s `ptr` parameter) needs the legacy
          // any-origin opt-in the Mojo parser otherwise rejects.
          state.structDefsBuilder.verbatim(
              MojoValue("@__allow_legacy_any_origin_fields"));
          selfPtrFields[j] = state.structDefsBuilder.varDecl(MojoValue(
              kStructPointer)[{scalarType, MojoValue(kComptimeMutAnyOrigin)}]);
          selfShapeFields[j] = state.structDefsBuilder.varDecl(indexListType);
          selfStridesFields[j] = state.structDefsBuilder.varDecl(indexListType);
        }
        for (unsigned k = 0; k < numShapeCaptures; ++k)
          shapeCaptureFields[k] = state.structDefsBuilder.varDecl(
              MojoValue(kStructIndexList)[MojoValue(
                  static_cast<uint64_t>(shapeCaptureRanks[k]))]);

        // The method must match the conformed trait (managed_tensor_slice.mojo)
        // for `_bind_to_fused_*` to accept it. Parameters match positionally;
        // names follow the trait for readability.
        //
        // A `mogg.yield`-terminated body returns that value; a store-lane body
        // returns nothing. Read it off the region so it can't disagree with the
        // lane.
        bool returnsValue = isa<YieldOp>(region.front().back());
        MojoValue simdType = MojoValue(kStructSIMD)[{
            MojoValue(kParameterDType), MojoValue(kParameterSimdWidth)}];
        SmallVector<NamedMojoDef> methodParameters = {
            NamedMojoDef::asVerbatim(kParameterDType, kStructDType),
            NamedMojoDef::asVerbatim(kParameterRank, kStructInt),
            NamedMojoDef::asVerbatim(kParameterSimdWidth, lane.widthParamType),
            NamedMojoDef::asVerbatim(kParameterElementAlignment, kStructInt,
                                     MojoValue(1)),
        };
        SmallVector<NamedMojoDef> methodArgs = {
            NamedMojoDef::asVerbatim(kParameterSelf),
            NamedMojoDef::asVerbatim(
                kParameterIdx,
                lane.coordIndex ? MojoValue(kStructCoord)
                                : MojoValue(kStructIndexList)[kParameterRank]),
        };
        if (lane.takesValue)
          methodArgs.push_back(NamedMojoDef::asVerbatim("value", simdType));

        state.structDefsBuilder.def(
            lane.methodName, methodParameters, methodArgs,
            returnsValue ? std::optional<MojoValue>(simdType)
                         : std::optional<MojoValue>{},
            /*exported=*/false, /*raises=*/false,
            /*isAlwaysInline=*/true, [&](const FunctionBody &fb) {
              EmitMojoState::LambdaInfosGuard lambdaInfosGuard(
                  state, {/*lambdaBuilder=*/&state.structDefsBuilder,
                          /*widthParameter=*/fb.parameters[2],
                          /*alignmentParameter=*/fb.parameters[3]});

              // The stride-aware primitive takes a `Coord`, so convert `idx`
              // once via `Coord`'s `IndexList` constructor (coord.mojo). Not
              // `_index_list_to_static_coord`, which ignores the input at
              // statically shaped positions: right for a shape, wrong for a
              // varying loop index. The trait-wrapper lane already has a
              // `Coord`.
              MojoValue coordIdx =
                  lane.coordIndex ? fb.arguments[1]
                                  : state.structDefsBuilder.var(MojoValue(
                                        kStructCoord)({}, {fb.arguments[1]}));

              // The store lanes' body leads with [value, index]; the load
              // lane's leads with [index] alone. Captures follow either way.
              unsigned captureBase = 1;
              if (lane.takesValue) {
                // `value` is `SIMD[dtype, width]` with the method's generic
                // `dtype`; nested elementwise calls hardcode `valueDtype`'s
                // literal, so rebind `value` to it.
                MojoValue reboundVal = state.structDefsBuilder.var(
                    fb.arguments[2].rebind(MojoValue(
                        kStructSIMD)[{valueDtype, fb.parameters[2]}]));
                state.setMojoValueFor(region.getArgument(0), reboundVal);
                state.setMojoValueFor(region.getArgument(1), coordIdx);
                captureBase = 2;
              } else {
                state.setMojoValueFor(region.getArgument(0), coordIdx);
              }

              for (unsigned j = 0; j < numCaptures; ++j) {
                // Each capture gets a fresh block arg. Record its decomposed
                // MTS components off the struct's fields/param for the body's
                // loads/stores; the spec carries the static alignment.
                Value captureArg = region.getArgument(captureBase + j);
                state.setFusedTensorArgFor(
                    captureArg,
                    {MojoValue(kParameterSelf).get(selfPtrFields[j]),
                     MojoValue(kParameterSelf).get(selfShapeFields[j]),
                     MojoValue(kParameterSelf).get(selfStridesFields[j]),
                     selfSpecParams[j]});
              }

              // Shape captures follow the tensor captures as trailing block
              // args; each resolves straight to its own `IndexList` field.
              for (unsigned k = 0; k < numShapeCaptures; ++k)
                state.setMojoValueFor(
                    region.getArgument(captureBase + numCaptures + k),
                    MojoValue(kParameterSelf).get(shapeCaptureFields[k]));

              Block &body = region.front();
              // A `mogg.yield` becomes the method's return, handled below.
              errOccurred = llvm::any_of(body, [&](Operation &op) {
                if (isa<YieldOp>(op))
                  return false;
                return state.lowerOpToMojo(&op).failed();
              });
              if (errOccurred || !returnsValue)
                return;

              // The method's generic `dtype` isn't statically known to equal
              // the yielded value's own dtype (which comes from the capture
              // tiles' literals), so rebind on the way out; the two are equal
              // at every instantiation.
              auto yieldOp = cast<YieldOp>(body.getTerminator());
              MojoValue result =
                  state.getMojoValueFor(yieldOp.getOperands().front());
              state.structDefsBuilder.returnsValue(result.rebind(simdType));
            });
      });

  if (errOccurred)
    return failure();

  // Instantiate the functor: per capture, the comptime `StaticTensorSpec` and
  // the (ptr, shape, strides) instance fields, in the declared order. The spec
  // param's inferred rank/layout are resolved from each passed spec.
  SmallVector<MojoValue> comptimeArgs;
  SmallVector<MojoValue> instanceArgs;
  for (const FusedCaptureInfo &capture : captureInfos) {
    comptimeArgs.push_back(capture.spec);
    instanceArgs.push_back(capture.ptr);
    instanceArgs.push_back(capture.shape);
    instanceArgs.push_back(capture.strides);
  }
  // Shape captures' `IndexList`s follow the (ptr, shape, strides) triples, in
  // the trailing field order the struct was declared with.
  for (MojoValue shapeArg : shapeCaptureArgs)
    instanceArgs.push_back(shapeArg);
  return state.builder.var(structNameVal(comptimeArgs, instanceArgs));
}

LogicalResult TensorCreateEpilogueOp::verify() {
  Block &body = getBody().front();
  if (body.empty())
    return emitOpError("requires a non-empty body");
  // Both shapes need the store-target capture: the store lane writes through
  // it, the compute lane leaves it for emit to bind onto.
  if (getCaptures().empty())
    return emitOpError("requires at least one (store-target) capture");
  if (body.getNumArguments() <
      1 + getCaptures().size() + getShapeCaptures().size())
    return emitOpError("body takes a value, an index, one arg per capture, and "
                       "one arg per shape capture");

  // `mogg.yield` is a terminator, so it can only be the last op: its presence
  // is what makes this a compute lambda rather than a store one.
  auto yieldOp = dyn_cast<YieldOp>(&body.back());
  if (!yieldOp)
    return success();

  if (!getComputeOutputArg())
    return emitOpError("a yield-terminated epilogue body is a compute lambda, "
                       "which requires `computeOutputArg`");
  if (yieldOp.getNumOperands() != 1 ||
      yieldOp.getOperand(0).getType() != body.getArgument(0).getType())
    return emitOpError("a compute-lambda epilogue must yield exactly the "
                       "value type it was handed");
  return success();
}

LogicalResult TensorCreateEpilogueOp::emitMojo(EmitMojoState &state) {
  // Emit the output-store fusion functor + `_MoggOutputStoreFusionTensor`
  // wrapper for a MOGG-ABI fused output; `__call__` recursively lowers this
  // op's epilogue body (as `emitMojoMOGGABI` walks a stub body).
  if (getCaptures().empty())
    return emitOpError(
        "requires at least one (store-target) capture to fuse into");

  // Emit the shared trait + wrapper definitions (`OutputStoreFusion` /
  // `_MoggOutputStoreFusionTensor`) once, exactly as the MOGG-ABI store path.
  state.emitOutputStoreFusionDefsOnce();

  // The epilogue's output dtype, to rebind the inbound `value` (typed by the
  // trait method's abstract `dtype`). Capture dtypes come from their defining
  // ops (see `collectFusedCaptureInfos`).
  ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(getElementType());
  if (dtypeOr.isError())
    return emitOpError("failed to convert epilogue element type to Mojo");
  MojoValue dtypeParam = state.mlirDTypeToMojoType(*dtypeOr);

  // The dtype the kernel hands in, which after a dtype-changing epilogue
  // fusion (`mo.cast`) is not the output dtype; rebinding to `dtypeParam` would
  // emit `rebind[SIMD[si32, w]](SIMD[f32, w])`, which Mojo rejects.
  ErrorOr<DType> inDtypeOr =
      M::MO::fromMLIRType(getBody().getArgument(0).getType());
  if (inDtypeOr.isError())
    return emitOpError("failed to convert epilogue input type to Mojo");
  MojoValue inputDtypeParam = state.mlirDTypeToMojoType(*inDtypeOr);

  // One `LayoutType` param and (ptr, layout) field pair per capture: the store
  // target, plus one per extra input of a fused binary elementwise op.
  FailureOr<SmallVector<FusedCaptureInfo>> captureInfosOr =
      collectFusedCaptureInfos(state, *this, getCaptures());
  if (failed(captureInfosOr))
    return failure();
  SmallVector<FusedCaptureInfo> captureInfos = *captureInfosOr;

  // `MAPToMOGG` builds a yield-terminated body exactly for a compute lambda;
  // `compute` is `store` plus a return.
  bool isCompute = isa<YieldOp>(getBody().front().back());

  const FusionLambdaInfo storeLane = {
      /*structName=*/"EpilogueStoreFunctor",
      /*traitName=*/kTraitOutputFusion,
      /*methodName=*/MojoValue("store"),
      /*widthParamType=*/kStructSIMDLength,
      /*takesValue=*/true,
  };
  const FusionLambdaInfo computeLane = {
      /*structName=*/"EpilogueComputeFunctor",
      /*traitName=*/kTraitComputeOutputFusion,
      /*methodName=*/MojoValue("compute"),
      /*widthParamType=*/kStructSIMDLength,
      /*takesValue=*/true,
  };

  // A trait-ABI fused output (`Some[Output & Fused & StoreAccess]`, `$contract`
  // stamped at MAP lowering) wants the inline `OutputStoreFusion` functor shape
  // (Coord `__call__`, `width: Int`) in `_MoggOutputStoreFusionTensor`; the MTS
  // API binds the `managed_tensor_slice.OutputFusion` shape onto a base slice.
  bool wrapStore =
      getContract() && bitEnumContainsAny(getContract()->getAccess(),
                                          MOGG::TensorArgAccessKind::store);
  const FusionLambdaInfo traitStoreLane = {
      /*structName=*/"OutputStoreFunctor",
      /*traitName=*/MojoValue("OutputStoreFusion"),
      /*methodName=*/MojoValue("__call__"),
      /*widthParamType=*/kStructInt,
      /*takesValue=*/true,
      /*coordIndex=*/true,
  };

  // A transform-access trait output (`Some[Output & Fused & TransformAccess]`)
  // wants the functor in the inline `TransformFusion` shape (Coord `__call__`
  // RETURNING the transformed value) wrapped in `_MoggTransformFusionTensor`,
  // which also holds the real output buffer (the wrapper stores the transformed
  // value through it). Its body is a compute body, so `MAPToMOGG` yields.
  bool wrapTransform =
      getContract() && bitEnumContainsAny(getContract()->getAccess(),
                                          MOGG::TensorArgAccessKind::transform);
  const FusionLambdaInfo traitTransformLane = {
      /*structName=*/"TransformFunctor",
      /*traitName=*/MojoValue("TransformFusion"),
      /*methodName=*/MojoValue("__call__"),
      /*widthParamType=*/kStructInt,
      /*takesValue=*/true,
      /*coordIndex=*/true,
  };
  if (wrapTransform && !state.emittedTransformFusionArgDefs) {
    state.structDefsBuilder.verbatim(MojoValue(kTransformFusionDefs.str()));
    state.emittedTransformFusionArgDefs = true;
  }

  // Inner-dynamic store-index reshapes carry their runtime from/to shapes as
  // trailing shape captures. Convert each `!mogg.shape` Coord to the
  // `IndexList` the reshape primitive wants, and pass its rank so the functor
  // can declare the matching field.
  SmallVector<unsigned> shapeCaptureRanks;
  SmallVector<MojoValue> shapeCaptureArgs;
  for (Value shapeCap : getShapeCaptures()) {
    FailureOr<unsigned> rankOr = fusedRegionRank(*this, shapeCap);
    if (failed(rankOr))
      return failure();
    shapeCaptureRanks.push_back(*rankOr);
    shapeCaptureArgs.push_back(state.builder.var(
        MojoValue(kFnCoordToIndexList)({}, {state.getMojoValueFor(shapeCap)})));
  }

  FailureOr<MojoValue> functorOr = emitMapFusionFunctor(
      state, getBody(), captureInfos,
      wrapTransform ? traitTransformLane
      : wrapStore   ? traitStoreLane
                    : (isCompute ? computeLane : storeLane),
      inputDtypeParam, shapeCaptureRanks, shapeCaptureArgs);
  if (failed(functorOr))
    return failure();
  MojoValue functorInstance = *functorOr;

  if (wrapTransform) {
    // Wrap the transform functor in `_MoggTransformFusionTensor`, which also
    // holds the real output buffer (`$outputPtr`): the wrapper stores the
    // transformed value through it. Sourced from this op's own device/shape.
    StringRef deviceLabel = getDevice();
    bool devicePassable = deviceLabel != kCPULabel;
    MojoValue alignmentParam(static_cast<uint64_t>(
        M::Driver::getPreferredAlignment(deviceLabel.str())));
    FailureOr<unsigned> baseRankT = fusedRegionRank(*this, getShape());
    if (failed(baseRankT))
      return failure();
    MojoValue layoutVar = state.builder.var(
        MojoValue(kFnRowMajor)({}, {state.getMojoValueFor(getShape())}));
    MojoValue baseSlice = emitBaseSliceFromPtr(
        state, dtypeParam, *baseRankT, layoutVar, alignmentParam,
        state.getMojoValueFor(getOutputPtr()));
    // The wrapper's `handle` is `Pointer[Scalar[dtype], MutUntrackedOrigin]`;
    // a slice's `unsafe_ptr()` yields `MutAnyOrigin`, so retag the origin.
    MojoValue basePtr = state.builder.var(MojoValue(
        llvm::formatv("{0}.unsafe_origin_cast[MutUntrackedOrigin]()",
                      ManagedTensorSliceRef(baseSlice).unsafePtr().text)
            .str()));
    MojoValue wrapper = MojoValue("_MoggTransformFusionTensor")(
        {dtypeParam, MojoValue("type_of")({}, {layoutVar}), alignmentParam,
         MojoValue(true), MojoValue("type_of")({}, {functorInstance}),
         MojoValue(devicePassable), MojoValue(!devicePassable)},
        {functorInstance, layoutVar, basePtr});
    state.setMojoValueFor(getTensor(), state.builder.var(wrapper));
    return success();
  }

  if (wrapStore) {
    // The functor owns its store to the captured target, so no base slice or
    // `_bind_to_fused_output` is needed.
    StringRef deviceLabel = getDevice();
    bool devicePassable = deviceLabel != kCPULabel;
    MojoValue alignmentParam(static_cast<uint64_t>(
        M::Driver::getPreferredAlignment(deviceLabel.str())));
    MojoValue layoutVar = state.builder.var(
        MojoValue(kFnRowMajor)({}, {state.getMojoValueFor(getShape())}));
    // The wrapper's dtype is what the KERNEL stores (the value it hands to
    // `store`), i.e. the computed dtype -- a dtype-changing epilogue casts to
    // the real output dtype inside the functor's own store to its capture.
    MojoValue wrapper = MojoValue("_MoggOutputStoreFusionTensor")(
        {inputDtypeParam, MojoValue("type_of")({}, {layoutVar}), alignmentParam,
         MojoValue("type_of")({}, {functorInstance}), MojoValue(devicePassable),
         MojoValue(!devicePassable)},
        {functorInstance, layoutVar});
    state.setMojoValueFor(getTensor(), state.builder.var(wrapper));
    return success();
  }

  // Bind the functor onto the store target's ManagedTensorSlice via
  // `_bind_to_fused_output`, giving a ManagedTensorSlice with a resolvable
  // `static_spec` that converts to the kernel's `FusedOutputTensor`. A
  // dtype-changing epilogue binds onto a slice of the computed dtype and
  // reaches the real output through capture 0.
  ASSERT_STREAM(!isCompute || inputDtypeParam.text == dtypeParam.text,
                "a compute lambda cannot change dtype; `MAPToMOGG` classifies "
                "a dtype-changing epilogue as a store lambda");
  // The base slice is over `$outputPtr` (the store target, or a pre-cast
  // accumulation buffer for a mixed-precision accumulating kernel): real
  // storage of the compute dtype, so an accumulating kernel accumulates
  // through it; otherwise the lambda does the store.
  FailureOr<unsigned> baseRank = fusedRegionRank(*this, getShape());
  if (failed(baseRank))
    return failure();
  MojoValue baseLayout = state.builder.var(
      MojoValue(kFnRowMajor)({}, {state.getMojoValueFor(getShape())}));
  MojoValue baseSlice = emitBaseSliceFromPtr(
      state, inputDtypeParam, *baseRank, baseLayout,
      MojoValue(static_cast<uint64_t>(
          M::Driver::getPreferredAlignment(getDevice().str()))),
      state.getMojoValueFor(getOutputPtr()));
  // Which entry point to bind through follows the kernel's declared argument
  // type, not the lane: a `_FusedComputeOutputTensor` argument binds with
  // `_bind_to_fused_compute_output` even when a dtype change or the
  // `enable-compute-lambdas` opt-out kept the store lane above.
  ManagedTensorSliceRef bindTarget(baseSlice);
  MojoValue bindMethod = getComputeOutputArg()
                             ? bindTarget.bindToFusedComputeOutput()
                             : bindTarget.bindToFusedOutput();
  MojoValue fusedTensor = bindMethod({}, {functorInstance});
  state.setMojoValueFor(getTensor(), state.builder.var(fusedTensor));
  return success();
}

LogicalResult TensorCreatePrologueOp::emitMojo(EmitMojoState &state) {
  // Emit the input-load fusion functor, the load-side sibling of
  // `TensorCreateEpilogueOp::emitMojo`: the tensor never materializes, and the
  // functor returns the loaded value, rebound to the method's generic dtype.
  // Captures are optional; a fused constant producer has none.
  //
  // `$contract` (stamped at MAP lowering) selects the consumer ABI:
  //   - trait (`Some[Input & Fused & LoadAccess]`): the inline
  //     `InputLoadFusion` shape (Coord `__call__`), wrapped in
  //     `_MoggInputLoadFusionTensor`.
  //   - MTS API (`FusedInputTensor`): the `managed_tensor_slice.InputFusion`
  //     shape (IndexList `load`), bound onto a type-only dangling base slice.

  // The dtype the kernel reads; capture dtypes come from their defining ops.
  ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(getElementType());
  if (dtypeOr.isError())
    return emitOpError("failed to convert prologue element type to Mojo");
  MojoValue dtypeParam = state.mlirDTypeToMojoType(*dtypeOr);

  // One `LayoutType` param and (ptr, layout) field pair per capture (a fused
  // binary producer captures both inputs).
  FailureOr<SmallVector<FusedCaptureInfo>> captureInfosOr =
      collectFusedCaptureInfos(state, *this, getCaptures());
  if (failed(captureInfosOr))
    return failure();
  SmallVector<FusedCaptureInfo> captureInfos = *captureInfosOr;

  bool traitFused = static_cast<bool>(getContract());

  // Both lanes manufacture the loaded value (width param `Int`); they differ in
  // the trait/method/index shape the consumer accepts.
  const FusionLambdaInfo bindLoadLane = {
      /*structName=*/"PrologueLoadFunctor",
      /*traitName=*/kTraitInputFusion,
      /*methodName=*/MojoValue("load"),
      /*widthParamType=*/kStructInt,
      /*takesValue=*/false,
      /*coordIndex=*/false,
  };
  const FusionLambdaInfo traitLoadLane = {
      /*structName=*/"InputLoadFunctor",
      /*traitName=*/MojoValue("InputLoadFusion"),
      /*methodName=*/MojoValue("__call__"),
      /*widthParamType=*/kStructInt,
      /*takesValue=*/false,
      /*coordIndex=*/true,
  };

  // The wrapper's inline `InputLoadFusion` trait must be defined before the
  // functor that conforms to it.
  if (traitFused && !state.emittedInputLoadFusionArgDefs) {
    state.structDefsBuilder.verbatim(MojoValue(kInputLoadFusionDefs.str()));
    state.emittedInputLoadFusionArgDefs = true;
  }

  // Inner-dynamic load-index reshapes carry runtime shapes as trailing shape
  // captures, as in the epilogue.
  SmallVector<unsigned> shapeCaptureRanks;
  SmallVector<MojoValue> shapeCaptureArgs;
  for (Value shapeCap : getShapeCaptures()) {
    FailureOr<unsigned> rankOr = fusedRegionRank(*this, shapeCap);
    if (failed(rankOr))
      return failure();
    shapeCaptureRanks.push_back(*rankOr);
    shapeCaptureArgs.push_back(state.builder.var(
        MojoValue(kFnCoordToIndexList)({}, {state.getMojoValueFor(shapeCap)})));
  }

  FailureOr<MojoValue> functorOr = emitMapFusionFunctor(
      state, getBody(), captureInfos, traitFused ? traitLoadLane : bindLoadLane,
      /*valueDtype=*/dtypeParam, shapeCaptureRanks, shapeCaptureArgs);
  if (failed(functorOr))
    return failure();
  MojoValue functorInstance = *functorOr;

  FailureOr<unsigned> rank = fusedRegionRank(*this, getShape());
  if (failed(rank))
    return failure();
  MojoValue layoutVar = state.builder.var(
      MojoValue(kFnRowMajor)({}, {state.getMojoValueFor(getShape())}));
  MojoValue alignmentParam(static_cast<uint64_t>(
      M::Driver::getPreferredAlignment(getDevice().str())));

  if (traitFused) {
    // On CPU the host proxy runs the load (`_access_enabled`); on GPU the
    // projected `device_type` does, so the host proxy disables it.
    StringRef deviceLabel = getDevice();
    bool devicePassable = deviceLabel != kCPULabel;
    MojoValue wrapper = MojoValue("_MoggInputLoadFusionTensor")(
        {dtypeParam, MojoValue("type_of")({}, {layoutVar}), alignmentParam,
         MojoValue("type_of")({}, {functorInstance}), MojoValue(devicePassable),
         MojoValue(!devicePassable)},
        {functorInstance, layoutVar});
    state.setMojoValueFor(getTensor(), state.builder.var(wrapper));
    return success();
  }

  // Bind the functor onto a type-only dangling base slice: the tensor the
  // kernel READS has no slice to bind onto (fusion replaced it), so every load
  // goes through the functor.
  MojoValue baseSlice = emitDanglingBaseSlice(state, dtypeParam, *rank,
                                              layoutVar, alignmentParam);
  MojoValue fusedTensor = baseSlice.get(
      ManagedTensorSliceRef::kBindToFusedInput)({}, {functorInstance});
  state.setMojoValueFor(getTensor(), state.builder.var(fusedTensor));
  return success();
}

bool CallForeachOp::isThrows() { return true; }

// Collects the distinct MO dtypes touched anywhere in `body` as a
// `StaticTuple[DType, N]` MojoValue for `get_simd_width_for_dtypes`.
static FailureOr<MojoValue> collectDtypesTuple(EmitMojoState &state,
                                               Region &body) {
  llvm::SmallSetVector<DType, 4> dtypes;
  body.walk([&](Operation *op) {
    for (Type t :
         llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
      ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(t);
      if (!dtypeOr.isError())
        dtypes.insert(*dtypeOr);
    }
  });
  if (dtypes.empty())
    return failure();

  SmallVector<MojoValue> dtypeMojoValues;
  for (DType dtype : dtypes)
    dtypeMojoValues.push_back(state.mlirDTypeToMojoType(dtype));
  return MojoValue(kStructStaticTuple)[{
      MojoValue(kStructDType),
      MojoValue(static_cast<uint64_t>(dtypeMojoValues.size()))}](
      {}, dtypeMojoValues);
}

LogicalResult CallForeachOp::emitMojo(EmitMojoState &state) {
  // Emits the per-index closure directly, then calls `_foreach` with it.
  // `_foreach`'s `Lambda` parameter needs a blanket `{var}` capture clause, the
  // only convention that works for a closure capturing register-passable
  // values (see `MojoBuilder::lambda`'s `captureConvention`). As in the fusion
  // functors, each source is captured as its decomposed `ManagedTensorSlice`
  // components, so loads/stores go through
  // `simd_{load_from,store_into}_tensor_pointer`.
  //
  // The closure's comptime parameters are `[width, element_alignment]`;
  // `TensorLoadOp`/`TensorStoreOp` read them by role from
  // `state.currentLambdaInfo` (set via `LambdaInfosGuard` below), since the
  // epilogue `__call__` has a different list.
  MojoValue shape = state.getMojoValueFor(getShape());
  MojoValue context = state.getMojoValueFor(getContext());
  ValueRange capturing = getCapturing();

  // Decompose each capture into the (ptr, layout) the closure captures plus its
  // static alignment, exactly as `emitMapFusionFunctor` does. Built here, in
  // the enclosing scope, so the `{var}` closure captures the (ptr, layout)
  // locals (not the slice) and can name the comptime alignment alias.
  FailureOr<SmallVector<FusedCaptureInfo>> captureInfosOr =
      collectFusedCaptureInfos(state, *this, capturing);
  if (failed(captureInfosOr))
    return failure();
  SmallVector<FusedCaptureInfo> captureInfos = *captureInfosOr;

  MojoValue targetStr = MojoValue(("\"" + getDevice() + "\"").str());

  // Infer simd_width from all dtypes touched anywhere in the closure body.
  FailureOr<MojoValue> dtypesTuple = collectDtypesTuple(state, getBody());
  if (failed(dtypesTuple))
    return emitOpError("mogg.call.foreach: could not determine capture dtype");
  MojoValue simdWidth =
      MojoValue(kFnGetSimdWidthForDtypes)[{*dtypesTuple, targetStr}]({}, {});

  bool errOccurred = false;
  // Blanket `{var}` capture -- see the capture-convention note above.
  MojoValue lambdaVal = state.builder.lambda(
      "lambda_",
      {NamedMojoDef::asVerbatim(kParameterWidth, kStructInt),
       NamedMojoDef::asVerbatim(kParameterElementAlignment, kStructInt,
                                MojoValue(1))},
      {NamedMojoDef::asVerbatim("idx", kStructCoord)},
      /*returnType=*/std::nullopt, MojoBuilder::LambdaInline::kAutomatic,
      /*isParameter=*/false,
      /*isAsync=*/false, /*raises=*/false, /*captureConvention=*/"var",
      [&](const FunctionBody &fb) {
        EmitMojoState::LambdaInfosGuard lambdaInfosGuard(
            state, {/*lambdaBuilder=*/&state.builder,
                    /*widthParameter=*/fb.parameters[0],
                    /*alignmentParameter=*/fb.parameters[1]});

        Block &body = getBody().front();
        ArrayRef<BlockArgument> args = body.getArguments();
        for (unsigned i = 0; i < captureInfos.size(); ++i) {
          // Record the captured decomposed MTS components for the body's
          // loads/stores; the closure captures trivial values only, so all are
          // nameable here.
          const FusedCaptureInfo &info = captureInfos[i];
          state.setFusedTensorArgFor(
              args[i], {info.ptr, info.shape, info.strides, info.spec});
        }
        state.setMojoValueFor(args.back(), fb.arguments[0]);

        errOccurred = llvm::any_of(body, [&](Operation &op) {
          return state.lowerOpToMojo(&op).failed();
        });
      });
  if (errOccurred)
    return failure();

  state.builder.call(MojoValue(getCallee()),
                     {MojoValue(kParameterSimdWidth).equal(simdWidth),
                      MojoValue(kMoggTargetParameterName).equal(targetStr)},
                     {shape, lambdaVal, context});
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

void KernelOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  // Same rule as `KernelOp::getEffects` above, on the MAP-lowered kernel.
  if (!getOutputParamDecls().empty())
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());

  bool addedEffect = false;
  for (auto &&opOperand : getOperation()->getOpOperands()) {
    if (!MO::isMutableOperand(opOperand.get()))
      continue;
    if (getHasWriteEffect())
      effects.emplace_back(mlir::MemoryEffects::Write::get(), &opOperand);
    if (getHasReadEffect())
      effects.emplace_back(mlir::MemoryEffects::Read::get(), &opOperand);
    addedEffect = true;
  }

  if (!addedEffect && (getHasReadEffect() || getHasWriteEffect()))
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

//===----------------------------------------------------------------------===//
// AsScalarOp
//===----------------------------------------------------------------------===//

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

  // Each body block argument is a `!mogg.async<T>` wrapping its operand type
  // `T`, unpacked via `mogg.async.unpack`; one consumed as the raw async slot
  // (e.g. a context) may be retyped to plain `T` during ABI lowering.
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

    // Erase the block args of the removed chain operands.
    Block &entryBlock = *op.getBody().begin();
    entryBlock.eraseArguments(blockArgsToRemove);
    entryBlock.addArguments(newTypes, newLocs);

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

  // The last chain output is always kept; removed chains are rewired to it.
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
  // In the emit-mojo path mo.opaque and mgp.opaque are the same value.
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
// AsTupleOp
//===----------------------------------------------------------------------===//

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

OpFoldResult AsyncUnpackOp::fold(FoldAdaptor adaptor) {
  // Fold `mogg.async.unpack(mogg.async.pack(x))` -> `x`.
  if (auto packOp = getInput().getDefiningOp<AsyncPackOp>())
    return packOp.getInput();
  return {};
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
    auto tensorRank = tensorSpec.getRank();
    auto bufferRank = tensorRank == 0 ? 1 : tensorRank;
    std::optional<DType> maybeDType = tensorSpec.getOptDType();
    if (!maybeDType)
      return emitError() << "Expected Tensor operand to have DType";

    SmallVector<MojoValue> params = {bufferRank, tensorRank,
                                     state.mlirDTypeToMojoType(*maybeDType)};
    if (tensorType.getMut())
      params.push_back(MojoValue(kParameterMut).equal(MojoValue(true)));
    if (M::isHostPlaced(tensorType.getDevice()))
      params.push_back(MojoValue(kParameterHost).equal(MojoValue(true)));
    unpackedValue = asyncUnpackFunc(params, state.getMojoValueFor(input));

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

  // Opaque types: bitcast from LegacyOpaquePointer to the parameterized type,
  // then dereference at the use site:
  //   var_XX = value.unsafe_bitcast[Type]()
  //   ... later: var_XX[]
  if (auto opaqueType = dyn_cast<MO::OpaqueType>(getType())) {
    MojoValue baseTy = state.mlirOpaqueTypeToMojoType(opaqueType);
    auto params = EmitMojoState::getOpaqueTypeParameters(opaqueType);
    MojoValue opaqueTypeStr = params.empty() ? baseTy : baseTy[params];
    OpaquePointerRef opaquePtr(finalValue);
    MojoValue ptrVar = state.builder.var(opaquePtr.bitcast(opaqueTypeStr)());
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
  // codegen handle a MOGG-ABI stub argument expects.
  Type resultType = getOutput().getType();

  // Context / dim / opaque casts are code-generation no-ops:
  // `mogg.async.unpack` already produced the Mojo type the stub expects (a
  // `DeviceContext`, an `Int`, or the kernel's own struct), so forward it
  // unchanged.
  if (isa<MOGG::ContextType, MOGG::DimType, MO::OpaqueType>(resultType)) {
    state.setMojoValueFor(getResult(), state.getMojoValueFor(getInput()));
    return success();
  }

  // Pointer cast: the input is the `OwnedByteBuffer` an `!mgp.buffer` unpacks
  // to, but the MOGG-ABI stub takes a raw `OpaquePointer[MutAnyOrigin]` (it
  // does `ptr.bitcast[NoneType]()` in `tensor.create`). Take the buffer's data
  // pointer and bitcast it to the opaque pointer type.
  if (isa<MOGG::PointerType>(resultType)) {
    MojoValue rawPtr =
        OwnedByteBufferRef(state.getMojoValueFor(getInput())).unsafePtr();
    MojoValue opaquePtr =
        OpaquePointerRef(rawPtr).bitcast(MojoValue(kComptimeDTypeNoneType))();
    state.setMojoValueFor(getResult(), state.builder.var(opaquePtr));
    return success();
  }

  return emitOpError("mogg.rebind: unsupported result type for Mojo emission: ")
         << resultType;
}

//===----------------------------------------------------------------------===//
// AsyncPackOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncPackOp::emitMojo(EmitMojoState &state) {
  // A `!mgp.tensor` value is a parametric `OwnedTensor[dtype, rank]`; its
  // generic overload would lose resolution to the `Some[Movable & Deinitable]`
  // overload (mis-packing it as an opaque Mojo value). Use the dedicated
  // tensor pack function, passing the true spec rank so a scalar (rank 0) is
  // not packed as the promoted rank-1 buffer view.
  MojoValue packed;
  if (auto tensorType = dyn_cast<MGP::DeviceTensorType>(getInput().getType())) {
    MGP::TensorSpecType tensorSpec = tensorType.getTensorSpec();
    MojoValue specRankParam(static_cast<uint64_t>(tensorSpec.getRank()));
    packed = state.builder.var(MojoValue(kFnMoggAsyncPackOwnedTensorValue)(
        {specRankParam}, {state.getMojoValueFor(getInput())}));
  } else {
    MojoValue input = state.getMojoValueFor(getInput());
    Type inputType = getInput().getType();

    // Trivial types are passed by value: transferring them produces compiler
    // warnings, or fails to compile when the emitter produced a literal
    // rather than a var.
    //
    // Opaque Mojo values may be move-only, so the pack must consume them.
    bool isMoveOnly =
        isa<MO::OpaqueType, MGP::OpaqueType, MGP::DeviceGraphType>(inputType);
    if (!isa<mlir::IndexType, mlir::IntegerType, KGEN::SIMDType>(inputType) &&
        (isMoveOnly || getInput().hasOneUse()))
      input = input.transfer();
    // For every other supported type, overload resolution on
    // `mogg_async_pack_value` selects the right value-returning packer.
    packed = state.builder.var(MojoValue(kFnMoggAsyncPackValue)({}, {input}));
  }

  state.setMojoValueFor(getResult(), packed);
  return success();
}

OpFoldResult AsyncPackOp::fold(FoldAdaptor adaptor) {
  // Fold `mogg.async.pack(mogg.async.unpack(x))` -> `x`.
  if (auto unpackOp = getInput().getDefiningOp<AsyncUnpackOp>())
    return unpackOp.getInput();
  return {};
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
  // `Scalar[DType.int](<dim>)`) and build a row-major `Layout`, whose Coord
  // element types convey the static/dynamic structure to `mogg_tensor_init`.
  auto shapeVar = state.builder.var(
      state.getShapeAsCoordMojoValue(tensorTy.getShape(), getDims()));
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

// Inline Mojo definitions for the dense trait tensor-argument struct, emitted
// once per module (guarded by `EmitMojoState::emittedDenseTensorArgDefs`). One
// struct serves CPU and GPU: storage is always a pointer, and
// `_is_device_passable` selects whether it crosses to the device (the
// projected `TileTensor` is itself `DevicePassable`).
// `_can_project_tile_tensor` gates top-level `TileTensorable` conformance:
// true for CPU and GPU kernels, false for accelerators whose handles can't be
// reinterpreted as tiles.
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
    _can_project_tile_tensor: Bool,
](
    DenseTensor,
    DevicePassable where _is_device_passable,
    Input where _io == _MoggTensorIO.INPUT,
    MutableInput where _io == _MoggTensorIO.MUTABLE_INPUT,
    Output where _io == _MoggTensorIO.OUTPUT,
    # Top-level `TileTensorable` projection is allowed only for CPU/GPU kernels,
    # where the dense arg's pointer-backed storage can be directly viewed as a
    # host or unified-memory `TileTensor`. On discrete-GPU targets the pointer
    # still crosses via the `TileTensor` value's own `DevicePassable` encoding.
    TileTensorable where _can_project_tile_tensor,
):
    comptime mut = Self._io.is_mut()
    comptime dtype = Self._dtype
    comptime LayoutType = Self._LayoutType
    comptime ShapeType = Coord[*Self.LayoutType._shape_types]
    comptime rank = Self.LayoutType.rank
    comptime alignment = Self._alignment
    var handle: Pointer[Scalar[Self.dtype], UntrackedOrigin[mut = Self.mut]]
    var _layout: Self.LayoutType

    @inline(.always)
    def __init__(
        out self,
        ptr: Pointer[NoneType, MutAnyOrigin],
        layout: Self.LayoutType,
    ):
        self.handle = ptr.unsafe_bitcast[Scalar[Self.dtype]]().unsafe_mut_cast[
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
        Engine = Self.Engine,
    ]

    @inline(.always)
    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ) where Self._is_device_passable:
        var device_ptr = Pointer[
            Scalar[Self.dtype], UntrackedOrigin[mut = Self.mut]
        ].unsafe_dangling()
        self.handle._to_device_type(
            encoder, Pointer(to=device_ptr).unsafe_bitcast[NoneType]()
        )
        var device_value = TileTensor[
            dtype = Self.dtype,
            LayoutType = Self.LayoutType,
            origin = UntrackedOrigin[mut = Self.mut],
            Engine = Self.Engine,
        ](ptr=device_ptr, layout=self.layout())
        encoder.encode(device_value, target)

    @staticmethod
    def get_type_name() -> String where Self._is_device_passable:
        return String("_MoggDenseTensorArg")

    @inline(.always)
    def shape(self) -> Self.ShapeType:
        return rebind[Self.ShapeType](self.layout().shape_coord())

    @inline(.always)
    def layout(self) -> Self.LayoutType:
        return self._layout

    @inline(.always)
    def to_tile_tensor(
        self,
    ) -> TileTensor[
        dtype = Self.dtype,
        LayoutType = Self.LayoutType,
        origin = UntrackedOrigin[mut = Self.mut],
        Engine = Self.Engine,
    ] where Self._can_project_tile_tensor:
        return TileTensor[
            dtype = Self.dtype,
            LayoutType = Self.LayoutType,
            origin = UntrackedOrigin[mut = Self.mut],
            Engine = Self.Engine,
        ](ptr=self.handle, layout=self.layout())
)mojo";

// `_MoggShapeTensorArg`: shape functions run host-only, so this struct
// skips `_MoggDenseTensorArg`'s device-passable machinery. `_data_visible`
// gates off `DenseTensor`/`TileTensorable` for a bare `Tensor` bound.
// Emitted once per module, guarded by
// `EmitMojoState::emittedShapeTensorArgDefs`.
static constexpr llvm::StringLiteral kShapeTensorArgStructName =
    "_MoggShapeTensorArg";
static constexpr llvm::StringLiteral kShapeTensorArgDefs = R"mojo(
from layout import Coord, TensorLayout, TileTensor
from extensibility import DenseTensor, Tensor, TileTensorable


struct _MoggShapeTensorArg[
    _dtype: DType,
    _LayoutType: TensorLayout,
    _alignment: Int,
    _data_visible: Bool,
](
    Tensor,
    # `_data_visible` gates `DenseTensor`/`TileTensorable` off for a bare
    # `Tensor` bound, so such an argument cannot reach element values.
    # TODO(GEX-4093): derive per-input data-dependence from the written
    # bound instead of materializing a data pointer for `Tensor`-tier args.
    DenseTensor where _data_visible,
    TileTensorable where _data_visible,
):
    comptime mut = False
    comptime dtype = Self._dtype
    comptime LayoutType = Self._LayoutType
    comptime ShapeType = Coord[*Self.LayoutType._shape_types]
    comptime rank = Self.LayoutType.rank
    comptime alignment = Self._alignment
    var handle: Pointer[Scalar[Self.dtype], UntrackedOrigin[mut = Self.mut]]
    var _layout: Self.LayoutType

    @inline(.always)
    def __init__(
        out self,
        ptr: Pointer[NoneType, MutAnyOrigin],
        layout: Self.LayoutType,
    ):
        self.handle = ptr.unsafe_bitcast[Scalar[Self.dtype]]().unsafe_mut_cast[
            Self.mut
        ]().unsafe_origin_cast[UntrackedOrigin[mut = Self.mut]]()
        self._layout = layout

    @inline(.always)
    def shape(self) -> Self.ShapeType:
        return rebind[Self.ShapeType](self.layout().shape_coord())

    @inline(.always)
    def layout(self) -> Self.LayoutType:
        return self._layout

    @inline(.always)
    def to_tile_tensor(
        self,
    ) -> TileTensor[
        dtype = Self.dtype,
        LayoutType = Self.LayoutType,
        origin = UntrackedOrigin[mut = Self.mut],
    ]:
        return TileTensor[
            dtype = Self.dtype,
            LayoutType = Self.LayoutType,
            origin = UntrackedOrigin[mut = Self.mut],
        ](ptr=self.handle, layout=self.layout())
)mojo";

/// Returns true when the device label supports top-level `TileTensorable`
/// projection for dense tensor arguments: only CPU and GPU kernels'
/// pointer-backed storage can be viewed as a `TileTensor`.
static bool canProjectTileTensorForDeviceLabel(StringRef label) {
  return label == kCPULabel || label == kGPULabel;
}

/// Emit `_MoggDenseTensorArg[dtype, layoutType, alignment, io, devicePassable,
/// canProjectTileTensor](ptr, layout)`, defining the struct once per module.
/// The runtime args are a `Pointer[NoneType, MutAnyOrigin]` and the layout;
/// callers source them from a stub-bound capture or a codegen slice (via
/// `mogg.trait.create`).
static MojoValue emitDenseTensorArg(EmitMojoState &state, MojoValue dtype,
                                    MojoValue layoutType, MojoValue alignment,
                                    MojoValue io, MojoValue devicePassable,
                                    MojoValue canProjectTileTensor,
                                    MojoValue ptr, MojoValue layout) {
  if (!state.emittedDenseTensorArgDefs) {
    state.structDefsBuilder.verbatim(MojoValue(kDenseTensorArgDefs.str()));
    state.emittedDenseTensorArgDefs = true;
  }
  return state.builder.var(MojoValue("_MoggDenseTensorArg")(
      {dtype, layoutType, alignment, io, devicePassable, canProjectTileTensor},
      {ptr, layout}));
}

/// Emit `_MoggShapeTensorArg[dtype, layoutType, alignment, dataVisible](ptr,
/// layout)`, the host-only shape-lane counterpart, defined once per module.
static MojoValue emitShapeTensorArg(EmitMojoState &state, MojoValue dtype,
                                    MojoValue layoutType, MojoValue alignment,
                                    MojoValue dataVisible, MojoValue ptr,
                                    MojoValue layout) {
  if (!state.emittedShapeTensorArgDefs) {
    state.structDefsBuilder.verbatim(MojoValue(kShapeTensorArgDefs.str()));
    state.emittedShapeTensorArgDefs = true;
  }
  return state.builder.var(MojoValue(kShapeTensorArgStructName.str())(
      {dtype, layoutType, alignment, dataVisible}, {ptr, layout}));
}

LogicalResult TensorArgInitOp::emitMojo(EmitMojoState &state) {
  auto tensorTy = MO::dynCastFromTensorLike(getType());
  if (!tensorTy)
    return emitOpError("expected a tensor-like result type");

  MojoValue dTypeParam = state.mlirDTypeToMojoType(tensorTy.getDtype());
  auto shapeVar = state.builder.var(
      state.getShapeAsCoordMojoValue(tensorTy.getShape(), getDims()));
  auto layoutVar = state.builder.var(MojoValue(kFnRowMajor)({}, {shapeVar}));
  MojoValue layoutType = MojoValue("type_of")({}, {layoutVar});

  // Shape-function args assemble the host-only `_MoggShapeTensorArg`;
  // unlike the execute lane, there's no device-passable flavor to probe
  // for. `_data_visible` reflects the contract's written bound.
  if (getContract().getRole() == MOGG::TensorArgRoleKind::none) {
    MOGG::TensorArgBoundKind bound = getContract().getBound();
    if (bound == MOGG::TensorArgBoundKind::unspecified)
      return emitOpError("shape-function tensor argument has an unspecified "
                         "bound; expected tensor, dense, or tile");
    bool dataVisible = bound != MOGG::TensorArgBoundKind::tensor;
    MojoValue alignmentParam(
        static_cast<uint64_t>(M::Driver::getPreferredAlignment(
            MO::getDeviceOrDefaultAny(tensorTy).getLabel().str())));
    MojoValue ptr = OpaquePointerRef(ManagedTensorSliceRef(
                                         state.getMojoValueFor(getBuffer()))
                                         .unsafePtr())
                        .bitcast(MojoValue(kComptimeDTypeNoneType))();
    state.setMojoValueFor(
        getResult(),
        emitShapeTensorArg(state, dTypeParam, layoutType, alignmentParam,
                           MojoValue(dataVisible), ptr, layoutVar));
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
      llvm::formatv("{0}.unsafe_bitcast[Scalar[{1}]]().unsafe_mut_cast[True]()"
                    ".unsafe_origin_cast[MutUntrackedOrigin]()",
                    rawUnsafePtr.text, dTypeParam.text)
          .str());
  MojoValue structName("_MoggTensorBufferCapture");
  state.setMojoValueFor(getResult(),
                        state.builder.var(structName({dTypeParam, layoutType},
                                                     {ptr, layoutVar})));

  return success();
}

LogicalResult TraitCreateOp::emitMojo(EmitMojoState &state) {
  // Re-present the codegen tensor (a `ManagedTensorSlice` from
  // `mogg.tensor.create`, or a view) as the trait argument, sourcing pointer
  // and layout off the slice so any producer works, including a non-row-major
  // view.
  MojoValue slice = state.getMojoValueFor(getTensor());
  MOGG::TensorArgContractAttr contract = getContract();

  MojoValue dTypeParam(llvm::formatv("type_of({0}).dtype", slice.text).str());
  MojoValue layoutArg = state.builder.var(slice.get("runtime_layout")());
  MojoValue layoutType = MojoValue("type_of")({}, {layoutArg});
  MojoValue ptrArg = state.builder.var(
      llvm::formatv("{0}.unsafe_bitcast[NoneType]().as_unsafe_any_origin()",
                    ManagedTensorSliceRef(slice).unsafePtr().text)
          .str());

  // Device is on the op (like `mogg.tensor.create`), so alignment and the
  // device-passable / tile-projection flavors are picked locally.
  StringRef deviceLabel = getDevice();
  MojoValue alignmentParam(static_cast<uint64_t>(
      M::Driver::getPreferredAlignment(deviceLabel.str())));

  // A role-less contract is the host-only shape lane; the written bound decides
  // whether element data is visible.
  if (contract.getRole() == MOGG::TensorArgRoleKind::none) {
    MOGG::TensorArgBoundKind bound = contract.getBound();
    if (bound == MOGG::TensorArgBoundKind::unspecified)
      return emitOpError("shape-function trait argument has an unspecified "
                         "bound; expected tensor, dense, or tile");
    bool dataVisible = bound != MOGG::TensorArgBoundKind::tensor;
    state.setMojoValueFor(
        getResult(),
        emitShapeTensorArg(state, dTypeParam, layoutType, alignmentParam,
                           MojoValue(dataVisible), ptrArg, layoutArg));
    return success();
  }

  MojoValue ioParam = [&] {
    MojoValue ioStruct("_MoggTensorIO");
    switch (contract.getRole()) {
    case MOGG::TensorArgRoleKind::input:
      return ioStruct.get("INPUT");
    case MOGG::TensorArgRoleKind::output:
      return ioStruct.get("OUTPUT");
    case MOGG::TensorArgRoleKind::mutable_input:
      return ioStruct.get("MUTABLE_INPUT");
    case MOGG::TensorArgRoleKind::none:
      llvm_unreachable("shape lane handled above");
    }
    llvm_unreachable("unhandled TensorArgRoleKind");
  }();
  bool devicePassable = deviceLabel != kCPULabel;
  bool canProjectTileTensor = canProjectTileTensorForDeviceLabel(deviceLabel);
  state.setMojoValueFor(
      getResult(),
      emitDenseTensorArg(state, dTypeParam, layoutType, alignmentParam, ioParam,
                         MojoValue(devicePassable),
                         MojoValue(canProjectTileTensor), ptrArg, layoutArg));
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

  // The MOGG ABI is the advanced-fusion outlining path (a stub outlined from a
  // `mogg.kernel`); it reuses `mogg.call.stub` as its call site.
  if (calleeFunc->getAbi() != ABIKind::MOGG)
    return emitError("Call site should be in the MOGG ABI");

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
      MojoBuilder::LambdaInline::kAlways, /*isParameter=*/true,
      /*isAsync=*/true,
      /*raises=*/false, /*captureConvention=*/"", [&](const FunctionBody &) {
        state.builder
            .try_([&]() {
              state.builder.call(stubFunc, /*parameters=*/{}, stubArgs);
            })
            .except([&](const MojoValue &errVar) {
              state.builder.assign(errorVar, MojoValue(errVar.text + "^"));
            });
      });

  // 3. Launch the task. GPU launches pin it to the device's NUMA-affine CPU
  // worker via `_create_task`; otherwise `create_task` (desired_worker_id=-1).
  // The device ID is `$affinity_device_id` (I64Attr) because codegen device
  // refs are label-only.
  MojoValue taskVar;
  auto affinityId = getAffinityDeviceIdAttr();
  if (affinityId) {
    int64_t deviceId = affinityId.getInt();
    // var worker_id_N = max.runtime.asyncrt.task_id_for_device(<deviceId>)
    MojoValue workerIdVar =
        state.builder.var(MojoValue(kFnTaskIdForDevice)(
                              /*params=*/{}, /*args=*/{MojoValue(deviceId)}),
                          /*cache=*/false);
    // var task_N = std.runtime._asyncrt._create_task(
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
    // var task_N = std.runtime._asyncrt.create_task(wrapper_N())
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
  // Produce a fresh chain AnyAsyncValueRef.
  MojoValue ready = state.builder.var(
      MojoValue(kFnMoggAsyncReadyValue)(/*parameters=*/{}, /*operands=*/{}));
  state.setMojoValueFor(getResult(), ready);
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
  // The producing op never returns a chain.
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
  const EmitMojoState::MojoLambdaInfos &lambdaInfo = state.currentLambdaInfo;
  ASSERT_STREAM(
      lambdaInfo.lambdaBuilder,
      "CallElementwiseOp::emitMojo reached without currentLambdaInfo set");

  SmallVector<MojoValue> inputs = state.getMojoValuesFor(getInputs());

  // `$index` (present only if `CallElementwiseOp::lowerToMOGG` wanted it) is a
  // `layout.Coord`, the only form `CallForeachOp::emitMojo`'s closure produces,
  // so convert it to the kernel's `IndexList`.
  if (Value index = getIndex())
    inputs.push_back(
        MojoValue(kFnCoordToIndexList)({}, {state.getMojoValueFor(index)}));

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
      outputVal, lambdaInfo.lambdaBuilder->var(
                     mojoSymbolName(maybeParams->functionParameters, inputs)));
  return success();
}

//===----------------------------------------------------------------------===//
// EmptyBufferOp
//===----------------------------------------------------------------------===//

LogicalResult EmptyBufferOp::emitMojo(EmitMojoState &state) {
  // Represent the empty buffer with a dangling pointer.
  MojoValue danglingPointer =
      MojoValue(
          kStructPointer)[{MojoValue(kStructScalar)[{
                               MojoValue(kStructDType).get(kEnumDTypeInt8)}],
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
// AsTupleOp
//===----------------------------------------------------------------------===//

// MOGG_KernelCallInterface methods

LogicalResult
CallShapeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto decl = symbolTable.lookupNearestSymbolFrom<ShapeFunctionDeclOp>(
      *this, getCalleeAttr());
  if (!decl)
    return emitOpError("'")
           << getCallee() << "' does not refer to a shape_func.decl op";
  return success();
}

bool CallShapeOp::isThrows() { return getIsThrows(); }

LogicalResult CallShapeOp::emitMojo(EmitMojoState &state) {
  SmallVector<MojoValue> kernelOperands;
  for (Value operand : getOperands())
    kernelOperands.push_back(state.getMojoValueFor(operand));

  // Infer any kernel parameters required by the shape function.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    return state.inferDefaultKernelParamValue(*this, paramName,
                                              getKernelParamAttr());
  };
  ShapeFunctionDeclOp shapeDecl =
      state.registry.lookupShapeFunctionBySymName(getCallee());
  auto maybeParams = inferKernelParameters(shapeDecl, inferParam);
  if (maybeParams.isError())
    return emitError() << "Failed to infer all the kernel parameters: "
                       << maybeParams.takeError();

  MojoValue shapeFn(getCallee());
  state.setMojoValueFor(
      getResult(), state.builder.var(shapeFn[maybeParams->structParameters](
                       maybeParams->functionParameters, kernelOperands)));
  state.genAllKeepAliveStatements();
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

  MojoValue indexValue = MojoValue(std::to_string(index));
  state.setMojoValueFor(getResult(), shapeValue[{indexValue}]);

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantScalarOp
//===----------------------------------------------------------------------===//

/// Convert a scalar-constant attribute (IntegerAttr or FloatAttr) to a Mojo
/// literal an elementwise call can splat (e.g.
/// `Add.elementwise[dtype = ...](lhs_simd, 2.0)`), or std::nullopt.
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

LogicalResult ConstantScalarOp::verify() {
  if (getValue().getType() != getResult().getType())
    return emitOpError("value attribute type ")
           << getValue().getType() << " does not match result type "
           << getResult().getType();
  return success();
}

LogicalResult ConstantScalarOp::emitMojo(EmitMojoState &state) {
  std::optional<MojoValue> literal =
      scalarConstantAttrToMojoLiteral(getValue());
  if (!literal)
    return emitError("TODO: unsupported scalar type");

  // Splat to the closure's SIMD width (as `resolveScalarCaptureInput` does for
  // `!mo.scalar`): Mojo infers an elementwise call's width from its SIMD
  // operands, so an unsplatted first-operand constant would fail inference.
  // Outside a closure the bare literal stands.
  const EmitMojoState::MojoLambdaInfos &lambdaInfo = state.currentLambdaInfo;
  ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(getResult().getType());
  if (lambdaInfo.lambdaBuilder && !dtypeOr.isError()) {
    MojoValue simdTy =
        EmitMojoState::getSIMDType(*dtypeOr, lambdaInfo.widthParameter);
    literal = simdTy({}, {*literal});
  }

  state.setMojoValueFor(getResult(), *literal);
  return success();
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

LogicalResult OutputOp::emitMojo(EmitMojoState &state) {
  // Only the MGP ABI hands results back through the `void**` slot vector, so
  // only it populates `state.mojoReturnValues`.
  auto kernelStub = getOperation()->getParentOfType<MOGG::KernelStubOp>();
  if (!kernelStub || kernelStub.getAbi() != MOGG::ABIKind::MGP)
    return success();

  // `mogg.output` operands are the async-returned outputs, in slot order. Each
  // producer (mogg.async.pack/ready/error) has emitted an AnyAsyncValueRef
  // value; assign them into the corresponding output slots here.
  for (auto [idx, operand] : llvm::enumerate(getOperands())) {
    MojoValue value = state.getMojoValueFor(operand);
    state.builder.call(MojoValue(kFnMoggAssignAsyncValue), {},
                       {state.mojoReturnValues[idx], value});
  }
  return success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::emitMojo(EmitMojoState &state) {
  // Routes a region's produced value to one of the parent's results:
  // `mogg.try` forwarding a branch's, or `mogg.with_trace` forwarding a traced
  // dispatch's. The op's other legal parents, guaranteed by its `ParentOneOf`,
  // carry nothing out this way.
  Operation *parent = getOperation()->getParentOp();
  if (!isa<MOGG::TryOp, MOGG::WithTraceOp>(parent))
    return success();

  for (auto [operand, result] :
       llvm::zip_equal(getOperands(), parent->getResults())) {
    MojoValue value = state.getMojoValueFor(operand);
    // An opaque struct need not be `ImplicitlyCopyable` (`SIMDPair` is only
    // `RegisterPassable`), so transfer it into the result variable rather than
    // copying -- as `emitMojoMOGGABI` does when returning one.
    if (isa<MO::OpaqueType>(operand.getType()))
      value = value.transfer();
    state.builder.assign(state.getMojoValueFor(result), value);
  }

  return success();
}

/// Renders a value crossing a stub boundary, where an opaque is the kernel's
/// own struct rather than the `Pointer[NoneType]` `mlirTypeToMojoType` erases
/// it to. Same mapping `emitMojoMOGGABI` builds inline.
static MojoValue boundaryTypeToMojo(EmitMojoState &state, Type type) {
  auto opaqueType = dyn_cast<MO::OpaqueType>(type);
  if (!opaqueType)
    return state.mlirTypeToMojoType(type);
  MojoValue baseTy = state.mlirOpaqueTypeToMojoType(opaqueType);
  auto params = EmitMojoState::getOpaqueTypeParameters(opaqueType);
  return params.empty() ? baseTy : baseTy[params];
}

//===----------------------------------------------------------------------===//
// WithTraceOp
//===----------------------------------------------------------------------===//

bool WithTraceOp::isThrows() { return true; }

LogicalResult WithTraceOp::emitMojo(EmitMojoState &state) {
  // Find the enclosing KernelStubOp to get the device and fusion info.
  auto kernelStubOp = getOperation()->getParentOfType<MOGG::KernelStubOp>();
  if (!kernelStubOp)
    return emitOpError("WithTraceOp must be nested in a KernelStubOp");

  std::string deviceLabel = kernelStubOp.getDevice().str();
  // Distinct chains mean a kernel body shared by two non-nested
  // profile_scope()s (MergeIdenticalKernelStubs merges their locations);
  // formatProfileScope marks that with `:` rather than `/`. The scope follows
  // the kernel name (unlike AddTryExceptBlocks' `[scope] region_N`) so the
  // fusion-info name leads the trace.
  std::string traceName =
      kernelStubOp.getFusionInfoAttr().toString(/*includeLineNumbers=*/false,
                                                /*quote=*/false,
                                                /*stripDialectPrefixes=*/true) +
      formatProfileScope(collectProfileScopeLayers(kernelStubOp.getLoc()),
                         false);

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

  // Declare a variable per result before the `with`, so the yield inside can
  // assign it and the op consuming this trace's results still sees it in scope.
  // Same shape `TryOp::emitMojo` uses for a non-MGP ABI.
  for (OpResult result : getResults())
    state.setMojoValueFor(result, state.builder.varDecl(boundaryTypeToMojo(
                                      state, result.getType())));

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
// ProfileScopeBeginOp / ProfileScopeEndOp
//===----------------------------------------------------------------------===//

bool ProfileScopeBeginOp::isThrows() { return true; }

LogicalResult ProfileScopeBeginOp::emitMojo(EmitMojoState &state) {
  // Build the Trace object: Trace[TraceLevel.OP, target=StaticString(device)]
  // ("[scope] region_name") and call start() on it.
  auto kernelStubOp = getOperation()->getParentOfType<MOGG::KernelStubOp>();
  if (!kernelStubOp)
    return emitOpError("ProfileScopeBeginOp must be nested in a KernelStubOp");

  std::string deviceLabel = kernelStubOp.getDevice().str();

  SmallVector<MojoValue> traceParams = {
      MojoValue(kEnumTraceLevelOp),
      MojoValue(kMoggTargetParameterName)
          .equal(MojoValue(kStructStaticString)(
              {}, {MojoValue(deviceLabel).quote()}))};
  SmallVector<MojoValue> traceOperands = {MojoValue(getName()).quote()};
  if (auto color = getColor())
    traceOperands.push_back(
        MojoValue(kParameterColor)
            .equal(MojoValue(kStructColor)({}, {MojoValue(*color).quote()})));
  MojoValue traceExpr = MojoValue(kStructTrace)(traceParams, traceOperands);

  MojoValue traceVar = state.builder.var(traceExpr, /*cache=*/false);
  state.setMojoValueFor(getRange(), traceVar);
  state.builder.call(MojoValue(traceVar).get(kFnTraceStart), {}, {});
  return success();
}

bool ProfileScopeEndOp::isThrows() { return true; }

LogicalResult ProfileScopeEndOp::emitMojo(EmitMojoState &state) {
  MojoValue traceVar = state.getMojoValueFor(getRange());
  state.builder.call(MojoValue(traceVar).get(kFnTraceEnd), {}, {});
  return success();
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
  // Declare result variables for both branches' yields and the consuming
  // mogg.output. MGP forwards async values (its except block sets an error on
  // each), so those are `AnyAsyncValueRef`; other ABIs forward the result type
  // and raise instead, so only the try branch assigns.
  auto stubOp = getOperation()->getParentOfType<MOGG::KernelStubOp>();
  bool isMGPAbi = stubOp && stubOp.getAbi() == MOGG::ABIKind::MGP;
  for (OpResult result : getResults()) {
    MojoValue resultVar =
        isMGPAbi ? state.builder.var(MojoValue(kStructAnyAsyncValueRef)(),
                                     /*cache=*/false)
                 : state.builder.varDecl(
                       boundaryTypeToMojo(state, result.getType()));
    state.setMojoValueFor(result, resultVar);
  }

  // MOGG ABI stubs catch, format with kernel context, and re-raise; this only
  // emits the try/except structure. AsyncErrorOp::emitMojo adds the region
  // name.
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

    SmallVector<unsigned> usedResultIndices;
    SmallVector<Type> usedResultTypes;
    for (auto [idx, result] : llvm::enumerate(tryOp.getResults())) {
      if (!result.use_empty()) {
        usedResultIndices.push_back(idx);
        usedResultTypes.push_back(result.getType());
      }
    }

    if (usedResultIndices.size() == tryOp.getNumResults())
      return failure();

    auto newTryOp =
        TryOp::create(rewriter, tryOp.getLoc(), TypeRange(usedResultTypes));

    rewriter.cloneRegionBefore(tryOp.getTryRegion(), newTryOp.getTryRegion(),
                               newTryOp.getTryRegion().begin());

    rewriter.cloneRegionBefore(tryOp.getExceptRegion(),
                               newTryOp.getExceptRegion(),
                               newTryOp.getExceptRegion().begin());

    auto tryYield =
        cast<YieldOp>(newTryOp.getTryRegion().front().getTerminator());
    SmallVector<Value> newTryYieldOperands;
    for (unsigned idx : usedResultIndices) {
      newTryYieldOperands.push_back(tryYield.getOperand(idx));
    }
    rewriter.setInsertionPoint(tryYield);
    rewriter.replaceOpWithNewOp<YieldOp>(tryYield, newTryYieldOperands);

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
  MojoValue errorExpr = state.getMojoValueFor(getError());

  // Get the parent MGP region name for the error header.
  std::string regionName;
  if (auto parentKernel = getOperation()->getParentOfType<MOGG::KernelStubOp>())
    regionName = parentKernel.getSymNameAttr().str();

  // Format the error with the region name prefix.
  auto fmtRegionErrPrimDecl =
      state.registry.lookupPrimitive(kMoggPrimitiveFormatRegionError);
  ASSERT_STREAM(fmtRegionErrPrimDecl,
                "format_region_error primitive not found");
  MojoValue formatRegionErrorFunc(primitiveMojoName(fmtRegionErrPrimDecl));
  MojoValue formattedErr = state.builder.var(
      formatRegionErrorFunc({}, {MojoValue(regionName).quote(), errorExpr}));

  // Produce a fresh error AnyAsyncValueRef. The actual output slot assignment
  // is emitted by the mogg.output terminator (which consumes the TryOp result).
  MojoValue errorValue = state.builder.var(
      MojoValue(kFnMoggAsyncErrorValue)({}, {formattedErr.transfer()}));
  state.setMojoValueFor(getResult(), errorValue);

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
  MojoValue shapeValue =
      state.getShapeAsIndexListMojoValue(toType.getShape(), getDynamicDims());
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
