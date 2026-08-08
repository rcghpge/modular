//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Support/AssertStream.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace mlir;
using namespace M::MAP;

//===----------------------------------------------------------------------===//
// map.call.opaque — bracketed kernel-params directive
//===----------------------------------------------------------------------===//

// Parses the `key = value, ...` body between the brackets in `@callee[...]`
// into a dictionary (the `[` and `]` are literals in the op's assembly format).
// The bracket form (rather than the builtin `{...}` dictionary syntax) keeps
// the call reading as `@callee[params]`; `@callee` is parsed by the builtin
// symbol parser, which stops at the `[`.
static ParseResult parseKernelParamsBody(OpAsmParser &parser,
                                         DictionaryAttr &params) {
  SmallVector<NamedAttribute> entries;
  StringRef key;
  while (succeeded(parser.parseOptionalKeyword(&key))) {
    Attribute value;
    if (parser.parseEqual() || parser.parseAttribute(value))
      return mlir::failure();
    entries.emplace_back(StringAttr::get(parser.getContext(), key), value);
    if (failed(parser.parseOptionalComma()))
      break;
  }
  params = DictionaryAttr::get(parser.getContext(), entries);
  return mlir::success();
}

static void printKernelParamsBody(OpAsmPrinter &p, Operation *,
                                  DictionaryAttr params) {
  llvm::interleaveComma(params, p, [&](NamedAttribute attr) {
    p.printKeywordOrString(attr.getName().strref());
    p << " = ";
    p.printAttribute(attr.getValue());
  });
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GraphCompiler/MAPDialect/Dialect/MAP.cpp.inc"

//===----------------------------------------------------------------------===//
// map.call.opaque
//===----------------------------------------------------------------------===//

LogicalResult
CallOpaqueOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto decl = symbolTable.lookupNearestSymbolFrom<M::MOGG::KernelDeclOp>(
      *this, getCalleeAttr());
  if (!decl)
    return emitOpError("'")
           << getCallee() << "' does not refer to a kernel.decl op";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.call.elementwise
//===----------------------------------------------------------------------===//

LogicalResult
CallElementwiseOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto decl = symbolTable.lookupNearestSymbolFrom<M::MOGG::KernelDeclOp>(
      *this, getCalleeAttr());
  if (!decl)
    return emitOpError("'")
           << getCallee() << "' does not refer to a kernel.decl op";
  if (!decl.getElementwise())
    return emitOpError("kernel '")
           << getCallee() << "' has no 'elementwise' function";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Shared parse/print helpers
//===----------------------------------------------------------------------===//

// Prints `%v : type, ...` (no surrounding parens).
static void printValuesWithTypes(OpAsmPrinter &p, ValueRange values) {
  llvm::interleaveComma(values, p,
                        [&](Value v) { p << v << " : " << v.getType(); });
}

//===----------------------------------------------------------------------===//
// map.iter.opaque
//===----------------------------------------------------------------------===//

ParseResult IterOpaqueOp::parse(OpAsmParser &parser, OperationState &result) {
  // inputs ( %A : tA, ... ) -- shadowed as body block arguments (isolated).
  SmallVector<OpAsmParser::Argument> allArgInfo;
  if (parser.parseKeyword("inputs") ||
      parseParenOperandListWithShadowing(parser, result, allArgInfo,
                                         /*isIsolatedFromAbove=*/true))
    return mlir::failure();

  // outputs ( %out : tOut, ... ) -> region entry args + op result types
  SmallVector<OpAsmParser::Argument> outArgs;
  if (parser.parseKeyword("outputs") ||
      parser.parseArgumentList(outArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return mlir::failure();
  for (const OpAsmParser::Argument &arg : outArgs) {
    result.addTypes(arg.type);
    allArgInfo.push_back(arg);
  }

  // body region (entry block args are ordered [inputs..., outputs...])
  Region *body = result.addRegion();
  if (parseRegion(parser, allArgInfo, *body, /*isIsolatedFromAbove=*/true))
    return mlir::failure();

  return parser.parseOptionalAttrDict(result.attributes);
}

void IterOpaqueOp::print(OpAsmPrinter &p) {
  p.increaseIndent();
  p.printNewline();
  p << "inputs ";
  printParenOperandListWithShadowing(p, getInputs(), getInputBlockArgs(),
                                     /*isIsolatedFromAbove=*/true);
  p.printNewline();
  p << "outputs (";
  printValuesWithTypes(p, getOutputBlockArgs());
  p << ") ";
  p.decreaseIndent();

  // Shadow the whole region so inner references to the inputs print with the
  // outer operand's own SSA name (mirrors IterForeachOp::print); output block
  // args keep whatever name getAsmBlockArgumentNames already assigned them,
  // so shadowing them with themselves here is a no-op.
  {
    llvm::DenseSet<Value> seen;
    bool inputsDistinct = true;
    for (Value v : getInputs()) {
      if (!seen.insert(v).second) {
        inputsDistinct = false;
        break;
      }
    }
    if (inputsDistinct) {
      SmallVector<Value> shadowValues(getInputs().begin(), getInputs().end());
      for (BlockArgument arg : getOutputBlockArgs())
        shadowValues.push_back(arg);
      p.shadowRegionArgs(getBody(), shadowValues);
    }
  }
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs());
}

Block::BlockArgListType IterOpaqueOp::getOutputBlockArgs() {
  return getBody().front().getArguments().take_back(getNumResults());
}

Block::BlockArgListType IterOpaqueOp::getInputBlockArgs() {
  return getBody().front().getArguments().drop_back(getNumResults());
}

void IterOpaqueOp::getAsmBlockArgumentNames(Region &region,
                                            OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  auto args = region.front().getArguments().take_back(getNumResults());
  for (auto [i, arg] : llvm::enumerate(args))
    setNameFn(arg, args.size() == 1 ? "out" : ("out" + std::to_string(i)));
}

//===----------------------------------------------------------------------===//
// map.iter.epilogue
//===----------------------------------------------------------------------===//

ParseResult IterEpilogueOp::parse(OpAsmParser &parser, OperationState &result) {
  // Leading block args: ( %val : tVal, %idx : tIdx )
  SmallVector<OpAsmParser::Argument> entryArgs;
  if (parser.parseArgumentList(entryArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return mlir::failure();

  // capturing ( %inner = %outer : type, ... ): the region is isolated from
  // above, so each captured value is an operand (%outer, resolved in the
  // enclosing scope) exposed inside the body as a fresh block argument
  // (%inner). The `%inner = %outer` sugar names that block argument.
  SmallVector<OpAsmParser::UnresolvedOperand> captures;
  SmallVector<Type> captureTypes;
  if (parser.parseKeyword("capturing") || parser.parseLParen())
    return mlir::failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::Argument innerArg;
      OpAsmParser::UnresolvedOperand outer;
      Type type;
      if (parser.parseArgument(innerArg) || parser.parseEqual() ||
          parser.parseOperand(outer) || parser.parseColonType(type))
        return mlir::failure();
      innerArg.type = type;
      entryArgs.push_back(innerArg);
      captures.push_back(outer);
      captureTypes.push_back(type);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return mlir::failure();
  }
  if (parser.resolveOperands(captures, captureTypes,
                             parser.getCurrentLocation(), result.operands))
    return mlir::failure();

  // body region (isolated; entry args are [leading..., captures...]).
  // enableNameShadowing lets the capture block args reuse the same SSA names as
  // the captured operands, so the body reads as if it referenced them directly
  // (valid only because the region is IsolatedFromAbove).
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, entryArgs, /*enableNameShadowing=*/true))
    return mlir::failure();

  // : result-type
  Type resultType;
  if (parser.parseColonType(resultType))
    return mlir::failure();
  result.addTypes(resultType);

  return parser.parseOptionalAttrDict(result.attributes);
}

void IterEpilogueOp::print(OpAsmPrinter &p) {
  Block &block = getBody().front();
  ValueRange captures = getCaptures();
  unsigned numLeading = block.getNumArguments() - captures.size();

  p << " (";
  printValuesWithTypes(p, block.getArguments().take_front(numLeading));
  p << ") capturing (";
  llvm::interleaveComma(
      llvm::seq<unsigned>(0, captures.size()), p, [&](unsigned i) {
        BlockArgument inner = block.getArgument(numLeading + i);
        p << inner << " = " << captures[i] << " : " << inner.getType();
      });
  p << ") ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p << " : " << getResult().getType();
  p.printOptionalAttrDict((*this)->getAttrs());
}

void IterEpilogueOp::getAsmBlockArgumentNames(Region &region,
                                              OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  auto args = region.front().getArguments();
  unsigned numCaptures = getCaptures().size();
  unsigned numLeading = args.size() - numCaptures;
  static constexpr const char *kLeadingNames[] = {"val", "idx"};
  for (auto [i, arg] : llvm::enumerate(args)) {
    if (i < numLeading) {
      setNameFn(arg, i < 2 ? kLeadingNames[i] : ("arg" + std::to_string(i)));
    } else {
      unsigned c = i - numLeading;
      setNameFn(arg, numCaptures == 1 ? "cap" : ("cap" + std::to_string(c)));
    }
  }
}

//===----------------------------------------------------------------------===//
// map.iter.prologue
//===----------------------------------------------------------------------===//

ParseResult IterPrologueOp::parse(OpAsmParser &parser, OperationState &result) {
  // Leading block arg: ( %idx : tIdx )
  SmallVector<OpAsmParser::Argument> entryArgs;
  if (parser.parseArgumentList(entryArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return failure();

  // capturing ( %inner = %outer : type, ... ): same sugar as
  // map.iter.epilogue -- see there for why the region can be isolated from
  // above yet have the body read as if it referenced the outer values
  // directly.
  SmallVector<OpAsmParser::UnresolvedOperand> captures;
  SmallVector<Type> captureTypes;
  if (parser.parseKeyword("capturing") || parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::Argument innerArg;
      OpAsmParser::UnresolvedOperand outer;
      Type type;
      if (parser.parseArgument(innerArg) || parser.parseEqual() ||
          parser.parseOperand(outer) || parser.parseColonType(type))
        return failure();
      innerArg.type = type;
      entryArgs.push_back(innerArg);
      captures.push_back(outer);
      captureTypes.push_back(type);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }
  if (parser.resolveOperands(captures, captureTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  // body region (isolated; entry args are [idx, captures...]).
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, entryArgs, /*enableNameShadowing=*/true))
    return failure();

  // : result-type
  Type resultType;
  if (parser.parseColonType(resultType))
    return failure();
  result.addTypes(resultType);

  return parser.parseOptionalAttrDict(result.attributes);
}

void IterPrologueOp::print(OpAsmPrinter &p) {
  Block &block = getBody().front();
  ValueRange captures = getCaptures();
  unsigned numLeading = block.getNumArguments() - captures.size();

  p << " (";
  printValuesWithTypes(p, block.getArguments().take_front(numLeading));
  p << ") capturing (";
  llvm::interleaveComma(
      llvm::seq<unsigned>(0, captures.size()), p, [&](unsigned i) {
        BlockArgument inner = block.getArgument(numLeading + i);
        p << inner << " = " << captures[i] << " : " << inner.getType();
      });
  p << ") ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p << " : " << getResult().getType();
  p.printOptionalAttrDict((*this)->getAttrs());
}

void IterPrologueOp::getAsmBlockArgumentNames(Region &region,
                                              OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  auto args = region.front().getArguments();
  unsigned numCaptures = getCaptures().size();
  unsigned numLeading = args.size() - numCaptures;
  static constexpr const char *kLeadingNames[] = {"idx"};
  for (auto [i, arg] : llvm::enumerate(args)) {
    if (i < numLeading) {
      setNameFn(arg, i < 1 ? kLeadingNames[i] : ("arg" + std::to_string(i)));
    } else {
      unsigned c = i - numLeading;
      setNameFn(arg, numCaptures == 1 ? "cap" : ("cap" + std::to_string(c)));
    }
  }
}

//===----------------------------------------------------------------------===//
// map.iter.foreach
//===----------------------------------------------------------------------===//

// Parse a parenthesized list of block-argument-only entries: (name: type, ...).
// Does NOT add operands to the state — only populates argumentInfo.
static ParseResult
parseBlockArgList(OpAsmParser &parser,
                  SmallVectorImpl<OpAsmParser::Argument> &argInfo) {
  if (parser.parseLParen())
    return mlir::failure();
  if (succeeded(parser.parseOptionalRParen()))
    return mlir::success();
  do {
    OpAsmParser::Argument arg;
    if (parser.parseArgument(arg, /*allowType=*/true))
      return mlir::failure();
    argInfo.push_back(arg);
  } while (succeeded(parser.parseOptionalComma()));
  return parser.parseRParen();
}

// Print a parenthesized list of block arguments (name: type) without operands.
static void printBlockArgList(OpAsmPrinter &p, Block::BlockArgListType args) {
  p << "(";
  llvm::interleaveComma(args, p, [&](BlockArgument arg) {
    p.printOperand(arg);
    p << ": ";
    p.printType(arg.getType());
  });
  p << ")";
}

ParseResult IterForeachOp::parse(OpAsmParser &parser, OperationState &state) {
  // Parse shape: <[d0, d1, ...]>
  // Uses MOSH::parseShapeAttr which handles static ints, symbolic params,
  // and parametric expressions inside the [...] brackets.
  if (parser.parseLess())
    return mlir::failure();
  mlir::TypedAttr shapeTypedAttr;
  auto shapeResult = MOSH::parseShapeAttr(parser, shapeTypedAttr);
  if (!shapeResult.has_value())
    return parser.emitError(parser.getCurrentLocation(),
                            "expected '[' for map.iter.foreach shape");
  if (failed(*shapeResult))
    return mlir::failure();
  if (parser.parseGreater())
    return mlir::failure();
  state.addAttribute(getShapeAttrName(state.name), shapeTypedAttr);

  // Parse inputs with shadowing: inputs (%x: T, %y: T)
  SmallVector<OpAsmParser::Argument> allArgInfo;
  if (parser.parseKeyword("inputs") ||
      parseParenOperandListWithShadowing(parser, state, allArgInfo,
                                         /*isIsolatedFromAbove=*/true))
    return mlir::failure();
  int32_t numInputOperands = static_cast<int32_t>(state.operands.size());

  // Parse outputs (block args only): outputs (%out: T)
  SmallVector<OpAsmParser::Argument> outputArgInfo;
  if (parser.parseKeyword("outputs") ||
      parseBlockArgList(parser, outputArgInfo))
    return mlir::failure();
  for (auto &arg : outputArgInfo)
    allArgInfo.push_back(arg);

  // Parse index (block arg only): index (%idx: map.index<N>)
  SmallVector<OpAsmParser::Argument> indexArgInfo;
  if (parser.parseKeyword("index") || parseBlockArgList(parser, indexArgInfo))
    return mlir::failure();
  if (indexArgInfo.size() != 1)
    return parser.emitError(
        parser.getCurrentLocation(),
        "'map.iter.foreach' expects exactly one index argument");
  allArgInfo.push_back(indexArgInfo.front());

  // Parse optional chain (%c) clause -- a plain operand, never a block arg:
  // nothing in the per-element body ever touches the chain (pure sequencing
  // plumbing, mirrors mogg.kernel's `chain(...)`).
  int32_t numChainOperands = 0;
  if (succeeded(parser.parseOptionalKeyword("chain"))) {
    OpAsmParser::UnresolvedOperand chainOperand;
    if (parser.parseLParen() || parser.parseOperand(chainOperand) ||
        parser.parseRParen())
      return mlir::failure();
    if (parser.resolveOperand(chainOperand,
                              MO::ChainType::get(parser.getContext()),
                              state.operands))
      return mlir::failure();
    numChainOperands = 1;
  }
  state.addAttribute(IterForeachOp::getOperandSegmentSizesAttrName(state.name),
                     parser.getBuilder().getDenseI32ArrayAttr(
                         {numInputOperands, numChainOperands}));

  // Parse result types: -> T or (T, T, ...)
  if (parser.parseArrowTypeList(state.types))
    return mlir::failure();

  // Parse body region with all block args.
  if (parseRegion(parser, allArgInfo, *state.addRegion(),
                  /*isIsolatedFromAbove=*/true))
    return mlir::failure();

  return parser.parseOptionalAttrDict(state.attributes);
}

void IterForeachOp::print(OpAsmPrinter &p) {
  // Print shape: <[d0, d1, ...]>
  // Uses MOSH::printShapeValue to render each dim (integer, symbolic, or
  // parametric expression) without the outer "#mosh<ape[...]>" envelope.
  p << "<[";
  MOSH::printShapeValue(p, getShape().getValues());
  p << "]>";
  p << "\n    ";

  Block &body = getBody().front();
  unsigned numInputs = getInputs().size();
  // The output chain (if any) is always the last result and, like the input
  // chain, never has a body block argument -- only tensor results do.
  bool hasOutChain = getOutputChain() != nullptr;
  unsigned numOutputs = getResults().size() - (hasOutChain ? 1 : 0);

  // inputs (%x: T, ...) — operands shadowed as block args
  p << "inputs  ";
  printParenOperandListWithShadowing(p, getInputs(),
                                     body.getArguments().slice(0, numInputs),
                                     /*isIsolatedFromAbove=*/true);
  p << "\n    ";

  // outputs (%out: T, ...) — block args only
  p << "outputs ";
  printBlockArgList(p, body.getArguments().slice(numInputs, numOutputs));
  p << "\n    ";

  // index (%idx: map.index<N>) — block arg only
  p << "index   ";
  printBlockArgList(p, body.getArguments().slice(numInputs + numOutputs, 1));
  p << " ";

  // chain (%c) — optional plain operand, no block arg.
  if (Value chain = getInputChain())
    p << "chain(" << chain << ") ";

  // -> result types (on same line as index, opening brace follows inline)
  p.printArrowTypeList(getResultTypes());
  p << " ";

  // Body region.
  // printRegion() from ParserUtils requires operands.size() ==
  // region.getNumArguments(), but IterForeachOp's region has inputs + outputs +
  // index block args, while getInputs() covers only the input operands.
  // Instead: shadow input block args with their outer operand SSA names so
  // the body prints "%x", "%y", etc.  Output and index block args were
  // already assigned names by the printBlockArgList calls above; shadowing
  // them with themselves is a no-op that keeps those names consistent.
  {
    llvm::DenseSet<Value> seen;
    bool inputsDistinct = true;
    for (Value v : getInputs()) {
      if (!seen.insert(v).second) {
        inputsDistinct = false;
        break;
      }
    }
    if (inputsDistinct) {
      SmallVector<Value> shadowValues(getInputs().begin(), getInputs().end());
      for (BlockArgument arg : body.getArguments().drop_front(numInputs))
        shadowValues.push_back(arg);
      p.shadowRegionArgs(getBody(), shadowValues);
    }
  }
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);

  p.printOptionalAttrDict(
      getOperation()->getAttrs(),
      {"shape", getOperandSegmentSizesAttrName().getValue()});
}

LogicalResult IterForeachOp::verify() {
  Block &body = getBody().front();
  unsigned numInputs = getInputs().size();

  // A chain result, if present, is always last (mirrors
  // MOGG::KernelOp::maybeGetOutChain()) and, like the input chain, never
  // gets a body block argument -- only the preceding tensor results do.
  ResultRange results = getResults();
  for (unsigned i = 0; i < results.size(); ++i) {
    bool isChain = llvm::isa<MO::ChainType>(results[i].getType());
    if (isChain && i + 1 != results.size())
      return emitOpError() << "only the last result may be a chain";
  }
  bool hasOutChain = getOutputChain() != nullptr;
  unsigned numOutputs = results.size() - (hasOutChain ? 1 : 0);
  unsigned expectedArgs = numInputs + numOutputs + 1; // +1 for index

  if (body.getNumArguments() != expectedArgs)
    return emitOpError() << "body expects " << expectedArgs
                         << " block arguments (" << numInputs << " inputs + "
                         << numOutputs << " outputs + 1 index), "
                         << "got " << body.getNumArguments();

  // Input block arg types must match input operand types.
  for (unsigned i = 0; i < numInputs; ++i) {
    if (body.getArgument(i).getType() != getInputs()[i].getType())
      return emitOpError() << "input block argument " << i
                           << " type mismatch with operand";
  }

  // Output block arg types must match the non-chain result types.
  for (unsigned i = 0; i < numOutputs; ++i) {
    if (body.getArgument(numInputs + i).getType() != results[i].getType())
      return emitOpError() << "output block argument " << i
                           << " type mismatch with result";
  }

  // Index arg must be map.index<N> where N == shape.size().
  BlockArgument indexArg = body.getArgument(numInputs + numOutputs);
  auto indexTy = llvm::dyn_cast<IndexType>(indexArg.getType());
  if (!indexTy)
    return emitOpError() << "index block argument must be map.index<N>";

  if (indexTy.getRank() != getShape().getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match shape rank "
                         << getShape().getRank();

  return mlir::success();
}

Value IterForeachOp::getOutputChain() {
  ResultRange results = getResults();
  if (results.empty())
    return nullptr;
  Value last = results.back();
  return llvm::isa<MO::ChainType>(last.getType()) ? last : Value();
}

// Mirrors MOGG::KernelOp::getEffects(): hasReadEffect/hasWriteEffect carry
// forward the original MO op's memory effects (see lowerMutableStoreToForeach
// in MOToMAP.cpp) so downstream passes see accurate effects instead of the
// conservative "unknown" default.
void IterForeachOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  bool addedEffect = false;
  for (OpOperand &opOperand : getOperation()->getOpOperands()) {
    if (!MO::isMutableOperand(opOperand.get()))
      continue;
    if (getHasWriteEffect())
      effects.emplace_back(MemoryEffects::Write::get(), &opOperand);
    if (getHasReadEffect())
      effects.emplace_back(MemoryEffects::Read::get(), &opOperand);
    addedEffect = true;
  }

  if (!addedEffect && (getHasReadEffect() || getHasWriteEffect()))
    effects.emplace_back(MemoryEffects::Write::get());
}

llvm::SmallVector<M::DeviceRefAttr> IterForeachOp::getExecutionDevices() {
  // Scan results then inputs for the first value with a device annotation.
  for (Value v : llvm::concat<Value>(getResults(), getInputs())) {
    if (auto dt = llvm::dyn_cast<M::MO::MOHasDeviceTypeInterface>(v.getType()))
      if (M::DeviceRefAttr device = dt.getDeviceRef())
        return {device};
  }
  return {M::DeviceRefAttr{}};
}

void IterForeachOp::insertInputsAt(unsigned idx, ValueRange vals) {
  ASSERT_STREAM(idx <= getInputs().size(),
                << "idx (" << idx << ") must stay within the inputs ("
                << getInputs().size() << "), before outputs/index block args");
  SmallVector<Value> ops(getInputs());
  ops.insert(ops.begin() + idx, vals.begin(), vals.end());
  getInputsMutable().assign(ops);

  for (auto v : llvm::reverse(vals))
    getBody().insertArgument(idx, v.getType(), v.getLoc());
}

void IterForeachOp::eraseInputsAt(unsigned idx, size_t len) {
  ASSERT_STREAM(idx + len <= getInputs().size(),
                << "[idx, idx + len) = [" << idx << ", " << (idx + len)
                << ") must stay within the inputs (" << getInputs().size()
                << "), before outputs/index block args");
  getInputsMutable().erase(idx, len);

  for (unsigned i = 0; i < len; ++i)
    getBody().eraseArgument(idx);
}

//===----------------------------------------------------------------------===//
// map.iter.foreach.body
//===----------------------------------------------------------------------===//

ParseResult IterForeachBodyOp::parse(OpAsmParser &parser,
                                     OperationState &state) {
  // Parse shape: <[d0, d1, ...]>
  if (parser.parseLess())
    return mlir::failure();
  mlir::TypedAttr shapeTypedAttr;
  auto shapeResult = MOSH::parseShapeAttr(parser, shapeTypedAttr);
  if (!shapeResult.has_value())
    return parser.emitError(parser.getCurrentLocation(),
                            "expected '[' for map.iter.foreach.body shape");
  if (failed(*shapeResult))
    return mlir::failure();
  if (parser.parseGreater())
    return mlir::failure();
  state.addAttribute(getShapeAttrName(state.name), shapeTypedAttr);

  // Parse index (block arg only): index (%idx: map.index<N>)
  SmallVector<OpAsmParser::Argument> indexArgInfo;
  if (parser.parseKeyword("index") || parseBlockArgList(parser, indexArgInfo))
    return mlir::failure();
  if (indexArgInfo.size() != 1)
    return parser.emitError(
        parser.getCurrentLocation(),
        "'map.iter.foreach.body' expects exactly one index argument");

  // Body region (not isolated; entry arg is just the index).
  Region *body = state.addRegion();
  if (parser.parseRegion(*body, indexArgInfo))
    return mlir::failure();

  return parser.parseOptionalAttrDict(state.attributes);
}

void IterForeachBodyOp::print(OpAsmPrinter &p) {
  p << "<[";
  MOSH::printShapeValue(p, getShape().getValues());
  p << "]> ";

  p.increaseIndent();
  p.printNewline();
  p << "index   ";
  printBlockArgList(p, getBody().front().getArguments());
  p << " ";
  p.decreaseIndent();

  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(getOperation()->getAttrs(), {"shape"});
}

LogicalResult IterForeachBodyOp::verify() {
  Block &body = getBody().front();
  if (body.getNumArguments() != 1)
    return emitOpError() << "body expects exactly one block argument (the "
                         << "index), got " << body.getNumArguments();

  auto indexTy = llvm::dyn_cast<IndexType>(body.getArgument(0).getType());
  if (!indexTy)
    return emitOpError() << "index block argument must be map.index<N>";

  if (indexTy.getRank() != getShape().getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match shape rank "
                         << getShape().getRank();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.load
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify() {
  auto tensorTy = llvm::dyn_cast<M::MO::TensorType>(getTensor().getType());
  if (!tensorTy)
    return emitOpError() << "tensor operand must be !mo.tensor type";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.store
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify() {
  if (!M::MO::dynCastFromTensorLike(getTensor().getType()))
    return emitOpError()
           << "tensor operand must be !mo.tensor or !mo.buffer type";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.index.slice
//===----------------------------------------------------------------------===//

// Shared by every view op's starts/stops/steps operands (map.index.slice,
// map.index.slice.stride): each must be a rank-1 !mo.tensor.
static LogicalResult verifySliceBoundsOperandsAreRank1(
    Operation *op, ArrayRef<std::pair<StringRef, Value>> bounds) {
  for (auto [name, value] : bounds) {
    auto tensorTy = llvm::dyn_cast<M::MO::TensorType>(value.getType());
    if (!tensorTy || tensorTy.getRank() != 1)
      return op->emitOpError()
             << name << " operand must be a rank-1 !mo.tensor";
  }
  return mlir::success();
}

LogicalResult SliceIndexOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto indexTy = llvm::cast<IndexType>(getIndex().getType());
  if (indexTy.getRank() != signature.getOut().getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match spec's out-shape rank "
                         << signature.getOut().getRank();

  auto resultTy = llvm::cast<IndexType>(getResult().getType());
  if (resultTy.getRank() != signature.getIn().getRank())
    return emitOpError() << "result rank " << resultTy.getRank()
                         << " does not match spec's in-shape rank "
                         << signature.getIn().getRank();

  return verifySliceBoundsOperandsAreRank1(getOperation(),
                                           {std::pair{"starts", getStarts()},
                                            {"stops", getStops()},
                                            {"steps", getSteps()}});
}
