//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the GraphRT dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/GRTDialect/GRTOps.h"
#include "Support/Compiler/ParserUtils.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace M;
using namespace GRT;

//===----------------------------------------------------------------------===//
// ChainCreateOp
//===----------------------------------------------------------------------===//

void ChainCreateOp::build(::mlir::OpBuilder &odsBuilder,
                          ::mlir::OperationState &odsState) {
  build(odsBuilder, odsState, odsBuilder.getType<MEF::ChainType>(), {});
}

ParseResult ChainCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  auto chainType = MEF::ChainType::get(parser.getContext());

  if (parseParenOperandListWithDefaultType(parser, result, chainType) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Result type is deterministic
  result.types.push_back(chainType);

  return success();
}

void ChainCreateOp::print(OpAsmPrinter &p) {
  auto chainType = MEF::ChainType::get(getContext());
  p << ' ';
  printParenOperandListWithDefaultType(p, getOperands(), chainType);
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
}

//===----------------------------------------------------------------------===//
// ChainFanoutOp
//===----------------------------------------------------------------------===//

LogicalResult ChainFanoutOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  assert(operands.size() == 1 && "mismatched operand arity");
  auto numResultsAttr =
      dyn_cast<IntegerAttr>(attributes.get(getNumResultsAttrName(name)));
  assert(numResultsAttr && "mismatched operand types");
  uint64_t size = numResultsAttr.getUInt();
  auto chainType = MEF::ChainType::get(context);
  for (uint64_t i = 0; i < size; ++i)
    inferredReturnTypes.emplace_back(chainType);
  return success();
}

LogicalResult ChainFanoutOp::verify() {
  if (getNumResults() == 0)
    return emitOpError("expects at least one result");
  return success();
}

//===----------------------------------------------------------------------===//
// InvokeOp
//===----------------------------------------------------------------------===//

LogicalResult InvokeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the target attribute was specified.
  MEF::InvokableOp fn = symbolTable.lookupNearestSymbolFrom<MEF::InvokableOp>(
      *this, getTargetAttr());
  if (!fn)
    return emitOpError() << "'" << getTargetAttr()
                         << "' does not reference a valid invokable";

  // Verify that the operand and result types match the target.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for target");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for target");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

ParseResult IfOp::parse(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::Argument> argumentInfo;
  OpAsmParser::UnresolvedOperand condOperand;
  Region *thenRegion = state.addRegion();
  Region *elseRegion = state.addRegion();
  if (parser.parseOperand(condOperand) ||
      parser.resolveOperand(condOperand,
                            IntegerType::get(state.getContext(), 1),
                            state.operands) ||
      parseParenOperandListWithShadowing(parser, state, argumentInfo) ||
      parser.parseArrowTypeList(state.types) ||
      parser.parseOptionalAttrDictWithKeyword(state.attributes) ||
      parseRegionWithShadowing(parser, argumentInfo, *thenRegion) ||
      parser.parseKeyword("else") ||
      parseRegionWithShadowing(parser, argumentInfo, *elseRegion))
    return failure();

  return success();
}

void IfOp::print(OpAsmPrinter &p) {
  p << ' ' << getCond() << ' ';
  printParenOperandListWithShadowing(p, getInputs(),
                                     getThenRegion().getArguments());
  p.printArrowTypeList(getResultTypes());
  p << ' ';
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  printRegionWithShadowing(p, getInputs(), getThenRegion());
  p << " else ";
  printRegionWithShadowing(p, getInputs(), getElseRegion());
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

ParseResult WhileOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> argumentInfo;
  Region *condRegion = result.addRegion();
  Region *bodyRegion = result.addRegion();
  if (parseParenOperandListWithShadowing(parser, result, argumentInfo) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parseRegionWithShadowing(parser, argumentInfo, *condRegion) ||
      parser.parseKeyword("do") ||
      parseRegionWithShadowing(parser, argumentInfo, *bodyRegion))
    return failure();

  // The op's return types match those of the signature arguments.
  result.types.reserve(result.operands.size());
  for (auto operand : result.operands)
    result.types.push_back(operand.getType());

  return success();
}

void WhileOp::print(OpAsmPrinter &p) {
  p << ' ';
  printParenOperandListWithShadowing(p, getInputs(),
                                     getCondRegion().getArguments());
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  p << ' ';
  printRegionWithShadowing(p, getInputs(), getCondRegion());
  p << " do ";
  printRegionWithShadowing(p, getInputs(), getBodyRegion());
}

LogicalResult WhileOp::verify() {
  // The 'cond' block should return an additional i1.
  Block *condBlock = &getCondRegion().front();
  auto condOutput = cast<MEF::OutputOp>(condBlock->getTerminator());
  if (condOutput->getNumOperands() != getInputs().size() + 1 ||
      condOutput->getOperand(getInputs().size()).getType() !=
          IntegerType::get(getContext(), 1))
    return emitOpError() << "expected 'cond' block to return additional i1 "
                            "result for loop condition";

  // The 'body' block should not return any additional results.
  Block *bodyBlock = &getBodyRegion().front();
  if (bodyBlock->getNumArguments() != getInputs().size())
    return emitOpError()
           << "expected 'body' block to return no additional results";

  return success();
}

//===----------------------------------------------------------------------===//
// RepeatI32Op
//===----------------------------------------------------------------------===//

ParseResult RepeatI32Op::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> argumentInfo;
  OpAsmParser::UnresolvedOperand countOperand;
  Region *body = result.addRegion();
  if (parser.parseOperand(countOperand) ||
      parser.resolveOperand(
          countOperand,
          IntegerType::get(result.getContext(), 32, IntegerType::Signed),
          result.operands) ||
      parseParenOperandListWithShadowing(parser, result, argumentInfo) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parseRegionWithShadowing(parser, argumentInfo, *body))
    return failure();

  // The result types are the same as the operands (ignoring 'tripCount').
  result.types.reserve(result.operands.size() - 1);
  for (auto operand : llvm::drop_begin(result.operands))
    result.types.push_back(operand.getType());

  return success();
}

void RepeatI32Op::print(OpAsmPrinter &p) {
  p << ' ' << getTripCount() << ' ';
  printParenOperandListWithShadowing(p, getInputs(),
                                     getBodyRegion().getArguments());
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  p << ' ';
  printRegionWithShadowing(p, getInputs(), getBodyRegion());
}

LogicalResult RepeatI32Op::verify() {
  // The 'body' block should not return any additional results.
  Block *bodyBlock = &getBodyRegion().front();
  if (bodyBlock->getNumArguments() != getInputs().size())
    return emitOpError()
           << "expected 'body' block to return no additional results";

  return success();
}

//===----------------------------------------------------------------------===//
// BenchmarkOp
//===----------------------------------------------------------------------===//

ParseResult BenchmarkOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the description string for the benchmark.
  StringAttr description;
  if (parser.parseAttribute(description, "description", result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument> operandInfo;
  llvm::SMLoc argLoc;
  if (parser.getCurrentLocation(&argLoc) ||
      parseParenOperandListWithShadowing(parser, result, operandInfo))
    return failure();

  if (operandInfo.empty() || !operandInfo[0].type.isa<MEF::ChainType>())
    return parser.emitError(
        argLoc, "grt.benchmark must take a chain as its first argument");

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Save the current location.
  llvm::SMLoc opLoc;
  if (parser.getCurrentLocation(&opLoc))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, operandInfo,
                         /*enableNameShadowing=*/true))
    return failure();

  // Infer the return types from the terminator's operands.
  Operation *bodyTerminator = body->front().getTerminator();
  result.types.assign(bodyTerminator->operand_type_begin(),
                      bodyTerminator->operand_type_end());
  if (!result.types.front().isa<MEF::ChainType>())
    return parser.emitError(
        opLoc, "grt.benchmark must return a chain as its first result.");

  return success();
}

void BenchmarkOp::print(OpAsmPrinter &p) {
  p << ' ' << getDescription() << " (";
  llvm::interleaveComma(getInputs(), p,
                        [&](Value arg) { p << arg << ": " << arg.getType(); });
  p << ") ";
  p.printOptionalAttrDict(getOperation()->getAttrs());
  // Reuse the argument names provided to the op for the arg names within
  // the region.
  p.shadowRegionArgs(getCodeUnderTest(), getOperands());
  p << " ";
  p.printRegion(getCodeUnderTest(), /*printEntryBlockArgs=*/false);
}

LogicalResult BenchmarkOp::verify() {
  bool isValidTimeUnit = llvm::StringSwitch<bool>(getTimeUnit())
                             .Cases("ns", "us", "ms", "s", true)
                             .Default(false);
  if (!isValidTimeUnit)
    return emitOpError(Twine("invalid timeUnit '") + getTimeUnit() + "'");
  return success();
}

//===----------------------------------------------------------------------===//
// TestcaseOp
//===----------------------------------------------------------------------===//

ParseResult TestcaseOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse just the symbol name first.
  StringAttr symbol;
  if (parser.parseSymbolName(symbol, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the body region. It has no arguments whatsoever.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {}, false))
    return failure();

  // Infer the return types from the terminator's operands.
  Operation *bodyTerminator = body->front().getTerminator();
  result.types.assign(bodyTerminator->operand_type_begin(),
                      bodyTerminator->operand_type_end());

  return success();
}

void TestcaseOp::print(OpAsmPrinter &p) { p.printRegion(getBody()); }

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GraphRT/GRTDialect/GRT.cpp.inc"
