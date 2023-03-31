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
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace M;
using namespace GRT;

/// Parse a parenthesized operand list with types included, e.g.
/// `(%a: i32, %b: f32)`.
static ParseResult
parseParenOperandList(OpAsmParser &parser, OperationState &result,
                      SmallVectorImpl<OpAsmParser::Argument> &argumentInfo) {
  auto parseOperandFn = [&]() -> ParseResult {
    OpAsmParser::Argument arg;
    if (parser.parseArgument(arg, /*allowType=*/true) ||
        parser.resolveOperand(arg.ssaName, arg.type, result.operands))
      return failure();
    argumentInfo.push_back(arg);
    return success();
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseOperandFn, "in operand list");
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

ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> operandInfo;
  llvm::SMLoc operandLoc;
  OpAsmParser::UnresolvedOperand condOperand;
  if (parser.parseOperand(condOperand) ||
      parser.getCurrentLocation(&operandLoc) ||
      parser.resolveOperand(condOperand,
                            IntegerType::get(result.getContext(), 1),
                            result.operands) ||
      parseParenOperandList(parser, result, operandInfo) ||
      parser.parseArrowTypeList(result.types) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the then region.
  Region *thenRegion = result.addRegion();
  if (parser.parseRegion(*thenRegion, operandInfo,
                         /*enableNameShadowing=*/true))
    return failure();

  Region *elseRegion = result.addRegion();
  if (parser.parseKeyword("else") ||
      parser.parseRegion(*elseRegion, operandInfo,
                         /*enableNameShadowing=*/true))
    return failure();

  return success();
}

void IfOp::print(OpAsmPrinter &p) {
  p << " " << getCond() << " (";
  llvm::interleaveComma(getInputs(), p,
                        [&](Value arg) { p << arg << ": " << arg.getType(); });
  p << ") ";
  p.printArrowTypeList(getResultTypes());
  p << ' ';
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());

  // Reuse the argument names provided to the op for the basic block arg names
  // within the region.
  p.shadowRegionArgs(getThenRegion(), getInputs());
  p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false);
  p << " else ";
  p.shadowRegionArgs(getElseRegion(), getInputs());
  p.printRegion(getElseRegion(), /*printEntryBlockArgs=*/false);
}

LogicalResult IfOp::verify() {
  // Verify that the operands match the bb arguments.  The ODS verifier already
  // checked the first argument to be present and i1.
  auto *thenBlock = &getThenRegion().front();
  if (getInputs().size() != thenBlock->getNumArguments())
    return emitOpError("incorrect number of arguments to 'then' block");
  auto *elseBlock = &getElseRegion().front();
  if (getInputs().size() != elseBlock->getNumArguments())
    return emitOpError("incorrect number of arguments to 'else' block");

  for (unsigned i = 0, e = getNumOperands() - 1; i != e; ++i)
    if (getInputs()[i].getType() != thenBlock->getArgument(i).getType() ||
        getInputs()[i].getType() != elseBlock->getArgument(i).getType())
      return emitOpError("operand/argument type mismatch");

  auto thenOutput = cast<MEF::OutputOp>(thenBlock->getTerminator());
  if (failed(thenOutput.checkArgumentTypes(getResultTypes())))
    return failure();

  auto elseOutput = cast<MEF::OutputOp>(elseBlock->getTerminator());
  return elseOutput.checkArgumentTypes(getResultTypes());
}

//===----------------------------------------------------------------------===//
// RepeatI32Op
//===----------------------------------------------------------------------===//

ParseResult RepeatI32Op::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> operandInfo;
  llvm::SMLoc operandLoc;
  OpAsmParser::UnresolvedOperand countOperand;
  if (parser.parseOperand(countOperand) ||
      parser.getCurrentLocation(&operandLoc) ||
      parser.resolveOperand(
          countOperand,
          IntegerType::get(result.getContext(), 32, IntegerType::Signed),
          result.operands) ||
      parseParenOperandList(parser, result, operandInfo) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // The result types are the same as the operands (ignoring count).
  for (Value operand : llvm::drop_begin(result.operands))
    result.types.push_back(operand.getType());

  // Parse the body region.
  Region *body = result.addRegion();
  return parser.parseRegion(*body, operandInfo,
                            /*enableNameShadowing=*/true);
}

void RepeatI32Op::print(OpAsmPrinter &p) {
  p << ' ' << getTripCount() << " (";
  llvm::interleaveComma(getInputs(), p,
                        [&](Value arg) { p << arg << ": " << arg.getType(); });
  p << ") ";
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());

  // Reuse the argument names provided to the op for the basic block arg names
  // within the region.
  p.shadowRegionArgs(getRegion(), getInputs());
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
}

LogicalResult RepeatI32Op::verify() {
  // Verify that the operand and result types match.
  if (getNumResults() != getInputs().size())
    return emitOpError("incorrect number of operands");

  for (unsigned i = 0, e = getNumResults(); i != e; ++i)
    if (getInputs()[i].getType() != getResult(i).getType())
      return emitOpError("operand/result type mismatch");

  auto output = cast<MEF::OutputOp>(getBody()->getTerminator());
  return output.checkArgumentTypes(getResultTypes());
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
      parseParenOperandList(parser, result, operandInfo))
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
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "GraphRT/GRTDialect/GRT.cpp.inc"
