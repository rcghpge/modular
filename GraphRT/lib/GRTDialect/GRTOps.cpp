//===- GraphRTOps.cpp -----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the GraphRT dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/GRTDialect/GRTOps.h"
#include "Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace M;
using namespace GRT;

/// Parse a parenthesized operand list with types included, e.g.
/// `(%a: i32, %b: f32)`.
static ParseResult
parseParenOperandList(OpAsmParser &parser, OperationState &result,
                      SmallVectorImpl<OpAsmParser::OperandType> &operandInfo,
                      SmallVectorImpl<Location> &operandLocs) {
  auto parseOperandFn = [&]() -> ParseResult {
    OpAsmParser::OperandType curOperand;
    Type type;
    if (parser.parseOperand(curOperand) || parser.parseColonType(type) ||
        parser.resolveOperand(curOperand, type, result.operands))
      return failure();
    operandInfo.push_back(curOperand);
    operandLocs.push_back(parser.getEncodedSourceLoc(curOperand.location));
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
  BEF::InvokableOp fn = symbolTable.lookupNearestSymbolFrom<BEF::InvokableOp>(
      *this, targetAttr());
  if (!fn)
    return emitOpError() << "'" << targetAttr()
                         << "' does not reference a valid invokable";

  // Verify that the operand and result types match the target.
  auto fnType = fn.getType();
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
  SmallVector<OpAsmParser::OperandType> operandInfo;
  SmallVector<Location> operandLocs;
  llvm::SMLoc operandLoc;
  OpAsmParser::OperandType condOperand;
  if (parser.parseOperand(condOperand) ||
      parser.getCurrentLocation(&operandLoc) ||
      parser.resolveOperand(condOperand,
                            IntegerType::get(result.getContext(), 1),
                            result.operands) ||
      parseParenOperandList(parser, result, operandInfo, operandLocs) ||
      parser.parseArrowTypeList(result.types) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  SmallVector<Type> bodyTypes;
  for (Value operand : llvm::drop_begin(result.operands))
    bodyTypes.push_back(operand.getType());

  // Parse the then region.
  Region *thenRegion = result.addRegion();
  if (parser.parseRegion(*thenRegion, operandInfo, bodyTypes, operandLocs,
                         /*enableNameShadowing=*/true))
    return failure();

  Region *elseRegion = result.addRegion();
  if (parser.parseKeyword("else") ||
      parser.parseRegion(*elseRegion, operandInfo, bodyTypes, operandLocs,
                         /*enableNameShadowing=*/true))
    return failure();

  return success();
}

void IfOp::print(OpAsmPrinter &p) {
  p << " " << cond() << " (";
  llvm::interleaveComma(inputs(), p,
                        [&](Value arg) { p << arg << ": " << arg.getType(); });
  p << ") ";
  p.printArrowTypeList(getResultTypes());
  p << ' ';
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());

  // Reuse the argument names provided to the op for the bbarg names within
  // the region.
  p.shadowRegionArgs(thenRegion(), inputs());
  p.printRegion(thenRegion(), /*printEntryBlockArgs=*/false);
  p << " else ";
  p.shadowRegionArgs(elseRegion(), inputs());
  p.printRegion(elseRegion(), /*printEntryBlockArgs=*/false);
}

LogicalResult IfOp::verify() {
  // Verify that the operands match the bb arguments.  The ODS verifier already
  // checked the first argument to be present and i1.
  auto *thenBlock = &thenRegion().front();
  if (inputs().size() != thenBlock->getNumArguments())
    return emitOpError("incorrect number of arguments to 'then' block");
  auto *elseBlock = &elseRegion().front();
  if (inputs().size() != elseBlock->getNumArguments())
    return emitOpError("incorrect number of arguments to 'else' block");

  for (unsigned i = 0, e = getNumOperands() - 1; i != e; ++i)
    if (inputs()[i].getType() != thenBlock->getArgument(i).getType() ||
        inputs()[i].getType() != elseBlock->getArgument(i).getType())
      return emitOpError("operand/argument type mismatch");

  auto thenOutput = cast<BEF::OutputOp>(thenBlock->getTerminator());
  if (failed(thenOutput.checkArgumentTypes(getResultTypes())))
    return failure();

  auto elseOutput = cast<BEF::OutputOp>(elseBlock->getTerminator());
  return elseOutput.checkArgumentTypes(getResultTypes());
}

//===----------------------------------------------------------------------===//
// RepeatI32Op
//===----------------------------------------------------------------------===//

ParseResult RepeatI32Op::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType> operandInfo;
  SmallVector<Location> operandLocs;
  llvm::SMLoc operandLoc;
  OpAsmParser::OperandType countOperand;
  if (parser.parseOperand(countOperand) ||
      parser.getCurrentLocation(&operandLoc) ||
      parser.resolveOperand(countOperand,
                            IntegerType::get(result.getContext(), 32),
                            result.operands) ||
      parseParenOperandList(parser, result, operandInfo, operandLocs) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // The result types are the same as the operands (ignoring count).
  for (Value operand : llvm::drop_begin(result.operands))
    result.types.push_back(operand.getType());

  // Parse the body region.
  Region *body = result.addRegion();
  return parser.parseRegion(*body, operandInfo, result.types, operandLocs,
                            /*enableNameShadowing=*/true);
}

void RepeatI32Op::print(OpAsmPrinter &p) {
  p << ' ' << tripCount() << " (";
  llvm::interleaveComma(inputs(), p,
                        [&](Value arg) { p << arg << ": " << arg.getType(); });
  p << ") ";
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());

  // Reuse the argument names provided to the op for the bbarg names within
  // the region.
  p.shadowRegionArgs(region(), inputs());
  p.printRegion(region(), /*printEntryBlockArgs=*/false);
}

LogicalResult RepeatI32Op::verify() {
  // Verify that the operand and result types match.
  if (getNumResults() != inputs().size())
    return emitOpError("incorrect number of operands");

  for (unsigned i = 0, e = getNumResults(); i != e; ++i)
    if (inputs()[i].getType() != getResult(i).getType())
      return emitOpError("operand/result type mismatch");

  auto output = cast<BEF::OutputOp>(getBody()->getTerminator());
  return output.checkArgumentTypes(getResultTypes());
}

//===----------------------------------------------------------------------===//
// BenchmarkOp
//===----------------------------------------------------------------------===//

ParseResult BenchmarkOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType> operandInfo;
  SmallVector<Location> operandLocs;
  OpAsmParser::OperandType chainOperand;
  if (parser.parseOperand(chainOperand) ||
      parser.resolveOperand(chainOperand,
                            parser.getBuilder().getType<ChainType>(),
                            result.operands) ||
      parseParenOperandList(parser, result, operandInfo, operandLocs))
    return failure();

  FloatAttr convergenceThreshold;
  if (parser.parseKeyword("convergenceThreshold") || parser.parseEqual() ||
      parser.parseAttribute(convergenceThreshold, "convergenceThreshold",
                            result.attributes))
    return failure();

  SmallVector<Type> operandTypes;
  // Drop the threshold operand.
  for (auto v : llvm::drop_begin(result.operands))
    operandTypes.push_back(v.getType());

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, operandInfo, operandTypes, operandLocs,
                         /*enableNameShadowing=*/true))
    return failure();

  // Infer the return types from the terminator's operands.
  Operation *bodyTerminator = body->front().getTerminator();
  result.types.assign(bodyTerminator->operand_type_begin(),
                      bodyTerminator->operand_type_end());

  return success();
}

void BenchmarkOp::print(OpAsmPrinter &p) {
  p << ' ' << chain() << " (";
  llvm::interleaveComma(operands(), p,
                        [&](Value arg) { p << arg << ": " << arg.getType(); });
  p << ") convergenceThreshold = ";
  p.printAttribute(convergenceThresholdAttr());

  // Reuse the argument names provided to the op for the bbarg names within
  // the region.
  p.shadowRegionArgs(codeUnderTest(), operands());
  p.printRegion(codeUnderTest(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "GraphRT/GRTDialect/GRT.cpp.inc"
