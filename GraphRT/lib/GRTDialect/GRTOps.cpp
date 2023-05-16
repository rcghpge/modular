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
// ChainMergeOp
//===----------------------------------------------------------------------===//

LogicalResult ChainMergeOp::canonicalize(ChainMergeOp op,
                                         PatternRewriter &rewriter) {
  // Using a set allows us to unique the operands on the fly.
  SmallPtrSet<Value, 4> uniqueOperands;

  // We collect the operands, and if an input comes from another ChainMergeOp,
  // we try to fold it into this one.
  for (Value operand : op->getOperands()) {
    if (auto producer = operand.getDefiningOp<ChainMergeOp>()) {
      auto producerArgs = producer->getOperands();
      uniqueOperands.insert(producerArgs.begin(), producerArgs.end());
    } else {
      uniqueOperands.insert(operand);
    }
  }

  // If there isn't any input, this becomes a new chain that's immediately ready
  if (uniqueOperands.empty()) {
    rewriter.replaceOpWithNewOp<ChainCreateOp>(op);
    return success();
  }

  // If there is a single input, we just fold this op.
  if (uniqueOperands.size() == 1) {
    op.replaceAllUsesWith(*uniqueOperands.begin());
    return success();
  }

  // If we have multiple operands, we sort them to have a canonicalize list.
  SmallVector<Value> newOperands(uniqueOperands.begin(), uniqueOperands.end());
  llvm::sort(newOperands, [](Value a, Value b) -> bool {
    return a.getAsOpaquePointer() < b.getAsOpaquePointer();
  });

  // If there is no change, we can return.
  if (newOperands == SmallVector<Value>(op->getOperands()))
    return failure();

  // Finally, we can update the op with the new (canonicalized) operands.
  op->setOperands(newOperands);
  return success();
}

ParseResult ChainMergeOp::parse(OpAsmParser &parser, OperationState &result) {
  auto chainType = MEF::ChainType::get(parser.getContext());

  SmallVector<OpAsmParser::Argument> operandInfo;
  // Parse the signature.
  if (failed(parseParenOperandList(parser, result, operandInfo,
                                   /*optDefaultType=*/chainType)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result.attributes))) {
    return failure();
  }

  // Result type is deterministic
  result.types.push_back(chainType);

  return success();
}

void ChainMergeOp::print(OpAsmPrinter &p) {
  auto printArg = [&](Value arg) {
    p << arg;
    if (!isa<MEF::ChainType>(arg.getType()))
      p << ": " << arg.getType();
  };

  p << " (";
  llvm::interleaveComma(getOperands(), p, printArg);
  p << ") ";
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
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
  // Parse the 'cond' operand and signature.
  SmallVector<OpAsmParser::Argument> operandInfo;
  OpAsmParser::UnresolvedOperand condOperand;
  if (parser.parseOperand(condOperand) ||
      parser.resolveOperand(condOperand,
                            IntegerType::get(result.getContext(), 1),
                            result.operands) ||
      parseParenOperandList(parser, result, operandInfo) ||
      parser.parseArrowTypeList(result.types) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the 'then' region, allowing shadowing of the op argument names.
  Region *thenRegion = result.addRegion();
  if (parser.parseRegion(*thenRegion, operandInfo,
                         /*enableNameShadowing=*/true))
    return failure();

  // Parse the 'else' region, again with shadowing.
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
  // within the regions.
  p.shadowRegionArgs(getThenRegion(), getInputs());
  p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false);
  p << " else ";
  p.shadowRegionArgs(getElseRegion(), getInputs());
  p.printRegion(getElseRegion(), /*printEntryBlockArgs=*/false);
}

LogicalResult IfOp::verify() {
  // The 'then' block should accept arguments matching the grt.if signature.
  auto *thenBlock = &getThenRegion().front();
  if (failed(
          checkArgumentTypes(*this, "then", thenBlock, getInputs().getTypes())))
    return failure();

  // The 'else' block should accept arguments matching the grt.if signature.
  auto *elseBlock = &getElseRegion().front();
  if (failed(
          checkArgumentTypes(*this, "else", elseBlock, getInputs().getTypes())))
    return failure();

  // The 'then' block should return results matching the grt.if results.
  auto thenOutput = cast<MEF::OutputOp>(thenBlock->getTerminator());
  if (failed(checkOperandTypes(thenOutput, getResultTypes())))
    return failure();

  // The 'else' block should return results matching the grt.if results.
  auto elseOutput = cast<MEF::OutputOp>(elseBlock->getTerminator());
  if (failed(checkOperandTypes(elseOutput, getResultTypes())))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

ParseResult WhileOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> operandInfo;
  // Parse the signature.
  if (parseParenOperandList(parser, result, operandInfo) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // The op's return types match those of the signature arguments.
  result.types.reserve(result.operands.size());
  for (auto operand : result.operands)
    result.types.push_back(operand.getType());

  // Parse the 'cond' region, allowing shadowing of the op argument names.
  Region *condRegion = result.addRegion();
  if (parser.parseRegion(*condRegion, operandInfo,
                         /*enableNameShadowing=*/true))
    return failure();

  // Parse the 'body' region, again allowing shadowing.
  Region *bodyRegion = result.addRegion();
  if (parser.parseKeyword("do") ||
      parser.parseRegion(*bodyRegion, operandInfo,
                         /*enableNameShadowing=*/true))
    return failure();

  return success();
}

void WhileOp::print(OpAsmPrinter &p) {
  p << " (";
  llvm::interleaveComma(getInputs(), p,
                        [&](Value arg) { p << arg << ": " << arg.getType(); });
  p << ") ";
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());

  // Reuse the argument names provided to the op for the basic block arg names
  // within the regions.
  p.shadowRegionArgs(getCondRegion(), getInputs());
  p.printRegion(getCondRegion(), /*printEntryBlockArgs=*/false);
  p << " do ";
  p.shadowRegionArgs(getBodyRegion(), getInputs());
  p.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false);
}

LogicalResult WhileOp::verify() {
  // The grt.while results must match the signature types.
  if (failed(checkMatchingTypes(*this, "result", getResultTypes(),
                                getInputs().getTypes())))
    return failure();

  // The 'cond' block should accept arguments matching the grt.while signature.
  auto *condBlock = &getCondRegion().front();
  if (failed(
          checkArgumentTypes(*this, "cond", condBlock, getInputs().getTypes())))
    return failure();

  // The 'body' block should accept arguments matching the grt.while signature.
  auto *bodyBlock = &getBodyRegion().front();
  if (failed(
          checkArgumentTypes(*this, "body", bodyBlock, getInputs().getTypes())))
    return failure();

  // The 'cond' block should return a result for each grt.while result,
  // followed by an i1.
  auto condOutput = cast<MEF::OutputOp>(condBlock->getTerminator());
  SmallVector<Type> condResultTypes;
  condResultTypes.reserve(getResultTypes().size() + 1);
  llvm::append_range(condResultTypes, getResultTypes());
  condResultTypes.emplace_back(IntegerType::get(getContext(), 1));
  if (failed(checkOperandTypes(condOutput, condResultTypes)))
    return failure();

  // The 'body' block should return a result for each grt.while result.
  auto bodyOutput = cast<MEF::OutputOp>(bodyBlock->getTerminator());
  if (failed(checkOperandTypes(bodyOutput, getResultTypes())))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// RepeatI32Op
//===----------------------------------------------------------------------===//

ParseResult RepeatI32Op::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the 'tripCount' operand and signature.
  SmallVector<OpAsmParser::Argument> operandInfo;
  OpAsmParser::UnresolvedOperand countOperand;
  if (parser.parseOperand(countOperand) ||
      parser.resolveOperand(
          countOperand,
          IntegerType::get(result.getContext(), 32, IntegerType::Signed),
          result.operands) ||
      parseParenOperandList(parser, result, operandInfo) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // The result types are the same as the operands (ignoring 'tripCount').
  result.types.reserve(result.operands.size() - 1);
  for (auto operand : llvm::drop_begin(result.operands))
    result.types.push_back(operand.getType());

  // Parse the 'body' region, allowing shadowing of the op argument names.
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
  p.shadowRegionArgs(getBodyRegion(), getInputs());
  p.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false);
}

LogicalResult RepeatI32Op::verify() {
  // The grt.repeat.i32 results should match the signature types.
  if (failed(checkMatchingTypes(*this, "result", getResultTypes(),
                                getInputs().getTypes())))
    return failure();

  // The 'body' block should accept arguments matching the grt.repeat.i32
  // signature.
  auto *bodyBlock = &getBodyRegion().front();
  if (failed(
          checkArgumentTypes(*this, "body", bodyBlock, getInputs().getTypes())))
    return failure();

  // The 'body' block should return a result for each grt.repeat.i32 result.
  auto output = cast<MEF::OutputOp>(bodyBlock->getTerminator());
  if (failed(checkOperandTypes(output, getResultTypes())))
    return failure();

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
