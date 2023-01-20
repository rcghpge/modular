//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MODialect/MOOps.h"
#include "GenericML/MODialect/MOInterfaces.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// MO_GraphOp support
//===----------------------------------------------------------------------===//

void GraphOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                    TypeRange inputTypes, TypeRange resultTypes) {
  // Below we basically inline and specialize `buildWithEntryBlock`, which
  // doesn't work because we use `signature` and not `function_type`.
  state.addAttribute(M::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));

  // Add signature attribute
  KGEN::ParamDeclArrayAttr inputParams;
  KGEN::TypeArrayAttr resultParamTypes;
  auto funcType = builder.getFunctionType(inputTypes, resultTypes);
  KGEN::ConventionsAttr conventions;
  auto sig = KGEN::SignatureType::get(inputParams, resultParamTypes, funcType,
                                      conventions);
  state.addAttribute(GraphOp::getSignatureAttrName(state.name),
                     mlir::TypeAttr::get(sig));

  // Add the function body.
  Region *bodyRegion = state.addRegion();
  Block &body = bodyRegion->emplaceBlock();
  for (Type input : inputTypes)
    body.addArgument(input, state.location);
}

/// Parses an MO Graph.
///
/// operation ::= `graph` function-signature function-attributes? function-body
///
ParseResult GraphOp::parse(OpAsmParser &parser, OperationState &result) {
  return KGEN::parseGeneratorOrFunc(parser, result,
                                    KGEN::GeneratorOrFuncKind::func);
}

/// Print the GraphOp.
void GraphOp::print(OpAsmPrinter &p) { KGEN::printGeneratorOrFunc(p, *this); }

//===----------------------------------------------------------------------===//
// OutputOp support
//===----------------------------------------------------------------------===//

LogicalResult OutputOp::verify() {
  // ODS checks the parent is a GraphOp.
  auto graph = cast<GraphOp>(getOperation()->getParentOp());

  return checkResultTypes(getOperation(), graph.getResultTypes());
}

//===----------------------------------------------------------------------===//
// MO_ConstantOp support
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  assert(operands.empty() && "constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// MO_ConcatOp support
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::verify() {
  size_t numInputs = getInputs().size();
  if (numInputs < 2)
    return emitError("concat must have at least 2 input tensors");

  SmallVector<TypedValue<TensorType>> tensors{getInputs()};
  tensors.emplace_back(getResult());
  if (failed(impl::verifyMatchingRanks(tensors)))
    return emitError("input and output ranks must match if static");

  return LogicalResult::success();
}

//===----------------------------------------------------------------------===//
// Windowed ops
//===----------------------------------------------------------------------===//

LogicalResult MaxPoolOp::verify() {
  auto op = cast<SlidingWindow>(this->getOperation());
  if (failed(op.verifyWindowParamShape(getFilterShape(), "filter_shape")))
    return failure();

  return op.verifyWindowParamSizes(
      {getFilterShape(), getDilations(), getStrides()});
}

LogicalResult ConvOp::verify() {
  // The sliding window interface already checks that the input and result have
  // matching ranks, so we can be more specific with the error message here
  if (failed(impl::verifyMatchingRanks({getInput(), getResult(), getFilter()})))
    return emitError("filter rank must match input/output ranks if static");

  auto op = cast<SlidingWindow>(this->getOperation());
  return op.verifyWindowParamSizes({getDilations(), getStrides()});
}

//===----------------------------------------------------------------------===//
// Matmul definitions.
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  TensorType aType = getTensorA().getType();

  // Check we have 2D * 2D tensors as input.
  if (aType.getRank() != 2) {
    return emitError("matmul input tensors must be rank 2, got: ")
           << aType.getRank();
  }

  return LogicalResult::success();
}

LogicalResult BatchMatmulOp::verify() {
  TensorType aType = getTensorA().getType();

  // Check we have at least 3d tensors
  if (aType.getRank() < 3) {
    return emitError(
               "batch_matmul input tensors must be at least rank 3, got: ")
           << aType.getRank();
  }

  return LogicalResult::success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

void MODialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/MODialect/MO.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "GenericML/MODialect/MO.cpp.inc"
