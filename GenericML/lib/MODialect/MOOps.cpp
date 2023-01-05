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
  buildWithEntryBlock(builder, state, name,
                      builder.getFunctionType(inputTypes, resultTypes), {},
                      inputTypes);
}

/// Parses an MO Graph.
///
/// operation ::= `graph` function-signature function-attributes? function-body
///
ParseResult GraphOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

/// Print the GraphOp.
void GraphOp::print(OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

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

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// MO_ConcatOp support
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::verify() {
  if (!getAxis().getType().getShape().empty())
    return emitError("concat axis must be a scalar tensor");

  size_t numInputs = getInputs().size();
  if (numInputs < 2)
    return emitError("concat must have at least 2 input tensors");

  llvm::SmallVector<size_t> ranks;
  for (size_t j = 0; j < numInputs; ++j) {
    TensorType t = getInputTensor(j).getType();
    if (t.hasRank())
      ranks.emplace_back(t.getRank());
  }
  if (!ranks.empty()) {
    if (!llvm::all_equal(ranks))
      return emitError("concat inputs with known rank must match");

    if (TensorType resType = getResult().getType(); resType.hasRank())
      if (resType.getRank() != ranks[0])
        return emitError("concat output with known rank must match inputs");
  }

  return LogicalResult::success();
}

//===----------------------------------------------------------------------===//
// Windowed ops
//===----------------------------------------------------------------------===//

// TODO: move this method to an interface.
LogicalResult MaxPoolOp::verifyWindowParamShapeAndInputMatch(
    TypedValue<::M::MO::TensorType> windowParam, StringRef name) {
  TensorType inputTy = getInput().getType();
  if (TensorType windowParamTy = windowParam.getType();
      windowParamTy.hasStaticShape())
    if (static_cast<size_t>(windowParamTy.getShape()[0]) != inputTy.getRank())
      return emitError(name) << " with static size must match rank of input";
  return LogicalResult::success();
}

LogicalResult MaxPoolOp::verify() {
  TensorType inputTy = getInput().getType();

  if (inputTy.hasRank()) {
    if (TensorType resultTy = getResult().getType(); resultTy.hasRank()) {
      if (inputTy.getRank() != resultTy.getRank())
        return emitError("input and result ranks must match if static");
    }

    if (failed(verifyWindowParamShapeAndInputMatch(getFilterShape(),
                                                   "filter_shape")))
      return failure();
    if (failed(verifyWindowParamShapeAndInputMatch(getStrides(), "strides")))
      return failure();
    if (failed(
            verifyWindowParamShapeAndInputMatch(getDilations(), "dilations")))
      return failure();
  }

  return LogicalResult::success();
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
