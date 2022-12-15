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

void MODialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/MODialect/MO.cpp.inc"
      >();
}

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// Matmul definitions.
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  auto matmul = cast<MatmulOp>(getOperation());

  // Get the two input tensors. ODS guarantees these are tensor types.
  auto aType = cast<TensorType>(matmul.getInputA().getType());
  auto bType = cast<TensorType>(matmul.getInputB().getType());
  auto outType = cast<TensorType>(matmul.getResult().getType());

  // Check we have 2D * 2D tensors as input.
  if (aType.getRank() != bType.getRank()) {
    return matmul.emitError("Matmul input tensors are not the same rank ")
           << aType.getRank() << " vs " << bType.getRank();
  }

  if (aType.getRank() != 2) {
    return matmul.emitError("Matmul input tensors must be rank 2, got: ")
           << aType.getRank();
  }

  // The tensor dims.
  llvm::ArrayRef<int64_t> aShape = aType.getShape();
  llvm::ArrayRef<int64_t> bShape = bType.getShape();
  llvm::ArrayRef<int64_t> outShape = outType.getShape();

  if (aShape[1] != bShape[0]) {
    return matmul.emitError("Matmul input tensors are expected to be (M, K) * "
                            "(K, N) but input shapes are ")
           << "(" << aShape << ") * (" << bShape << ")";
  }

  if (outShape[0] != aShape[0] || outShape[1] != bShape[1]) {
    return matmul.emitError(
               "Matmul of (M, K) * (K, N) does not produce shape (M, N): ")
           << "(" << aShape << ") * (" << bShape << ") -> (" << outShape << ")";
  }

  return LogicalResult::success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/MODialect/MO.cpp.inc"
