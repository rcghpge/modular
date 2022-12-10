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
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/MODialect/MO.cpp.inc"
