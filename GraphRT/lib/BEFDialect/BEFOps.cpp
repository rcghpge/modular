//===- BEFOps.cpp ---------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the BEF dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFDialect/BEFOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace M;
using namespace BEF;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder &builder, OperationState &state, StringAttr name,
                   ArrayRef<Type> inputTypes, ArrayRef<Type> resultTypes,
                   ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(::SymbolTable::getSymbolAttrName(), name);
  auto fnType = builder.getFunctionType(inputTypes, resultTypes);
  state.addAttribute(getTypeAttrName(), TypeAttr::get(fnType));
  state.attributes.append(attrs.begin(), attrs.end());
  Region *bodyRegion = state.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add input arguments to the body block.
  for (auto elt : inputTypes)
    body->addArgument(elt, state.location);
}

// Parses a BEF Function.
//
// operation ::= `func` function-signature function-attributes? function-body
//
ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  SmallVector<OpAsmParser::OperandType> entryArgs;
  SmallVector<NamedAttrList> argAttrs;
  SmallVector<NamedAttrList> resultAttrs;
  SmallVector<Type> argTypes;
  SmallVector<Type> resultTypes;
  SmallVector<Location> argLocations;
  bool isVariadic;

  if (parser.parseSymbolName(nameAttr, ::SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      mlir::function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/false, entryArgs, argTypes, argAttrs,
          argLocations, isVariadic, resultTypes, resultAttrs))
    return failure();

  auto type = FunctionType::get(parser.getContext(), argTypes, resultTypes);
  result.addAttribute(mlir::FunctionOpInterface::getTypeAttrName(),
                      TypeAttr::get(type));

  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();
  mlir::function_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, argAttrs, resultAttrs);

  auto *body = result.addRegion();
  return parser.parseRegion(*body, entryArgs,
                            entryArgs.empty() ? ArrayRef<Type>() : argTypes);
}

// Print the FuncOp. Collects argument and result types and passes them to
// helper functions. Drops "void" result since it cannot be parsed back.
void FuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getName());

  FunctionType fnType = getType();
  mlir::function_interface_impl::printFunctionSignature(
      p, *this, fnType.getInputs(), /*isVarArg=*/false, fnType.getResults());
  mlir::function_interface_impl::printFunctionAttributes(
      p, *this, fnType.getNumInputs(), fnType.getNumResults(), {});

  // Print the body if this is not an external function.
  Region &body = getBody();
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

/// Hook for FunctionOpInterface, called after verifying that the 'type'
/// attribute is present and checks if it holds a function type. Ensures
/// getType, getNumFuncArguments, and getNumFuncResults can be called
/// safely.
LogicalResult FuncOp::verifyType() {
  auto llvmType = getTypeAttr().getValue().dyn_cast_or_null<FunctionType>();
  if (!llvmType)
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");

  return success();
}

LogicalResult FuncOp::verify() {
  auto output = cast<OutputOp>(getBodyBlock()->getTerminator());
  return output.checkArgumentTypes(getResultTypes());
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

/// Containers verify that the operands of this OutputOp match the specified set
/// of types.
mlir::LogicalResult OutputOp::checkArgumentTypes(TypeRange types) {
  // Verify our result types match up with the enclosing func result type.
  if (getNumOperands() != types.size())
    return emitOpError("expected ")
           << types.size() << " operands for enclosing op";

  for (size_t i = 0, e = getNumOperands(); i != e; ++i) {
    if (getOperand(i).getType() != types[i])
      return emitOpError("operand #")
             << i << " has type " << getOperand(i).getType()
             << " but should be " << types[i];
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "GraphRT/BEFDialect/BEF.cpp.inc"
