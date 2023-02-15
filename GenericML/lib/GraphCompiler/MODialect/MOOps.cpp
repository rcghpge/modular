//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

using namespace M;
using namespace MO;
using namespace Shape;

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
  FunctionType funcType = builder.getFunctionType(inputTypes, resultTypes);
  auto sig = KGEN::SignatureType::get(funcType);
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
  if (graph.getResultParams().size() != this->getParameters().size())
    return emitError(
        "output bindings and result parameters must have the same size");

  // Build a mapping from result param references to their binding attribute,
  // example:
  //  { (D1, 42), (Sh0, [D0, 10]) }
  DenseMap<KGEN::ParamDeclRefAttr, TypedAttr> ResParamRefToBindingAttr;
  for (auto [resParamDecl, bindingParam] :
       llvm::zip(graph.getResultParams(), this->getParameters())) {
    ResParamRefToBindingAttr.insert(
        {KGEN::ParamDeclRefAttr::get(resParamDecl), bindingParam});
  }

  // Substitute param references inside graph result types using the map, which
  // gives us the expected output operands type.
  SmallVector<Type> expectedOutputOperandTypes;
  for (Type ty : graph.getResultTypes()) {
    if (auto tensorTy = dyn_cast<TensorType>(ty)) {
      TypedAttr newShapeAttr = tensorTy.getShapeAttr().replace(
          [&ResParamRefToBindingAttr](KGEN::ParamDeclRefAttr paramRef) {
            TypedAttr resolvedAttr = ResParamRefToBindingAttr.lookup(paramRef);
            return resolvedAttr ? resolvedAttr : paramRef;
          });
      expectedOutputOperandTypes.emplace_back(tensorTy.cloneWith(newShapeAttr));
    } else {
      expectedOutputOperandTypes.emplace_back(ty);
    }
  }

  return checkResultTypes(getOperation(), expectedOutputOperandTypes);
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

  SmallVector<TypedValue<TensorType>> tensors;
  for (Value input : getInputs())
    tensors.emplace_back(cast<TypedValue<TensorType>>(input));

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
  // TODO: improve this verifier
  TensorType aType = getTensorA().getType();

  // Check we have at least 3d tensors
  if (aType.hasRank() && aType.getRank() < 3) {
    return emitError(
               "batch_matmul input tensors must be at least rank 3, got: ")
           << aType.getRank();
  }

  return LogicalResult::success();
}

//===----------------------------------------------------------------------===//
// MO_RebindOp support
//===----------------------------------------------------------------------===//

static ParseResult parseParamDecls(OpAsmParser &p,
                                   KGEN::ParamDeclArrayAttr &paramDecls) {
  if (succeeded(p.parseOptionalLess()) && failed(p.parseOptionalGreater())) {
    if (p.parseLParen() || p.parseRParen() || p.parseArrow() ||
        KGEN::parseParamDecls(p, paramDecls) || p.parseGreater())
      return failure();
  } else {
    paramDecls = KGEN::ParamDeclArrayAttr::get(p.getContext(), {});
  }
  return success();
}

static void printParamDecls(OpAsmPrinter &p, Operation *,
                            KGEN::ParamDeclArrayAttr paramDecls) {
  if (!paramDecls.empty()) {
    p << "<() -> ";
    printParamDecls(p, paramDecls);
    p << '>';
  }
}

OpFoldResult RebindOp::fold(FoldAdaptor adaptor) {
  // We fold if the result and operand types are the same.
  if (getInput().getType() == getType())
    return getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// MO_TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  // The permutation has known shape iff the argument and result are statically
  // ranked.
  TensorType permTy = getPerm().getType();
  TensorType inputTy = getInput().getType();
  TensorType resTy = getResult().getType();
  if (!permTy.hasStaticShape()) {
    if (inputTy.hasRank() || resTy.hasRank()) {
      return emitError("argument and result must be dynamically ranked if perm "
                       "has unknown shape");
    }
    return success();
  }

  if (!inputTy.hasRank() || !resTy.hasRank()) {
    return emitError("argument and result must be statically ranked if perm "
                     "has known shape");
  }

  if (inputTy.getRank() != resTy.getRank())
    return emitError("input and result ranks do not match");

  // ODS constraints ensure that perm has rank = 1.
  if ((int64_t)resTy.getRank() != permTy.getIntDim(0))
    return emitError("input and result ranks do not match perm size");
  return success();
}

//===----------------------------------------------------------------------===//
// Shape ops
//===----------------------------------------------------------------------===//

void Shape::GetDimOp::build(OpBuilder &builder, OperationState &state,
                            Value input, int64_t dim) {
  build(builder, state, input, builder.getIndexAttr(dim));
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

void MODialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/GraphCompiler/MODialect/MO.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MODialect/MO.cpp.inc"
