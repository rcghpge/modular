//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/ParameterLowering.h"
#include "Support/AssertStream.h"

#include "mlir/IR/BuiltinAttributes.h"

using namespace M;
using namespace mlir;

ErrorOr<Value> M::ParameterLoweringState::lowerExpression(OpBuilder &b,
                                                          Location loc,
                                                          Region *parentRegion,
                                                          TypedAttr attr,
                                                          LowerExprFunc func) {
  if (Value lowered = lookupLowered(parentRegion, attr))
    return lowered;

  Value newValue;
  if (auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(attr)) {
    // A decl ref we haven't seen before gets a placeholder.
    newValue = PlaceholderOp::create(b, loc, declRef).getResult();
  } else if (auto opAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr)) {
    ArrayRef<TypedAttr> operands = opAttr.getOperands();
    TypedAttr newOpAttr = attr;
    if (opAttr.getOpcode() == KGEN::POC::Apply) {
      // Apply is special, since its first argument is an mlir op.
      newOpAttr = operands.front();
      operands = operands.drop_front();
    }
    auto newOperandsOr = lowerOperands(b, loc, parentRegion, operands, func);
    if (newOperandsOr.isError())
      return newOperandsOr.takeError();
    auto newOperands = std::move(*newOperandsOr);
    auto newValueOr = func(*this, b, loc, parentRegion, newOpAttr, newOperands);
    if (newValueOr.isError())
      return newValueOr.takeError();
    newValue = std::move(*newValueOr);
  } else {
    auto newValueOr = func(*this, b, loc, parentRegion, attr, ValueRange{});
    if (newValueOr.isError())
      return newValueOr.takeError();
    newValue = std::move(*newValueOr);
  }

  // We update the map and return the lowered value.
  emplaceLowered(parentRegion, attr, newValue);
  return newValue;
}

Value M::ParameterLoweringState::lowerParamDecl(Region *parentRegion,
                                                KGEN::ParamDeclAttr declAttr,
                                                Value value) {
  auto declRef = KGEN::ParamDeclRefAttr::get(declAttr);
  if (Value lowered = lookupLowered(parentRegion, declRef)) {
    auto placeholder = lowered.getDefiningOp<PlaceholderOp>();
    // The parameter verification should ensure we don't have duplicate
    // declarations, so we just assert
    ASSERT_STREAM(placeholder, << "duplicate declaration of parameter");

    // Otherwise, we update the mapping and return the placeholder.
    eraseLowered(parentRegion, declRef);
    emplaceLowered(parentRegion, declRef, value);
    return placeholder;
  }

  emplaceLowered(parentRegion, declRef, value);
  return {};
}

ErrorOr<SmallVector<Value>> M::ParameterLoweringState::lowerOperands(
    OpBuilder &b, Location loc, Region *parentRegion,
    ArrayRef<TypedAttr> operands, LowerExprFunc func) {
  SmallVector<Value> newOperands;
  for (TypedAttr operand : operands) {
    // Expressions can be nested, so we lower the operands recursively.
    auto newOperandOr = lowerExpression(b, loc, parentRegion, operand, func);
    if (newOperandOr.isError())
      return newOperandOr.takeError();
    newOperands.emplace_back(std::move(*newOperandOr));
  }
  return newOperands;
}
