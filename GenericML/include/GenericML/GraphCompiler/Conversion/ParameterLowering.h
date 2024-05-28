//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_PARAMETERLOWERING_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_PARAMETERLOWERING_H

#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/AssertStream.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

namespace M {

class ParameterLoweringState {
public:
  using PlaceholderOp = MOSH::ParamToValueOp;

  /// Register what SSA value the parameter attribute is lowered to. Prefer to
  /// use the higher level lowering methods where placeholders are concerned.
  void emplaceLowered(Region *parentRegion, Attribute attr, Value value) {
    loweredParams[parentRegion].try_emplace(attr, value);
  }

  /// Returns the SSA value that the parameter attribute is lowered to, or a
  /// null value if it hasn't been lowered yet.
  Value lookupLowered(Region *parentRegion, Attribute attr) {
    if (loweredParams[parentRegion].contains(attr))
      return loweredParams[parentRegion].lookup(attr);
    if (auto parent = parentRegion->getParentRegion(); parent)
      return lookupLowered(parent, attr);
    return {};
  }

  void eraseLowered(Region *parentRegion, Attribute attr) {
    if (loweredParams[parentRegion].contains(attr))
      return (void)loweredParams[parentRegion].erase(attr);
    if (auto parent = parentRegion->getParentRegion(); parent)
      return eraseLowered(parent, attr);
    assert(false);
  }

  /// Returns a lowered value for the parameter expression attribute. Parameter
  /// references not seen before are lowered to a placeholder, parameter
  /// operators are lowered recursively, and all other attributes are delegated
  /// to the provided hook (using the given builder and location). The hook
  /// should not update the value mapping.
  template <typename LowerExprFunc>
  ErrorOr<Value> lowerExpression(OpBuilder &b, Location loc,
                                 Region *parentRegion, TypedAttr attr,
                                 LowerExprFunc func) {
    if (Value lowered = lookupLowered(parentRegion, attr))
      return lowered;

    Value newValue;
    if (auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(attr)) {
      // A decl ref we haven't seen before gets a placeholder.
      newValue = b.create<PlaceholderOp>(loc, declRef).getResult();
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
      auto newValueOr =
          func(*this, b, loc, parentRegion, newOpAttr, newOperands);
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

  /// Lower a parameter declaration, resolving the placeholder if necessary. If
  /// a placeholder is resolved, its SSA value is returned so that callers can
  /// replace their uses if necessary. Otherwise, a null Value signifies a
  /// successful lowering without placeholder resolution.
  Value lowerParamDecl(Region *parentRegion, KGEN::ParamDeclAttr declAttr,
                       Value value) {
    auto declRef = KGEN::ParamDeclRefAttr::get(declAttr);
    if (Value lowered = lookupLowered(parentRegion, declRef)) {
      auto placeholder = lowered.getDefiningOp<PlaceholderOp>();
      // The parameter verification should ensure we don't have duplicate
      // declarations, so we just assert
      GENERICML_ASSERT(placeholder) << "duplicate declaration of parameter";

      // Otherwise, we update the mapping and return the placeholder.
      eraseLowered(parentRegion, declRef);
      emplaceLowered(parentRegion, declRef, value);
      return placeholder;
    }

    emplaceLowered(parentRegion, declRef, value);
    return {};
  }

private:
  /// Helper to lower an array of expression operands.
  template <typename LowerExprFunc>
  ErrorOr<SmallVector<Value>>
  lowerOperands(OpBuilder &b, Location loc, Region *parentRegion,
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

  /// Tracks what parameters attributes (e.g. expressions) are lowered to.
  ///  - Dimension parameter attributes are lowered to values of type `index`.
  ///  - Shape parameter attributes are lowered to values of type
  ///    `mgp.tensor_spec<...xunk>`.
  DenseMap<Region *, DenseMap<Attribute, Value>> loweredParams;
};

} // namespace M

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_PARAMETERLOWERING_H
