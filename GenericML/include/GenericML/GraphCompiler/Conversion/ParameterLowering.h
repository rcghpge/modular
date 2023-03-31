//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_PARAMETERLOWERING_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_PARAMETERLOWERING_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

namespace M {

class ParameterLoweringState {
public:
  using PlaceholderOp = KGEN::ParamConstantOp;

  /// Register what SSA value the parameter attribute is lowered to. Prefer to
  /// use the higher level lowering methods where placeholders are concerned.
  void emplaceLowered(Attribute attr, Value value) {
    loweredParams.try_emplace(attr, value);
  }

  /// Returns the SSA value that the parameter attribute is lowered to, or a
  /// null value if it hasn't been lowered yet.
  Value lookupLowered(Attribute attr) { return loweredParams.lookup(attr); }

  /// Returns a lowered value for the parameter expression attribute. Parameter
  /// references not seen before are lowered to a placeholder, parameter
  /// operators are lowered recursively, and all other attributes are delegated
  /// to the provided hook (using the given builder and location). The hook
  /// should not update the value mapping.
  template <typename LowerExprFunc>
  ErrorOr<Value> lowerExpression(OpBuilder &b, Location loc, TypedAttr attr,
                                 LowerExprFunc func) {
    if (Value lowered = lookupLowered(attr))
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
      UNWRAP_ERROR(newOperands, lowerOperands(b, loc, operands, func));
      UNWRAP_ERROR_OR_SET(newValue,
                          func(*this, b, loc, newOpAttr, newOperands));
    } else {
      UNWRAP_ERROR_OR_SET(newValue, func(*this, b, loc, attr, {}));
    }

    // We update the map and return the lowered value.
    emplaceLowered(attr, newValue);
    return newValue;
  }

  /// Lower a parameter declaration, resolving the placeholder if necessary. A
  /// declaration can only be lowered once, and an error is returned otherwise.
  /// If a placeholder is resolved, its SSA value is returned so that callers
  /// can replace their uses if necessary. Otherwise, a null Value signifies a
  /// successful lowering without placeholder resolution.
  ErrorOr<PlaceholderOp> lowerParamDecl(KGEN::ParamDeclAttr declAttr,
                                        Value value) {
    auto declRef = KGEN::ParamDeclRefAttr::get(declAttr);
    if (Value lowered = lookupLowered(declRef)) {
      auto placeholder = lowered.getDefiningOp<PlaceholderOp>();
      // If the parameter is already declared, we return an error.
      if (!placeholder)
        return Error("Duplicate declaration of " + declRef.getName().strref());

      // Otherwise, we update the mapping and return the placeholder.
      loweredParams.erase(declRef);
      emplaceLowered(declRef, value);
      return placeholder;
    }

    emplaceLowered(declRef, value);
    return PlaceholderOp();
  }

private:
  /// Helper to lower an array of expression operands.
  template <typename LowerExprFunc>
  ErrorOr<SmallVector<Value>> lowerOperands(OpBuilder &b, Location loc,
                                            ArrayRef<TypedAttr> operands,
                                            LowerExprFunc func) {
    SmallVector<Value> newOperands;
    for (TypedAttr operand : operands) {
      // Expressions can be nested, so we lower the operands recursively.
      UNWRAP_ERROR(newOperand, lowerExpression(b, loc, operand, func));
      newOperands.emplace_back(newOperand);
    }
    return newOperands;
  }

  /// Tracks what parameters attributes (e.g. expressions) are lowered to.
  DenseMap<Attribute, Value> loweredParams;
};

} // namespace M

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_PARAMETERLOWERING_H
