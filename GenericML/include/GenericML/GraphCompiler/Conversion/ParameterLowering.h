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
  /// Register what SSA value the parameter attribute is lowered to. Prefer to
  /// use the higher level lowering methods where placeholders are concerned.
  void emplaceLowered(Attribute attr, Value value) {
    loweredParams.try_emplace(attr, value);
  }

  /// Returns the SSA value that the parameter attribute is lowered to, or a
  /// null value if it hasn't been lowered yet.
  Value lookupLowered(Attribute attr) { return loweredParams.lookup(attr); }

  /// Lower the given graph argument using the provided builder and hook. The
  /// hook (using the given builder and argument) is responsible for updating
  /// the value mapping.
  template <typename LowerArgFunc>
  ErrorOr<bool> lowerArg(OpBuilder &b, BlockArgument &arg, LowerArgFunc func) {
    return func(*this, b, arg);
  }

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
      newValue = b.create<KGEN::ParamConstantOp>(loc, declRef).getResult();
    } else if (auto opAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr)) {
      // Expressions can be nested, so we lower the operands recursively.
      SmallVector<Value> newOperands;
      for (TypedAttr operand : opAttr.getOperands()) {
        UNWRAP_ERROR(newOperand, lowerExpression(b, loc, operand, func));
        newOperands.emplace_back(newOperand);
      }

      // Then we lower the operator using the hook.
      UNWRAP_ERROR_OR_SET(newValue, func(*this, b, loc, opAttr, newOperands));
    } else {
      UNWRAP_ERROR_OR_SET(newValue, func(*this, b, loc, attr, {}));
    }

    // We update the map and return the lowered value.
    emplaceLowered(attr, newValue);
    return newValue;
  }

  /// Lower a parameter declaration, resolving the placeholder if necessary. A
  /// declaration can only be lowered once, and an error is returned otherwise.
  ErrorOrSuccess lowerParamDecl(KGEN::ParamDeclAttr declAttr, Value value) {
    auto declRef = KGEN::ParamDeclRefAttr::get(declAttr);
    if (Value lowered = lookupLowered(declRef)) {
      auto placeholder = lowered.getDefiningOp<KGEN::ParamConstantOp>();
      // If the parameter is already declared, we return an error.
      if (!placeholder)
        return Error("Duplicate declaration of " + declRef.getName().strref());

      // Otherwise, we remove it from the map and replace the SSA values.
      lowered.replaceAllUsesWith(value);
      placeholder->erase();
      loweredParams.erase(declRef);
    }

    // Finally, update the map with the new lowering.
    emplaceLowered(declRef, value);
    return success();
  }

private:
  /// Tracks what parameters attributes (e.g. expressions) are lowered to.
  DenseMap<Attribute, Value> loweredParams;
};

} // namespace M

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_PARAMETERLOWERING_H
