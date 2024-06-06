//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Implementations of `Staticization` interface for various `mo` ops.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/ShapeFunctions.h"
#include "KGEN/KGENDialect/KGENAttrs.h"

namespace M::MO {

ErrorOr<TypedAttr> BroadcastShapeOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                                  ParamExprBuilder &b) {

  // Requires inputs to be lists of param exprs.
  auto lhsShapeAttr = dyn_cast<ShapeAttr>(inputs[0]);
  auto rhsShapeAttr = dyn_cast<ShapeAttr>(inputs[1]);
  if (lhsShapeAttr && rhsShapeAttr) {

    // Ensure rank(lhsShapeAttr) >= rank(rhsShapeAttr)
    if (lhsShapeAttr.getRank() < rhsShapeAttr.getRank())
      std::swap(lhsShapeAttr, rhsShapeAttr);

    // Build up output value.
    SmallVector<TypedAttr> resDims;
    size_t rankDiff = lhsShapeAttr.getRank() - rhsShapeAttr.getRank();
    for (size_t axis = 0; axis < lhsShapeAttr.getRank(); axis++) {
      if (axis < rankDiff) {
        // Extra leading dimensions get broadcasted trivially.
        resDims.push_back(lhsShapeAttr[axis]);

      } else {
        // Otherwise this dimension is broadcasted by the following rule:
        //   def bcast(D1, D2):
        //       assert `D1 == D2 or D1 == 1 or D2 == 1`
        //       if D1 == 1:
        //           return D2
        //       else:
        //           return D1
        Var dim1 = b.createVar(lhsShapeAttr[axis]);
        Var dim2 = b.createVar(rhsShapeAttr[axis - rankDiff]);
        // Insert assertion
        b.createAssert(dim1 == 1 || dim2 == 1 || dim1 == dim2,
                       "[broadcast_shape] dimensions are not broadcastable");

        Var newResDim = b.createCond(dim1 == 1, dim2, dim1);
        resDims.push_back(newResDim.getAttr());
      }
    }
    auto res = MOSH::ShapeAttr::get(resDims, lhsShapeAttr.getType());
    return res;
  }

  // Can't staticize.
  return TypedAttr{};
}

ErrorOr<TypedAttr> RebindOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                          ParamExprBuilder &b) {
  return inputs[0];
}

} // namespace M::MO
