//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"

namespace M::MO {

#define GEN_PASS_DEF_ADDFALLBACKSHAPEFUNCTIONS
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Fallback guard creation (to ensure all MO ops have one or error)
//===----------------------------------------------------------------------===//

// C++ workaround so `false` depends on `T`
template <class>
[[maybe_unused]] constexpr bool FALSE = false;

// Eventually all MO types will implement this function and we will instantiate
// all of them using the macros in the MLIR generated headers. Once we flip the
// switch on that adding a MO op without a fallback or a placeholder entry
// designating that it has no fallback will produce a compile time error.
template <typename T>
struct Fallback {
  static_assert(
      FALSE<T>,
      "All MO operations must have registered fallback shape "
      "functions but no function found for this MO op. (see template error)");
};

#define HAS_FALLBACK(MOOp)                                                     \
  template <>                                                                  \
  struct Fallback<MOOp> {                                                      \
    void build(GraphOp &graph, OpBuilder &builder, MOOp op);                   \
  };

#define NEEDS_NO_FALLBACK(MOOp)                                                \
  template <>                                                                  \
  struct Fallback<MOOp> {};

HAS_FALLBACK(MO::WhereOp)

// Eventually all ops must be accounted for here. They either must have a
// fallback or an explict vouch that they need no fallback.
NEEDS_NO_FALLBACK(MO::Static::BroadcastToOp)

// Creates the fallback for this op if it has parameters still needing decls.
#define CALL_FALLBACK(MOOp)                                                    \
  if (auto asOp = dyn_cast<MOOp>(op.getOperation())) {                         \
    Fallback<MOOp>{}.build(graph, builder, asOp);                              \
    return mlir::WalkResult::advance();                                        \
  }

//===----------------------------------------------------------------------===//
// Fallback creation pass.
//===----------------------------------------------------------------------===//

struct AddFallbackShapeFunctions
    : public impl::AddFallbackShapeFunctionsBase<AddFallbackShapeFunctions> {

  void runOnOperation() override {
    MO::GraphOp graph = getOperation();
    OpBuilder builder{graph.getContext()};

    graph.walk([&](ParamDeclarationInterface op) {
      // Shape function was already materialized or never needed.
      if (op.getOutputParamDecls().empty())
        return mlir::WalkResult::advance();

      // Call the fallbacks.
      builder.setInsertionPoint(op);
      CALL_FALLBACK(MO::WhereOp)

      return mlir::WalkResult::advance();
    });
  }
};

//===----------------------------------------------------------------------===//
// Fallback helpers
//===----------------------------------------------------------------------===//

// Each fallback needs to produce a parameter for every dimension in the output
// even if those dimensions are static. This simplifies the lowering as without
// doing this we would need the lowering to know how many should be produced and
// to which dimension they should be refering to.
KGEN::ParamDeclArrayAttr getNewParameterMapping(GraphOp &graph,
                                                TensorType outTy) {
  // New attrs. These are either the exact same as the old attribute or a new
  // attribute
  SmallVector<KGEN::ParamDeclAttr> params;

  // Parameters assumed to be in order. Add the index of each symbolic
  // dimension.
  for (auto dim : outTy.getTypedShapeAttr().getValues()) {
    if (auto asSymbolic = dyn_cast<KGEN::ParamDeclRefAttr>(dim)) {
      // Since this pass happens before shape propagation, for any op that
      // declares output parameters, the parameters in its output shape must
      // come from those output parameters.
      params.push_back(
          KGEN::ParamDeclAttr::get(asSymbolic.getName(), asSymbolic.getType()));
    } else {
      // Add a new placeholder parameter. F for fallback.
      params.push_back(graph.getNewDimParam("F"));
    }
  }
  return KGEN::ParamDeclArrayAttr::get(graph.getContext(), params);
}

//===----------------------------------------------------------------------===//
// Actual fallback implementations
//===----------------------------------------------------------------------===//

void Fallback<MO::WhereOp>::build(GraphOp &graph, OpBuilder &builder,
                                  MO::WhereOp op) {
  KGEN::ParamDeclArrayAttr params =
      getNewParameterMapping(graph, op.getResult().getType());

  // Create the shape function.
  builder.create<MO::InvokeShapeFuncOp>(
      op.getLoc(), ValueRange{op.getInput()}, "where_shape", params,
      /*kgenAttrs*/ DictionaryAttr{},
      /*Data dependent input indices*/ builder.getI32ArrayAttr({0}));

  // Clear the parameters from the old op.
  op.setOutputParamDecls({});
}

} // namespace

} // namespace M::MO

#undef HAS_FALLBACK
#undef NEEDS_NO_FALLBACK
#undef CALL_FALLBACK
