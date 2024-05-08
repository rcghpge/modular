//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Analysis/MODialect/ParametricTopologyAnalysis.h"
#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/SetOperations.h"
#include <type_traits>

using namespace M;
using namespace MO;

namespace M::MO {

#define GEN_PASS_DEF_PROPAGATESHAPES

#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

} // namespace M::MO

namespace {

bool isParamOperatorPure(KGEN::ParamOperatorAttr paramOperator) {
  if (paramOperator.getOpcode() == KGEN::POC::Apply)
    return false;
  return true;
}

bool isParamDeclOpPure(KGEN::ParamDeclareOp paramDeclOp) {
  if (auto paramOperatorAttr =
          dyn_cast<KGEN::ParamOperatorAttr>(paramDeclOp.getValue()))
    return isParamOperatorPure(paramOperatorAttr);
  return true;
}

// Assume `useDefGraph` is constructed for `graphOp`, remove all the dead
// `kgen.param.declare` ops (basically aggressive DCE applied to
// `kgen.param.declare`s).
void removeDeadParamDeclares(GraphOp graphOp,
                             const KGEN::ParameterUseDefGraph &useDefGraph) {
  // Invariants:
  // Any op that has been on `worklist`
  // 1. is in `processed`
  // 2. must be alive
  DenseSet<Operation *> processed;
  SmallVector<Operation *> worklist;
  auto pushOntoWorklist = [&](Operation *op) {
    if (auto [_, inserted] = processed.insert(op); inserted)
      worklist.push_back(op);
  };

  // Initialize worklist
  graphOp.walk([&](Operation *op) {
    // Conservatively collect seeding ops which must be alive -- the only
    // exceptions are `kgen.param.declare` that are pure (they _might_ not
    // need to be alive), everything else must be alive.
    auto paramDeclOp = dyn_cast<KGEN::ParamDeclareOp>(op);
    if (paramDeclOp && isParamDeclOpPure(paramDeclOp))
      return;
    else
      pushOntoWorklist(op);
  });

  // Process the worklist and find more ops that must be alive.
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    // `op` is considered alive, so all of the parameter it uses must be
    // alive, so the definition ops of those parameters must be alive.
    for (KGEN::ParamDeclRefAttr usedParamDeclRef :
         useDefGraph.opUses.lookup(op)) {
      const KGEN::ParamDefinition &usedParamDef =
          useDefGraph.defs.at(usedParamDeclRef.getName());
      pushOntoWorklist(usedParamDef.defOp);
    }
  }

  // Remove all the `kgen.param.declarle` ops that do not have to be alive.
  graphOp.walk([&](KGEN::ParamDeclareOp paramDeclOp) {
    if (!processed.contains(paramDeclOp))
      paramDeclOp.erase();
  });
}

/// Handle output op and graph result parameters. This is special in that
/// 1. We need to manually check for the binding parameters, because they
/// never materialize kgen.param.decl (the result params are declared in the
/// graph signature).
/// 2. We need to be careful about mapping graph result parameters.
///
/// NOTE this assumes we've run shape propagation for all the ops within a
/// `mo.graph`.
void handleOutputBindings(GraphOp graphOp,
                          KGEN::ParameterEvaluator &evaluator) {
  // First we collect the declared parameter names so we can check for them.
  SmallPtrSet<StringAttr, 4> resDeclNames;
  for (KGEN::ParamDeclAttr decl : graphOp.getResultParams())
    resDeclNames.insert(decl.getName());
  SmallPtrSet<StringAttr, 4> argDeclNames;
  for (KGEN::ParamDeclAttr decl : graphOp.getInputParams())
    argDeclNames.insert(decl.getName());

  // Check if `attr` can appear as a dimension attribute in graph output.
  auto isValidOutputDim = [&](Attribute dimAttr) {
    // Don't allow any reference to parameters declared inside the graph.
    if (auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(dimAttr))
      return argDeclNames.contains(declRef.getName());

    if (isa<IntegerAttr>(dimAttr))
      return true;

    return false;
  };

  // Check if `attr` can appear as a dimension attribute in graph output.
  auto isValidOutputShape = [&](Attribute attr) {
    // Don't allow any reference to parameters declared inside the graph.
    if (auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(attr))
      return argDeclNames.contains(declRef.getName());

    if (auto shapeAttr = dyn_cast<MOSH::ShapeAttr>(attr))
      return llvm::all_of(shapeAttr.getValues(), isValidOutputDim);

    return false;
  };

  auto outputOp = graphOp.getOutputOp();
  for (auto [bindingParam, resultParam] :
       llvm::zip(outputOp.getParameters(), graphOp.getResultParams())) {
    // NOTE the binding parameter has already been rebound.
    // Output parameter could be either a shape or dimension, we check to make
    // sure the resolved parameter is valid.
    if ((isa<ShapeType>(bindingParam.getType()) &&
         isValidOutputShape(bindingParam)) ||
        isValidOutputDim(bindingParam)) {
      evaluator.setParameterValue(resultParam.getName(), bindingParam);
    }
    // Otherwise `KGEN::ParameterEvaluator` will map the result parameter to
    // itself by default, meaning it won't change in the signature.
  }
}

/// Update the signature's result declarations and output bindings to account
/// for any output parameter simplification (e.g., maybe an old output
/// parameter `D1` is deduced to be the same as an input parameter `D0`).
void updateGraphSignature(GraphOp graphOp) {
  SmallPtrSet<StringAttr, 4> resParamNames;
  for (Type resTy : graphOp.getResultTypes())
    if (auto resTensorTy = dyn_cast<MO::TensorType>(resTy))
      llvm::set_union(resParamNames, resTensorTy.getUsedParamNames());

  auto outputOp = graphOp.getOutputOp();
  SmallVector<KGEN::ParamDeclAttr> newResultParams{};
  SmallVector<TypedAttr> newBindingParams{};
  for (auto [bindingParam, resParamDecl] :
       llvm::zip(outputOp.getParameters(), graphOp.getResultParams())) {
    if (resParamNames.contains(resParamDecl.getName())) {
      newResultParams.emplace_back(resParamDecl);
      newBindingParams.emplace_back(bindingParam);
    }
  }

  outputOp.setParameters(newBindingParams);

  // Update the result parameters.
  auto newResultParamsAttr =
      KGEN::ParamDeclArrayAttr::get(graphOp->getContext(), newResultParams);
  NamedAttrList attrs = graphOp->getAttrDictionary();
  attrs.set(graphOp.getResultParamsAttrName(), newResultParamsAttr);

  SignatureType signature = KGEN::SignatureType::remapToSignature(
      graphOp.getInputParams(), newResultParamsAttr, graphOp.getFunctionType());
  attrs.set(graphOp.getSignatureAttrName(), TypeAttr::get(signature));
  graphOp->setAttrs(attrs.getDictionary(graphOp->getContext()));
}

// Forward declaration for `Var`.
class Var;

/// This class helps one create parameter expression in a less elaborate way.
class ParamExprBuilder {
public:
  /// Create a `ParamExprBuilder` with the given metadata.
  /// - `graphOp` is used for new parameter generation.
  /// - `loc` is used for newly created ops like `kgen.param.declare X = <...>`.
  /// - `createdOps` will be appended with newly created ops.
  ParamExprBuilder(GraphOp graphOp, Location loc,
                   SmallVector<Operation *> *createdOps)
      : builder(graphOp.getContext()), graphOp(graphOp), loc(loc),
        createdOps(createdOps) {}

  /// Create a Var that wraps around given `attr`, and allows one to build more
  /// parameter expression with the returned Var.
  Var createVar(TypedAttr attr);

  /// Create an op of the following form
  ///   kgen.param.declare declPrefix... = <poc(operands...)>
  ///
  /// Return a parameter decl ref to the newly created parameter
  /// `declPrefix...`.
  KGEN::ParamDeclRefAttr createDeclToKgenExpr(KGEN::POC poc,
                                              ArrayRef<TypedAttr> operands,
                                              StringRef declPrefix = "expr");

  /// Create an op of the following form
  ///   kgen.param.declare declPrefix... = <attr>
  ///
  /// Return a parameter decl ref to the newly created parameter
  /// `declPrefix...`.
  KGEN::ParamDeclRefAttr createDeclToKgenAttr(TypedAttr attr,
                                              StringRef declPrefix = "expr");

  /// Create an assert on the boolean parameter expression in `condVar`, with
  /// given error message.
  void createAssert(const Var &condVar, StringRef errorMsg);

  /// Create an op of the following form
  ///   kgen.param.declare ... = <cond(condAttr, thenAttr, elseAttr)>
  ///
  /// Where `(condAttr, thenAttr, elseAttr)` are parameter expressions within
  /// `(convV, thenV, elseV)` respectively.
  Var createCond(const Var &condV, const Var &thenV, const Var &elseV);

  /// Assume given a variadic number of Var instances, create a new `ShapeAttr`
  /// consisted of the TypedAttr within those Var, in the given order.
  template <class... V, typename = std::enable_if_t<(true && ... &&
                                                     std::is_same_v<V, Var>)>>
  ShapeAttr getShapeAttr(const V &...dim);

  /// Create a new `ShapeAttr` consisted of the `dimAttrs` in the given order.
  ShapeAttr getShapeAttr(ArrayRef<TypedAttr> dimAttrs);

private:
  OpBuilder builder;
  GraphOp graphOp;
  Location loc;
  SmallVector<Operation *> *createdOps;
};

/// This class wraps around a parameter attribute and allows one to build new
/// parameter expression easily via operator overloading.
///
/// Example:
///   ParamExprBuilder b(...);
///   Var x = b.createVar(xAttr);
///   Var y = (x * x) / 2;
///   TypedAttr yAttr = y.getAttr();
class Var {
private:
  // Either constant IntegerAttr or DeclRef, to minimize expression nesting and
  // make debugging possible (otherwise expressions will fold by construction).
  TypedAttr attr;
  ParamExprBuilder *builder;

  // Tightly control how one can construct a `Var`.
  Var(TypedAttr attr, ParamExprBuilder *builder)
      : attr(attr), builder(builder) {}

  // Allow access to the controlled constructor of Var.
  friend class ParamExprBuilder;

public:
  Var(const Var &other) {
    this->attr = other.attr;
    this->builder = other.builder;
  }
  Var &operator=(const Var &other) = default;
  Var &operator=(Var &&other) = default;

  // Allows one to write `v1 + v2`.
  Var operator+(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Add, attr, other.attr),
        "var_add");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 - v2`.
  Var operator-(const Var &other) const { return *this + (other * -1); }

  // Allows one to write `v1 * v2`.
  Var operator*(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::get(KGEN::POC::MulNuw, attr, other.attr),
        "var_mul");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 / v2`.
  Var operator/(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Div, attr, other.attr),
        "var_div");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 != v2`.
  Var operator!=(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::getNE(attr, other.attr), "var_ne");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 == v2`.
  Var operator==(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::get(KGEN::POC::EQ, {attr, other.attr}),
        "var_eq");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 < v2`.
  Var operator<(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::get(KGEN::POC::LT, attr, other.attr),
        "var_lt");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 <= v2`.
  Var operator<=(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::get(KGEN::POC::LE, attr, other.attr),
        "var_le");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 > v2`.
  Var operator>(const Var &other) const { return other < *this; }

  // Allows one to write `v1 >= v2`.
  Var operator>=(const Var &other) const { return other <= *this; }

  // Allows one to write `v1 || v2`.
  Var operator||(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Or, attr, other.attr),
        "var_or");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 && v2`.
  Var operator&&(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::get(KGEN::POC::And, attr, other.attr),
        "var_and");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 + 42`.
  Var operator+(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this + otherVar;
  }

  // Allows one to write `v1 - 42`.
  Var operator-(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this - otherVar;
  }

  // Allows one to write `v1 * 42`.
  Var operator*(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this * otherVar;
  }

  // Allows one to write `v1 / 42`.
  Var operator/(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this / otherVar;
  }

  // Allows one to write `v1 != 42`.
  Var operator!=(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this != otherVar;
  }

  Var operator==(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this == otherVar;
  }

  // Allows one to write `v1 < 42`.
  Var operator<(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this < otherVar;
  }

  // Allows one to write `v1 <= 42`.
  Var operator<=(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this <= otherVar;
  }

  // Allows one to write `v1 > 42`.
  Var operator>(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this > otherVar;
  }

  // Allows one to write `v1 >= 42`.
  Var operator>=(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this >= otherVar;
  }

  TypedAttr getAttr() const { return attr; }
};

Var ParamExprBuilder::createVar(TypedAttr attr) {
  auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(attr);
  if (!declRef)
    declRef = this->createDeclToKgenAttr(attr, "var");
  return Var(declRef, this);
}

KGEN::ParamDeclRefAttr ParamExprBuilder::createDeclToKgenExpr(
    KGEN::POC poc, ArrayRef<TypedAttr> operands, StringRef declPrefix) {
  auto exprAttr = KGEN::ParamOperatorAttr::get(poc, operands);
  return createDeclToKgenAttr(exprAttr, declPrefix);
};

KGEN::ParamDeclRefAttr
ParamExprBuilder::createDeclToKgenAttr(TypedAttr attr, StringRef declPrefix) {
  Type resTy = attr.getType();
  KGEN::ParamDeclAttr declAttr = graphOp.getNewParamOfType(declPrefix, resTy);
  auto declOp = builder.create<KGEN::ParamDeclareOp>(loc, declAttr, attr);
  auto declRefAttr = KGEN::ParamDeclRefAttr::get(declOp.getParamDecl());
  createdOps->push_back(declOp);
  return declRefAttr;
};

Var ParamExprBuilder::createCond(const Var &condV, const Var &thenV,
                                 const Var &elseV) {
  auto exprAttr = KGEN::ParamOperatorAttr::get(
      KGEN::POC::Cond, {condV.getAttr(), thenV.getAttr(), elseV.getAttr()});
  return createVar(createDeclToKgenAttr(exprAttr, "cond"));
};

template <class... V, typename>
ShapeAttr ParamExprBuilder::getShapeAttr(const V &...dim) {
  return getShapeAttr({dim.getAttr()...});
}

ShapeAttr ParamExprBuilder::getShapeAttr(ArrayRef<TypedAttr> dimAttrs) {
  return ShapeAttr::get(dimAttrs, ShapeType::get(graphOp.getContext()));
}

void ParamExprBuilder::createAssert(const Var &condVar, StringRef errorMsg) {
  TypedAttr condAttr = condVar.getAttr();
  MLIRContext *ctx = condAttr.getContext();
  Type chainType = MOSH::ChainType::get(ctx);

  // Build a new operand list which starts with MLIROpAttr.
  auto signature = SignatureType::get(ctx, {condAttr.getType()}, chainType);
  NamedAttribute msgAttr(StringAttr::get(ctx, "message"),
                         StringAttr::get(errorMsg, KGEN::StringType::get(ctx)));
  auto dictAttr = DictionaryAttr::get(ctx, {msgAttr});
  auto opNameAttr = StringAttr::get(ctx, MOSH::AssertOp::getOperationName());
  SmallVector<TypedAttr> applyOperands{
      KGEN::MLIROpAttr::get(opNameAttr, dictAttr, signature), condAttr};

  (void)createDeclToKgenExpr(KGEN::POC::Apply, applyOperands, "assert");
};

/// A class that helps `PropagateShapesPass` handle ops that could be
/// decomposed when the shape parameters they use have been simplified
/// already. For instance, in order to simplify `mosh.broadcast`, we often need
/// its two input shapes to be simplified first (e.g., `[D0, 1]` instead of
/// `Sh0`). However, the simplification happens _during_ `PropagateShapesPass`,
/// so we use this `OpDecomposer` to decompose ops on the fly, and letting
/// `PropagateShapesPass` to further simplify the decomposed ops.
class OpDecomposer {
  // State that stays invariant across calls.
  OpBuilder builder;

  // State that might change across calls.
  GraphOp graphOp;
  Location loc;
  SmallVector<Operation *> decomposedOps;

public:
  /// Create an OpDecomposer with given MLIRContext.
  OpDecomposer(MLIRContext *ctx, Location loc) : builder(ctx), loc(loc) {}

  /// Try decomposing the given Operation if possible. Return either
  /// - a failure as fatal failure (e.g., some shape assertion failed)
  /// - a list of decomposed ops (ordered by SSA and parametric dependencies) if
  ///   decomposition was possible
  /// - else return an empty vector.
  /// The input operation won't be modified.
  FailureOr<SmallVector<Operation *>> tryDecomposeOp(Operation *op) {
    // Reset transient state.
    graphOp = op->getParentOfType<GraphOp>();
    loc = op->getLoc();
    decomposedOps.clear();

    if (auto declOp = dyn_cast<KGEN::ParamDeclareOp>(op)) {
      tryDecomposeParamDeclOp(declOp);

    } else if (auto invokeShapeFuncOp = dyn_cast<InvokeShapeFuncOp>(op)) {
      if (failed(tryDecomposeInvokeShapeFuncOp(invokeShapeFuncOp)))
        return failure();
    }

    return decomposedOps;
  }

private:
  // Some parameter declaration can be simplified as long as there are extra
  // assertions emitted to legalize the simplification. However, not all such
  // assertions can proven to be trivial statically. In order to still benefit
  // from the expression simplification and avoid having to provide a runtime
  // primitive for the original expression, we try to decompose these
  // expressions into smaller primitives we already support, and try to simplify
  // those smaller primitives as much as possible.
  void tryDecomposeParamDeclOp(KGEN::ParamDeclareOp declOp) {
    // Only try to decompose expression that applies an mlir op.
    auto operatorAttr = dyn_cast<KGEN::ParamOperatorAttr>(declOp.getValue());
    if (!operatorAttr || operatorAttr.getOpcode() != KGEN::POC::Apply)
      return;

    ArrayRef<TypedAttr> operands = operatorAttr.getOperands();
    auto mlirOpAttr = dyn_cast<KGEN::MLIROpAttr>(operands.front());
    if (!mlirOpAttr)
      return;

    // Check to see if the mlir op expression can be decomposed.
    if (mlirOpAttr.getName() == MOSH::BroadcastOp::getOperationName()) {
      auto shapeAttr1 = dyn_cast<MOSH::ShapeAttr>(operands[1]);
      auto shapeAttr2 = dyn_cast<MOSH::ShapeAttr>(operands[2]);
      if (shapeAttr1 && shapeAttr2) {
        // Ensure rank(shapeAttr1) >= rank(shapeAttr2)
        if (shapeAttr1.getRank() < shapeAttr2.getRank())
          std::swap(shapeAttr1, shapeAttr2);

        // Accumulate ops created into `decomposedOps`
        ParamExprBuilder b(graphOp, loc, &decomposedOps);
        SmallVector<TypedAttr> resDims;
        size_t rankDiff = shapeAttr1.getRank() - shapeAttr2.getRank();
        for (size_t axis = 0; axis < shapeAttr1.getRank(); axis++) {
          if (axis < rankDiff) {
            resDims.push_back(shapeAttr1[axis]);
          } else {
            Var dim1 = b.createVar(shapeAttr1[axis]);
            Var dim2 = b.createVar(shapeAttr2[axis - rankDiff]);
            // Insert assertion
            b.createAssert(
                dim1 == 1 || dim2 == 1 || dim1 == dim2,
                "[broadcast_shape] dimensions are not broadcastable");

            // def bcast(D1, D2):
            //     assert `D1 == D2 or D1 == 1 or D2 == 1`
            //     if D1 == 1:
            //         return D2
            //     else:
            //         return D1
            Var newResDim = b.createCond(dim1 == 1, dim2, dim1);
            resDims.push_back(newResDim.getAttr());
          }
        }
        auto newDeclOp = cast<KGEN::ParamDeclareOp>(declOp->clone());
        newDeclOp.setValueAttr(
            MOSH::ShapeAttr::get(resDims, shapeAttr1.getType()));
        decomposedOps.push_back(newDeclOp);
      }
    }
  }

  // Assume `invokeShapeFuncOp` applies the shape function of `MO::BroadcastTo`,
  // try decomposing the shape function call into `kgen.param.declare`
  // primitives if possible, return null otherwise.
  ShapeAttr tryDecomposeBroadcastToShape(InvokeShapeFuncOp invokeShapeFuncOp) {
    auto inputTensorVal = cast<TensorValue>(invokeShapeFuncOp.getOperand(0));
    auto inputShape =
        dyn_cast<MOSH::ShapeAttr>(inputTensorVal.getType().getShapeAttr());
    auto targetShape = tryGetShapeAttr(invokeShapeFuncOp.getOperand(1));
    if (!inputShape || !targetShape)
      return {};

    int inputRank = inputShape.getRank();
    int outputRank = targetShape.getRank();
    // Illegal broadcast, let runtime handle this error.
    if (inputRank > outputRank)
      return {};

    // Accumulate ops created into `decomposedOps`
    ParamExprBuilder b(graphOp, loc, &decomposedOps);
    for (int i = 0; i < inputRank; i++) {
      int inputAxis = inputRank - i - 1;
      int outputAxis = outputRank - i - 1;
      Var inputDim = b.createVar(inputShape[inputAxis]);
      Var outputDim = b.createVar(targetShape[outputAxis]);

      // Insert assertions
      b.createAssert(
          inputDim == 1 || inputDim == outputDim,
          "[broadcast_to] input dim must be either 1 or equal to "
          "corresponding output dim starting from the rightmost dim");
    }
    return targetShape;
  }

  /// If target shape contains a `-1`, infer it based on the invariant that
  /// `inputShapeAttr` and `targetShapeAttr` represent equal number of elements.
  ///
  /// Example:
  ///   inputShapeAttr = [3, D4]
  ///   targetShapeAttr = [D0, -1, 42]
  ///   returns [D0, RS0, 42]
  /// with the following inserted by `rewriter`
  ///   kgen.param.declare RS0 = <div(mul(3, D4), mul(D0, 42)>
  ShapeAttr resolveReshapeTargetShape(ParamExprBuilder &b,
                                      ShapeAttr inputShapeAttr,
                                      ShapeAttr targetShapeAttr) {
    auto indexType = builder.getType<IndexType>();
    // Find index of `-1` dimension (if any), and collect all other dimensions.
    SmallVector<TypedAttr> targetNonNegOneDims;
    std::optional<size_t> maybeTargetNegOneDimIdx = std::nullopt;
    for (auto [idx, targetDim] : llvm::enumerate(targetShapeAttr.getValues())) {
      auto intDim = dyn_cast<IntegerAttr>(targetDim);
      if (intDim && intDim.getInt() == -1) {
        // Detected multiple negative one, gives up simplifying this reshape
        // function call and let runtime handle the error for now.
        if (maybeTargetNegOneDimIdx.has_value())
          return {};
        maybeTargetNegOneDimIdx = idx;
      } else {
        // TODO: we are using the assumption that all non-integer attribute
        // (e.g., `D0` or `mul(D0, D1)`) will be non-negative. To make this
        // completely safe, we'll need to extend and leverage the range
        // analysis here.
        targetNonNegOneDims.push_back(targetDim);
      }
    }

    // no literal `-1` dimension, nothing to resolve
    if (!maybeTargetNegOneDimIdx.has_value())
      return targetShapeAttr;

    // Infer the actual value of `-1` dimension based on other dimensions
    Var numerator = b.createVar(inputShapeAttr.getSymbolicNumElements());

    // Add a 1 to the shapes to multiply to ensure the denominator is valid
    auto mulDims = targetNonNegOneDims;
    mulDims.emplace_back(IntegerAttr::get(indexType, 1));
    Var denominator =
        b.createVar(KGEN::ParamOperatorAttr::get(KGEN::POC::MulNuw, mulDims));

    // Make sure we check for 0 denominator
    b.createAssert(denominator != 0,
                   "[reshape] cannot infer dimension when a specified output "
                   "dimension is 0");

    Var resolvedDim = numerator / denominator;
    SmallVector<TypedAttr> resolvedDims = targetNonNegOneDims;
    resolvedDims.insert(resolvedDims.begin() + maybeTargetNegOneDimIdx.value(),
                        resolvedDim.getAttr());
    return b.getShapeAttr(resolvedDims);
  }

  // Assume `invokeShapeFuncOp` applies the shape function of `MO::ReshapeOp`,
  // try decomposing the shape function call into `kgen.param.declare`
  // primitives if possible, return null otherwise.
  ShapeAttr tryDecomposeReshapeShape(InvokeShapeFuncOp invokeShapeFuncOp) {
    // Make sure both input and target shapes are ranked symbolic shapes.
    auto inputShape = dyn_cast<ShapeAttr>(
        cast<TensorType>(invokeShapeFuncOp.getOperand(0).getType())
            .getShapeAttr());
    ShapeAttr targetShape = tryGetShapeAttr(invokeShapeFuncOp.getOperand(1));
    if (!inputShape || !targetShape)
      return {};

    // Accumulate ops created into `decomposedOps`
    ParamExprBuilder b(graphOp, loc, &decomposedOps);

    // Make sure we are able to resolve any `-1` (inferred dimension) in the
    // target shape.
    ShapeAttr outputShape =
        resolveReshapeTargetShape(b, inputShape, targetShape);
    if (!outputShape)
      return {};

    // Insert assertions
    Var inputNumeElems = b.createVar(inputShape.getSymbolicNumElements());
    Var outputNumeElems = b.createVar(outputShape.getSymbolicNumElements());
    b.createAssert(inputNumeElems == outputNumeElems,
                   "[reshape] input and output number of elements must match");

    return outputShape;
  }

  // Assume `invokeShapeFuncOp` applies the shape function of `MO::RangeOp`,
  // try decomposing the shape function call into `kgen.param.declare`
  // primitives if possible, return null otherwise.
  ShapeAttr tryDecomposeRangeShape(InvokeShapeFuncOp invokeShapeFuncOp) {
    // Make sure both input and target shapes are ranked symbolic shapes.
    TypedAttr startAttr = tryGetDimAttr(invokeShapeFuncOp.getOperand(0));
    TypedAttr stopAttr = tryGetDimAttr(invokeShapeFuncOp.getOperand(1));
    TypedAttr stepAttr = tryGetDimAttr(invokeShapeFuncOp.getOperand(2));
    if (!startAttr || !stopAttr || !stepAttr)
      return {};

    // Accumulate ops created into `decomposedOps`
    ParamExprBuilder b(graphOp, loc, &decomposedOps);
    Var start = b.createVar(startAttr);
    Var stop = b.createVar(stopAttr);
    Var step = b.createVar(stepAttr);

    // Check step != 0
    b.createAssert(step != 0, "[range] step must be non-zero");

    // Check start/stop are in bounds
    b.createAssert(start <= stop || step < 0,
                   "[range] requires (start <= stop) for positive step");
    b.createAssert(stop <= start || step > 0,
                   "[range] requires (stop <= start) for negative step");

    // Compute output shape by simulating a ceiling-div -- think the length of
    // range(0, 3, 2).
    Var diff = stop - start;
    Var offset = b.createCond(step > 0, step - 1, step + 1);
    Var outDim = (diff + offset) / step;
    ShapeAttr outputShape = b.getShapeAttr(outDim);
    return outputShape;
  }

  // Assume `invokeShapeFuncOp` applies the shape function of
  // `MO::BatchMatmulOp`, try decomposing the shape function call into
  // `kgen.param.declare` primitives if possible, return null otherwise.
  ShapeAttr tryDecomposeBatchMatmulShape(InvokeShapeFuncOp invokeShapeFuncOp) {
    // Make sure both input and target shapes are ranked symbolic shapes.
    auto lhsShape = dyn_cast<ShapeAttr>(
        cast<TensorType>(invokeShapeFuncOp.getOperand(0).getType())
            .getShapeAttr());
    auto rhsShape = dyn_cast<ShapeAttr>(
        cast<TensorType>(invokeShapeFuncOp.getOperand(1).getType())
            .getShapeAttr());
    if (!lhsShape || !rhsShape)
      return {};

    size_t inputRank = lhsShape.getRank();
    if (rhsShape.getRank() != inputRank || inputRank <= 2)
      return {};

    // Accumulate ops created into `decomposedOps`
    ParamExprBuilder b(graphOp, loc, &decomposedOps);
    SmallVector<TypedAttr> resDims;
    // Check batch dimensions are equal and collect them into `resDims`.
    for (size_t batchDim = 0; batchDim < inputRank - 2; batchDim++) {
      Var lhsDim = b.createVar(lhsShape[batchDim]);
      Var rhsDim = b.createVar(rhsShape[batchDim]);
      b.createAssert(lhsDim == rhsDim,
                     "[batch_matmul] inputs batch dimensions must match");
      resDims.push_back(lhsDim.getAttr());
    }

    // Check inner dimensions are equal
    Var lhsInnerDim = b.createVar(lhsShape[inputRank - 1]);
    Var rhsInnerDim = b.createVar(rhsShape[inputRank - 2]);
    b.createAssert(lhsInnerDim == rhsInnerDim,
                   "[batch_matmul] inputs inner dimensions must match");

    // Collect outer dims.
    resDims.push_back(lhsShape[inputRank - 2]);
    resDims.push_back(rhsShape[inputRank - 1]);

    return b.getShapeAttr(resDims);
  }

  // Assume `invokeShapeFuncOp` applies the shape function of
  // `MO::ConcatOp`, try decomposing the shape function call into
  // `kgen.param.declare` primitives if possible, return null otherwise.
  ShapeAttr tryDecomposeConcatShape(InvokeShapeFuncOp invokeShapeFuncOp) {
    // Make sure axis is constant.
    std::optional<int64_t> axisOr =
        tryGetConstantInt(cast<TensorValue>(invokeShapeFuncOp.getOperand(0)));
    if (!axisOr.has_value())
      return {};

    // Collect statically-ranked input shapes.
    SmallVector<ShapeAttr> inputShapes;
    for (Value input : invokeShapeFuncOp.getOperands().drop_front()) {
      auto inputTensorTy = cast<TensorType>(input.getType());
      if (auto inputShape = dyn_cast<ShapeAttr>(inputTensorTy.getShapeAttr())) {
        inputShapes.push_back(inputShape);
      } else {
        return {}; // Give up on any dyn-ranked input shape
      }
    }
    // `MO::ConcatOp` verifier ensures there's at least 1 input, and ranks of
    // inputs all match.
    ShapeAttr firstInputShape = inputShapes.front();
    auto restInputShapes = ArrayRef(inputShapes).drop_front();
    int64_t inputRank = firstInputShape.getRank();

    // Normalize axis and make sure it's in bound.
    int64_t axis = axisOr.value();
    axis = axis < 0 ? axis + inputRank : axis;
    if (axis < 0 || inputRank <= axis)
      return {}; // TODO handle this error at compile time.

    // Construct output shape and make sure input shapes are compatible.
    SmallVector<TypedAttr> resDims(firstInputShape.getValues());
    // Accumulate ops created into `decomposedOps`
    ParamExprBuilder b(graphOp, loc, &decomposedOps);
    Var concatDimSum = b.createVar(resDims[axis]);
    for (ShapeAttr inputShape : restInputShapes) {
      for (auto [idx, firstInputDimAttr, inputDimAttr] : llvm::enumerate(
               firstInputShape.getValues(), inputShape.getValues())) {
        if (static_cast<int64_t>(idx) != axis) {
          Var firstInputDim = b.createVar(firstInputDimAttr);
          Var inputDim = b.createVar(inputDimAttr);
          b.createAssert(
              firstInputDim == inputDim,
              "[concat] input shapes must match except at concat axis");
        }
      }
      Var inputAxisDim = b.createVar(inputShape[axis]);
      concatDimSum = concatDimSum + inputAxisDim;
    }
    resDims[axis] = concatDimSum.getAttr();
    return b.getShapeAttr(resDims);
  }

  // Treat certain fallback shape function call as special intrinsics and
  // effectively symbolicizing them _during_ shape propagation.
  //
  // TODO this should eventually become general symbolicization applied to a
  // single `InvokeShapeFuncOp`, however we can't do it yet since we need to
  // symbolicize in batch in order to minimize the overhead.
  LogicalResult
  tryDecomposeInvokeShapeFuncOp(InvokeShapeFuncOp invokeShapeFuncOp) {
    // Dispatch on certain shape function names.
    StringRef shapeFuncName = invokeShapeFuncOp.getShapeFuncName();
    ShapeAttr outputShape;
    // Why we have these here (instead of just relying on symbolicization):
    //
    // - because they can't always be symbolicized (due to KGEN IR complexity or
    // phasing), and they are often critical to e2e shape inference and other
    // optimizations like MHA pattern patching.
    //
    // - `broadcast_shape` also because it might rely on simplification of
    // `mosh.broadcast`, it's _possible_ that we could move these 2 logic into
    // staticization and symbolicization respectively.
    if (shapeFuncName == "broadcast_to_shape") {
      outputShape = tryDecomposeBroadcastToShape(invokeShapeFuncOp);

    } else if (shapeFuncName == "reshape_shape") {
      outputShape = tryDecomposeReshapeShape(invokeShapeFuncOp);

    } else if (shapeFuncName == "range_shape") {
      outputShape = tryDecomposeRangeShape(invokeShapeFuncOp);

    } else if (shapeFuncName == "batch_matmul_shape") {
      outputShape = tryDecomposeBatchMatmulShape(invokeShapeFuncOp);

    } else if (shapeFuncName == "concat_shape") {
      outputShape = tryDecomposeConcatShape(invokeShapeFuncOp);
    }

    // Create new declaration of the shape function call's output parameter if
    // decomposition succeeded; `outputShape` has the form `[N0, N1, ...]`
    if (outputShape) {
      // outputDecls is always one of
      // - Sh2: !mosh.ape
      // - [D0, D1, ...]
      ArrayRef<KGEN::ParamDeclAttr> outputDecls =
          invokeShapeFuncOp.getOutputParamDecls();

      // NOTE symbolicization currently can't handle dynamic output shape, since
      // it relies on MOGG compiling the shape function down to KGEN IR, which
      // relies on the mojo shape function (and the bottom part of our stack) to
      // support dynamic rank.
      //
      // However, we don't need to support dynamic
      if (outputDecls.size() == 1 &&
          isa<ShapeType>(outputDecls.front().getType())) {
        // Create definition the output shape parameter to output shape, i.e.,
        //   kgen.param.decare D0 = <[N0, N1, ...]>
        auto shapeDeclOp = builder.create<KGEN::ParamDeclareOp>(
            loc, outputDecls.front(), outputShape);
        decomposedOps.push_back(shapeDeclOp);

      } else {
        size_t returnedRank = outputShape.getRank();
        size_t annnotatedRank = outputDecls.size();
        if (returnedRank != annnotatedRank) {
          invokeShapeFuncOp.emitError()
              << "shape function returned shape of rank " << returnedRank
              << " but the type annotation has shape of rank "
              << annnotatedRank;
          return failure();
        }
        // Create definition from each output parameter to its dimension, i.e.,
        //   kgen.param.decare D0 = <N0>
        //   kgen.param.decare D1 = <N1>
        //   ...
        for (auto [outputDimDecl, outputDim] :
             llvm::zip(outputDecls, outputShape.getValues())) {
          auto dimDeclOp = builder.create<KGEN::ParamDeclareOp>(
              loc, outputDimDecl, outputDim);
          decomposedOps.push_back(dimDeclOp);
        }
      }
    }
    return success();
  }
};

} // namespace

struct PropagateShapesPass
    : public MO::impl::PropagateShapesBase<PropagateShapesPass> {
  using PropagateShapesBase::PropagateShapesBase;

  /// For each op inside `graphOp` (including nested blocks within), make sure
  /// any depended ops (SSA or parameter dependencies) appear before the op.
  void topoSortWithParameters(GraphOp graphOp) {
    ParametricTopologyAnalysis analysis(graphOp);
    analysis.topologicallySortAllOpsWithin(&graphOp.getRegion());
  }

  /// Return the depth of a KGEN param operator, where
  /// - <add(D0, 1)> has depth 1
  /// - <add(mul(D0, 2), 1)> has depth 2
  size_t getParamAttrDepth(KGEN::ParamOperatorAttr paramOperator) {
    if (!paramAttrToDepth.contains(paramOperator)) {
      size_t depth = 0;
      for (TypedAttr operand : paramOperator.getOperands()) {
        if (auto nestedParamOperator =
                dyn_cast<KGEN::ParamOperatorAttr>(operand)) {
          depth = std::max(depth, getParamAttrDepth(nestedParamOperator));
        }
      }
      depth += 1;
      paramAttrToDepth[paramOperator] = depth;
    }
    return paramAttrToDepth.at(paramOperator);
  }

  bool canParamOperatorBeNested(KGEN::ParamOperatorAttr paramOperator) {
    if (!isParamOperatorPure(paramOperator))
      return false;

    // A heuristic to avoid having super nested KGEN expression which cannot be
    // folded away. `kMaxDepth` should be picked such that, folding for
    // practical models should only require at most this amount of expression
    // nesting.
    constexpr size_t kMaxDepth = 4;
    auto size = getParamAttrDepth(paramOperator);
    if (size > kMaxDepth)
      return false;

    return true;
  }

  /// Track `attr` as definitely non-negative.
  void markDefinitelyNonNegative(TypedAttr attr) {
    definitelyNonNegativeAttrs.insert(attr);
  }

  /// A conservative predicate that returns true iff `attr` is provable to be
  /// non-positive, and return false we can't claim anything.
  bool isDefinitelyNonPositive(TypedAttr attr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getValue().isNonPositive();

    auto exprAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr);
    if (!exprAttr || exprAttr.getNumOperands() != 2)
      return false;

    // If we have an integer operand, it'll be `rhs` because the order is
    // canonicalized by attribute type.
    TypedAttr lhs = exprAttr.getOperand(0);
    TypedAttr rhs = exprAttr.getOperand(1);

    // mul(non_negative_expr, -1) must be non-positive
    if (llvm::is_contained({KGEN::POC::Mul, KGEN::POC::MulNuw},
                           exprAttr.getOpcode())) {
      auto intAttr = dyn_cast<IntegerAttr>(rhs);
      if (intAttr && intAttr.getInt() == -1 && isDefinitelyNonNegative(lhs))
        return true;
    }

    return false;
  }

  /// A conservative predicate that returns true iff `attr` is provable to be
  /// non-negative, and return false we can't claim anything.
  bool isDefinitelyNonNegative(TypedAttr attr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getValue().isNonNegative();

    if (definitelyNonNegativeAttrs.contains(attr))
      return true;

    auto exprAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr);
    if (!exprAttr)
      return false;

    // If operands are all non-negative, these operations will preserve the
    // property.
    if (llvm::is_contained({KGEN::POC::Max, KGEN::POC::Add},
                           exprAttr.getOpcode())) {
      return llvm::all_of(exprAttr.getOperands(), [&](TypedAttr operand) {
        return isDefinitelyNonNegative(operand);
      });
    }

    return false;
  }

  /// A conservative predicate that returns true iff `attr` is provable to be
  /// negative, and return false we can't claim anything.
  bool isDefinitelyNegative(TypedAttr attr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getValue().isNegative();

    return false;
  }

  /// A conservative predicate that returns true iff `attr` is provable to be
  /// positive, and return false we can't claim anything.
  bool isDefinitelyPositive(TypedAttr attr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getValue().isStrictlyPositive();

    return false;
  }

  /// This function serves as shape inference system's extension to KGEN folder
  /// on `POC::Cond`. It leverages things like range information to enable more
  /// expression folding, and applies it only to `cond(operands)`, assuming
  /// `operands` have been folded as much as possible.
  ///
  /// Return null if we can't apply any special folding.
  TypedAttr tryFoldCond(ArrayRef<TypedAttr> operands) {

    // A helper to check whether `attr` is an IntegerAttr of value `val`.
    auto isIntAttrOfVal = [](TypedAttr attr, int val) {
      auto intAttr = dyn_cast<IntegerAttr>(attr);
      return intAttr && intAttr.getInt() == val;
    };

    // Look for patterns that arise from abs(X), which could materialize into
    //   cond(ge(X, 1), X, mul_nuw(X, -1))
    //
    // Which gets canonicalized into
    //   cond(xor(lt(X, 1), true), X, mul_nuw(X, -1))
    TypedAttr cond = operands[0];
    TypedAttr thenVal = operands[1];
    TypedAttr elseVal = operands[2];

    // Match xor(..., true)
    auto trueAttr = BoolAttr::get(cond.getContext(), true);
    auto xorExpr = dyn_cast<KGEN::ParamOperatorAttr>(cond);
    if (!xorExpr || xorExpr.getOpcode() != KGEN::POC::Xor ||
        xorExpr.getOperand(1) != trueAttr)
      return {};

    // Match lt(X, 1)
    auto ltExpr = dyn_cast<KGEN::ParamOperatorAttr>(xorExpr.getOperand(0));
    if (!ltExpr || ltExpr.getOpcode() != KGEN::POC::LT ||
        ltExpr.getOperand(0) != thenVal ||
        !isIntAttrOfVal(ltExpr.getOperand(1), 1))
      return {};

    // Match mul_nuw(X, -1)
    auto mulExpr = dyn_cast<KGEN::ParamOperatorAttr>(elseVal);
    if (!mulExpr || mulExpr.getOpcode() != KGEN::POC::MulNuw ||
        mulExpr.getOperand(0) != thenVal ||
        !isIntAttrOfVal(mulExpr.getOperand(1), -1))
      return {};

    // Now we know the whole cond is equivalent to `abs(thenVal)`, and
    // `abs(nonNegative) == nonNegative`
    if (isDefinitelyNonNegative(thenVal))
      return thenVal;

    return {};
  }

  /// This function serves as shape inference system's extension to KGEN
  /// folders. It leverages things like range information to enable more
  /// expression folding, and applies it only to `opcode(operands)`, assuming
  /// `operands` have been folded as much as possible.
  ///
  /// Return null if we can't apply any special folding.
  TypedAttr tryFoldExprAttr(KGEN::POC opcode, ArrayRef<TypedAttr> operands) {
    if (opcode == KGEN::POC::Cond)
      return tryFoldCond(operands);

    // Only fold certain binary expression for now.
    if (operands.size() != 2)
      return {};

    // If we have an integer operand, it'll be `rhs` because the order is
    // canonicalized by attribute type.
    TypedAttr lhs = operands[0];
    TypedAttr rhs = operands[1];
    MLIRContext *ctx = lhs.getContext();

    if (opcode == KGEN::POC::Max) {
      // TODO: need a pattern matcher real bad...
      // <max(add(A, B), B)>
      // `lhs` and `rhs` are the operands to the max
      auto maxAddOptimization = [&](TypedAttr maxLHS,
                                    TypedAttr maxRHS) -> TypedAttr {
        // Assume add operand is on the lhs.
        auto lhsOp = dyn_cast<M::KGEN::ParamOperatorAttr>(maxLHS);
        if (!lhsOp || lhsOp.getOpcode() != KGEN::POC::Add)
          return {};

        if (lhsOp.getNumOperands() != 2)
          return {};

        // ensure add operands are parameters
        TypedAttr addLHS = lhsOp.getOperand(0);
        TypedAttr addRHS = lhsOp.getOperand(1);

        // If both params `A` and `B` are non negative, then `add(A, B) >= B`
        // and the max of both sides is `add(A, B)`
        if (!isDefinitelyNonNegative(addLHS) ||
            !isDefinitelyNonNegative(addRHS))
          return {};

        // Trick to handle `<max(add(A, B), B)>` and `<max(add(A, B), A)>`
        if (addLHS == rhs || addRHS == maxRHS)
          return maxLHS;

        return {};
      };
      // Check with both permutations
      if (auto maybeMaxAddOptimization = maxAddOptimization(lhs, rhs))
        return maybeMaxAddOptimization;
      if (auto maybeMaxAddOptimization = maxAddOptimization(rhs, lhs))
        return maybeMaxAddOptimization;

      // max(non_negative_expr, non_positive_expr) = non_negative_expr
      if (isDefinitelyNonNegative(lhs) && isDefinitelyNonPositive(rhs))
        return lhs;

      // max(non_positive_expr, non_negative_expr) = non_negative_expr
      if (isDefinitelyNonPositive(lhs) && isDefinitelyNonNegative(rhs))
        return rhs;

    } else if (opcode == KGEN::POC::Min) {
      // min(non_negative_expr, non_positive_expr) = non_positive_expr
      if (isDefinitelyNonNegative(lhs) && isDefinitelyNonPositive(rhs))
        return rhs;

      // min(non_positive_expr, non_negative_expr) = non_positive_expr
      if (isDefinitelyNonPositive(lhs) && isDefinitelyNonNegative(rhs))
        return lhs;

    } else if (opcode == KGEN::POC::LT) {
      // NOTE we have to put this logic here because for some reason the KGEN
      // folder doesn't consider it valid to fold things like
      // `lt(9223372036854775807, 100)`, maybe due to potential overflow? But we
      // know this is safe as far as shape inference is concerned.
      auto intLhs = dyn_cast<IntegerAttr>(lhs);
      auto intRhs = dyn_cast<IntegerAttr>(rhs);
      if (intLhs && intRhs) {
        bool isLessThan = intLhs.getValue().slt(intRhs.getValue());
        return BoolAttr::get(ctx, isLessThan);
      }

      // lt(non_negative_expr, non_positive_expr) = false
      if (isDefinitelyNonNegative(lhs) && isDefinitelyNonPositive(rhs))
        return BoolAttr::get(ctx, false);

      // lt(non_positive_expr, non_negative_expr) = true
      if (isDefinitelyNonPositive(lhs) && isDefinitelyNonNegative(rhs))
        return BoolAttr::get(ctx, true);

      // Try more things, since we get this pattern from slice: lt(add(X, Y), X)
      //     lhs < rhs
      // <=> lhs - rhs < 0
      auto diff = tryFoldExprAttr(KGEN::ParamOperatorAttr::getSub(lhs, rhs));
      if (isDefinitelyNonNegative(diff))
        return BoolAttr::get(ctx, false);

      if (isDefinitelyNegative(diff))
        return BoolAttr::get(ctx, true);

    } else if (opcode == KGEN::POC::EQ) {
      MLIRContext *ctx = lhs.getContext();
      // eq(non_negative_expr, negative_expr) = false
      if ((isDefinitelyNonNegative(lhs) && isDefinitelyNegative(rhs)) ||
          (isDefinitelyNonNegative(rhs) && isDefinitelyNegative(lhs)))
        return BoolAttr::get(ctx, false);

      // eq(non_positive_expr, positive_expr) = false
      if ((isDefinitelyNonPositive(lhs) && isDefinitelyPositive(rhs)) ||
          (isDefinitelyNonPositive(rhs) && isDefinitelyPositive(lhs)))
        return BoolAttr::get(ctx, false);
    }

    return {};
  }

  /// This function serves as shape inference system's extension to KGEN
  /// folders. It leverages things like range information to enable more
  /// expression folding, and applies it recursively to all expression nested
  /// inside `attr`.
  TypedAttr tryFoldExprAttr(TypedAttr attr) {
    auto exprAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr);
    if (!exprAttr)
      return attr;

    KGEN::POC opcode = exprAttr.getOpcode();
    SmallVector<TypedAttr> foldedOperands;
    for (TypedAttr operand : exprAttr.getOperands())
      foldedOperands.push_back(tryFoldExprAttr(operand));

    if (TypedAttr foldedAttr = tryFoldExprAttr(opcode, foldedOperands))
      return foldedAttr;

    // Otherwise fallback to KGEN's folding hooks.
    return KGEN::ParamOperatorAttr::get(opcode, foldedOperands);
  }

  /// If `attr` is a `mosh.assert false`, return the error message associated
  /// with it, else return null.
  static StringAttr tryGetAssertFalseErrMsg(TypedAttr attr) {
    // Check it's an expression attribute.
    auto paramOperatorAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr);
    if (!paramOperatorAttr)
      return {};

    // Check it's a `mosh.assert`.
    auto mlirOpAttr =
        dyn_cast<KGEN::MLIROpAttr>(paramOperatorAttr.getOperand(0));
    if (!mlirOpAttr ||
        mlirOpAttr.getName() != MOSH::AssertOp::getOperationName())
      return {};

    // Check assert condition is a static false.
    if (auto cond = dyn_cast<BoolAttr>(paramOperatorAttr.getOperand(1));
        cond && cond.getValue() == false)
      return dyn_cast<StringAttr>(mlirOpAttr.getAttrs().get("message"));

    return {};
  }

  /// An enum to signal the result of op `tryDecomposeAndReplaceOp`.
  enum class ReplacementResult {
    Replaced,
    NotReplaced,
    FatalError,
  };

  /// Try decomposing `op` with `opDecomposer`. If successful, replace `op` with
  /// the decomposed ops in the graph, and push the decomposed ops onto
  /// `worklist` so they will be processed next immediately.
  static ReplacementResult
  tryDecomposeAndReplaceOp(OpDecomposer &opDecomposer, Operation *op,
                           std::deque<Operation *> &worklist) {

    // Try decompose the op, and check if it was successful.
    FailureOr<SmallVector<Operation *>> decomposedOpsOr =
        opDecomposer.tryDecomposeOp(op);
    if (failed(decomposedOpsOr))
      return ReplacementResult::FatalError;

    SmallVector<Operation *> &decomposedOps = decomposedOpsOr.value();
    if (decomposedOps.empty())
      return ReplacementResult::NotReplaced;

    OpBuilder builder(op);
    // Replace the original op with decomposed ops in correct order.
    for (Operation *decomposedOp : decomposedOps)
      builder.insert(decomposedOp);
    op->erase();

    // The first decomposed op must be processed next in the worklist, so we
    // push the decomposed ops onto the worklist in reversed order.
    for (Operation *decomposedOp : llvm::reverse(decomposedOps))
      worklist.push_back(decomposedOp);

    return ReplacementResult::Replaced;
  }

  void runOnOperation() override {
    // Clear state variables (in case the pass instance is reused)
    paramAttrToDepth.clear();
    definitelyNonNegativeAttrs.clear();
    exprToRegionToDeclRef.clear();

    GraphOp graphOp = getOperation();
    topoSortWithParameters(graphOp);

    // There are 3 special cases for adding param mappings:
    // 1. Input parameters of the graph, e.g., `mo.graph<D0, Sh1 -> ...>`
    // 2. Output param decls in an op that could not be resolved by
    // shape inference, e.g., `mo.reshape<() -> D0>`.
    // 3. Mappings to selected expressions, to avoid extremely large expressions
    // that cannot be folded, e.g., `max(D42, max(D772, 42, add(...)), ...))`.
    // TODO this limits folding as well, we need a better approach.
    //
    // In these cases, we map the declared parameters to themselves, so that
    // evaluator won't panic when looking them up. Note that (1) and (2) are
    // handled by default in `KGEN::ParameterEvaluator`, and (3) is handled in
    // the worklist loop below.
    KGEN::ParameterEvaluator evaluator;
    auto markIntParamDeclAsNonNegative = [&](KGEN::ParamDeclAttr paramDecl) {
      if (paramDecl.getType().isIntOrIndex())
        markDefinitelyNonNegative(KGEN::ParamDeclRefAttr::get(paramDecl));
    };
    // The integer input parameters to graph are dynamic dimensions, so they
    // must be non-negative.
    llvm::for_each(graphOp.getInputParams(), markIntParamDeclAsNonNegative);

    // TODO(#9008) inline inputs to `mosh.assert`
    // A replacer to help replacing parameters in an op with its evaluated form.
    // TODO(#26115) we should switch to replacing manually via
    // `evaluator.getReboundAttribute(shapeAttr)` for each op, since we only
    // care about specific region of an op, and to avoid having 2 caches -- 1 in
    // replacer and 1 in evaluator.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](KGEN::ParamDeclRefAttr declRef) -> Attribute {
      Attribute resolvedAttr = evaluator.getReboundAttribute(declRef);
      return resolvedAttr;
    });

    // A stack of ops to be "evaluated", i.e., the op in the back is the one to
    // be processed next.
    std::deque<Operation *> worklist;
    graphOp.walk([&](Operation *op) {
      // Skip the graph op itself since it'll be handled at the very end, after
      // we process all the ops inside the graph.
      if (!isa<GraphOp>(op))
        worklist.push_front(op);
    });

    // NOTE think of this as a stack machine that also decomposes "pseudo
    // instructions".
    OpDecomposer opDecomposer(graphOp.getContext(), graphOp.getLoc());
    while (!worklist.empty()) {
      Operation *op = worklist.back();
      worklist.pop_back();

      // Use accumulated param-to-expression mapping to simplify the op first.
      // This is needed to help DCE get rid of the decls with simplified exprs.
      replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                                 /*replaceLocs=*/true, /*replaceTypes=*/true);

      ReplacementResult result =
          tryDecomposeAndReplaceOp(opDecomposer, op, worklist);
      switch (result) {
      // If there was a fatal error, terminate pass immediately.
      case ReplacementResult::FatalError:
        return signalPassFailure();

      // If op was replaced, don't process it anymore, instead continue to
      // process the decomposed ops which were pushed onto worklist.
      case ReplacementResult::Replaced:
        continue;

      // If op wasn't replaced, continue processing it.
      case ReplacementResult::NotReplaced:
        break;
      }

      // Every single `kgen.param.declare x = <expr>` should add a mapping from
      // `x` to some attribute in `evaluator`.
      if (auto declOp = dyn_cast<KGEN::ParamDeclareOp>(op)) {
        KGEN::ParamDeclAttr declAttr = declOp.getParamDecl();
        auto foldedRhsAttr = cast<TypedAttr>(declOp.getValue());

        // Apply shape-inference specific folding
        foldedRhsAttr = tryFoldExprAttr(foldedRhsAttr);
        declOp.setValueAttr(foldedRhsAttr);

        auto paramOpAttr = dyn_cast<KGEN::ParamOperatorAttr>(foldedRhsAttr);
        // If the param expression cannot be nested, we either map the decl to
        // itself or to a previous decl that has the same expression. The goal
        // here is maximize textual equality for expression, e.g.,
        // - `D0` and `D0` for decls that map to themselves or other decls.
        // - `add(D0, 2)` and `add(D0, 2)` for decls that map to expressions.
        // This enables more simplifications.
        if (paramOpAttr && !canParamOperatorBeNested(paramOpAttr)) {
          // CSE on the fly for expression that cannot be nested, which helps
          // the following scenario, where we'd otherwise lose textual equality
          // for `D0` and `D1`:
          //   kgen.param.declare D0 = <...nested_expr_e1...>
          //   kgen.param.declare D1 = <...nested_expr_e1...>
          //   static.broadcast_to(...) : <[D0], f32> -> <[D1], f32>
          //
          // NOTE this CSE formulation is limited when the expression appears
          // within a _nested_ region of where it was defined, for instance,
          // currently we _do not_ CSE in this scenario:
          //   if {
          //     D0 = expr
          //     if {
          //       D1 = expr // we'd like this to become `D1 = D0`
          //     }
          //   }
          Region *opParentRegion = op->getParentRegion();
          if (exprToRegionToDeclRef.contains(paramOpAttr) &&
              exprToRegionToDeclRef[paramOpAttr].contains(opParentRegion)) {
            KGEN::ParamDeclRefAttr existingDeclRef =
                exprToRegionToDeclRef[paramOpAttr][opParentRegion];
            evaluator.setParameterValue(declAttr, existingDeclRef);
            declOp.setValueAttr(existingDeclRef);

            // Else we remember this decl has this expression (if pure), and map
            // the parameter to itself since we don't want to inline it.
          } else {
            auto declRefAttr = KGEN::ParamDeclRefAttr::get(declAttr);
            if (isParamOperatorPure(paramOpAttr))
              exprToRegionToDeclRef[paramOpAttr][opParentRegion] = declRefAttr;
            evaluator.setParameterValue(declAttr.getName(), declRefAttr);
          }

        } else {
          // The param expression can be nested, so we just record it as is.
          evaluator.setParameterValue(declAttr, foldedRhsAttr);
        }

        // Emit compile time error for any `mosh.assert false`
        if (StringAttr errMsg = tryGetAssertFalseErrMsg(foldedRhsAttr)) {
          op->emitError(errMsg.strref());
          return signalPassFailure();
        }
      }
    }

    handleOutputBindings(graphOp, evaluator);
    replacer.replaceElementsIn(graphOp, /*replaceAttrs=*/true,
                               /*replaceLocs=*/true, /*replaceTypes=*/true);
    updateGraphSignature(graphOp);

    // DCE on `kgen.param.declare`s for better readability/debuggability.
    auto &paramCache = getAnalysis<KGEN::ParameterCollector::Analysis>();
    KGEN::ParameterUseDefGraph useDefGraph(graphOp.getRegion());
    useDefGraph.calculate(paramCache);
    removeDeadParamDeclares(graphOp, useDefGraph);
  }

  // State variables used by the pass
  DenseMap<KGEN::ParamOperatorAttr, size_t> paramAttrToDepth;
  DenseSet<TypedAttr> definitelyNonNegativeAttrs;
  // This maps from
  // - expressions that we have processed, but decided not to inline (i.e.,
  //   we'll keep it as a `kgen.param.declare X = <expr>` and refer to `X`
  //   elsewheres).
  // to another map from
  // - the immediately enclosing region where the expression was defined.
  // to
  // - Reference to the definition parameter of the expression, for instance,
  //   given `kgen.param.declare X = <expr>`, this would be `X`.
  DenseMap<KGEN::ParamOperatorAttr, DenseMap<Region *, KGEN::ParamDeclRefAttr>>
      exprToRegionToDeclRef;
};
