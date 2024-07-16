//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Analysis/MODialect/ParametricTopologyAnalysis.h"
#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MODialect/ShapeFunctions.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/AssertStream.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include <optional>
#include <sstream>
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

/// A variant class that contains result of staticizing a series of computation
/// on shapes tensors.
/// - Success (with parameter bindings and result attribute).
/// - NonStaticizable (when inputs are too dynamic).
/// - Error (when a fatal error has been detected).
class StaticizeResult {
public:
  struct Success {
    // A series of bindings which model `kgen.param.declare` ops that define
    // either
    // - parameter expressions like `D2 = <add(D0, D1)>`
    // - assertions like `Chain = <...assert(cond, msg)...>`
    //
    // These could then be consumed by shape propagation logic which does
    // substitution & folding, and handles these asserts.
    SmallVector<ParamBinding> bindings;

    // This is the result of staticization, which could be either
    // - `IntegerAttr` for a constant integer or boolean.
    // - `ParamDeclRefAttr` for an integer or boolean parameter.
    // - `ParamOperatorAttr` for an integer or boolean parameter expression.
    // - `ShapeAttr` which is a list where each item could be any of the above.
    TypedAttr resultAttr;

    // Constructors
    Success() = default;
    Success(SmallVector<ParamBinding> &&bindings, TypedAttr resultAttr)
        : bindings(std::move(bindings)), resultAttr(resultAttr) {}

    // Disable copy and move.
    Success(const Success &) = delete;
    Success(Success &&) = delete;
    Success &operator=(const Success &) = delete;
    Success &operator=(Success &&) = delete;
  };
  struct NonStaticizable {};
  struct Error {
    std::string msg;

    // Constructors
    Error(const Twine &msg) : msg(msg.str()) {}

    // Disable copy and move.
    Error(const Error &) = delete;
    Error(Error &&) = delete;
    Error &operator=(const Error &) = delete;
    Error &operator=(Error &&) = delete;
  };

  bool isSuccess() { return std::holds_alternative<Success>(result); }
  bool isNonStaticizable() {
    return std::holds_alternative<NonStaticizable>(result);
  }
  bool isError() { return std::holds_alternative<Error>(result); }

  /// Returning a reference so that we could either use return as
  /// - a rvalue, e.g., `res.getSuccess().bindings` or
  /// - a lvalue, e.g., `const auto& success = res.getSuccess()`
  Success &getSuccess() { return std::get<Success>(result); }
  Error &getError() { return std::get<Error>(result); }

  static StaticizeResult success(SmallVector<ParamBinding> &&bindings,
                                 TypedAttr resultAttr) {
    return StaticizeResult(std::move(bindings), resultAttr);
  }
  static StaticizeResult nonStaticizable() { return StaticizeResult(); }
  static StaticizeResult error(const Twine &errMsg) {
    return StaticizeResult(errMsg);
  }

private:
  StaticizeResult(SmallVector<ParamBinding> &&bindings, TypedAttr resultAttr) {
    result.emplace<Success>(std::move(bindings), resultAttr);
  }
  StaticizeResult() { result = NonStaticizable{}; }
  StaticizeResult(const Twine &errMsg) { result.emplace<Error>(errMsg.str()); }
  std::variant<Success, NonStaticizable, Error> result;
};

} // namespace

namespace {

/// Try converting `val` to either
/// - a symbolic int/bool (if `val` is a scalar tensor) or
/// - a list of the above (if `val` is a 1D tensor)
///
/// `b` is used to build up and track parameter expressions, if needed.
static ErrorOr<TypedAttr> tryStaticizeImpl(Value val, ParamExprBuilder &b) {
  // Base case -- source of shape tensor.
  if (auto shapeOfOp = val.getDefiningOp<ShapeOfOp>())
    return shapeOfOp.getInput().getType().getShapeAttr();

  // Only staticize tensor.
  auto tensorVal = dyn_cast<TensorValue>(val);
  if (!tensorVal)
    return TypedAttr{};

  // Only staticize scalar, 1D or 2D tensor.
  TensorType tensorTy = tensorVal.getType();
  if (!tensorTy.hasRank() || tensorTy.getRank() > 2)
    return TypedAttr{};

  // Only staticize int/bool tensor.
  if (!isa<IntegerType>(tensorTy.getElementType()))
    return TypedAttr{};

  // TODO replace `tryGetDimAttr` with `Staticization` implementation of
  // relevant ops. Examples: `D0`
  if (auto dimAttr = tryGetDimAttr(val))
    return dimAttr;

  // TODO replace `tryGetShapeAttr` with `Staticization` implementation of
  // relevant ops. Examples: `[D0, 10, ...]`
  if (auto shapeAttr = tryGetShapeAttr(val))
    return shapeAttr;

  // Handle ops that implement the Staticization interface.
  if (auto staticization = val.getDefiningOp<Staticization>()) {
    // Staticize inputs and collect the results.
    SmallVector<TypedAttr> staticizedInputs;
    for (Value input : staticization->getOperands()) {
      // Recur on the input value.
      ErrorOr<TypedAttr> staticizedInputOr = tryStaticizeImpl(input, b);

      // Short-circuit on error.
      if (staticizedInputOr.isError())
        return Error(staticizedInputOr.getError());

      // Short-circuit on non-staticizable inputs.
      TypedAttr staticizedInput = staticizedInputOr.get();
      if (!staticizedInput)
        return TypedAttr{};

      // Collect the staticized input.
      staticizedInputs.push_back(staticizedInput);
    }
    // Invoke the interface method.
    return staticization.tryStaticize(staticizedInputs, b);
  }

  return TypedAttr{};
}

/// Try converting `val` to either
/// - a symbolic int/bool (if `val` is a scalar tensor) or
/// - a list of the above (if `val` is a 1D tensor)
static StaticizeResult tryStaticize(Value val) {
  SmallVector<ParamBinding> bindings;
  auto graphOp = val.getParentRegion()->getParentOfType<GraphOp>();
  ParamExprBuilder b(graphOp, val.getLoc(), &bindings);
  ErrorOr<TypedAttr> resOr = tryStaticizeImpl(val, b);
  if (resOr.isError())
    return StaticizeResult::error(resOr.getError());

  TypedAttr res = resOr.get();
  if (!res)
    return StaticizeResult::nonStaticizable();

  return StaticizeResult::success(std::move(bindings), res);
}

/// Try to shape infer the input op if possible, given the input shapes or
/// staticized input data. Return either
/// - a failure as fatal failure (e.g., some shape assertion failed)
/// - a list of decomposed ops (ordered by SSA and parametric dependencies) if
///   materialization was possible
/// - else return an empty vector.
static ErrorOr<SmallVector<KGEN::ParamDeclareOp>>
shapeInferOpImpl(ShapeMaterialization op, ArrayRef<TypedAttr> inputAttrs) {

  ShapeMaterializeResult result = op.materializeShapeDefs(inputAttrs);
  // Give up if we can't materialize the shape logic.
  if (result.isNonMaterializable())
    return SmallVector<KGEN::ParamDeclareOp>();

  // Propagate error to caller.
  if (result.isError())
    return Error(result.getError().msg);

  // Successfully inferred, package and propagate information to caller.
  SmallVector<ParamBinding> bindings;
  if (result.isSuccess()) {
    // Create new declaration of the shape function call's output parameter;
    auto &success = result.getSuccess();
    llvm::append_range(bindings, success.bindings);

    // `outputShape` has the form `[N0, N1, ...]`
    ShapeAttr outputShape = success.resultShape;

    // outputDecls is always one of
    // - Sh2: !mosh.ape
    // - [D0, D1, ...]
    ArrayRef<KGEN::ParamDeclAttr> outputDecls = op.getOutputParamDecls();

    // NOTE symbolicization currently can't handle dynamic output shape, since
    // it relies on MOGG compiling the shape function down to KGEN IR, which
    // relies on the mojo shape function (and the bottom part of our stack) to
    // support dynamic rank.
    //
    // However, we need to support dynamic rank here in order to resolve
    // pseudo-dynamic-rank which we might get from RMO or Torch.
    // TODO(GRA-927) remove this branch.
    if (isa<InvokeShapeFuncOp>(op)) {
      if (outputDecls.size() == 1 &&
          isa<ShapeType>(outputDecls.front().getType())) {
        // Create binding for the output shape parameter to output shape which
        // models ops like the following:
        //   kgen.param.decare D0 = <[N0, N1, ...]>
        bindings.push_back(ParamBinding(outputDecls.front(), outputShape));

      } else {
        size_t returnedRank = outputShape.getRank();
        size_t annnotatedRank = outputDecls.size();
        if (returnedRank != annnotatedRank) {
          std::stringstream ss;
          ss << "shape function returned shape of rank " << returnedRank
             << " but the type annotation has shape of rank " << annnotatedRank;
          return Error(ss.str());
        }
        // Create binding from each output parameter to its dimension, to model
        // ops like the following:
        //   kgen.param.decare D0 = <N0>
        //   kgen.param.decare D1 = <add(N1, 10)>
        //   ...
        for (auto [outputDimDecl, outputDim] :
             llvm::zip(outputDecls, outputShape.getValues())) {
          bindings.push_back(ParamBinding(outputDimDecl, outputDim));
        }
      }

    } else {
      if (outputDecls.size() == 1 &&
          isa<ShapeType>(outputDecls.front().getType())) {
        // Case 0 (output decl is a shape)
        //   mo.reshape<() -> Sh0: !mosh.ape> : ... -> Sh0
        //
        // Create binding for the output shape parameter to output shape which
        // models ops like the following:
        //   kgen.param.decare Sh0 = <[N0, N1, ...]>
        bindings.push_back(ParamBinding(outputDecls.front(), outputShape));
      } else {
        // Case 1 (output decls are dimensions)
        //   mo.reshape<() -> D1, D2> : ... -> [D1, D2]
        //   mo.reshape<() -> D1> : ... -> [10, D1]
        //
        // Create binding for the output shape parameter to output dims which
        // models ops like the following:
        //   kgen.param.decare D1 = <N0>
        //   kgen.param.decare D2 = <add(N1, 10)>
        //
        // TODO support variadic output
        ShapeAttr annotatedOutShape =
            cast<TensorType>(op->getResult(0).getType()).getTypedShapeAttr();
        size_t returnedRank = outputShape.getRank();
        size_t annnotatedRank = annotatedOutShape.getRank();
        if (returnedRank != annnotatedRank) {
          std::stringstream ss;
          ss << "shape function returned shape of rank " << returnedRank
             << " but the type annotation has shape of rank " << annnotatedRank;
          return Error(ss.str());
        }
        // Given `outputShape` for
        //   mo.reshape<() -> D1> : ... -> [10, D1]
        // Generate
        //   D1 = <outputShape[1]>
        llvm::SmallSetVector<ParamDeclAttr, 9> outputDeclSet(
            outputDecls.begin(), outputDecls.end());
        for (auto [axis, dim] :
             llvm::enumerate(annotatedOutShape.getValues())) {
          // Skip if `dim` is not a dimension parameter
          auto paramDim = dyn_cast<KGEN::ParamDeclRefAttr>(dim);
          if (!paramDim)
            continue;

          // Skip if `dim` is defined elsewhere.
          auto paramDecl = KGEN::ParamDeclAttr::get(paramDim);
          if (!outputDeclSet.contains(paramDecl))
            continue;

          bindings.push_back(ParamBinding(paramDecl, outputShape[axis]));
        }
      }
    }
  }

  // Convert the bindings to actual `kgen.param.declare` ops.
  OpBuilder builder(op.getContext());
  SmallVector<KGEN::ParamDeclareOp> ops;
  Location loc = op->getLoc();
  for (const ParamBinding &binding : bindings) {
    auto declOp =
        builder.create<KGEN::ParamDeclareOp>(loc, binding.decl, binding.value);
    ops.push_back(declOp);
  }
  return ops;
}

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

  /// Try to shape infer through given op. If successful, replace `op` with the
  /// newly generated ops, and push the new ops onto `worklist` so they will be
  /// processed next immediately.
  static ErrorOrSuccess shapeInferOp(ShapeMaterialization op,
                                     std::deque<Operation *> &worklist) {
    // Turn inputs into parameter attributes; the mechanism depends on whether
    // they are data-dependent or not.
    SmallVector<ParamBinding> staticizationBindings;
    DenseSet<int32_t> dataDepInputIdxs;
    for (int32_t idx : op.getDataDependentInputIndices()) {
      ASSERT_STREAM(0 <= idx && uint32_t(idx) < op->getNumOperands(),
                    << "data depedent input index must be in bound");
      dataDepInputIdxs.insert(idx);
    }
    SmallVector<TypedAttr> inputAttrs;
    for (auto [idx, input] : llvm::enumerate(op->getOperands())) {
      if (dataDepInputIdxs.contains(idx)) {
        StaticizeResult res = tryStaticize(input);

        // Short circuit if a data-dependent input can't be staticized.
        if (res.isNonStaticizable())
          return success();

        // Short circuit on any fatal error.
        if (res.isError())
          return Error(res.getError().msg);

        auto &success = res.getSuccess();
        llvm::append_range(staticizationBindings, success.bindings);
        inputAttrs.push_back(success.resultAttr);

      } else {
        // Short circuit if a non-data-dependent input doesn't have a shape; not
        // exactly sure how this could happen but just in case.
        TensorValue tensorInput = dyn_cast<TensorValue>(input);
        if (!tensorInput)
          return success();

        inputAttrs.push_back(tensorInput.getType().getShapeAttr());
      }
    }

    // Try shape infer the op, and check if it was successful.
    ErrorOr<SmallVector<KGEN::ParamDeclareOp>> paramDeclOpsOr =
        shapeInferOpImpl(op, inputAttrs);
    if (paramDeclOpsOr.isError())
      return Error(paramDeclOpsOr.getError());

    SmallVector<KGEN::ParamDeclareOp> &paramDeclOps = paramDeclOpsOr.get();
    if (paramDeclOps.empty())
      return success();

    // Materialize staticized values; we do this because for certain ops (e.g.,
    // `mo.slice`), the data dependent input might still get fed to the op
    // itself, even after the op has been shape inferred. In such cases it's
    // better to have those input something like
    //   %tmp = mosh.param.to_value: !mosh.ape = <[10, D0]>
    //   %input = mo.shape.to_tensor(%tmp) : !mo.tensor<[2], si32>
    // rather than
    //   %shape = mo.shape_of(%x)
    //   %d0 = mo.gather(%shape)
    //   %input = mo.concat(%ten, %d0) : ... -> !mo.tensor<[2], si32>
    OpBuilder builder(op);
    Location loc = op.getLoc();
    SmallVector<Operation *> newOps; // Tracks all the newly created ops.
    for (ParamBinding &binding : staticizationBindings)
      newOps.push_back(binding.toOp(loc, builder));
    for (auto [idx, input] : llvm::enumerate(op->getOperands())) {
      // Skip if it's not a data dependent input, or if it's already been
      // materialized (this occurs if the input is shared between multiple ops).
      if (!dataDepInputIdxs.contains(idx) ||
          !input.getDefiningOp<Staticization>())
        continue;

      TensorType inputTensorTy = cast<TensorType>(input.getType());
      DType inputDtype = inputTensorTy.getDtype();
      // We only staticize input of type `si32` or `si64`, this is a practical
      // restriction we can easily extend if there are perf needs.
      if (!inputDtype.isInt() ||
          !llvm::is_contained({32, 64}, inputDtype.getWidthInBits()))
        continue;

      // Grab the saved staticization result, and create
      //   %tmp   = mosh.param.to_value: !mosh.ape = <[10, D0]>
      //   %input = mo.shape.to_tensor(%tmp) : !mo.tensor<...>
      TypedAttr inputAttr = inputAttrs[idx];
      auto paramToValOp =
          builder.create<MOSH::ParamToValueOp>(loc, inputAttrs[idx]);
      Value newInputVal;
      if (isa<ShapeType>(inputAttr.getType())) {
        newInputVal = builder.create<Shape::ToTensorOp>(
            loc, inputTensorTy, paramToValOp.getResult());
      } else {
        newInputVal = builder.create<IndexToTensorOp>(loc, inputTensorTy,
                                                      paramToValOp.getResult());
      }
      // Track the new ops so they'll get processed in the worklist loop later.
      newOps.push_back(paramToValOp);
      newOps.push_back(newInputVal.getDefiningOp());

      // Replace old uses with the new op.
      input.replaceAllUsesWith(newInputVal);
    }

    // Replace the original op with param decl ops in correct order.
    // Note that the ops from staticization has already been inserted correctly
    // since they were built with `builder`.
    for (Operation *declOp : paramDeclOps)
      builder.insert(declOp);
    // TODO(GRA-627) remove the true branch
    bool isInvokeShapeFuncOp = isa<InvokeShapeFuncOp>(op);
    if (isInvokeShapeFuncOp)
      op.erase();
    else
      op.setOutputParamDecls({});

    // The first new op must be processed next in the worklist, so we push the
    // new ops onto the worklist in reversed order.
    llvm::append_range(newOps, paramDeclOps);
    // Note that we also put `op` itself back onto the worklist, to be
    // re-visited after we visit all the new ops, since that's when we'll have
    // all the information needede to update the output shape of `op` (if
    // needed).
    // TODO(GRA-627) remove this branch
    if (!isInvokeShapeFuncOp)
      worklist.push_back(op);
    for (Operation *newOp : llvm::reverse(newOps))
      worklist.push_back(newOp);

    return success();
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

    // TODO(#9008) inline inputs to `mo.assert`
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

    // A loop to process ops from worklist.
    while (!worklist.empty()) {
      Operation *op = worklist.back();
      worklist.pop_back();

      if (auto shapeMaterialization = dyn_cast<ShapeMaterialization>(op)) {
        // If output param decl is empty, then we skip this branch, and proceed
        // to then end of this loop body, where replacer updates the parameter
        // within `op`.
        if (!shapeMaterialization.getOutputParamDecls().empty()) {
          ErrorOrSuccess result = shapeInferOp(shapeMaterialization, worklist);
          // Short-circuit on fatal error.
          if (result.isError()) {
            op->emitError(result.getError());
            return signalPassFailure();
          }
          // Otherwise the op has been shape-inferred, continue to process the
          // new ops that were pushed onto the worklist.
          continue;
        }

      } else if (auto shapeOfOp = dyn_cast<ShapeOfOp>(op)) {
        // TODO merge this branch with the `InvokeShapeFuncOp` branch once we
        // have a common interface to model shape function logic of MO ops.

        // Skip if input shape is still dynamically ranked. At this stage, it
        // indicates a rare scenario where there might be a real dynamic rank,
        // which our stack doesn't support at the moment.
        if (auto inputShape = dyn_cast<ShapeAttr>(
                shapeOfOp.getInput().getType().getShapeAttr())) {
          // Change
          //   mo.shape_of<() -> D1> : [N1, N2] -> [D1]
          // to
          //   kgen.param.declare D1 = <2>
          //   mo.shape_of : [N1, N2] -> [D1]
          ArrayRef<KGEN::ParamDeclAttr> outParamDecls =
              shapeOfOp.getOutputParamDecls();
          if (!outParamDecls.empty()) {
            KGEN::ParamDeclAttr decl = outParamDecls.front();
            ASSERT_STREAM(outParamDecls.size() == 1 && decl.getType().isIndex(),
                          << "expect only 1 output dimension parameter");
            ImplicitLocOpBuilder builder(op->getLoc(), op);
            auto rankAttr = builder.getIndexAttr(inputShape.getRank());
            auto declOp = builder.create<KGEN::ParamDeclareOp>(decl, rankAttr);
            shapeOfOp.setOutputParamDecls({});
            worklist.push_back(op);
            worklist.push_back(declOp);
            // Continue to let worklist process the new ops.
            continue;
          }
        }
      }

      // Then handle existing `kgen.param.declare`.
      if (auto declOp = dyn_cast<KGEN::ParamDeclareOp>(op)) {
        KGEN::ParamDeclAttr declAttr = declOp.getParamDecl();
        auto foldedRhsAttr =
            cast<TypedAttr>(evaluator.getReboundAttribute(declOp.getValue()));

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

        // Emit compile time error for any `mo.assert false`
        if (StringAttr errMsg = tryGetAssertFalseErrMsg(foldedRhsAttr)) {
          op->emitError(errMsg.strref());
          return signalPassFailure();
        }
      }

      // At this point, we know replacer has absorbed as much relevant parameter
      // definitions as possible in order to update parameters in `op`. So we
      // just use the accumulated param-to-expression mapping to simplify the
      // op. This helps DCE get rid of the decls with simplified exprs.
      replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                                 /*replaceLocs=*/true, /*replaceTypes=*/true);
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

    // Extra processing to convert
    //   mo.shape_of(%input) : [N1, 32] -> ...
    // to
    //   const_shape = mosh.param.to_value <N, 32>
    //   shape       = mo.shape.to_tensor(const_shape) : ...
    //
    // We need this because it allows us to get rid of the false data dependency
    // from this op (and all its immediate & transitive users) to the `%input`
    // value.
    //
    // We put them here because
    // 1. Putting it in the worklist loop above doesn't enable more shape
    //    propagation, and it adds more complexity to the loop.
    // 2. Putting it in `ShapeOfOp` canonicalizer would make this happen in _too
    //    many_ places, which might affect debugging, especially when we want to
    //    examine the output of frontend lowering (e.g., to check correctness or
    //    understand model semantics)
    getOperation().walk([&](ShapeOfOp op) {
      ImplicitLocOpBuilder builder(op.getLoc(), op);
      TypedAttr inputShape = op.getInput().getType().getShapeAttr();
      auto shConst = builder.create<MOSH::ParamToValueOp>(inputShape);
      auto toTensor = builder.create<Shape::ToTensorOp>(op.getType(), shConst);
      op.replaceAllUsesWith(toTensor.getResult());
      op.erase();
      return WalkResult::advance();
    });
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
