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
#include "mlir/Analysis/TopologicalSortUtils.h"
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
  Success &getSuccess() { return std::get<Success>(result); };
  Error &getError() { return std::get<Error>(result); };

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

/// A class that helps `PropagateShapesPass` handle ops that could be
/// decomposed when the shape parameters they use have been simplified
/// already. For instance, in order to simplify shape function calls, we often
/// need its two input shapes to be simplified first (e.g., `[D0, 1]` instead of
/// `Sh0`). However, the simplification happens _during_ `PropagateShapesPass`,
/// so we use this `OpDecomposer` to decompose ops on the fly, and letting
/// `PropagateShapesPass` to further simplify the decomposed ops.
class OpDecomposer {
  // State that stays invariant across calls.
  OpBuilder builder;

  // State that might change across calls.
  GraphOp graphOp;
  Location loc;
  // Parameter bindings created during op decomposition, which will be converted
  // to `KGEN::ParamDeclareOp` if decomposition is successful.
  SmallVector<ParamBinding> bindings;

public:
  /// Create an OpDecomposer with given MLIRContext.
  OpDecomposer(MLIRContext *ctx, Location loc) : builder(ctx), loc(loc) {}

  /// Try decomposing the given Operation if possible. Return either
  /// - a failure as fatal failure (e.g., some shape assertion failed)
  /// - a list of decomposed ops (ordered by SSA and parametric dependencies) if
  ///   decomposition was possible
  /// - else return an empty vector.
  /// The input operation won't be modified.
  ErrorOr<SmallVector<KGEN::ParamDeclareOp>> tryDecomposeOp(Operation *op) {
    // Reset transient state.
    graphOp = op->getParentOfType<GraphOp>();
    loc = op->getLoc();
    bindings.clear();

    SmallVector<KGEN::ParamDeclareOp> ops;
    if (auto invokeShapeFuncOp = dyn_cast<InvokeShapeFuncOp>(op)) {
      ErrorOr<LogicalResult> errOrRes =
          tryDecomposeInvokeShapeFuncOp(invokeShapeFuncOp);
      if (errOrRes.isError())
        return errOrRes.takeError();

      // Populate `ops` only if decomposition was successful.
      if (errOrRes.get().succeeded()) {
        // Convert the bindings to actual `kgen.param.declare` ops.
        for (const ParamBinding &binding : bindings) {
          KGEN::ParamDeclareOp declOp = binding.toOp(loc, builder);
          ops.push_back(declOp);
        }
      }
    }
    return ops;
  }

private:
  /// Try converting `val` to either
  /// - a symbolic int/bool (if `val` is a scalar tensor) or
  /// - a list of the above (if `val` is a 1D tensor)
  ///
  /// `b` is used to build up and track parameter expressions, if needed.
  static TypedAttr tryStaticizeImpl(Value val, ParamExprBuilder &b) {
    // Only staticize tensor.
    auto tensorVal = dyn_cast<TensorValue>(val);
    if (!tensorVal)
      return {};

    // TODO move all these to a new interface implementations of the ops.

    // Look through rebinds which often gets generated by parameterization.
    if (auto rebindOp = val.getDefiningOp<RebindOp>())
      return tryStaticizeImpl(rebindOp.getInput(), b);

    // Only staticize scalar, 1D or 2D tensor.
    TensorType tensorTy = tensorVal.getType();
    if (!tensorTy.hasRank() || tensorTy.getRank() > 2)
      return {};

    // Only staticize int/bool tensor.
    if (!isa<IntegerType>(tensorTy.getElementType()))
      return {};

    // Examples: `D0`
    if (auto dimAttr = tryGetDimAttr(val))
      return dimAttr;

    // Examples: `[D0, 10, ...]`
    if (auto shapeAttr = tryGetShapeAttr(val))
      return shapeAttr;

    MLIRContext *ctx = val.getContext();
    // Example: 42.
    if (auto intOr = MO::tryGetConstantInt(tensorVal))
      return IntegerAttr::get(IndexType::get(ctx), intOr.value());

    // Example: [10, 20, ...]
    if (auto intsOr = MO::tryGetConstantIntVector(tensorVal))
      return ShapeAttr::get(intsOr.value(), MOSH::ShapeType::get(ctx));

    // Example:
    //   lhs =     [D0, 1]
    //   rhs = [D1, 1, 10]
    //   mo.broadcast_shape(lhs, rhs) --> [D1, D0, 10]
    if (auto bcastOp = val.getDefiningOp<MO::BroadcastShapeOp>()) {
      auto shapeAttr1 =
          dyn_cast_or_null<ShapeAttr>(tryStaticizeImpl(bcastOp.getInputX(), b));
      auto shapeAttr2 =
          dyn_cast_or_null<ShapeAttr>(tryStaticizeImpl(bcastOp.getInputY(), b));
      if (shapeAttr1 && shapeAttr2) {
        // Ensure rank(shapeAttr1) >= rank(shapeAttr2)
        if (shapeAttr1.getRank() < shapeAttr2.getRank())
          std::swap(shapeAttr1, shapeAttr2);

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
            //
            // NOTE we optimize for the case where `D1` is a complex expression
            // but `D2` is 1, in this case the output dim is just `D1`, rather
            // than `cond(D1 == 1, D2, D1)` which won't be simplified.
            //
            // Maybe we should create an intrinsic for `broadcast_dim`, i.e.,
            // another POC just like `add`, `div`, etc.?
            Var newResDim = dim1;
            if (tryGetConstantInt(dim2.getAttr()) != 1)
              newResDim = b.createCond(dim1 == 1, dim2, dim1);
            resDims.push_back(newResDim.getAttr());
          }
        }
        auto res = MOSH::ShapeAttr::get(resDims, shapeAttr1.getType());
        return res;
      }
    }

    return {};
  }

  /// Try converting `val` to either
  /// - a symbolic int/bool (if `val` is a scalar tensor) or
  /// - a list of the above (if `val` is a 1D tensor)
  static StaticizeResult tryStaticize(Value val) {
    SmallVector<ParamBinding> bindings;
    auto graphOp = val.getParentRegion()->getParentOfType<GraphOp>();
    ParamExprBuilder b(graphOp, val.getLoc(), &bindings);
    TypedAttr res = tryStaticizeImpl(val, b);
    if (!res)
      return StaticizeResult::nonStaticizable();

    return StaticizeResult::success(std::move(bindings), res);
  }

  // Treat certain fallback shape function call as special intrinsics and
  // effectively symbolicizing them _during_ shape propagation.
  //
  // To interpret the return value:
  // 1. Error if encountered a fatal error
  // 2. Failure if unable to decompose the op.
  // 3. Success otherwise.
  ErrorOr<LogicalResult>
  tryDecomposeInvokeShapeFuncOp(InvokeShapeFuncOp invokeShapeFuncOp) {
    // Turn inputs into parameter attributes; the mechanism depends on whether
    // they are data-dependent or not.
    DenseSet<int64_t> dataDepInputIdxs;
    if (auto idxs = invokeShapeFuncOp.getDataDeptTensors()) {
      for (llvm::APInt index : idxs->getAsValueRange<IntegerAttr>())
        dataDepInputIdxs.insert(index.getLimitedValue());
    }
    SmallVector<TypedAttr> inputAttrs;
    for (auto [idx, input] : llvm::enumerate(invokeShapeFuncOp.getOperands())) {
      if (dataDepInputIdxs.contains(idx)) {
        StaticizeResult res = tryStaticize(input);

        // Short circuit if a data-dependent input can't be staticized.
        if (res.isNonStaticizable())
          return success();

        // Short circuit on any fatal error.
        if (res.isError())
          return Error(res.getError().msg);

        auto &success = res.getSuccess();
        llvm::append_range(bindings, success.bindings);
        inputAttrs.push_back(success.resultAttr);

      } else {
        // Short circuit if a non-data-dependent input doesn't have a shape; not
        // exactly sure how this could happen but just in case.
        TensorValue tensorInput = dyn_cast<TensorValue>(input);
        if (!tensorInput) {
          return success();
        }

        inputAttrs.push_back(tensorInput.getType().getShapeAttr());
      }
    }

    // Dispatch on certain shape function names. TODO instead of using a map
    // loopkup, guard this against a new interface.
    StringRef shapeFuncName = invokeShapeFuncOp.getShapeFuncName();
    auto shapeMaterializeFunc = tryGetShapeMaterializeFunc(shapeFuncName);
    // Give up if we can't figure out how to infer this shape function.
    if (!shapeMaterializeFunc.has_value())
      return success();

    DictionaryAttr kgenParams = invokeShapeFuncOp.getKgenParamsAttr();
    ShapeMaterializeResult result =
        shapeMaterializeFunc.value()(inputAttrs, kgenParams, graphOp, loc);
    // Give up if we can't materialize the shape logic.
    if (result.isNonMaterializable())
      return success();

    // Propagate error to caller.
    if (result.isError())
      return Error(result.getError().msg);

    // Successfully inferred, package and propagate information to caller.
    if (result.isSuccess()) {
      // Create new declaration of the shape function call's output parameter;
      auto &success = result.getSuccess();
      llvm::append_range(bindings, success.bindings);

      // `outputShape` has the form `[N0, N1, ...]`
      ShapeAttr outputShape = success.resultShape;

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
      // However, we need to support dynamic rank here in order to resolve
      // pseudo-dynamic-rank which we might get from RMO or Torch.
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
    ErrorOr<SmallVector<KGEN::ParamDeclareOp>> decomposedOpsOr =
        opDecomposer.tryDecomposeOp(op);
    if (decomposedOpsOr.isError()) {
      op->emitError(decomposedOpsOr.getError());
      return ReplacementResult::FatalError;
    }

    SmallVector<KGEN::ParamDeclareOp> &decomposedOps = decomposedOpsOr.get();
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

        // Emit compile time error for any `mo.assert false`
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
