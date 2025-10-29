//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Analysis/MODialect/ParametricTopologyAnalysis.h"
#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MODialect/ShapeFunctions.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/AssertStream.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"

#include <deque>

using namespace M;
using namespace MO;

namespace M::MO {

#define GEN_PASS_DEF_PROPAGATESHAPES

#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

} // namespace M::MO

// Look up the callee graph op by name in the containing module op.
static GraphOp lookupCallee(ModuleOp moduleOp, StringRef name) {
  for (GraphOp graph : moduleOp.getOps<GraphOp>())
    if (graph.getName() == name)
      return graph;

  return nullptr;
}

// Collect unique parameter declarations referenced by the shape of a tensor
// type; used to populate rebind's outputParamDecls when the expected type
// refers to params not yet declared in the caller graph.
static SmallVector<KGEN::ParamDeclAttr> collectDeclsFromShape(TensorType ty) {
  llvm::SmallSetVector<KGEN::ParamDeclAttr, 4> decls;
  if (auto shape = dyn_cast<MOSH::ShapeAttr>(ty.getShapeAttr())) {
    for (TypedAttr dim : shape.getValues())
      if (auto ref = dyn_cast<KGEN::ParamDeclRefAttr>(dim))
        decls.insert(KGEN::ParamDeclAttr::get(ref));
  }
  return SmallVector<KGEN::ParamDeclAttr>(decls.begin(), decls.end());
}

// Normalize mo.call operands to match the callee signature by inserting
// mo.rebind where shape symbols differ across the boundary.
static void normalizeCallOperandsToCallee(GraphOp graphOp) {
  ModuleOp moduleOp = graphOp->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return;

  graphOp.walk([&](CallOp call) {
    GraphOp callee = lookupCallee(moduleOp, call.getCallee());
    if (!callee)
      return WalkResult::advance();

    auto calleeInputs = callee.getFunctionType().getInputs();
    if (calleeInputs.size() != call->getNumOperands())
      return WalkResult::advance();

    OpBuilder b(call);
    Location loc = call.getLoc();
    for (auto [idx, operand] : llvm::enumerate(call.getOperands())) {
      auto expected = dyn_cast<TensorType>(calleeInputs[idx]);
      auto actual = dyn_cast<TensorType>(operand.getType());
      if (!expected || !actual)
        continue;

      if (expected.cloneWithoutLayout() == actual.cloneWithoutLayout())
        continue;

      // Only rebind if dtype/device agree; otherwise let verifier report.
      if ((expected.getDtype() != actual.getDtype()) ||
          (expected.getDeviceRef() != actual.getDeviceRef()))
        continue;

      Value reb = RebindOp::create(b, loc, expected, operand,
                                   collectDeclsFromShape(expected));
      call->setOperand(idx, reb);
    }
    return WalkResult::advance();
  });
}

namespace {

bool isParamOperatorPure(KGEN::ParamOperatorAttr paramOperator) {
  return paramOperator.getOpcode() != KGEN::POC::Apply;
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
  for (KGEN::ParamDeclAttr decl : graphOp.getResultParameters())
    resDeclNames.insert(decl.getName());
  SmallPtrSet<StringAttr, 4> argDeclNames;
  for (KGEN::ParamDeclAttr decl : graphOp.getInputParameters())
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
       llvm::zip(outputOp.getParameters(), graphOp.getResultParameters())) {
    // NOTE the binding parameter has already been rebound.
    // Output parameter could be either a shape or dimension, we check to make
    // sure the resolved parameter is valid.
    if ((isa<ShapeType>(bindingParam.getType()) &&
         isValidOutputShape(bindingParam)) ||
        isValidOutputDim(bindingParam)) {
      evaluator.setDeclBinding(resultParam.getName(), bindingParam);
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
       llvm::zip(outputOp.getParameters(), graphOp.getResultParameters())) {
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
  attrs.set(graphOp.getResultParametersAttrName(), newResultParamsAttr);

  FuncTypeGeneratorType signature =
      KGEN::FuncTypeGeneratorType::remapToFuncTypeGenerator(
          graphOp.getInputParameters(), graphOp.getFunctionType());
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

  if (auto paramToValue = val.getDefiningOp<MOSH::ParamToValueOp>())
    if (isa<IntegerType, MOSH::ShapeType>(paramToValue.getType()))
      return paramToValue.getValue();

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

static TypedAttr tryFoldMoshOp(Value val) {
  auto op = val.getDefiningOp();
  if (!op || !isa<M::MOSH::MOSHDialect>(op->getDialect()))
    return TypedAttr{};

  auto attr = MOSH::tryFoldOp(op);
  if (failed(attr))
    return TypedAttr{};

  return *attr;
}

/// Try converting `val` to either
/// - a symbolic int/bool (if `val` is a scalar tensor) or
/// - a list of the above (if `val` is a 1D tensor)
static StaticizeResult tryStaticize(Value val) {
  // First, try folding, which is likely to be cheaper than the staticization
  // interfaces.
  if (TypedAttr folded = tryFoldMoshOp(val))
    return StaticizeResult::success({}, folded);

  SmallVector<ParamBinding> bindings;
  auto graphOp = val.getParentRegion()->getParentOfType<GraphOp>();
  ParamExprBuilder b(graphOp, val.getLoc(), &bindings);
  ErrorOr<TypedAttr> resOr = tryStaticizeImpl(val, b);
  if (resOr.isError())
    return StaticizeResult::error(resOr.getError());

  if (TypedAttr res = resOr.get())
    return StaticizeResult::success(std::move(bindings), res);

  return StaticizeResult::nonStaticizable();
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

    // outputDecls is always one of
    // - Sh2: !mosh.ape
    // - [D0, D1, ...]
    ArrayRef<KGEN::ParamDeclAttr> outputDecls = op.getOutputParamDecls();

    if (outputDecls.size() == 1 &&
        isa<ShapeType>(outputDecls.front().getType())) {
      // `outputShape` has the form `[N0, N1, ...]`
      ASSERT_STREAM(success.resultShapes.size() == 1,
                    << "expect a single output");
      ShapeAttr outputShape = success.resultShapes.front();

      // Case 0 (output decl is a shape)
      //   mo.reshape<() -> Sh0: !mosh.ape> : ... -> Sh0
      //
      // Create binding for the output shape parameter to output shape which
      // models ops like the following:
      //   kgen.param.decare Sh0 = <[N0, N1, ...]>
      bindings.push_back(ParamBinding(outputDecls.front(), outputShape));
    } else {
      llvm::DenseSet<ParamDeclAttr> outputDeclSet(outputDecls.begin(),
                                                  outputDecls.end());
      llvm::DenseSet<KGEN::ParamDeclRefAttr> processedDeclRefs;
      for (auto [i, res, outputShape] :
           llvm::enumerate(op->getResults(), success.resultShapes)) {
        // Case 1 (output decls are dimensions)
        //   mo.reshape<() -> D1, D2> : ... -> [D1, D2]
        //   mo.reshape<() -> D1> : ... -> [10, D1]
        //
        // Create binding for the output shape parameter to output dims which
        // models ops like the following:
        //   kgen.param.decare D1 = <N0>
        //   kgen.param.decare D2 = <add(N1, 10)>
        ShapeAttr annotatedOutShape =
            cast<TensorType>(res.getType()).getTypedShapeAttr();
        size_t returnedRank = outputShape.getRank();
        size_t annnotatedRank = annotatedOutShape.getRank();
        if (returnedRank != annnotatedRank) {
          std::stringstream ss;
          ss << "shape function result #" << i << " returned shape of rank "
             << returnedRank << ", but the type annotation has shape of rank "
             << annnotatedRank;
          return Error(ss.str());
        }
        // Given `outputShape` for
        //   mo.reshape<() -> D1> : ... -> [10, D1]
        // Generate
        //   D1 = <outputShape[1]>
        for (auto [axis, dim] :
             llvm::enumerate(annotatedOutShape.getValues())) {
          // Skip if `dim` is not a dimension parameter
          auto paramDim = dyn_cast<KGEN::ParamDeclRefAttr>(dim);
          if (!paramDim)
            continue;

          // Skip if already added binding to define this parameter.
          if (auto [_, inserted] = processedDeclRefs.insert(paramDim);
              !inserted)
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
        KGEN::ParamDeclareOp::create(builder, loc, binding.decl, binding.value);
    ops.push_back(declOp);
  }
  return ops;
}

struct PropagateShapes {
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
    unsigned bitwidth =
        mlir::ConstantIntRanges::getStorageBitwidth(attr.getType());

    if (isSignedType(attr.getType()))
      ranges.try_emplace(attr, mlir::ConstantIntRanges::fromSigned(
                                   APInt::getZero(bitwidth),
                                   APInt::getSignedMaxValue(bitwidth)));
    else
      ranges.try_emplace(
          attr, mlir::ConstantIntRanges::fromUnsigned(
                    APInt::getZero(bitwidth), APInt::getMaxValue(bitwidth)));
  }

  static bool isSignedType(Type type) {
    if (auto integer = dyn_cast<IntegerType>(type))
      return integer.getSignedness() !=
             IntegerType::SignednessSemantics::Unsigned;
    return isa<IndexType>(type);
  }

  static mlir::ConstantIntRanges getMaxRange(Type type) {
    unsigned width = mlir::ConstantIntRanges::getStorageBitwidth(type);
    APInt umin = APInt::getMinValue(width);
    APInt umax = APInt::getMaxValue(width);
    APInt smin = width != 0 ? APInt::getSignedMinValue(width) : umin;
    APInt smax = width != 0 ? APInt::getSignedMaxValue(width) : umax;
    return mlir::ConstantIntRanges{umin, umax, smin, smax};
  }

  mlir::ConstantIntRanges
  getRange(KGEN::POC opcode, ArrayRef<TypedAttr> operands, Type resultType) {
    // Get ranges for operands.
    auto inputs = llvm::map_to_vector(
        operands, [&](TypedAttr arg) { return getRange(arg); });

    // Evaluate the range of comparison operations.
    auto evaluateCmp = [&](mlir::intrange::CmpPredicate pred) {
      const mlir::ConstantIntRanges &lhs = inputs[0], &rhs = inputs[1];

      APInt min = APInt::getZero(1);
      APInt max = APInt::getAllOnes(1);

      std::optional<bool> truthValue =
          mlir::intrange::evaluatePred(pred, lhs, rhs);
      if (truthValue.has_value() && *truthValue)
        min = max;
      else if (truthValue.has_value() && !(*truthValue))
        max = min;

      return mlir::ConstantIntRanges::fromUnsigned(min, max);
    };

    switch (opcode) {
    case KGEN::POC::Add:
      return mlir::intrange::inferAdd(inputs);
    case KGEN::POC::Mul:
      return mlir::intrange::inferMul(inputs,
                                      mlir::intrange::OverflowFlags::None);
    case KGEN::POC::MulNoWrap:
      return mlir::intrange::inferMul(inputs,
                                      mlir::intrange::OverflowFlags::Nuw);
    case KGEN::POC::And:
      return mlir::intrange::inferAnd(inputs);
    case KGEN::POC::Or:
      return mlir::intrange::inferOr(inputs);
    case KGEN::POC::Xor:
      return mlir::intrange::inferXor(inputs);
    case KGEN::POC::Max:
      return isSignedType(resultType) ? mlir::intrange::inferMaxS(inputs)
                                      : mlir::intrange::inferMaxU(inputs);
    case KGEN::POC::Min:
      return isSignedType(resultType) ? mlir::intrange::inferMinS(inputs)
                                      : mlir::intrange::inferMinU(inputs);
    case KGEN::POC::Shl:
      return mlir::intrange::inferShl(inputs);
    case KGEN::POC::Shr:
      return isSignedType(resultType) ? mlir::intrange::inferShrS(inputs)
                                      : mlir::intrange::inferShrU(inputs);
    case KGEN::POC::Div:
      return isSignedType(resultType) ? mlir::intrange::inferDivS(inputs)
                                      : mlir::intrange::inferDivU(inputs);
    case KGEN::POC::Mod:
      return isSignedType(resultType) ? mlir::intrange::inferRemS(inputs)
                                      : mlir::intrange::inferRemU(inputs);
    case KGEN::POC::EQ:
      return evaluateCmp(mlir::intrange::CmpPredicate::eq);
    case KGEN::POC::LT:
      return isSignedType(operands[0].getType())
                 ? evaluateCmp(mlir::intrange::CmpPredicate::slt)
                 : evaluateCmp(mlir::intrange::CmpPredicate::ult);
    case KGEN::POC::LE:
      return isSignedType(operands[0].getType())
                 ? evaluateCmp(mlir::intrange::CmpPredicate::sle)
                 : evaluateCmp(mlir::intrange::CmpPredicate::ule);
    case KGEN::POC::Cond: {
      auto constant = inputs[0].getConstantValue();
      auto &trueCase = inputs[1];
      auto &falseCase = inputs[2];

      if (constant)
        return constant->isZero() ? falseCase : trueCase;

      return falseCase.rangeUnion(trueCase);
    }
    default:
      return getMaxRange(resultType);
    };
  }

  mlir::ConstantIntRanges getRange(TypedAttr attr) {
    if (!isa<IntegerType, IndexType>(attr.getType()))
      return mlir::ConstantIntRanges::maxRange(1);

    if (auto it = ranges.find(attr); it != ranges.end())
      return it->second;

    // Constants have trivially inferrable ranges.
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return mlir::ConstantIntRanges::constant(intAttr.getValue());

    // Bail out for attributes we don't recognize.
    auto exprAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr);
    if (!exprAttr)
      return getMaxRange(attr.getType());

    return getRange(exprAttr.getOpcode(), exprAttr.getOperands(),
                    exprAttr.getType());
  }

  static APInt getZero(Type type) {
    unsigned bitwidth = mlir::ConstantIntRanges::getStorageBitwidth(type);
    return APInt::getZero(bitwidth);
  }

  static mlir::ConstantIntRanges getZeroRange(TypedAttr attr) {
    return mlir::ConstantIntRanges::constant(getZero(attr.getType()));
  }

  bool isDefinitelyGreater(TypedAttr lhs, TypedAttr rhs) {
    auto pred = isSignedType(lhs.getType()) ? mlir::intrange::CmpPredicate::sge
                                            : mlir::intrange::CmpPredicate::uge;

    mlir::ConstantIntRanges lhsRange = getRange(lhs);
    mlir::ConstantIntRanges rhsRange = getRange(rhs);

    if (auto cmpResult =
            mlir::intrange::evaluatePred(pred, lhsRange, rhsRange)) {
      if (cmpResult.value())
        return true;
    }

    return false;
  }

  /// A conservative predicate that returns true iff `attr` is provable to be
  /// non-negative, and return false we can't claim anything.
  bool isDefinitelyNonNegative(TypedAttr attr) {
    auto pred = isSignedType(attr.getType())
                    ? mlir::intrange::CmpPredicate::sge
                    : mlir::intrange::CmpPredicate::uge;

    mlir::ConstantIntRanges range = getRange(attr);
    mlir::ConstantIntRanges zero = getZeroRange(attr);

    if (std::optional<bool> cmpResult =
            mlir::intrange::evaluatePred(pred, range, zero)) {
      if (cmpResult.value())
        return true;
    }

    return false;
  }

  /// This function serves as shape inference system's extension to KGEN folder
  /// on `POC::Cond`. It leverages things like range information to enable more
  /// expression folding, and applies it only to `cond(operands)`, assuming
  /// `operands` have been folded as much as possible.
  ///
  /// Return null if we can't apply any special folding.
  TypedAttr tryFoldCond(ArrayRef<TypedAttr> operands) {
    if (auto constant = getRange(operands[0]).getConstantValue())
      return constant->isZero() ? operands[2] : operands[1];
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

      if (isDefinitelyGreater(lhs, rhs))
        return lhs;

      if (isDefinitelyGreater(rhs, lhs))
        return rhs;
    } else if (opcode == KGEN::POC::Min) {
      if (isDefinitelyGreater(lhs, rhs))
        return rhs;

      if (isDefinitelyGreater(rhs, lhs))
        return lhs;
    } else if (opcode == KGEN::POC::LT || opcode == KGEN::POC::LE) {
      // Try more things, since we get this pattern from slice: lt(add(X, Y), X)
      //     lhs < rhs
      // <=> lhs - rhs < 0
      //
      // This works due to the symbolic optimizations which simplify add
      // opcodes.
      auto zero = IntegerAttr::get(lhs.getType(), 0);
      auto diff = KGEN::ParamOperatorAttr::getSub(lhs, rhs);
      auto i1Type = IntegerType::get(lhs.getContext(), 1);

      auto cmp = getRange(opcode, {diff, zero}, i1Type);

      if (auto truthValue = cmp.getConstantValue())
        return BoolAttr::get(lhs.getContext(), !truthValue->isZero());
    }

    return {};
  }

  TypedAttr tryRangeBasedFolding(TypedAttr attr) {
    auto integerType = dyn_cast<IntegerType>(attr.getType());
    if (!integerType)
      return {};

    std::optional<APInt> constant = getRange(attr).getConstantValue();
    if (!constant)
      return {};

    return IntegerAttr::get(integerType, constant.value());
  }

  /// This function serves as shape inference system's extension to KGEN
  /// folders. It leverages things like range information to enable more
  /// expression folding, and applies it recursively to all expression nested
  /// inside `attr`.
  TypedAttr tryFoldExprAttr(TypedAttr attr) {
    auto exprAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr);
    if (!exprAttr)
      return attr;

    if (auto value = tryRangeBasedFolding(attr))
      return value;

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
                    << "data dependent input index must be in bound");
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
          MOSH::ParamToValueOp::create(builder, loc, inputAttrs[idx]);
      Value newInputVal;
      if (isa<ShapeType>(inputAttr.getType())) {
        newInputVal = Shape::ToTensorOp::create(builder, loc, inputTensorTy,
                                                paramToValOp.getResult());
      } else {
        newInputVal = IndexToTensorOp::create(builder, loc, inputTensorTy,
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

    ASSERT_STREAM(!isa<InvokeShapeFuncOp>(op),
                  << "We do not expect InvokeShapeFuncOp to be in the IR "
                     "before PropagateShape!");
    op.setOutputParamDecls({});

    // The first new op must be processed next in the worklist, so we push the
    // new ops onto the worklist in reversed order.
    llvm::append_range(newOps, paramDeclOps);
    // Note that we also put `op` itself back onto the worklist, to be
    // re-visited after we visit all the new ops, since that's when we'll have
    // all the information needede to update the output shape of `op` (if
    // needed).
    worklist.push_back(op);
    for (Operation *newOp : llvm::reverse(newOps))
      worklist.push_back(newOp);

    return success();
  }

  LogicalResult propagateShape(GraphOp graphOp) {
    // Clear state variables (in case the pass instance is reused)
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
    llvm::for_each(graphOp.getInputParameters(), markIntParamDeclAsNonNegative);

    // TODO(#9008) inline inputs to `mo.assert`
    // A replacer to help replacing parameters in an op with its evaluated form.
    // TODO(#26115) we should switch to replacing manually via
    // `evaluator.getReboundAttribute(shapeAttr)` for each op, since we only
    // care about specific region of an op, and to avoid having 2 caches -- 1 in
    // replacer and 1 in evaluator.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](KGEN::ParamDeclRefAttr declRef) -> Attribute {
      return evaluator.getReboundAttribute(declRef);
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
            return failure();
          }
          // Otherwise the op has been shape-inferred, continue to process the
          // new ops that were pushed onto the worklist.
          continue;
        }

      } else if (auto shapeOfOp = dyn_cast<ShapeOfOp>(op)) {
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
            ASSERT_STREAM(outParamDecls.size() == 1 &&
                              ShapeAttr::isDimType(decl.getType()),
                          << "expect only 1 output dimension parameter");
            ImplicitLocOpBuilder builder(op->getLoc(), op);
            auto rankAttr =
                ShapeAttr::getDim(builder.getContext(), inputShape.getRank());
            auto declOp = KGEN::ParamDeclareOp::create(builder, decl, rankAttr);
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
        auto foldedRhsAttr = evaluator.getReboundAttribute(declOp.getValue());

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
            evaluator.setDeclBinding(declAttr, existingDeclRef);
            declOp.setValueAttr(existingDeclRef);

            // Else we remember this decl has this expression (if pure), and map
            // the parameter to itself since we don't want to inline it.
          } else {
            auto declRefAttr = KGEN::ParamDeclRefAttr::get(declAttr);
            if (isParamOperatorPure(paramOpAttr))
              exprToRegionToDeclRef[paramOpAttr][opParentRegion] = declRefAttr;
            evaluator.setDeclBinding(declAttr.getName(), declRefAttr);
          }

        } else {
          // The param expression can be nested, so we just record it as is.
          evaluator.setDeclBinding(declAttr, foldedRhsAttr);
        }

        // Emit compile time error for any `mo.assert false`
        if (StringAttr errMsg = tryGetAssertFalseErrMsg(foldedRhsAttr)) {
          op->emitError(errMsg.strref());
          return failure();
        }
      }

      if (auto paramFromValue = dyn_cast<MOSH::ParamFromValueOp>(op)) {
        StaticizeResult input = tryStaticize(paramFromValue.getValue());
        if (input.isError()) {
          op->emitError(input.getError().msg);
          return failure();
        }

        if (input.isSuccess()) {
          StaticizeResult::Success &success = input.getSuccess();

          OpBuilder builder(op);
          Location loc = op->getLoc();
          SmallVector<Operation *> newOps;
          for (ParamBinding &binding : success.bindings)
            newOps.push_back(binding.toOp(loc, builder));

          llvm::append_range(worklist, llvm::reverse(newOps));

          auto result = evaluator.getReboundAttribute(success.resultAttr);
          evaluator.setDeclBinding(paramFromValue.getParamDecl(), result);
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
    graphOp.walk([&](ShapeOfOp op) {
      ImplicitLocOpBuilder builder(op.getLoc(), op);
      TypedAttr inputShape = op.getInput().getType().getShapeAttr();
      auto shConst = MOSH::ParamToValueOp::create(builder, inputShape);
      auto toTensor = Shape::ToTensorOp::create(builder, op.getType(), shConst);
      op.replaceAllUsesWith(toTensor.getResult());
      op.erase();
      return WalkResult::advance();
    });

    // Normalize cross-graph call sites.
    normalizeCallOperandsToCallee(graphOp);

    return success();
  }

  PropagateShapes(KGEN::ParameterCollector::Analysis &paramCache)
      : paramCache{paramCache} {}

  KGEN::ParameterCollector::Analysis &paramCache;

  // State variables used by the pass
  DenseMap<KGEN::ParamOperatorAttr, size_t> paramAttrToDepth{};
  DenseMap<TypedAttr, mlir::ConstantIntRanges> ranges;
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
      exprToRegionToDeclRef{};
};

} // namespace

struct PropagateShapesPass
    : public MO::impl::PropagateShapesBase<PropagateShapesPass> {
  using Base::Base;

  void runOnOperation() override {
    auto &paramCache = getAnalysis<KGEN::ParameterCollector::Analysis>();

    PropagateShapes propagator(paramCache);
    if (failed(propagator.propagateShape(getOperation())))
      signalPassFailure();
  }
};
