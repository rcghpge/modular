//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/Interfaces/LoweringState.h"
#include "GraphCompiler/GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"

#include <numeric>

using namespace M;
using namespace M::MGP;

static Value popScalarToBuiltinType(OpBuilder &builder, Location loc, Value v) {
  auto type = cast<KGEN::SIMDType>(v.getType());
  auto maybeDtype = type.getResolvedDType();
  assert(maybeDtype && "unable to resolve dtype");

  auto dtype = *maybeDtype;

  MLIRContext *ctx = builder.getContext();

  auto resultType =
      dtype.isIndex()
          ? (Type)IndexType::get(ctx)
          : (Type)M::getEquivalentIntegerType(ctx, DType(dtype.getValue()));

  return builder.createOrFold<KGEN::POP::CastToBuiltinOp>(loc, resultType, v);
}

//===----------------------------------------------------------------------===//
// MOGraphLoweringState
//===----------------------------------------------------------------------===//

ErrorOr<ElementsAttr>
MOGraphLoweringState::lowerElementsAttr(ElementsAttr elementsAttr) {
  MLIRContext *ctx = getExecuteOpBuilder().getContext();

  // All attributes must have a minimum alignment of this value.
  uint64_t alignment = static_cast<uint64_t>(getMinimumCpuAlignment());

  auto hasAlignedAttr = dyn_cast<HasAlignedBytesInterface>(elementsAttr);
  if (hasAlignedAttr == nullptr)
    return Error("unsupported constant attribute type");
  auto origType = hasAlignedAttr.getAlignedBytesType();

  // Increase alignment to respect what the attribute requests.
  alignment = std::max(alignment, origType.getAlign());

  // Incorporate the alignment and size into an AlignedBytesType.
  auto finalType = AlignedBytesType::get(ctx, origType.getSize(), alignment);

  if (auto arrayAttr = dyn_cast<ArrayElementsAttr>(elementsAttr)) {
    // In the following we'll reuse the underlying array elements data as is,
    // ignoring its current alignment. The desired MEF alignment is instead
    // conveyed via the aligned_bytes type.
    static constexpr size_t kLargeDataThreshold = 512;
    if (arrayAttr.getRawData().size() < kLargeDataThreshold) {
      // Small attributes can remain inline.
      return AlignedBytesAttr::get(ctx, arrayAttr.getRawData(), alignment);
    } else {
      // Push large attributes to dense_resources.
      std::string name = "const_" + std::to_string(getExecuteNextConstId());
      ArrayRef<char> data(
          reinterpret_cast<const char *>(arrayAttr.getRawData().data()),
          arrayAttr.getRawData().size());
      return DenseResourceElementsAttr::get(
          finalType,
          DenseResourceElementsHandle::getManagerInterface(ctx).insert(
              name, mlir::UnmanagedAsmResourceBlob::allocateWithAlign(
                        data, alignment)));
    }
  }

  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(elementsAttr)) {
    // We'll not try to allocate a new blob. Instead we'll reuse the original
    // blob and rely on the type annotation to capture our desired alignment.
    return DenseResourceElementsAttr::get(finalType,
                                          resourceAttr.getRawHandle());
  }

  if (auto alignedAttr = dyn_cast<AlignedBytesAttr>(elementsAttr))
    return AlignedBytesAttr::get(ctx, alignedAttr.getData(), alignment);

  llvm_unreachable(
      "missing case for implementation of HasAlignedBytesInterface");
}

TypedValue<MGP::TensorSpecType>
MOGraphLoweringState::getShape(Location loc, Value loweredTensor,
                               DeviceRefAttr device) {
  OpBuilder &builder = getExecuteOpBuilder();
  return getBufferBuilder(device)->tensorExtractTensorSpec(builder, loc,
                                                           loweredTensor);
}

MGP::TensorSpecGetDimOp MOGraphLoweringState::getDim(Location loc,
                                                     Value loweredTensor,
                                                     uint64_t axis,
                                                     DeviceRefAttr device) {
  OpBuilder &builder = getExecuteOpBuilder();
  // NOTE: We could use getShape here, however usually we'll also require the
  // loweredTensor's buffer in which case we'll be emitting a full
  // tensor.extract op anyway.
  TypedValue<MGP::TensorSpecType> tensorSpec =
      getBufferBuilder(device)->tensorExtractTensorSpec(builder, loc,
                                                        loweredTensor);
  return MGP::TensorSpecGetDimOp::create(builder, loc, /*spec=*/tensorSpec,
                                         /*axis=*/axis);
}

SmallVector<Value> MOGraphLoweringState::getDynamicDims(Location loc,
                                                        Value loweredTensor,
                                                        DeviceRefAttr device) {
  OpBuilder &builder = getExecuteOpBuilder();
  // NOTE: We could use getShape here, however usually we'll also require the
  // loweredTensor's buffer in which case we'll be emitting a full
  // tensor.extract op anyway.
  TypedValue<MGP::TensorSpecType> tensorSpec =
      getBufferBuilder(device)->tensorExtractTensorSpec(builder, loc,
                                                        loweredTensor);
  TensorSpecType specType = tensorSpec.getType();
  SmallVector<Value> dynDims;

  for (size_t i = 0; i < specType.getRank(); ++i)
    if (specType.isDynamicDim(i))
      dynDims.push_back(
          MGP::TensorSpecGetDimOp::create(builder, loc, tensorSpec, i));

  return dynDims;
}

static Value assertSymbolicDimensionMatchesPreviousValue(
    OpBuilder &builder, Location loc, Value previousValue, Value newValue,
    KGEN::ParamDeclRefAttr symbolicDimension, unsigned argumentIndex) {

  FailureOr<Type> targetType = previousValue.getType();
  // TODO: cast?
  Value isEqual = KGEN::POP::CmpOp::create(builder, loc, KGEN::CmpPredicate::EQ,
                                           previousValue, newValue);

  isEqual = popScalarToBuiltinType(builder, loc, isEqual);

  auto errorMessage = llvm::formatv(
      "symbolic dimension '{}' for input {} does not match prior uses "
      "of that dimension",
      symbolicDimension.getName().strref(), argumentIndex);

  return MGP::AssertOp::create(builder, loc, isEqual,
                               builder.getStringAttr(errorMessage));
}

ErrorOr<MOGraphLoweringState::ArgLoweringResult>
MOGraphLoweringState::lowerArg(Region *parentRegion, BlockArgument inArg,
                               BlockArgument outArg) {
  bool hasParams = false;
  Location loc = inArg.getLoc();

  auto isSubgraphRegion = [](Region *region) {
    if (auto graph = region->getParentOfType<MO::GraphOp>())
      return graph.getIsSubgraph();
    return false;
  };

  Value loweredArg = outArg;
  if (auto argType = MO::dynCastFromTensorLike(inArg.getType())) {
    DeviceRefAttr argDevice = MO::getDeviceOrDefault(argType);
    if (argType.hasStaticShape()) {
      // No parameters to deal with.
      /* no-op */
    } else if (auto inShapeParam =
                   dyn_cast<KGEN::ParamDeclRefAttr>(argType.getShapeAttr())) {
      // Argument has dynamic rank. The parameter needs to be represented by
      // the runtime MGP 'shape' of the tensor, ie a tensor_spec with unknown
      // dtype. However, we only need to capture the mapping for the first
      // occurrence of the parameter.
      hasParams = true;
      if (!params.lookupLowered(parentRegion, inShapeParam))
        params.emplaceLowered(parentRegion, inShapeParam,
                              getShape(loc, outArg, argDevice));
    } else {
      // Argument has at least one dynamic dimension. The parameters need to
      // be represented by the corresponding runtime dimensions. Since
      // parameters may be shared we only update the parameter map for the
      // first occurrence of each parameter. However, since dimensions are
      // extract as a group, we always do the extraction, and rely on DCE to
      // cleanup if all the parameters turn out to have already been mapped.
      hasParams = true;

      SmallVector<Value> dynDims = getDynamicDims(loc, outArg, argDevice);
      // getDynamicDims returns indexes only for dimensions
      // which are unknown at compile time. So build the corresponding
      // vector of parameters. Note we do not support arbitrary expressions in
      // the MO shape signature.
      SmallVector<KGEN::ParamDeclRefAttr> inDimParams;
      for (TypedAttr inDim : argType.getDims()) {
        if (auto inDimParam = dyn_cast<KGEN::ParamDeclRefAttr>(inDim))
          inDimParams.emplace_back(inDimParam);
      }
      if (dynDims.size() != inDimParams.size()) {
        // TODO(#14531) handle param expr
        return Error("Expressions in parameter arguments are not supported");
      }

      OpBuilder &builder = getExecuteOpBuilder();
      SmallVector<Value> assertChains;
      for (auto [inDimParam, outDim] : llvm::zip_equal(inDimParams, dynDims)) {
        // Only update the parameter mapping for the first parameter
        // occurrence.
        auto existingValue = params.lookupLowered(parentRegion, inDimParam);
        auto loweredValue = indexToDimension(builder, loc, outDim);

        if (existingValue && !isSubgraphRegion(parentRegion)) {
          // If the parameter has been lowered previously, then assert that the
          // current instance is equal to the previously lowered value.
          // This check is only needed for entry-point graphs, as subgraphs
          // will inherit correctness from their caller.
          auto assert = assertSymbolicDimensionMatchesPreviousValue(
              builder, loc, existingValue, loweredValue, inDimParam,
              inArg.getArgNumber());

          assertChains.push_back(assert);
        } else {
          // Otherwise lower the dimension value and record the lowering.
          params.emplaceLowered(parentRegion, inDimParam, loweredValue);
        }
      }

      // If any assertions were generated, guard the outArg tensor value by
      // those assertions.
      if (!assertChains.empty()) {
        auto consolidatedChains =
            GRT::ChainCreateOp::create(builder, loc, assertChains);

        auto guarded =
            MGP::GuardOp::create(builder, loc, consolidatedChains, outArg);

        loweredArg = guarded.getResult(0);
      }
    }
  }
  return ArgLoweringResult{hasParams, loweredArg};
}

ErrorOr<Value> MOGraphLoweringState::lowerExpression(Region *parentRegion,
                                                     Location loc,
                                                     TypedAttr attr) {
  auto newValOr = params.lowerExpression(
      getExecuteOpBuilder(), loc, parentRegion, attr,
      [this](auto &&...args) { return this->lowerExpressionImpl(args...); });

  if (newValOr.isError())
    return newValOr.takeError();
  auto newVal = std::move(*newValOr);

  // If we got a placeholder, we need to remember it, in case we map an SSA
  // value to it.
  if (auto placeholder = newVal.getDefiningOp<ParamPlaceholderOp>())
    placeholderMap.try_emplace(placeholder, SmallVector<Value>{});
  return newVal;
}

ErrorOrSuccess MOGraphLoweringState::lowerParamDecl(
    Region *parentRegion, KGEN::ParamDeclAttr declAttr, Value value) {
  Value maybePlaceholder = params.lowerParamDecl(parentRegion, declAttr, value);

  if (maybePlaceholder) {
    // If we just resolved a placeholder, we need to make sure no one will
    // ever map to it in the future.
    if (auto it = placeholderMap.find(maybePlaceholder);
        it != placeholderMap.end()) {
      for (Value key : it->second) {
        execute.loweredValues.erase(key);
        execute.loweredValues.try_emplace(key, value);
      }
      placeholderMap.erase(it);
    }

    // We also need to replace all its existing uses.
    maybePlaceholder.replaceAllUsesWith(value);
    maybePlaceholder.getDefiningOp<ParamPlaceholderOp>().erase();
  }
  return success();
}

ErrorOr<SmallVector<Value>> MOGraphLoweringState::lowerShapeAttrDynDims(
    Region *parentRegion, MOSH::ShapeAttr shapeAttr, Location loc) {
  SmallVector<Value> runtimeDims;
  for (TypedAttr dim : shapeAttr.getValues()) {
    if (!MOSH::ShapeAttr::getStaticDim(dim).has_value()) {
      auto vOr = lowerDimensionExpression(params, getExecuteOpBuilder(), loc,
                                          parentRegion, dim);
      if (vOr.isError())
        return vOr.takeError();
      runtimeDims.emplace_back(vOr.takeValue());
    }
  }
  return runtimeDims;
}

/// Build a sequence of binary operations reducing the given set of values
/// to a single value. This is used to convert variadic KGEN::POC ops to
/// the binary op implementing them.
template <typename Op>
static ErrorOr<Value> buildVariadicOp(OpBuilder &builder, Location loc,
                                      ValueRange operands, Type resultType) {
  if (operands.empty())
    return Error("Variadic op without operands");

  Value start = operands.front();
  ValueRange rest = operands.drop_front();

  auto createBinop = [&](Value lhs, Value rhs) {
    return Op::create(builder, loc, resultType, lhs, rhs);
  };

  return std::accumulate(rest.begin(), rest.end(), start, createBinop);
}

/// Convert the given KGEN::POC operation to the equivalent operation in
/// in the POP dialect. This function assumes that input/output types
/// are !kgen.scalar values.
static ErrorOr<Value> pocToMopInteger(KGEN::POC poc, OpBuilder &builder,
                                      Location loc, ValueRange operands,
                                      Type resultType) {
  auto getCmpPredicate = [](KGEN::POC poc) -> KGEN::CmpPredicate {
    switch (poc) {
    case KGEN::POC::EQ:
      return KGEN::CmpPredicate::EQ;
    case KGEN::POC::LT:
      return KGEN::CmpPredicate::LT;
    case KGEN::POC::LE:
      return KGEN::CmpPredicate::LE;
    default:
      llvm_unreachable("unrecognized predicate");
    }
  };

  switch (poc) {
  // All comparisons lower to `pop.cmp`
  case KGEN::POC::EQ:
  case KGEN::POC::LT:
  case KGEN::POC::LE:
    return KGEN::POP::CmpOp::create(builder, loc, resultType,
                                    getCmpPredicate(poc), operands[0],
                                    operands[1]);

  // All potentially variadic ops
  case KGEN::POC::Add:
    return buildVariadicOp<KGEN::POP::AddOp>(builder, loc, operands,
                                             resultType);
  case KGEN::POC::Div:
    return buildVariadicOp<KGEN::POP::DivOp>(builder, loc, operands,
                                             resultType);
  case KGEN::POC::Max:
    return buildVariadicOp<KGEN::POP::MaxOp>(builder, loc, operands,
                                             resultType);
  case KGEN::POC::Min:
    return buildVariadicOp<KGEN::POP::MinOp>(builder, loc, operands,
                                             resultType);
  case KGEN::POC::Mul:
  case KGEN::POC::MulNoWrap:
    return buildVariadicOp<KGEN::POP::MulOp>(builder, loc, operands,
                                             resultType);
  case KGEN::POC::And:
    return buildVariadicOp<KGEN::POP::SIMDAndOp>(builder, loc, operands,
                                                 resultType);
  case KGEN::POC::Or:
    return buildVariadicOp<KGEN::POP::SIMDOrOp>(builder, loc, operands,
                                                resultType);
  case KGEN::POC::Xor:
    return buildVariadicOp<KGEN::POP::SIMDXOrOp>(builder, loc, operands,
                                                 resultType);

  // Remaining ops
  case KGEN::POC::Shl:
    return KGEN::POP::ShlOp::create(builder, loc, resultType, operands);
  case KGEN::POC::Shr:
    return KGEN::POP::ShrOp::create(builder, loc, resultType, operands);
  case KGEN::POC::Mod:
    return KGEN::POP::RemOp::create(builder, loc, resultType, operands);
  case KGEN::POC::Cond: {
    auto condition = operands[0];
    auto trueValue = operands[1];
    auto falseValue = operands[2];
    return KGEN::POP::SIMDSelectOp::create(builder, loc, resultType, condition,
                                           trueValue, falseValue);
  }
  default:
    return Error(Twine("unsupported POC '") + stringifyPOC(poc) + "'");
  }
}

ErrorOr<Value>
MOGraphLoweringState::lowerParamOperator(OpBuilder &builder, Location loc,
                                         KGEN::ParamOperatorAttr opAttr,
                                         ValueRange operands) {
  auto resultType = opAttr.getType();

  ASSERT_STREAM(isa<KGEN::SIMDType>(resultType),
                "must be a simd parameter expression");

  SmallVector<Value> popOperands;
  popOperands.reserve(operands.size());

  // Convert operands to equivalent !kgen.scalar type
  for (auto operand : operands) {
    ASSERT_STREAM(isa<KGEN::SIMDType>(operand.getType()),
                  "must be a simd parameter expression");
    popOperands.push_back(operand);
  }

  // Create the implementing pop dialect operation.
  auto maybeResult = pocToMopInteger(opAttr.getOpcode(), builder, loc,
                                     popOperands, resultType);

  if (maybeResult.isError())
    return maybeResult;

  return maybeResult.takeValue();
}

Operation *MOGraphLoweringState::lowerKGENConstantAttr(OpBuilder &builder,
                                                       Location loc,
                                                       TypedAttr value) {
  return KGEN::ParamConstantOp::create(builder, loc, value);
}

/// sequenced assert lowering helper functions
namespace {

bool followOperands(Operation *op) {
  // We uses POP::And/Or/XOrOp instead of index::And/Or/XOr so that we can
  // handle more than IndexType.
  return isa<KGEN::POP::POPDialect>(op->getDialect());
}

DenseSet<Operation *> getAncestors(Operation *root) {

  DenseSet<Operation *> ancestors;
  SmallVector<Operation *> worklist;

  worklist.push_back(root);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    if (!op || !followOperands(op))
      continue;

    ancestors.insert(op);

    for (Value val : op->getOperands()) {
      Operation *defOp = val.getDefiningOp();

      if (!ancestors.contains(defOp))
        worklist.push_back(defOp);
    }
  }

  return ancestors;
}

SmallVector<OpOperand *>
getBoundaryOpOperands(Operation *root,
                      const llvm::DenseSet<Operation *> &prevAncestors) {

  DenseSet<OpOperand *> boundaryOpOperands;
  SmallVector<Operation *> worklist;

  worklist.push_back(root);

  while (!worklist.empty()) {

    Operation *op = worklist.pop_back_val();

    if (!op || !followOperands(op) || prevAncestors.contains(op))
      continue;

    for (OpOperand &opOperand : op->getOpOperands()) {
      auto defOp = opOperand.get().getDefiningOp();
      if (!defOp)
        continue;

      const bool isTrivial = defOp->hasTrait<OpTrait::ConstantLike>() ||
                             isa<mlir::CastOpInterface>(defOp);

      if (prevAncestors.contains(defOp) && !isTrivial)
        boundaryOpOperands.insert(&opOperand);
      else
        worklist.push_back(defOp);
    }
  }

  return {boundaryOpOperands.begin(), boundaryOpOperands.end()};
}

/// Given the operand chain and condition to an assertion gather all
/// expressions rooted at the current condition and the condition from the
/// previous producing assertion. The values (OpOperands really) that cross
/// from the previous assertion's tree to the current assertion's tree.
SmallVector<OpOperand *> getSequencedAssertBoundaryOperands(
    Value cond, const SmallVector<Operation *> &parentAsserts) {

  DenseSet<Operation *> allAncestors;

  for (auto parentAssert : parentAsserts) {
    auto prevAssertCondRoot =
        cast<MGP::AssertOp>(parentAssert).getCond().getDefiningOp();

    auto prevAncestors = getAncestors(prevAssertCondRoot);

    allAncestors.insert(prevAncestors.begin(), prevAncestors.end());
  }

  return getBoundaryOpOperands(cond.getDefiningOp(), allAncestors);
}

SmallVector<Operation *>
getParentAsserts(Operation *currentAssert,
                 const DenseMap<Operation *, Operation *> &parentMap) {
  SmallVector<Operation *> parents = {currentAssert};

  auto it = parentMap.find(currentAssert);
  bool hasParent = it != parentMap.end();

  while (hasParent) {
    currentAssert = it->second;
    parents.push_back(currentAssert);
    it = parentMap.find(currentAssert);
    hasParent = it != parentMap.end();
  }

  return parents;
}

} // namespace

ErrorOr<Value> MOGraphLoweringState::lowerDimensionExpression(
    ParameterLoweringState &state, OpBuilder &builder, Location loc,
    Region *parentRegion, TypedAttr dim) {
  auto dimValOr = state.lowerExpression(
      builder, loc, parentRegion, dim,
      [this](auto &&...args) { return this->lowerExpressionImpl(args...); });

  if (failed(dimValOr))
    return dimValOr;

  return dimensionToIndex(builder, loc, dimValOr.takeValue());
}

ErrorOr<Value> MOGraphLoweringState::lowerExpressionImpl(
    ParameterLoweringState &state, OpBuilder &builder, Location loc,
    Region *parentRegion, TypedAttr attr, ValueRange operands) {
  Operation *newOp;
  if (isa<IntegerAttr, KGEN::SIMDAttr>(attr)) {
    newOp = lowerKGENConstantAttr(builder, loc, attr);
  } else if (auto cast = dyn_cast<KGEN::CastFromBuiltinAttr>(attr)) {
    ErrorOr<Value> castFrom = lowerExpression(parentRegion, loc, cast.getArg());
    if (failed(castFrom))
      return castFrom;
    newOp = KGEN::POP::CastFromBuiltinOp::create(builder, loc, cast.getType(),
                                                 castFrom.takeValue());
  } else if (auto opAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr)) {
    return lowerParamOperator(builder, loc, opAttr, operands);
  } else if (auto opAttr = dyn_cast<KGEN::MLIROpAttr>(attr)) {
    StringAttr opName = opAttr.getName();
    if (opName != MO::AssertOp::getOperationName()) {
      return Error(Twine("cannot lower mlir operation '") + opName.strref() +
                   "' in parameter expression");
    }

    auto [cond, maybeParentAssert] = [&]() -> std::pair<Value, Operation *> {
      bool hasChain = isa<MEF::ChainType>(operands[0].getType());

      if (hasChain) {
        Value chain = operands[0];
        Value cond = operands[1];

        ASSERT_STREAM(
            isa<MGP::AssertOp>(chain.getDefiningOp()),
            << "Encountered a mo.assert with a chain that was not produced by "
               "another mo.assert!");

        SmallVector<Operation *> parentAsserts =
            getParentAsserts(chain.getDefiningOp(), execute.parentAssert);

        // collects the values (OpOperands really) that span from the tree
        // of expressions forming condition for the assertion we're looking
        // at and the assertion that produced the chain
        auto bOperands =
            getSequencedAssertBoundaryOperands(cond, parentAsserts);

        // This logic will materialize guards only in the case where there were
        // shared values between the expression tree rooted at the previous
        // assertion and the current
        //
        // This is desirable as the mgp.guard op doesn't have a Mojo
        // implementation (and consequently KGEN lowering), blocking the
        // possibility of sequence fusion
        if (!bOperands.empty()) {
          SmallVector<Value> v;
          for (OpOperand *opOperand : bOperands) {
            v.push_back(opOperand->get());
          }

          ValueRange guardedValRange{v};

          auto guardOp = MGP::GuardOp::create(
              builder, loc, guardedValRange.getTypes(), chain, guardedValRange);

          for (auto &&[guarded, original] :
               llvm::zip(guardOp.getResults(), bOperands)) {
            original->assign(guarded);
          }
        }

        return {cond, chain.getDefiningOp()};
      } else {
        return {operands[0], nullptr};
      }
    }();

    if (cond.getType() != builder.getIntegerType(1)) {
      assert(KGEN::isScalarOf<KGEN::KGENDType::kBool>(cond.getType()));
      cond = builder.createOrFold<KGEN::POP::CastToBuiltinOp>(
          loc, builder.getIntegerType(1), cond);
    }
    // We must be dealing with an assert.
    MGP::AssertOp assertOp = MGP::AssertOp::create(
        builder, loc, ValueRange{cond}, opAttr.getAttrs().getValue());

    newOp = assertOp;
    if (maybeParentAssert)
      execute.parentAssert[newOp] = maybeParentAssert;
  } else if (auto shapeAttr = dyn_cast<MOSH::ShapeAttr>(attr)) {
    // Lower dimensions recursively.
    SmallVector<Value> dimVals;
    for (TypedAttr dim : shapeAttr.getValues()) {
      auto dimValOr =
          lowerDimensionExpression(state, builder, loc, parentRegion, dim);

      if (dimValOr.isError())
        return dimValOr.takeError();
      dimVals.emplace_back(std::move(*dimValOr));
    }

    // There is no runtime shape type in MGP, so we map shape attributes to
    // a fully dynamic spec.
    SmallVector<std::optional<uint64_t>> dummyShape(dimVals.size());
    auto specTy =
        MGP::TensorSpecType::get(loc.getContext(), dummyShape, DType::si64);
    auto specOp =
        MGP::TensorSpecCreateOp::create(builder, loc, specTy, dimVals);
    return specOp.getResult();
  } else {
    return Error("cannot lower parameter expression");
  }

  return newOp->getResult(0);
}

/// Helper function to create AsyncPackBorrowOp with an async unpack from a
/// source value. This is used for operations that need to borrow a
/// buffer/tensor and track its async reference.
static Value createAsyncPackBorrowWithUnpack(OpBuilder &builder, Location loc,
                                             Value value, Value sourceValue) {
  // The async value ref must be extracted from the async-backed value (the
  // `!mogg.async<T>` runtime slot), not its unpacked `T` projection. If the
  // source is a `mogg.async.unpack`, look through it to the async operand.
  if (auto unpackOp = sourceValue.getDefiningOp<MOGG::AsyncUnpackOp>())
    sourceValue = unpackOp.getInput();
  // The result type (an AnyAsyncValueRef handle) is supplied by the op builder.
  auto asyncUnpackAnyAsyncValueRefOp =
      MOGG::AsyncUnpackAnyAsyncValueRefOp::create(builder, loc, sourceValue);
  return MOGG::AsyncPackBorrowOp::create(builder, loc, value,
                                         asyncUnpackAnyAsyncValueRefOp);
}

/// Returns the element type of a `!mogg.async<T>`, or the type itself if it is
/// not an async type. Region body arguments are async-wrapped, so checks on the
/// underlying value type (e.g. chain detection) must unwrap first.
static Type getAsyncElementOrSelf(Type type) {
  if (auto asyncType = dyn_cast<MOGG::AsyncType>(type))
    return asyncType.getElementType();
  return type;
}

/// Rewires the consumers of an async value `asyncValue` to its synchronous
/// `projection` (an unpack or a pass-through reinterpret). The projection op's
/// own use of `asyncValue` is left untouched, as are the async-consuming ops
/// (`mogg.async.unpack[.AnyAsyncValueRef]`) that must keep operating on the
/// async value directly.
static void rewireAsyncConsumersToProjection(Value asyncValue,
                                             Value projection) {
  Operation *projectionOp = projection.getDefiningOp();
  asyncValue.replaceUsesWithIf(projection, [&](mlir::OpOperand &operand) {
    Operation *owner = operand.getOwner();
    return owner != projectionOp &&
           !isa<MOGG::AsyncUnpackOp, MOGG::AsyncUnpackAnyAsyncValueRefOp>(
               owner);
  });
}

void MOGraphLoweringState::lowerMOGGABI(Operation *kernelOp,
                                        const mlir::BitVector &shouldUnpack,
                                        const mlir::BitVector &shouldPack) {
  auto kernel = cast<MOGG::GenericCompiledRegionOp>(kernelOp);
  OpBuilder builder{&kernel.getBody().front(),
                    kernel.getBody().front().begin()};
  auto loc = kernel->getLoc();
  Operation *terminator = kernel.getBody().front().getTerminator();

  // Chain inputs should never be used by any ops. Replace them with a dummy
  // chain. Block arguments are `!mogg.async<T>`, so unwrap before checking.
  for (auto &&arg : kernel.getBody().front().getArguments()) {
    if (isa<MEF::ChainType>(getAsyncElementOrSelf(arg.getType()))) {
      arg.replaceAllUsesWith(
          MOGG::ChainDummyOp::create(builder, loc).getResult());
    }
  }

  // Unpack all inputs that should be unpacked first, so the body ops consume
  // the unpacked `T` values. This must happen before packing outputs below,
  // which inspects the (now unpacked) operands of the body ops.
  builder.setInsertionPointToStart(&kernel.getBody().front());
  for (auto v : kernel.getBody().getArguments()) {
    // Chains are left as `!mogg.async<chain>` (already replaced with a dummy
    // chain above); nothing consumes them directly.
    if (isa<MEF::ChainType>(getAsyncElementOrSelf(v.getType())))
      continue;

    if (shouldUnpack.test(v.getArgNumber())) {
      // Project the `!mogg.async<T>` argument to `T` and rewire consumers.
      auto unpackOp = MOGG::AsyncUnpackOp::create(builder, loc, v);
      rewireAsyncConsumersToProjection(v, unpackOp);
    } else {
      // This argument is deliberately not unpacked: the consuming op wants the
      // raw async slot as a plain `T` value (e.g. the context for
      // `mgp.buffer.get_cached`). Insert a pass-through reinterpret to a `T`
      // value (its Mojo value remains the runtime slot) and rewire consumers,
      // leaving the block argument as `!mogg.async<T>`. Keeping the argument
      // async lets a single argument feed both a raw-slot consumer (via this
      // reinterpret) and a consumer that unpacks it, which can happen after
      // sequence fusion merges regions.
      auto reinterpret = MOGG::ToMGPFromOpaqueOp::create(
          builder, loc, getAsyncElementOrSelf(v.getType()), v);
      rewireAsyncConsumersToProjection(v, reinterpret);
    }
  }

  builder.setInsertionPoint(terminator);
  // Track chain operands to create AsyncReadyOp AFTER all other pack ops.
  // This ensures downstream consumers don't start before all outputs are ready.
  SmallVector<std::pair<size_t, Value>> deferredChainOps;

  for (auto &&[i, value] : llvm::enumerate(terminator->getOperands())) {
    Value newValue;
    if (isa<MEF::ChainType>(value.getType())) {
      if (shouldPack[i])
        deferredChainOps.emplace_back(i, value);
      continue;
    } else if (shouldPack.test(i)) {
      SmallVector<Value, 2> packOperands{value};
      if (isa<MGP::DeviceBufferType>(value.getType())) {
        if (auto mgpBufferAllocOp = mlir::dyn_cast_or_null<MGP::BufferAllocOp>(
                value.getDefiningOp()))
          packOperands.push_back(mgpBufferAllocOp.getDeviceContext());
      }
      newValue = MOGG::AsyncPackOp::create(
          builder, loc, MOGG::AsyncType::get(value.getType()), packOperands);
    } else if (value.getDefiningOp<MGP::BufferConstantOp>()) {
      newValue = MOGG::AsyncPackBorrowOp::create(
          builder, loc, value,
          MOGG::AsyncPlaceholderAnyAsyncValueRefOp::create(builder, loc));
    } else if (auto bufferGetCachedOp =
                   value.getDefiningOp<MGP::BufferGetCachedOp>()) {
      if (bufferGetCachedOp.getOptionalMemStorageHandle().empty()) {
        auto memStorageType = KGEN::PointerType::get(
            KGEN::PointerType::get(KGEN::NoneType::get(builder.getContext())));
        auto oldBufferGetCachedOp = bufferGetCachedOp;
        bufferGetCachedOp = MGP::BufferGetCachedOp::create(
            builder, loc, {value.getType(), memStorageType},
            bufferGetCachedOp->getOperands(), bufferGetCachedOp->getAttrs());
        value.replaceAllUsesWith(bufferGetCachedOp.getOutBuffer());
        value = bufferGetCachedOp.getOutBuffer();
        oldBufferGetCachedOp->erase();
      }
      newValue = MOGG::AsyncPackBorrowOp::create(
          builder, loc, bufferGetCachedOp.getOutBuffer(),
          bufferGetCachedOp.getMemStorageHandle());
    } else if (auto tensorExtractBufferOp =
                   value.getDefiningOp<MGP::TensorExtractBufferOp>()) {
      newValue = createAsyncPackBorrowWithUnpack(
          builder, loc, value, tensorExtractBufferOp.getTensor());
    } else if (auto tensorCreateOp =
                   value.getDefiningOp<MGP::TensorCreateOp>()) {
      newValue = createAsyncPackBorrowWithUnpack(builder, loc, value,
                                                 tensorCreateOp.getBuffer());
    } else if (auto bufferSliceOp = value.getDefiningOp<MGP::BufferSliceOp>()) {
      newValue = createAsyncPackBorrowWithUnpack(builder, loc, value,
                                                 bufferSliceOp.getInBuffer());
    } else if (auto bulkSliceOp =
                   value.getDefiningOp<MGP::BufferBulkSliceOp>()) {
      newValue = createAsyncPackBorrowWithUnpack(builder, loc, value,
                                                 bulkSliceOp.getBaseBuffer());
    } else if (auto tensorSliceOp = value.getDefiningOp<MGP::TensorSliceOp>()) {
      newValue = createAsyncPackBorrowWithUnpack(builder, loc, value,
                                                 tensorSliceOp.getInput());
    }

    if (newValue) {
      value.replaceUsesWithIf(newValue, [&](OpOperand &use) {
        return use.getOwner() == terminator;
      });

      // If the value being replaced is a chain, be sure to drop it in a
      // ChainSinkOp to avoid it being DCE-ed if it's the only output.
      if (isa<MEF::ChainType>(value.getType()) &&
          !isa<MOGG::AsyncReadyOp>(value.getDefiningOp()))
        MOGG::ChainSinkOp::create(builder, loc, value);
    }
  }

  // Create AsyncReadyOp for chain outputs AFTER all other pack ops.
  // This ensures downstream consumers don't start until all outputs are packed.
  for (auto &&[i, value] : deferredChainOps) {
    Value newValue = MOGG::AsyncReadyOp::create(builder, loc);
    value.replaceUsesWithIf(
        newValue, [&](OpOperand &use) { return use.getOwner() == terminator; });

    // If the value being replaced is a chain, be sure to drop it in a
    // ChainSinkOp to avoid it being DCE-ed if it's the only output.
    if (!isa<MOGG::AsyncReadyOp>(value.getDefiningOp()))
      MOGG::ChainSinkOp::create(builder, loc, value);
  }
}

//===----------------------------------------------------------------------===//
// MOOpLoweringState
//===----------------------------------------------------------------------===//

SmallVector<Value> MOOpLoweringState::getLoweredOperands() {
  SmallVector<Value> loweredOperands;
  for (Value operand : opBeingLowered->getOperands()) {
    Value loweredInput = loweringState.lookupExecuteLoweredValue(operand);

    if (isa<MO::TensorType>(operand.getType()) ||
        isa<MO::BufferType>(operand.getType())) {
      BufferizedTensor bufferizedInput = getBufferizedInput(operand);
      loweredOperands.push_back(bufferizedInput.buffer);
    } else {
      loweredOperands.push_back(loweredInput);
    }
  }
  return loweredOperands;
}

Value MOOpLoweringState::getGuardChain(Value chain) {
  SmallVector<Value> inputChains;
  for (Value operand : opBeingLowered->getOperands()) {
    if (Value c = loweringState.getChainGuardingValueOrNull(operand))
      inputChains.push_back(c);
  }
  if (!inputChains.empty()) {
    inputChains.push_back(chain);
    chain = create<GRT::ChainCreateOp>(inputChains).getResult();
  }
  return chain;
}

FailureOr<SmallVector<MOOutputStorage>>
MOOpLoweringState::createOutputStorage(Value inChain) {
  FailureOr<SmallVector<TypedValue<MGP::TensorSpecType>>> outSpecsOr =
      createOutputTensorSpecs();
  if (failed(outSpecsOr))
    return failure();
  return createOutputStorage(inChain, outSpecsOr.value());
}

FailureOr<SmallVector<MOOutputStorage>> MOOpLoweringState::createOutputStorage(
    Value inChain, ArrayRef<TypedValue<MGP::TensorSpecType>> outSpecs) {
  SmallVector<BufferizedTensor> bufferizedTensors;
  bufferizedTensors.reserve(outSpecs.size());

  // We only create the mgp storage for the pre-allocated tensors. For those
  // have in-place allocation, we skip them.
  DenseSet<int> inplaceAllocIndicesSet;
  if (auto kernel = dyn_cast<MOGG::KernelOp>(opBeingLowered)) {
    if (auto allocIndicesAttr = kernel.getAllocIndicesAttr()) {
      for (auto attr : allocIndicesAttr) {
        inplaceAllocIndicesSet.insert(cast<mlir::IntegerAttr>(attr).getInt());
      }
    }
  }

  SmallVector<Value> results;
  for (auto [idx, res] : llvm::enumerate(opBeingLowered->getResults())) {
    if (!inplaceAllocIndicesSet.contains(idx) &&
        !isa<MO::ChainType, MO::OpaqueType>(res.getType()))
      results.push_back(opBeingLowered->getResult(idx));
  }

  for (auto [spec, result] : llvm::zip(outSpecs, results)) {
    auto bufOr = createBuffer(inChain, spec, result.getType());
    if (failed(bufOr))
      return failure();
    bufferizedTensors.emplace_back(*bufOr);
  }
  return bufferizedTensors;
}

void MOOpLoweringState::createOutputs(
    Value inChain, SmallVector<MOOutputStorage> &loweredOutputsStorage) {
  if (auto kernel = dyn_cast<MOGG::KernelOp>(opBeingLowered)) {
    DenseSet<int> inplaceAllocIndicesSet;
    if (auto allocIndicesAttr = kernel.getAllocIndicesAttr()) {
      for (auto attr : allocIndicesAttr) {
        inplaceAllocIndicesSet.insert(cast<mlir::IntegerAttr>(attr).getInt());
      }
    }
    int index = 0;
    for (const auto &[idx, res] : llvm::enumerate(kernel.getResults())) {
      // ignore chain and opaque types as there are currently special carveouts
      // in all instances of KernelOp lowerings
      if (!inplaceAllocIndicesSet.contains(idx) &&
          !isa<MO::ChainType, MO::OpaqueType>(res.getType())) {
        auto outputStorage = loweredOutputsStorage[index];
        createOutputTensor(res, inChain, outputStorage.buffer,
                           outputStorage.tensorSpec);
        ++index;
      }
    }
  } else {
    for (const auto &[idx, outputStorage] :
         llvm::enumerate(loweredOutputsStorage)) {
      createOutputTensor(opBeingLowered->getResult(idx), inChain,
                         outputStorage.buffer, outputStorage.tensorSpec);
    }
  }
}

BufferizedTensor MOOpLoweringState::getBufferizedInput(Value moTensor) {
  Value loweredTensor = loweringState.lookupExecuteLoweredValue(moTensor);
  ASSERT_STREAM(loweredTensor, << "referenced SSA value has not been lowered");

  MO::TensorType ty = MO::castFromTensorLike(moTensor.getType());
  BufferBuilder *bufferBuilder = getStorageBufferBuilder(ty);
  auto buffer = bufferBuilder->tensorExtractBuffer(builder, loc, loweredTensor);
  auto tensorSpec =
      bufferBuilder->tensorExtractTensorSpec(builder, loc, loweredTensor);

  // we are extracting a buffer from a tensor, so the buffer needs to be guarded
  // by the same chain as the tensor
  loweringState.guardValueWithChain(
      buffer, loweringState.getChainGuardingValueOrNull(loweredTensor));

  return BufferizedTensor(buffer, tensorSpec);
}

FailureOr<TypedValue<MGP::TensorSpecType>>
MOOpLoweringState::createTensorSpec(MO::TensorType moTensorType) {
  if (!moTensorType.hasRank())
    return opBeingLowered->emitOpError(
        "dynamically ranked MO tensors are not yet supported");
  MGP::TensorSpecType tensorSpecType = MGP::createTensorSpecType(moTensorType);
  if (tensorSpecType.getSizeInBytes()) {
    // Statically shaped.
    return MGP::TensorSpecCreateOp::create(builder, loc,
                                           /*spec=*/tensorSpecType)
        .getResult();
  }

  // Dynamically shaped.
  auto runtimeDimsOr = loweringState.lowerShapeAttrDynDims(
      opBeingLowered->getParentRegion(), moTensorType.getTypedShapeAttr(), loc);
  if (runtimeDimsOr)
    return opBeingLowered->emitOpError(runtimeDimsOr.getError());
  return MGP::TensorSpecCreateOp::create(builder, loc,
                                         /*spec=*/tensorSpecType,
                                         /*runtimeDims=*/*runtimeDimsOr)
      .getResult();
}

FailureOr<SmallVector<TypedValue<MGP::TensorSpecType>>>
MOOpLoweringState::createOutputTensorSpecs() {
  SmallVector<TypedValue<MGP::TensorSpecType>> tensorSpecs;
  tensorSpecs.reserve(opBeingLowered->getNumResults());

  DenseSet<int> inplaceAllocIndicesSet;
  if (auto kernel = dyn_cast<MOGG::KernelOp>(opBeingLowered)) {
    if (auto allocIndicesAttr = kernel.getAllocIndicesAttr()) {
      for (auto attr : allocIndicesAttr) {
        inplaceAllocIndicesSet.insert(cast<mlir::IntegerAttr>(attr).getInt());
      }
    }
  }

  for (auto [idx, type] :
       llvm::enumerate(opBeingLowered->getResults().getType())) {
    if (inplaceAllocIndicesSet.contains(idx))
      continue;

    // Skip non-tensor types like chains and opaques - they are handled
    // separately by the kernel lowering logic.
    if (isa<MO::ChainType, MO::OpaqueType>(type))
      continue;

    auto moTensorType = dyn_cast<MO::TensorType>(type);
    ASSERT_STREAM(moTensorType, << "result of op is not a !mo.tensor");
    FailureOr<TypedValue<MGP::TensorSpecType>> tensorSpecOr =
        createTensorSpec(moTensorType);
    if (failed(tensorSpecOr))
      return failure();
    tensorSpecs.emplace_back(*tensorSpecOr);
  }
  return tensorSpecs;
}

FailureOr<BufferizedTensor> MOOpLoweringState::createBuffer(
    Value inChain, TypedValue<TensorSpecType> tensorSpec, Type tensorType) {
  BufferBuilder *bufferBuilder = getStorageBufferBuilder(tensorType);
  MO::TensorType moTensorType = MO::dynCastFromTensorLike(tensorType);
  ASSERT_STREAM(moTensorType, << "input is not a !mo.tensor or !mo.buffer");

  Value numBytes = bufferSizeBytes(tensorSpec, moTensorType);

  auto bufferDevice = bufferBuilder->getDevice();
  auto preferredAlignment =
      M::Driver::getPreferredAlignment(bufferDevice.getLabel().str());
  if (bufferDevice.getLabel() == M::kCPULabel)
    preferredAlignment = std::gcd(preferredAlignment, minimumCpuAlignment);
  auto deviceContext = loweringState.getExecuteDevice(bufferDevice);
  return BufferizedTensor(
      bufferBuilder->bufferAlloc(builder, loc, context, inChain, numBytes,
                                 deviceContext, preferredAlignment),
      tensorSpec);
}

void MOOpLoweringState::createOutputTensor(
    Value moTensor, Value inChain, Value buffer,
    TypedValue<MGP::TensorSpecType> tensorSpec) {
  Value loweredTensor =
      getStorageBufferBuilder(moTensor.getType())
          ->tensorCreate(builder, loc, context, inChain, buffer, tensorSpec);
  loweringState.emplaceExecuteLoweredBufferValue(moTensor, loweredTensor,
                                                 inChain);
}

void MOOpLoweringState::createOutputTensors(
    Value inChain, SmallVector<Value> &buffers,
    SmallVector<TypedValue<MGP::TensorSpecType>> &tensorSpecs) {
  for (const auto &[idx, buffer, tensorSpec] :
       llvm::enumerate(buffers, tensorSpecs))
    createOutputTensor(opBeingLowered->getResult(idx), inChain, buffer,
                       tensorSpec);
}

BufferBuilder *MOOpLoweringState::getStorageBufferBuilder(Type moTensorType) {
  auto type = MO::dynCastFromTensorLike(moTensorType);
  ASSERT_STREAM(type, << "expecting a !mo.tensor or !mo.buffer type");
  return loweringState.getBufferBuilder(MO::getDeviceOrDefault(type));
}

TypeBuilder *MOOpLoweringState::getStorageTypeBuilder(Type moTensorType) {
  auto type = dyn_cast<MO::TensorType>(moTensorType);
  ASSERT_STREAM(type, << "expecting !mo.tensor type");
  return loweringState.getTypeBuilder(MO::getDeviceOrDefault(type));
}
