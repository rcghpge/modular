//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Interfaces/LoweringState.h"
#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "GenericML/GraphCompiler/Interfaces/EmitMojoState.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"

#include <numeric>

using namespace M;
using namespace M::MGP;

static FailureOr<Type> builtinTypeToDType(mlir::MLIRContext *ctx, Type type) {
  KGEN::KGENDType dtype;

  if (type.isIndex())
    dtype = KGEN::KGENDType{KGEN::KGENDType::ExtraCases::index};
  else if (auto integer = dyn_cast<IntegerType>(type))
    dtype = KGEN::KGENDType(M::getEquivalentDType(integer));
  else
    return failure();

  return KGEN::POP::SIMDType::get(ctx, 1, dtype);
}

static Value popScalarToBuiltinType(OpBuilder &builder, Location loc, Value v) {
  auto type = cast<KGEN::POP::SIMDType>(v.getType());
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

Value MOGraphLoweringState::indexToDimension(OpBuilder &builder, Location loc,
                                             Value index) {
  if (MOSH::ShapeAttr::isDimType(index.getType()))
    return index;

  return builder.createOrFold<mlir::index::CastSOp>(
      loc, MOSH::ShapeAttr::getDimType(builder.getContext()), index);
}

Value MOGraphLoweringState::dimensionToIndex(OpBuilder &builder, Location loc,
                                             Value dim) {
  if (dim.getType().isIndex())
    return dim;

  return builder.createOrFold<mlir::index::CastSOp>(loc, builder.getIndexType(),
                                                    dim);
}

static Value assertSymbolicDimensionMatchesPreviousValue(
    OpBuilder &builder, Location loc, Value previousValue, Value newValue,
    KGEN::ParamDeclRefAttr symbolicDimension, unsigned argumentIndex) {
  FailureOr<Type> targetType =
      builtinTypeToDType(builder.getContext(), previousValue.getType());

  Value isEqual = KGEN::POP::CmpOp::create(
      builder, loc, KGEN::POP::CmpPredicate::EQ,
      builder.createOrFold<KGEN::POP::CastFromBuiltinOp>(loc, *targetType,
                                                         previousValue),
      builder.createOrFold<KGEN::POP::CastFromBuiltinOp>(loc, *targetType,
                                                         newValue));

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
    } else if (argType.hasUnknownShape()) {
      // Also no parameters to deal with, however this argument won't be
      // usable by any MO ops.
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
    if (!isa<IntegerAttr>(dim)) {
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
/// are !pop.scalar values.
static ErrorOr<Value> pocToMopInteger(KGEN::POC poc, OpBuilder &builder,
                                      Location loc, ValueRange operands,
                                      Type resultType) {
  auto getPopCmpPredicate = [](KGEN::POC poc) -> KGEN::POP::CmpPredicate {
    switch (poc) {
    case KGEN::POC::EQ:
      return KGEN::POP::CmpPredicate::EQ;
    case KGEN::POC::LT:
      return KGEN::POP::CmpPredicate::LT;
    case KGEN::POC::LE:
      return KGEN::POP::CmpPredicate::LE;
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
                                    getPopCmpPredicate(poc), operands[0],
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

  // Get !pop.scalar dtype that corresponds to the expected output type.
  auto popResultType = builtinTypeToDType(builder.getContext(), resultType);
  if (failed(popResultType))
    return Error("unable to convert result type");

  SmallVector<Value> popOperands;
  popOperands.reserve(operands.size());

  // Convert operands to equivalent !pop.scalar type
  for (auto operand : operands) {
    auto scalarType =
        builtinTypeToDType(builder.getContext(), operand.getType());

    if (failed(scalarType))
      return Error("unable to convert operand to !pop.scalar");

    if (scalarType == operand.getType()) {
      popOperands.push_back(operand);
      continue;
    }

    popOperands.push_back(builder.createOrFold<KGEN::POP::CastFromBuiltinOp>(
        loc, *scalarType, operand));
  }

  // Create the implementing pop dialect operation.
  auto maybeResult = pocToMopInteger(opAttr.getOpcode(), builder, loc,
                                     popOperands, *popResultType);

  if (maybeResult.isError())
    return maybeResult;

  // Convert the !pop.scalar value back to the corresponding builtin type.
  return popScalarToBuiltinType(builder, loc, maybeResult.takeValue());
}

Operation *MOGraphLoweringState::lowerIntAttr(OpBuilder &builder, Location loc,
                                              IntegerAttr intAttr) {
  return KGEN::ParamConstantOp::create(builder, loc, intAttr);
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
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    newOp = lowerIntAttr(builder, loc, intAttr);
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

void MOGraphLoweringState::lowerMOGGABI(Operation *kernelOp,
                                        const mlir::BitVector &shouldUnpack,
                                        const mlir::BitVector &shouldPack) {
  auto kernel = cast<MOGG::GenericCompiledRegionOp>(kernelOp);
  OpBuilder builder{&kernel.getBody().front(),
                    kernel.getBody().front().begin()};
  auto loc = kernel->getLoc();
  Operation *terminator = kernel.getBody().front().getTerminator();

  builder.setInsertionPoint(terminator);
  SmallVector<Value> packedValues;
  for (auto &&[i, value] : llvm::enumerate(terminator->getOperands())) {
    Value newValue;
    if (isa<MEF::ChainType>(value.getType())) {
      if (shouldPack[i])
        newValue = MOGG::AsyncReadyOp::create(builder, loc);
    } else if (shouldPack.test(i)) {
      SmallVector<Value, 2> packOperands{value};
      if (isa<MGP::DeviceBufferType>(value.getType())) {
        if (auto mgpBufferAllocOp = mlir::dyn_cast_or_null<MGP::BufferAllocOp>(
                value.getDefiningOp()))
          packOperands.push_back(mgpBufferAllocOp.getDeviceContext());
      }
      newValue = MOGG::AsyncPackOp::create(builder, loc, value.getType(),
                                           packOperands);
    } else if (value.getDefiningOp<MGP::BufferConstantOp>()) {
      newValue = MOGG::AsyncPackBorrowV2Op::create(
          builder, loc, value.getType(), value,
          MOGG::AsyncPlaceholderAnyAsyncValueRefOp::create(builder, loc));
    } else if (auto bufferGetCachedOp =
                   value.getDefiningOp<MGP::BufferGetCachedOp>()) {
      auto memStorageType = KGEN::PointerType::get(
          KGEN::PointerType::get(KGEN::NoneType::get(builder.getContext())));
      auto newBufferGetCachedOp = MGP::BufferGetCachedOp::create(
          builder, loc, {value.getType(), memStorageType},
          bufferGetCachedOp->getOperands(), bufferGetCachedOp->getAttrs());
      newValue = MOGG::AsyncPackBorrowV2Op::create(
          builder, loc, value.getType(), newBufferGetCachedOp.getOutBuffer(),
          newBufferGetCachedOp.getMemStorageHandle());
    } else if (auto tensorExtractBufferOp =
                   value.getDefiningOp<MGP::TensorExtractBufferOp>()) {
      auto asyncUnpackAnyAsyncValueRefOp =
          MOGG::AsyncUnpackAnyAsyncValueRefOp::create(
              builder, loc, tensorExtractBufferOp.getTensor());
      newValue = MOGG::AsyncPackBorrowV2Op::create(
          builder, loc, value.getType(), value, asyncUnpackAnyAsyncValueRefOp);
    } else if (auto tensorCreateOp =
                   value.getDefiningOp<MGP::TensorCreateOp>()) {
      auto asyncUnpackAnyAsyncValueRefOp =
          MOGG::AsyncUnpackAnyAsyncValueRefOp::create(
              builder, loc, tensorCreateOp.getBuffer());
      newValue = MOGG::AsyncPackBorrowV2Op::create(
          builder, loc, value.getType(), value, asyncUnpackAnyAsyncValueRefOp);
    } else if (auto bufferSliceOp = value.getDefiningOp<MGP::BufferSliceOp>()) {
      auto asyncUnpackAnyAsyncValueRefOp =
          MOGG::AsyncUnpackAnyAsyncValueRefOp::create(
              builder, loc, bufferSliceOp.getInBuffer());
      newValue = MOGG::AsyncPackBorrowV2Op::create(
          builder, loc, value.getType(), value, asyncUnpackAnyAsyncValueRefOp);
    } else {
      newValue = MOGG::AsyncPackBorrowOp::create(builder, loc, value);
    }

    if (newValue) {
      value.replaceUsesWithIf(newValue, [&](OpOperand &use) {
        return use.getOwner() == terminator;
      });
      packedValues.push_back(newValue);
    }
  }

  // Then, unpack all inputs that should be unpacked.
  SmallVector<Value> asyncValues;
  builder.setInsertionPointToStart(&kernel.getBody().front());
  for (auto v : kernel.getBody().getArguments()) {
    // Do not attempt to unpack a chain as there is nothing that we could be
    // doing with it internally but still decrement its async reference. Also,
    // do not unpack certain arguments if the primitive being lowered requests
    // it.
    if (shouldUnpack.test(v.getArgNumber()) &&
        !isa<MEF::ChainType>(v.getType())) {
      auto unpackOp = MOGG::AsyncUnpackOp::create(builder, loc, v);
      v.replaceUsesWithIf(unpackOp, [](mlir::OpOperand &operand) {
        return !isa<MOGG::AsyncUnpackOp, MOGG::AsyncUnpackAnyAsyncValueRefOp>(
            operand.getOwner());
      });
    }
    asyncValues.emplace_back(v);
  }

  builder.setInsertionPoint(terminator);
  for (auto asyncValue : asyncValues)
    MOGG::AsyncDelOp::create(builder, loc, asyncValue);
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
  if (auto kernel = dyn_cast<MOGG::KernelOp>(opBeingLowered))
    inplaceAllocIndicesSet = kernel.getInplaceAllocIndicesSet();

  SmallVector<Value> results;
  for (auto [idx, res] : llvm::enumerate(opBeingLowered->getResults())) {
    if (!inplaceAllocIndicesSet.contains(idx) &&
        !isa<MO::ChainType>(res.getType()))
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
    DenseSet<int> inplaceAllocIndicesSet = kernel.getInplaceAllocIndicesSet();
    int index = 0;
    for (const auto &[idx, res] : llvm::enumerate(kernel.getResults())) {
      // ignore chain types as there are currently special carveouts in all
      // instances of KernelOp lowerings
      if (!inplaceAllocIndicesSet.contains(idx) &&
          !isa<MO::ChainType>(res.getType())) {
        auto outputStorage = loweredOutputsStorage[index];
        createOutputTensor(res, inChain, outputStorage.buffer,
                           outputStorage.tensorSpec);
        index++;
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
  if (auto kernel = dyn_cast<MOGG::KernelOp>(opBeingLowered))
    inplaceAllocIndicesSet = kernel.getInplaceAllocIndicesSet();

  for (auto [idx, type] :
       llvm::enumerate(opBeingLowered->getResults().getType())) {
    if (inplaceAllocIndicesSet.contains(idx))
      continue;

    if (isa<MO::ChainType>(type))
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
      M::AsyncRT::getPreferredAlignment(bufferDevice.getLabel().str());
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
