//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements canonicalization and folding methods for
// MO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOAttrs.h"
#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/GraphCompiler/Matchers.h"
#include "GenericML/Support/Dims.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/Permutation.h"
#include "Support/ML/RangeUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {
bool isOne(Value val) {
  Operation *definingOp = val.getDefiningOp();
  if (!definingOp)
    return false;

  // Look through views.
  if (isa<RebindOp, Static::BroadcastToOp, SliceOp, TransposeOp>(definingOp))
    return isOne(definingOp->getOperand(0));

  if (auto asCst = dyn_cast<ConstantOp>(definingOp)) {
    TensorValue out = asCst.getResult();
    size_t rank = asCst.getType().getRank();
    Type elemTy = out.getType().getElementType();
    if (elemTy.isInteger(1)) {
      if (auto maybeBlob = tryGetConstantOfRank<bool>(out, rank)) {
        return llvm::all_of(maybeBlob.value(), [](bool b) { return b; });
      }
    } else if (elemTy.isInteger()) {
      if (auto maybeBlob = tryGetConstantOfRank<int64_t>(out, rank)) {
        return llvm::all_of(maybeBlob.value(),
                            [](int64_t v) { return v == 1; });
      }
    } else if (elemTy.isIntOrFloat()) {
      if (auto maybeBlob = tryGetConstantOfRank<float>(out, rank)) {
        return llvm::all_of(maybeBlob.value(),
                            [](float v) { return v == 1.0f; });
      }
    }
  }

  return false;
}

/// If `op`'s axis is constant and input rank is known, modify `op`'s `axis` to
/// be the canonical version, i.e., it's guaranteed to be in [0, rank).
/// NOTE this should be used as the first canonicalization, so other logic in
/// canonicalization will be able to utilize it.
LogicalResult
canonicalizeStaticNegativeIndicesInPlace(Reduction op, int operandIndex,
                                         PatternRewriter &rewriter) {
  TensorValue axisArr = op.getAxis();
  TensorType inputTy = op.getInput().getType();
  std::optional<int64_t> maybeAxis = tryGetConstantInt(axisArr);
  if (!maybeAxis.has_value() || !inputTy.hasRank())
    return failure();

  int64_t axis = maybeAxis.value();
  if (axis >= 0)
    return failure();

  OpBuilder builder(op);
  builder.setInsertionPoint(op);
  int64_t newAxis = axis + inputTy.getRank();
  auto newAxisOp = builder.create<ConstantOp>(op.getLoc(), newAxis,
                                              getHostDevice(op.getContext()));
  rewriter.modifyOpInPlace(op,
                           [&] { op->setOperand(operandIndex, newAxisOp); });
  return success();
}

template <typename SliceTy, typename AdaptorTy>
OpFoldResult foldSliceHelper(SliceTy slice, AdaptorTy adaptor) {
  // We must be able to see the steps are all `1` as a condition of folding.
  auto steps = dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getStep());
  if (!steps)
    return {};

  TypedValue<TensorType> input = slice.getInput();
  TensorType inputTy = input.getType();

  // If the input and the output are the same and neither have ? in them then
  // we can maybe fold.
  if (slice.getResult().getType() != inputTy || inputTy.hasUnknownShape())
    return {};

  // If all of the steps are `1` then this is an identity. We need to check
  // this as you could produce the same shape by walking backwards through the
  // slice i.e -1. So we must check they are 1 which means the only way to
  // produce the same shape as input is through start of 0 and end of -1. I.E
  // an identity.
  auto stepsData = tryGetConstantBlob<int64_t>(
      cast<ElementsAttr>(steps), slice.getStep().getType().getDtype());
  if (!stepsData.has_value())
    return {};

  for (int64_t step : *stepsData) {
    if (step != 1)
      return {};
  }
  return input;
}

/// Helper to fold reshape ops.
ArrayElementsAttr foldReshape(ArrayRef<int64_t> outIntDims,
                              ArrayElementsAttr inputArr) {
  if (llvm::is_contained(outIntDims, -1))
    return {};
  auto outTy = RankedTensorType::get(outIntDims, inputArr.getElementType());
  return ArrayElementsAttr::get(inputArr.getRawData(), outTy);
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                         Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

/// Find the closes IsolatedFromAbove parent region for moving an operation to.
/// This logic is taken from `mlir/Transforms/Utils/FoldUtils.cpp`, which drives
/// the hoisting of ConstantLike values as high in the region tree as possible.
Region *getInsertionRegion(Block *insertionBlock) {
  while (Region *region = insertionBlock->getParent()) {
    // Insert in this region if the parent is a top-level operation.
    auto *parentOp = region->getParentOp();
    if (parentOp->hasTrait<OpTrait::IsIsolatedFromAbove>() ||
        !parentOp->getBlock())
      return region;
    // Traverse up the parent looking for an insertion region.
    insertionBlock = parentOp->getBlock();
  }
  llvm_unreachable("expected valid insertion region");
}

LogicalResult materializeShapeInference(RebindOp op,
                                        PatternRewriter &rewriter) {
  ArrayRef<KGEN::ParamDeclAttr> paramDecls = op.getOutputParamDecls();
  if (paramDecls.empty())
    return failure();
  // TODO lift into a method on MO::TensorType?
  auto doesShapeAttrContainUnknown = [](TypedAttr shapeAttr) {
    if (auto dimAttrs = dyn_cast<ShapeAttr>(shapeAttr))
      return dimAttrs.hasUnknownDims();
    return isa<KGEN::UnknownAttr>(shapeAttr);
  };
  Location loc = op.getLoc();
  TypedAttr inputShapeAttr = op.getInput().getType().getShapeAttr();
  if (paramDecls.size() == 1) {
    KGEN::ParamDeclAttr newDecl = paramDecls.front();
    if (isa<ShapeType>(newDecl.getType()) &&
        !doesShapeAttrContainUnknown(inputShapeAttr)) {
      // If there is a single declaration in rebind that is a shape parameter,
      // we materialize into a kgen.param.declare.
      rewriter.create<KGEN::ParamDeclareOp>(loc, newDecl, inputShapeAttr);
      rewriter.modifyOpInPlace(op, [&] { op.setOutputParamDecls({}); });
      return success();
    }
  }
  TypedAttr outputShapeAttr = op.getResult().getType().getShapeAttr();
  auto outputDimAttrs = dyn_cast<ShapeAttr>(outputShapeAttr);
  bool allDeclsAreDimAttrs = llvm::all_of(paramDecls, [](ParamDeclAttr decl) {
    return ShapeAttr::isDimType(decl.getType());
  });
  if (allDeclsAreDimAttrs && outputDimAttrs) {
    llvm::SmallDenseMap<StringAttr, KGEN::ParamDeclAttr> nameToParamDecl;
    for (KGEN::ParamDeclAttr paramDecl : paramDecls)
      nameToParamDecl.try_emplace(paramDecl.getName(), paramDecl);
    if (auto inputShapeParamRef =
            dyn_cast<KGEN::ParamDeclRefAttr>(inputShapeAttr)) {
      // Example:
      //   rebind<() -> D1> : Sh0 -> [D0, D1, 44]
      // --->
      //   kgen.param.declare D1 = <apply("mosh.get_dim", Sh0, 1)>
      //   rebind : Sh0 -> [D0, D1, 44]
      for (auto [axis, outputDimAttr] :
           llvm::enumerate(outputDimAttrs.getValues())) {
        KGEN::ParamDeclAttr paramDecl;
        auto outputDimParamRef =
            dyn_cast<KGEN::ParamDeclRefAttr>(outputDimAttr);
        if (outputDimParamRef &&
            (paramDecl = nameToParamDecl.lookup(outputDimParamRef.getName()))) {
          TypedAttr mlirOpCallAttr =
              MOSH::GetDimOp::emitStaticCall(inputShapeParamRef, axis);
          rewriter.create<KGEN::ParamDeclareOp>(loc, paramDecl, mlirOpCallAttr);
        }
      }
      rewriter.modifyOpInPlace(op, [&] { op.setOutputParamDecls({}); });
      return success();
    }
    if (auto inputDimAttrs = dyn_cast<ShapeAttr>(inputShapeAttr)) {
      // Example:
      //   rebind<() -> D4> : [D0, D1, D2] -> [D3, D4, 42]
      // --->
      //   kgen.param.declare D4 = <D1>
      //   rebind : [D0, D1, D2] -> [D0, D4, 42]
      llvm::DenseSet<KGEN::ParamDeclAttr> unknownParamDecls;
      // Avoid redeclaring multiple parameters. Example:
      //  rebind<() -> C> : [A, B] -> [C, C]
      // --->
      //   kgen.param.declare C = <A>
      //   rebind : [A, B] -> [C, C]
      llvm::SmallSetVector<KGEN::ParamDeclAttr, 4> declaredParams;
      for (auto [inputDimAttr, outputDimAttr] :
           llvm::zip(inputDimAttrs.getValues(), outputDimAttrs.getValues())) {
        if (auto outputDimParamRef =
                dyn_cast<KGEN::ParamDeclRefAttr>(outputDimAttr)) {
          if (KGEN::ParamDeclAttr paramDecl =
                  nameToParamDecl.lookup(outputDimParamRef.getName())) {
            if (isa<KGEN::UnknownAttr>(inputDimAttr)) {
              unknownParamDecls.insert(paramDecl);
            } else if (declaredParams.insert(paramDecl)) {
              rewriter.create<KGEN::ParamDeclareOp>(loc, paramDecl,
                                                    inputDimAttr);
            }
          }
        }
      }
      if (unknownParamDecls.size() == paramDecls.size())
        return failure();
      rewriter.modifyOpInPlace(op, [&] {
        op.setOutputParamDecls(SmallVector<KGEN::ParamDeclAttr>(
            unknownParamDecls.begin(), unknownParamDecls.end()));
      });
      return success();
    }
  }
  // TODO: support materialization for other rebinds
  return failure();
}

} // namespace

namespace M::MO {

//===----------------------------------------------------------------------===//
// AllReduceSumOp
//===----------------------------------------------------------------------===//
LogicalResult AllReduceSumOp::fold(FoldAdaptor adaptor,
                                   SmallVectorImpl<OpFoldResult> &results) {
  // If the all-reduce has 1 input, return the input (identity op).
  if (getInputs().size() == 1 &&
      getInputs()[0].getType() == getOutput().getType()) {
    results.push_back(getInputs()[0]);
    results.push_back(getInChain());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// AllGatherOp
//===----------------------------------------------------------------------===//
LogicalResult AllGatherOp::fold(FoldAdaptor adaptor,
                                SmallVectorImpl<OpFoldResult> &results) {
  // If the all-gather has 1 input, return the input (identity op).
  // The verifier ensures that with 1 input, there is exactly 1 output.
  if (getInputs().size() == 1 &&
      getInputs()[0].getType() == getOutputs()[0].getType()) {
    results.push_back(getInputs()[0]);
    results.push_back(getInChain());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//
OpFoldResult AssertOp::fold(FoldAdaptor adaptor) {
  // If the condition is true, we return a chain attribute, otherwise we leave
  // the op as is and a pass can either check for failing asserts or it becomes
  // a guaranteed runtime failure.
  if (auto boolAttr = dyn_cast_if_present<BoolAttr>(adaptor.getCond()))
    if (boolAttr.getValue())
      return ChainAttr::get(ChainType::get(getContext()));
  return {};
}

//===----------------------------------------------------------------------===//
// ArgMaxOp
//===----------------------------------------------------------------------===//
LogicalResult ArgMaxOp::canonicalize(ArgMaxOp op, PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

//===----------------------------------------------------------------------===//
// ArgMinOp
//===----------------------------------------------------------------------===//
LogicalResult ArgMinOp::canonicalize(ArgMinOp op, PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//
OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  // Cast to the same type is just identity.
  TensorType resTy = getResult().getType();
  TensorType inTy = getInput().getType();
  if (inTy == resTy)
    return getInput();

  // TODO(#10811): support other types and do this using a MOJO kernel? Waiting
  // on #23197
  Attribute inputAttr = adaptor.getInput();
  if (!inputAttr)
    return {};

  auto rankedResTy =
      RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
  auto resElemTy = rankedResTy.getElementType();
  if (auto intElemsAttr =
          dyn_cast_if_present<IntArrayElementsAttr>(inputAttr)) {
    if (isa<IntegerType>(resElemTy)) {
      // This is inefficient, but simple.
      SmallVector<APInt> res = llvm::map_to_vector(
          intElemsAttr.getValues(), [&](const llvm::APInt &dim) {
            if (inTy.getDtype().isSInt())
              return APInt(64, dim.getSExtValue(), /*isSigned=*/true);

            // Else is unsigned integer.
            return APInt(64, dim.getZExtValue(), /*isSigned=*/false);
          });

      return IntArrayElementsAttr::get(rankedResTy, res);
    }
  }

  return {};
}

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//
LogicalResult BroadcastToOp::canonicalize(BroadcastToOp op,
                                          PatternRewriter &rewriter) {
  TensorValue inputVal = op.getInput();
  TensorType inputTy = inputVal.getType();
  TensorType resTy = op.getType();

  // If we have a known input and an known output with no parameter decls then
  // we can convert this to static.
  if (!resTy.hasUnknownShape() && !inputTy.hasUnknownShape() &&
      op.getOutputParamDecls().empty()) {
    rewriter.replaceOpWithNewOp<Static::BroadcastToOp>(op, op.getType(),
                                                       inputVal);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// ChainCreateOp
//===----------------------------------------------------------------------===//
OpFoldResult ChainCreateOp::fold(FoldAdaptor adaptor) {
  if (getInputs().empty())
    return nullptr;

  // Forward the input if its just a single chain value.
  if (getNumOperands() == 1 && isa<ChainType>(getOperand(0).getType()))
    return getOperand(0);

  // Try to de-duplicate repeated values to the chain create.
  SmallVector<Value> keptValues;
  llvm::SmallPtrSet<Value, 4> seenValues;

  for (Value val : getInputs()) {
    auto [_, inserted] = seenValues.insert(val);
    if (!inserted)
      continue;

    auto chainDefOp = val.getDefiningOp<MOSH::ParamToValueOp>();

    if (!chainDefOp) {
      keptValues.push_back(val);
      continue;
    }

    auto isConstantChain = isa<MO::ChainAttr>(chainDefOp.getValue());

    if (!isConstantChain) {
      keptValues.push_back(val);
      continue;
    }
  }

  // In this case all the inputs were from 'constant' chains, that is:
  // mosh.param.to_value: !mo.chain = <#mo.chain>
  //
  // replace the output of this op with one of these
  if (keptValues.empty())
    return getInputs().front();

  // In this case none of the inputs were constant chains so leave the op
  // unperturbed.
  if (keptValues.size() == getInputs().size())
    return nullptr;

  getOperation()->setOperands(keptValues);

  // De-duplication may have simplified the chain to have just a single operand,
  // check again if its a chain and if so forward it.
  if (getNumOperands() == 1 && isa<ChainType>(getOperand(0).getType()))
    return getOperand(0);

  return nullptr;
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//
OpFoldResult ConcatOp::fold(FoldAdaptor adaptor) {
  TensorType outType = getResult().getType();

  Value axis = getAxis();

  SplitOp split = nullptr;

  // If the concat is the sole product of a split along the same axis we can
  // fold and just use the split input.
  size_t operandIndex = 0;
  for (Value operand : getInputs()) {
    auto definingOp = operand.getDefiningOp<MO::SplitOp>();
    if (!definingOp) {
      split = nullptr;
      break;
    }

    // Check if we have a match, all operands must have the same defining op.
    if (!split) {
      split = definingOp;
    } else if (split != definingOp) {
      split = nullptr;
      break;
    }

    // Split must have same axis.
    if (definingOp.getAxis() != axis) {
      split = nullptr;
      break;
    }

    if (cast<OpResult>(operand).getResultNumber() != operandIndex) {
      split = nullptr;
      break;
    }

    // Each split operand must be in the same order as they appear here.
    operandIndex++;
  }

  // If we are a concat followed by split we can fold and use the split input
  // directly.
  if (split && split.getInput().getType() == outType &&
      split.getNumResults() == getInputs().size())
    return split.getInput();

  // Enforce rank 1 so we don't need to check axis or deal with complex
  // stride stuff.
  if (!outType.hasRank() || outType.getRank() != 1 || !outType.hasStaticShape())
    return {};

  SmallVector<APInt> concatTensor;

  DType outDtype = outType.getDtype();
  for (Attribute attr : adaptor.getInputs()) {
    // Enforce that it only runs on ints.
    auto asIntAttr = dyn_cast_if_present<IntArrayElementsAttr>(attr);
    if (!asIntAttr)
      return {};

    auto maybeValue =
        tryGetConstantBlob<int64_t>(cast<ElementsAttr>(asIntAttr), outDtype);
    if (!maybeValue)
      return {};

    // Append the values to the output tensor. We can do this since it is
    // rank 1.
    for (size_t idx = 0, e = maybeValue->size(); idx < e; idx++)
      concatTensor.push_back(APInt(64, (*maybeValue)[idx], true));
  }

  auto rankedResTy =
      RankedTensorType::get(outType.getIntDims(), outType.getElementType());
  return IntArrayElementsAttr::get(rankedResTy, concatTensor);
}

LogicalResult ConcatOp::canonicalize(ConcatOp op, PatternRewriter &rewriter) {
  Operation::operand_range inputs = op.getInputs();
  SmallVector<Value> nonEmptyInputs;
  if (!op.getOutputParamDecls().empty())
    return failure();
  for (Value arg : inputs) {
    TensorType argTy = cast<TensorType>(arg.getType());
    auto isNonZero = [](int64_t dim) { return dim != 0; };
    if (!argTy.hasRank() || llvm::all_of(argTy.getIntDims(), isNonZero))
      nonEmptyInputs.emplace_back(arg);
  }
  // TODO: implement this
  if (nonEmptyInputs.empty())
    return failure();
  if (nonEmptyInputs.size() == 1) {
    rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(),
                                          nonEmptyInputs.front());
    return success();
  }
  if (nonEmptyInputs.size() != inputs.size()) {
    op.getInputsMutable().assign(nonEmptyInputs);
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ConcatFromListOp
//===----------------------------------------------------------------------===//
LogicalResult ConcatFromListOp::canonicalize(ConcatFromListOp op,
                                             PatternRewriter &rewriter) {
  auto resultType = cast<TensorType>(op->getResultTypes()[0]);
  MO::ListValue inputList = op.getInput();
  if (resultType.getElementType().isInteger()) {
    if (auto constData = tryGetConstantAPIntVector(inputList)) {
      rewriter.replaceOpWithNewOp<ConstantOp>(op, *constData,
                                              resultType.getElementType(),
                                              resultType.getDeviceRef());
      return success();
    }
  } else if (resultType.getElementType().isIntOrFloat()) {
    if (auto constData = tryGetConstantAPFloatVector(inputList)) {
      rewriter.replaceOpWithNewOp<ConstantOp>(op, *constData,
                                              resultType.getElementType(),
                                              resultType.getDeviceRef());
      return success();
    }
  }
  // If the input is a mo.list.create, maybe we can simplify to a mo.concat.
  if (auto listCreateOp = inputList.getDefiningOp<MO::ListCreateOp>()) {
    auto listElementTy =
        cast<TensorType>(listCreateOp.getResult().getType().getElementType());
    if (!listElementTy.hasRank())
      return failure();
    // If the list elements are rank-0, we can simplify, but we need to
    // broadcast each input to rank-1 with a single element.
    if (listElementTy.getRank() == 0) {
      // If the result type is not rank-1, we won't simplify (to be safe).
      if (!resultType.hasRank() || resultType.getRank() != 1)
        return failure();
      auto rankOneTy = listElementTy.cloneWith(SmallVector<int64_t>{1},
                                               listElementTy.getElementType());
      size_t numInputs = listCreateOp->getNumOperands();
      SmallVector<Value> rankOneInputs;
      rankOneInputs.reserve(numInputs);
      for (size_t i = 0; i < numInputs; ++i) {
        rankOneInputs.push_back(rewriter.create<Static::ReshapeOp>(
            op->getLoc(), rankOneTy, listCreateOp.getInputTensor(i)));
      }
      rewriter.replaceOpWithNewOp<MO::ConcatOp>(op, resultType, op.getAxis(),
                                                rankOneInputs,
                                                op.getOutputParamDeclsAttr());
      return success();
    }
    // If any of the inputs are dynamically ranked, we won't simplify.
    for (Value input : listCreateOp.getInputs())
      if (!cast<MO::TensorType>(input.getType()).hasRank())
        return failure();
    rewriter.replaceOpWithNewOp<MO::ConcatOp>(op, resultType, op.getAxis(),
                                              listCreateOp.getInputs(),
                                              op.getOutputParamDeclsAttr());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValueAttr(); }

LogicalResult ConstantOp::canonicalize(ConstantOp op,
                                       PatternRewriter &rewriter) {
  if (op.getType().hasUnknownShape()) {
    Location loc = op.getLoc();
    ElementsAttr elements = op.getValue();
    auto staticTensorType = op.getResult().getType().cloneWith(
        cast<ShapedType>(elements.getType()).getShape(),
        elements.getElementType());
    auto constantOp =
        rewriter.create<ConstantOp>(loc, staticTensorType, elements);
    rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(), constantOp);
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ConstantExternalOp
//===----------------------------------------------------------------------===//
LogicalResult ConstantExternalOp::canonicalize(ConstantExternalOp op,
                                               PatternRewriter &rewriter) {
  // Special case the hoisting of mo.constant.external operations, since they
  // cannot be ConstantLike and hoisted by default in the
  // GreedyPatternRewriteDriver.
  Region *insertRegion = getInsertionRegion(op->getBlock());
  Block *insertBlock = &insertRegion->front();
  if (insertBlock == op->getBlock())
    return failure();
  rewriter.moveOpBefore(op, insertBlock, insertBlock->begin());
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantScalarOp
//===----------------------------------------------------------------------===//
OpFoldResult ConstantScalarOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//
LogicalResult ConvOp::canonicalize(ConvOp op, PatternRewriter &rewriter) {
  StringRef filterLayoutStr = "";
  LayoutAttr filterLayout = op.getFilter().getType().getLayoutAttr();
  if (filterLayout)
    filterLayoutStr = filterLayout.getFormat();
  if (op.getFilterLayout() == filterLayoutStr)
    return failure();
  rewriter.modifyOpInPlace(op, [&]() { op.setFilterLayout(filterLayoutStr); });
  return success();
}

//===----------------------------------------------------------------------===//
// ConvTransposeOp
//===----------------------------------------------------------------------===//
LogicalResult ConvTransposeOp::canonicalize(ConvTransposeOp op,
                                            PatternRewriter &rewriter) {
  StringRef filterLayoutStr = "";
  LayoutAttr filterLayout = op.getFilter().getType().getLayoutAttr();
  if (filterLayout)
    filterLayoutStr = filterLayout.getFormat();
  if (op.getFilterLayout() == filterLayoutStr)
    return failure();
  rewriter.modifyOpInPlace(op, [&]() { op.setFilterLayout(filterLayoutStr); });
  return success();
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

// Helper function for empty array creation
static ArrayElementsAttr getEmptyArray(Type elementType) {
  return ArrayElementsAttr::get({}, RankedTensorType::get({0}, elementType));
}

OpFoldResult GatherOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?
  if (!getOutputParamDecls().empty())
    return {};

  auto inputElems = dyn_cast_if_present<ElementsAttr>(adaptor.getInput());
  auto idxElems = dyn_cast_if_present<ElementsAttr>(adaptor.getIndices());

  // The verifier should ensure that indices is empty, but we don't check it.
  if (inputElems && inputElems.empty())
    return getEmptyArray(inputElems.getElementType());

  if (idxElems && idxElems.empty())
    // DType of empty array must match
    return getEmptyArray(getResult().getType().getElementType());

  if (!inputElems || !idxElems)
    return {};

  TensorType resTy = getResult().getType();
  TensorType inTy = getInput().getType();

  if (!resTy.hasRank() || !inTy.hasRank())
    return {};

  // We only perform this on rank 1 gathers.
  if (resTy.getRank() != inTy.getRank() || inTy.getRank() != 1 ||
      !resTy.hasStaticShape() || !inTy.hasStaticShape())
    return {};

  DType dataTy = getInput().getType().getDtype();
  DType indicesTy = getIndices().getType().getDtype();

  // If the input/indices/axis is known this is something we can fold.
  // We have to enforce that they are only ints for now.
  // We only run on rank 1 so we know the only possible value for axis is 1.
  auto dataAttr = dyn_cast<IntArrayElementsAttr>(inputElems);
  auto iAttr = dyn_cast<IntArrayElementsAttr>(idxElems);
  if (!dataAttr || !iAttr)
    return {};

  auto maybeData =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(dataAttr), dataTy);
  auto maybeInd =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(iAttr), indicesTy);
  if (!maybeData.has_value() || !maybeInd.has_value())
    return {};

  ArrayRef<int64_t> data = *maybeData;
  ArrayRef<int64_t> indices = *maybeInd;

  SmallVector<APInt> gatheredValues(indices.size());
  for (size_t idx = 0, e = indices.size(); idx < e; idx++) {
    int64_t index = indices[idx];
    if (index < 0 || static_cast<size_t>(index) > data.size())
      return {};

    int64_t val = data[index];
    gatheredValues[idx] = APInt(64, val, /*isSigned=*/true);
  }

  auto rankedResTy =
      RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
  return IntArrayElementsAttr::get(rankedResTy, gatheredValues);
}

//===----------------------------------------------------------------------===//
// GuardOp
//===----------------------------------------------------------------------===//

namespace {

/// Canonicalization pattern that removes duplicate operands from guard
/// operations.
///
/// For example, transforms:
///   %r0, %r1, %r2, %r3 = mo.guard[%ch](%a, %b, %a, %b)
/// Into:
///   %f0, %f1 = mo.guard[%ch](%a, %b)
struct DedupeGuardOperands : public OpRewritePattern<GuardOp> {
  using OpRewritePattern<GuardOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GuardOp guard,
                                PatternRewriter &rewriter) const override {
    // Map each unique operand to the index of its first occurrence.
    llvm::MapVector<Value, size_t> uniqueOperands;
    for (auto [index, operand] : llvm::enumerate(guard.getInputs()))
      uniqueOperands.try_emplace(operand, index);
    // If all operands are already unique, nothing to do.
    if (uniqueOperands.size() == guard.getInputs().size())
      return rewriter.notifyMatchFailure(guard, "no duplicated operands");
    // Create a new guard with only the unique operands.
    auto chain = guard.getChain();
    auto newOperands = llvm::to_vector(llvm::make_first_range(uniqueOperands));
    auto newGuard =
        GuardOp::create(rewriter, guard.getLoc(), chain, newOperands);
    // Map each original result to the corresponding result from the new guard.
    // For duplicate operands, multiple results will map to the same new result.
    SmallVector<Value> newResults;
    newResults.reserve(guard.getNumResults());
    for (Value operand : guard.getInputs()) {
      size_t index = uniqueOperands[operand];
      newResults.push_back(newGuard.getResults()[index]);
    }
    rewriter.replaceOp(guard, newResults);
    return success();
  }
};

} // namespace

LogicalResult GuardOp::fold(FoldAdaptor adaptor,
                            SmallVectorImpl<OpFoldResult> &results) {
  auto chainDefOp = getChain().getDefiningOp<MOSH::ParamToValueOp>();
  if (!chainDefOp)
    return failure();
  bool isTrivialChain = isa<MO::ChainAttr>(chainDefOp.getValue());
  if (!isTrivialChain)
    return failure();
  results.insert(results.begin(), getInputs().begin(), getInputs().end());
  return success();
}

void GuardOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<DedupeGuardOperands>(ctx);
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//
namespace {
struct ConstantFoldIf : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  static std::optional<bool> getConstantCondition(IfOp op) {
    if (auto tensor = dyn_cast<TensorValue>(op.getCond()))
      if (auto value = tryGetConstantBool(tensor))
        return value;

    BoolAttr attr;
    if (matchPattern(op.getCond(), m_Constant(&attr)))
      return attr.getValue();

    return {};
  }

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    auto condValue = getConstantCondition(op);
    if (!condValue)
      return rewriter.notifyMatchFailure(op, "non-constant condition");

    if (*condValue)
      replaceOpWithRegion(rewriter, op, op.getThenRegion());
    else
      replaceOpWithRegion(rewriter, op, op.getElseRegion());

    return success();
  }
};

struct RemoveUnusedResults : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  static void transferBody(Block *source, Block *dest,
                           ArrayRef<OpResult> usedResults,
                           PatternRewriter &rewriter) {
    // Move all operations to the destination block.
    rewriter.mergeBlocks(source, dest);
    // Replace the yield op by one that returns only the used values.
    auto yieldOp = cast<YieldOp>(dest->getTerminator());

    auto usedOperands = llvm::map_to_vector(usedResults, [&](OpResult res) {
      return yieldOp.getOperand(res.getResultNumber());
    });

    rewriter.modifyOpInPlace(yieldOp,
                             [&]() { yieldOp->setOperands(usedOperands); });
  }

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    auto usedResults = llvm::filter_to_vector(
        op.getResults(), [](OpResult res) { return !res.use_empty(); });

    if (usedResults.size() == op.getNumResults())
      return failure();

    auto newTypes = llvm::map_to_vector(
        usedResults, [](OpResult res) { return res.getType(); });

    auto newOp = rewriter.create<IfOp>(op.getLoc(), newTypes, op.getCond());
    rewriter.createBlock(&newOp.getThenRegion());
    rewriter.createBlock(&newOp.getElseRegion());

    // Move the bodies and replace the terminators (note there is a then and
    // an else region since the operation returns results).
    transferBody(op.getBody(0), newOp.getBody(0), usedResults, rewriter);
    transferBody(op.getBody(1), newOp.getBody(1), usedResults, rewriter);

    // Replace the operation by the new one.
    SmallVector<Value> repResults(op.getNumResults());
    for (const auto &en : llvm::enumerate(usedResults))
      repResults[en.value().getResultNumber()] = newOp.getResult(en.index());
    rewriter.replaceOp(op, repResults);
    return success();
  }
};

/// Replace uses of the condition value with a true constant in the then branch
/// and false in the else branch.
struct ConditionPropagation : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  static Value createBoolOfType(PatternRewriter &rewriter, Location loc,
                                Type type, bool value) {
    if (auto tensorTy = dyn_cast<TensorType>(type))
      return rewriter.create<ConstantOp>(loc, value, tensorTy.getDeviceRef());

    assert(isa<ScalarType>(type) && "must be a tensor or scalar type");
    return rewriter.create<ConstantScalarOp>(loc, rewriter.getBoolAttr(value));
  }

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;

    // These variables serve to prevent creating duplicate constants
    // and hold constant true or false values.
    Value constantTrue = nullptr;
    Value constantFalse = nullptr;

    for (OpOperand &use : llvm::make_early_inc_range(op.getCond().getUses())) {
      auto useRegion = use.getOwner()->getParentRegion();

      if (op.getThenRegion().isAncestor(useRegion)) {
        changed = true;

        if (!constantTrue)
          constantTrue = createBoolOfType(rewriter, op.getLoc(),
                                          op.getCond().getType(), true);

        rewriter.modifyOpInPlace(use.getOwner(),
                                 [&]() { use.set(constantTrue); });
      } else if (op.getElseRegion().isAncestor(useRegion)) {
        changed = true;

        if (!constantFalse)
          constantFalse = createBoolOfType(rewriter, op.getLoc(),
                                           op.getCond().getType(), false);

        rewriter.modifyOpInPlace(use.getOwner(),
                                 [&]() { use.set(constantFalse); });
      }
    }

    return success(changed);
  }
};

} // end namespace

void IfOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx) {
  patterns.add<ConstantFoldIf, ConditionPropagation, RemoveUnusedResults>(ctx);
}

//===----------------------------------------------------------------------===//
// IndexToTensorOp
//===----------------------------------------------------------------------===//
OpFoldResult IndexToTensorOp::fold(FoldAdaptor adaptor) {
  auto inputAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getInput());
  if (!inputAttr)
    return {};

  SmallVector<APInt> res{inputAttr.getValue()};
  auto rankedResTy = RankedTensorType::get({}, getType().getElementType());
  return IntArrayElementsAttr::get(rankedResTy, res);
}

LogicalResult IndexToTensorOp::canonicalize(IndexToTensorOp op,
                                            PatternRewriter &rewriter) {
  auto paramToValueOp = op.getInput().getDefiningOp<MOSH::ParamToValueOp>();
  if (!paramToValueOp)
    return failure();

  auto inputIntAttr = dyn_cast<IntegerAttr>(paramToValueOp.getValue());
  if (!inputIntAttr)
    return failure();

  SmallVector<APInt> res{inputIntAttr.getValue()};
  auto rankedResTy = RankedTensorType::get({}, op.getType().getElementType());
  auto elementsAttr =
      cast<ElementsAttr>(IntArrayElementsAttr::get(rankedResTy, res));
  rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getType(), elementsAttr);
  return success();
}

//===----------------------------------------------------------------------===//
// LayoutTransformOp
//===----------------------------------------------------------------------===//
LogicalResult LayoutTransformOp::canonicalize(LayoutTransformOp op,
                                              PatternRewriter &rewriter) {
  LayoutAttr inputLayout = op.getInput().getType().getLayoutAttr();
  LayoutAttr outputLayout = op.getResult().getType().getLayoutAttr();
  if (inputLayout == outputLayout) {
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//
LogicalResult MatmulOp::canonicalize(MatmulOp op, PatternRewriter &rewriter) {
  bool transposeB = false;
  bool packedB = false;

  LayoutAttr weightsLayout = op.getInputB().getType().getLayoutAttr();
  if (weightsLayout) {
    transposeB = weightsLayout.getFormat() == "NK";
    packedB = weightsLayout.getFormat() == "KNkni";
  }

  if (transposeB == op.getTransposeB() && packedB == op.getPackedB())
    return failure();

  rewriter.modifyOpInPlace(op, [&]() {
    op.setTransposeB(transposeB);
    op.setPackedB(packedB);
  });

  return success();
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//
LogicalResult MeanOp::canonicalize(MeanOp op, PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//
LogicalResult MulOp::canonicalize(MulOp op, PatternRewriter &rewriter) {
  if (isOne(op.getInputX())) {
    rewriter.replaceOp(op, op.getInputY());
    return success();
  } else if (isOne(op.getInputY())) {
    rewriter.replaceOp(op, op.getInputX());
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// MutableLoadOp
//===----------------------------------------------------------------------===//

// In instances of two sequential loads on the same buffer this will
// canonicalize the two loads to just one. This replaces the uses of the current
// load op's chain and tensor with the upstream load op's chain and tensor.
LogicalResult MutableLoadOp::canonicalize(MutableLoadOp loadOp,
                                          mlir::PatternRewriter &rewriter) {
  auto inChain = loadOp.getInChain();
  auto upstreamLoadOp =
      dyn_cast_if_present<MutableLoadOp>(inChain.getDefiningOp());

  if (!upstreamLoadOp) {
    return rewriter.notifyMatchFailure(
        loadOp.getLoc(), "Upstream op (on chain) is not a mo.mutable.load op.");
  }

  if (upstreamLoadOp.getInBuffer() != loadOp.getInBuffer()) {
    return rewriter.notifyMatchFailure(
        loadOp.getLoc(),
        "Upstream mo.mutable.load op does not operate on same buffer.");
  }

  // Only collapse loads if they're in the same block
  if (upstreamLoadOp->getBlock() != loadOp->getBlock()) {
    return rewriter.notifyMatchFailure(
        loadOp.getLoc(),
        "Upstream mo.mutable.load op is in a different block.");
  }

  rewriter.replaceAllUsesWith(loadOp.getOutChain(),
                              upstreamLoadOp.getOutChain());
  rewriter.replaceAllUsesWith(loadOp.getOutTensor(),
                              upstreamLoadOp.getOutTensor());

  // DCE in the canonicalization pass will take care of deleting loadOp
  return success();
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//
OpFoldResult RangeOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?

  auto startAttr =
      dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getStart());
  auto limitAttr =
      dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getLimit());
  auto stepAttr = dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getStep());
  if (!startAttr || !limitAttr || !stepAttr)
    return {};

  int64_t start = getIntBlob(startAttr).front();
  int64_t limit = getIntBlob(limitAttr).front();
  int64_t step = getIntBlob(stepAttr).front();

  auto numResElementsOr = getRangeNumElements(start, limit, step);
  if (numResElementsOr.isError())
    return {};
  int64_t numResElements = *numResElementsOr;
  SmallVector<APInt> res((size_t)numResElements);
  for (size_t idx = 0; idx < (size_t)numResElements; ++idx) {
    res[idx] = APInt(64, start, /*isSigned=*/true);
    start += step;
  }

  auto attrTy =
      RankedTensorType::get({numResElements}, startAttr.getElementType());
  auto foldedAttr = IntArrayElementsAttr::get(attrTy, res);
  TensorType outTy = getType();
  if (outTy.hasStaticShape())
    return foldedAttr;

  Location loc = getLoc();
  OpBuilder builder(*this); // insert before this op
  auto resTy = outTy.cloneWith({numResElements}, outTy.getElementType());
  auto constantOp =
      builder.create<ConstantOp>(loc, resTy, cast<ElementsAttr>(foldedAttr));
  auto rebindOp =
      builder.create<RebindOp>(loc, outTy, constantOp, getOutputParamDecls());
  return rebindOp.getResult();
}

//===----------------------------------------------------------------------===//
// ReduceAddOp
//===----------------------------------------------------------------------===//
OpFoldResult ReduceAddOp::fold(FoldAdaptor adaptor) {
  auto axisArr = dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getAxis());
  TensorType inputTy = getInput().getType();

  if (!axisArr || !inputTy.hasRank())
    return {};

  int64_t axis = getIntBlob(axisArr).front();
  int64_t rank = inputTy.getRank();
  axis = (axis + rank) % rank;

  // If we are reducing along a dimension of 1
  if (inputTy.getIntDim(axis) == 1 &&
      getInput().getType() == getResult().getType())
    return getInput();
  return {};
}

LogicalResult ReduceAddOp::canonicalize(ReduceAddOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

//===----------------------------------------------------------------------===//
// ReduceMaxOp
//===----------------------------------------------------------------------===//
LogicalResult ReduceMaxOp::canonicalize(ReduceMaxOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

//===----------------------------------------------------------------------===//
// ReduceMinOp
//===----------------------------------------------------------------------===//
LogicalResult ReduceMinOp::canonicalize(ReduceMinOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

//===----------------------------------------------------------------------===//
// ReduceMulOp
//===----------------------------------------------------------------------===//
LogicalResult ReduceMulOp::canonicalize(ReduceMulOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//
LogicalResult RebindOp::canonicalize(RebindOp op,
                                     PatternRewriter &origRewriter) {
  DeviceAwareRewriter rewriter(op, origRewriter);
  // We fold if the result and operand types are the same.
  if (op.getInput().getType() == op.getType()) {
    assert(op.getOutputParamDecls().empty());
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
  // We materialize the parameter declarations explicitly.
  LogicalResult paramsResolved = materializeShapeInference(op, origRewriter);
  // Likely already processed, or doesn't help shape inference.
  if (op.getInput().getDefiningOp<RebindOp>())
    return success(succeeded(paramsResolved));
  //  %0 = mo.rebind<...>(%arg) // dominator
  //  %1 = mo.rebind<...>(%arg)
  //  %2 = mo.rebind<...>(%arg)
  // ==>
  //  %0 = mo.rebind<...>(%arg) // dominator
  //  %1 = mo.rebind<...>(%0)
  //  %2 = mo.rebind<...>(%0)
  //
  // NOTE This helps shape inference propagate parameter constraints across
  // these rebinds, in case `%arg`'s type is never parameterized and contains
  // unknown dimension `?`.
  if (op.getInput().getType().hasUnknownShape()) {
    SmallVector<RebindOp> rebindsWithSameInput;
    RebindOp dominatorRebindOp = op;
    for (Operation *rebindInputUser : op.getInput().getUsers()) {
      auto rebindOp = dyn_cast<RebindOp>(rebindInputUser);
      // TODO To support this across blocks, we need `DominanceInfo`, which
      // might require us to move this into a pass
      if (rebindOp && rebindOp->getBlock() == dominatorRebindOp->getBlock()) {
        rebindsWithSameInput.push_back(rebindOp);
        if (rebindOp->isBeforeInBlock(dominatorRebindOp))
          dominatorRebindOp = rebindOp;
      }
    }
    if (rebindsWithSameInput.size() > 1) {
      for (RebindOp rebindOp : rebindsWithSameInput)
        if (rebindOp != dominatorRebindOp)
          rebindOp.getInputMutable().assign(dominatorRebindOp);
      return success();
    }
  }
  return success(succeeded(paramsResolved));
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//
OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) {
  auto inputArr = dyn_cast_if_present<ArrayElementsAttr>(adaptor.getInput());
  auto shapeArr =
      dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getNewShape());
  auto outputShapeAttr = dyn_cast<ShapeAttr>(getType().getShapeAttr());
  if (!inputArr || !shapeArr || !outputShapeAttr)
    return {};

  // Compute number of elements specified by target shape
  int64_t shapeNumElems = 1;
  for (llvm::APInt dim : shapeArr.getValues()) {
    if (dim == -1)
      return {};
    shapeNumElems *= dim.getSExtValue();
  }

  // Compute number of elements specified by output type
  int64_t outputNumElems = 1;
  SmallVector<int64_t> outIntDims = outputShapeAttr.getIntDims();
  for (int64_t intDim : outIntDims) {
    if (mlir::ShapedType::isDynamic(intDim) || intDim == -1)
      return {};
    outputNumElems *= intDim;
  }

  // Reshape is valid if number of elements match.
  if (shapeNumElems == outputNumElems)
    return foldReshape(outIntDims, inputArr);

  return {};
}

LogicalResult ReshapeOp::canonicalize(ReshapeOp op, PatternRewriter &rewriter) {
  TensorValue inputVal = op.getInput();
  TensorType inputTy = inputVal.getType();
  if (inputTy.hasUnknownShape())
    return failure();

  TensorType resTy = op.getType();
  if (!inputTy.hasRank() || !resTy.hasRank())
    return failure();

  // TODO(#11269) Emit runtime checks.
  // [] --> [?]
  if (inputTy.getRank() == 0 && resTy.getRank() == 1) {
    auto inferredResTy =
        resTy.cloneWith(SmallVector<int64_t>({1}), resTy.getElementType());
    auto newReshapeOp = rewriter.create<Static::ReshapeOp>(
        op.getLoc(), inferredResTy, inputVal);
    rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, newReshapeOp,
                                          op.getOutputParamDecls());
    return success();
  }

  // [?] --> []
  if (inputTy.getRank() == 1 && resTy.getRank() == 0) {
    auto inferredResTy =
        resTy.cloneWith(SmallVector<int64_t>({}), resTy.getElementType());
    auto newReshapeOp = rewriter.create<Static::ReshapeOp>(
        op.getLoc(), inferredResTy, inputVal);
    rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, newReshapeOp,
                                          op.getOutputParamDecls());
    return success();
  }

  if (resTy.hasUnknownShape() || !op.getOutputParamDecls().empty())
    return failure();

  rewriter.replaceOpWithNewOp<Static::ReshapeOp>(op, resTy, inputVal);
  return success();
}

//===----------------------------------------------------------------------===//
// ShapeOf
//===----------------------------------------------------------------------===//
LogicalResult ShapeOfOp::canonicalize(ShapeOfOp op, PatternRewriter &rewriter) {
  TensorType inputTy = op.getInput().getType();
  TensorType resTy = op.getResult().getType();
  if (inputTy.hasUnknownShape() || !inputTy.hasRank() ||
      resTy.hasUnknownShape())
    return failure();

  auto moshVal = rewriter.create<MOSH::ParamToValueOp>(op.getLoc(),
                                                       inputTy.getShapeAttr());
  rewriter.replaceOpWithNewOp<MO::Shape::ToTensorOp>(op, resTy, moshVal);
  return success();
}

//===----------------------------------------------------------------------===//
// Shape::FromTensorOp
//===----------------------------------------------------------------------===//
LogicalResult Shape::FromTensorOp::canonicalize(Shape::FromTensorOp op,
                                                PatternRewriter &rewriter) {
  // mo.shape.from_tensor(mo.shape.to_tensor(x)) == x
  auto input = op.getInput();
  if (auto toTensor = input.getDefiningOp<Shape::ToTensorOp>()) {
    if (toTensor.getInput().getType() == op.getType()) {
      rewriter.replaceOp(op, toTensor.getInput());
      return success();
    }
  }

  // fold mo.shape.from_tensor(<constant>)
  IntArrayElementsAttr newShape;
  if (mlir::matchPattern(input, mlir::m_Constant(&newShape))) {
    auto shape = ShapeAttr::get(newShape, op.getType());
    rewriter.replaceOpWithNewOp<MOSH::ParamToValueOp>(op, shape);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape::ToTensorOp
//===----------------------------------------------------------------------===//
LogicalResult Shape::ToTensorOp::canonicalize(Shape::ToTensorOp op,
                                              PatternRewriter &rewriter) {
  auto input = op.getInput();
  if (auto paramToValueOp = input.getDefiningOp<MOSH::ParamToValueOp>()) {
    auto inputShapeAttr = dyn_cast<MOSH::ShapeAttr>(paramToValueOp.getValue());
    if (!inputShapeAttr)
      return failure();

    TensorType resTy = op.getType();
    if (!resTy.hasStaticShape()) {
      TensorType newTy = resTy.cloneWith(ShapeAttr::get(
          {(int64_t)inputShapeAttr.getRank()}, rewriter.getType<ShapeType>()));
      Location loc = op.getLoc();
      auto newOp =
          rewriter.create<Shape::ToTensorOp>(loc, newTy, paramToValueOp);
      rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, newOp);
      return success();
    }

    if (inputShapeAttr.isConstant()) {
      auto outTy =
          RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
      SmallVector<APInt> res =
          llvm::map_to_vector(inputShapeAttr.getIntDims(), [](int64_t dim) {
            return APInt(64, dim, /*isSigned=*/true);
          });
      auto elementsAttr =
          cast<ElementsAttr>(IntArrayElementsAttr::get(outTy, res));
      rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getType(), elementsAttr);
      return success();
    }
  }

  // mo.shape.to_tensor(mo.shape.from_tensor(x)) => x
  if (auto fromTensor = input.getDefiningOp<Shape::FromTensorOp>()) {
    if (fromTensor.getInput().getType() == op.getType()) {
      rewriter.replaceOp(op, fromTensor.getInput());
      return success();
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//
OpFoldResult SliceOp::fold(FoldAdaptor adaptor) {
  return foldSliceHelper(*this, adaptor);
}

//===----------------------------------------------------------------------===//
// SliceDimOp
//===----------------------------------------------------------------------===//
OpFoldResult SliceDimOp::fold(FoldAdaptor adaptor) {
  return foldSliceHelper(*this, adaptor);
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//
OpFoldResult SelectOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?
  // TODO refactor parts of this with other folders

  // If the x and y are the same it's an identity op regardless of condition.
  if (getX() == getY())
    return getX();

  auto condAttr = dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getCond());
  auto xAttr = dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getX());
  auto yAttr = dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getY());
  if (!condAttr)
    return {};

  DType condDType = getCond().getType().getDtype();

  // If either or both inputs are unknown we can still fold symbolically. If
  // the condition is known to be all of one value then we can just return X if
  // the cond is true or Y if the cond is false.
  if (!xAttr || !yAttr) {
    auto maybeCond =
        tryGetConstantBlob<int64_t>(cast<ElementsAttr>(condAttr), condDType);
    if (!maybeCond.has_value())
      return {};

    ArrayRef<int64_t> condArr = *maybeCond;

    int64_t value = condArr[0];

    // Check the values are all the same otherwise exit.
    for (size_t idx = 1, e = condAttr.size(); idx < e; idx++) {
      if (value != condArr[idx])
        return {};
    }
    return value ? getX() : getY();
  }

  TensorType resTy = getType();
  DType resDType = resTy.getDtype();
  auto maybeCond =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(condAttr), condDType);
  auto maybeX =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(xAttr), resDType);
  auto maybeY =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(yAttr), resDType);
  if (!maybeX.has_value() || !maybeY.has_value())
    return {};

  ArrayRef<int64_t> condArr = *maybeCond;
  ArrayRef<int64_t> xArr = *maybeX;
  ArrayRef<int64_t> yArr = *maybeY;
  SmallVector<APInt> res(condAttr.size());
  for (size_t idx = 0, e = condAttr.size(); idx < e; idx++) {
    int64_t resVal = condArr[idx] ? xArr[idx] : yArr[idx];
    res[idx] = APInt(64, resVal, /*isSigned=*/true);
  }

  auto rankedResTy =
      RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
  return IntArrayElementsAttr::get(rankedResTy, res);
}

//===----------------------------------------------------------------------===//
// SplitDimOp
//===----------------------------------------------------------------------===//
LogicalResult SplitDimOp::canonicalize(SplitDimOp op,
                                       PatternRewriter &rewriter) {
  int64_t axis = op.getAxis().getLimitedValue();
  TensorType resultTy = op.getResult().getType();

  if (!resultTy.hasRank())
    return rewriter.notifyMatchFailure(op->getLoc(),
                                       "Result has dynamic rank.");

  if (resultTy.getIntDim(axis) == 1) {
    rewriter.replaceOpWithNewOp<AddSingletonDimOp>(
        op, resultTy, op.getInput(),
        IntegerAttr::get(op.getAxisAttr().getType(), axis));
    return success();
  }
  if (resultTy.getIntDim(axis + 1) == 1) {
    rewriter.replaceOpWithNewOp<AddSingletonDimOp>(
        op, resultTy, op.getInput(),
        IntegerAttr::get(op.getAxisAttr().getType(), axis + 1));
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//
LogicalResult SplitOp::fold(FoldAdaptor adaptor,
                            SmallVectorImpl<OpFoldResult> &results) {
  // If the split is splitting into one tensor then it is identity.
  if (getNumResults() == 1 && getInput().getType() == getResult(0).getType()) {
    results.push_back(getInput());
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// SqueezeShapeOp
//===----------------------------------------------------------------------===//
LogicalResult SqueezeShapeOp::canonicalize(SqueezeShapeOp op,
                                           PatternRewriter &rewriter) {
  // Normalize negative indices and make result in sorted order
  TensorValue indicesValue = op.getRemoveIndices();
  TensorType inputShapeType = op.getInputShape().getType();
  std::optional<SmallVector<int64_t>> maybeIndices =
      tryGetConstantIntVector(indicesValue);
  if (!maybeIndices || !inputShapeType.getNumElements().has_value())
    return failure();

  SmallVector<int64_t> &indices = maybeIndices.value();
  bool changedIndices = false;
  for (int64_t &index : indices) {
    if (index < 0) {
      index += inputShapeType.getNumElements().value();
      changedIndices = true;
    }
  }

  bool isSorted = true;
  for (size_t i = 0, e = indices.size() - 1; i < e; ++i)
    isSorted = indices[i] <= indices[i + 1];
  if (!isSorted) {
    llvm::sort(indices);
    changedIndices = true;
  }

  if (!changedIndices)
    return failure();

  // Construct new op with same dtypes for indices
  TensorValue newIndices = rewriter.create<ConstantOp>(
      op->getLoc(), indices, inputShapeType.getDeviceRef());
  newIndices = rewriter.create<CastOp>(op->getLoc(), newIndices,
                                       indicesValue.getType().getDtype());
  rewriter.replaceOpWithNewOp<SqueezeShapeOp>(op, op.getResult().getType(),
                                              op.getInputShape(), newIndices);
  return success();
}

//===----------------------------------------------------------------------===//
// Static::ReshapeOp
//===----------------------------------------------------------------------===//
OpFoldResult Static::ReshapeOp::fold(FoldAdaptor adaptor) {
  if (auto inputArr =
          dyn_cast_if_present<ArrayElementsAttr>(adaptor.getInput())) {
    auto outputShape = dyn_cast<ShapeAttr>(getType().getShapeAttr());
    if (outputShape && outputShape.isStatic())
      return foldReshape(outputShape.getIntDims(), inputArr);
  }

  if (getInput().getType() == getType())
    return getInput();

  return {};
}

LogicalResult Static::ReshapeOp::canonicalize(Static::ReshapeOp op,
                                              PatternRewriter &rewriter) {
  // We fold consecutive static reshapes into a single reshape.
  if (auto precedingOp = op.getInput().getDefiningOp<Static::ReshapeOp>()) {
    rewriter.modifyOpInPlace(op,
                             [&] { op.setOperand(precedingOp.getInput()); });
    return success();
  }

  TensorValue inputVal = op.getInput();
  TensorType inputTy = inputVal.getType();
  TensorType resTy = op.getType();
  if (!resTy.hasRank() || !inputTy.hasRank())
    return failure();

  // If we are preceded by ranked -> ranked rebind, we fold that into this op.
  if (auto rebindOp = inputVal.getDefiningOp<RebindOp>()) {
    TensorValue rebIn = rebindOp.getInput();
    TensorType rebInTy = rebIn.getType();
    if (rebInTy.hasRank() && !rebInTy.hasUnknownShape()) {
      rewriter.modifyOpInPlace(op, [&] { op.setOperand(rebIn); });
      return success();
    }
  }

  size_t inputRank = inputTy.getRank();
  size_t resRank = resTy.getRank();
  if (inputRank == 0 && resRank == 1) {
    // We can fold static reshapes from scalars to singletons.
    if (TypedAttr param = getParamAttrThrough<IndexToTensorOp>(inputVal)) {
      auto shape = ShapeAttr::get(SmallVector<TypedAttr>{param},
                                  rewriter.getType<ShapeType>());
      auto newConstOp =
          rewriter.create<MOSH::ParamToValueOp>(op->getLoc(), shape);
      rewriter.replaceOpWithNewOp<Shape::ToTensorOp>(op, resTy, newConstOp);
      return success();
    }
  }
  if (inputRank == 1 && resRank == 0) {
    // We can fold static reshapes from singletons to scalars.
    if (TypedAttr param = getParamAttrThrough<Shape::ToTensorOp>(inputVal)) {
      if (auto shape = dyn_cast<ShapeAttr>(param);
          shape && shape.getRank() == 1) {
        auto newConstOp =
            rewriter.create<MOSH::ParamToValueOp>(op->getLoc(), shape[0]);
        rewriter.replaceOpWithNewOp<IndexToTensorOp>(op, resTy, newConstOp);
        return success();
      }
    }
  }

  if (inputRank != resRank)
    return failure();

  // A reshape that changes only a single dimension is just a rebind.
  bool foundDifferentDim = false;
  for (auto [inDim, resDim] : llvm::zip(inputTy.getDims(), resTy.getDims())) {
    if (inDim != resDim) {
      if (foundDifferentDim)
        return failure();
      foundDifferentDim = true;
    }
  }

  rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, op.getInput());
  return success();
}

//===----------------------------------------------------------------------===//
// Static::BroadcastToOp
//===----------------------------------------------------------------------===//
LogicalResult Static::BroadcastToOp::canonicalize(Static::BroadcastToOp op,
                                                  PatternRewriter &rewriter) {
  TensorType resTy = op.getType();
  TensorValue inputVal = op.getInput();
  TensorType inputTy = inputVal.getType();

  if (auto pastBroadcastOp =
          op.getInput().getDefiningOp<Static::BroadcastToOp>()) {
    rewriter.replaceOpWithNewOp<Static::BroadcastToOp>(
        op, op.getType(), pastBroadcastOp.getInput());
    return success();
  }

  if (resTy.getShapeAttr() == inputTy.getShapeAttr()) {
    // Be careful not to drop any non-shape info like metadata in TensorType.
    if (resTy == inputTy)
      rewriter.replaceAllUsesExcept(op, inputVal, op);
    else
      rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, inputVal);

    return success();
  }

  if (!inputTy.hasRank() || !resTy.hasRank())
    return failure();

  // If a broadcast has an empty input, it is not equivalent to a reshape.
  auto resTyAttr = dyn_cast<IntegerAttr>(resTy.getSymbolicNumElements());
  auto inputTyAttr = dyn_cast<IntegerAttr>(inputTy.getSymbolicNumElements());
  if ((resTyAttr && resTyAttr.getSInt() == 0) ||
      (inputTyAttr && inputTyAttr.getSInt() == 0))
    return failure();

  // If a broadcast does not change the number of elements, it is equivalent to
  // a reshape. This is something we can check symbolically.
  if (resTy.getSymbolicNumElements() == inputTy.getSymbolicNumElements()) {
    rewriter.replaceOpWithNewOp<Static::ReshapeOp>(op, resTy, inputVal);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//
LogicalResult TileOp::canonicalize(TileOp op, PatternRewriter &rewriter) {
  TensorType inputTy = op.getInput().getType();
  if (!inputTy.hasRank())
    return failure();

  // Extract constant repeats.
  auto maybeRepeatsVec = tryGetConstantIntVector(op.getRepeats());
  if (!maybeRepeatsVec.has_value())
    return failure();
  ArrayRef<int64_t> repeatsVec(maybeRepeatsVec.value());

  // Check if the tile operation can be replaced with a broadcast. Each repeats
  // value leaves the input dimension unchanged or the input dimension must be
  // a broadcast.
  for (auto [inDim, repeats] : llvm::zip(inputTy.getIntDims(), repeatsVec)) {
    if (repeats < 1 || (repeats > 1 && inDim != 1))
      return failure();
  }

  TensorType shapeTy =
      inputTy.cloneWith({(int64_t)inputTy.getRank()},
                        rewriter.getIntegerType(64, /*isSigned=*/true));
  TensorValue inputShape =
      rewriter.create<ShapeOfOp>(op.getLoc(), shapeTy, op.getInput());
  TensorValue bcastShape = rewriter.create<BroadcastShapeOp>(
      op.getLoc(), shapeTy, inputShape, op.getRepeats());
  rewriter.replaceOpWithNewOp<BroadcastToOp>(
      op, op.getType(), op.getInput(), bcastShape, op.getOutputParamDecls());
  return success();
}

//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//
LogicalResult TransferOp::canonicalize(TransferOp op,
                                       PatternRewriter &rewriter) {
  // transfer(transfer(input)) = input, provided devices line up.
  TypedValue<TensorType> input = op.getInput();
  TypedValue<TensorType> output = op.getResult();
  if (auto prevOp = input.getDefiningOp<TransferOp>()) {
    TypedValue<TensorType> prevOpInput = prevOp.getInput();
    if (output.getType().getDeviceRef() ==
        prevOpInput.getType().getDeviceRef()) {
      rewriter.replaceOp(op, ValueRange{prevOpInput, prevOp.getInChain()});
      if (prevOp->use_empty()) {
        // Erase the now unused previous transfer as well.
        rewriter.eraseOp(prevOp);
      }
      return success();
    }
  }

  // If the source and destination device are the same, we can elide.
  if (op.getAlwaysElideSameDeviceCopy() &&
      input.getType() == output.getType()) {
    // Replace uses of `op`'s outChain with its inChain to preserve ordering.
    rewriter.replaceOp(op, ValueRange{input, op.getInChain()});
    return success();
  }

  DeviceRefAttr inputDevice = input.getType().getDeviceRef();
  // If the transfer feeds directly into a cast to a larger bit type then we
  // should rotate the cast to cast first on the CPU then transfer the smaller
  // size.
  if (output.hasOneUse() && inputDevice == getHostDevice(op)) {
    Operation *user = *output.getUsers().begin();

    if (auto cast = dyn_cast<MO::CastOp>(user)) {
      TypedValue<TensorType> castOut = cast.getResult();

      Type srcTy = output.getType().getElementType();
      Type dstTy = castOut.getType().getElementType();

      if (srcTy.isF32() && dstTy.isBF16()) {
        // Switch the cast to run on the cpu and convert the transfer input.
        cast->setOperand(0, input);
        castOut.setType(castOut.getType().cloneWith(inputDevice));

        // Make the transfer take the cast as input.
        op->setOperand(0, castOut);
        output.setType(
            output.getType().cloneWith(castOut.getType().getDtype()));

        cast->moveBefore(op);
        castOut.replaceAllUsesExcept(output, op);
        return success();
      }
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
LogicalResult TransposeOp::canonicalize(TransposeOp op,
                                        PatternRewriter &rewriter) {
  // Nothing we can do if op is dynamically ranked.
  if (!op.getType().hasRank())
    return failure();
  size_t opRank = op.getType().getRank();

  // Extract constant permutation.
  auto maybePermVec = tryGetConstantIntVector(op.getPerm());
  if (!maybePermVec.has_value())
    return failure();
  ArrayRef<int64_t> permVec(maybePermVec.value());

  // Fold away identity permutation, e.g., [0, 1, 2] for 3D transpose.
  auto identityPerm = llvm::iota_range<int64_t>(0, opRank, /*Inclusive=*/false);
  if (llvm::equal(permVec, identityPerm)) {
    rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(), op.getInput(),
                                          op.getOutputParamDecls());
    return success();
  }

  // NOTE The following (transposes merging or transpose-->reshape) assumes
  // the verifier checks against invalid permutations.

  // Merge consecutive transposes by merging their constant permutations
  if (auto inputTransposeOp = op.getInput().getDefiningOp<TransposeOp>()) {
    auto maybeInputPermVec =
        tryGetConstantIntVector(inputTransposeOp.getPerm());
    if (!maybeInputPermVec.has_value())
      return failure();
    ArrayRef<int64_t> inputPermVec(maybeInputPermVec.value());

    // Compute the merged permutation, and replace the consecutive transpose
    // with a new transpose with the merged permutation.
    SmallVector<int64_t> mergedPerm = permute(inputPermVec, permVec);
    auto mergedPermVal = rewriter.create<ConstantOp>(
        op.getLoc(), mergedPerm, op.getPerm().getType().getDeviceRef());
    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, op.getType(), inputTransposeOp.getInput(), mergedPermVal,
        op.getOutputParamDecls());
    return success();
  }

  // Transpose == Reshape if the relative order of non-1 dims remains the same
  // e.g. transpose from shape [8, 1, 1, 5, 1, 1, 1, 50] to
  // - [8, 1, 1, 5, 1, 1, 50, 1]
  // - [1, 1, 1, 8, 5, 50, 1, 1]
  // - [8, 5, 50, 1, 1, 1, 1, 1]
  // replacing with equivalent reshapes are all allowed
  int64_t indexLastNonOneDim = -1; // should always be increasing
  MOSH::ShapeAttr inputShape = op.getInput().getType().getTypedShapeAttr();
  bool meetsReshapeRewriteCriteria = inputShape != nullptr;
  for (int64_t inputDimIndex : permVec) {
    if (!meetsReshapeRewriteCriteria) {
      break;
    }
    int64_t dimInteger = inputShape.getIntDim(inputDimIndex);
    if (dimInteger == mlir::ShapedType::kDynamic) {
      // violates static reshape which only allows int dims
      meetsReshapeRewriteCriteria = false;
    } else if (dimInteger != 1) {
      if (inputDimIndex < indexLastNonOneDim)
        meetsReshapeRewriteCriteria = false;
      indexLastNonOneDim = inputDimIndex;
    }
  }
  if (meetsReshapeRewriteCriteria) {
    rewriter.replaceOpWithNewOp<Static::ReshapeOp>(op, op.getType(),
                                                   op.getInput());
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//
namespace {

struct RemoveDuplicateLoopCarriedValues : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  // Compute a mapping of duplicated operands back to their first occurrence
  // in the operand sequence.
  static llvm::DenseMap<unsigned, unsigned> duplicateMap(ValueRange inputs) {
    llvm::DenseMap<Value, unsigned> firstIndex;
    llvm::DenseMap<unsigned, unsigned> result;
    for (auto [index, value] : llvm::enumerate(inputs)) {
      auto [it, inserted] = firstIndex.try_emplace(value, index);
      if (!inserted)
        result[index] = it->second;
    }
    return result;
  }

  // Two values in the same block are duplicated if
  // 1. Their backing values are identical,
  // 2. They refer to block arguments that are known to be duplicates
  static bool isDuplicate(Block *block, ValueRange values, unsigned index,
                          unsigned firstOccurrence) {
    auto x = values[index];
    auto y = values[firstOccurrence];

    // Same values count as duplicates
    if (x == y)
      return true;

    auto blockArgX = dyn_cast<BlockArgument>(x);
    auto blockArgY = dyn_cast<BlockArgument>(y);

    // Block arguments that correspond to duplicate input operands also
    // count as duplicates.
    return blockArgX && blockArgY && blockArgX.getParentBlock() == block &&
           blockArgY.getParentBlock() == block &&
           blockArgX.getArgNumber() == index &&
           blockArgY.getArgNumber() == firstOccurrence;
  };

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    WhileConditionOp condOp = op.getCondTerminator();
    YieldOp yieldOp = op.getBodyTerminator();

    ValueRange whileOpResults = op.getResults();

    Block::BlockArgListType condBlockArgs = op.getCondArguments();
    Block::BlockArgListType bodyBlockArgs = op.getBodyArguments();

    ValueRange condOpArgs = condOp.getArgs();
    ValueRange yieldOpArgs = yieldOp.getOperands();

    // The duplication map acts as a seed for checking duplicates in the body
    // and condition regions. Every duplicate operand must alias in an identical
    // way on the region terminators.
    auto duplicatedWhileOperands = duplicateMap(op.getOperands());
    if (duplicatedWhileOperands.empty())
      return rewriter.notifyMatchFailure(op, "no operands to remove");

    llvm::SmallVector<Value> operandsToKeep;
    llvm::SmallVector<Value> resultsToKeep;
    llvm::BitVector toErase(op.getNumOperands());

    // Check if each duplication candidate from the operands is also a
    // duplicate within the predicate and body regions of the while op.
    for (unsigned index : llvm::seq(op.getNumOperands())) {
      auto it = duplicatedWhileOperands.find(index);
      if (it == duplicatedWhileOperands.end()) {
        operandsToKeep.push_back(op.getOperand(index));
        resultsToKeep.push_back(op.getResult(index));
        continue;
      }

      unsigned firstOccurrence = it->second;
      // First check if cond op has duplicate candidates
      if (!isDuplicate(op.getCondBlock(), condOpArgs, index, firstOccurrence) ||
          !isDuplicate(op.getBodyBlock(), yieldOpArgs, index,
                       firstOccurrence)) {
        operandsToKeep.push_back(op.getOperand(index));
        resultsToKeep.push_back(op.getResult(index));
        continue;
      }

      // Found some duplicated operands/results, remap them to refer to the
      // first occurrence of that value.
      toErase.set(index);
      rewriter.replaceAllUsesWith(condBlockArgs[index],
                                  condBlockArgs[firstOccurrence]);
      rewriter.replaceAllUsesWith(bodyBlockArgs[index],
                                  bodyBlockArgs[firstOccurrence]);
      rewriter.replaceAllUsesWith(whileOpResults[index],
                                  whileOpResults[firstOccurrence]);
    }

    if (toErase.none())
      return rewriter.notifyMatchFailure(op,
                                         "no duplicate loop carried values");

    // Construct the new arguments for the mo.while.condition and mo.yield
    // terminators (ops do not have nice functions to erase arguments).
    SmallVector<Value> newCondArgs;
    SmallVector<Value> newYieldArgs;

    for (auto index : llvm::seq(op.getNumOperands())) {
      if (!toErase.test(index)) {
        newCondArgs.push_back(condOp.getArgs()[index]);
        newYieldArgs.push_back(yieldOpArgs[index]);
      }
    }

    // Erase the duplicated block arguments
    op.getCondBlock()->eraseArguments(toErase);
    op.getBodyBlock()->eraseArguments(toErase);

    // Erase the duplicated terminator arguments
    condOp.getArgsMutable().assign(newCondArgs);
    yieldOp.getOperandsMutable().assign(newYieldArgs);

    auto newWhileOp = WhileOp::create(
        rewriter, op.getLoc(), TypeRange(operandsToKeep), operandsToKeep);

    newWhileOp.getCondRegion().takeBody(op.getCondRegion());
    newWhileOp.getBodyRegion().takeBody(op.getBodyRegion());

    rewriter.replaceAllUsesWith(resultsToKeep, newWhileOp.getResults());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Find unused results of the while loop that are also unused in the
/// body block and remove them.
///
/// Results are produced by the `mo.while.condition` operation, which yields
/// values to the parent `mo.while` operation and to the body region.
struct RemoveLoopInvariantCarriedValues : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  /// Can the given value to the terminator op be simplified if it appears as
  /// the index'th operand to the terminator.
  static bool canSimplifyValue(Value val, Value initVal, Block *block,
                               unsigned index) {
    // If the operand correspond to the initial value at the given index,
    // then the value can be simplified.
    if (val == initVal)
      return true;

    // Check if the given value is a trivial forwarding of the index'th block
    // argument from the parent block.
    auto blockArg = dyn_cast<BlockArgument>(val);
    if (blockArg && blockArg.getParentBlock() == block &&
        index == blockArg.getArgNumber())
      return true;

    return false;
  }

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    Block::BlockArgListType condBlockArgs = op.getCondArguments();
    Block::BlockArgListType bodyBlockArgs = op.getBodyArguments();
    WhileConditionOp condOp = op.getCondTerminator();
    OperandRange condOpArgs = condOp.getArgs();
    YieldOp yieldOp = op.getBodyTerminator();
    ValueRange yieldOpArgs = yieldOp.getOperands();

    // Perform a simplification check before proceeding to avoid allocating
    // all the extra dstat structures which may be unused.
    bool canSimplify = false;
    for (const auto [index, initVal, yieldOpArg, condOpArg] :
         llvm::enumerate(op.getOperands(), yieldOpArgs, condOpArgs)) {

      // We can simplify the index'th loop carried value if both the YieldOp's
      // and WhileConditionOp's index'th value can be simplified.
      canSimplify |=
          canSimplifyValue(yieldOpArg, initVal, yieldOp->getBlock(), index) &&
          canSimplifyValue(condOpArg, initVal, condOp->getBlock(), index);
    }

    if (!canSimplify)
      return rewriter.notifyMatchFailure(
          op, "no loop carried values can be simplified");

    SmallVector<Value> newInitArgs, newYieldOpArgs, newCondOpArgs;
    SmallVector<Location> newCondBlockArgLocs, newBodyBlockArgLocs;

    SmallVector<Value> blockInitValMap(condOpArgs.size());
    for (const auto [index, initVal, yieldOpArg, condOpArg] :
         llvm::enumerate(op.getOperands(), yieldOpArgs, condOpArgs)) {

      bool simplifyArg =
          canSimplifyValue(yieldOpArg, initVal, yieldOp->getBlock(), index) &&
          canSimplifyValue(condOpArg, initVal, condOp->getBlock(), index);

      // If the index'th loop carried value can be simplified, record the
      // mapping to the corresponding initVal.
      if (simplifyArg) {
        blockInitValMap[index] = initVal;
        continue;
      }

      newInitArgs.emplace_back(initVal);
      newYieldOpArgs.emplace_back(yieldOpArg);
      newCondOpArgs.emplace_back(condOpArg);
      newCondBlockArgLocs.emplace_back(condBlockArgs[index].getLoc());
      newBodyBlockArgLocs.emplace_back(bodyBlockArgs[index].getLoc());
    }

    { // Update body YieldOp
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(yieldOp);
      rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, newYieldOpArgs);
    }

    { // Update cond WhileConditionOp
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(condOp);
      rewriter.replaceOpWithNewOp<WhileConditionOp>(
          condOp, condOp.getCondition(), newCondOpArgs);
    }

    auto newWhile = rewriter.create<WhileOp>(
        op.getLoc(), TypeRange(newInitArgs), newInitArgs);

    Block &newCondBlock = *rewriter.createBlock(
        &newWhile.getCondRegion(), /*insertPt*/ {},
        ValueRange(newYieldOpArgs).getTypes(), newCondBlockArgLocs);

    Block &newBodyBlock = *rewriter.createBlock(
        &newWhile.getBodyRegion(), /*insertPt*/ {},
        ValueRange(newYieldOpArgs).getTypes(), newBodyBlockArgLocs);

    Block &condBlock = *op.getCondBlock();
    Block &bodyBlock = *op.getBodyBlock();

    SmallVector<Value> newCondBlockArgs(condBlock.getNumArguments());
    SmallVector<Value> newBodyBlockArgs(condBlock.getNumArguments());
    SmallVector<Value> newResults(condBlock.getNumArguments());

    // For each i-th before block argument we find it's replacement value as :-
    //   1. If i-th before block argument is a loop invariant, we fetch it's
    //      initial value from `beforeBlockInitValMap` by querying for key `i`.
    //   2. Else we fetch j-th new before block argument as the replacement
    //      value of i-th before block argument.
    for (unsigned i = 0, j = 0, n = condBlock.getNumArguments(); i < n; i++) {
      // If the index 'i' argument was a loop invariant we fetch it's initial
      // value from `beforeBlockInitValMap`.
      if (blockInitValMap[i] != nullptr) {
        newCondBlockArgs[i] = blockInitValMap[i];
        newBodyBlockArgs[i] = blockInitValMap[i];
        newResults[i] = blockInitValMap[i];
      } else {
        newCondBlockArgs[i] = newCondBlock.getArgument(j);
        newBodyBlockArgs[i] = newBodyBlock.getArgument(j);
        newResults[i] = newWhile.getResult(j);
        j++;
      }
    }

    // Inline new blocks with updated block arguments
    rewriter.mergeBlocks(&condBlock, &newCondBlock, newCondBlockArgs);
    rewriter.mergeBlocks(&bodyBlock, &newBodyBlock, newBodyBlockArgs);

    // Replace old op with the new results
    rewriter.replaceOp(op, newResults);
    return success();
  }
};

} // end namespace

void WhileOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns
      .add<RemoveLoopInvariantCarriedValues, RemoveDuplicateLoopCarriedValues>(
          ctx);
}

} // namespace M::MO

//===----------------------------------------------------------------------===//
// UnsqueezeShapeOp
//===----------------------------------------------------------------------===//
LogicalResult UnsqueezeShapeOp::canonicalize(UnsqueezeShapeOp op,
                                             PatternRewriter &rewriter) {
  // Normalize negative indices and make result in sorted order
  TensorValue indicesValue = op.getPaddingIndices();
  TensorType inputShapeType = op.getInputShape().getType();
  std::optional<SmallVector<int64_t>> maybeIndices =
      tryGetConstantIntVector(indicesValue);
  if (!maybeIndices || !inputShapeType.getNumElements().has_value())
    return failure();
  SmallVector<int64_t> &indices = maybeIndices.value();
  bool changedIndices = false;
  for (int64_t &index : indices) {
    if (index < 0) {
      index += inputShapeType.getNumElements().value() + indices.size();
      changedIndices = true;
    }
  }
  bool isSorted = true;
  for (size_t i = 0; i < indices.size() - 1 && isSorted; i++)
    isSorted = indices[i] <= indices[i + 1];
  if (!isSorted) {
    llvm::sort(indices);
    changedIndices = true;
  }
  if (!changedIndices)
    return failure();
  // Construct new op with same dtypes for indices
  TensorValue newIndices = rewriter.create<ConstantOp>(
      op->getLoc(), indices, inputShapeType.getDeviceRef());
  newIndices = rewriter.create<CastOp>(op->getLoc(), newIndices,
                                       indicesValue.getType().getDtype());
  rewriter.replaceOpWithNewOp<UnsqueezeShapeOp>(op, op.getResult().getType(),
                                                op.getInputShape(), newIndices);
  return success();
}

namespace M::MO {

namespace {
/// A scatter-like operation with an empty indices tensor is a no-op.
struct FoldNoOpPadLikeOp : public OpInterfaceRewritePattern<PadLike> {
  using OpInterfaceRewritePattern<PadLike>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(PadLike op,
                                PatternRewriter &rewriter) const override {
    // Try to extract constant paddings
    auto maybePaddings = tryGetConstantIntVector(op.getPaddings());
    if (!maybePaddings.has_value())
      return failure();
    ArrayRef<int64_t> paddings = maybePaddings.value();
    // If paddings are all 0s, the pad operation is identity. We replace it with
    // a rebind from the input because the output shape of `mo.pad` might not
    // always equal to the input shape, e.g., it could be fully dynamic from the
    // frontends, e.g.,
    //     mo.pad(input, zeros, val) : ... -> tensor<[?, ?, ?], f32>
    if (llvm::all_of(paddings, [](int64_t n) { return n == 0; })) {
      rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(), op.getInput(),
                                            op.getOutputParamDecls());
      return success();
    }
    return failure();
  }
};

/// A scatter-like operation with an empty indices tensor is a no-op.
struct FoldNoOpScatterLikeOp : public OpInterfaceRewritePattern<ScatterLike> {
  using OpInterfaceRewritePattern<ScatterLike>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(ScatterLike op,
                                PatternRewriter &rewriter) const override {
    if (op.getInput().getType() != op.getResult().getType())
      return rewriter.notifyMatchFailure(op,
                                         "input and result types do not match");

    std::optional<int64_t> numIndices =
        op.getIndices().getType().getNumElements();

    if (numIndices != 0)
      return rewriter.notifyMatchFailure(op, "non-empty indices tensor");

    rewriter.replaceAllOpUsesWith(op, op.getInput());
    return success();
  }
};
} // namespace

void MODialect::getCanonicalizationPatterns(RewritePatternSet &patterns) const {
  patterns.add<FoldNoOpPadLikeOp, FoldNoOpScatterLikeOp>(patterns.getContext());
}

} // namespace M::MO
