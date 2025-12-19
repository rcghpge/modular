//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MODialect/MOAttrs.h"
#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MojoParameters.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENStubBuilderHelpers.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOGGTypeTraits.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoTargetFunctions.h"
#include "GenericML/Support/DataflowUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MOGGPreElab/MojoKernelAdaptor.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/AssertStream.h"
#include "llvm/ADT/StringSet.h"

namespace M::MOGG {

StringAttr getSummaryString(Operation *op) {
  if (auto kernelOp = dyn_cast<MOGG::KernelOp>(op)) {
    if (auto summaryAttr = kernelOp.getSummaryAttr())
      return summaryAttr;
  } else if (auto kernelOp = dyn_cast<MOGG::ExpKernelOp>(op)) {
    if (auto summaryAttr = kernelOp.getSummaryAttr())
      return summaryAttr;
  } else if (auto kernelOp = dyn_cast<MOGG::GenericCompiledRegionOp>(op)) {
    if (auto summaryAttr = kernelOp.getSummaryAttr())
      return summaryAttr;
  }

  std::string name;
  if (auto shapeFuncOp = dyn_cast<MO::InvokeShapeFuncOp>(op)) {
    name = "shapeof(";
    name += shapeFuncOp.getShapeFuncName();
    name += ")";
  } else if (auto layoutTransformOp = dyn_cast<MO::LayoutTransformOp>(op)) {
    name = layoutTransformOp->getName().getIdentifier().getValue();
    name += "_";
    name += layoutTransformOp.getInput()
                .getType()
                .getLayoutAttr()
                .getFormat()
                .getValue();
    name += "_to_";
    name += layoutTransformOp.getResult()
                .getType()
                .getLayoutAttr()
                .getFormat()
                .getValue();
  } else if (auto customOp = dyn_cast<MO::CustomOp>(op)) {
    name = "custom__";
    name += customOp.getSymbol();
  } else if (auto loadExternalOpMGP =
                 dyn_cast<MGP::BufferConstantExternalOp>(op)) {
    name = (llvm::Twine(op->getName().getIdentifier().getValue()) + "(" +
            loadExternalOpMGP.getBName() + ")")
               .str();
  } else {
    name = op->getName().getIdentifier().getValue();
  }

  if (auto fileLineLoc = dyn_cast<mlir::FileLineColLoc>(op->getLoc())) {
    if (fileLineLoc.getLine() > 0) {
      name += ":";
      name += std::to_string(fileLineLoc.getLine());
    }
  }

  return StringAttr::get(op->getContext(), name);
}

StringAttr getSummaryString(ArrayRef<Operation *> ops, FusionType fusionType) {
  if (ops.empty())
    return StringAttr::get(ops.front()->getContext(), "");

  if (ops.size() == 1)
    return getSummaryString(ops.front());

  std::string str;
  llvm::raw_string_ostream os(str);
  if (fusionType == FusionType::HARD) {
    llvm::interleave(
        ops, os,
        [&os](Operation *op) { os << getSummaryString(op).getValue(); }, "+");
  } else {
    ASSERT_STREAM(fusionType == FusionType::SOFT,
                  << "Expected soft fusion by default");
    for (auto [i, op] : llvm::enumerate(ops)) {
      os << "(" << getSummaryString(op).getValue() << ")";

      if (i != ops.size() - 1)
        os << "_";
    }
  }
  return StringAttr::get(ops.front()->getContext(), str);
}

// The max number of elements in a constant that we will allow to be fused into
// the body of a kernel.
constexpr int64_t kMaxConstantSize = 2048;

/// Try to get the original constant op that produced `val`, with optional
/// rebind (i.e., reinterpret cast) in the way.
static MO::ConstantOp tryGetConstantOp(Value val) {
  // Look through potential rebind op
  if (auto rebindOp = val.getDefiningOp<MO::RebindOp>())
    val = rebindOp.getInput();

  return val.getDefiningOp<MO::ConstantOp>();
}

/// Check whether the `index`-th operand of `kernel` is a fusable constant op.
static bool canFuseOperand(MOGG::ExpKernelOp kernel, int32_t index) {
  // HACK HACK HACK https://github.com/modularml/modular/issues/22959
  // Don't fuse constant unless on the host. TEMPORARY until constant
  // lowering fixed.
  if (!MO::isOnHost(kernel))
    return false;

  MO::ConstantOp cst = tryGetConstantOp(kernel.getOperand(index));
  if (!cst)
    return false;

  if (!MO::isOnHost(cst))
    return false;

  // Check it is a small constant with a attribute we understand.
  ElementsAttr elems = cst.getValueAttr();
  if (elems.getNumElements() > kMaxConstantSize)
    return false;

  Type dtype = elems.getElementType();
  if (!MOGGTypeTraits::isSupportedMOGGType(dtype))
    return false;

  if (!isa<M::ArrayElementsAttr, DenseResourceElementsAttr>(elems))
    return false;

  return true;

  return false;
}

/// Fold the constant at `idx`-th operand of `kernel` into the kernel body.
static void fuseInputAtIndex(MOGG::ExpKernelOp kernel, int32_t idx) {
  Value inputVal = kernel.getOperand(idx);
  MO::ConstantOp cst = tryGetConstantOp(inputVal);

  Block *block = kernel.getBody();

  mlir::OpBuilder builder{kernel->getContext()};
  builder.setInsertionPointToStart(block);

  // Clone the constant and add it to the kernel.
  MO::ConstantOp newOp = cst.clone();

  builder.insert(newOp);

  // Replace all uses of it with the now in-kernel constant.
  block->getArgument(idx).replaceAllUsesWith(newOp.getResult());

  // Remove the argument.
  kernel.eraseInputsAt(idx);

  // Kill the constant if that was the last use.
  if (cst->use_empty())
    cst->erase();
}

void fuseConstant(MOGG::ExpKernelOp kernel) {
  // Traverse from the back to simplify removing operands.
  int32_t preFuseNumOperands = kernel.getNumOperands();
  for (int32_t index = preFuseNumOperands - 1; index >= 0; --index) {
    if (canFuseOperand(kernel, index))
      fuseInputAtIndex(kernel, index);
  }
}

void stageKernelsForFusion(ArrayRef<Operation *> kernelOps) {
  static const std::string prefix = "_";
  llvm::DenseSet<StringAttr> parameters;

  for (Operation *kernelOp : kernelOps) {
    auto ctx = kernelOp->getContext();

    llvm::DenseMap<StringAttr, TypedAttr> nameToDecl;
    mlir::AttrTypeReplacer walker;
    walker.addReplacement([&](KGEN::ParamDeclAttr oldDecl) {
      std::string oldName =
          KGEN::LIT::demangleParameterName(oldDecl.getName()).str();

      StringAttr newName = StringAttr::get(ctx, oldName);
      while (parameters.contains(newName))
        newName = StringAttr::get(ctx, prefix + newName.strref());

      auto newDeclRef = KGEN::ParamDeclRefAttr::get(newName, oldDecl.getType());
      nameToDecl.insert({oldDecl.getName(), newDeclRef});
      parameters.insert(newDeclRef.getName());

      return KGEN::ParamDeclAttr::get(newDeclRef);
    });
    walker.recursivelyReplaceElementsIn(kernelOp, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);

    // mogg.to_kgen operations hold StringAttrs rather than ParamDeclRefAttr,
    // so those must be explicitly renamed to ensure consistency.
    kernelOp->walk([&](MOGG::ToKGENOp toKGEN) {
      if (std::optional<StringAttr> specName =
              toKGEN.getStaticTensorSpecName()) {
        if (auto it = nameToDecl.find(*specName); it != nameToDecl.end()) {
          StringAttr name = cast<KGEN::ParamDeclRefAttr>(it->second).getName();
          toKGEN.setStaticTensorSpecNameAttr(name);
        }
      }
    });

    MOGG::remapTypesAndAttrs(kernelOp, nameToDecl);

    if (auto asKernelOp = dyn_cast<KernelOp>(kernelOp)) {
      // Rename lambdas if they exist and they were renamed.
      for (uint64_t i = 0; i < asKernelOp->getNumOperands(); ++i) {
        if (auto lambdaNameAttr = asKernelOp.getInputFusionInterfaceAt(i)) {
          auto it = nameToDecl.find(lambdaNameAttr);
          if (it != nameToDecl.end()) {
            if (auto decl = dyn_cast<KGEN::ParamDeclRefAttr>(it->second))
              asKernelOp.setInputFusionInterface(i, decl.getName());
          }
        }
      }
    }
  }
}

void fuseAndReplaceCompiledRegions(bool allowDuplication, bool genReleaseOpaque,
                                   ArrayRef<Operation *> toFuse,
                                   fuseRegionInlineCallBackT inlineOpCallBack) {
  ASSERT_STREAM(llvm::all_of(toFuse,
                             [](auto op) {
                               return isa<MOGG::GenericCompiledRegionOp>(op);
                             }),
                << "Expected all operations to be "
                << MOGG::GenericCompiledRegionOp::getOperationName());

  MOGG::stageKernelsForFusion(toFuse);

  // Iterate through all the ops to fuse and gather all operand and results.
  llvm::SetVector<Value> allResults;
  llvm::SetVector<Value> allOperands;
  for (Operation *op : toFuse) {
    auto results = op->getResults();
    auto operands = op->getOperands();
    allResults.insert(results.begin(), results.end());
    allOperands.insert(operands.begin(), operands.end());
  }

  // The fused op's operands will be allOperands - allResults.
  llvm::SetVector<Value> fusedRegionOperands = allOperands;
  fusedRegionOperands.set_subtract(allResults);

  // Iterate through all the results and identify values that need to be return
  // values after fusion.
  llvm::SetVector<Value> resultsThatShouldBeReplaced;
  SmallVector<Type> fusedRegionOutputTypes;
  DenseSet<Operation *> compiledRegionOpsToFuseTogetherAsSet(toFuse.begin(),
                                                             toFuse.end());

  bool isFirstChain = false;
  for (auto result : allResults) {
    // We keep at least one chain output such that error could be propagated
    // properly. The duplicated chain will be removed later in the pipeline
    // (namely, remove-redundant-operands pass).
    isFirstChain = isa<MEF::ChainType>(result.getType()) && !isFirstChain;
    if (allowDuplication) {
      // If none of the users of this result are in the fusion set, then it is
      // something that is solely used from the outside.
      if (isFirstChain || llvm::none_of(result.getUsers(), [&](auto user) {
            return compiledRegionOpsToFuseTogetherAsSet.contains(user);
          })) {
        resultsThatShouldBeReplaced.insert(result);
        fusedRegionOutputTypes.push_back(result.getType());
      }
    } else {
      // If any of the users of this result are used outside, then it must be
      // output too.
      if (isFirstChain || llvm::any_of(result.getUsers(), [&](auto user) {
            return !compiledRegionOpsToFuseTogetherAsSet.contains(user);
          })) {
        resultsThatShouldBeReplaced.insert(result);
        fusedRegionOutputTypes.push_back(result.getType());
      }
    }
  }

  // Maintain attributes.
  auto firstRegionOp = cast<MOGG::GenericCompiledRegionOp>(toFuse.front());
  const bool isAsync = llvm::any_of(toFuse, [&](auto region) {
    return cast<MOGG::GenericCompiledRegionOp>(region).getIsAsync().value_or(
        false);
  });
  llvm::SmallSetVector<Attribute, 4> resources;
  for (Operation *regionOp : toFuse) {
    auto res = cast<MOGG::GenericCompiledRegionOp>(regionOp)
                   .getConstantResourcesAttr()
                   .getValue();
    resources.insert(res.begin(), res.end());
  }
  const bool canFuseIntoOther = llvm::all_of(toFuse, [&](auto region) {
    return cast<MOGG::GenericCompiledRegionOp>(region).getCanFuseIntoOther();
  });
  const bool canBeFusedInto = llvm::all_of(toFuse, [&](auto region) {
    return cast<MOGG::GenericCompiledRegionOp>(region).getCanBeFusedInto();
  });
  const bool canDuplicate = llvm::all_of(toFuse, [&](auto region) {
    return cast<MOGG::GenericCompiledRegionOp>(region).getCanDuplicate();
  });
  const bool isStructuredKernel = llvm::any_of(toFuse, [&](auto region) {
    return cast<MOGG::GenericCompiledRegionOp>(region).getIsStructuredKernel();
  });
  DeviceRefAttr device =
      cast<MOGG::GenericCompiledRegionOp>(toFuse.front()).getExecutionDevice();
  // For now we set the device to be a cuda device if one exists.
  for (auto region : toFuse) {
    auto regionDevice =
        cast<MOGG::GenericCompiledRegionOp>(region).getExecutionDevice();
    if (regionDevice.getLabel() == kGPULabel)
      device = regionDevice;
  }

  auto ctx = firstRegionOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointAfter(toFuse.back());

  SmallVector<Location> locs;
  for (auto region : toFuse)
    locs.push_back(region->getLoc());
  auto loc = mlir::FusedLoc::get(ctx, locs);

  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics;
  auto fusedCompiledRegionOp = MOGG::GenericCompiledRegionOp::create(
      builder, loc, fusedRegionOutputTypes, fusedRegionOperands.getArrayRef(),
      KGEN::ParamDeclArrayAttr::get(ctx, {}),
      InOutSignatureAttr::get(ctx, operandSemantics),
      /*summary=*/
      MOGG::getSummaryString(toFuse, MOGG::FusionType::SOFT),
      /*isAsync=*/BoolAttr::get(ctx, isAsync),
      /*constantResources=*/builder.getArrayAttr(resources.getArrayRef()),
      canFuseIntoOther, canBeFusedInto, canDuplicate, false,
      isStructuredKernel);
  fusedCompiledRegionOp.setDeviceRef(device);
  Block &newBody = fusedCompiledRegionOp.getBody().emplaceBlock();
  builder.setInsertionPoint(&newBody, newBody.begin());

  // Create a mapping from outside SSA value to internal SSA value for inputs.
  DenseMap<Value, Value> outsideValueToInternalValue;
  for (Value operand : fusedRegionOperands) {
    auto newArg = newBody.addArgument(operand.getType(), loc);
    outsideValueToInternalValue[operand] = newArg;
  }

  SmallVector<Value> valuesToOutput;

  IRMapping mapper;
  for (auto op : toFuse) {
    auto asRegionOp = cast<MOGG::GenericCompiledRegionOp>(op);
    for (auto [operand, argument] :
         llvm::zip(asRegionOp->getOperands(),
                   asRegionOp.getBody().front().getArguments())) {
      mapper.map(argument, outsideValueToInternalValue[operand]);
    }

    for (auto &internalOp : asRegionOp.getBody().front()) {
      // Actual fusion.

      // When handled specially by user-provided callback
      if (inlineOpCallBack &&
          inlineOpCallBack(builder, mapper, &internalOp, asRegionOp,
                           fusedCompiledRegionOp))
        continue;

      auto newOp = builder.clone(internalOp, mapper);
      if (auto newOutputOp = dyn_cast<MOGG::OutputOp>(newOp)) {
        auto resultRange =
            llvm::make_filter_range(asRegionOp->getResults(), [](Value val) {
              return !isa<MEF::ChainType>(val.getType());
            });

        auto updateMetadataFunc = [&](Value oldResult, Value internalValue) {
          outsideValueToInternalValue[oldResult] = internalValue;
          if (resultsThatShouldBeReplaced.contains(oldResult))
            valuesToOutput.push_back(internalValue);
        };

        if (asRegionOp->getNumResults() != newOutputOp->getNumOperands()) {
          for (auto [oldResult, internalValue] :
               llvm::zip(resultRange, newOutputOp.getOperands()))
            updateMetadataFunc(oldResult, internalValue);
        } else {
          for (auto [oldResult, internalValue] :
               llvm::zip(asRegionOp->getResults(), newOutputOp.getOperands()))
            updateMetadataFunc(oldResult, internalValue);
        }

        newOutputOp->erase();
      }
    }
  }

  // The `valuesToOutput` can have an explicit mef.chain or not. If it has a
  // mef.chain, then valuesToOutput.size() ==
  // resultsThatShouldBeReplaced.size(). Otherwise, the `fusedCompiledRegionOp`
  // output still have an implicit mef.chain and valuesToOutput.size() + 1 ==
  // resultsThatShouldBeReplaced.size().
  ASSERT_STREAM(valuesToOutput.size() + 1 >= resultsThatShouldBeReplaced.size(),
                << "Expected the kernel to have at least as many outputs as "
                   "internal result values that should be replaced");

  // After fusion, some opaque outputs become intermediates (they are leaking).
  // Manually insert destructor calls.
  // Inserting them at the end of the block is conservative.
  // Insert after last use isn't trivial, another pass will take care of moving
  // up the destructor call.
  if (genReleaseOpaque) {
    for (Operation *op : toFuse) {
      for (auto [idx, resultVal] : llvm::enumerate(op->getResults())) {
        if (!isa<MGP::OpaqueType>(resultVal.getType()) ||
            resultsThatShouldBeReplaced.contains(resultVal))
          continue;

        Value oldVal =
            op->getRegion(0).front().getTerminator()->getOperand(idx);
        Value newVal = mapper.lookup(oldVal);
        MOGG::ReleaseMojoValueOp::create(builder, oldVal.getLoc(), newVal);
      }
    }
  }

  // Creating the output op and replacing outside SSA values.
  MOGG::OutputOp::create(builder, loc, valuesToOutput);
  for (auto [oldValue, newValue] :
       llvm::zip(resultsThatShouldBeReplaced.takeVector(),
                 fusedCompiledRegionOp.getResults())) {
    oldValue.replaceAllUsesWith(newValue);
  }
}

static bool isExclusive(MOGG::KernelOp kernelOp, Value moValue) {
  bool isInAViewKernel = kernelOp.getIsViewOp().value_or(false);
  bool hasUnknownProvenance = moValue.getDefiningOp<MOGG::CallOp>();
  return !isInAViewKernel && !hasUnknownProvenance;
}

// Build the static tensor spec used as a parameter for autospecialized
// kernels. Note, this returns a parametric result which isn't bound to a name.
ErrorOr<TypedAttr>
buildTensorSpecAttr(DeviceRefAttr targetDevice, Value moValue, bool isExclusive,
                    MOGG::MojoLibraryAnalysis &libraryAnalysis,
                    MOGG::TypeManager &typeManager, OpBuilder &builder) {
  MO::TensorType tensorType = MO::dynCastFromTensorLike(moValue.getType());
  if (!tensorType)
    return Error("unsupported value type");

  // Build the proper output type
  TypedAttr dtypeParam = typeManager.lookupDType(tensorType.getDtype());
  TypedAttr rankParam = builder.getIndexAttr(tensorType.getRankNonZero());
  TypedAttr alignmentParam = builder.getIndexAttr(
      AsyncRT::getPreferredAlignment(targetDevice.getLabel().str()));

  // Lower address space
  CompileDiagnosticOr<TypedAttr> maybeAddressSpace =
      typeManager.lowerAddressSpace(builder);
  if (maybeAddressSpace.isError())
    return Error("could not lower address space");

  TypedAttr exclusive = builder.getBoolAttr(isExclusive);

  SmallVector<TypedAttr> buildSpecParams{dtypeParam, rankParam};

  // Populate static shape information
  SmallVector<int64_t> shapeVals = tensorType.getIntDims();
  if (shapeVals.empty())
    shapeVals.push_back(1);
  auto shapeOrError = typeManager.lowerStaticDims(builder, shapeVals);
  if (shapeOrError.isError())
    return Error("failed to populate static shape.");

  // Populate static stride information
  auto strideVals = MOGG::getStridesForTensor(tensorType, /*unknown=*/false);
  auto stridesOrError = typeManager.lowerStaticDims(builder, strideVals);
  if (stridesOrError.isError())
    return Error("failed to populate static strides.");

  // Create parametric call which produces the static spec
  SmallVector<mlir::TypedAttr> lambdaArgs{
      shapeOrError.takeValue(), stridesOrError.takeValue(), alignmentParam,
      maybeAddressSpace.takeValue(), exclusive};

  return libraryAnalysis.buildParameterCall(MOGG::Utils::DPSBuildTensorSpecs,
                                            buildSpecParams, lambdaArgs);
}

KGEN::ParamDeclAttr bindAttrToParam(Location loc, TypedAttr toBind,
                                    StringRef paramName, OpBuilder &builder) {
  // Redefine the spec attr with a new decl attached to the ToKGEN.
  // We need this during view fusion to update the strides.
  auto newParameterDecl = KGEN::ParamDeclAttr::get(
      MOGG::defineNewUniqueParameter(paramName, toBind.getType()));

  // We insert the KGEN::ParamDeclareOp at the beginning of the block.
  OpBuilder::InsertionGuard guard(builder);

  Block *kernelBody = builder.getInsertionBlock();
  builder.setInsertionPointToStart(kernelBody);
  KGEN::ParamDeclareOp::create(builder, loc, newParameterDecl, toBind);

  return newParameterDecl;
}

// Build the static tensor spec used as a parameter for autospecialized
// kernels. Note, this returns a parametric result which isn't bound to a name.
ErrorOr<TypedAttr>
buildTensorSpecAttr(MOGG::KernelOp kernelOp, Value moValue,
                    MOGG::MojoLibraryAnalysis &libraryAnalysis,
                    MOGG::TypeManager &typeManager, OpBuilder &builder) {
  MO::TensorType tensorType = MO::dynCastFromTensorLike(moValue.getType());
  if (!tensorType)
    return Error("unsupported value type");

  DeviceRefAttr targetDevice =
      kernelOp ? MO::getDeviceOrDefaultAny(kernelOp)
               : DeviceRefAttr::getDefault(tensorType.getContext());

  return buildTensorSpecAttr(targetDevice, moValue,
                             isExclusive(kernelOp, moValue), libraryAnalysis,
                             typeManager, builder);
}

bool dependsOnNoAliasConstantLikesOnly(mlir::Operation *operation) {
  return dependsOnlyOnProperty(operation, [](mlir::Operation *op) -> bool {
    if (!isa<MO::ConstantLike, MO::ChainCreateOp>(op))
      return false;

    auto extConst = dyn_cast<MO::ConstantExternalOp>(op);
    return !extConst || !extConst.getHasAlias();
  });
}

static bool valueDependsTransitivelyOnImpl(Value start, Value targetValue,
                                           mlir::Operation *targetOp) {
  SmallVector<Value> worklist{start};
  llvm::SmallPtrSet<Value, 16> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    if (current == targetValue)
      return true;

    if (Operation *def = current.getDefiningOp()) {
      if (def == targetOp)
        return true;
      worklist.append(def->operand_begin(), def->operand_end());
    }
  }
  return false;
}

bool valueDependsTransitivelyOn(Value start, Value target) {
  if (start == target)
    return true;
  return valueDependsTransitivelyOnImpl(start, target, nullptr);
}

bool valueDependsTransitivelyOn(Value start, mlir::Operation *target) {
  return valueDependsTransitivelyOnImpl(start, nullptr, target);
}

ErrorOr<KernelOutputFusionLambdaType>
getKernelOutputFusionLambdaType(MojoLibraryAnalysis &mojoLibraryAnalysis,
                                MOGG::TensorFusedOp fusedOp,
                                bool enableComputeLambdas) {
  MOGG::BindLambdaOp lambdaOp = fusedOp.getBindLambdaOp();
  if (lambdaOp.getBindKind() != MOGG::BindLambdaKind::OutputLambda)
    return Error("BindLambdaOp kind must be a OutputLambda");

  Value fusedTensor = fusedOp.getResult();
  // Find the kernel call op.
  if (!fusedTensor.hasOneUse())
    return Error("TensorFusedOp should have a unique user");
  auto executeOp = dyn_cast<CallExecuteOp>(*fusedTensor.getUsers().begin());
  if (!executeOp) {
    // If the user is not a CallExecuteOp (e.g., after epilogue fusion with
    // distributed operations where the user is a BindLambdaOp), default to
    // Output fusion type which uses a store.
    return KernelOutputFusionLambdaType::Output;
  }

  // Find the corresponding output index.
  std::optional<unsigned> outputIdx = std::nullopt;
  for (auto [idx, outputVal] : llvm::enumerate(executeOp.getOutputs())) {
    if (outputVal == fusedTensor) {
      outputIdx = idx;
      break;
    }
  }

  if (!outputIdx.has_value())
    return Error("TensorFusedOp result must be a kernel call output");

  // Check if the output operand corresponds to an output lambda.
  auto function = mojoLibraryAnalysis.getRegisteredLITMojoExecuteFunctionFor(
      executeOp.getKernelName());
  KGEN::MOGGPreElab::MojoKernelFunctionAdaptor<KGEN::LIT::FnOp> functionAdaptor(
      function);
  if (!functionAdaptor.outputArguments[*outputIdx].isTensorWithIOSpec(
          KGEN::MOGGPreElab::IOSpec::FusedComputeOutputTensor)) {
    return KernelOutputFusionLambdaType::Output;
  }

  // Now that we know the expect expects a _FusedComputeOutputTensor, let's
  // check if we build an output or compute lambda.
  if (!enableComputeLambdas)
    return KernelOutputFusionLambdaType::OutputLegacyForCompute;

  // Mixed-precision lambdas (eg cast) must use a classic output lambda.
  MO::TensorType inputTy =
      MO::dynCastFromTensorLike(lambdaOp.getFusibleTensors()[0].getType());
  MO::TensorType outputTy =
      MO::dynCastFromTensorLike(lambdaOp.getOutput().getType());
  if (!inputTy || !outputTy || inputTy.getDtype() != outputTy.getDtype())
    return KernelOutputFusionLambdaType::OutputLegacyForCompute;

  return KernelOutputFusionLambdaType::Compute;
}

} // namespace M::MOGG
