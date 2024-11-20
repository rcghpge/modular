//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"

#include "GenericML/GraphCompiler/MODialect/MOAttrs.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MojoParameters.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENStubBuilderHelpers.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOGGTypeTraits.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "Support/AssertStream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringSet.h"

namespace M::MOGG {

StringAttr getSummaryString(Operation *op) {
  if (auto kernelOp = dyn_cast<MOGG::KernelOp>(op)) {
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
  assert(!ops.empty());

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

bool inlineLambdasInKernel(KernelOp kernel) {
  bool changed = false;
  mlir::SymbolTable symtable{kernel};

  // Inline lambdas.
  kernel.walk([&](MOGG::LambdaInvokeOp invoke) {
    auto name = invoke.getLambda().getLeafReference().strref();
    auto lambda = symtable.lookup<MOGG::LambdaOp>(name);

    if (lambda) {
      inlineInvoke(invoke, lambda);
      invoke.erase();
      changed = true;
    }
  });

  // Remove any now unused lambdas.
  kernel.walk([&](MOGG::LambdaOp lambda) {
    if (mlir::SymbolTable::symbolKnownUseEmpty(lambda, kernel)) {
      lambda.erase();
      changed = true;
    }
  });

  // If we touched any of the invokes we must disable fusion on the inputs &
  // outputs as we will have removed the input/output lambda.
  if (changed) {
    kernel.removeInputFusionInterfaceAttr();
    kernel.removeOutputFusionInterfaceAttr();
  }

  return changed;
}

SmallVector<Value> inlineInvoke(MOGG::LambdaInvokeOp invokeOp,
                                MOGG::LambdaOp invokeTarget) {
  IRMapping mapper;

  // Remap the values on the call site to the values inside the block.
  for (auto [callValue, bodyValue] :
       llvm::zip(invokeOp.getOperands(),
                 invokeTarget.getBody().front().getArguments()))
    mapper.map(bodyValue, callValue);

  mlir::OpBuilder inserter{invokeOp};

  // Clone everything in the lambda.
  for (Operation &op : invokeTarget.getBody().front()) {
    // Don't add the terminator obviously
    if (isa<MOGG::OutputOp>(op)) {
      ASSERT_STREAM(
          invokeOp->getNumResults() == op.getNumOperands(),
          << "[LAMBDA_INLINE] Return doesn't have same number of results as "
             "caller.");
      if (invokeOp.use_empty())
        continue;

      SmallVector<Value> newVals;
      for (Value v : op.getOperands())
        newVals.push_back(mapper.lookup(v));
      invokeOp.replaceAllUsesWith(newVals);
      return newVals;
    } else {
      inserter.insert(op.clone(mapper));
    }
  }

  return {};
}

MOGG::CallOp createMOGGReshape(mlir::OpBuilder &builder, mlir::Location loc,
                               Value tensorIn, MO::TensorType newShape,
                               const char *overrideName) {
  MLIRContext *ctx = builder.getContext();
  // Create the new output shape.
  auto shape = MOGG::ShapeType::get(ctx, newShape.getTypedShapeAttr());

  // Materialize the runtime value of the shape in the MOGG kernel.
  Value shapeInKGEN = MOGG::lowerShapeToKGEN(loc, builder, newShape.getDims());
  Value runtimeShape = builder.create<MOGG::ToMOGGOp>(loc, TypeRange{shape},
                                                      ValueRange{shapeInKGEN});

  // Add the non-blocking flag.
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(
      builder.getNamedAttr(MOGG::PARAMETER_SINGLE_THREAD_BLOCKING_OVERRIDE,
                           builder.getBoolAttr(true)));
  mlir::DictionaryAttr kgenParams = builder.getDictionaryAttr(attrs);

  // Create the call to the new reshape.
  return builder.create<MOGG::CallOp>(
      loc,
      /*outputTypes=*/TypeRange{newShape},
      /*kernelTarget=*/
      overrideName ? llvm::StringRef(overrideName)
                   : MO::Static::ReshapeOp::getOperationName(),
      /*inputOperands=*/ValueRange{tensorIn, runtimeShape},
      /*lambdaParams=*/mlir::ArrayAttr{},
      /*otherParams=*/kgenParams);
}

// Create a call to the slice kernel directly in MOGG.
MOGG::CallOp createMOGGSlice(mlir::OpBuilder &builder, mlir::Location loc,
                             Value tensorIn, Type outType,
                             const SmallVector<int64_t> &start,
                             const SmallVector<int64_t> &stop,
                             const SmallVector<int64_t> &step) {
  // Create a constant for the axis and each of the dimensions being sliced.
  Value startVal = builder.create<MO::ConstantOp>(loc, start);
  Value stopVal = builder.create<MO::ConstantOp>(loc, stop);
  Value stepVal = builder.create<MO::ConstantOp>(loc, step);

  // Add the non-blocking flag.
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(
      builder.getNamedAttr(MOGG::PARAMETER_SINGLE_THREAD_BLOCKING_OVERRIDE,
                           builder.getBoolAttr(true)));
  mlir::DictionaryAttr kgenParams = builder.getDictionaryAttr(attrs);

  // Create the call to the new reshape.
  return builder.create<MOGG::CallOp>(
      loc,
      /*outputTypes=*/TypeRange{outType},
      /*kernelTarget=*/MO::SliceOp::getOperationName(),
      /*inputOperands=*/ValueRange{tensorIn, startVal, stopVal, stepVal},
      /*lambdaParams=*/mlir::ArrayAttr{},
      /*otherParams=*/kgenParams);
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
static bool canFuseOperand(MOGG::KernelOp kernel, int32_t index) {
  // HACK HACK HACK https://github.com/modularml/modular/issues/22959
  // Don't fuse constant unless on the host. TEMPORARY until constant
  // lowering fixed.
  if (!MO::isOnHost(kernel))
    return false;

  if (kernel.isInPlaceAlloc())
    return false;

  // Check it is a small constant with a attribute we understand.
  if (MO::ConstantOp cst = tryGetConstantOp(kernel.getOperand(index))) {
    ElementsAttr elems = cst.getValueAttr();
    if (elems.getNumElements() > kMaxConstantSize)
      return false;

    Type dtype = elems.getElementType();
    if (!MOGGTypeTraits::isSupportedMOGGType(dtype))
      return false;

    if (!isa<M::ArrayElementsAttr, DenseResourceElementsAttr>(elems))
      return false;

    return true;
  }

  return false;
}

/// Fold the constant at `idx`-th operand of `kernel` into the kernel body.
static void fuseInputAtIndex(MOGG::KernelOp kernel, int32_t idx) {
  Value inputVal = kernel.getOperand(idx);
  MO::ConstantOp cst = tryGetConstantOp(inputVal);

  Block &block = kernel.getBody().front();

  mlir::OpBuilder builder{kernel->getContext()};
  kernel.setInsertionPointToTop(builder);

  // Clone the constant and add it to the kernel.
  MO::ConstantOp newOp = cst.clone();

  builder.insert(newOp);

  // Replace all uses of it with the now in-kernel constant.
  block.getArgument(idx).replaceAllUsesWith(newOp.getResult());

  // Remove the argument.
  kernel.removeOperandAtIndex(idx);

  // Kill the constant if that was the last use.
  if (cst->use_empty())
    cst->erase();
}

void fuseConstant(MOGG::KernelOp kernel) {
  // Traverse from the back to simplify removing operands.
  int32_t preFuseNumOperands = kernel.getNumOperands();
  for (int32_t index = preFuseNumOperands - 1; index >= 0; --index) {
    if (canFuseOperand(kernel, index))
      fuseInputAtIndex(kernel, index);
  }
}

void stageKernelsForFusion(ArrayRef<Operation *> kernelOps) {
  static const std::string prefix = "_";
  llvm::StringSet<> parameters;
  for (Operation *kernelOp : kernelOps) {
    llvm::StringMap<TypedAttr> nameToDecl;
    mlir::AttrTypeReplacer walker;
    walker.addReplacement([&](KGEN::ParamDeclAttr oldDecl) {
      std::string oldName =
          KGEN::LIT::demangleIfNeeded(oldDecl).getName().str();
      std::string newName = oldName;
      while (parameters.contains(newName))
        newName = prefix + newName;

      auto newDeclRef = KGEN::ParamDeclRefAttr::get(newName, oldDecl.getType());
      nameToDecl.insert({oldDecl.getName(), cast<TypedAttr>(newDeclRef)});
      parameters.insert(newName);

      return KGEN::ParamDeclAttr::get(newDeclRef);
    });
    walker.recursivelyReplaceElementsIn(kernelOp, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
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

/// Helper function for "soft" fusing kernels.
/// - It is expected that all kernels in `kernelOpsToFuseTogether` are sorted
/// topologically.
/// - The end result is a single kernel where results are guaranteed to be used
/// from the outside.
/// - Operands can be duplicated.
/// - There is no attempt at maintaining optional attributes which means some
/// optional attributes are set to their reasonable default (isAsync = false,
/// isViewOp = false).
/// - There is no attempt at checking if the fusion would create a cycle.
Operation *sequenceFuseKernelsAndReplaceUses(
    ArrayRef<Operation *> kernelOpsToFuseTogether) {
  if (kernelOpsToFuseTogether.size() < 2)
    return nullptr;

  // Gather all outside results and operands of all kernels to be fused.
  llvm::SetVector<Value> allResults;
  llvm::SetVector<Value> allOperands;
  for (Operation *op : kernelOpsToFuseTogether) {
    auto results = op->getResults();
    auto operands = op->getOperands();
    allResults.insert(results.begin(), results.end());
    allOperands.insert(operands.begin(), operands.end());
  }

  // FusedOperations = AllOperands - AllResults
  // This is because operands that belong to AllResults will be accessible from
  // within the kernel. Otherwise, a cycle would be created.
  llvm::SetVector<Value> fusedKernelOperands = allOperands;
  fusedKernelOperands.set_subtract(allResults);

  // Determine the outputs of the fused kernel by checking if all users are
  // inside the kernels to be fused. Otherwise, there is no point in returning
  // them.
  mlir::DenseSet<Operation *> kernelOpsToFuseTogetherAsSet(
      kernelOpsToFuseTogether.begin(), kernelOpsToFuseTogether.end());
  SmallVector<Value> valuesThatShouldBeReplaced;
  for (Value result : allResults) {
    if (!llvm::all_of(result.getUsers(), [&](auto user) {
          return kernelOpsToFuseTogetherAsSet.contains(user);
        })) {
      valuesThatShouldBeReplaced.push_back(result);
    }
  }

  // Create the output types which is just a 1:1 translation from Value -> Type.
  SmallVector<Type> fusedKernelResultTypes;
  for (Value result : valuesThatShouldBeReplaced)
    fusedKernelResultTypes.push_back(result.getType());

  // Create the kernel skeleton which is possible now that we have input
  // operands and output types.
  auto ctx = kernelOpsToFuseTogether.front()->getContext();
  OpBuilder builder(ctx);
  Location loc = kernelOpsToFuseTogether.back()->getLoc();
  builder.setInsertionPointAfter(kernelOpsToFuseTogether.back());

  bool hasReadEffect =
      llvm::any_of(kernelOpsToFuseTogether, [&](auto kernelOp) {
        return cast<KernelOp>(kernelOp).getHasReadEffect();
      });
  bool hasWriteEffect =
      llvm::any_of(kernelOpsToFuseTogether, [&](auto kernelOp) {
        return cast<KernelOp>(kernelOp).getHasWriteEffect();
      });
  // All kernels should have the same device.
  auto deviceRef =
      cast<KernelOp>(kernelOpsToFuseTogether.front()).getDeviceRef();
  ASSERT_STREAM(llvm::all_of(kernelOpsToFuseTogether,
                             [&](auto kernelOp) {
                               return cast<KernelOp>(kernelOp).getDeviceRef() ==
                                      deviceRef;
                             }),
                << "Not all kernels to fuse have the same device!");

  auto fusedKernelOp = builder.create<KernelOp>(
      loc,
      /*results=*/fusedKernelResultTypes,
      /*operands=*/fusedKernelOperands.getArrayRef(),
      /*outputParamDecls=*/
      builder.getAttr<KGEN::ParamDeclArrayAttr>(
          ArrayRef<KGEN::ParamDeclAttr>()),
      /*inputFusionInterface=*/ArrayAttr{},
      /*outputFusionInterface=*/ArrayAttr{},
      /*fusionBodyLambda=*/StringAttr{},
      /*isViewOp=*/BoolAttr{},
      /*isKGEN=*/BoolAttr::get(ctx, true),
      /*summary=*/getSummaryString(kernelOpsToFuseTogether, FusionType::SOFT),
      /*isAsync=*/BoolAttr{},
      /*allocIndices=*/ArrayAttr{},
      /*hasReadEffect*/ hasReadEffect,
      /*hasWriteEffect*/ hasWriteEffect,
      /*device*/ deviceRef);

  mlir::Block &newBody = fusedKernelOp.getBody().emplaceBlock();

  // Create the kernel arguments and map outside values to inside arguments.
  mlir::DenseMap<Value, Value> outsideOperandToNewArgument;
  for (Value operand : fusedKernelOperands) {
    auto newArg = newBody.addArgument(operand.getType(), loc);
    outsideOperandToNewArgument[operand] = newArg;
  }

  // Iterate through all memory allocations of all kernels to fuse and create
  // vectors of Type/Value which we will use to replicate them in the fused
  // kernel. Note that there is no guarantee that there will be any allocation
  // or any return value which is why everything is guarded by if-statements.
  SmallVector<Type> outputPlaceholderTypes;
  SmallVector<Type> stackAllocationTypes;
  SmallVector<Type> emptyAllocationTypes;
  SmallVector<Value> oldOutputPlaceholderValues;
  SmallVector<Value> oldStackAllocationTypes;
  SmallVector<Value> oldEmptyAllocationValuesForBothKernels;
  SmallVector<Value> oldReturnValues;
  for (Operation *op : kernelOpsToFuseTogether) {
    auto asKernelOp = cast<KernelOp>(op);
    auto placeholdersOp = asKernelOp.getOutputPlaceholder();
    auto stackAllocationOp = asKernelOp.getStackAllocateTensor();
    auto emptyAllocationOp = asKernelOp.getCreateEmptyTensor();
    auto outputOp = asKernelOp.getOutputOp();

    if (placeholdersOp) {
      auto resultTypes = placeholdersOp->getResultTypes();
      auto results = placeholdersOp->getResults();
      outputPlaceholderTypes.append(resultTypes.begin(), resultTypes.end());
      oldOutputPlaceholderValues.append(results.begin(), results.end());
    }
    if (stackAllocationOp) {
      auto resultTypes = stackAllocationOp->getResultTypes();
      auto results = stackAllocationOp->getResults();
      stackAllocationTypes.append(resultTypes.begin(), resultTypes.end());
      oldStackAllocationTypes.append(results.begin(), results.end());
    }
    if (emptyAllocationOp) {
      auto resultTypes = emptyAllocationOp->getResultTypes();
      auto results = emptyAllocationOp->getResults();
      emptyAllocationTypes.append(resultTypes.begin(), resultTypes.end());
      oldEmptyAllocationValuesForBothKernels.append(results.begin(),
                                                    results.end());
    }
    if (outputOp) {
      auto operands = outputOp.getOperands();
      auto outsideValues = op->getResults();
      // For all <output operands, kernel results>, if the result is used at
      // least once outside of the set of kernels to fuse, then it will be part
      // of the new return op.
      for (auto [insideValue, outsideValue] :
           llvm::zip(operands, outsideValues)) {
        if (!llvm::all_of(outsideValue.getUsers(), [&](auto user) {
              return kernelOpsToFuseTogetherAsSet.contains(user);
            })) {
          oldReturnValues.push_back(insideValue);
        }
      }
    }
  }

  // Use the previously created vectors of Value to create new allocations in
  // the fused kernel and remap them.
  IRMapping remapper;
  builder.setInsertionPoint(&newBody, newBody.begin());
  if (!outputPlaceholderTypes.empty()) {
    auto newPlaceholdersOp =
        builder.create<MOGG::OutputPlaceholdersOp>(loc, outputPlaceholderTypes);
    auto results = newPlaceholdersOp->getResults();
    for (auto [oldValue, newValue] :
         llvm::zip(oldOutputPlaceholderValues, results)) {
      remapper.map(oldValue, newValue);
    }
  }
  if (!stackAllocationTypes.empty()) {
    auto newStackAllocateOp =
        builder.create<MOGG::StackAllocateTensorsOp>(loc, stackAllocationTypes);
    auto results = newStackAllocateOp->getResults();
    for (auto [oldValue, newValue] :
         llvm::zip(oldStackAllocationTypes, results)) {
      remapper.map(oldValue, newValue);
    }
  }
  if (!emptyAllocationTypes.empty()) {
    auto newEmptyTensorsOp =
        builder.create<MOGG::CreateEmptyTensorsOp>(loc, emptyAllocationTypes);
    auto results = newEmptyTensorsOp->getResults();
    for (auto [oldValue, newValue] :
         llvm::zip(oldEmptyAllocationValuesForBothKernels, results)) {
      remapper.map(oldValue, newValue);
    }
  }

  // For each kernel to fuse, create a mapping of old outside results to new
  // inside value (one of the allocations we created). This is used to map old
  // kernel body arguments to new allocations. Without this, we can't know if we
  // should map to a new body argument or an internal allocation.
  mlir::DenseMap<Value, Value> outsideResultToNewInsideValue;
  for (Operation *op : kernelOpsToFuseTogether) {
    auto asKernelOp = cast<KernelOp>(op);
    for (auto [outputOperand, kernelResult] :
         llvm::zip(asKernelOp.getOutputOp()->getOperands(),
                   asKernelOp->getResults())) {
      auto newInternalValue = remapper.lookup(outputOperand);
      outsideResultToNewInsideValue[kernelResult] = newInternalValue;
    }
  }

  // Remap old kernel body arguments to either new kernel body arguments or new
  // allocations.
  for (Operation *op : kernelOpsToFuseTogether) {
    auto asKernelOp = cast<KernelOp>(op);
    for (auto [kernelOperand, oldKernelArgument] :
         llvm::zip(asKernelOp.getOperands(),
                   asKernelOp.getBody().front().getArguments())) {
      auto it = outsideResultToNewInsideValue.find(kernelOperand);
      if (it == outsideResultToNewInsideValue.end()) {
        remapper.map(oldKernelArgument,
                     outsideOperandToNewArgument[kernelOperand]);
      } else {
        remapper.map(oldKernelArgument, it->second);
      }
    }
  }

  // Actual fusion.
  mlir::SymbolTable symTable(fusedKernelOp);
  stageKernelsForFusion(kernelOpsToFuseTogether);
  for (Operation *op : kernelOpsToFuseTogether) {
    llvm::StringMap<StringRef> oldNameToNew;
    const auto skipAllocations = [](Operation *op) {
      return isa<OutputPlaceholdersOp, CreateEmptyTensorsOp,
                 StackAllocateTensorsOp>(op);
    };
    cast<KernelOp>(op).fuseBodyIntoNewWithSkip(builder, symTable, remapper,
                                               oldNameToNew, skipAllocations);
  }

  // Finally, create the return op.
  SmallVector<Value> newOutputOpOperands;
  for (Value oldValue : oldReturnValues)
    newOutputOpOperands.push_back(remapper.lookup(oldValue));
  builder.create<MOGG::OutputOp>(loc, newOutputOpOperands);

  // Use-def replacement.
  for (auto [oldValue, newValue] :
       llvm::zip(valuesThatShouldBeReplaced, fusedKernelOp.getResults())) {
    oldValue.replaceAllUsesWith(newValue);
  }

  return fusedKernelOp;
}

void clearUnusedParamDecls(Operation *fusedKernel,
                           ArrayRef<KGEN::ParamDeclareOp> newParamDeclsOps) {
  SmallVector<KGEN::ParamDeclareOp> toDeleteDecls(newParamDeclsOps);

  mlir::AttrTypeWalker walker;
  walker.addWalk([&](KGEN::ParamDeclRefAttr refAttr) {
    auto it = std::find_if(toDeleteDecls.begin(), toDeleteDecls.end(),
                           [&](KGEN::ParamDeclareOp declOp) {
                             return declOp.getParamDeclAttr().getName() ==
                                    refAttr.getName();
                           });

    // Attr is used, don't remove the declaration
    if (it != toDeleteDecls.end())
      toDeleteDecls.erase(it);
  });

  // Walk the whole kernel
  fusedKernel->walk([&](Operation *op) {
    for (const NamedAttribute &attr : op->getAttrs())
      walker.walk(attr.getValue());
    for (Type type : op->getOperandTypes())
      walker.walk(type);
    for (Region &region : op->getRegions()) {
      for (Type type : region.getArgumentTypes())
        walker.walk(type);
    }
  });

  // Remove the unused ops.
  for (auto op : toDeleteDecls)
    op->erase();
}

} // namespace M::MOGG
