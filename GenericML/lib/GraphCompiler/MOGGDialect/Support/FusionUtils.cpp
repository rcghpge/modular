//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"

#include "GenericML/GraphCompiler/MODialect/MOAttrs.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MojoParameters.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOGGTypeTraits.h"
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
    name = shapeFuncOp.getShapeFuncName();
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

StringAttr getSummaryString(ArrayRef<Operation *> ops) {
  ASSERT_STREAM(!ops.empty());

  if (ops.size() == 1)
    return getSummaryString(ops.front());

  std::string str;
  llvm::raw_string_ostream os(str);
  llvm::interleave(
      ops, os, [&os](Operation *op) { os << getSummaryString(op).getValue(); },
      "+");
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
      ASSERT_STREAM(invokeOp->getNumResults() == op.getNumOperands())
          << "[LAMBDA_INLINE] Return doesn't have same number of results as "
             "caller.";
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
  auto getShape = builder.create<MOGG::ParamShapeToValueOp>(loc, shape);
  Value runtimeShape = getShape.getResult();

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
  newOp.getResult().setType(cast<MO::TensorType>(inputVal.getType()));

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

} // namespace M::MOGG
