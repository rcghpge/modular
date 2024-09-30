//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/PackingShapeFunctions.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/MArchTarget/MArchTargetMinimal.h"

#include "mlir/Pass/PassManager.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"

#include <mutex>

#define DEBUG_TYPE "invoke-shape-packing-funcs"

namespace M::MO {

#define GEN_PASS_DEF_INVOKESHAPEPACKINGFUNCTIONS
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

using ShapeInferenceFunction =
    llvm::function_ref<std::optional<SmallVector<int64_t>>(InvokeShapeFuncOp,
                                                           TargetInfoAttr)>;

static ShapeInferenceFunction
getShapeInferencePackingFunction(InvokeShapeFuncOp invoke) {
  if (invoke.getShapeFuncName() == "pack_matmul_b_shape_func")
    return materializePackMatmulShapeFunc;
  return nullptr;
}

static void initLLVM() {
  static std::once_flag flag;

  std::call_once(flag, []() {
    llvm::InitializeNativeTarget();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
  });
}

/// Try to pull target info off of the graph operation.
/// Mostly for testing purposes to ensure changing the host does not affect
/// the tests.
///
/// If we cannot find such an attribute, extract the target information for
/// the current host.
static ErrorOr<TargetInfoAttr> getTargetInfoForHost(GraphOp graph,
                                                    bool useHostSystem) {
  if (!useHostSystem) {
    if (auto target = graph->getAttrOfType<TargetInfoAttr>("target"))
      return target;
    return Error("unable to extract target information from graph");
  }

  MLIRContext *ctx = graph.getContext();

  initLLVM();
  return M::getTargetInfoFor(ctx, llvm::sys::getDefaultTargetTriple(),
                             llvm::sys::getHostCPUName(),
                             M::getHostCPUFeatures());
}

//===----------------------------------------------------------------------===//
// InvokeShapePackingFunctions pass.
//===----------------------------------------------------------------------===//

struct InvokeShapePackingFunctions
    : public impl::InvokeShapePackingFunctionsBase<
          InvokeShapePackingFunctions> {
  using Base::Base;

  void runOnOperation() override {
    GraphOp graph = getOperation();

    ErrorOr<TargetInfoAttr> targetOr =
        getTargetInfoForHost(graph, useHostSystem);

    if (failed(targetOr)) {
      signalPassFailure();
      return;
    }

    TargetInfoAttr targetInfo = targetOr.takeValue();

    graph.walk([&](InvokeShapeFuncOp invoke) {
      // Find corresponding inference function
      ShapeInferenceFunction infer = getShapeInferencePackingFunction(invoke);
      if (!infer)
        return WalkResult::advance();

      // Try to infer the concrete values of the output parameters
      std::optional<llvm::SmallVector<int64_t>> maybeInferredShapes =
          infer(invoke, targetInfo);

      if (!maybeInferredShapes) {
        invoke.emitRemark("unable to infer shapes for operation");
        return WalkResult::advance();
      }

      SmallVector<int64_t> inferredShapes = std::move(*maybeInferredShapes);

      ArrayRef<ParamDeclAttr> outputDecls = invoke.getOutputParamDecls();
      if (outputDecls.size() != inferredShapes.size()) {
        invoke.emitError("number of inferred parameters does not match number "
                         "of output parameters");
        return WalkResult::interrupt();
      }

      // Emit a sequence of `kgen.param.declare` ops mapping each output
      // parameter of the `mo.invoke_shape_func` to an inferred value.
      OpBuilder rewriter(invoke);
      for (auto [decl, value] : llvm::zip(outputDecls, inferredShapes)) {
        auto valueAttr = rewriter.getIndexAttr(value);
        rewriter.create<KGEN::ParamDeclareOp>(invoke.getLoc(), decl, valueAttr);
      }

      invoke.erase();
      return WalkResult::advance();
    });
  }
};

} // namespace M::MO
