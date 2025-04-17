//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H
#define GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H

#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/TypeManager.h"

using namespace M;

namespace M::MOGG {

inline bool useExperimentalMOGGPath() {
  // return true;
  auto var = llvm::sys::Process::GetEnv("MOGG_USE_EXP_KERNELS");
  return var.has_value() && var.value() == "1";
}

enum class FusionType {
  SOFT, // e.g. simply combine invocations into a single execution unit
  HARD  // e.g. use lambda fusion interface
};

/// Returns a compact, human readable 'summary' of op, which can be used
/// at runtime for debugging and profiling.
StringAttr getSummaryString(Operation *op);

/// Returns a compact, human readable 'summary' of the combination of ops.
// `fusionType` is used to help formatting to differentiate between soft and
// hard fusion.
StringAttr getSummaryString(ArrayRef<Operation *> ops, FusionType fusionType);

/// Create a call to the reshape kernel directly in MOGG. The overrideName is
/// used in case call site needs to call a particular implementation of reshape.
/// This occurs when we have multiple implementations and we would like to
/// bypass the mojo library analysis.
MOGG::CallOp createMOGGReshape(mlir::OpBuilder &builder, mlir::Location loc,
                               Value tensorIn, MO::TensorType newShape,
                               const char *overrideName = nullptr);

// Create a call to the slice kernel directly in MOGG.
MOGG::CallOp createMOGGSlice(mlir::OpBuilder &builder, mlir::Location loc,
                             Value tensorIn, Type outType,
                             const SmallVector<int64_t> &start,
                             const SmallVector<int64_t> &stop,
                             const SmallVector<int64_t> &step);

/// Fuse small constant inputs to `kernel` into the kernel, and erase the
/// original constant op if it has no remaining users.
///
/// Example:
///   %const = mo.constant ...
///   %res   = mogg.kernel(%const, ...) {
///     ...
///   }
///
/// ===>
///
///   %res = mogg.kernel(...) {
///     %const = mo.constant ...
///     ...
///   }
void fuseConstant(MOGG::KernelOp kernel);

/// Given a set of kernels to fuse together, rename just the right number of
/// parameters to avoid collisions. Any attempt at fusing the bodies of two
/// kernels should be prefaced with a call to this function.
void stageKernelsForFusion(ArrayRef<Operation *> kernelOps);

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
    ArrayRef<Operation *> kernelOpsToFuseTogether);

/// Remove all ParamDeclareOp in newParamDeclsOps that are unused in
/// fusedKernel. We need this because MoToMOGG define some staticTensorSpec attr
/// that references a lambda. But during fusion that lambda might be removed
/// (and the spec left unused) Eg: `kgen.param.declare spec0 =
/// build_static_tensor_spec(..., in_lambda, ...)`
/// But `in_lambda` was removed by the fusion pass.
void clearUnusedParamDecls(Operation *fusedKernel,
                           ArrayRef<KGEN::ParamDeclareOp> newParamDeclsOps);

/// Callback type to intercept inlining on a specific operation.
/// Return true if the operation is handled by the callback, false otherwise.
///
/// Parameters:
///   OpBuilder &: the builder being used
///   IRMapping &: the current IRMapping for the callback to lookup/update
///   Operation *: the operation being inlined
///   GenericCompiledRegionOp src: parent of the operation being inlined
///   GenericCompiledRegionOp dst: region that the operation is inlining into
using fuseRegionInlineCallBackT = llvm::function_ref<bool(
    OpBuilder &, mlir::IRMapping &, Operation *, MOGG::GenericCompiledRegionOp,
    MOGG::GenericCompiledRegionOp)>;

/// This function will sequence fuse a set of MOGG::GenericCompiledRegionOp.
/// It expects the input vector of MOGG::GenericCompiledRegionOp to be
/// topologically sorted.
/// `allowDuplication` indicates whether the operation can be duplicated in
/// order to fuse.
/// An optional inlineOpCallBack can be provided to handle special
/// operations that need to be inlined differently.
void fuseAndReplaceCompiledRegions(
    bool allowDuplication, bool genReleaseOpaque, ArrayRef<Operation *> toFuse,
    fuseRegionInlineCallBackT inlineOpCallBack = nullptr);

/// This function will merge multiple ops of the same OpTy type. This is needed
/// for cases where only one instance of OpTy should be present within a block.
template <typename OpTy>
static void mergeAllocationOfType(MOGG::KernelOp kernelOp) {
  SmallVector<OpTy> allocs;
  SmallVector<Type> resultTypes;
  for (auto alloc : kernelOp.getOps<OpTy>()) {
    allocs.push_back(alloc);
    for (auto type : alloc.getResultTypes())
      resultTypes.push_back(type);
  }

  if (allocs.size() < 2)
    return;

  OpBuilder builder(kernelOp.getContext());
  builder.setInsertionPoint(allocs.front());
  auto mergedAlloc = builder.create<OpTy>(kernelOp->getLoc(), resultTypes);

  int idx = 0;
  for (auto alloc : allocs) {
    for (auto result : alloc.getResults())
      result.replaceAllUsesWith(mergedAlloc.getResult(idx++));

    alloc.erase();
  }
}

/// Build the static tensor spec used as a parameter for kernels.
/// The tensor spec value does not include fusion lambdas.
ErrorOr<TypedAttr>
buildTensorSpecAttr(MOGG::KernelOp kernelOp, Value moValue,
                    MOGG::MojoLibraryAnalysis &libraryAnalysis,
                    M::MOGG::TypeManager &typeManager, OpBuilder &builder);

// Aliased constant (introduced in MAXPLAT-70) will not be lifted to init, which
// might reduce the performance gains from certain optimization. We probably
// need to revisit it when the aliased constant is handled correctly.
bool dependsOnNoAliasConstantLikesOnly(mlir::Operation *operation);

void setKGENKernelParamValueIfPresent(MOGG::KernelOp kernelOp,
                                      MojoLibraryAnalysis &analysis,
                                      StringRef paramLabel, TypedAttr value);

} // namespace M::MOGG

#endif // GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H
