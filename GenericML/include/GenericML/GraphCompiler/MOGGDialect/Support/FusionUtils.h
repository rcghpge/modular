//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H
#define GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H

#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"

using namespace M;

namespace M::MOGG {

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

/// Inline the given lambda into the call site of the invoke and return the new
/// results which have replaced the old operation.
SmallVector<Value> inlineInvoke(MOGG::LambdaInvokeOp invokeOp,
                                MOGG::LambdaOp invokeTarget);

/// This inlines all lambdas which are only invoked within other lambdas
/// and removed any unused lambdas. Returns true if there was any
/// modification
bool inlineLambdasInKernel(KernelOp kernel);

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

/// This function will sequence fuse a set of MOGG::GenericCompiledRegionOp. It
/// expects the input vector of MOGG::GenericCompiledRegionOp to be
/// topologically sorted.
/// `allowDuplication` indicates whether the operation can be duplicated in
/// order to fuse.
Operation *fuseAndReplaceCompiledRegions(bool allowDuplication,
                                         ArrayRef<Operation *> toFuse);

} // namespace M::MOGG

#endif // GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H
