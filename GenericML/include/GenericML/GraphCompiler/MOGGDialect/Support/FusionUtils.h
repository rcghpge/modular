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

enum class FusionType {
  SOFT, // e.g. simply combine invocations into a single execution unit
  HARD  // e.g. use lambda fusion interface
};

inline bool useExperimentalMojoPath() {
  auto var = llvm::sys::Process::GetEnv("EMIT_MOJO");
  return var.has_value() && var.value() == "1";
}

/// Returns a compact, human readable 'summary' of op, which can be used
/// at runtime for debugging and profiling.
StringAttr getSummaryString(Operation *op);

/// Returns a compact, human readable 'summary' of the combination of ops.
// `fusionType` is used to help formatting to differentiate between soft and
// hard fusion.
StringAttr getSummaryString(ArrayRef<Operation *> ops, FusionType fusionType);

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
void fuseConstant(MOGG::ExpKernelOp kernel);

/// Given a set of kernels to fuse together, rename just the right number of
/// parameters to avoid collisions. Any attempt at fusing the bodies of two
/// kernels should be prefaced with a call to this function.
void stageKernelsForFusion(ArrayRef<Operation *> kernelOps);

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

/// Build the static tensor spec used as a parameter for kernels.
/// The tensor spec value does not include fusion lambdas.
ErrorOr<TypedAttr>
buildTensorSpecAttr(DeviceRefAttr targetDevice, Value moValue, bool isExclusive,
                    MOGG::MojoLibraryAnalysis &libraryAnalysis,
                    MOGG::TypeManager &typeManager, OpBuilder &builder);
/// Same as above, except extracting `isExclusive`from MOGG::KernelOp.
ErrorOr<TypedAttr>
buildTensorSpecAttr(MOGG::KernelOp kernelOp, Value moValue,
                    MOGG::MojoLibraryAnalysis &libraryAnalysis,
                    M::MOGG::TypeManager &typeManager, OpBuilder &builder);

/// Generate a kgen.param.decl `paramName` `toBind`
KGEN::ParamDeclAttr bindAttrToParam(Location loc, TypedAttr toBind,
                                    StringRef paramName, OpBuilder &builder);

// Aliased constant (introduced in MAXPLAT-70) will not be lifted to init, which
// might reduce the performance gains from certain optimization. We probably
// need to revisit it when the aliased constant is handled correctly.
bool dependsOnNoAliasConstantLikesOnly(mlir::Operation *operation);

} // namespace M::MOGG

#endif // GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H
