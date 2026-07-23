//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPH_COMPILER_MOGG_SUPPORT_MOTOMOGGLOWERING_H
#define GENERICML_GRAPH_COMPILER_MOGG_SUPPORT_MOTOMOGGLOWERING_H

#include "GraphCompiler/Conversion/OperandInference.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"

/*
This file contains helpers to manage the lowering between MO and KGEN. This
translation is often done during MOGG lowering, where under the new
extensibility API KGEN IR is lowered very early on in the process.

This file contains components which helps with the following tasks:
1. The marshalling between MO types (e.g. !mo.tensor, !mo.buffer) to their KGEN
equivalents (e.g. Scalar[type: DType], ManagedTensorSlice).

2. The translation of calling convention between MO/MOGG operations to their
KGEN calls.

3. The forwarding of special op and kernel specific parameters to KGEN calls.
For example, the `synchronous` keyword, and things like `transpose_b` for
matmul.

It differs from LoweringHelper.h in that the paths it covers are exclusively
for those of the new ExtensibilityAPI.
*/

namespace M::MOGG {

class MojoRegistry;

/// Validate that the user-supplied parameters on a `mo.custom` op match the
/// compile-time parameters declared by its registered Mojo kernel. Checks
/// both struct-level and execute-fn-level params (Mojo merges them in
/// `@register` parameter lists). Used by both `--mo-to-mogg` lowering and
/// the standalone `verifyCustomOp` entry point.
ErrorOrSuccess validateCustomOpParameters(const MojoRegistry &registry,
                                          MO::CustomOp customOp);

/// A helper class which translates MO operations to MOGG Kernels with KGEN
/// inlined.
class MOGGKernelBuilder {
public:
  MOGGKernelBuilder(Operation *op, OpBuilder &insertPoint, bool forceSync,
                    MOGG::MojoRegistry &registry);

  /// Main entry point for building most other kernels.
  /// TODO: unroll and factor common components to helpers.
  ErrorOr<MOGG::KernelOp> lowerToKernel();

private:
  DeviceRefAttr getDevice() const {
    // For multi-device ops, use the first execution device rather than
    // falling back to host.  Downstream code (e.g. isGpuKernel checks in
    // BindLambdaOp::emitMojo) uses the kernel's device label to decide
    // whether GPU-safe pointer decomposition and @__copy_capture are needed.
    // Returning "host" here would skip those paths even though the kernel's
    // output lambdas ultimately execute inside GPU sub-kernels.
    // TODO: Store all the devices here, see GEX-2278
    if (auto hasDevice = dyn_cast<MO::MOHasDeviceInterface>(moOp)) {
      auto devices = hasDevice.getExecutionDevices();
      if (!devices.empty())
        return devices.front();
    }

    return MO::getDeviceOrDefaultAny(moOp);
  }

  DeviceRefAttr getHostDevice() const { return M::getHostDevice(graphOp); }

  ErrorOr<KernelOp> lowerToKernelBase();

  KernelCallInterface lowerToElementwiseKernel(
      KernelOp kernelOp, const KernelDeclOperandMapping &mapping,
      DictionaryAttr forwardedParameters, const KernelDeclAdaptor &adaptor);

  ErrorOr<KernelCallInterface>
  lowerToViewKernel(KernelOp kernelOp, const KernelDeclOperandMapping &mapping,
                    const KernelDeclAdaptor &adaptor);

  KernelCallInterface
  lowerToNormalKernel(KernelOp kernelOp,
                      const KernelDeclOperandMapping &mapping,
                      const KernelDeclAdaptor &adaptor);

  KernelCallInterface
  lowerToShapeKernel(KernelOp kernelOp, const KernelDeclOperandMapping &mapping,
                     const KernelDeclAdaptor &adaptor);

  ErrorOr<KernelOp>
  buildKernelOpImpl(KernelDeclAdaptor &adaptor,
                    const KernelDeclOperandMapping &operandMapping);

  /// Stamp the per-argument tensor-trait contracts (trait lane, D1/D5) onto
  /// `kernelOp` as discardable attributes (`kInputTensorArgContractsAttrName` /
  /// `kOutputTensorArgContractsAttrName`). No-op for pure legacy kernels.
  void stampTensorArgContracts(KernelOp kernelOp,
                               const KernelDeclOperandMapping &operandMapping);

  ErrorOr<DictionaryAttr> forwardParameters();

  /// The original MO operation we are lowering.
  Operation *moOp;

  /// The MO graph containing the above op.
  MO::GraphOp graphOp;

  /// True if the op is to be implemented by a Mojo async function.
  bool isAsync;

  OpBuilder &builder;
  MLIRContext *ctx;
  Location loc;

  MOGG::MojoRegistry &registry;
};

} // namespace M::MOGG

#endif // GENERICML_GRAPH_COMPILER_MOGG_SUPPORT_MOTOMOGGLOWERING_H
