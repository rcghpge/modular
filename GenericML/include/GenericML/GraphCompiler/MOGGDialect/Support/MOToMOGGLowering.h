//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPH_COMPILER_MOGG_SUPPORT_MOTOMOGGLOWERING_H
#define GENERICML_GRAPH_COMPILER_MOGG_SUPPORT_MOTOMOGGLOWERING_H

#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/OperandMapper.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/MOGGPreElab/MojoKernelAdaptor.h"
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

// Forward declarations
class MojoLibraryAnalysis;

/// A helper class which translates MO operations to MOGG Kernels with KGEN
/// inlined.
class MOGGExpKernelBuilder {
public:
  MOGGExpKernelBuilder(Operation *op, OpBuilder &insertPoint, bool forceSync,
                       MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis);

  /// Main entry point for building most other kernels.
  /// TODO: unroll and factor common components to helpers.
  ErrorOr<MOGG::KernelOp> lowerToKernel();

private:
  DeviceRefAttr getDevice() const {
    // GEX-2278 If this is a multi-device op, we can't return a single device.
    // Existing code expects the default host device in this case.
    if (moOp->hasTrait<MO::MOMultiDevice>()) {
      return getHostDevice();
    }

    return MO::getDeviceOrDefaultAny(moOp);
  }

  DeviceRefAttr getHostDevice() const { return M::getHostDevice(graphOp); }

  ErrorOr<KernelOp> lowerToKernelBase();

  KernelCallInterface
  lowerToElementwiseKernel(KernelOp kernelOp, const OperandMapping &mapping,
                           const KGEN::MOGGPreElab::MojoKernelAdaptor &adaptor);

  ErrorOr<KernelCallInterface>
  lowerToViewKernel(KernelOp kernelOp, const OperandMapping &mapping,
                    const KGEN::MOGGPreElab::MojoKernelAdaptor &adaptor);

  KernelCallInterface
  lowerToNormalKernel(KernelOp kernelOp, const OperandMapping &mapping,
                      const KGEN::MOGGPreElab::MojoKernelAdaptor &adaptor);

  KernelCallInterface
  lowerToShapeKernel(KernelOp kernelOp, const OperandMapping &mapping,
                     const KGEN::MOGGPreElab::MojoKernelAdaptor &adaptor);

  ErrorOr<KernelOp>
  buildKernelOpImpl(KGEN::MOGGPreElab::MojoKernelAdaptor &adaptor,
                    const OperandMapping &operandMapping);

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

  MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis;
};

} // namespace M::MOGG

#endif // GENERICML_GRAPH_COMPILER_MOGG_SUPPORT_MOTOMOGGLOWERING_H
