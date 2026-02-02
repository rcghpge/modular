//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOTarget/Conversion.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/RMODialect/RMODialect.h"

using namespace M;
using namespace MO;
using namespace RMO;

RMOConversionTarget::RMOConversionTarget(MLIRContext &ctx)
    : ConversionTarget(ctx) {
  addLegalDialect<RMO::RMODialect>();

  // RMO reuses the following operations for now:
  addLegalOp<
      // Reuses the same graph region
      MO::GraphOp, MO::OutputOp,
      // Constants
      MO::ConstantOp, MO::ConstantExternalOp,
      // Control flow
      MO::IfOp, MO::YieldOp, MO::WhileOp, MO::WhileConditionOp,
      // Custom ops allowed for now
      MO::CustomOp,
      // TODO (GEX-723)
      MO::LayerNormOp,
      // Buffer operations
      MO::ChainCreateOp,
      // Copied from MOConversionTarget
      ModuleOp, mlir::UnrealizedConversionCastOp>();
}

MOConversionTarget::MOConversionTarget(MLIRContext &ctx)
    : ConversionTarget(ctx) {
  addLegalDialect<MO::MODialect>();
  addLegalOp<ModuleOp, mlir::UnrealizedConversionCastOp>();
}
