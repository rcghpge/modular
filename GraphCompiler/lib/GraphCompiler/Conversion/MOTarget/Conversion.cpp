//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/Conversion/MOTarget/Conversion.h"
#include "GraphCompiler/GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/GraphCompiler/RMODialect/RMODialect.h"

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
      MO::ReduceGroupNormOp, MO::LayerNormOp, MO::ReduceMinAndMaxOp,
      MO::Composite::ReduceRMSNormFusedResidualAddOp,
      MO::Composite::RMSNormResidualAddOp, MO::ReduceRMSNormOp,
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
