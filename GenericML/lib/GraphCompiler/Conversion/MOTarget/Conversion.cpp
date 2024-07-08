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
      MO::ConstantOp,
      // Control flow
      MO::IfOp, MO::YieldOp, MO::WhileOp,
      // List operations
      MO::ListAppendOp, MO::ListCreateOp, MO::ListInsertOp, MO::ListGetOp,
      MO::ConcatFromListOp,
      // Custom ops allowed for now
      MO::CustomOp,
      // TODO (GRA-723)
      MO::LayerNormOp,
      // Copied from MOConversionTarget
      ModuleOp, mlir::UnrealizedConversionCastOp>();
}

MOConversionTarget::MOConversionTarget(MLIRContext &ctx)
    : ConversionTarget(ctx) {
  addLegalDialect<MO::MODialect>();
  addLegalOp<ModuleOp, mlir::UnrealizedConversionCastOp>();
}
