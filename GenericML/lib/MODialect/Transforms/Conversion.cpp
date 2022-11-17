//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MODialect/Transforms/Conversion.h"
#include "GenericML/MODialect/MODialect.h"

using namespace M;
using namespace MO;

MOConversionTarget::MOConversionTarget(MLIRContext &ctx)
    : ConversionTarget(ctx) {
  addLegalDialect<MO::MODialect>();
  addLegalOp<ModuleOp, mlir::UnrealizedConversionCastOp>();
}
