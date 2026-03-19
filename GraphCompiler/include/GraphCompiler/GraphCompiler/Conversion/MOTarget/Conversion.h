//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTARGET_CONVERSION_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTARGET_CONVERSION_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Transforms/DialectConversion.h"

namespace M::MO {

/// ConversionTarget for converting to the MO dialect.
class MOConversionTarget : public ConversionTarget {
public:
  explicit MOConversionTarget(MLIRContext &ctx);
};

} // namespace M::MO

namespace M::RMO {
/// ConversionTarget for converting to the MO dialect.
class RMOConversionTarget : public ConversionTarget {
public:
  explicit RMOConversionTarget(MLIRContext &ctx);
};
} // namespace M::RMO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTARGET_CONVERSION_H
