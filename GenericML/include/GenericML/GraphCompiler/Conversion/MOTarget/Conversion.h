//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_TRANSFORMS_CONVERSION_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_TRANSFORMS_CONVERSION_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Transforms/DialectConversion.h"

namespace M::MO {

/// ConversionTarget for converting to the MO dialect.
class MOConversionTarget : public ConversionTarget {
public:
  explicit MOConversionTarget(MLIRContext &ctx);
};

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_TRANSFORMS_CONVERSION_H
