//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGSUPPORT_KGENSTUBCOMPILER_H
#define GENERICML_GRAPHCOMPILER_MOGGSUPPORT_KGENSTUBCOMPILER_H

#include "mlir/IR/BuiltinAttributes.h"

#include "Support/Buffer.h"
#include "Support/MDialect/MAttrs.h"

namespace M::MOGG {

/// Create a DenseResourceElementsAttr from a buffer. The name is used as part
/// of the attribute name.
DenseResourceElementsAttr createArchiveAttr(MLIRContext *ctx, BufferRef buffer,
                                            StringRef name);

/// Get the target info for the current host.
ErrorOr<TargetInfoAttr> getTargetInfo(MLIRContext *ctx);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGSUPPORT_KGENSTUBCOMPILER_H
