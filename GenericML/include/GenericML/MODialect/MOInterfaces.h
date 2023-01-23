//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MODIALECT_MOINTERFACES_H
#define GENERICML_MODIALECT_MOINTERFACES_H

#include "GenericML/MODialect/MOTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

namespace mlir {
class RewriterBase;
} // namespace mlir

namespace M::MO {
class MatmulLike;
class SlidingWindow;

namespace impl {
LogicalResult
verifyMatchingRanks(ArrayRef<TypedValue<M::MO::TensorType>> values);
LogicalResult verifyMatmulLike(MatmulLike op);
LogicalResult verifySlidingWindow(SlidingWindow op);
} // namespace impl
} // namespace M::MO

#include "GenericML/MODialect/MOInterfaces.h.inc"

#endif // GENERICML_MODIALECT_MOINTERFACES_H
