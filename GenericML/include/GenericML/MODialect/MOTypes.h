//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares types for the Modular dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MODIALECT_MOTYPES_H
#define GENERICML_MODIALECT_MOTYPES_H

#include "GenericML/MODialect/MODialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Types.h"

namespace M {
namespace MO {
static constexpr int64_t kDynamicRank = -2;
} // namespace MO
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GenericML/MODialect/MOTypes.h.inc"

#endif // GENERICML_MODIALECT_MOTYPES_H
