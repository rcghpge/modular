//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares types for the Modular dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOPDIALECT_MOPTYPES_H
#define GENERICML_MOPDIALECT_MOPTYPES_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Types.h"

#include "GenericML/MOPDialect/MOPDialect.h"

namespace M {
class DType;
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GenericML/MOPDialect/MOPTypes.h.inc"

#endif // GENERICML_MOPDIALECT_MOPTYPES_H
