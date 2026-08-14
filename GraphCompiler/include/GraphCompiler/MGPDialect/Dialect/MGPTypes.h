//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares types for the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTYPES_H
#define GENERICML_MGPDIALECT_MGPTYPES_H

#include "MLRT/GraphRT/MEFDialect/MEFTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Types.h"

#include "GraphCompiler/MGPDialect/Dialect/MGPDialect.h"

namespace M {
class DType;
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Dialect/MGPTypeInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h.inc"

namespace M::MGP {

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Parses a compact tensor type description.
ParseResult parseTensorSpec(AsmParser &p, TensorSpecType &tensorSpec);

/// Prints a compact tensor type description.
void printTensorSpec(AsmPrinter &p, TensorSpecType tensorSpec);

/// Parses the optional `, <device>` and `, mut` tail of a device tensor type.
ParseResult parseOptDeviceAndMut(AsmParser &p, M::DeviceRefAttr &device,
                                 mlir::UnitAttr &mut);

/// Prints the optional `, <device>` and `, mut` tail of a device tensor type.
void printOptDeviceAndMut(AsmPrinter &p, M::DeviceRefAttr device,
                          mlir::UnitAttr mut);

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_MGPTYPES_H
