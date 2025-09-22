//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPOPS_H
#define GENERICML_MGPDIALECT_MGPOPS_H

#include "GenericML/GraphCompiler/Interfaces/EmitsMojoInterface.h"
#include "GenericML/MGPDialect/MGPAttrs.h"
#include "GenericML/MGPDialect/MGPDialect.h"
#include "GenericML/MGPDialect/MGPOpInterfaces.h"
#include "GenericML/MGPDialect/MGPTypes.h"
#include "GraphRT/GRTDialect/GRTChainInterface.h"
#include "GraphRT/GRTDialect/GRTExtendsLifetimeInterface.h"
#include "GraphRT/GRTDialect/GRTMayAliasInterface.h"
#include "GraphRT/MEFDialect/MEFOps.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/CommonAttributes.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class ModuleOp;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/MGPDialect/MGP.h.inc"

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

namespace M::MGP {

/// Returns the alignment constraint for the given attribute, or none if it
/// has no such constraint.
ErrorOr<std::optional<uint64_t>> getImpliedAlignment(ElementsAttr value);

/// Verifies all the types and attributes of op uses devices drawn from
/// expectedDevices.
LogicalResult
verifyConsistentDevices(Operation *op,
                        std::initializer_list<DeviceRefAttr> expectedDevices);

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_MGPOPS_H
