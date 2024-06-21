//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_H

#include "GenericML/GraphCompiler/MODialect/MOAttrs.h"
#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MODialect/MOUtils.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

// Needed because the ODS definition of some KGEN ops don't have the namespace.
using M::KGEN::ParamDeclArrayAttr;
using M::KGEN::ParamDeclAttr;
using M::KGEN::ParameterExprArrayAttr;
using M::KGEN::SignatureType;

namespace M::MO {
/// Returns the default CPU device.
///
/// Forward declared so can be used in MOOps.td, see MODevices.h for rest of
/// helpers and MODevices.cpp for implementation.
DeviceRefAttr getHostDevice(Operation *op);
} // namespace M::MO

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

namespace M::MO {
class CoordinateTransformModeAttr;
} // namespace M::MO

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MODialect/MO.h.inc"

//===----------------------------------------------------------------------===//
// MO Utilities
//===----------------------------------------------------------------------===//

namespace M::MO {

/// Parse param decls in MO dialect
ParseResult parseParamDecls(OpAsmParser &p,
                            KGEN::ParamDeclArrayAttr &paramDecls);

/// Print param decls in MO dialect
void printParamDecls(OpAsmPrinter &p, Operation *,
                     KGEN::ParamDeclArrayAttr paramDecls);

/// If `val` comes from constant tensors (defined by a ConstantOp or
/// concatenation or identity chain of ConstantOps), return the defining ops.
SmallVector<MO::ConstantOp> tryGetConstantOpsOfRank(TensorValue val,
                                                    size_t rank);

/// If `val` is defined by a ConstantOp, RebindOp, ConcatOp, ReshapeOp or
/// unrealized conversion (that takes in a ConstantOp with given rank),
/// convert it to an vector of specificed type, else return None.
template <typename T>
static std::optional<SmallVector<T>> tryGetConstantOfRank(TensorValue val,
                                                          size_t rank) {
  SmallVector<ConstantOp> constantOps = tryGetConstantOpsOfRank(val, rank);
  if (constantOps.empty())
    return std::nullopt;

  SmallVector<T> vec;
  // Multiple constant operations: Concatenation of constant ops.
  for (ConstantOp constantOp : constantOps) {
    std::optional<SmallVector<T>> blob = tryGetConstantBlob<T>(
        constantOp.getValueAttr(), constantOp.getType().getDtype());
    if (!blob.has_value())
      return std::nullopt;
    vec.insert(vec.end(), blob.value().begin(), blob.value().end());
  }
  return vec;
}

/// If `val` comes from a constant float scalar tensor (defined by a
/// ConstantOp or RebindOp that takes in a ConstantOp), convert it to a raw
/// float, else return None.
std::optional<float> tryGetConstantFloat(TensorValue val);
std::optional<double> tryGetConstantDouble(TensorValue val);

/// If `val` comes from a constant integer scalar tensor (defined by a
/// ConstantOp or RebindOp that takes in a ConstantOp), convert it to a raw
/// integer, else return None.
std::optional<int64_t> tryGetConstantInt(TensorValue val);

/// If `val` comes from a constant boolean scalar tensor (defined by a
/// ConstantOp or RebindOp that takes in a ConstantOp), convert it to a raw
/// boolean, else return None.
std::optional<bool> tryGetConstantBool(TensorValue val);

/// If `val` comes from a constant integer 1D tensor (defined by a ConstantOp or
/// RebindOp that takes in a ConstantOp), convert it to an integer vector, else
/// return None.
std::optional<SmallVector<int64_t>> tryGetConstantIntVector(TensorValue val);

/// If `val` comes from a list.create op whose inputs are all scalar integer
/// constants, convert it to an integer vector, else return None.
std::optional<SmallVector<int64_t>> tryGetConstantIntVector(ListValue val);

/// If `val` comes from a constant integer 1D tensor (defined by a ConstantOp or
/// RebindOp that takes in a ConstantOp), convert it to a boolean vector, else
/// return None.
std::optional<SmallVector<bool>> tryGetConstantBoolVector(TensorValue val);

/// Convert `val` to a value (of MOSH::ShapeType) if possible, else return
/// `null`.
/// NOTE possible side-effect: if `val` is a tensor constant, a
/// `ParamToValueOp` will be created to hold the shape value.
Value tryConvertToShapeVal(Value val, Location loc);

/// Extract the `MOSH::ShapeAttr` represented by `val` if possible, else return
/// `null`.
ShapeAttr tryGetShapeAttr(Value val);

/// Extract the single dimension (symbolic, constant, or expression) represented
/// by `val` if possible, else return `null`.
TypedAttr tryGetDimAttr(Value val);

/// If any of the input/output tensor type's metadata changed, update the given
/// GraphOp's function type and signature to reflect those changes as well.
void synchronizeMetadataUpdates(GraphOp op);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_H
