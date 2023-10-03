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

#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MODialect/MOUtils.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Matchers.h"
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

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MODialect/MO.h.inc"

//===----------------------------------------------------------------------===//
// MO Utilities
//===----------------------------------------------------------------------===//

namespace M::MO {

/// Name of optional attribute holding the #M.device_spec_collection describing
/// the (virtual) devices available to the graph.
constexpr StringRef kGraphDeviceSpecCollectionAttrName = "device_specs";

/// Returns the #M.device_ref annotation for op, or the 'host' device
/// if there is no such annotation on the op itself.
DeviceRefAttr getDeviceOrDefault(MOHasDeviceInterface op);

/// As above, but for any operation.
DeviceRefAttr getDeviceOrDefaultAny(Operation *op);

/// Returns true if op should be executed via the host.
bool isViaHostAny(Operation *op);

/// Returns the #M.device_ref annotation for type which appears in the
/// the given MO graphOp. Returns the 'host' device if there is no such
/// annotation on the type itself.
DeviceRefAttr getDeviceOrDefault(GraphOp graphOp, TensorType type);

/// Returns the #M.device_ref for the 'host' device of graphOp from its device
/// specs annotation, or <"cpu", 0> if the graph has no such annotation.
DeviceRefAttr getHostDevice(GraphOp graphOp);

/// Returns the #M.device_spec_collection annotation for graphOp, or a
/// description of the current host CPU if it has no such annotation.
///
/// NOTE: AddDevicesPass will fill in the device specs annotation on the graph
/// based on all available devices. This helper is to support units tests which
/// don't want to have to deal with devices.
ErrorOr<DeviceSpecCollectionAttr> getDeviceSpecsOrDefault(GraphOp graphOp);

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

/// If `val` comes from a constant integer 1D tensor (defined by a ConstantOp or
/// RebindOp that takes in a ConstantOp), convert it to a boolean vector, else
/// return None.
std::optional<SmallVector<bool>> tryGetConstantBoolVector(TensorValue val);

/// Returns a parameter constant attribute through a variant of ToTensor ops if
/// possible. Otherwise returns a null attribute.
template <typename ToTensorOpTy>
static TypedAttr getParamConstThrough(TensorValue inputVal) {
  if (auto toTensorOp = inputVal.template getDefiningOp<ToTensorOpTy>()) {
    TypedAttr value;
    if (mlir::matchPattern(toTensorOp.getInput(), mlir::m_Constant(&value)))
      return value;
  }
  return {};
}

/// If any of the input/output tensor type's metadata changed, update the given
/// GraphOp's function type and signature to reflect those changes as well.
void synchronizeMetadataUpdates(GraphOp op);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_H
