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

#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MODialect/MOUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
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

/// If `val` comes from a constant tensor (defined by a ConstantOp or
/// RebindOp that takes in a ConstantOp), return the defining op
MO::ConstantOp tryGetConstantOpOfRank(TensorValue val, size_t rank);

/// If `val` is defined by a ConstantOp or a RebindOp that takes in a
/// ConstantOp with given rank, convert it to an vector of specificed type,
/// else return None.
template <typename T>
static std::optional<SmallVector<T>> tryGetConstantOfRank(TensorValue val,
                                                          size_t rank) {
  if (ConstantOp constantOp = tryGetConstantOpOfRank(val, rank)) {
    return tryGetConstantBlob<T>(constantOp.getValueAttr(),
                                 constantOp.getType().getDtype());
  }
  return std::nullopt;
}

/// If `val` comes from a constant float scalar tensor (defined by a
/// ConstantOp or RebindOp that takes in a ConstantOp), convert it to a raw
/// float, else return None.
std::optional<float> tryGetConstantFloat(TensorValue val);

/// If `val` comes from a constant integer scalar tensor (defined by a
/// ConstantOp or RebindOp that takes in a ConstantOp), convert it to a raw
/// integer, else return None.
std::optional<int64_t> tryGetConstantInt(TensorValue val);

/// If `val` comes from a constant integer 1D tensor (defined by a ConstantOp or
/// RebindOp that takes in a ConstantOp), convert it to an integer vector, else
/// return None.
std::optional<SmallVector<int64_t>> tryGetConstantIntVector(TensorValue val);

/// Returns a parameter constant attribute through a variant of ToTensor ops if
/// possible. Otherwise returns a null attribute.
template <typename ToTensorOpTy>
static TypedAttr getParamConstThrough(TensorValue inputVal) {
  if (auto toTensorOp = inputVal.template getDefiningOp<ToTensorOpTy>()) {
    if (auto constOp = toTensorOp.getInput()
                           .template getDefiningOp<KGEN::ParamConstantOp>()) {
      return constOp.getValue();
    }
  }
  return {};
}

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_H
