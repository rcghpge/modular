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

#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

// Needed because the ODS definition of some KGEN ops don't have the namespace.
using M::KGEN::FuncType;
using M::KGEN::FuncTypeGeneratorType;
using M::KGEN::ParamDeclArrayAttr;
using M::KGEN::ParamDeclAttr;
using M::KGEN::ParameterExprArrayAttr;

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

namespace M::MO {
class CoordinateTransformModeAttr;
} // namespace M::MO

#define GET_OP_CLASSES
#include "GraphCompiler/MODialect/MO.h.inc"

//===----------------------------------------------------------------------===//
// MO Utilities
//===----------------------------------------------------------------------===//

namespace M::MO {

ParseResult parseParamDecls(OpAsmParser &p,
                            KGEN::ParamDeclArrayAttr &paramDecls);

void printParamDecls(OpAsmPrinter &p, Operation *,
                     KGEN::ParamDeclArrayAttr paramDecls);

ParseResult parseTransferResult(AsmParser &p, Type inputType, Type &resultType);

void printTransferResult(AsmPrinter &p, Operation *, TensorType inputType,
                         TensorType resultType);

/// If `val` comes from constant tensors (defined by a ConstantOp or
/// concatenation or identity chain of ConstantOps), return the defining ops.
SmallVector<MO::ConstantOp> tryGetConstantOpsOfRank(TensorValue val,
                                                    size_t rank);

/// If `val` is defined by a ConstantOp, RebindOp, ConcatOp, ReshapeOp or
/// unrealized conversion (that takes in a ConstantOp with given rank),
/// convert it to an vector of specified type, else return None.
template <typename T>
static std::optional<SmallVector<T>> tryGetConstantOfRank(TensorValue val,
                                                          size_t rank) {
  SmallVector<ConstantOp> constantOps = tryGetConstantOpsOfRank(val, rank);
  if (constantOps.empty())
    return std::nullopt;

  SmallVector<T> vec;
  for (ConstantOp constantOp : constantOps) {
    std::optional<SmallVector<T>> blob = tryGetConstantBlob<T>(
        constantOp.getValueAttr(), constantOp.getType().getDtype());
    if (!blob.has_value())
      return std::nullopt;
    vec.insert(vec.end(), blob.value().begin(), blob.value().end());
  }
  return vec;
}

/// If `val` comes from a constant scalar tensor (defined by a ConstantOp or
/// RebindOp that takes in a ConstantOp), return its value, else None.
std::optional<float> tryGetConstantFloat(TensorValue val);
std::optional<double> tryGetConstantDouble(TensorValue val);
std::optional<int64_t> tryGetConstantInt(TensorValue val);
std::optional<bool> tryGetConstantBool(TensorValue val);
std::optional<APFloat> tryGetConstantAPFloat(TensorValue val);
std::optional<APInt> tryGetConstantAPInt(TensorValue val);

/// If `val` comes from a constant 1D tensor (defined by a ConstantOp or
/// RebindOp that takes in a ConstantOp), convert it to an integer vector, else
/// return None.
std::optional<SmallVector<int64_t>> tryGetConstantIntVector(TensorValue val);

/// If `val` comes from a constant integer 1D tensor (defined by a ConstantOp or
/// RebindOp that takes in a ConstantOp), convert it to a boolean vector, else
/// return None.
std::optional<SmallVector<bool>> tryGetConstantBoolVector(TensorValue val);

/// If `val` comes from a list.create op whose inputs are all scalar
/// constants, return their values, else None.
std::optional<SmallVector<int64_t>> tryGetConstantIntVector(ListValue val);
std::optional<SmallVector<APFloat>> tryGetConstantAPFloatVector(ListValue val);
std::optional<SmallVector<APInt>> tryGetConstantAPIntVector(ListValue val);

/// Extract the `MOSH::ShapeAttr` represented by `val` if possible, else return
/// `null`.
ShapeAttr tryGetShapeAttr(Value val);

/// Extract the single dimension (symbolic, constant, or expression) represented
/// by `val` if possible, else return `null`.
TypedAttr tryGetDimAttr(Value val);

/// If any of the input/output tensor type's metadata changed, update the given
/// GraphOp's function type and signature to reflect those changes as well.
void synchronizeMetadataUpdates(GraphOp op);

/// Return the set of MO ops which are expected to only run on host.
///
/// TODO: Temporary deny-list of ops without CUDA kernels and MOGG entry
/// points; remove entries as ops gain kernels or
/// MODeviceConstraintsInterface.
const llvm::DenseSet<mlir::TypeID> &getHostOnlyMOOps();
} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_H
