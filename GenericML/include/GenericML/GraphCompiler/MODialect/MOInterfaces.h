//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_H

#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
class RewriterBase;
} // namespace mlir

namespace M::KGEN {
class FuncOp;
class GeneratorOp;
} // namespace M::KGEN

namespace M::MO {
class ElementWiseBinary;
class MatmulLike;
class ParamDeclarationInterface;
class ParameterizationInterface;
class Reduction;
class ScatterLike;
class ScatterNdLike;
class PadLike;
class ParamExprBuilder;
class SlidingWindow;

using M::KGEN::ParamDeclAttr;
using M::KGEN::ParamDefValue;

/// Name of optional attribute holding the #M.device_ref on which a MO or MOGG
/// op should be executed.
constexpr StringRef kOperationDeviceRefAttrName = "device";

/// Name of optional attribute signalling that a MO or MOGG op is to be called
/// via a 'host' shim, which in turn calls a implementation on device matching
/// the 'device' attribute.
constexpr StringRef kOperationViaHostAttrName = "via_host";

namespace impl {
/// Verify that ranks are either all static and match or all dynamic.
LogicalResult verifyMatchingRanks(ArrayRef<TensorValue> values);
/// Check compatiblity of shapes for the expression AB = C, supporting leading
/// batch axes and dynamic ranks/dynamic shapes.
LogicalResult checkMatmulCompatibility(Operation *op, TensorValue A,
                                       TensorValue B, TensorValue C);

LogicalResult verifyMatmulLike(MatmulLike op);
LogicalResult verifyReduction(Reduction op);
LogicalResult verifyScatterLike(ScatterLike op);
LogicalResult verifyScatterNdLike(ScatterNdLike op);
LogicalResult verifySlidingWindow(SlidingWindow op);
LogicalResult verifyPadLike(PadLike op);
LogicalResult verifyMOSameOperandsAndResultElementType(Operation *op);
LogicalResult verifyMOSameOperandsAndResultShapeAttr(Operation *op);
LogicalResult verifyMOSameOperandsAndResultShape(Operation *op);
LogicalResult verifyMOSameOperandsElementType(Operation *op);
LogicalResult verifyMOSameFirstOperandAndResultsElementType(Operation *op);

LogicalResult defaultResolveUnknownParameters(ParameterizationInterface op,
                                              mlir::RewriterBase &rewriter);
LogicalResult resolveSameOperandsAndResultShapeUnknownParameters(
    ParameterizationInterface op, mlir::RewriterBase &rewriter);
LogicalResult resolveSameFirstOperandAndResultShapeUnknownParameters(
    ParameterizationInterface op, mlir::RewriterBase &rewriter);
} // namespace impl

template <typename ConcreteType>
class MOSameOperandsAndResultElementType
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      MOSameOperandsAndResultElementType> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyMOSameOperandsAndResultElementType(op);
  }
};

template <typename ConcreteType>
class MOSameOperandsAndResultShapeAttr
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      MOSameOperandsAndResultShapeAttr> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyMOSameOperandsAndResultShapeAttr(op);
  }
};

template <typename ConcreteType>
class MOSameOperandsElementType
    : public mlir::OpTrait::TraitBase<ConcreteType, MOSameOperandsElementType> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyMOSameOperandsElementType(op);
  }
};

template <typename ConcreteType>
class MOSameFirstOperandAndResultsElementType
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      MOSameFirstOperandAndResultsElementType> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyMOSameFirstOperandAndResultsElementType(op);
  }
};
} // namespace M::MO

#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h.inc"

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_H
