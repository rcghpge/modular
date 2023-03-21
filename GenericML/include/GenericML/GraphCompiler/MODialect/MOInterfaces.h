//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_H

#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class RewriterBase;
} // namespace mlir

namespace M::MO {
class MatmulLike;
class SlidingWindow;

namespace impl {
LogicalResult verifyMatchingRanks(ArrayRef<TensorValue> values);
LogicalResult verifyMatmulLike(MatmulLike op);
LogicalResult verifySlidingWindow(SlidingWindow op);
LogicalResult verifyMOSameOperandsAndResultElementType(Operation *op);
LogicalResult verifyMOSameOperandsAndResultShapeAttr(Operation *op);
LogicalResult verifyMOSameOperandsAndResultShape(Operation *op);
LogicalResult verifyMOSameOperandsElementType(Operation *op);
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

} // namespace M::MO

#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h.inc"

namespace M::MO {
// These declarations must follow the include above since they reference
// `ParameterizationInterface`.

namespace impl {
LogicalResult resolveSameOperandsAndResultShapeUnknownParameters(
    ParameterizationInterface op, mlir::RewriterBase &rewriter);

LogicalResult resolveReductionParameters(Reduction op,
                                         mlir::RewriterBase &rewriter);

LogicalResult materializeReductionShapeInference(Reduction op,
                                                 mlir::RewriterBase &rewriter);
} // namespace impl

template <typename ConcreteType>
class SameInputOutputShapesParameterization
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      SameInputOutputShapesParameterization> {
public:
  mlir::LogicalResult resolveUnknownParameters(mlir::RewriterBase &rewriter) {
    auto paramInterface =
        mlir::cast<ParameterizationInterface>(this->getOperation());
    return impl::resolveSameOperandsAndResultShapeUnknownParameters(
        paramInterface, rewriter);
  }
};

template <typename ConcreteType>
class ReductionParameterization
    : public mlir::OpTrait::TraitBase<ConcreteType, ReductionParameterization> {
public:
  mlir::LogicalResult resolveUnknownParameters(mlir::RewriterBase &rewriter) {
    auto reduction = mlir::cast<Reduction>(this->getOperation());
    return impl::resolveReductionParameters(reduction, rewriter);
  }
};

template <typename ConcreteType>
class ReductionShapeInferenceMaterialization
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      ReductionShapeInferenceMaterialization> {
public:
  LogicalResult materializeShapeInference(mlir::RewriterBase &rewriter) {
    auto reduction = mlir::cast<Reduction>(this->getOperation());
    return impl::materializeReductionShapeInference(reduction, rewriter);
  }
};

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_H
