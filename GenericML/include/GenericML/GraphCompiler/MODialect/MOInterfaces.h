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

namespace mlir {
class RewriterBase;
} // namespace mlir

namespace M::MO {
class MatmulLike;
class SlidingWindow;

namespace impl {
LogicalResult
verifyMatchingRanks(ArrayRef<TypedValue<M::MO::TensorType>> values);
LogicalResult verifyMatmulLike(MatmulLike op);
LogicalResult verifySlidingWindow(SlidingWindow op);
LogicalResult verifyMOSameOperandsAndResultElementType(Operation *op);
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

} // namespace M::MO

#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h.inc"

namespace M::MO {
// These declarations must follow the include above since they reference
// `ParameterizationInterface`.

namespace impl {
LogicalResult
resolveSameOperandsAndResultTypeUnknownParameters(ParameterizationInterface op,
                                                  mlir::RewriterBase &rewriter);
LogicalResult resolveReductionParameters(ParameterizationInterface op,
                                         mlir::RewriterBase &rewriter);
} // namespace impl

template <typename ConcreteType>
class SameInputOutputTypesParameterization
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      SameInputOutputTypesParameterization> {
public:
  mlir::LogicalResult resolveUnknownParameters(mlir::RewriterBase &rewriter) {
    // dependent trait ensures this is valid
    auto paramInterface =
        mlir::cast<ParameterizationInterface>(this->getOperation());
    return impl::resolveSameOperandsAndResultTypeUnknownParameters(
        paramInterface, rewriter);
  }
};

template <typename ConcreteType>
class ReductionParameterization
    : public mlir::OpTrait::TraitBase<ConcreteType, ReductionParameterization> {
public:
  mlir::LogicalResult resolveUnknownParameters(mlir::RewriterBase &rewriter) {
    // dependent trait ensures this is valid
    auto paramInterface =
        mlir::cast<ParameterizationInterface>(this->getOperation());
    return impl::resolveReductionParameters(paramInterface, rewriter);
  }
};

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_H
