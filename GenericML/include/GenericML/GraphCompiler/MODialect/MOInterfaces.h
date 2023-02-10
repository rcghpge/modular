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
} // namespace impl
} // namespace M::MO

#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h.inc"

namespace M::MO::impl {
LogicalResult resolveSameOperandsAndResultTypeUnknownParameters(
    MOOpParameterizationInterface op, mlir::RewriterBase &rewriter);
} // namespace M::MO::impl

namespace M::MO {
// This class must follow include above since it references
// `MOOpParameterizationInterface`
template <typename ConcreteType>
class SameInputOutputTypesParameterization
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      SameInputOutputTypesParameterization> {
public:
  mlir::LogicalResult resolveUnknownParameters(mlir::RewriterBase &rewriter) {
    // dependent trait ensures this is valid
    auto paramInterface = mlir::cast<::M::MO::MOOpParameterizationInterface>(
        this->getOperation());
    return ::M::MO::impl::resolveSameOperandsAndResultTypeUnknownParameters(
        paramInterface, rewriter);
  }
};
} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_H
