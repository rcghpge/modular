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
class ScatterNdLike;
class SlidingWindow;
class ShapeFuncMaterializationState;

using M::KGEN::ParamDeclAttr;
using M::KGEN::ParamDefValue;

namespace impl {
/// Verify that ranks are either all static and match or all dynamic.
LogicalResult verifyMatchingRanks(ArrayRef<TensorValue> values);
/// Check compatiblity of shapes for the expression AB = C, supporting leading
/// batch axes and dynamic ranks/dynamic shapes.
LogicalResult checkMatmulCompatibility(Operation *op, TensorValue A,
                                       TensorValue B, TensorValue C);

LogicalResult verifyMatmulLike(MatmulLike op);
LogicalResult verifyReduction(Reduction op);
LogicalResult verifyScatterNdLike(ScatterNdLike op);
LogicalResult verifySlidingWindow(SlidingWindow op);
LogicalResult verifyMOSameOperandsAndResultElementType(Operation *op);
LogicalResult verifyMOSameOperandsAndResultShapeAttr(Operation *op);
LogicalResult verifyMOSameOperandsAndResultShape(Operation *op);
LogicalResult verifyMOSameOperandsElementType(Operation *op);

LogicalResult defaultResolveUnknownParameters(ParameterizationInterface op,
                                              mlir::RewriterBase &rewriter);
LogicalResult resolveSameOperandsAndResultShapeUnknownParameters(
    ParameterizationInterface op, mlir::RewriterBase &rewriter);

LogicalResult
materializeReductionShapeInference(Reduction op,
                                   ShapeFuncMaterializationState &state);
LogicalResult
materializeScatterNdLikeShapeInference(ScatterNdLike op,
                                       ShapeFuncMaterializationState &state);

ErrorOrSuccess elementwiseSinkStaticBroadcastTo(Operation *op,
                                                mlir::RewriterBase &rewriter);
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

/// A state shared by shape function materilization implementations in
/// individual ops.
class ShapeFuncMaterializationState {
public:
  /// Constructor which loads a copy of the shape function module, so we can
  /// manipulate it (insert functions or elaborate the module)
  ShapeFuncMaterializationState(MLIRContext *ctx);

  /// Return the shape function module, where users can look up for existing
  /// shape functions or insert new ones.
  ModuleOp getShapeFuncModule();

  /// Return the rewriter for rewriting MO ops.
  mlir::RewriterBase &getRewriter();

  /// Set the rewriter for rewriting MO ops. NOTE the pattern rewrite must set
  /// this before passing the state to the op implementation.
  void setRewriter(mlir::RewriterBase &rewriter);

  /// Mark given shape function as used, and make sure it won't be removed
  /// during elaboration.
  void markShapeFuncAsUsed(KGEN::GeneratorOp func);

  /// ASSUME elaboration has run on the shape function module, return all the
  /// used shape functions.
  SmallVector<KGEN::FuncOp, 6> getUsedShapeFuncsAfterElaboration();

private:
  // A module that contains the following (as pre-elaboration KGEN IR)
  // - a copy of the mojo shape function module
  // - newly instantiated shape functions (e.g., a slice shape function with a
  // specific rank)
  mlir::OwningOpRef<ModuleOp> shapeFuncModule;

  // The rewriter passed down from the rewrite pattern.
  mlir::RewriterBase *rewriter;

  // Names of shape functions that are used inside the MO graph.
  llvm::SetVector<StringAttr> usedShapeFuncNames;
};

} // namespace M::MO

#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h.inc"

namespace M::MO {
// These declarations must follow the include above since they reference
// classes declared in the generated header.

template <typename ConcreteType>
class ReductionShapeInferenceMaterialization
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      ReductionShapeInferenceMaterialization> {
public:
  LogicalResult
  materializeShapeInference(ShapeFuncMaterializationState &state) {
    auto reduction = mlir::cast<Reduction>(this->getOperation());
    return impl::materializeReductionShapeInference(reduction, state);
  }
};

template <typename ConcreteType>
class ScatterNdLikeShapeInferenceMaterialization
    : public mlir::OpTrait::TraitBase<
          ConcreteType, ScatterNdLikeShapeInferenceMaterialization> {
public:
  LogicalResult
  materializeShapeInference(ShapeFuncMaterializationState &state) {
    auto reduction = mlir::cast<ScatterNdLike>(this->getOperation());
    return impl::materializeScatterNdLikeShapeInference(reduction, state);
  }
};

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_H
