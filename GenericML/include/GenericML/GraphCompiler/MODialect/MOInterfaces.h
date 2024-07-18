//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_H

#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENOps.h"
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

/// A struct which which contains the ingredients (lhs parameter name and rhs
/// parameter expr) to create a `KGEN::ParamDeclareOp` .
///
/// This allows shape function materialization and staticization to build up
/// parameter expressions without having to worry about memory management of any
/// op, since we won't create the `KGEN::ParamDeclareOp` ops until it's actually
/// needed. This is especially helpful since we could be building up some
/// expressions and end up having to short-circuit the process (e.g., due to a
/// shape error).
struct ParamBinding {
  KGEN::ParamDeclAttr decl;
  TypedAttr value;

  /// A simple constructor.
  ParamBinding(KGEN::ParamDeclAttr decl, TypedAttr value)
      : decl(decl), value(value) {}

  /// Create an op based on this param binding, using the given location and
  /// builder.
  KGEN::ParamDeclareOp toOp(Location loc, OpBuilder b) const;
};

/// A variant class that contains result of materializing the shape logic of an
/// op, where the variant is one of
/// - Success (with the shape function ops and result shape).
/// - NonMaterializable (when inputs are too dynamic).
/// - Error (when a fatal error has been detected).
class ShapeMaterializeResult {
public:
  struct Success {
    // A series of bindings which model `kgen.param.declare` ops that define
    // either
    // - parameter expressions like `D2 = <add(D0, D1)>`
    // - assertions like `Chain = <...assert(cond, msg)...>`
    //
    // These could then be consumed by shape propagation logic which does
    // substitution & folding, and handles any asserts.
    SmallVector<ParamBinding> bindings;
    SmallVector<ShapeAttr> resultShapes;

    // Constructors
    Success() = default;
    Success(SmallVector<ParamBinding> &&bindings,
            SmallVector<ShapeAttr> &&resultShapes)
        : bindings(std::move(bindings)), resultShapes(resultShapes) {}

    // Disable copy and move.
    Success(const Success &) = delete;
    Success(Success &&) = delete;
    Success &operator=(const Success &) = delete;
    Success &operator=(Success &&) = delete;
  };
  struct NonMaterializable {};
  struct Error {
    std::string msg;

    // Constructors
    Error(const Twine &msg) : msg(msg.str()) {}

    // Disable copy and move.
    Error(const Error &) = delete;
    Error(Error &&) = delete;
    Error &operator=(const Error &) = delete;
    Error &operator=(Error &&) = delete;
  };

  bool isSuccess() { return std::holds_alternative<Success>(result); }
  bool isNonMaterializable() {
    return std::holds_alternative<NonMaterializable>(result);
  }
  bool isError() { return std::holds_alternative<Error>(result); }

  /// Returning a reference so that we could either use return as
  /// - a rvalue, e.g., `res.getSuccess().bindings` or
  /// - a lvalue, e.g., `const auto& success = res.getSuccess()`
  Success &getSuccess() { return std::get<Success>(result); }
  Error &getError() { return std::get<Error>(result); }

  static ShapeMaterializeResult success(SmallVector<ParamBinding> &&bindings,
                                        ShapeAttr resultShape) {
    return ShapeMaterializeResult(std::move(bindings), {resultShape});
  }
  static ShapeMaterializeResult success(SmallVector<ParamBinding> &&bindings,
                                        SmallVector<ShapeAttr> &&resultShapes) {
    return ShapeMaterializeResult(std::move(bindings), std::move(resultShapes));
  }
  static ShapeMaterializeResult nonMaterializable() {
    return ShapeMaterializeResult();
  }
  static ShapeMaterializeResult error(const Twine &errMsg) {
    return ShapeMaterializeResult(errMsg);
  }

private:
  ShapeMaterializeResult(SmallVector<ParamBinding> &&bindings,
                         SmallVector<ShapeAttr> &&resultShapes) {
    result.emplace<Success>(std::move(bindings), std::move(resultShapes));
  }
  ShapeMaterializeResult() { result = NonMaterializable{}; }
  ShapeMaterializeResult(const Twine &errMsg) {
    result.emplace<Error>(errMsg.str());
  }
  std::variant<Success, NonMaterializable, Error> result;
};

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
