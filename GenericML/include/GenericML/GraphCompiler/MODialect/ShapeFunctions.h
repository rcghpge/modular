//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares a series of utilities that are useful for materializing
// shape function as parametric expressions. They can be used by components like
// GraphAPI or Graph Compiler Shape Inference.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_SHAPE_FUNCTIONS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_SHAPE_FUNCTIONS_H

namespace M::MO {
// Forward declaration for `Var`.
class Var;

/// This class helps one create parameter expression in a less elaborate way.
class ParamExprBuilder {
public:
  /// Create a `ParamExprBuilder` with the given metadata.
  /// - `graphOp` is used for new parameter generation.
  /// - `loc` is used for newly created ops like `kgen.param.declare X = <...>`.
  /// - `createdOps` will be appended with newly created ops.
  ParamExprBuilder(GraphOp graphOp, Location loc,
                   SmallVector<ParamBinding> *createdOps)
      : graphOp(graphOp), loc(loc), createdOps(createdOps) {}

  /// Create a Var that wraps around given `attr`, and allows one to build more
  /// parameter expression with the returned Var.
  Var createVar(TypedAttr attr);

  /// Create a Var that wraps around a newly `IntegerAttr` that contains
  /// `n`, and allows one to build more parameter expression with the returned
  /// Var.
  Var createVar(int64_t n);

  /// Create a parameter expression `<poc(operands...)>`, and if it doesn't get
  /// folded into a trivial attribute (parameter reference or constant), create
  /// and track a `ParamBinding` which models the following op:
  ///
  ///   kgen.param.declare declPrefix... = <poc(operands...)>
  ///
  /// Return either the folded attribute or a parameter reference.
  TypedAttr createKgenExpr(KGEN::POC poc, ArrayRef<TypedAttr> operands,
                           StringRef declPrefix = "expr");

  /// If `attr` is a a trivial attribute (parameter reference or constant),
  /// return it as is, otherwise create and track a `ParamBinding` which models
  /// the following op:
  ///
  ///   kgen.param.declare declPrefix... = <poc(operands...)>
  ///
  /// Return either `attr` itself or a parameter reference.
  TypedAttr wrapNonTrivialAttrWithDecl(TypedAttr attr,
                                       StringRef declPrefix = "expr");

  /// Creates and tracks a `ParamBinding` which represents an assert on the
  /// boolean parameter expression in `condVar`, with given error message.
  ///
  /// Create and tracks a `ParamBinding` which models the following op:
  ///   kgen.param.declare ... = <...assert(condAttr, errorMsg)...>
  ///
  /// Where `condAttr` is the parameter expression within `condVar`.
  void createAssert(const Var &condVar, StringRef errorMsg);

  /// Create and tracks a `ParamBinding` which models the following op:
  ///   kgen.param.declare ... = <cond(condAttr, thenAttr, elseAttr)>
  ///
  /// Where `(condAttr, thenAttr, elseAttr)` are parameter expressions within
  /// `(convV, thenV, elseV)` respectively.
  Var createCond(const Var &condV, const Var &thenV, const Var &elseV);

  /// Create and tracks a `ParamBinding` which models the following op:
  ///   kgen.param.declare ... = <max(lhsAttr, rhsAttr)>
  ///
  /// Where `(lhsAttr, rhsAttr)` are parameter expressions within `(lhs, rhs)`
  /// respectively.
  Var createMax(const Var &lhs, const Var &rhs);

  /// Create an op of the following form
  ///   kgen.param.declare ... = <min(lhsAttr, rhsAttr)>
  ///
  /// Where `(lhsAttr, rhsAttr)` are parameter expressions within `(lhs, rhs)`
  /// respectively.
  Var createMin(const Var &lhs, const Var &rhs);

  /// Assume given a variadic number of Var instances, create a new `ShapeAttr`
  /// consisted of the TypedAttr within those Var, in the given order.
  template <class... V, typename = std::enable_if_t<(true && ... &&
                                                     std::is_same_v<V, Var>)>>
  ShapeAttr getShapeAttr(const V &...dim) {
    return getShapeAttr({dim.getAttr()...});
  }

  /// Create a new `ShapeAttr` consisted of the `dimAttrs` in the given order.
  ShapeAttr getShapeAttr(ArrayRef<TypedAttr> dimAttrs);

private:
  GraphOp graphOp;
  Location loc;
  SmallVector<ParamBinding> *createdOps;
};

/// This class wraps around a parameter attribute and allows one to build new
/// parameter expression easily via operator overloading.
///
/// Example:
///   ParamExprBuilder b(...);
///   Var x = b.createVar(xAttr);
///   Var y = (x * x) / 2;
///   TypedAttr yAttr = y.getAttr();
class Var {
private:
  // Either constant IntegerAttr or DeclRef, to minimize expression nesting and
  // make debugging possible (otherwise expressions will fold by construction).
  TypedAttr attr;
  ParamExprBuilder *builder;

  // Tightly control how one can construct a `Var`.
  Var(TypedAttr attr, ParamExprBuilder *builder)
      : attr(attr), builder(builder) {}

  // Allow access to the controlled constructor of Var.
  friend class ParamExprBuilder;

public:
  Var(const Var &other) {
    this->attr = other.attr;
    this->builder = other.builder;
  }
  Var &operator=(const Var &other) = default;
  Var &operator=(Var &&other) = default;

  // Allows one to write `v1 + v2`.
  Var operator+(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Add, attr, other.attr),
        "var_add");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 - v2`.
  Var operator-(const Var &other) const { return *this + (other * -1); }

  // Allows one to write `v1 * v2`.
  Var operator*(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::MulNuw, attr, other.attr),
        "var_mul");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 / v2`.
  Var operator/(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Div, attr, other.attr),
        "var_div");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 != v2`.
  Var operator!=(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::getNE(attr, other.attr), "var_ne");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 == v2`.
  Var operator==(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::EQ, {attr, other.attr}),
        "var_eq");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 < v2`.
  Var operator<(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::LT, attr, other.attr),
        "var_lt");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 <= v2`.
  Var operator<=(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::LE, attr, other.attr),
        "var_le");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 % v2`.
  Var operator%(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Mod, attr, other.attr),
        "var_mod");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 > v2`.
  Var operator>(const Var &other) const { return other < *this; }

  // Allows one to write `v1 >= v2`.
  Var operator>=(const Var &other) const { return other <= *this; }

  // Allows one to write `v1 || v2`.
  Var operator||(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Or, attr, other.attr),
        "var_or");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 && v2`.
  Var operator&&(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::And, attr, other.attr),
        "var_and");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 + 42`.
  Var operator+(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this + otherVar;
  }

  // Allows one to write `v1 - 42`.
  Var operator-(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this - otherVar;
  }

  // Allows one to write `v1 * 42`.
  Var operator*(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this * otherVar;
  }

  // Allows one to write `v1 / 42`.
  Var operator/(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this / otherVar;
  }

  // Allows one to write `v1 != 42`.
  Var operator!=(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this != otherVar;
  }

  Var operator==(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this == otherVar;
  }

  // Allows one to write `v1 < 42`.
  Var operator<(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this < otherVar;
  }

  // Allows one to write `v1 <= 42`.
  Var operator<=(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this <= otherVar;
  }

  // Allows one to write `v1 > 42`.
  Var operator>(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this > otherVar;
  }

  // Allows one to write `v1 >= 42`.
  Var operator>=(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this >= otherVar;
  }

  // Allows one to write `v1 % 42`.
  Var operator%(int64_t other) const {
    Var otherVar(IntegerAttr::get(attr.getType(), other), builder);
    return *this % otherVar;
  }

  TypedAttr getAttr() const { return attr; }
  ParamExprBuilder &getBuilder() const { return *builder; }
};

/// Try converting `attr` to a constant integer.
std::optional<int64_t> tryGetConstantInt(TypedAttr attr);

/// Try converting `attr` to a vector of constant integers.
std::optional<SmallVector<int64_t>> tryGetConstantIntVector(TypedAttr attr);

// Calculates the full rank shapes for matmul inputs.
// These shapes should be equivalent after broadcasting.
// The new dimensions will just be static 1 dimensions.
std::pair<SmallVector<TypedAttr>, SmallVector<TypedAttr>>
expandMatmulInputsForBroadcasting(ShapeAttr lhsShape, ShapeAttr rhsShape);

//----------------------------------------------------------------------------//
//------------------------Shape functions begin-------------------------------//

// NOTE these helpers will eventually be shared between shape inference and
// GraphAPI shape propagation.
//
// These helpers have pretty uniform input types, where
// - `GraphOp` is used for new parameter name generation
// - `Location` is used for newly created instructions
// - All the `TypedAttr`s are parameteric representation of the op's input,
//   specifically, it should be one of the following:
//   - `IntegerAttr` (a constant integer)
//   - `KGEN::ParamDeclRefAttr` (a parameter)
//   - `KGEN::ParamOperatorAttr` (a parameter expression)
//   - `MOSH::ShapeAttr` (a list of the aforementioned integers)

/// Materialize the shape logic for `MO::BatchMatmulOp`, assuming given
ShapeMaterializeResult materializeBatchMatmulShape(GraphOp graphOp,
                                                   Location loc,
                                                   TypedAttr lhsShapeAttr,
                                                   TypedAttr rhsShapeAttr);

/// Materialize the shape logic for `RMO::MatmulOp`.
/// Note, this is a high level op that broadcasts.
/// That said, it is restricted to only broadcasting static `1` dimensions at
/// construction time.
ShapeMaterializeResult materializeRmoMatmulShape(GraphOp graphOp, Location loc,
                                                 TypedAttr lhsShapeAttr,
                                                 TypedAttr rhsShapeAttr);

/// Materialize the shape logic for RMO binary elementwise ops.
ShapeMaterializeResult materializeRmoBinaryElementwiseShape(GraphOp graphOp,
                                                            Location loc,
                                                            ShapeAttr lhsShape,
                                                            ShapeAttr rhsShape);

/// Materialize the shape logic for RMO select ops.
ShapeMaterializeResult materializeRmoSelectShape(GraphOp graphOp, Location loc,
                                                 ShapeAttr condShape,
                                                 ShapeAttr xShape,
                                                 ShapeAttr yShape);

/// Materialize the shape logic for `MO::BroadcastShapeOp`.
ShapeMaterializeResult materializeBroadcastShapeShape(GraphOp graphOp,
                                                      Location loc,
                                                      TypedAttr lhsShapeAttr,
                                                      TypedAttr rhsShapeAttr);

/// Materialize the shape logic for `MO::BroadcastToOp`.
ShapeMaterializeResult materializeBroadcastToShape(GraphOp graphOp,
                                                   Location loc,
                                                   TypedAttr inputShapeAttr,
                                                   TypedAttr targetShapeAttr);

/// Materialize the shape logic for `MO::ConcatOp`.
ShapeMaterializeResult
materializeConcatShape(GraphOp graphOp, Location loc, TypedAttr axisAttr,
                       ArrayRef<TypedAttr> inputShapeAttrs);

/// Materialize the shape logic for `MO::ConvOp`.
ShapeMaterializeResult
materializeConvShape(GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
                     TypedAttr filterShapeAttr, TypedAttr stridesAttr,
                     TypedAttr dilationsAttr, TypedAttr paddingsAttr,
                     TypedAttr numGroupsAttr);

/// Materialize the shape logic for `MO::ConvTransposeOp`.
ShapeMaterializeResult materializeConvTransposeShape(
    GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
    TypedAttr filterShapeAttr, TypedAttr stridesAttr, TypedAttr dilationsAttr,
    TypedAttr paddingsAttr, TypedAttr outputPaddingsAttr);

/// Materialize the shape logic for `MO::MatrixSolveOp`.
ShapeMaterializeResult materializeMatrixSolveShape(GraphOp graphOp,
                                                   Location loc,
                                                   TypedAttr lhsShapeAttr,
                                                   TypedAttr rhsShapeAttr);

/// Materialize the shape logic for `MO::GatherOp`.
ShapeMaterializeResult materializeGatherShape(GraphOp graphOp, Location loc,
                                              TypedAttr inputShapeAttr,
                                              TypedAttr indicesShapeAttr,
                                              TypedAttr axisAttr);

/// Materialize the shape logic for `MO::GatherNdOp`.
ShapeMaterializeResult materializeGatherNdShape(GraphOp graphOp, Location loc,
                                                TypedAttr inputShapeAttr,
                                                TypedAttr indicesShapeAttr,
                                                int64_t batchDims);

/// Materialize the shape logic for `MO::PadLike` ops.
ShapeMaterializeResult materializePadShape(GraphOp graphOp, Location loc,
                                           TypedAttr inputShapeAttr,
                                           TypedAttr paddingsAttr);

/// Materialize the shape logic for Pooling ops (e.g., `MO::AvgPoolOp`,
/// `MO::MaxPoolCeilOp`).
ShapeMaterializeResult
materializePoolShape(GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
                     TypedAttr filterAttr, TypedAttr stridesAttr,
                     TypedAttr dilationsAttr, TypedAttr paddingsAttr,
                     bool ceilMode);

/// Materialize the shape logic for `MO::RangeOp`.
ShapeMaterializeResult materializeRangeShape(GraphOp graphOp, Location loc,
                                             TypedAttr startAttr,
                                             TypedAttr stopAttr,
                                             TypedAttr stepAttr);

/// Materialize the shape logic for reduction ops like `MO::MeanOp`.
ShapeMaterializeResult materializeReduceShape(GraphOp graphOp, Location loc,
                                              TypedAttr inputShape,
                                              TypedAttr axis);

/// Materialize the shape logic for `MO::ReshapeOp`.
///
/// Specifically, if target shape contains a `-1`, infer it based on the
/// invariant that `inputShapeAttr` and `targetShapeAttr` represent equal number
/// of elements.
///
/// allowParametricDenominator removes the `denominator != 0` check when
/// parametric expressions are involved (e.g. [-1, X]). The default shape
/// inference behavour is very pessimistic. It assumes that `X` might be `0`,
/// which fails the check. This flag allows a more optimistic view where `X` is
/// assumed to not be `0` (requires runtime assert to be guaranteed).
///
/// Example:
///   inputShape = [3, D4]
///   targetShape = [D0, -1, 42]
///   returns [D0, RS0, 42]
/// with the following operation accumulated in `b`
///   kgen.param.declare RS0 = <div(mul(3, D4), mul(D0, 42)>
ShapeMaterializeResult
materializeReshapeShape(GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
                        TypedAttr targetShapeAttr,
                        bool allowParametricDenominator = false);

/// Materialize the shape logic for `MO::RoiAlignOp`.
ShapeMaterializeResult materializeRoiAlignShape(GraphOp graphOp, Location loc,
                                                TypedAttr inputShapeAttr,
                                                TypedAttr roisShapeAttr,
                                                TypedAttr outHeightAttr,
                                                TypedAttr outWidthAttr);

/// Materialize the shape logic for `MO::ScatterOp`.
ShapeMaterializeResult materializeScatterShape(GraphOp graphOp, Location loc,
                                               TypedAttr inputShapeAttr,
                                               TypedAttr updatesShapeAttr,
                                               TypedAttr indicesShapeAttr,
                                               TypedAttr axisAttr);

/// Materialize the shape logic for `MO::ScatterNd`-family ops.
ShapeMaterializeResult materializeScatterNdShape(GraphOp graphOp, Location loc,
                                                 TypedAttr inputShapeAttr,
                                                 TypedAttr updatesShapeAttr,
                                                 TypedAttr indicesShapeAttr);

/// Materialize the shape logic for `MO::SliceOp`.
ShapeMaterializeResult materializeSliceShape(GraphOp graphOp, Location loc,
                                             TypedAttr inputShapeAttr,
                                             TypedAttr startsAttr,
                                             TypedAttr stopsAttr,
                                             TypedAttr stepsAttr);

/// Materialize the shape logic for `MO::SplitOp`.
ShapeMaterializeResult materializeSplitOutputShapes(GraphOp graphOp,
                                                    Location loc,
                                                    TypedAttr inputShapeAttr,
                                                    TypedAttr splitSizesAttr,
                                                    TypedAttr axisAttr);

/// Materialize shape logic for operations whose output shape is equal to one
/// operand `targetShapeAttr`. E.g. `mo.resize_*` family of operations allows
/// resizing of image to arbitrary dimension.
ShapeMaterializeResult materializeTargetShapeLike(GraphOp graphOp, Location loc,
                                                  TypedAttr targetShapeAttr);

/// Materialize the shape logic for `MO::SqueezeShapeOp`.
ShapeMaterializeResult materializeSqueezeShapeShape(GraphOp graphOp,
                                                    Location loc,
                                                    TypedAttr inputShapeAttr,
                                                    TypedAttr indicesShapeAttr);

/// Materialize the shape logic for `MO::TileOp`.
ShapeMaterializeResult materializeTileShape(GraphOp graphOp, Location loc,
                                            TypedAttr inputShapeAttr,
                                            TypedAttr repeatsAttr);

/// Materialize the shape logic for `MO::TopKOp` and `MO::BottomKOp`
ShapeMaterializeResult materializeTopKShape(GraphOp graphOp, Location loc,
                                            TypedAttr inputShape, TypedAttr k,
                                            TypedAttr axis);

/// Materialize the shape logic for `MO::TransposeOp`.
ShapeMaterializeResult materializeTransposeShape(GraphOp graphOp, Location loc,
                                                 TypedAttr inputShapeAttr,
                                                 TypedAttr permutationsAttr);

/// Materialize the shape logic for `MO::UnsqueezeShapeOp`.
ShapeMaterializeResult
materializeUnsqueezeShapeShape(GraphOp graphOp, Location loc,
                               TypedAttr inputShapeAttr,
                               TypedAttr indicesShapeAttr);

//------------------------Shape functions end---------------------------------//
//----------------------------------------------------------------------------//

/// If `attr` is a `mo.assert false`, return the error message associated
/// with it, else return null.
StringAttr tryGetAssertFalseErrMsg(TypedAttr attr);

/// A naive shape propagator that takes all the `ParamBinding`s materialized by
/// the shape functions, and do a substitute & fold via a single traverse. At
/// the end it returns a simplified version of `targetShapes` after digesting
/// all the bindings.
///
/// It returns an error when it encounters an assert whose condition doesn't
/// fold to `true`. The error message will be the same as the assert error
/// message.
ErrorOr<SmallVector<ShapeAttr>>
propagateShape(ArrayRef<ParamBinding> bindings,
               ArrayRef<ShapeAttr> targetShapes);

// TODO move the following into a new interface.
// NOTE some of the input arguments will be removed once we move the
// implementation to a MLIR interface:
// - we won't need `kgenParams` anymore because it's really a temporary backward
//   compatibility thing for the fallback shape function.
// - `loc` will just become the location of the operation itself.;
using MaterializeShapeFuncTy = std::function<ShapeMaterializeResult(
    ArrayRef<TypedAttr> inputs, DictionaryAttr kgenParams, GraphOp graphOp,
    Location loc)>;

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_SHAPE_FUNCTIONS_H
