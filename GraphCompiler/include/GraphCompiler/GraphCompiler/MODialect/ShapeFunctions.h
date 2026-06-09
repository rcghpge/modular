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

#include "GraphCompiler/GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOOps.h"
#include "Support/AssertStream.h"

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
                   SmallVectorImpl<ParamBinding> *createdOps)
      : graphOp(graphOp), loc(loc), createdOps(createdOps) {}

  /// No copy and no move. The builder classes need pointer stability for
  /// the Var mechanism to work correctly.
  ParamExprBuilder(const ParamExprBuilder &other) = delete;
  ParamExprBuilder(ParamExprBuilder &&other) = delete;

  /// Create a Var that wraps around given `attr`, and allows one to build more
  /// parameter expression with the returned Var.
  Var createVar(TypedAttr attr);

  /// Create a Var that wraps around a newly created that contains  `n` of
  /// DimType, and allows one to build more parameter expression with the
  /// returned Var.
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
  SmallVectorImpl<ParamBinding> *createdOps;
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
      : attr(attr), builder(builder) {
    ASSERT_STREAM(
        builder, << "Var should be constructed with non-null builder");
  }

  // Allow access to the controlled constructor of Var.
  friend class ParamExprBuilder;

public:
  // Disallow default construction since this always needs a builder.
  Var() = delete;
  Var(const Var &other) = default;
  Var &operator=(const Var &other) = default;
  Var(Var &&other) = default;
  Var &operator=(Var &&other) = default;

  //===--------------------------------------------------------------------===//
  // Arithmetic
  //===--------------------------------------------------------------------===//

  // Allows one to write `v1 + v2` where v2 is a TypedAtr.
  Var operator+(TypedAttr otherAttr) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Add, attr, otherAttr),
        "var_add");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 + v2`.
  Var operator+(const Var &other) const { return *this + other.attr; }

  // v1 += v2 with TypedAttr.
  Var &operator+=(TypedAttr otherAttr) {
    *this = *this + otherAttr;
    return *this;
  }

  // v1 += v2
  Var &operator+=(const Var &other) { return *this += other.attr; }

  // Allows one to write `v1 + 42`.
  Var operator+(int64_t other) const {
    return *this + builder->createVar(other);
  }

  // Allows one to write `v1 - v2`.
  Var operator-(const Var &other) const { return *this + (other * -1); }

  // v1 -= v2
  Var operator-=(const Var &other) {
    *this = *this - other;
    return *this;
  }

  // Allows one to write `v1 - 42`.
  Var operator-(int64_t other) const {
    return *this - builder->createVar(other);
  }

  // Unary minus: `-v`.
  Var operator-() const { return *this * -1; }

  // Allows one to write `v1 * v2`.
  Var operator*(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::MulNoWrap, attr, other.attr),
        "var_mul");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 * 42`.
  Var operator*(int64_t other) const {
    return *this * builder->createVar(other);
  }

  // Allows one to write `v1 / v2`.
  Var operator/(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Div, attr, other.attr),
        "var_div");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 / 42`.
  Var operator/(int64_t other) const {
    return *this / builder->createVar(other);
  }

  // Allows one to write `v1 % v2`.
  Var operator%(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Mod, attr, other.attr),
        "var_mod");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 % 42`.
  Var operator%(int64_t other) const {
    return *this % builder->createVar(other);
  }

  //===--------------------------------------------------------------------===//
  // Comparison
  //===--------------------------------------------------------------------===//

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

  // Allows one to write `v1 > v2`.
  Var operator>(const Var &other) const { return other < *this; }

  // Allows one to write `v1 >= v2`.
  Var operator>=(const Var &other) const { return other <= *this; }

  // Allows one to write `v1 != 42`.
  Var operator!=(int64_t other) const {
    return *this != builder->createVar(other);
  }

  Var operator==(int64_t other) const {
    return *this == builder->createVar(other);
  }

  // Allows one to write `v1 < 42`.
  Var operator<(int64_t other) const {
    return *this < builder->createVar(other);
  }

  // Allows one to write `v1 <= 42`.
  Var operator<=(int64_t other) const {
    return *this <= builder->createVar(other);
  }

  // Allows one to write `v1 > 42`.
  Var operator>(int64_t other) const {
    return *this > builder->createVar(other);
  }

  // Allows one to write `v1 >= 42`.
  Var operator>=(int64_t other) const {
    return *this >= builder->createVar(other);
  }

  //===--------------------------------------------------------------------===//
  // Logical
  //===--------------------------------------------------------------------===//

  // Allows one to write `v1 && v2`.
  Var operator&&(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::And, attr, other.attr),
        "var_and");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 || v2`.
  Var operator||(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Or, attr, other.attr),
        "var_or");
    return Var(resDeclRef, builder);
  }

  TypedAttr getAttr() const { return attr; }
  ParamExprBuilder &getBuilder() const { return *builder; }
};

/// Try converting `attr` to a constant integer.
inline std::optional<int64_t> tryGetConstantInt(TypedAttr attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().getSExtValue();
  return MOSH::ShapeAttr::getStaticDim(attr);
}

/// Try converting `attr` to a constant double.
inline std::optional<double> tryGetConstantFloat(TypedAttr attr) {
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValueAsDouble();
  return std::nullopt;
}

/// Return true if `attr` is a statically known integer dimension
inline bool isStaticDimAttr(TypedAttr attr) {
  return tryGetConstantInt(attr).has_value();
}

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
// - All the `TypedAttr`s are parametric representation of the op's input,
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

/// Materialize the shape logic for RMO slice ops.
/// NOTE: the following preconditions should be checked by the caller:
/// - step != 0
/// - step > 0 => start < stop
/// - step < 0 => stop < start
ShapeMaterializeResult
materializeRmoSliceShape(GraphOp graphOp, Location loc, ShapeAttr inputShape,
                         ShapeAttr starts, ShapeAttr stops,
                         const SmallVector<int64_t> &steps);

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
                     TypedAttr numGroupsAttr, StringAttr inputLayoutAttr,
                     LayoutAttr filterLayoutAttr);

/// Materialize the shape logic for `MO::ConvTransposeOp`.
ShapeMaterializeResult materializeConvTransposeShape(
    GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
    TypedAttr filterShapeAttr, TypedAttr stridesAttr, TypedAttr dilationsAttr,
    TypedAttr paddingsAttr, TypedAttr outputPaddingsAttr,
    StringAttr inputLayoutAttr, LayoutAttr filterLayoutAttr);

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
/// The `-1` dimension is computed as mul(input_dims) / mul(target_dims).
/// Constant zero dimensions in the target shape are excluded from the
/// denominator to avoid division by zero.
///
/// Example:
///   inputShape = [3, D4]
///   targetShape = [D0, -1, 42]
///   returns [D0, RS0, 42]
/// with the following operation accumulated in `b`
///   kgen.param.declare RS0 = <div(mul(3, D4), mul(D0, 42)>
ShapeMaterializeResult materializeReshapeShape(GraphOp graphOp, Location loc,
                                               TypedAttr inputShapeAttr,
                                               TypedAttr targetShapeAttr);

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
ErrorOr<SmallVector<ShapeAttr>> propagateShape(ArrayRef<ParamBinding> bindings,
                                               ArrayRef<ShapeAttr> targetShapes,
                                               bool validateAsserts = true);

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
