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

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_SHAPE_FUNCTIONS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_SHAPE_FUNCTIONS_H

namespace M::MO {
// Forward declaration for `Var`.
class Var;

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

  /// Create and tracks a `ParamBinding` which models the following op:
  ///   kgen.param.declare declPrefix... = <poc(operands...)>
  ///
  /// Return a parameter decl ref to the newly created parameter
  /// `declPrefix...`.
  KGEN::ParamDeclRefAttr createDeclToKgenExpr(KGEN::POC poc,
                                              ArrayRef<TypedAttr> operands,
                                              StringRef declPrefix = "expr");

  /// Create and tracks a `ParamBinding` which models the following op:
  ///   kgen.param.declare declPrefix... = <attr>
  ///
  /// Return a parameter decl ref to the newly created parameter
  /// `declPrefix...`.
  KGEN::ParamDeclRefAttr createDeclToKgenAttr(TypedAttr attr,
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
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Add, attr, other.attr),
        "var_add");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 - v2`.
  Var operator-(const Var &other) const { return *this + (other * -1); }

  // Allows one to write `v1 * v2`.
  Var operator*(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::get(KGEN::POC::MulNuw, attr, other.attr),
        "var_mul");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 / v2`.
  Var operator/(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Div, attr, other.attr),
        "var_div");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 != v2`.
  Var operator!=(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::getNE(attr, other.attr), "var_ne");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 == v2`.
  Var operator==(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::get(KGEN::POC::EQ, {attr, other.attr}),
        "var_eq");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 < v2`.
  Var operator<(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::get(KGEN::POC::LT, attr, other.attr),
        "var_lt");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 <= v2`.
  Var operator<=(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::get(KGEN::POC::LE, attr, other.attr),
        "var_le");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 % v2`.
  Var operator%(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
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
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Or, attr, other.attr),
        "var_or");
    return Var(resDeclRef, builder);
  }

  // Allows one to write `v1 && v2`.
  Var operator&&(const Var &other) const {
    KGEN::ParamDeclRefAttr resDeclRef = builder->createDeclToKgenAttr(
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
    ShapeAttr resultShape;

    // Constructors
    Success() = default;
    Success(SmallVector<ParamBinding> &&bindings, ShapeAttr resultShape)
        : bindings(std::move(bindings)), resultShape(resultShape) {}

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
  Success &getSuccess() { return std::get<Success>(result); };
  Error &getError() { return std::get<Error>(result); };

  static ShapeMaterializeResult success(SmallVector<ParamBinding> &&bindings,
                                        ShapeAttr resultShape) {
    return ShapeMaterializeResult(std::move(bindings), resultShape);
  }
  static ShapeMaterializeResult nonMaterializable() {
    return ShapeMaterializeResult();
  }
  static ShapeMaterializeResult error(const Twine &errMsg) {
    return ShapeMaterializeResult(errMsg);
  }

private:
  ShapeMaterializeResult(SmallVector<ParamBinding> &&bindings,
                         ShapeAttr resultShape) {
    result.emplace<Success>(std::move(bindings), resultShape);
  }
  ShapeMaterializeResult() { result = NonMaterializable{}; }
  ShapeMaterializeResult(const Twine &errMsg) {
    result.emplace<Error>(errMsg.str());
  }
  std::variant<Success, NonMaterializable, Error> result;
};

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
ShapeMaterializeResult matierlizeRoiAlignShape(GraphOp graphOp, Location loc,
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
///
/// NOTE this one is odd because `MO::SplitOp` has multiple outputs (so one'd
/// think the result should contain multiple output shapes). However since
/// currently we wire these C++ shape function with fallback shape function
/// calls, which contain one shape function call for each split output, the C++
/// shape function will mirror the fallback shape function.
///
/// TODO once we finish migrating staticization and are ready to move
/// `AddFallbackShapeFuncs` to after shape inference, this will materialize a
/// single shape function for a `MO::SplitOp`, with _multiple_ output.
ShapeMaterializeResult materializeSplitIthOutputShape(
    GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
    TypedAttr splitSizesAttr, TypedAttr axisAttr, size_t i);

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
ShapeMaterializeResult materializesqueezeShapeShape(GraphOp graphOp,
                                                    Location loc,
                                                    TypedAttr inputShapeAttr,
                                                    TypedAttr indicesShapeAttr);

//------------------------Shape functions end---------------------------------//
//----------------------------------------------------------------------------//

/// If `attr` is a `mo.assert false`, return the error message associated
/// with it, else return null.
StringAttr tryGetAssertFalseErrMsg(TypedAttr attr);

/// A naive shape propagator that takes all the `ParamBinding`s materialized by
/// the shape functions, and do a substitute & fold via a single traverse. At
/// the end it returns a simplified version of `targetShape` after digesting all
/// the bindings.
///
/// It returns an error when it encounters an assert whose condition doesn't
/// fold to `true`. The error message will be the same as the assert error
/// message.
ErrorOr<ShapeAttr> propagateShape(ArrayRef<ParamBinding> bindings,
                                  ShapeAttr targetShape);

// TODO move the following into a new interface.
// NOTE some of the input arguments will be removed once we move the
// implementation to a MLIR interface:
// - we won't need `kgenParams` anymore because it's really a temporary backward
//   compatibility thing for the fallback shape function.
// - `loc` will just become the location of the operation itself.;
using MaterializeShapeFuncTy = std::function<ShapeMaterializeResult(
    ArrayRef<TypedAttr> inputs, DictionaryAttr kgenParams, GraphOp graphOp,
    Location loc)>;
/// Return a shape materialization function for the mojo shape function with
/// given name.
std::optional<MaterializeShapeFuncTy>
tryGetShapeMaterializeFunc(StringRef shapeFuncName);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_SHAPE_FUNCTIONS_H
