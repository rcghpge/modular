//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MAP;

//===----------------------------------------------------------------------===//
// MAPDialect
//===----------------------------------------------------------------------===//

void MAPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "GraphCompiler/MAPDialect/Dialect/MAP.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "GraphCompiler/MAPDialect/Dialect/MAPTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// SliceSpecAttr
//===----------------------------------------------------------------------===//

// Parses `[` (`?` | int) (`,` (`?` | int))* `]`, using `?` for
// `ShapedType::kDynamic` -- the same convention MLIR shaped types use for an
// unknown dimension (e.g. `tensor<?x4xf32>`).
static ParseResult parseSliceSpecBounds(AsmParser &parser,
                                        SmallVectorImpl<int64_t> &out) {
  if (parser.parseLSquare())
    return failure();
  if (succeeded(parser.parseOptionalRSquare()))
    return success();
  do {
    if (succeeded(parser.parseOptionalQuestion())) {
      out.push_back(ShapedType::kDynamic);
      continue;
    }
    int64_t value;
    if (parser.parseInteger(value))
      return failure();
    out.push_back(value);
  } while (succeeded(parser.parseOptionalComma()));
  return parser.parseRSquare();
}

static void printSliceSpecBounds(AsmPrinter &printer,
                                 ArrayRef<int64_t> values) {
  printer << '[';
  llvm::interleaveComma(values, printer, [&](int64_t v) {
    if (ShapedType::isDynamic(v))
      printer << '?';
    else
      printer << v;
  });
  printer << ']';
}

static ParseResult parseSliceSpecBoundsField(AsmParser &parser,
                                             StringRef keyword,
                                             SmallVectorImpl<int64_t> &out) {
  if (parser.parseKeyword(keyword) || parser.parseEqual())
    return failure();
  return parseSliceSpecBounds(parser, out);
}

//===----------------------------------------------------------------------===//
// ViewSignatureAttr
//===----------------------------------------------------------------------===//

// Parses `[in-shape] -> [out-shape]` (no surrounding `<...>` -- shared by
// ViewSignatureAttr's own custom format and SliceSpecAttr's inline embedding
// of it, the same convention MOSH::ShapeAttr's printer/parser helpers use).
static ParseResult parseViewSignatureBody(AsmParser &parser,
                                          MOSH::ShapeAttr &in,
                                          MOSH::ShapeAttr &out) {
  auto parseShape = [&](MOSH::ShapeAttr &shape) -> ParseResult {
    TypedAttr shapeAttr;
    auto result = MOSH::parseShapeAttr(parser, shapeAttr);
    if (!result.has_value())
      return parser.emitError(parser.getCurrentLocation(),
                              "expected '[' for shape");
    if (failed(*result))
      return failure();
    shape = llvm::cast<MOSH::ShapeAttr>(shapeAttr);
    return success();
  };
  return failure(failed(parseShape(in)) || parser.parseArrow() ||
                 failed(parseShape(out)));
}

static void printViewSignatureBody(AsmPrinter &printer, MOSH::ShapeAttr in,
                                   MOSH::ShapeAttr out) {
  MOSH::printShapeTypedAttr(printer, in);
  printer << " -> ";
  MOSH::printShapeTypedAttr(printer, out);
}

Attribute ViewSignatureAttr::parse(AsmParser &parser, Type /*type*/) {
  MOSH::ShapeAttr in, out;
  if (parser.parseLess() || parseViewSignatureBody(parser, in, out) ||
      parser.parseGreater())
    return {};
  return ViewSignatureAttr::get(parser.getContext(), in, out);
}

void ViewSignatureAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printViewSignatureBody(printer, getIn(), getOut());
  printer << ">";
}

//===----------------------------------------------------------------------===//
// SliceSpecAttr
//===----------------------------------------------------------------------===//

llvm::SmallVector<int64_t> MAP::inferSliceSteps(MOSH::ShapeAttr out,
                                                llvm::ArrayRef<int64_t> starts,
                                                llvm::ArrayRef<int64_t> stops) {
  SmallVector<int64_t> outDims = out.getIntDims();
  llvm::SmallVector<int64_t> steps;
  steps.reserve(outDims.size());
  for (size_t i = 0, e = outDims.size(); i < e; ++i) {
    if (ShapedType::isDynamic(outDims[i]) || ShapedType::isDynamic(starts[i]) ||
        ShapedType::isDynamic(stops[i])) {
      steps.push_back(ShapedType::kDynamic);
      continue;
    }
    int64_t span = stops[i] - starts[i];
    if (outDims[i] == 0 || span % outDims[i] != 0) {
      steps.push_back(ShapedType::kDynamic);
      continue;
    }
    steps.push_back(span / outDims[i]);
  }
  return steps;
}

Attribute SliceSpecAttr::parse(AsmParser &parser, Type /*type*/) {
  MOSH::ShapeAttr inShape, outShape;
  SmallVector<int64_t> starts, stops, steps;
  if (parser.parseLess() || parseViewSignatureBody(parser, inShape, outShape) ||
      parser.parseComma() ||
      parseSliceSpecBoundsField(parser, "starts", starts) ||
      parser.parseComma() ||
      parseSliceSpecBoundsField(parser, "stops", stops) ||
      parser.parseComma() ||
      parseSliceSpecBoundsField(parser, "steps", steps) ||
      parser.parseGreater())
    return {};

  MLIRContext *ctx = parser.getContext();
  return parser.getChecked<SliceSpecAttr>(
      ctx, ViewSignatureAttr::get(ctx, inShape, outShape),
      mlir::DenseI64ArrayAttr::get(ctx, starts),
      mlir::DenseI64ArrayAttr::get(ctx, stops),
      mlir::DenseI64ArrayAttr::get(ctx, steps));
}

void SliceSpecAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printViewSignatureBody(printer, getSignature().getIn(),
                         getSignature().getOut());
  printer << ", starts = ";
  printSliceSpecBounds(printer, getStarts().asArrayRef());
  printer << ", stops = ";
  printSliceSpecBounds(printer, getStops().asArrayRef());
  printer << ", steps = ";
  printSliceSpecBounds(printer, getSteps().asArrayRef());
  printer << ">";
}

LogicalResult SliceSpecAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    MAP::ViewSignatureAttr signature, mlir::DenseI64ArrayAttr starts,
    mlir::DenseI64ArrayAttr stops, mlir::DenseI64ArrayAttr steps) {
  MOSH::ShapeAttr in = signature.getIn();
  MOSH::ShapeAttr out = signature.getOut();
  // Slice never changes rank -- the one invariant ViewSignatureAttr itself
  // does not enforce, since other views (e.g. a future reshape) can.
  if (in.getRank() != out.getRank())
    return emitError() << "in/out shapes must have the same rank, got "
                       << in.getRank() << " and " << out.getRank();

  int64_t rank = out.getRank();
  if (starts.size() != rank || stops.size() != rank || steps.size() != rank)
    return emitError() << "starts/stops/steps must each have " << rank
                       << " entries (one per dimension), got " << starts.size()
                       << "/" << stops.size() << "/" << steps.size();

  SmallVector<int64_t> inDims = in.getIntDims();
  SmallVector<int64_t> outDims = out.getIntDims();
  ArrayRef<int64_t> startVals = starts.asArrayRef();
  ArrayRef<int64_t> stopVals = stops.asArrayRef();
  ArrayRef<int64_t> stepVals = steps.asArrayRef();
  for (int64_t i = 0; i < rank; ++i) {
    if (ShapedType::isDynamic(inDims[i]) ||
        ShapedType::isDynamic(startVals[i]) ||
        ShapedType::isDynamic(stopVals[i]))
      continue;
    if (startVals[i] < 0 || startVals[i] > stopVals[i] ||
        stopVals[i] > inDims[i])
      return emitError() << "dim " << i << ": start/stop out of bounds for "
                         << "in-shape dim " << inDims[i]
                         << ", got start=" << startVals[i]
                         << ", stop=" << stopVals[i];

    if (ShapedType::isDynamic(outDims[i]) || ShapedType::isDynamic(stepVals[i]))
      continue;
    // MO sizes a slice by ceiling division (see computeRangeSize), so
    // start=0, stop=5, step=2 yields 3, not an exact multiple of the step.
    if (stepVals[i] <= 0 ||
        outDims[i] !=
            llvm::divideCeilSigned(stopVals[i] - startVals[i], stepVals[i]))
      return emitError() << "dim " << i << ": start/stop/step do not produce "
                         << "out-shape dim " << outDims[i]
                         << ", got start=" << startVals[i]
                         << ", stop=" << stopVals[i]
                         << ", step=" << stepVals[i];
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastSpecAttr
//===----------------------------------------------------------------------===//

Attribute BroadcastSpecAttr::parse(AsmParser &parser, Type /*type*/) {
  MOSH::ShapeAttr inShape, outShape;
  if (parser.parseLess() || parseViewSignatureBody(parser, inShape, outShape) ||
      parser.parseGreater())
    return {};

  MLIRContext *ctx = parser.getContext();
  return parser.getChecked<BroadcastSpecAttr>(
      ctx, ViewSignatureAttr::get(ctx, inShape, outShape));
}

void BroadcastSpecAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printViewSignatureBody(printer, getSignature().getIn(),
                         getSignature().getOut());
  printer << ">";
}

LogicalResult
BroadcastSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          MAP::ViewSignatureAttr signature) {
  MOSH::ShapeAttr in = signature.getIn();
  MOSH::ShapeAttr out = signature.getOut();
  // Broadcast can add rank (numpy semantics) -- never remove it.
  if (in.getRank() > out.getRank())
    return emitError() << "broadcast cannot reduce rank, got in-rank "
                       << in.getRank() << " > out-rank " << out.getRank();

  SmallVector<int64_t> inDims = in.getIntDims();
  SmallVector<int64_t> outDims = out.getIntDims();
  int64_t inRank = in.getRank();
  int64_t delta = static_cast<int64_t>(out.getRank()) - inRank;
  for (int64_t i = 0; i < inRank; ++i) {
    int64_t inDim = inDims[i];
    int64_t outDim = outDims[i + delta];
    if (ShapedType::isDynamic(inDim) || ShapedType::isDynamic(outDim))
      continue;
    if (inDim != 1 && inDim != outDim)
      return emitError() << "dim " << i << ": in-shape dim " << inDim
                         << " is not broadcastable to out-shape dim " << outDim;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeSpecAttr
//===----------------------------------------------------------------------===//

Attribute ReshapeSpecAttr::parse(AsmParser &parser, Type /*type*/) {
  MOSH::ShapeAttr inShape, outShape;
  if (parser.parseLess() || parseViewSignatureBody(parser, inShape, outShape) ||
      parser.parseGreater())
    return {};

  MLIRContext *ctx = parser.getContext();
  return parser.getChecked<ReshapeSpecAttr>(
      ctx, ViewSignatureAttr::get(ctx, inShape, outShape));
}

void ReshapeSpecAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printViewSignatureBody(printer, getSignature().getIn(),
                         getSignature().getOut());
  printer << ">";
}

LogicalResult
ReshapeSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        MAP::ViewSignatureAttr signature) {
  // Reshape can change rank arbitrarily -- the only invariant to check is
  // that the total element count matches, shared with
  // MO::Static::ReshapeOp's own verifier.
  return MO::checkReshapeShapesCompatible(signature.getIn(), signature.getOut(),
                                          emitError);
}

//===----------------------------------------------------------------------===//
// TransposeSpecAttr
//===----------------------------------------------------------------------===//

Attribute TransposeSpecAttr::parse(AsmParser &parser, Type /*type*/) {
  MOSH::ShapeAttr inShape, outShape;
  SmallVector<int64_t> perm;
  if (parser.parseLess() || parseViewSignatureBody(parser, inShape, outShape) ||
      parser.parseComma() || parseSliceSpecBoundsField(parser, "perm", perm) ||
      parser.parseGreater())
    return {};

  MLIRContext *ctx = parser.getContext();
  return parser.getChecked<TransposeSpecAttr>(
      ctx, ViewSignatureAttr::get(ctx, inShape, outShape),
      mlir::DenseI64ArrayAttr::get(ctx, perm));
}

void TransposeSpecAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printViewSignatureBody(printer, getSignature().getIn(),
                         getSignature().getOut());
  printer << ", perm = ";
  printSliceSpecBounds(printer, getPerm().asArrayRef());
  printer << ">";
}

LogicalResult
TransposeSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          MAP::ViewSignatureAttr signature,
                          mlir::DenseI64ArrayAttr perm) {
  MOSH::ShapeAttr in = signature.getIn();
  MOSH::ShapeAttr out = signature.getOut();
  if (in.getRank() != out.getRank())
    return emitError() << "in/out shapes must have the same rank, got "
                       << in.getRank() << " and " << out.getRank();

  int64_t rank = out.getRank();
  if (static_cast<int64_t>(perm.size()) != rank)
    return emitError() << "perm must have " << rank
                       << " entries (one per dimension), got " << perm.size();

  SmallVector<int64_t> permVec(perm.asArrayRef());
  // A dynamic perm (the eager interpreter's one-graph-per-op transpose) is
  // resolved at runtime; the permutation-validity and dim-match checks below
  // only apply to the entries known at compile time. `mo.transpose`'s own perm
  // is all-or-nothing (the whole tensor is constant or it isn't), so in
  // practice this is either fully static or fully dynamic.
  bool fullyStatic = llvm::none_of(permVec, ShapedType::isDynamic);
  if (fullyStatic) {
    SmallVector<int64_t> sortedPerm(permVec);
    llvm::sort(sortedPerm);
    for (int64_t i = 0; i < rank; ++i) {
      if (sortedPerm[i] != i)
        return emitError() << "perm must be a permutation of [0, " << rank
                           << ")";
    }
  }

  SmallVector<int64_t> inDims = in.getIntDims();
  SmallVector<int64_t> outDims = out.getIntDims();
  for (int64_t i = 0; i < rank; ++i) {
    if (ShapedType::isDynamic(permVec[i]))
      continue;
    int64_t inDim = inDims[permVec[i]];
    int64_t outDim = outDims[i];
    if (ShapedType::isDynamic(inDim) || ShapedType::isDynamic(outDim))
      continue;
    if (inDim != outDim)
      return emitError() << "dim " << i << ": out-shape dim " << outDim
                         << " does not match in-shape dim " << inDim
                         << " at perm[" << i << "]=" << permVec[i];
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "GraphCompiler/MAPDialect/Dialect/MAPTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.cpp.inc"
