//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODialect.h"
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
    if (stepVals[i] <= 0 ||
        (stopVals[i] - startVals[i]) != outDims[i] * stepVals[i])
      return emitError() << "dim " << i << ": start/stop/step do not produce "
                         << "out-shape dim " << outDims[i]
                         << ", got start=" << startVals[i]
                         << ", stop=" << stopVals[i]
                         << ", step=" << stepVals[i];
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
