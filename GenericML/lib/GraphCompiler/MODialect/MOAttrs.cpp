//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect attributes.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOAttrs.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MO;

#include "GenericML/GraphCompiler/MODialect/MOEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// MODialect attribute support
//===----------------------------------------------------------------------===//

void MODialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GenericML/GraphCompiler/MODialect/MOAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// LayoutAttr
//===----------------------------------------------------------------------===//

LayoutAttr LayoutAttr::get(MLIRContext *ctx, StringRef formatStr) {
  return get(ctx, StringAttr::get(ctx, formatStr));
}

Attribute LayoutAttr::parse(AsmParser &p, Type t) {
  // parse '<'
  if (p.parseLess())
    return {};

  // parse the layout format string
  StringRef layoutFormat;
  if (failed(p.parseKeyword(&layoutFormat)))
    return {};

  // parse '>'
  if (p.parseGreater())
    return {};

  MLIRContext *ctx = p.getContext();
  auto formatAttr = StringAttr::get(ctx, layoutFormat);
  return p.getChecked<LayoutAttr>(p.getCurrentLocation(), ctx, formatAttr);
}

void LayoutAttr::print(AsmPrinter &p) const {
  p << '<' << getFormat().strref() << '>';
}

LogicalResult
LayoutAttr::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                   mlir::StringAttr format) {
  // TODO(#18622) make this more extensible
  static StringSet<> kSupportedLayoutStrs = {
      // conv and convTranspose inputs
      // 2d
      "NHWC",
      "NCHW",
      // 3d
      "NDHWC",

      // conv filter
      // 2d
      "RSCF",
      "FCRS",
      // 3d
      "QRSCF",

      // convTranspose filter
      // 2d
      "RSFC",
      "QRSFC",

      // packed conv and convTranspose filter
      // 2d
      "FRSCf",
      // 3d
      "FQRSCf",

      // matmul weights
      "KN",

      // transposed matmul weights
      "NK",

      // packed matmul weights
      "KNkni",
  };

  if (!kSupportedLayoutStrs.contains(format.strref()))
    return emitError() << "invalid layout format: " << format;
  return success();
}

//===----------------------------------------------------------------------===//
// ChainAttr
//===----------------------------------------------------------------------===//

bool ChainAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GenericML/GraphCompiler/MODialect/MOAttrs.cpp.inc"
