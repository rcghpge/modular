//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/ParameterInference.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"

using namespace mlir;

namespace M {

DictionaryAttr inferParameters(Operation *op, OpBuilder &builder) {
  if (auto custom = dyn_cast<MO::CustomOp>(op)) {
    if (DictionaryAttr params = custom.getParameters())
      return params;
    return builder.getDictionaryAttr({});
  }

  SmallVector<NamedAttribute> params;
  for (NamedAttribute attr : op->getAttrs()) {
    Attribute value = attr.getValue();
    if (isa<M::DeviceRefAttr>(value))
      continue;
    if (auto typeAttr = dyn_cast<TypeAttr>(value)) {
      params.push_back(NamedAttribute(
          attr.getName(),
          KGEN::DTypeConstantAttr::get(
              value.getContext(), MO::getDTypeOrInvalid(typeAttr.getValue()))));
      continue;
    }
    if (isa<TypedAttr>(value))
      params.push_back(attr);
  }
  return builder.getDictionaryAttr(params);
}

} // namespace M
