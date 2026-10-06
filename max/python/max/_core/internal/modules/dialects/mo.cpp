//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "Support/Nanobind/OptionalPimpl.h"
#include "Support/Nanobind/TypeCasters.h" // IWYU pragma: keep (type casters)
#include "_core/Graph/Bindings.h"
#include "_core/Graph/Checked.h"
#include "_core/Graph/InferContext.h"
#include "_core/Graph/InferTypeOpBuilders.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

void M::Graph::Python::bindMODialect(nb::module_ m) {
  using namespace M::MO;
/// Bind types first, they're used in defaults for op builders
#include "GraphCompiler/MODialect/Nanobind/MOTypes.cpp.inc"

#include "GraphCompiler/MODialect/Nanobind/MOAttrs.cpp.inc"
#include "GraphCompiler/MODialect/Nanobind/MOCompositeOps.cpp.inc"
#include "GraphCompiler/MODialect/Nanobind/MOEnums.cpp.inc"
#include "GraphCompiler/MODialect/Nanobind/MOInterfaces.cpp.inc"
#include "GraphCompiler/MODialect/Nanobind/MOOps.cpp.inc"

  nb::class_<ParamExprBuilder>(m, "ParamExprBuilder");
  nb::class_<ShapeMaterializeResult>(m, "ShapeMaterializeResult");
  nb::module_ passes = m.def_submodule("passes");
  {
    nb::module_ m = passes;
#include "GraphCompiler/MODialect/Nanobind/MOTransforms.cpp.inc"
  }
}
