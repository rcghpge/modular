//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/RMODialect/RMODialect.h"
#include "GraphCompiler/RMODialect/RMOOps.h"
#include "GraphCompiler/RMODialect/Transforms/RMOTransforms.h"
#include "Support/Nanobind/OptionalPimpl.h"
#include "Support/Nanobind/TypeCasters.h" // IWYU pragma: keep (type casters)
#include "_core/Graph/Bindings.h"
#include "_core/Graph/Checked.h"
#include "_core/Graph/InferContext.h"
#include "_core/Graph/InferTypeOpBuilders.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

void M::Graph::Python::bindRMODialect(nb::module_ m) {
  using namespace M::RMO;
#include "GraphCompiler/RMODialect/Nanobind/RMOAttrs.cpp.inc"
#include "GraphCompiler/RMODialect/Nanobind/RMOInterfaces.cpp.inc"
#include "GraphCompiler/RMODialect/Nanobind/RMOMOAnalogueOps.cpp.inc"
#include "GraphCompiler/RMODialect/Nanobind/RMOOps.cpp.inc"

  nb::module_ passes = m.def_submodule("passes");
  {
    nb::module_ m = passes;
#include "GraphCompiler/RMODialect/Nanobind/RMOTransforms.cpp.inc"
  }
}
