//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MDialect.h"
#include "Support/Nanobind/OptionalPimpl.h"
#include "Support/Nanobind/TypeCasters.h" // IWYU pragma: keep (type casters)
#include "_core/Graph/Bindings.h"
#include "_core/Graph/Checked.h"
#include "_core/Graph/InferContext.h"
#include "_core/Graph/InferTypeOpBuilders.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

void M::Graph::Python::bindMDialect(nb::module_ m) {
  using namespace M;
  using InOutSemantics = M::InOutSignatureAttr::InOutSemantics;

  nb::enum_<InOutSemantics>(m, "InOutSemantics")
      .value("none", InOutSemantics::kNone)
      .value("in_", InOutSemantics::kIn) // in is a python keyword
      .value("out", InOutSemantics::kOut)
      .value("mut", InOutSemantics::kMut);

#include "Support/MDialect/Nanobind/MAttrInterfaces.cpp.inc"
#include "Support/MDialect/Nanobind/MAttrs.cpp.inc"
#include "Support/MDialect/Nanobind/MTypes.cpp.inc"

  nb::class_<DataLayout>(m, "DataLayout");
  nb::class_<IntArrayElementsAttr>(m, "IntArrayElementsAttr");
}
