//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_PACKAGEVARIADICSANDSCALARS_TD
#define GENERICML_GRAPHCOMPILER_CONVERSION_PACKAGEVARIADICSANDSCALARS_TD

include "mlir/Pass/PassBase.td"

def PackageVariadicsAndScalars
    : Pass<"package-variadics-and-scalars", "mlir::ModuleOp"> {
  let summary = "Materialize tuple and scalar operands on `mogg` calls.";
  let description = [{
    Inspects the signatures of `mogg` call ops and inserts the tuple (variadic)
    and scalar conversion ops they require. Deferred out of `MAPToMOGG` to keep
    the fusion IR simpler.

    This is scaffolding gated on `M::MO::useAdvancedFusion()`; it currently does
    nothing.
  }];

  let dependentDialects = [
    "M::MOGG::MOGGDialect",
  ];
}

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_PACKAGEVARIADICSANDSCALARS_TD
