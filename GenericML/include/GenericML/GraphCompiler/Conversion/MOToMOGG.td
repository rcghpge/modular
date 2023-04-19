//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMOGG_TD
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMOGG_TD

include "mlir/Pass/PassBase.td"

def MOToMOGG : Pass<"mo-to-mogg", "ModuleOp"> {
  let summary = "Convert `mo` dialect to mogg ops.";
  let description = [{
    This pass performs a conversion from known `mo` dialect ops to registered
    `mogg` dialect ops, performing kernel selection along the way.

    TODO: Expand this to also perform lowering search
  }];

  let options = [
      Option<"useJit", "use-jit", "bool", /*default=*/"false",
             "Compile lowered kernels into the MEF as JITed kernels">,
  ];

  let dependentDialects = [
      "M::MO::MODialect",
      "M::MOGG::MOGGDialect",
      "M::KGEN::KGENDialect",
      "M::KGEN::POP::POPDialect",
      "mlir::index::IndexDialect",
      "M::DebugInfo::DebugInfoDialect",
      "M::HLCF::HLCFDialect",
  ];
}

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMOGG_TD
