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
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo kernel definitions used "
      "to replace MO ops.">,
    Option<"forceSync", "force-sync", "bool", "false",
      "Make all kernels synchronous.">
  ];

  let dependentDialects = [
      "M::MO::MODialect",
      "M::MOGG::MOGGDialect",
      "M::KGEN::KGENDialect",
      "M::KGEN::Custom::CustomDialect",
      "M::KGEN::LIT::LITDialect",
      "M::KGEN::POP::POPDialect",
      "M::HLCF::HLCFDialect",
      "M::DebugInfo::DebugInfoDialect",
      "mlir::NVVM::NVVMDialect",
      "mlir::index::IndexDialect",
  ];
}

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMOGG_TD
