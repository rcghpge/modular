//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_TD
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_TD

include "mlir/Pass/PassBase.td"

def MOToPrimitives : Pass<"mo-to-primitives", "ModuleOp"> {
  let summary = "Convert `mo` dialect to primitives.";
  let description = [{
    This pass performs a conversion from known `mo` dialect ops to `mgp` and
    `mef` primitives.
  }];
  let options = [
    Option<"compiledFrameworkLabel", "compiled-framework-label", "std::string",
      /*default=*/"\"mof\"",
      "The label for resulting MGP graph. Default is mof.">,
    Option<"minimumCpuAlignment", "min-cpu-alignment", "size_t",
      /*default=*/"kPreferredMemoryAlignment",
     "Minimum alignment for all CPU buffers (both constants and allocated)">
  ];
  let dependentDialects = ["MEF::MEFDialect",
                           "MGP::MGPDialect",
                           "GRT::GRTDialect"];
}


def LowerMOGGRegionCompiles: Pass<"lower-mogg-region-compile", "ModuleOp"> {
  let summary = "Convert MOGG compile ops into MGP versions";
  let description = [{
    After lowering we will have raw MOGG compiled kernels in the graph, this
    converts them into the same MGP format used for the other compiled code.
  }];

  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect"];
}

def LowerPrimitives: Pass<"lower-primitives", "ModuleOp"> {
  let summary = "Lower MGP operations to primitives";
  let description = [{
    Using the loaded kernel library this looks for implementations for all
    primitives and if one exists will lower the primitive to that mojo
    implementation.

    At this stage these implementations will use "unpacked" synchronous values,
    not the runtime async values.
  }];

  let options = [
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo kernel definitions to be loaded.">,
  ];

  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect",
                          "M::KGEN::KGENDialect",
                          "M::KGEN::LIT::LITDialect",
                          "M::KGEN::POP::POPDialect",
                          "M::HLCF::HLCFDialect",
                          "M::DebugInfo::DebugInfoDialect",
                          "mlir::NVVM::NVVMDialect",
                          "mlir::index::IndexDialect",
  ];
}

def LowerMOGGABI: Pass<"lower-abi", "ModuleOp"> {
  let summary = "Lower MGP kernels into the correct ABI";
  let description = [{
    For each generic compilable function this pass will convert them from their
    synchronous representation into a runtime executable representation which
    contains the code to unpack from and to async values.
  }];

  let options = [
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo kernel definitions to be loaded.">,
  ];

  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect",
                          "M::KGEN::KGENDialect",
                          "M::KGEN::LIT::LITDialect",
                          "M::KGEN::POP::POPDialect",
                          "M::HLCF::HLCFDialect",
                          "M::DebugInfo::DebugInfoDialect",
                          "mlir::NVVM::NVVMDialect",
                          "mlir::index::IndexDialect",
  ];
}

def FuseGenericCompiledRegions: Pass<"fuse-generic-compiled-regions", "ModuleOp"> {
  let summary = "Fuse sequences of GenericCompiledRegionOp";
  let description = [{}];

  let options = [
    Option<"enableFusion", "enable-fusion", "bool", "true",
      "Flag to allow control of fusion pass">
  ];
  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect"];
}

def RemoveNonLowerableResults: Pass<"remove-non-lowerable-results", "ModuleOp"> {
  let summary = "Removes non lowerable result types from mogg ops";
  let description = [{}];

  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect"];
}

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_TD
