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
  let dependentDialects = ["GRT::GRTDialect",
                           "KGEN::POP::POPDialect",
                           "MEF::MEFDialect",
                           "MGP::MGPDialect",
                           "mlir::index::IndexDialect"];
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

def LowerPrimitives: Pass<"lower-primitives", "MGP::ModelOp"> {
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
                          "mlir::ROCDL::ROCDLDialect",
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
                          "mlir::ROCDL::ROCDLDialect",
                          "mlir::index::IndexDialect",
  ];
}

def FuseGenericCompiledRegions: Pass<"fuse-generic-compiled-regions", "MGP::ModelOp"> {
  let summary = "Fuse sequences of GenericCompiledRegionOp";
  let description = [{}];

  let options = [
    Option<"enableTransitiveFusion", "enable-transitive-fusion", "bool", "false",
      "Flag to allow control of fusion pass">
  ];
  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect"];
}

def PostABIFuseGenericRegions: Pass<"post-abi-fuse-generic-regions", "ModuleOp"> {
  let summary = "Fuse sequences of GenericCompiledRegionOp (after lowering abi)";
  let description = [{
    Fuses all (currently only non-async) kernels into one. This helps reduce
    overhead in the MEF interpreter by minimizing the number of mpg primitives.
    The fusion need to be done after lowering abi to be able to analyze and
    optimize the lifetime of async values (e.g., due to mgp.buffer.borrow).
  }];

  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect"];
}

def RemoveRedundantOperands: Pass<"remove-redundant-operands", "ModuleOp"> {
  let summary = "Removes all redundant operands as a result of seq fusion";
  let description = [{}];

  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect"];
}

def LowerAsyncRefs: Pass<"lower-async-refs", "ModuleOp"> {
  let summary = "Lower mogg.pack/unpack to mogg.calls";
  let description = [{
    Lower mogg.pack/upack calls to/from async refs to the corresponding
    mogg.calls.
  }];

  let options = [
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo kernel definitions to be loaded.">,
  ];

  let dependentDialects = ["MOGG::MOGGDialect"];
}


def LegalizeAsyncRefDeallocations: Pass<"legalize-async-ref-deallocations", "ModuleOp"> {
  let summary = "Legalize deallocation calls on async values";
  let description = [{
    Legalizes deallocation calls on async values in a compiled region. Double
    free might occur after post-abi fusion and/or cse. This pass tries to remove
    extra deallocation calls such that the compiled region will be free of
    memory errors.
  }];

  let dependentDialects = ["MOGG::MOGGDialect"];
}


def CoalesceAsyncRefs: Pass<"coalesce-async-refs", "ModuleOp"> {
  let summary = "coalesce async ref (de)allocation sites";
  let description = [{
    Coalesces async ref (de)allocation sites and hoist the allocation sites to
    the beginning of the block. This pass will make it easier to
    async refs when a runtime error happens.

    Note that the correspondence between async ref and the wrapped object will
    no longer be available after the pass.

    Before:
    ```
    ...
    %a0 = allocate_async_ref_for(%a)
    use(%a0)
    ...
    %a1 = allocate_async_ref_for(%b)
    use(%a1)
    ...
    ```

    After:
    ```
    // This is beginning of the block
    %a:2 = mgp.allocate_async_refs
    ...
    use(%a#0)
    ...
    use(%a#1)
    ```
  }];

  let dependentDialects = ["MOGG::MOGGDialect",
                           "MGP::MGPDialect"];
}

def EmitMojo: Pass<"emit-mojo", "ModuleOp"> {
  let summary = "Emit Mojo text";
  let description = [{
    Emit Mojo text from kernel stubs and `mogg.generic.compiled.region`. The
    Mojo is put in a stub that will be held until JIT compilation.
  }];

  let options = [
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo kernel definitions to be loaded.">,
    Option<"dumpAsText", "dump-as-text", "bool", "false",
       "Flag to dump the emitted mojo as text">
  ];

  let dependentDialects = ["MOGG::MOGGDialect",
                           "MGP::MGPDialect"];
}


#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_TD
