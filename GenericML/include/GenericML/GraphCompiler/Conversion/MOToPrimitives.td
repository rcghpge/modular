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
    Option<"minimumCpuAlignment", "min-cpu-alignment", "size_t",
      /*default=*/"kPreferredMemoryAlignment",
     "Minimum alignment for all CPU buffers (both constants and allocated)">
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

def LowerMOGGABI: Pass<"lower-abi", "MGP::ModelOp"> {
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

def RemoveRedundantOperands: Pass<"remove-redundant-operands", "MGP::ModelOp"> {
  let summary = "Removes all redundant operands as a result of seq fusion";
  let description = [{}];

  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect"];
}

def LowerAsyncRefs: Pass<"lower-async-refs"> {
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


def CoalesceAsyncRefs: Pass<"coalesce-async-refs"> {
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

def DumpMojo: Pass<"dump-mojo", "ModuleOp"> {
  let summary = "Dump Mojo text";
  let description = [{
    Dump to stderr the previously emitted Mojo as text.
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

def LoadMojo: Pass<"load-mojo", "ModuleOp"> {
  let summary = "Load Mojo text";
  let description = [{
    Load the provided Mojo text and replace the existing module. This can be
    achieved by dumping Mojo, modifying it and then running this pass. Then, the
    -jit-pipeline can be run to compile all the way to the end.

    Example:
    EMIT_MOJO=1 mo-opt program.mlir -mo-to-mgp -o /dev/null -mlir-print-ir-after=emit-mojo &> program.mgp.mlir
    EMIT_MOJO=1 mo-opt program.mgp.mlir -dump-mojo -o /dev/null &> program.mojo
    # Modify Mojo here
    EMIT_MOJO=1 mo-opt program.mgp.mlir -load-mojo=path=program.mojo -jit-pipeline
  }];

  let options = [
    Option<"path", "path", "std::string", "\"\"",
      "The label for resulting MGP graph. Default is mof.">
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

// NOT safe to run in //. This has to run on the ModuleOp because of how we name
// kernels.
def EmitMojo: Pass<"emit-mojo", "ModuleOp"> {
  let summary = "Emit Mojo text";
  let description = [{
    Emit Mojo text from kernel stubs and `mogg.generic.compiled.region`. The
    Mojo is put in a stub that will be held until JIT compilation.
  }];

  let options = [
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo kernel definitions to be loaded.">,
    Option<"enableOpTraceComments", "enable-op-trace-comments", "bool", "false", "Flag to enable adding comments in the Mojo code detailing which operation is being emitted.">,
    Option<"enableComputeLambdas", "enable-compute-lambdas", "bool", "true",
      "Enable compute lambdas for the fused I/O kernels">,
    Option<"moggDebugDumpFusedTensorSpecs", "mogg-debug-dump-fused-tensor-specs", "bool", "false",
      "Flag to add prints for the fused tensor specs in the lambdas">
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

def AddTryExceptBlocks: Pass<"add-try-except-blocks", "MOGG::KernelStubOp"> {
  let summary = "Adds a try-except block to raising kernels";
  let description = [{
    This pass should be run after sequence fusion and before Mojo codegen. It
    adds a try-except block to raising kernels such that the emit mojo pass can
    generate adequate Mojo.
  }];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect"];
}

def OutlineKernels: Pass<"outline-kernels", "ModuleOp"> {
  let summary = "Outline kernels into stubs";
  let description = [{
    The same way MOGG kernels are outlined into KernelStubOp as we lower to MGP,
    outline GenericCompiledRegionOp into KernelStubOp. This should be done after
    all optimization passes, just before Mojo codegen. The purpose of this pass
    is to stage things for the merging of identical kernel stubs.
  }];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect"];
}

def MergeIdenticalKernelStubs: Pass<"merge-identical-kernel-stubs", "ModuleOp"> {
  let summary = "Merge kernel stubs that are identical";
  let description = [{
    This is the equivalent of CSE but for MOGG::KernelStubOp. It can improve
    compilation time by reducing the IR size and emitted Mojo text.
  }];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect"];

  let statistics = [
    Statistic<"numMergedStubs", "# of stubs that were merged", "Number of stubs that were merged">
  ];
}

def LookupConstantResources: Pass<"lookup-constant-resources", "MOGG::KernelStubOp"> {
  let summary = "Replace mgp.buffer.constant with mgp.resource.constant";
  let description = [{
    Retrieves a pointer to the corresponding constant resources attached to the
    parent `mogg.kernel.stub` operation. This makes compilation faster because
    the value of the constant can be different but identical kernels will still
    merge if the index is the same.
  }];

  let dependentDialects = ["MGP::MGPDialect"];
}


#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_TD
