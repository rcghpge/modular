//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD

include "mlir/Pass/PassBase.td"

def JitCompileKernels: Pass<"mogg-jit-compile-kernels", "ModuleOp"> {
  let summary = "Replace kernel op with calls into compiled kernel";
  let description = [{
    This pass finds all the kernels ops, compile the kernels with the KGEN
    pipeline, and replaces the orinal kernel ops with calls into the compiled
    code.

    Note that the compiled kernels are stored as resource blobs, and the pass
    inserts an op to associate the generated kernel names with those blobs.
  }];
  let dependentDialects = [
    "M::MDialect",
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::HLCF::HLCFDialect",
    "M::DebugInfo::DebugInfoDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::ROCDL::ROCDLDialect",
    "mlir::index::IndexDialect",
  ];

  let options = [
    Option<"saveTempPrefix", "save-temp-prefix", "std::string", /*default=*/"",
            "The prefix to use when saving temporary files. Setting this "
            "option toggles on saveTemps.">,
    Option<"dumpStub", "dump-stub", "bool", /*default=*/"false",
            "Prints the generated KGEN stub for use in tests and debugging">,
    Option<"createMOGGReproducers", "create-mogg-reproducers", "bool", /*default=*/"false",
            "Creates isolated reproducers of each kernel in a new mo graph.">,
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo kernel definitions to be loaded.">,
    Option<"minimumCpuAlignment", "min-cpu-alignment", "size_t",
      /*default=*/"kPreferredMemoryAlignment",
     "Minimum alignment for all CPU buffers (both constants and allocated)">,
    Option<"timeMOGGCompile", "time-mogg-compile", "bool",
      /*default=*/"false",
     "Turns on timing for the KGEN passes invoked by mogg">,
    Option<"dryRun", "dry-run", "bool",
      /*default=*/"false",
     "Run the pass but don't actually compile.">,
    Option<"debugKGENStubs", "debug-kgen-stubs", "bool",
      /*default=*/"false",
      "Create debuginfo for kgen stubs such that the debugger will give linenumbers for the stubs">
  ];
}

def PatternFusion: Pass<"mogg-pattern-fusion", "MO::GraphOp"> {
  let summary = "Fuse kernels based on patterns.";
  let description = [{
    This pass includes a variety of pattern rewrites which replaces a series of
    operations with a primitive call to a hand-fused (instead of synthesized)
    kernel, or more specific operations (e.g., concat with axis 0).
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
  let options = [
    Option<"enableMHAPattern", "mha-enable", "bool", /*default=*/"true",
            "Enable multi-head attention pattern.">,
    Option<"debugMHAPattern", "mha-debug", "bool", /*default=*/"false",
            "Enable multi-head attention pattern matching logs. Useful for debugging pattern matching failures.">,
    Option<"mhaAllowUnsafeBCast", "mha-allow-unsafe-bcast", "bool", /*default=*/"true",
            "Allow potentially unsafe broadcasts to be pattern matched for MHA patterns.">,
    Option<"mhaAllowInternalReshape", "mha-allow-internal-reshape", "bool", /*default=*/"true",
            "Allow potentially unsafe broadcasts to be pattern matched for MHA patterns.">,
    Option<"targetDevice", "target-device",  "std::string", "\"\"",
           "If the graph operation does not already have a device specified, "
           "use this device for pattern fusion passes that depend on the target device.">,
    Option<"targetDeviceId", "target-device-id",  "size_t", "0",
           "If target-device is provided, this is the default-id to target">,
  ];
}

// TODO update this to account for general view operation (e.g., transpose in
// addition to broadcast)
def FuseViews: Pass<"mogg-fuse-views", "MO::GraphOp"> {
  let summary = "Fuse views into other views or kernels";
  let description = [{
    This pass identifies kernels that (1). contains only view operations and
    (2). are consumed by some other non-view kernel; then it fuses the view
    operations into the consumer kernel, thereby eliminating the copying at the
    end of the original view kernel.
  }];

  let dependentDialects = [
    "M::MDialect",
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::HLCF::HLCFDialect",
    "M::DebugInfo::DebugInfoDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::ROCDL::ROCDLDialect",
    "mlir::index::IndexDialect",
  ];

  let statistics = [
    Statistic<"numberOfFusedViews", "# of fused views", "Number of fused views">
  ];
}

def FuseAffineViews: Pass<"mogg-fuse-affine-views", "MO::GraphOp"> {
  let summary = "Fuse affine views into kernels";
  let description = [{
    A subset of view operations can be directly represented as affine expressions
    on index operations. These can fuse both into normal view operations and into
    elementwise lambdas.
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
  let statistics = [
    Statistic<"numberOfFusedViews", "# of fused views", "Number of fused views">
  ];
}

def FuseConstants: Pass<"mogg-fuse-consts", "MO::GraphOp"> {
  let summary = "Fold constants into the jit kernels";
  let description = [{
    This pass identifies small constant inputs to a kernel, and fuse those
    constants into the kernel body, thereby eliminating the need to pass them
    into the kernel at runtime.
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
}

def FuseElementwise: Pass<"mogg-fuse-elementwise", "MO::GraphOp"> {
  let summary = "Fuse blocks of elementwise operations if possible";
  let description = [{
    This pass identifies non-view kernels that are consumed by some other
    non-view kernel, and fuses the body lambda of the former into that of the
    consumer kernel.
  }];
  // Balloons due to dependency on MojoLibraryAnalysis
  let dependentDialects = [
    "M::MDialect",
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::HLCF::HLCFDialect",
    "M::DebugInfo::DebugInfoDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::ROCDL::ROCDLDialect",
    "mlir::index::IndexDialect",
  ];
  let options = [
    Option<"dumpDotGraph", "dump-dot-graph",
           "bool", /*default=*/"false",
           "Prints the connected elementwise graph we traverse to find fusions.">,
  ];
  let statistics = [
    Statistic<"numberOfFusedKernels", "# of fused kernels", "Number of fused kernels">
  ];
}

def FuseProloguesEpilogues: Pass<"mogg-fuse-prologues-epilogues", "MO::GraphOp"> {
  let summary = "Fuse kernels into the lambda of their users";
  let description = [{
    An elementwise kernel can be directly fused into the lambda body of another
    kernel. I.E fused directly into a matmul output loop.
  }];

  // Balloons due to dependency on MojoLibraryAnalysis
  let dependentDialects = [
    "M::MDialect",
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::HLCF::HLCFDialect",
    "M::DebugInfo::DebugInfoDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::ROCDL::ROCDLDialect",
    "mlir::index::IndexDialect",
  ];

  let statistics = [
    Statistic<"numberOfFusedEpilogues", "# of fused epiloges", "Number of fused epiloges">,
    Statistic<"numberOfFusedPrologues", "# of fused prologues", "Number of fused prologues">
  ];
}

def FuseMutableLoads: Pass<"mogg-fuse-mutable-loads", "MO::GraphOp"> {
  let summary = "'Fuse' mo.mutable.load ops with mogg.kernel ops.";
  let description = [{
    Will perform 'fusion' of a mo.mutable.load op with a mogg.kernel op.
    This amounts to:
      1. Replacing the value semantic tensor created by the mutable.load op
         with the buffer semantic tensor
      2. Making the input chain of the load op an input of the mogg.kernel op
      3. Adding the output chain of the mutable.load op to the output of the
         mogg.kernel op
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
}

def SimplifyKernelOperands: Pass<"mogg-simplify-kernel-operands", "MO::GraphOp"> {
  let summary = "Gets rid of duplicate kernel operands";
  let description = [{
    This pass uniques kernel operands.
  }];
  let dependentDialects = [
    "M::MDialect",
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
  ];
}

def HashModuleContents: Pass<"mogg-hash-module-contents", "mlir::ModuleOp"> {
  let summary = "Compute and attach hash of module contents";
  let description = [{
    This pass computes a hash of the module contents and attaches it as an
    attribute. This can be used for caching, verification, or tracking changes
    to the module.
  }];
  let dependentDialects = [
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
  ];
}

def NormalizeAllNames: Pass<"mogg-normalize-all-names", "mlir::ModuleOp"> {
  let summary = "Normalize all names in the module";
  let description = [{
    This pass normalizes all symbol and parameter names to a canonical form.
    Symbol names are renamed to __S0, __S1, etc., and parameter names are
    renamed to __P0, __P1, etc. This makes the IR less sensitive to naming
    variations, which is useful for caching and hashing purposes.
  }];
  let dependentDialects = [
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
  ];
}

// Passes moved from MOToPrimitives (mogg-* passes that operate on MOGG dialect)
def LowerMOGGRegionCompiles: Pass<"mogg-lower-mogg-region-compile", "ModuleOp"> {
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

def SequenceFusion: Pass<"mogg-sequence-fusion", "MGP::ModelOp"> {
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

def RemoveRedundantOperands: Pass<"mogg-remove-redundant-operands", "MGP::ModelOp"> {
  let summary = "Removes all redundant operands as a result of seq fusion";
  let description = [{}];

  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect"];
}

def AddTryExceptBlocks: Pass<"mogg-add-try-except-blocks", "MOGG::KernelStubOp"> {
  let summary = "Adds a try-except block to raising kernels";
  let description = [{
    This pass should be run after sequence fusion and before Mojo codegen. It
    adds a try-except block to raising kernels such that the emit mojo pass can
    generate adequate Mojo.
  }];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect"];
}

def OutlineKernels: Pass<"mogg-outline-kernels", "ModuleOp"> {
  let summary = "Outline kernels into stubs";
  let description = [{
    The same way MOGG kernels are outlined into KernelStubOp as we lower to MGP,
    outline GenericCompiledRegionOp into KernelStubOp. This should be done after
    all optimization passes, just before Mojo codegen. The purpose of this pass
    is to stage things for the merging of identical kernel stubs.
  }];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect"];
}

def MergeIdenticalKernelStubs: Pass<"mogg-merge-identical-kernel-stubs", "ModuleOp"> {
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

def LookupConstantResources: Pass<"mogg-lookup-constant-resources", "MOGG::KernelStubOp"> {
  let summary = "Replace mgp.buffer.constant with mgp.resource.constant";
  let description = [{
    Retrieves a pointer to the corresponding constant resources attached to the
    parent `mogg.kernel.stub` operation. This makes compilation faster because
    the value of the constant can be different but identical kernels will still
    merge if the index is the same.
  }];

  let dependentDialects = ["MGP::MGPDialect"];
}

def EmitMojo: Pass<"mogg-emit-mojo", "ModuleOp"> {
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

def DumpMojo: Pass<"mgp-dump-mojo", "ModuleOp"> {
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

def LoadMojo: Pass<"mgp-load-mojo", "ModuleOp"> {
  let summary = "Load Mojo text";
  let description = [{
    Load the provided Mojo text and replace the existing module. This can be
    achieved by dumping Mojo, modifying it and then running this pass. Then, the
    -jit-pipeline can be run to compile all the way to the end.

    Example:
    mo-opt program.mlir -mo-to-mgp -o /dev/null -mlir-print-ir-after=mogg-emit-mojo &> program.mgp.mlir
    mo-opt program.mgp.mlir -mgp-dump-mojo -o /dev/null &> program.mojo
    # Modify Mojo here
    mo-opt program.mgp.mlir -mgp-load-mojo=path=program.mojo -jit-pipeline
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

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD
