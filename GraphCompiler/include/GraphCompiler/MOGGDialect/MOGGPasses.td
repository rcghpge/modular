//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD

include "mlir/Pass/PassBase.td"

def JitCompileKernels: Pass<"jit", "ModuleOp"> {
  let summary = "Replace kernel op with calls into compiled kernel";
  let description = [{
    Compiles kernel ops with the KGEN pipeline into resource blobs and replaces
    them with calls into the compiled code, inserting an op that maps kernel
    names to blobs.
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
      "Create debuginfo for kgen stubs such that the debugger will give linenumbers for the stubs">,
    Option<"defaultDeviceTargetArch", "default-device-target-arch", "std::string",
      /*default=*/"",
      "Together with default-device-target-triple, overrides the accelerator "
      "arch (e.g. sm_100a) instead of probing the host GPU. Enables "
      "cross-compilation for a target GPU on a machine that lacks it (or has no "
      "GPU at all).">,
    Option<"defaultDeviceTargetTriple", "default-device-target-triple", "std::string",
      /*default=*/"",
      "Target triple (e.g. nvptx64-nvidia-cuda) identifying the accelerator "
      "vendor for default-device-target-arch. The override applies only when "
      "this triple names a GPU target.">,
    Option<"compileForHostExecution", "compile-for-host-execution", "bool",
      /*default=*/"false",
      "Compile kernels for this machine, ignoring any virtual CPU target "
      "override. For artifacts that execute in the compiling process "
      "(constant folding).">
  ];
}

def PatternFusion: Pass<"mo-pattern-fusion", "MO::GraphOp"> {
  let summary = "Fuse kernels based on patterns.";
  let description = [{
    Replaces op sequences with a call to a hand-fused kernel, or with more
    specific ops (e.g. concat with axis 0).
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
    Option<"enableMegaFFNPattern", "mega-ffn-enable", "bool", /*default=*/"true",
            "Enable the NVFP4 MegaFFN fusion (grouped_matmul_swiglu_nvfp4 + "
            "grouped_matmul_block_scaled -> mega_ffn_nvfp4). Only fires on SM100 "
            "targets; the kernel is MAX-only, so this is disabled in OSS builds.">,
    Option<"targetDevice", "target-device",  "std::string", "\"\"",
           "If the graph operation does not already have a device specified, "
           "use this device for pattern fusion passes that depend on the target device.">,
    Option<"targetDeviceId", "target-device-id",  "size_t", "0",
           "If target-device is provided, this is the default-id to target">
  ];
}

def HashModuleContents: Pass<"mogg-hash-module-contents", "mlir::ModuleOp"> {
  let summary = "Compute and attach hash of module contents";
  let description = [{
    Attaches a hash of the module contents as an attribute, for caching.
  }];
  let dependentDialects = [
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
  ];
}

def NormalizeAllNames: Pass<"mogg-normalize-all-names", "mlir::ModuleOp"> {
  let summary = "Normalize all names in the module";
  let description = [{
    Renames symbols to __S0, __S1, ... and parameters to __P0, __P1, ... so
    caching and hashing ignore naming variations.
  }];
  let dependentDialects = [
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
  ];
}

def LowerMOGGRegionCompiles: Pass<"mogg-lower-mogg-region-compile", "ModuleOp"> {
  let summary = "Convert MOGG compile ops into MGP versions";
  let description = [{
    Converts raw MOGG compiled kernels into the MGP format used by other
    compiled code.
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

def AddProfileScopeTracing: Pass<"mogg-add-profile-scope-tracing", "MOGG::KernelStubOp"> {
  let summary = "Bracket Graph profile scope ranges inside kernel stub bodies";
  let description = [{
    Synthesizes profiler trace ranges (`mogg.profile_scope.begin` /
    `mogg.profile_scope.end`) from contiguous runs of ops that share the same
    `Graph.profile_scope(...)` layer set, plus coarser ancestor ranges over
    adjacent runs nested under a common outer scope.

    Only MGP-ABI stubs are bracketed. Runs before `mogg-add-try-except-blocks`
    because begin/end can raise and must land inside its try/except.
  }];

  let options = [
    Option<"profileScopeTracingTraceUnlabelled", "profile-scope-tracing-trace-unlabelled", "bool", /*default=*/"false",
      "Emit ranges for regions carrying no profile scope. Useful for visualizing compiler output.">
  ];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect"];
}

def WrapKernelsWithTraceOp: Pass<"mogg-wrap-kernels-with-trace-op", "MOGG::KernelStubOp"> {
  let summary = "Wraps a MOGG ABI kernel stub's body in a trace range";
  let description = [{
    `mogg.with_trace` emits the per-kernel `Trace` range. MGP-ABI stubs get
    theirs from `mogg-add-profile-scope-tracing`. Runs before
    `mogg-add-try-except-blocks`, which moves this op into its try.
  }];

  let dependentDialects = ["MOGG::MOGGDialect"];
}

def AddTryExceptBlocks: Pass<"mogg-add-try-except-blocks", "MOGG::KernelStubOp"> {
  let summary = "Adds a try-except block to raising kernels";
  let description = [{
    Runs after sequence fusion and before Mojo codegen, so emitted Mojo for
    raising kernels handles errors.
  }];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect"];
}

def OutlineGenericCompiledRegionsToFunctions: Pass<"mogg-outline-generic-compiled-regions-to-functions", "ModuleOp"> {
  let summary = "Outline kernels into stubs";
  let description = [{
    Outlines GenericCompiledRegionOp into KernelStubOp so identical stubs can
    be merged. Runs after all optimization passes, just before Mojo codegen.
  }];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect"];
}

def OutlineKernelsIntoFunctions
    : Pass<"mogg-outline-kernels-into-functions", "mlir::ModuleOp"> {
  let summary = "Outline each `mogg.kernel` body into a `mogg.kernel.stub`.";
  let description = [{
    Moves each `mogg.kernel` body into a module-level `mogg.kernel.stub` with
    the `MAP` ABI (so `EmitMojo` can branch on it), leaving a single
    `mogg.call.stub` that forwards the block arguments. The `mogg.kernel` is
    kept for its `!mo.tensor` interface, which `MOGGToMGP` lowers.
  }];

  let dependentDialects = [
    "M::MO::MODialect",
    "M::MOGG::MOGGDialect",
  ];
}

def AnonymizeDimNames: Pass<"mogg-anonymize-dim-names", "ModuleOp"> {
  let summary = "Drop symbolic dim names from every kernel stub's signature.";
  let description = [{
    Turns every `!mogg.dim<name>` in a `mogg.kernel.stub` signature into
    `!mogg.dim`, so `MergeIdenticalKernelStubs` can merge stubs that differ
    only in dim names. Runs inside `EmitMojo` just before that merge.
  }];

  let dependentDialects = ["MOGG::MOGGDialect"];
}


def MergeIdenticalKernelStubs: Pass<"mogg-merge-identical-kernel-stubs", "ModuleOp"> {
  let summary = "Merge kernel stubs that are identical";
  let description = [{
    CSE for MOGG::KernelStubOp, reducing IR size and emitted Mojo.
  }];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect"];

  let statistics = [
    Statistic<"numMergedStubs", "# of stubs that were merged", "Number of stubs that were merged">
  ];
}

def CanonicalizeKernelStubDeviceIds:
    Pass<"mogg-canonicalize-kernel-stub-device-ids", "ModuleOp"> {
  let summary = "Erase device ids inside kernel stubs so per-device copies merge";
  let description = [{
    Rewrites every `#M.device_ref<label, id>` inside each `mogg.kernel.stub`
    body (attributes and `!mgp.tensor` types) to id 0. Emission reads only the
    label, so per-device copies of a kernel (e.g. an EP-sharded MoE) then merge
    in `MergeIdenticalKernelStubs`. Call sites keep real ids. See GEX-4239.
  }];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect"];
}

def LiftBufferCacheSlots: Pass<"mogg-lift-buffer-cache-slots", "ModuleOp"> {
  let summary = "Lift inline `index.constant` slot operands of mgp.buffer.get_cached to kernel args";
  let description = [{
    In each MGP-ABI `mogg.kernel.stub`, promotes `index.constant` ops used
    only as `mgp.buffer.get_cached` slots to trailing kernel arguments,
    materialized at each `mogg.execute_kernel` call site. Kernels differing
    only in slot indices can then merge. See GEX-3685.
  }];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect", "mlir::index::IndexDialect"];
}

def LookupConstantResources: Pass<"mogg-lookup-constant-resources", "MOGG::KernelStubOp"> {
  let summary = "Replace mgp.buffer.constant with mgp.resource.constant";
  let description = [{
    Reads constants from resources attached to the parent `mogg.kernel.stub`,
    so kernels differing only in constant values still merge.
  }];

  let dependentDialects = ["MGP::MGPDialect"];
}

def EmitMojo: Pass<"mogg-emit-mojo", "ModuleOp"> {
  let summary = "Emit Mojo text";
  let description = [{
    Emits Mojo from kernel stubs and `mogg.generic.compiled.region`, held
    until JIT compilation.
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

def DumpMojo: Pass<"mogg-dump-mojo", "ModuleOp"> {
  let summary = "Dump Mojo text";
  let description = [{
    Dumps the emitted Mojo to stderr.
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

def LoadMojo: Pass<"mogg-load-mojo", "ModuleOp"> {
  let summary = "Load Mojo text";
  let description = [{
    Replaces the module's Mojo with the given file, for editing dumped Mojo
    before running -jit-pipeline.

    Example:
    mo-opt program.mlir -gc-pipeline -o /dev/null -mlir-print-ir-after=mogg-emit-mojo &> program.mgp.mlir
    mo-opt program.mgp.mlir -mogg-dump-mojo -o /dev/null &> program.mojo
    # Modify Mojo here
    mo-opt program.mgp.mlir -mogg-load-mojo=path=program.mojo -jit-pipeline
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

def ImportPackages : Pass<"mogg-import-packages", "ModuleOp"> {
  let summary = "Import Mojo kernel packages and emit mogg.kernel.decl ops";
  let description = [{
    Emits one mogg.kernel.decl per registered kernel, recording its signatures,
    so later passes read them from the IR instead of loading packages. Runs
    early in the pipeline.

    - mojo-pkg-paths: compiled .mojoc files.
    - lit-mlir-paths: kgen-translate -import-mojo output; skips built-in
      packages, for self-contained lit tests.
  }];
  let options = [
    ListOption<"mojoPkgPaths", "mojo-pkg-paths", "std::string",
      "Paths to compiled .mojoc files containing kernel definitions.">,
    ListOption<"litMlirPaths", "lit-mlir-paths", "std::string",
      "Paths to MLIR files (kgen-translate -import-mojo output) for testing. "
      "Built-in packages are not loaded when this option is used alone.">,
    Option<"append", "append", "bool", /*default=*/"false",
      "Append mode: seed the registration/primitive deduplication sets from "
      "decl ops already present in the module, and skip loading built-in "
      "packages. Callers pass only the delta paths not yet loaded; existing "
      "decls in the module are preserved.">,
  ];
  let dependentDialects = [
    "M::MOGG::MOGGDialect",
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

def RewriteOpaqueTypes : Pass<"mogg-rewrite-opaque-types", "ModuleOp"> {
  let summary = "Rewrite !mo.opaque<\"simple\"> to !mo.opaque<\"fully.qualified\">";
  let description = [{
    Rewrites each `!mo.opaque<"simple_name">` to its fully-qualified name using
    the module's `mogg.opaque_type_mapping` (written by `mogg-import-packages`).
    Unmapped types are unchanged. Kept separate so callers that loaded packages
    elsewhere (e.g. `MAXG_addKernelPackage`) can copy the mapping and run only
    this step; see the `kernel-decls-pre-populated` pipeline option.
  }];
}

def StripKernelDecls : Pass<"mogg-strip-kernel-decls", "ModuleOp"> {
  let summary = "Remove mogg.kernel.decl and mogg.primitive.decl ops from a module";
  let description = [{
    Run after the last pass that reads the kernel registry; the decls are not
    part of the final output.
  }];
}

def VerifyKernels: Pass<"mogg-verify-kernels", "ModuleOp"> {
  let summary = "Verifies that kernels are well constructed";
  let description = [{
    Emits user-facing diagnostics for malformed `mogg.kernel.decl` signatures
    (missing `execute`, mismatched shape/execute inputs, ...).
  }];
}

def DumpKernels: Pass<"mogg-dump-kernels", "ModuleOp"> {
  let summary = "Prints the internal representation of kernels";
  let description = [{
    For debugging. Works on both `LIT` and `KGEN` kernels.
  }];
}

def ConformCallOperands
    : Pass<"mogg-conform-call-operands", "mlir::ModuleOp"> {
  let summary = "Conform `mogg` call operands to the callee's signature.";
  let description = [{
    Wraps scalar arguments in `mogg.scalar.create`, packs variadic runs into
    `mogg.tuple.create`, and wraps trait-bound arguments in
    `mogg.trait.create`. Kept out of `MAPToMOGG` so fusion IR stays simpler and
    each `::emitMojo` is a local, 1:1 lowering.
  }];

  let dependentDialects = [
    "M::MOGG::MOGGDialect",
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD
