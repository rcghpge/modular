//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD

include "mlir/Pass/PassBase.td"

def JitCompileKernels: Pass<"jit-compile-kernels", "ModuleOp"> {
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
    "M::Cache::CacheDialect",
    "M::DebugInfo::DebugInfoDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::index::IndexDialect",
  ];

  let options = [
    Option<"useSearch", "use-search", "bool", /*default=*/"false",
            "Tell KGEN elaborator to search for best kernel.">,
    Option<"saveTempPrefix", "save-temp-prefix", "std::string", /*default=*/"",
            "The prefix to use when saving temporary files. Setting this "
            "option toggles on saveTemps.">,
    Option<"dumpStub", "dump-stub", "bool", /*default=*/"false",
            "Prints the generated KGEN stub for use in tests and debugging">,
    Option<"createMOGGReproducers", "create-mogg-reproducers", "bool", /*default=*/"false",
            "Creates isolated reproducers of each kernel in a new mo graph.">,
    Option<"minimumCpuAlignment", "min-cpu-alignment", "size_t",
      /*default=*/"kPreferredMemoryAlignment",
     "Minimum alignment for all CPU buffers (both constants and allocated)">,
  ];
}

def PatternFusion: Pass<"mo-pattern-fusion", "MO::GraphOp"> {
  let summary = "Fuse kernels based on patterns.";
  let description = [{
    This pass includes a variety of pattern rewrites which replaces a series of
    operations with a primitive call to a hand-fused (instead of synthesized)
    kernel, or more specific operations (e.g., concat with axis 0).
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
}

// TODO update this to account for general view operation (e.g., transpose in
// addition to broadcast)
def FuseBroadcasts: Pass<"fuse-bcasts", "MO::GraphOp"> {
  let summary = "Fuse broadcasts into other kernels";
  let description = [{
    This pass identifies kernels that (1). contains only view operations and
    (2). are consumed by some other non-view kernel; then it fuses the view
    operations into the consumer kernel, thereby eliminating the copying at the
    end of the original view kernel.
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
}

def FuseConstants: Pass<"fuse-consts", "MO::GraphOp"> {
  let summary = "Fold constants into the jit kernels";
  let description = [{
    This pass identifies small constant inputs to a kernel, and fuse those
    constants into the kernel body, thereby eliminating the need to pass them
    into the kernel at runtime.
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
}

def FuseElementwise: Pass<"fuse-elementwise", "MO::GraphOp"> {
  let summary = "Fuse blocks of elementwise operations if possible";
  let description = [{
    This pass identifies non-view kernels that are consumed by some other
    non-view kernel, and fuses the body lambda of the former into that of the
    consumer kernel.
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
  let options = [
    Option<"dumpDotGraph", "dump-dot-graph",
           "bool", /*default=*/"false",
           "Prints the connected elementwise graph we traverse to find fusions.">,
  ];
}

def SoftFuse: Pass<"soft-fuse", "MO::GraphOp"> {
  let summary = "Soft fuse kernels together";
  let description = [{
    When we have mogg blocks of trivial kernels we can soft fuse them together.
    Soft fusion is the process of colocating kernels into the same invocation.
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
}

def MarkTrivialKernels: Pass<"mark-trivial-kernels", "MO::GraphOp"> {
  let summary = "Statically detect if kernels are trivial";
  let description = [{
    Check if a kernel is trivial and only contains calls to `single_thread_blocking_override`
    compatible kernels. If so then we are able to potentially avoid the async overhead.
    This enables soft fusion of those kernels.
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
}

def AddExplicitChainReady: Pass<"add-explicit-chain-ready", "MO::GraphOp"> {
  let summary = "Adds a chain ready if needed";
  let description = [{
    When we mark kernels as being single thread blocking we are opting them into
    behaviour which prevents them from setting the chain ready themselves. 
    Instead the chain ready should be set at the end of a cluster of these
    calls. This pass will detect when we need to insert such a call.
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
}

def LoadMojoLibrary : Pass<"load-mojo-library", "ModuleOp"> {
  let summary = "Load the mojo library";
  let description = [{
    This pass loads the mojo library off the disk, find all available kernels
    from the library, and add a `MojoRegisterOp` that includes all those
    kernels.
  }];
  let dependentDialects = [
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "mlir::index::IndexDialect",
    "M::DebugInfo::DebugInfoDialect",
    "M::HLCF::HLCFDialect",
  ];
}

def LowerSimdLoads : Pass<"lower-simd-loads", "ModuleOp"> {
  let summary = "Lower simd loads to optimal implementations";
  let description = [{
    SIMD loads can be lowered to one of many implementations depending on kernel
    level circumstances. This pass looks at the kernel and picks the best load
    for each simd_load within that kernel.
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
 }

def InlineLambdas: Pass<"inline-lambdas", "MO::GraphOp"> {
  let summary = "Inline lambda invokes and remove unused lambdas";
  let description = [{
    This passes replaces individual `LambdaInvokeOp` by inlining the body of the
    invoked lambda, if the lambda body is not available. It also erases all the
    lambdas that become unused after inlining.

    Note that this also disables fusion because the lambda hooks will be inlined
    and erased.
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD
