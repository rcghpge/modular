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
    "M::KGEN::Custom::CustomDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::HLCF::HLCFDialect",
    "M::DebugInfo::DebugInfoDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
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
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo kernel definitions to be loaded.">,
    Option<"minimumCpuAlignment", "min-cpu-alignment", "size_t",
      /*default=*/"kPreferredMemoryAlignment",
     "Minimum alignment for all CPU buffers (both constants and allocated)">,
    Option<"splitBinaryCompile", "split-binary-compile", "bool",
      /*default=*/"false",
     "Split compilation of kernels into seperate output kernel binaries">,
    Option<"timeMOGGCompile", "time-mogg-compile", "bool",
      /*default=*/"false",
     "Turns on timing for the KGEN passes invoked by mogg">,
    Option<"skipJitCompileMiddle", "skip-jit-compile-middle", "bool",
      /*default=*/"true",
     "Skip jit compile for mogg kernels, which get lowered to compiled region">,
  ];
}

def TestRunningMogg: Pass<"test-run-mogg", "ModuleOp"> {
  let summary = "A pass which executes and prints the given graph";
  let description = [{
    A pass to test we can execute a kernel directly without MGP.

    This has no other purpose than a test target.
  }];

  let dependentDialects = [
    "M::MDialect",
    "M::MOGG::MOGGDialect",
    "M::KGEN::Custom::CustomDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::HLCF::HLCFDialect",
    "M::DebugInfo::DebugInfoDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::index::IndexDialect",
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
  let options = [
    Option<"enableMHAPattern", "mha-enable", "bool", /*default=*/"true",
            "Enable multi-head attention pattern.">,
    Option<"debugMHAPattern", "mha-debug", "bool", /*default=*/"false",
            "Enable multi-head attention pattern matching logs. Useful for debugging pattern matching failures.">,
    Option<"enableRank4GPUMask", "mha-gpu-enable-rank4-mask", "bool", /*default=*/"false",
            "Enable masks on GPU to be rank 4.">,
    Option<"mhaAllowUnsafeBCast", "mha-allow-unsafe-bcast", "bool", /*default=*/"true",
            "Allow potentially unsafe broadcasts to be pattern matched for MHA patterns.">,
    Option<"mhaAllowInternalReshape", "mha-allow-internal-reshape", "bool", /*default=*/"true",
            "Allow potentially unsafe broadcasts to be pattern matched for MHA patterns.">,
  ];
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
  let summary = "Fuse a sequence of kernels into a single kernel.";
  let description = [{
    This pass will combine a sequence of MOGG::KernelOp invocations into a
    single invocation. The original kernels will be called in some topological
    order. Intermediate tensors may be stack-allocated within the fused kernel.

    Generally only 'small' kernels and 'small' intermediate tensors may be
    fused.
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];

  let options = [
    Option<"smallTensorLimit", "small-tensor-limit", "int64_t", "1024",
           "Maximum number of elements a tensor may have to be considered "
           "'small'. These may be stack allocated within the fused kernel.">,
    Option<"nonHostKernelLimit", "non-host-kernel-limit", "size_t", "3",
           "Maximum number of non-host kernels to place in a single fused "
           "kernel.">,
  ];
}

def FuseInOutLambdas: Pass<"fuse-in-out-lambdas", "MO::GraphOp"> {
  let summary = "Fuse kernels into the lambda of their users";
  let description = [{
    An elementwise kernel can be directly fused into the lambda body of another
    kernel. I.E fused directly into a matmul output loop.
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
}

def FuseMutableOps: Pass<"fuse-mutable-ops", "MO::GraphOp"> {
  let summary = "Fuse mo.mutable.load/store ops into kernel bodies.";
  let description = [{
    Will fuse in a mo.mutable.load and mo.mutable.store op pair surrounding a
    mogg.kernel under appropriate conditions.

    Mechanically this is done by:
    1. Replacing the value semantic tensor created by the mo.mutable.load op
    that is input to the kernel with the buffer semantic tensor it's derived
    from.
    2. Adding the chain consumed by the load op as operand of the kernel op.
    3. Replacing the old output of the kernel with a chain and having all
    former consumers of the output chain produced by the store op take this new
    chain.
    4. Rewriting the kernel body to make use of the buffer semantic tensor in
    the output lambda rather than the one created by the output placeholder op.
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
}

def MarkTrivialKernels: Pass<"mark-trivial-kernels", "MO::GraphOp"> {
  let summary = "Decide which `MOGG::KernelOp`s can be sequentially fused.";
  let description = [{
    Adds the attribute `single_thread_blocking_override=true` to all
    `MOGG::KernelOp` which can be fused into a sequence of kernel calls.
    Such a sequence can avoid the overhead of the LLCL::AsyncValue machinery
    to coordinate dataflow between kernel calls.

    On CPU this is only enabled for 'trivial' kernels, ie those with small
    input and output shapes. For GPU this is enabled generally.
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

def SimplifyKernelOperands: Pass<"simplify-kernel-operands", "MO::GraphOp"> {
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

def StackAllocateTemporaries: Pass<"stack-allocate-temporaries", "MO::GraphOp"> {
  let summary = "Converts small dynamic temporaries to stack allocations";
  let description = [{
    Any output placeholder that is small enough and that is not an actual output
    is converted to a stack allocation.
  }];
  let options = [
    Option<"smallTensorLimit", "small-tensor-limit", "uint64_t", "5000",
           "Maximum number of elements a tensor may have to be eligible for a"
           "stack allocation.">,
  ];
  let dependentDialects = [
    "M::MDialect",
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
  ];
}

// This pass has to match a ModuleOp because it lowers operations which is not
// thread safe.
def KernelSequenceFuse: Pass<"sequence-fuse-kernels", "ModuleOp"> {
  let summary = "Fuse a sequence of kernels into a single kernel.";
  let description = [{
    This pass will combine a sequence of MOGG::KernelOp invocations into a
    single invocation. The original kernels will be called in some topological
    order. Intermediate tensors may be stack-allocated within the fused kernel.

    Generally only 'small' kernels and 'small' intermediate tensors may be
    fused.
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];

  let options = [
    Option<"nonHostKernelLimit", "non-host-kernel-limit", "size_t", "3",
           "Maximum number of non-host kernels to place in a single fused "
           "kernel.">,
    Option<"heuristic", "heuristic", "std::string", "\"conservative\"",
           "Which of the heuristics to use for evaluating which kernels to"
           "fuse. Choices are (greedy|conservative|floodfill).">,
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo kernel definitions to be loaded.">,
  ];

  let dependentDialects = [
    "M::MDialect",
    "M::MOGG::MOGGDialect",
    "M::KGEN::Custom::CustomDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::HLCF::HLCFDialect",
    "M::DebugInfo::DebugInfoDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::index::IndexDialect",
  ];
}

def TranslateMOGGToKGEN: Pass<"translate-mogg-to-kgen", "ModuleOp"> {
  let summary = "Translates MOGG kernels into KGEN kernels";
  let description = [{
    This pass will convert MOGG kernels (buffer semantics NDBuffer kernels) into
    KGEN kernels (Buffer/Value semantics Tensor API).
  }];
  let dependentDialects = [
    "M::MDialect",
    "M::MOGG::MOGGDialect",
    "M::KGEN::Custom::CustomDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::HLCF::HLCFDialect",
    "M::DebugInfo::DebugInfoDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::index::IndexDialect",
  ];

  let options = [
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo kernel definitions to be loaded.">,
    Option<"minimumCpuAlignment", "min-cpu-alignment", "size_t",
      /*default=*/"kPreferredMemoryAlignment",
     "Minimum alignment for all CPU buffers (both constants and allocated)">
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD
