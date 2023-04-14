//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD

include "mlir/Pass/PassBase.td"

def JitCompileKernels: Pass<"jit-compile-kernels", "ModuleOp"> {
  let summary = "Compile MOGG::JitTargetOps into KGEN calls";
  let dependentDialects = [
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "mlir::index::IndexDialect",
    "M::Cache::CacheDialect",
    "M::DebugInfo::DebugInfoDialect",
    "mlir::LLVM::LLVMDialect",
    "M::HLCF::HLCFDialect",
  ];

  let options = [
    Option<"useSearch", "use-search", "bool", /*default=*/"false",
            "Tell KGEN elaborator to search for best kernel.">,
    Option<"saveTempPrefix", "save-temp-prefix", "std::string", /*default=*/"",
            "The prefix to use when saving temporary files. Setting this "
            "option toggles on saveTemps.">,
    Option<"dumpStub", "dump-stub", "bool", /*default=*/"false",
            "Prints the generated KGEN stub for use in tests and debugging">,
  ];
}

def PatternFusion: Pass<"mo-pattern-fusion", "MO::GraphOp"> {
  let summary = "Fuse kernels based on patterns.";
  let dependentDialects = ["M::MOGG::MOGGDialect"];
  let options = [
    Option<"useMatmulPacking", "use-matmul-packing",
           "bool", /*default=*/"true", "Enable matmul bias packing.">,
  ];
}

def FuseBroadcasts: Pass<"fuse-bcasts", "MO::GraphOp"> {
  let summary = "Fuse broadcasts into other kernels.";
  let dependentDialects = ["M::MOGG::MOGGDialect"];
}

def FuseConstants: Pass<"fuse-consts", "MO::GraphOp"> {
  let summary = "Fold constants into the jit kernels";
  let dependentDialects = ["M::MOGG::MOGGDialect"];
}

def FuseElementwise: Pass<"fuse-elementwise", "MO::GraphOp"> {
  let summary = "Fuse blocks of elementwise operations if possible";
  let dependentDialects = ["M::MOGG::MOGGDialect"];
}

def LoadMojoLibrary : Pass<"load-mojo-library", "ModuleOp"> {
  let summary = "Load the mojo library (noop if already loaded).";
  let description = [{
    Load the mojo library off the disk and include the avaliable kernels in the
    MO graph.
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
  let summary = "Lower simd loads to an optimal implementation";
  let description = [{
    SIMD loads can be lowered to one of many implementations depending on kernel
    level circumstances. This pass looks at the kernel and picks the best load
    for each simd_load within that kernel.
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
 }

def InlineLambdas: Pass<"inline-lambdas", "MO::GraphOp"> {
  let summary = "Inline lambda invokes into lambdas and remove unused lambdas";
  let dependentDialects = ["M::MOGG::MOGGDialect"];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD
