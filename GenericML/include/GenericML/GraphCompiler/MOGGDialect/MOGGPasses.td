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
        "M::MO::MODialect",
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
        Option<"useSearch", "useSearch", "bool", /*default=*/"false",
                "Tell KGEN elaborator to search for best kernel.">,
    ];
}

def PatternFusion: Pass<"mo-pattern-fusion", "MO::GraphOp"> {
    let summary = "Fuse kernels based on patterns.";
    let dependentDialects = [
        "M::MO::MODialect",
        "M::MOGG::MOGGDialect"
    ];
}

def FuseBroadcasts: Pass<"fuse-bcasts", "MO::GraphOp"> {
    let summary = "Fuse broadcasts into other kernels.";
    let dependentDialects = [
        "M::MO::MODialect",
        "M::MOGG::MOGGDialect"
    ];
}

def FuseElementwise: Pass<"fuse-elementwise", "MO::GraphOp"> {
    let summary = "Fuse blocks of elementwise operations if possible";
    let dependentDialects = [
        "M::MO::MODialect",
        "M::MOGG::MOGGDialect"
    ];
}

def LoadLitLibrary : Pass<"load-lit-library", "ModuleOp"> {
  let summary = "Load the lit library. (nop if already loaded)";
  let description = [{
    Load the lit library off the disk and include the avaliable kernels in the mo graph.
  }];


  let dependentDialects = [
      "M::MO::MODialect",
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
     SIMD loads can be lowered to one of many implementations depending
     on kernel level circumstances. This pass looks at the kernel and picks
     the best load for each simd_load within that kernel.
   }];


   let dependentDialects = [
     "M::MO::MODialect",
     "M::MOGG::MOGGDialect"
   ];
 }


#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD
