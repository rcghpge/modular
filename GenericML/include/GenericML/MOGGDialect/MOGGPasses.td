//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOGGDIALECT_MOGGPASSES_TD
#define GENERICML_MOGGDIALECT_MOGGPASSES_TD

include "mlir/Pass/PassBase.td"

def JitCompileKernels: Pass<"jit-compile-kernels", "ModuleOp"> {
    let summary = "Compile MOGG::JitTargetOps into KGEN calls";
    let dependentDialects = [
        "M::MO::MODialect",
        "M::MOGG::MOGGDialect",
        "M::KGEN::KGENDialect",
        "M::KGEN::POP::POPDialect",
        "mlir::index::IndexDialect",
        "mlir::scf::SCFDialect",
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

def FuseElementwise: Pass<"fuse-elementwise", "MO::GraphOp"> {
    let summary = "Fuse blocks of elementwise operations if possible";
    let dependentDialects = [
        "M::MO::MODialect",
        "M::MOGG::MOGGDialect"
    ];
}

#endif // GENERICML_MOGGDIALECT_MOGGPASSES_TD
