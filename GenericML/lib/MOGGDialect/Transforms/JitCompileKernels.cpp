//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include <string>
#include <unordered_map>

#include "GenericML/KGENStubBuilder/Builder.h"

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "GenericML/MODialect/MOOps.h"
#include "GenericML/MODialect/MOTypes.h"
#include "GenericML/MOGGDialect/MOGGOps.h"
#include "GenericML/MOGGDialect/MOGGPasses.h"
#include "GenericML/Support/Tensor.h"

#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "Kernels/KGEN.h"

#include "llvm/ADT/StringExtras.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_JITCOMPILEKERNELS
#include "GenericML/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Convert all possible operations directly to the avaliable operation.
struct JitCompileKernels
    : public impl::JitCompileKernelsBase<JitCompileKernels> {
  using JitCompileKernelsBase::JitCompileKernelsBase;

  void runOnOperation() override {
    MO::GraphOp graph = getOperation();

    // The path to the MOGG standard library lit bindings. The macro is set by
    // CMake.
    StringLiteral path = MOGG_LIT_BINDINGS_PATH;

    // Load the module. The stub will be appended to this.
    auto baseModuleRef =
        mlir::parseSourceFile<mlir::ModuleOp>(path, graph.getContext());
    mlir::ModuleOp baseModule = *baseModuleRef;

    // Strip all the export functions from the imported module.
    baseModule.walk([&](KGEN::ExportOp exportOp) { exportOp.erase(); });

    // Map operations with the same output sizes onto the same kernel. Longer
    // term we may want fancier optimizations like unifying aliased inputs but
    // this is legal for now.
    std::unordered_map<std::string, MOGG::ExecuteCompiledOp> jittedKernels;

    // Compile all the fused chains into a KGEN kernel.
    graph.walk([&, pass = this](MOGG::JitTargetOp op) {
      // Create a unique name for the kernel.
      std::string stubKernelName =
          "__Gen_" + op.getOriginalOpName().str() + "_";

      // Add the types to mangle the name.
      for (Type inType : op->getOperandTypes()) {
        auto tensor = cast<MO::TensorType>(inType);

        for (int64_t dim : tensor.getShape()) {
          stubKernelName +=
              ShapedType::isDynamic(dim) ? "?" : std::to_string(dim);
          stubKernelName += "x";
        }

        assert(tensor.getElementType().isF32());
        stubKernelName += "f32_";
      }

      // If we've added this exact kernel before reuse it.
      auto itr = jittedKernels.find(stubKernelName);
      if (itr != jittedKernels.end()) {
        // Clone the jit operation (with the same jit kernel attached)
        MOGG::ExecuteCompiledOp exe = itr->second.clone();

        mlir::OpBuilder opBuilder{op->getContext()};
        opBuilder.setInsertionPoint(op);
        opBuilder.insert(exe);

        // Fix up the IR and replace all the uses.
        exe->setOperands(op->getOperands());
        op->replaceAllUsesWith(exe.getResults());
        op->erase();
        return;
      }

      // Each kernel works on a clean module. We would need to compile them all
      // into the same one if not. The module will not be legal post elaboration
      // and will have lost any references.
      auto freshModule = baseModule.clone();

      // Find the KGEN kernel to call.
      KGEN::GeneratorOp kernel;
      freshModule.walk([&](KGEN::GeneratorOp generator) {
        if (generator.getSymName().starts_with("$MOGG::__mogg__")) {
          // Strip the arguments from the name.
          size_t begin = strlen("$MOGG::__mogg__");
          size_t end = generator.getSymName().find("(") - begin;
          std::string name = generator.getSymName().substr(begin, end).str();
          if (name == op.getKernelTarget()) {
            kernel = generator;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::skip();
      });

      if (!kernel) {
        llvm::errs() << "[JITKernel] ERROR could not find a kernel for "
                     << op.getKernelTarget() << "\n";
        return;
      }

      // Start building up the KGEN kernel.
      KGENStubBuilder builder{op, freshModule, stubKernelName,
                              op.getOriginalOpName().str(),
                              useSearch.getValue()};

      // Lower the basic stub function.
      builder.startLowering();

      // Create the actual KGEN call.
      auto err = builder.lowerKGENCall(kernel);
      if (err.failed()) {
        pass->signalPassFailure();
        return;
      }

      // Run KGEN elaboration then compile to ELF.
      auto result = builder.compile();
      if (result.isError()) {
        pass->signalPassFailure();
        return;
      }

      // Replace the original kernel with a call to the compiled jit code.
      MOGG::ExecuteCompiledOp jitExecute = result.takeValue();
      op->replaceAllUsesWith(jitExecute.getResults());

      jittedKernels.insert({stubKernelName, jitExecute});

      // Clean up the KGEN module.
      freshModule->destroy();

      // Remove the original operation from the graph.
      op->erase();
    });
  }
};
} // namespace

} // namespace M::MOGG
