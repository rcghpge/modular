//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include <string>
#include <unordered_map>

#include "GenericML/KGENStubBuilder/Builder.h"
#include "GenericML/KGENStubBuilder/StubCompiler.h"

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

#include "llvm/ADT/StringExtras.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_JITCOMPILEKERNELS
#include "GenericML/MOGGDialect/MOGGPasses.h.inc"

namespace {

// If this is a single elementwise operation we can pass along the shape
// inference function hack.
std::string getShapeInferenceName(MOGG::ElementwiseOp op) {
  mlir::Block &block = op.getBody().front();
  if (block.getOperations().size() != 2)
    return "NoShapeInferFunc";

  // In practice shouldn't be reachable, but legally could just be a return
  // block for the case of an empty elementwise op.
  auto asLit = dyn_cast<MOGG::LitOp>(*block.begin());
  if (!asLit)
    return "NoShapeInferFunc";
  return asLit.getOriginalOpName().str();
}

// We create a mangled name which exactly matches the operation. The number of
// inputs, the contents of the block, and which are returned. It needs to be
// this verbose to ensure we don't recompile for identical blocks but also allow
// for blocks of the same operations with different returns to compile legally.
std::string createMangledName(MOGG::ElementwiseOp op) {
  std::string name = "__gen__ins_";

  // Track where each value is in the block order. This is to mangle the returns
  // to ensure legality.
  DenseMap<Value, std::string> valueToPositionInBlock;
  size_t argIndex = 0;

  // Get the MLIR block.
  mlir::Block &block = op.getBody().front();
  for (Value v : block.getArguments()) {
    valueToPositionInBlock[v] = "i" + std::to_string(argIndex);
    argIndex++;

    // Add input id to the name.
    name += valueToPositionInBlock[v] + "_";

    // Mangle the type into the name.
    auto tensor = cast<MO::TensorType>(v.getType());
    for (int64_t dim : tensor.getShape()) {
      name += ShapedType::isDynamic(dim) ? "?" : std::to_string(dim);
      name += "x";
    }
    assert(tensor.getElementType().isF32());
    name += "f32_";
  }

  size_t blockIndex = 0;

  // Each operation in the block of ops contributes to the name.
  for (mlir::Operation &innerOp : op.getBody().front()) {
    // Handle the output as a special case, report which operations returned a
    // value from the block.
    if (isa<MOGG::OutputOp>(innerOp)) {
      name += "outs_";
      for (Value operand : innerOp.getOperands())
        name += valueToPositionInBlock[operand] + "_";
      break;
    }

    name += cast<MOGG::LitOp>(innerOp).getOriginalOpName().str() + "_";

    // Add operands to the mangling.
    for (Value input : innerOp.getOperands())
      name += valueToPositionInBlock[input] + "_";

    // Track the outputs for future mangling.
    for (Value output : innerOp.getResults())
      valueToPositionInBlock[output] = std::to_string(blockIndex);

    // Add double seperator between operations.
    name += "_";
    blockIndex++;
  }

  return name;
}

// Convert all possible operations directly to the avaliable operation.
struct JitCompileKernels
    : public impl::JitCompileKernelsBase<JitCompileKernels> {
  using JitCompileKernelsBase::JitCompileKernelsBase;

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    for (auto graph : m.getOps<MO::GraphOp>())
      this->runPass(graph);
  }

  void runPass(MO::GraphOp graph) {
    // The path to the MOGG standard library lit bindings. The macro is set by
    // CMake.
    StringLiteral path = MOGG_LIT_BINDINGS_PATH;

    // Load the module. The stub will be appended to this.
    mlir::OwningOpRef<mlir::ModuleOp> baseModuleRef =
        mlir::parseSourceFile<mlir::ModuleOp>(path, graph.getContext());

    if (!baseModuleRef) {
      graph.emitError() << "[JitCompileKernels] Could not load MOGG LIT "
                           "binding library at path "
                        << path;
      return;
    }

    mlir::ModuleOp baseModule = *baseModuleRef;

    // Helper to track all the types to prevent constant reallocation.
    Stub::TypeManager manager{graph.getContext()};

    // Strip all the export functions from the imported module.
    baseModule.walk([&](KGEN::ExportOp exportOp) { exportOp.erase(); });

    // Map operations with the same output sizes onto the same kernel. Longer
    // term we may want fancier optimizations like unifying aliased inputs but
    // this is legal for now.
    std::unordered_map<std::string, MOGG::ExecuteCompiledOp> jittedKernels;

    // Track all the attributes we have added.
    SmallVector<mlir::Attribute> symbolNamesAdded;

    // Start building up the KGEN kernel.
    Stub::KGENBuilder builder{manager, baseModule};

    // Build stubs for all JIT operations and replace them with symbolic calls
    // to the kernels in the KGEN stub.
    graph.walk([&, pass = this](MOGG::ElementwiseOp op) {
      // Create a unique name for the kernel.
      std::string stubKernelName = createMangledName(op);

      // Don't duplicate KGEN kernels for the same call.
      auto itr = jittedKernels.find(stubKernelName);
      if (itr != jittedKernels.end()) {
        // Clone the jit operation (with the same jit kernel attached)
        MOGG::ExecuteCompiledOp exe = itr->second.clone();

        // Add the execute op into the block at the same point as the current
        // op.
        mlir::OpBuilder opBuilder{op->getContext()};
        opBuilder.setInsertionPoint(op);
        opBuilder.insert(exe);

        // Fix up the IR and replace all the uses.
        exe->setOperands(op->getOperands());
        op->replaceAllUsesWith(exe.getResults());
        op->erase();
        return;
      }

      // Lower the basic stub function.
      builder.startLowering(op, stubKernelName);

      // Lower the operation using a vectorized map over an elementwise lambda.
      auto errOr = builder.lowerToElementwiseLambda(op.getBody().front());
      if (errOr.isError()) {
        pass->signalPassFailure();
        return;
      }

      // The propagate the hacky shape inference name.
      std::string shapeInferFunc = getShapeInferenceName(op);

      // Replace the original kernel with a call to the compiled jit code.
      MOGG::ExecuteCompiledOp jitExecute =
          builder.createExecuteOperation(stubKernelName, shapeInferFunc);
      op->replaceAllUsesWith(jitExecute.getResults());
      jittedKernels.insert({stubKernelName, jitExecute});
      op->erase();

      // Tracking attributes makes it easier to create an array.
      symbolNamesAdded.push_back(jitExecute.getKernelNameAttr());
    });

    // Convert our list into a combined attribute.
    auto allSyms = mlir::ArrayAttr::get(graph->getContext(), symbolNamesAdded);

    // Add the compiled binary to the file.
    Stub::StubCompiler compiler{baseModule, graph, useSearch.getValue()};
    auto errOr = compiler.compile(allSyms);
    if (errOr.isError()) {
      graph.emitError() << "[JitCompileKernels] " << errOr.getError();
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace M::MOGG
