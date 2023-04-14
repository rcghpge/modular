//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include <string>
#include <unordered_map>

#include "GenericML/GraphCompiler/KGENStubBuilder/Builder.h"
#include "GenericML/GraphCompiler/KGENStubBuilder/StubCompiler.h"

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/Support/Tensor.h"

#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"

#include "llvm/ADT/StringExtras.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_JITCOMPILEKERNELS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Directly embed the mojo file rather than loading dynamically.
#include MOGG_MOJO_BINDING_H

std::string mangleBlock(mlir::Block &block,
                        DenseMap<Value, std::string> &valueToPositionInBlock) {
  std::string name = "";
  size_t blockIndex = 0;
  size_t numConstants = 0;

  // Each operation in the block of ops contributes to the name.
  for (mlir::Operation &innerOp : block) {
    // Handle the output as a special case, report which operations returned a
    // value from the block.
    if (isa<MOGG::OutputOp>(innerOp)) {
      name += "outs_";
      for (Value operand : innerOp.getOperands())
        name += valueToPositionInBlock[operand] + "_";
    } else if (auto lambda = dyn_cast<MOGG::LambdaOp>(innerOp)) {
      mlir::Block &lambdaBlock = lambda.getBody().front();

      // Mangle the lambda block arguments.
      for (size_t argIndex = 0, e = lambdaBlock.getNumArguments(); argIndex < e;
           ++argIndex)
        valueToPositionInBlock[lambdaBlock.getArgument(argIndex)] =
            "L" + std::to_string(argIndex);

      name += mangleBlock(lambdaBlock, valueToPositionInBlock);
    } else if (auto out = dyn_cast<MOGG::OutputPlaceholdersOp>(innerOp)) {
      for (Value output : out.getResults())
        valueToPositionInBlock[output] = "O" + std::to_string(blockIndex);
    } else if (auto invokeOp = dyn_cast<MOGG::LambdaInvokeOp>(innerOp)) {
      name += "invoke_" + invokeOp.getLambda().getLeafReference().str() + "_";

      // Add operands to the mangling.
      for (Value input : innerOp.getOperands())
        name += valueToPositionInBlock[input] + "_";

      // Track the outputs for future mangling.
      for (Value output : innerOp.getResults())
        valueToPositionInBlock[output] = std::to_string(blockIndex);

    } else if (auto call = dyn_cast<MOGG::CallOp>(innerOp)) {
      name += "call_" + call.getLitKernelTarget().str();

      // If we have parameters we have to mangle those too.
      if (call.getParams()) {
        for (SymbolRefAttr sym :
             call.getParamsAttr().getAsRange<SymbolRefAttr>())
          name += sym.getLeafReference().str() + "_";
      }

      for (Value input : innerOp.getOperands())
        name += valueToPositionInBlock[input] + "_";

      // Track the outputs name for future mangling.
      for (Value output : innerOp.getResults())
        valueToPositionInBlock[output] = std::to_string(blockIndex);
    } else if (auto cst = dyn_cast<MO::ConstantOp>(innerOp)) {
      std::string constantID = "CST" + std::to_string(numConstants);
      numConstants++;

      valueToPositionInBlock[cst.getResult()] = constantID;

      // Add to the mangling soup.
      name += constantID + "_";

      // Add the actual constant to the soup.
      auto attr = cast<M::ArrayElementsAttr>(cst.getValueAttr());
      const float *asFloat =
          reinterpret_cast<const float *>(attr.getRawData().data());
      for (int64_t i = 0, e = attr.getNumElements(); i < e; ++i)
        name += std::to_string(asFloat[i]) + "_";
    }

    // Add double seperator between operations.
    name += "_";
    blockIndex++;
  }

  return name;
}

// We create a mangled name which exactly matches the operation. The number of
// inputs, the contents of the block, and which are returned. It needs to be
// this verbose to ensure we don't recompile for identical blocks but also allow
// for blocks of the same operations with different returns to compile legally.
std::string createMangledName(MOGG::KernelOp op) {
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
    for (int64_t dim : tensor.getIntDims()) {
      name += ShapedType::isDynamic(dim) ? "?" : std::to_string(dim);
      name += "x";
    }
    assert(tensor.getElementType().isF32());
    name += "f32_";
  }

  // Add the output type as the kernel could reference this independently.
  name += "_out_ty_";
  for (Type res : op->getResultTypes()) {
    auto tensor = cast<MO::TensorType>(res);
    for (int64_t dim : tensor.getIntDims()) {
      name += ShapedType::isDynamic(dim) ? "?" : std::to_string(dim);
      name += "x";
    }
    assert(tensor.getElementType().isF32());
    name += "f32_";
  }
  name += "_";

  name += mangleBlock(op.getBody().front(), valueToPositionInBlock);
  return name;
}

// Convert all possible operations directly to the avaliable operation.
struct JitCompileKernels
    : public impl::JitCompileKernelsBase<JitCompileKernels> {
  using JitCompileKernelsBase::JitCompileKernelsBase;

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();

    // Load the mojo kernel library info.
    FailureOr<MOGG::MojoRegisterOp> registryOr = getOrLoadMojoLibrary(m);
    if (failed(registryOr)) {
      m.emitError() << "[JitCompileKernels] Couldn't load mojo kernel registry";
      return signalPassFailure();
    }
    MOGG::MojoRegisterOp registry = *registryOr;

    // Run the jit required passes.
    mlir::PassManager pm(m->getContext());
    pm.addPass(createLowerSimdLoads());
    pm.addNestedPass<MO::GraphOp>(createInlineLambdas());
    if (failed(pm.run(m)))
      return signalPassFailure();

    for (auto graph : m.getOps<MO::GraphOp>()) {
      auto errOr = this->runPass(graph, registry);
      if (errOr.isError()) {
        graph.emitError() << "[JitCompileKernels] " << errOr.getError();
        return signalPassFailure();
      }
    }
  }

  ErrorOr<mlir::OwningOpRef<mlir::ModuleOp>> loadMojoLibrary() {

    // Load the module. The stub will be appended to this.
    llvm::StringRef source{
        reinterpret_cast<const char *>(MojoMoggBindings::data),
        MojoMoggBindings::size};
    mlir::OwningOpRef<mlir::ModuleOp> baseModuleRef =
        mlir::parseSourceString<mlir::ModuleOp>(source,
                                                getOperation()->getContext());

    if (!baseModuleRef) {
      return Error("[JitCompileKernels] Could not load the embedded MOGG Mojo "
                   "binding library at path ");
    }

    // Strip all the export functions from the imported module.
    baseModuleRef->walk([&](KGEN::ExportOp exportOp) { exportOp.erase(); });

    return baseModuleRef;
  }

  ErrorOrSuccess runPass(MO::GraphOp graph, MOGG::MojoRegisterOp registry) {
    // Load the mojo library. We use this KGEN module as the starting point to
    // build up the stub functions. It contains all the kernels and functions we
    // will need to call.
    UNWRAP_ERROR(kgenModule, loadMojoLibrary());

    // Helper to track all the types to prevent constant reallocation.
    Stub::TypeManager manager{graph.getContext()};

    // Map operations with the same output sizes onto the same kernel. Longer
    // term we may want fancier optimizations like unifying aliased inputs but
    // this is legal for now.
    std::unordered_map<std::string, MOGG::ExecuteCompiledOp> jittedKernels;

    // Track all the attributes we have added.
    SmallVector<mlir::Attribute> symbolNamesAdded;

    // Start building up the KGEN kernel.
    Stub::KGENBuilder builder{manager, *kgenModule, registry};

    // Build stubs for all JIT operations and replace them with symbolic calls
    // to the kernels in the KGEN stub.
    for (MOGG::KernelOp op :
         llvm::make_early_inc_range(graph.getOps<MOGG::KernelOp>())) {
      // Create a unique name for the kernel.
      std::string stubKernelName = createMangledName(op);

      // Don't duplicate KGEN kernels for the same call.
      auto itr = jittedKernels.find(stubKernelName);
      if (itr != jittedKernels.end()) {
        MOGG::ExecuteCompiledOp cachedExeOp = itr->second;

        // We cannot just clone the cached operation because its result type may
        // be different from the currently processed kernel's result type, if
        // they are parameterized.
        mlir::IRRewriter rewriter{OpBuilder(op)};
        rewriter.replaceOpWithNewOp<MOGG::ExecuteCompiledOp>(
            op, op->getResultTypes(), op->getOperands(),
            cachedExeOp->getAttrDictionary().getValue());

        continue;
      }

      // Lower the basic stub function.
      RETURN_ERROR(builder.startLowering(op, stubKernelName));

      // Lower the operation using a vectorized map over an elementwise lambda.
      RETURN_ERROR(builder.lowerToElementwiseLambda(op));

      // Replace the original kernel with a call to the compiled jit code.
      MOGG::ExecuteCompiledOp jitExecute =
          builder.createExecuteOperation(stubKernelName, op.getShapeFunc());
      op->replaceAllUsesWith(jitExecute.getResults());
      jittedKernels.insert({stubKernelName, jitExecute});
      op->erase();

      // Tracking attributes makes it easier to create an array.
      symbolNamesAdded.push_back(jitExecute.getKernelNameAttr());
    }

    if (dumpStub.getValue())
      builder.dumpStub();

    if (!symbolNamesAdded.empty()) {
      // Convert our list into a combined attribute.
      auto allSyms =
          mlir::ArrayAttr::get(graph->getContext(), symbolNamesAdded);

      // Add the compiled binary to the file.
      Stub::StubCompiler compiler{*kgenModule, graph, useSearch.getValue(),
                                  saveTempPrefix.getValue()};
      RETURN_ERROR(compiler.compile(allSyms));
    }
    return success();
  }
};

} // namespace

} // namespace M::MOGG
