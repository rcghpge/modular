//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGInterfaces.h" // IWYU: needed for type declarations used via MOGG::
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/LowerABIHelper.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_EMITMOJO
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

namespace M::MGP {

namespace {

StringAttr lowerGenericCompiledRegionOp(MOGG::GenericCompiledRegionOp kernel,
                                        MOGG::EmitMojoState &state,
                                        OpBuilder &builder) {
  static const MOGG::MojoValue vUnsafePointer =
      MOGG::HardcodedMojoSymbols::struct_UnsafePointer;
  static const MOGG::MojoValue vOpaquePointer =
      MOGG::HardcodedMojoSymbols::alias_OpaquePointer;

  static constexpr StringLiteral baseKernelName = "kernel_wrapper";
  // Just a static local var should be fine for now, but if we ever parallelize
  // this function we should change this.
  static int fooIdx = 0;

  auto skernelName = llvm::formatv("{}_{}", baseKernelName, fooIdx++).str();
  StringRef kernelName = skernelName;

  // Clear any previous kernel's return values from the state
  state.mojoReturnValues.clear();

  bool errOccurred = false;
  state.builder.fn(
      kernelName, {},
      {MOGG::NamedMojoDef("arg", vUnsafePointer[vOpaquePointer]),
       MOGG::NamedMojoDef("arg", vOpaquePointer)},
      std::nullopt, true, false, [&](const MOGG::FunctionBody &functionBody) {
        state.builder.comment(
            "This is the entry point of the kernel. The first argument is a "
            "pointer to a list of inputs and outputs. The second is a pointer "
            "to a runtime context.");
        auto runtimeArgument = functionBody.arguments.front();
        auto &kernelBody = kernel.getBody().front();
        // The order of arguments in runtimeArgument:
        // 1. Inputs and DPS outputs
        // 2. Return values (chain and non DPS outputs)
        for (auto argument : kernelBody.getArguments()) {
          state.setMojoValueFor(
              argument,
              runtimeArgument[std::to_string(argument.getArgNumber())]);
        }
        // Return values, although in DPS form from the runtime's perspective,
        // are still in value semantics inside of the kernel. Therefore, there
        // is no value to associate them with. Instead, we have a mapping that
        // mogg.return and mogg.async.pack will use.
        for (uint64_t i = 0; i < kernel->getNumResults(); ++i) {
          state.mojoReturnValues.emplace_back(
              runtimeArgument[std::to_string(i + kernel->getNumOperands())]);
        }

        state.builder
            .try_([&]() {
              errOccurred |=
                  llvm::any_of(kernel.getOps(), [&](mlir::Operation &op) {
                    return state.lowerOpToMojo(&op).failed();
                  });
              state.genAllKeepAliveStatements();
            })
            .except([&](const MOGG::MojoValue &e) {
              /// TODO: Set an error on the context.
              state.builder.pass();
            });
      });

  if (errOccurred)
    return nullptr;

  // Create a new ExecuteCompiledOp to replace the original kernel
  auto insertPoint = builder.saveInsertionPoint();
  builder.setInsertionPoint(kernel);
  auto newOp = builder.create<MOGG::ExecuteCompiledOp>(
      kernel->getLoc(),
      /*resultTypes=*/kernel->getResultTypes(),
      /*operands=*/kernel->getOperands(),
      /*kernelName=*/StringAttr::get(kernel->getContext(), kernelName),
      /*usedParams=*/builder.getArrayAttr({}),
      /*outputParamDecls=*/kernel.getOutputParamDecls(),
      /*summary=*/MOGG::getSummaryString(kernel),
      /*isAsync=*/*kernel.getIsAsync(),
      /*constantResources=*/kernel.getConstantResources());
  builder.restoreInsertionPoint(insertPoint);

  newOp.setDeviceRef(kernel.getExecutionDevice());
  kernel->replaceAllUsesWith(newOp.getResults());
  kernel->erase();

  return newOp.getKernelNameAttr();
}

LogicalResult lowerKernelStubOp(MOGG::KernelStubOp kernelStub,
                                MOGG::EmitMojoState &state) {
  SmallVector<MOGG::NamedMojoDef> inputTypes;
  for (auto arg : kernelStub.getBody().getArguments())
    inputTypes.push_back(
        MOGG::NamedMojoDef("arg", state.mlirTypeToMojoType(arg.getType())));

  bool errOccurred = false;
  state.builder.fn(
      kernelStub.getNameAttr().str(), {}, inputTypes, std::nullopt, false, true,
      [&](const MOGG::FunctionBody &functionBody) {
        for (auto &&[mlirArg, mojoArg] : llvm::zip(
                 kernelStub.getBody().getArguments(), functionBody.arguments)) {
          state.setMojoValueFor(mlirArg, mojoArg);
        }

        errOccurred |=
            llvm::any_of(kernelStub.getOps(), [&](mlir::Operation &op) {
              return state.lowerOpToMojo(&op).failed();
            });
        state.genAllKeepAliveStatements();
      });

  if (errOccurred)
    return failure();
  kernelStub.erase();

  return success();
}

struct EmitMojo
    : public MOGG::MojoLibraryAnalysisPass<impl::EmitMojoBase<EmitMojo>> {
  using Base::Base;

  void runOnOperation() override {
    if (!MOGG::useExperimentalMojoPath())
      return;

    auto moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    auto analysisOr = getLibraryAnalysis();
    if (analysisOr.isError()) {
      emitError(getOperation().getLoc(),
                Twine("[MO_TO_MOGG] Failed to load MOJO analysis: ") +
                    analysisOr.getError());
      return;
    }

    OpBuilder builder(ctx);
    for (auto modelOp : moduleOp.getOps<MGP::ModelOp>()) {
      DenseSet<StringAttr> kernelsToLoad;
      MOGG::EmitMojoState state(*analysisOr.takeValue());
      for (auto package : MOGG::HardcodedMojoSymbols::packagesToImport)
        state.builder.import(package, "*");
      // Import mojo packages for the custom kernel libraries.
      for (const auto &kernelLibraryPath : kernelLibraryPaths) {
        auto moduleName = std::filesystem::path(kernelLibraryPath).stem();
        state.builder.import(moduleName.string(), "*");
      }

      // First do all the stubs (in case they appear after their references).
      auto lowerKernelStubOps = [&](MOGG::KernelStubOp kernelStub) {
        if (lowerKernelStubOp(kernelStub, state).failed())
          return WalkResult::interrupt();
        return WalkResult::advance();
      };
      if (modelOp->walk(lowerKernelStubOps).wasInterrupted()) {
        signalPassFailure();
        return;
      }

      // Then do all the kernels that may reference them.
      auto lowerGenericCompiledRegionOps =
          [&](MOGG::GenericCompiledRegionOp kernel) {
            // If none of the ops emits Mojo, skip it.
            if (kernel.getEmitsMojo()) {
              auto kernelName =
                  lowerGenericCompiledRegionOp(kernel, state, builder);
              if (!kernelName)
                return WalkResult::interrupt();
              kernelsToLoad.insert(kernelName);
            }
            return WalkResult::advance();
          };
      if (modelOp->walk(lowerGenericCompiledRegionOps).wasInterrupted()) {
        signalPassFailure();
        return;
      }

      // Store the emitted text into a resource and point to it from a
      // MojoModuleOp.
      auto mojoText = state.builder.str();
      auto attrType =
          ArrayType::get((int64_t)mojoText.size(),
                         IntegerType::get(ctx, 8, IntegerType::Unsigned));
      auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<char>(
          {mojoText.data(), mojoText.size()});
      auto resourceManager =
          DenseResourceElementsHandle::getManagerInterface(ctx);
      /// TODO: Find a better name than mojo, maybe a hash.
      auto mojoTextAttr = DenseResourceElementsAttr::get(
          attrType, resourceManager.insert("mojo", std::move(blob)));

      auto loc = moduleOp.getLoc();
      builder.setInsertionPointToStart(modelOp.getInitBlock());
      builder.create<MOGG::MojoModuleOp>(loc, mojoTextAttr);
      /// TODO: Add debug info support. For now, defaults to false.
      builder.create<MOGG::LoadCompiledKernelOp>(
          loc,
          builder.getArrayAttr(llvm::to_vector_of<Attribute>(kernelsToLoad)),
          mojoTextAttr, /*debug*/ false);

      if (dumpMojoText)
        llvm::errs() << mojoText << "\n";

      if (!saveTempPrefix.empty()) {
        std::string outPath = saveTempPrefix + ".kernels.mojo";
        auto outFile = mlir::openOutputFile(outPath);
        if (!outFile) {
          modelOp->emitError()
              << "Could not open pre-elaboration dump file named " << outPath;
          signalPassFailure();
          return;
        }

        outFile->os() << mojoText << "\n";
        outFile->keep();
      }
    }
  }
};

} // namespace

} // namespace M::MGP

std::unique_ptr<mlir::Pass> M::MGP::createEmitMojo(
    const EmitMojoOptions &opts,
    std::shared_ptr<MOGG::MojoLibraryAnalysis> libraryAnalysis) {
  return std::make_unique<EmitMojo>(opts, std::move(libraryAnalysis));
}
