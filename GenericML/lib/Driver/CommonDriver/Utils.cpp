//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/Utils.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "AsyncRT/Runtime/Utils.h"
#include "GenericML/Driver/DriverCLOptions.h"
#include "GraphRT/MEFExecutor/MEFFile.h"
#include "Support/ADT/GenericUniquePtrSet.h"
#include "Support/Buffer.h"
#include "Support/Configuration.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/ToolOutputFile.h"

int M::setupMLIRAndRun(ContextRef context, const ExecutorDriverOptions &options,
                       const char *testName,
                       std::unique_ptr<llvm::MemoryBuffer> &&file,
                       BodyFn &&bodyFn) {
  if (failed(options.configureMLIRContextAndExecute(
          std::move(file), [&](MLIRContext *ctx) {
            AsyncRT::Runtime *runtime = context->get<Runtime>();
            assert(runtime && "caller should have set up runtime for this");
            registerContext(*ctx, context);
            return AsyncRT::runWithLeakCheckedRuntime(
                testName, [&]() { return bodyFn(*runtime, ctx); });
          })))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

M::ErrorOr<std::string> M::getFileContentIdentity(const BufferRef &file,
                                                  FileFormat format) {
  switch (format) {
  default:
    return "input has unknown contents";
  case FileFormat::kMONNXDialect:
    return "input is an ONNX model in MLIR form";
  case FileFormat::kONNXProtobuf:
    return "input is an ONNX model in binary protobuf form";
  case FileFormat::kONNXTextProtobuf:
    return "input is an ONNX model in text protobuf form";
  case FileFormat::kTorchDialect:
    return "input is a PyTorch model in Torch MLIR dialect form";
  case FileFormat::kPTPickle:
    return "input is a PyTorch model in PT pickled form";
  case FileFormat::kTorchPDialect:
    return "input is a PyTorch model in MLIR form consisting of primitive ops";
  case FileFormat::kMEFFile: {
    ErrorOr<RCRef<MEFFile>> mefFileOr = MEFFile::open(file->getBuffer());
    if (failed(mefFileOr))
      return Error(mefFileOr.getError());

    auto mefContents = mefFileOr.get()->sniffFileContents();
    return std::string("input is a MEF file containing a ") +
           mefContents.getAsString();
  }
  }
}

void M::writeIntermediateFileIfRequested(const std::string &identifier,
                                         ModuleOp module,
                                         const CommonOptions *options,
                                         const BufferRef &file) {
  std::string ext = "." + identifier + ".mlir";
  std::string filePath;

  // Call the file "temp" if `file` is null or not backed by a file.
  auto filePathOr = file ? file->getFilePath() : std::nullopt;
  if (filePathOr)
    filePath = filePathOr->string();
  else
    filePath = "temp";

  auto writeFile = [&](const CommonOptions *opt) {
    std::unique_ptr<llvm::ToolOutputFile> intermediateFile =
        opt->getIntermediateFile(filePath, ext);
    if (opt->printTempsLocalScope)
      module->print(intermediateFile->os(),
                    mlir::OpPrintingFlags().useLocalScope());
    else
      module->print(intermediateFile->os());
    intermediateFile->keep();
  };

  // If there are CL options use those.
  if (options && options->saveTemps) {
    writeFile(options);
    return;
  }

#ifndef MODULAR_PRODUCTION
  // Else check in config and use the value there.
  auto configOr = Config::open();
  if (!configOr.isError()) {
    StringRef tempsDir = configOr->getValue("max.temps_dir");
    bool writeFileLocally =
        configOr->getValueAsBool("max.temps_use_local_scope", false);
    if (!tempsDir.empty()) {
      CommonOptions tempOptions;
      tempOptions.saveTemps = true;
      tempOptions.printTempsLocalScope = writeFileLocally;
      tempOptions.tempsDir = tempsDir.str();
      writeFile(&tempOptions);
    }
  }
#endif // MODULAR_PRODUCTION
}
