//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_UTILS_H
#define GENERICML_DRIVER_UTILS_H

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

namespace M {

/// This is the type we expect to have when
/// `setupMLIRAndRun` is called. The runtime reference
/// and MLIRContext pointer are the only two things that the body function
/// could not capture outside this call.
using BodyFn = std::function<LogicalResult(Runtime &, MLIRContext *)>;

/// Run the given body function with runtime from given context.
int setupMLIRAndRun(M::ContextRef context, const ExecutorDriverOptions &options,
                    const char *testName,
                    std::unique_ptr<llvm::MemoryBuffer> &&file,
                    BodyFn &&bodyFn) {
  if (failed(options.configureMLIRContextAndExecute(
          std::move(file), [&](MLIRContext *ctx) {
            LLCL::Runtime *runtime = context->get<Runtime>();
            assert(runtime && "caller should have setup runtime for this");
            registerContext(*ctx, context);
            return LLCL::runWithLeakCheckedRuntime(
                testName, [&]() { return bodyFn(*runtime, ctx); });
          })))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

// FIXME: We have to duplicate this here because of #1639. Remove this when
//        possible to do so.
ErrorOr<std::string> getFileContentIdentity(const M::BufferRef &file,
                                            FileFormat format) {
  switch (format) {
  default:
    return "input has unknown contents";
  case FileFormat::kTFDialect:
    return "input is a TensorFlow model in MLIR form";
  case FileFormat::kTFSavedModel:
    return "input is a TensorFlow model in SavedModel form";
  case FileFormat::kTFPDialect:
    return "input is a compiled TensorFlow MEF model in MLIR form";
  case FileFormat::kFauxDialect:
    return "input is Faux model in MLIR form";
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
  case FileFormat::kSafeTensors:
    return "input is a SafeTensors weights archive";
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

/// Saves an intermediate file with a dump of `module` if --saved-temps is set.
inline void writeIntermediateFileIfRequested(const std::string &identifier,
                                             ModuleOp module,
                                             const CommonOptions *options,
                                             const M::BufferRef &file) {
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
  auto configOr = M::Config::open();
  if (!configOr.isError()) {
    StringRef tempsDir = configOr->getValue("max.temps_dir");
    if (!tempsDir.empty()) {
      CommonOptions tempOptions;
      tempOptions.saveTemps = true;
      tempOptions.tempsDir = tempsDir.str();
      writeFile(&tempOptions);
    }
  }
#endif // MODULAR_PRODUCTION
}

} // namespace M

#endif // GENERICML_DRIVER_UTILS_H
