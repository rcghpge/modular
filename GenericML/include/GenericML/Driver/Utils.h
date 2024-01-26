//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_UTILS_H
#define GENERICML_DRIVER_UTILS_H

#include "GenericML/Driver/DriverCLOptions.h"
#include "GraphRT/MEFExecutor/MEFFile.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/Utils.h"
#include "Support/ADT/GenericUniquePtrSet.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/ToolOutputFile.h"

namespace M {

/// This is the type we expect to have when
/// `setupMLIRAndRunWithLeakCheckedRuntime` is called. The runtime reference
/// and MLIRContext pointer are the only two things that the body function
/// could not capture outside this call.
using BodyFn = std::function<LogicalResult(Runtime &, MLIRContext *)>;

/// Run the given body function with a leak checked runtime.
int setupMLIRAndRunWithLeakCheckedRuntime(
    const ExecutorDriverCLOptions &options, const char *testName,
    std::unique_ptr<llvm::MemoryBuffer> &&file, BodyFn &&bodyFn) {
  if (failed(options.configureMLIRContextAndExecute(
          std::move(file), [&](MLIRContext *ctx) {
            return LLCL::runWithLeakCheckedRuntime(
                options, testName,
                [&](Runtime &runtime) { return bodyFn(runtime, ctx); });
          })))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

// FIXME: We have to duplicate this here because of #1639. Remove this when
//        possible to do so.
ErrorOr<std::string> getFileContentIdentity(const llvm::MemoryBufferRef &file,
                                            FileFormat format) {
  switch (format) {
  default:
    return "input has unknown contents";
  case FileFormat::kTFLFlatBuffer:
    return "input is a binary TFLite flatbuffer file";
  case FileFormat::kTFLFlatBufferJSON:
    return "input is a TFLite flatbuffer file in JSON format";
  case FileFormat::kMTFLDialect:
    return "input is TFLite model in MLIR form";
  case FileFormat::kTFLPDialect:
    return "input is a compiled TFLite MEF model in MLIR form";
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
  case FileFormat::kMEFFile: {
    ErrorOr<RCRef<MEFFile>> mefFileOr = MEFFile::open(file.getBuffer());
    if (failed(mefFileOr))
      return Error(mefFileOr.getError());

    auto mefContents = mefFileOr.get()->sniffFileContents();
    return std::string("input is a MEF file containing a ") +
           mefContents.getAsString();
  }
  }
}

/// Saves an intermediate file with a dump of `module` if --saved-temps is set.
inline void
writeIntermediateFileIfRequested(const std::string &identifier, ModuleOp module,
                                 const CommonCLOptions *options,
                                 const llvm::MemoryBufferRef &file) {
  std::string ext = "." + identifier + ".mlir";
  if (options && options->saveTemps) {
    std::unique_ptr<llvm::ToolOutputFile> intermediateFile =
        options->getIntermediateFile(file.getBufferIdentifier(), ext);
    module->print(intermediateFile->os());
    intermediateFile->keep();
  }
}

} // namespace M

#endif // GENERICML_DRIVER_UTILS_H
