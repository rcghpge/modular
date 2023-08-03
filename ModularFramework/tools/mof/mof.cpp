//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Config/Version.h"
#include "GenericML/Driver/CompileDriver.h"
#include "GenericML/Driver/DriverAPI.h"
#include "GenericML/Driver/DriverCLOptions.h"
#include "GenericML/Driver/DriverInputFormat.h"
#include "GenericML/Driver/Utils.h"
#include "GenericML/GraphCompiler/GraphCompilerCLOptions.h"
#include "GenericML/Support/CASConfig.h"
#include "GenericML/Support/ModelCollection.h"
#include "GraphRT/MEFExecutor/MEFFile.h"
#include "MOF/Driver/DriverConfig.h"
#include "MOF/Driver/DriverImpl.h"
#include "MOF/ModularModel/ModularModelFile.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ToolOutputFile.h"
#include <string>
#include <utility>

using namespace M;
using namespace M::LLCL;
using namespace M::MOF;
using llvm::MemoryBuffer;

namespace {

class MOFDriverCLOptions : public DriverCLOptions,
                           public MO::GraphCompilerCLOptions {
public:
  using DriverCLOptions::DriverCLOptions;

  cl::opt<bool> enableJitKernels{"enable-jit-kernels",
                                 cl::desc("Enable jitting of MOGG kernels."),
                                 cl::init(true)};
};

class MOFContext {
public:
  MOFContext(int &argc, char **argv)
      : clOptions(std::make_unique<MOFDriverCLOptions>(argc, argv,
                                                       /*skipInitLLVM=*/true)),
        commonCasConfig{} {
    setConfigFromOptions();
  }

  MOFContext() : clOptions(nullptr) {}

  bool hasOptions() { return clOptions != nullptr; }

  MOFDriverCLOptions *getOptions() { return clOptions.get(); }

  ErrorOrSuccess setOptions(ArrayRef<std::string> opts) {
    if (!hasOptions())
      return Error("CL Options parser is not initialized");

    // Reset all option occurrences every time this is called.
    llvm::cl::ResetAllOptionOccurrences();
    // Parse the options one at a time, so we don't have to combine the list of
    // strings.
    for (StringRef opt : opts) {
      const char *optChars[] = {clOptions->getProgramName().data(), opt.data()};
      if (!llvm::cl::ParseCommandLineOptions(std::size(optChars), optChars))
        return Error("unable to parse argument: " + opt);
    }

    setConfigFromOptions();

    return success();
  }

  void setCasConfig(const CASConfig &casConfig) { commonCasConfig = casConfig; }
  CASConfig getCasConfig() { return commonCasConfig; }

  ErrorOrSuccess setConfigs(ArrayRef<std::unique_ptr<DriverConfig>> configs) {
    return success();
  }

  ~MOFContext() {
    if (hasOptions())
      llvm::cl::ResetCommandLineParser();
  }

private:
  void setConfigFromOptions() {}
  std::unique_ptr<MOFDriverCLOptions> clOptions;
  CASConfig commonCasConfig;
};

} // namespace

//===----------------------------------------------------------------------===//
// getFileContentIdentity
//===----------------------------------------------------------------------===//

// TODO(#1639): Remove. No really do, it's gross.
static ErrorOr<std::string>
getFileContentIdentity(const llvm::MemoryBufferRef &file, FileFormat format) {
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
  case FileFormat::kMODialect:
    [[fallthrough]];
  case FileFormat::kMGPDialect:
    return "input is a Modular model";
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

void M_getVersionString(std::string *rev) { *rev = getModularVersionString(); }

M::ErrorOrSuccess M_initFrameworkContext(M::FrameworkContextHandle *ctx,
                                         int &argc, char **&argv) {
  ctx->ctx = new MOFContext(argc, argv);
  return success();
}

M::ErrorOrSuccess
M_initFrameworkContextWithConfig(M::FrameworkContextHandle *ctx) {
  ctx->ctx = new MOFContext();
  return success();
}

void M_freeFrameworkContext(M::FrameworkContextHandle ctx) {
  delete ctx.unwrap<MOFContext>();
}

ErrorOrSuccess M_setOptions(FrameworkContextHandle ctx,
                            ArrayRef<std::string> opts) {
  if (opts.empty())
    return success();

  return ctx.unwrap<MOFContext>()->setOptions(opts);
}

ErrorOrSuccess M_setConfigs(FrameworkContextHandle ctx,
                            ArrayRef<std::unique_ptr<DriverConfig>> configs,
                            const CASConfig &casConfig) {
  ctx.unwrap<MOFContext>()->setCasConfig(casConfig);
  return ctx.unwrap<MOFContext>()->setConfigs(configs);
}

bool M_isBinary(FrameworkContextHandle ctx, const llvm::MemoryBufferRef &file,
                FileFormat &format) {
  return false;
}

ErrorOrSuccess M_compileToBinary(FrameworkContextHandle ctxHandle,
                                 const llvm::MemoryBufferRef &file,
                                 raw_ostream &os, FileFormat &format) {
  // CAUTION: Options will be null for C API calls.
  auto *options = ctxHandle.unwrap<MOFContext>()->getOptions();
  auto casConfig = ctxHandle.unwrap<MOFContext>()->getCasConfig();
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module;

  // Convert the input file to an MLIR module.
  switch (format) {
  case FileFormat::kMODialect:
    [[fallthrough]];
  case FileFormat::kMGPDialect:
    module = parseMODialect(file.getBuffer(), &ctx);
    break;

  default: {
    UNWRAP_ERROR(identity, getFileContentIdentity(file, format));
    return Error("cannot compile input with format " + identity);
  }
  }

  // We have an MLIR model, try to lower it to a MEF-capable encoding.
  switch (format) {
  case FileFormat::kMODialect: {

    auto writeIntermediateFileIfRequested = [&](const std::string &identifier) {
      std::string ext = "." + identifier + ".mlir";
      if (options && options->saveTemps) {
        std::unique_ptr<llvm::ToolOutputFile> intermediateFile =
            options->getIntermediateFile(file.getBufferIdentifier(), ext);
        module->print(intermediateFile->os());
        intermediateFile->keep();
      }
    };

    writeIntermediateFileIfRequested("mo");

    // The MO graph dialect can be lowered to TFP, which is MEF emittable.
    // TODO: Set the options flags.
    if (failed(lowerMOToMGP(module.get())))
      return Error("unable to lower MO to MGP");

    writeIntermediateFileIfRequested("mgp");

    format = FileFormat::kMGPDialect;
    break;
  }
  case FileFormat::kMGPDialect: {
    break;
  }
  default:
    UNWRAP_ERROR(identity, getFileContentIdentity(file, format));
    return Error("cannot lower to MEF data of format " + identity);
  }

  bool enableMEFMetadata = options ? options->enableMEFMetadata : true;
  if (failed(convertToMEF(module.get(), os, enableMEFMetadata)))
    return Error("unable to convert file to a MEF binary");
  return success();
}

ErrorOrSuccess M_getMetadataFromBinary(FrameworkContextHandle ctx,
                                       DriverHandle modelBinaryHandle,
                                       llvm::ToolOutputFile &outputFile,
                                       int command) {

  return Error("Not implemented");
}

ErrorOrSuccess M_openBinaryFile(FrameworkContextHandle ctx,
                                const llvm::MemoryBuffer &file,
                                DriverHandle *modelBinaryHandle) {
  UNWRAP_ERROR(mefFile, MEFFile::open(file.getBuffer()));
  *modelBinaryHandle = DriverHandle{std::move(mefFile).release()};
  return success();
}

void M_closeBinaryFile(DriverHandle modelBinaryHandle) {
  // We're going to use RCRef to destroy the MEFFile handle for simplicity's
  // sake. We take the RCRef and then immediately destroy it, which should
  // essentially just decrement the refcount to 0 and destroy the MEFFile.
  (void)takeRCRef(modelBinaryHandle.unwrap<MEFFile>());
}

ErrorOrSuccess M_checkExpectedContents(FrameworkContextHandle ctx,
                                       DriverHandle modelBinaryHandle) {
  auto expectedFramework =
      CompiledFrameworkLabel{CompiledFrameworkLabel::kModularModel};
  auto actualFramework =
      modelBinaryHandle.unwrap<MEFFile>()->sniffFileContents();
  if (actualFramework != expectedFramework)
    return Error(Twine("expected ") + expectedFramework.getAsString() +
                 ", found " + actualFramework.getAsString() + " instead");
  return success();
}

void M_sniffBinaryFileContents(FrameworkContextHandle ctx,
                               DriverHandle modelBinaryHandle,
                               std::string &str) {
  auto mefContents = modelBinaryHandle.unwrap<MEFFile>()->sniffFileContents();
  str = std::string("input is a MEF file containing a ") +
        mefContents.getAsString();
}

ErrorOrSuccess M_loadModels(FrameworkContextHandle ctx,
                            llvm::MemoryBufferRef fileBuffer,
                            DriverHandle handle,
                            std::unique_ptr<ModelCollection> &modelsPtr) {
  UNWRAP_ERROR_OR_SET(modelsPtr,
                      loadMGPMEFFile(copyRCRef(handle.unwrap<MEFFile>())));
  return success();
}

int M_runBinaryFile(FrameworkContextHandle ctx,
                    std::unique_ptr<llvm::MemoryBuffer> &&fileBuffer,
                    DriverHandle modelBinaryHandle) {
  auto *opts = ctx.unwrap<MOFContext>()->getOptions();
  std::unique_ptr<ModelCollection> models;
  if (auto err = M_loadModels(ctx, fileBuffer->getMemBufferRef(),
                              modelBinaryHandle, models)) {
    return opts->reportError("unable to load models: " + Twine(err.getError()));
  }

  auto runModels = [&](Runtime &runtime, MLIRContext *ctx) -> LogicalResult {
    for (const auto &model : (*models)) {
      if (failed(runModel(ctx, model.get(), runtime, *opts))) {
        if (opts->stopOnFirstError())
          return failure();
      }
    }
    return success();
  };

  // Run it.
  return setupMLIRAndRunWithLeakCheckedRuntime(
      *opts, "Modular model", std::move(fileBuffer), runModels);
}

int M_runModelWithData(FrameworkContextHandle ctx,
                       std::unique_ptr<llvm::MemoryBuffer> &&fileBuffer,
                       const M::Model *model, DriverHandle inputsHandle,
                       DriverHandle outputsHandle) {
  auto *opts = ctx.unwrap<MOFContext>()->getOptions();
  auto inputs = inputsHandle.unwrap<llvm::SmallVector<const void *>>();
  auto outputs = outputsHandle.unwrap<llvm::SmallVector<void *>>();

  auto runModels = [&](Runtime &runtime, MLIRContext *ctx) -> LogicalResult {
    if (runModelWithData(ctx, model, runtime, *opts, *inputs, *outputs)
            .isError()) {
      if (opts->stopOnFirstError())
        return failure();
    }

    return success();
  };

  // Run it.
  return setupMLIRAndRunWithLeakCheckedRuntime(
      *opts, "Modular model", std::move(fileBuffer), runModels);
}

ErrorOrSuccess M_openPath(M::FrameworkContextHandle ctx,
                          const std::filesystem::path &path,
                          std::unique_ptr<llvm::MemoryBuffer> &buffer) {
  llvm_unreachable("Not implemented");
}
