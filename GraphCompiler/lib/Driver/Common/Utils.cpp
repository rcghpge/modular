//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Driver/Common/Utils.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"
#include "Support/ADT/GenericUniquePtrSet.h"
#include "Support/Buffer.h"
#include "Support/CommonCLOptions.h"
#include "Support/Configuration.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/ToolOutputFile.h"

M::ErrorOr<std::string> M::getFileContentIdentity(const BufferRef &file,
                                                  FileFormat format) {
  switch (format) {
  default:
    return "input has unknown contents";
  case FileFormat::kMODialect:
    [[fallthrough]];
  case FileFormat::kMGPDialect:
    return "input is a Modular model";
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

std::string M::getMaxIRDumpDir() {
  auto configOr = Config::open();
  if (configOr.isError())
    return {};

  // Prefer the user-facing debug option; fall back to the legacy save-temps
  // key (which is also where `MODULAR_MAX_TEMPS_DIR` lands).
  StringRef dir = configOr->getValue("max-debug.ir-output-dir");
  if (dir.empty())
    dir = configOr->getValue("max.temps_dir");
  return dir.str();
}

void M::writeIntermediateFileIfRequested(const std::string &identifier,
                                         ModuleOp module,
                                         const CommonOptions *options,
                                         const BufferRef &file,
                                         llvm::StringRef baseName) {
  std::string ext = "." + identifier + ".mlir";
  std::string filePath;

  // Call the file "temp" if `file` is null or not backed by a file.
  auto filePathOr = file ? file->getFilePath() : std::nullopt;
  if (filePathOr)
    filePath = filePathOr->string();
  else
    filePath = baseName.str();

  auto writeFile = [&](const CommonOptions *opt) {
    std::unique_ptr<llvm::ToolOutputFile> intermediateFile =
        opt->getIntermediateFile(filePath, ext);
    mlir::OpPrintingFlags flags;
    if (opt->printTempsLocalScope)
      flags.useLocalScope();
    // Print each op's source location (the printer omits loc attributes
    // by default), so dump consumers can trace ops back to the code that
    // built them.
    flags.enableDebugInfo(true);
    module->print(intermediateFile->os(), flags);
    intermediateFile->keep();
  };

  // If there are CL options use those.
  if (options && options->saveTemps) {
    writeFile(options);
    return;
  }

  // Else fall back to the unified Config-driven directory (which combines
  // `max-debug.ir-output-dir` and the legacy `max.temps_dir` keys).
  std::string tempsDir = getMaxIRDumpDir();
  if (tempsDir.empty())
    return;

  bool writeFileLocally = false;
  if (auto configOr = Config::open(); !configOr.isError()) {
    writeFileLocally =
        configOr->getValueAsBool("max.temps_use_local_scope", false);
  }
  CommonOptions tempOptions;
  tempOptions.saveTemps = true;
  tempOptions.printTempsLocalScope = writeFileLocally;
  tempOptions.tempsDir = std::move(tempsDir);
  writeFile(&tempOptions);
}
