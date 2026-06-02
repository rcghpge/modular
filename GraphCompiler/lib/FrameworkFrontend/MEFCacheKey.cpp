//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MEFCacheKey.h"

#include "Cache/CachedTransform.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/ParametersUtils.h"
#include "GraphCompiler/GraphCompiler/Passes.h"
#include "GraphCompiler/MOFOptions.h"
#include "KGEN/KGENDialect/KGENCompilationContext.h"
#include "Support/Buffer.h"
#include "Support/Configuration.h"
#include "Support/FileSystemExtras.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace M {

namespace {

class BLAKE3Stream : public llvm::raw_ostream {
public:
  explicit BLAKE3Stream(llvm::BLAKE3 &hasher) : hasher(hasher) {
    SetUnbuffered();
  }

private:
  void write_impl(const char *ptr, size_t size) override {
    hasher.update(llvm::StringRef(ptr, size));
    pos += size;
  }
  uint64_t current_pos() const override { return pos; }

  llvm::BLAKE3 &hasher;
  uint64_t pos = 0;
};

void feedMOToMGPPipelineKey(llvm::BLAKE3 &hasher, mlir::MLIRContext *ctx,
                            const MO::GCPipelineOptions &options) {
  auto gcPipelinePasses = mlir::PassManager::on<mlir::ModuleOp>(
      ctx, mlir::OpPassManager::Nesting::Implicit);
  MO::buildGCPipeline(gcPipelinePasses, options);
  BLAKE3Stream os(hasher);
  gcPipelinePasses.printAsTextualPipeline(os);
}

ErrorOrSuccess feedFile(llvm::BLAKE3 &hasher, llvm::StringRef libraryPath,
                        llvm::StringRef errorPrefix) {
  auto bufferOr = M::openInputFile(libraryPath);
  if (bufferOr.isError())
    return Error(Twine(errorPrefix) + libraryPath);
  llvm::StringRef bytes = (*bufferOr)->getBuffer();
  hasher.update(bytes);
  return success();
}

std::string finalizeHasher(const llvm::BLAKE3 &hasher) {
  // `llvm_blake3_hasher_finalize` does not mutate the hasher.
  auto digest = const_cast<llvm::BLAKE3 &>(hasher).final();
  return std::string(reinterpret_cast<const char *>(digest.data()),
                     digest.size());
}

} // namespace

ErrorOr<MEFCacheKeyHasher> MEFCacheKeyHasher::build(
    mlir::MLIRContext &mlirCtx, MOF::MOFLoweringOptions &mofLoweringOptions,
    const MO::GCPipelineOptions &gcPipelineOptions, ContextRef maxContext) {
  MEFCacheKeyHasher hasher;
  llvm::BLAKE3 &b3 = hasher.prefixHasher;
  BLAKE3Stream os(b3);

  // Lowering options + pipeline options + textual pipeline
  mofLoweringOptions.print(os);
  gcPipelineOptions.print(os);
  feedMOToMGPPipelineKey(b3, &mlirCtx, gcPipelineOptions);

  // KGEN compilation context (Mojo compile-time defines)
  if (auto *compilationCtx = maxContext->get<KGEN::CompilationContext>())
    compilationCtx->print(os);

  // Graph compiler version, pipeline, config
  if (auto err = M::MOGG::graphCompilerCacheKey(os, &mlirCtx))
    return err.takeError();

  // User-provided kernel library files (ref MOCO-1570)
  for (const std::string &libraryPath : gcPipelineOptions.kernelLibraryPaths) {
    if (auto err = feedFile(
            b3, libraryPath,
            "failed to open file specified as additional kernel library: "))
      return err.takeError();
  }

  // Built-in kernel packages
  auto builtInPathsOr = MOGG::getBuiltInKernelPackagePaths(&mlirCtx);
  if (builtInPathsOr.isError())
    return builtInPathsOr.takeError();
  for (const std::string &libraryPath : *builtInPathsOr) {
    if (auto err = feedFile(b3, libraryPath,
                            "failed to open built-in kernel package: "))
      return err.takeError();
  }

  // Selected debug-config flags (GEX-2678).
  // Interpret config read failure as no config.
  auto configOr = M::Config::open();
  M::Config config = configOr.isError() ? M::Config() : configOr.takeValue();
  b3.update(
      llvm::formatv("max-debug.source-tracebacks={}",
                    config.getValueAsBool("max-debug.source-tracebacks", false))
          .str());
  b3.update(llvm::formatv("max-debug.nan-check={}",
                          config.getValueAsBool("max-debug.nan-check", false))
                .str());
  if (llvm::StringRef stride = config.getValue("max-debug.nan-check-stride");
      !stride.empty())
    b3.update(llvm::formatv("max-debug.nan-check-stride={}", stride).str());

  return hasher;
}

ErrorOr<std::string>
MEFCacheKeyHasher::keyForModule(mlir::ModuleOp module) const {
  // Strip debug info and normalize symbols before hashing the IR
  auto stripDebugInfo = mlir::PassManager::on<mlir::ModuleOp>(
      module.getContext(), mlir::OpPassManager::Nesting::Implicit);
  stripDebugInfo.addPass(mlir::createStripDebugInfoPass());
  mlir::OwningOpRef<mlir::ModuleOp> moduleOpNoDebug = module.clone();
  if (failed(stripDebugInfo.run(*moduleOpNoDebug)))
    return Error::getStaticString(
        "failed to strip location information from the module op");
  MOGG::normalizeAllNames(*moduleOpNoDebug);

  WriteableBufferRef moduleBuf = WriteableBuffer::get();
  if (failed(Cache::writeOperationToCacheKey(*moduleOpNoDebug, moduleBuf)))
    return Error::getStaticString("failed to write module op to cache key");

  llvm::BLAKE3 finalizer = prefixHasher;
  finalizer.update(
      llvm::StringRef(moduleBuf->getBufferStart(), moduleBuf->getBufferSize()));
  return finalizeHasher(finalizer);
}

ErrorOr<std::string> computeMEFCacheKey(
    mlir::ModuleOp module, MOF::MOFLoweringOptions &mofLoweringOptions,
    const MO::GCPipelineOptions &gcPipelineOptions, ContextRef maxContext) {
  auto hasherOr = MEFCacheKeyHasher::build(
      *module.getContext(), mofLoweringOptions, gcPipelineOptions, maxContext);
  if (hasherOr.isError())
    return hasherOr.takeError();
  return hasherOr.takeValue().keyForModule(module);
}

} // namespace M
