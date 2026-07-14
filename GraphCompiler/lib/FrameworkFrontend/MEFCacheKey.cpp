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

// Clears the absolute-path-valued options on `options` in place, so the cache
// key derived from it does not depend on the install location (MOCO-4152).
// These paths leak into the key via `print` (which emits every option) and the
// textual pipeline string, but they are not relevant to the compiled result
// *as paths*:
//   - kernel-library-paths: their file *contents* are hashed separately (see
//     `feedFile` in `build`), so the path strings are redundant for
//     correctness.
//   - save-temp-prefix / mlir-timing-out-dir: pure diagnostic output locations
//     (debug temp-saving and pass-timing JSON); no effect on the artifact.
//   - cache-dir: set-vs-unset DOES change the artifact (it toggles CAS
//     dense-attribute outlining), but the path value only selects where the CAS
//     store lives. So we strip the path here and instead fold a boolean
//     "CAS enabled" into the key in `build`.
void stripInstallPaths(MO::GCPipelineOptions &options) {
  options.cacheDir = "";
  options.kernelLibraryPaths = {};
  options.saveTempPrefix = "";
  options.mlirTimingOutDir = "";
}

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

ErrorOr<MEFCacheKeyHasher>
MEFCacheKeyHasher::build(mlir::MLIRContext &mlirCtx,
                         const MO::GCPipelineOptions &gcPipelineOptions,
                         ContextRef maxContext) {
  MEFCacheKeyHasher hasher;
  llvm::BLAKE3 &b3 = hasher.prefixHasher;
  BLAKE3Stream os(b3);

  // Pipeline options + textual pipeline. `stripInstallPaths` removes the
  // path options so the key is relocatable (MOCO-4152); since it also strips
  // cache-dir, fold in a "CAS enabled" bit separately (set-vs-unset changes
  // the artifact even though the path does not). `GCPipelineOptions` is
  // non-copyable, hence the local + `copyOptionValuesFrom`.
  MO::GCPipelineOptions keyOptions;
  keyOptions.copyOptionValuesFrom(gcPipelineOptions);
  stripInstallPaths(keyOptions);
  keyOptions.print(os);
  os << "cas-enabled="
     << (gcPipelineOptions.cacheDir.hasValue() &&
         !gcPipelineOptions.cacheDir.getValue().empty());
  feedMOToMGPPipelineKey(b3, &mlirCtx, keyOptions);

  // KGEN compilation context (Mojo compile-time defines)
  if (auto *compilationCtx = maxContext->get<KGEN::CompilationContext>())
    compilationCtx->print(os);

  // Graph compiler version, pipeline, config. The host target is resolved
  // here, outside the key computation itself.
  auto hostTargetOr = MOGG::getHostTargetInfoAttr(&mlirCtx);
  if (hostTargetOr.isError())
    return hostTargetOr.takeError();
  if (auto err = M::MOGG::graphCompilerCacheKey(os, hostTargetOr.takeValue()))
    return err.takeError();

  // User-provided kernel library files (ref MOCO-1570)
  for (const std::string &libraryPath : gcPipelineOptions.kernelLibraryPaths) {
    if (auto err = feedFile(
            b3, libraryPath,
            "failed to open file specified as additional kernel library: "))
      return err.takeError();
  }

  // Built-in kernel packages plus every binary package on the module import
  // path (see `getKernelPackagePathsForCacheKey` for why both must reach the
  // key). Only contents are hashed, not paths, so the key stays relocatable
  // (MOCO-4152).
  auto kernelPackagePathsOr = MOGG::getKernelPackagePathsForCacheKey(&mlirCtx);
  if (kernelPackagePathsOr.isError())
    return kernelPackagePathsOr.takeError();
  for (const std::string &libraryPath : *kernelPackagePathsOr) {
    if (auto err = feedFile(b3, libraryPath, "failed to open kernel package: "))
      return err.takeError();
  }

  // Interpret config read failure as no config.
  auto configOr = M::Config::open();
  M::Config config = configOr.isError() ? M::Config() : configOr.takeValue();
  b3.update(
      llvm::formatv("max-debug.source-tracebacks={}",
                    config.getValueAsBool("max-debug.source-tracebacks", false))
          .str());

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

ErrorOr<std::string>
computeMEFCacheKey(mlir::ModuleOp module,
                   const MO::GCPipelineOptions &gcPipelineOptions,
                   ContextRef maxContext) {
  auto hasherOr = MEFCacheKeyHasher::build(*module.getContext(),
                                           gcPipelineOptions, maxContext);
  if (hasherOr.isError())
    return hasherOr.takeError();
  return hasherOr.takeValue().keyForModule(module);
}

} // namespace M
