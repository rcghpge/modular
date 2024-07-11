//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/CompileDriver.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "Config/Version.h"
#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "GraphRT/MLIRToMEF.h"
#include "Support/ADT/ConcatenationTree.h"
#include "Support/Configuration.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace M;

#define _STRINGIFY(str) #str
#define _X_STRINGIFY(str) _STRINGIFY(str)
#define STRINGIFY_MAX_CONFIG(path) _X_STRINGIFY(MAX_CONFIG_SECTION) path

std::optional<std::filesystem::path> M::getMaxCacheDir() {
  auto configOr = Config::open();

  // If config is accessible look if it's set or not. If it's set we are done.
  if (!configOr.isError()) {
    M::Config config = configOr.takeValue();
    auto maxCacheDir = config.getValue(STRINGIFY_MAX_CONFIG(".cache_dir"));
    if (!maxCacheDir.empty())
      return std::filesystem::path(maxCacheDir.str());
  }

  // Try using modular data directory.
  auto dataDirOr = Config::getModularDataFolderPath(/*create=*/true);
  if (dataDirOr.isError())
    return std::nullopt;

  return dataDirOr.takeValue() / ".max_cache";
}

bool M::isCompileProgressEnabled() {
  auto configOr = Config::open();

  if (configOr.isError())
    return false;

  auto compileProgressEnabled =
      configOr->getValueAsBool(STRINGIFY_MAX_CONFIG(".enable_compile_progress"),
                               /*defaultValue=*/true);
  return !compileProgressEnabled.isError() && *compileProgressEnabled;
}

bool M::isModelIRCacheEnabled() {
  auto configOr = Config::open();

  // We will ignore the errors and disable caching if config is not accessible.
  if (configOr.isError())
    return false;

  auto cacheEnabled =
      configOr->getValueAsBool(STRINGIFY_MAX_CONFIG(".enable_model_ir_cache"),
                               /*defaultValue=*/false);
  return !cacheEnabled.isError() && *cacheEnabled;
}

std::optional<std::filesystem::path> M::getWeightsCacheDir() {
  auto configOr = Config::open();
  if (configOr.isError())
    return std::nullopt;

  auto weightsCacheEnabledOr = configOr->getValueAsBool(
      /*key=*/STRINGIFY_MAX_CONFIG(".enable_weight_cache"),
      /*defaultValue=*/false);

  if (weightsCacheEnabledOr.isError() || !(*weightsCacheEnabledOr))
    return std::nullopt;

  auto baseDirOr = getMaxCacheDir();
  if (baseDirOr)
    return *baseDirOr / "weights";

  // For some reason if we can't find the base directory, we will let the client
  // of API define with it however they want.
  return std::nullopt;
}

ErrorOr<BufferRef>
M::cacheMEFTransform(StringRef frameworkName,
                     llvm::function_ref<ErrorOr<BufferRef>()> loweringFunction,
                     WriteableBufferRef key) {
  // We will ignore the errors and disable caching if config is not accessible.
  if (!isModelIRCacheEnabled())
    return loweringFunction();

  auto baseCacheDirOr = M::getMaxCacheDir();

  // If we can't find any suitable location to cache then also we will disable
  // caching.
  if (!baseCacheDirOr)
    return loweringFunction();

  const std::filesystem::path &baseCacheDir = *baseCacheDirOr;

  auto mefBackendOr = Cache::getLocalDefaultBackendChain(
      (baseCacheDir / frameworkName.str() / "mef").string(),
      std::string(M::getModularVersionString()));

  // We couldn't create a cache. Don't cache.
  if (mefBackendOr.isError())
    return loweringFunction();

  using MEFCache = RCRef<Cache::BlobCache<Cache::TransformCacheKey>>;
  auto mefCache = MEFCache::create(mefBackendOr.takeValue());

  // Check if cache contains compiled model.
  auto foundOr = mefCache->findSync(key.copy());
  if (!foundOr.isError() && foundOr->has_value())
    return std::move(**foundOr);

  // Compile model and insert into cache.
  auto mefOr = loweringFunction();
  if (mefOr.isError())
    return mefOr.takeError();

  BufferRef mefBuffer = mefOr.takeValue();

  // Try inserting to cache. If fail to cache, ignore.
  (void)mefCache->insertSync(std::move(key), mefBuffer.copy());

  return mefBuffer;
}
