//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/CompileDriver.h"
#include "Config/Version.h"
#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "GraphRT/MLIRToMEF.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Support/UnknownLocationDecoder.h"
#include "Support/ADT/ConcatenationTree.h"
#include "Support/Configuration.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace M;

std::optional<std::filesystem::path> M::getMaxCacheDir() {
  auto configOr = Config::open();

  // If config is accessible look if it's set or not. If it's set we are done.
  if (!configOr.isError()) {
    M::Config config = configOr.takeValue();
    auto maxCacheDir = config.getValue("max.cache_dir");
    if (!maxCacheDir.empty())
      return std::filesystem::path(maxCacheDir.str());
  }

  // Try using a temporary directory.
  std::error_code ec;
  std::filesystem::path basePath = std::filesystem::temp_directory_path(ec);
  if (ec)
    return std::nullopt;
  return basePath / ".max_cache";
}

bool M::isModelIRCacheEnabled() {
  auto configOr = Config::open();

  // We will ignore the errors and disable caching if config is not accessible.
  if (configOr.isError())
    return false;

  auto cacheEnabled =
      configOr->getValueAsBool("max.enable_model_ir_cache", true);
  return !cacheEnabled.isError() && *cacheEnabled;
}

std::optional<std::filesystem::path> M::getWeightsCacheDir() {
  auto configOr = Config::open();
  if (configOr.isError())
    return std::nullopt;

  auto weightsCacheEnabledOr = configOr->getValueAsBool(
      /*key=*/"max.enable_weight_cache", /*defaultValue=*/false);

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
                     WriteableBufferRef key, LLCL::Runtime &runtime) {

  // We will ignore the errors and disable caching if config is not accessible.
  if (!isModelIRCacheEnabled())
    return loweringFunction();

  auto mefBackendOr = Cache::getLocalDefaultBackendChain(
      runtime.getCurrentRuntime(),
      (std::filesystem::path(".max_cache") / frameworkName.str() / "mef")
          .string(),
      std::string(M::getModularVersionString()));

  if (mefBackendOr.isError())
    return mefBackendOr.takeError();

  auto mefCache = RCRef<Cache::BlobCache<Cache::TransformCacheKey>>::create(
      mefBackendOr.takeValue());

  // Function to run when there is a cache miss. This should be same as our
  // function where we skip caching with async wrappers.
  auto loweringTransForm = [&](LLCL::AnyAsyncValueRef inputChain) {
    auto outputBuffer =
        LLCL::AsyncValueRef<BufferRef>::allocate(inputChain.getRuntime());

    auto waiter = [&, output = outputBuffer.copy()]() mutable {
      auto setError = [errorChain = output.copy()](Error err) mutable {
        return std::move(errorChain)
            .setToError({std::move(err),
                         LLCL::UnknownLocationDecoder::getEncodedLocation()});
      };

      auto bufferOr = loweringFunction();

      if (bufferOr.isError())
        return setError(bufferOr.takeError());

      // This will be returned to us.
      return std::move(output).emplace(bufferOr.takeValue());
    };
    inputChain.andThenSync(std::move(waiter));
    return outputBuffer;
  };

  // Function to run when the model is in cache. We only need to take it out of
  // cache.
  auto cacheHitFn = [](BufferRef out) { return out.copy(); };

  auto inputChain =
      LLCL::AnyAsyncValueRef::createReady<Chain>(runtime.getCompactPtr());
  auto outputChain = cachedTransform(
      LLCL::UnknownLocationDecoder::getEncodedLocation(), mefCache.copy(),
      inputChain.copy(), key.copy(), std::move(loweringTransForm), cacheHitFn);
  LLCL::await(outputChain);
  if (outputChain.isError())
    return outputChain.getDiagnostic().getMessage().copy();

  // Now that we know it ran successfully we can safely assume the output is a
  // BufferRef.
  BufferRef &outputBuffer = outputChain.get<BufferRef>();

  return std::move(outputBuffer);
}
