//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/CompileDriver.h"
#include "GraphRT/MLIRToMEF.h"
#include "Support/ADT/ConcatenationTree.h"
#include "Support/Configuration.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace M;

static std::optional<std::filesystem::path> getMaxCacheDir() {
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
