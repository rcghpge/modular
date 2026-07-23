//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Version.h"
#include "AsyncRT/Runtime/Globals/Globals.h"
#include "Config/Version.h"
#include "GraphCompiler/MGPDialect/RT/BinaryID.h"
#include "KGEN/CompilerRT/BinaryID.h"
#include "MLRT/Driver/DeviceContext/BinaryID.h"
#include "Support/BinaryID.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/xxhash.h"

#include "xxh3.h"
#include "xxhash.h"

std::string M::MO::getGraphCompilerVersionString() {
  // TODO(GEX-1565): The binary ID for graph compiler library is not sufficient
  // to solely determine whether a cache invalidation is needed. Updates on
  // shared libraries, e.g., MOGGLoader, might result in a different compiled
  // MEF file too, include the binary ID for relevant shared libraries too.
  return M::getModularVersionString() + M::getBinaryID() + "-" +
         M::MGPRT::getMGPRTBinaryID() + "-" +
         M::Driver::getMojoBindingsBinaryID() + "-" +
         M::AsyncRT::getRuntimeGlobalsBinaryID() + "-" +
         M::KGEN::getCompilerRTBinaryID() + "-" +
         M::getModularVersion().buildType;
}

std::string M::MO::getMOGGCompilerVersionString() {
  // Compute version hash once using static initialization. This combines:
  // - Modular framework version
  // - Binary ID for current build
  // - MOGG embeddings ID
  // - KGEN compiler runtime ID
  // - Build type
  // Uses lambda initialization to ensure thread-safe single computation
  static std::string version = []() {
    std::string versionString = M::getModularVersionString() +
                                M::getBinaryID() + "-" +
                                M::KGEN::getCompilerRTBinaryID() + "-" +
                                M::getModularVersion().buildType;
    return llvm::utohexstr(llvm::xxh3_64bits(versionString),
                           /*LowerCase=*/true,
                           /*Width=*/16);
  }();
  return version;
}
