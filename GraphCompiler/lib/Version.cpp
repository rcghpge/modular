//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Version.h"
#include "AsyncRT/Runtime/Globals/Globals.h"
#include "Config/Version.h"
#include "GraphCompiler/MGPDialect/RT/BinaryID.h"
#include "MLRT/Driver/DeviceContext/BinaryID.h"
#include "Mojo/CompilerRT/BinaryID.h"
#include "Support/BinaryID.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/xxhash.h"

#include "xxh3.h"
#include "xxhash.h"

std::string M::MO::getGraphCompilerVersionString() {
  // TODO(GEX-1565): Also include binary IDs of other shared libraries (e.g.
  // MOGGLoader) whose updates can change the compiled MEF.
  return M::getModularVersionString() + M::getBinaryID() + "-" +
         M::MGPRT::getMGPRTBinaryID() + "-" +
         M::Driver::getMojoBindingsBinaryID() + "-" +
         M::AsyncRT::getRuntimeGlobalsBinaryID() + "-" +
         M::KGEN::getCompilerRTBinaryID() + "-" +
         M::getModularVersion().buildType;
}

std::string M::MO::getMOGGCompilerVersionString() {
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
