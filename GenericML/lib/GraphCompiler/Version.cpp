//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Version.h"
#include "AsyncRT/DeviceContext/BinaryID.h"
#include "AsyncRT/Runtime/Globals/Globals.h"
#include "Config/Version.h"
#include "GenericML/MGPRT/BinaryID.h"
#include "KGEN/CompilerRT/BinaryID.h"
#include "Kernels/mojo/Mogg/MOGGLoader.h"
#include "PyTorch/ATenRT/BinaryID.h"
#include "Support/BinaryID.h"

std::string M::MO::getGraphCompilerVersionString() {
  // TODO(GEX-1565): The binary ID for graph compiler library is not sufficient
  // to solely determine whether a cache invalidation is needed. Updates on
  // shared libraries, e.g., MOGGLoader, might result in a different compiled
  // MEF file too, include the binary ID for relevant shared libraries too.
  return M::getModularVersionString() + M::getBinaryID() + "-" +
         M::MOGG::Embeddings::getMOGGEmbeddingID() + "-" +
         M::MGPRT::getMGPRTBinaryID() + "-" + M::Torch::getATenRTBinaryID() +
         "-" + M::AsyncRT::getMojoBindingsBinaryID() + "-" +
         M::AsyncRT::getRuntimeGlobalsBinaryID() + "-" +
         M::KGEN::getCompilerRTBinaryID() + "-" +
         M::getModularVersion().buildType;
}
