//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "MetalCpp.h"
#include "llvm/ADT/StringRef.h"

#include <string>

struct M_driver_function {
  [[nodiscard]] static DriverResult<M_driver_function_t>
  create(M_driver_context_t context, llvm::StringRef code,
         llvm::StringRef name);
  static M_driver_status_t destroy(M_driver_function_t function);

  M_driver_context_t context;
  std::string name;
  NS::SharedPtr<MTL::ComputePipelineState> pipelineState;
  size_t argumentCount;
};
