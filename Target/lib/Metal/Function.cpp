//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Function.h"
#include "Common.h"
#include "Context.h"
#include "Device.h"
#include "HAL/M_driver_api.h"

#include <algorithm>

using namespace llvm;

#define RETURN_ERROR(...) return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);

[[nodiscard]] DriverResult<M_driver_function_t>
M_driver_function::create(M_driver_context_t context, llvm::StringRef code,
                          llvm::StringRef name) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  // Get the Metal device
  MTL::Device *mDevice = context->device->mDevice;
  if (mDevice == nullptr)
    return tl::unexpected(M_DRIVER_STATUS_NO_DEVICES);

  NS::SharedPtr<MTL::Library> library;

  // Detect if input is binary or MSL source code
  bool isBinary = false;
  if (code.size() >= 4) {
    // Check if it's NOT MSL text (which would start with valid ASCII/UTF-8)
    // Look for non-printable characters in the first few bytes
    isBinary = false;
    for (size_t i = 0; i < std::min(size_t(8), code.size()); ++i) {
      unsigned char c = code[i];
      if (c < 32 && c != '\n' && c != '\r' && c != '\t') {
        isBinary = true;
        break;
      }
    }
  }

  if (!isBinary) {
    // Compile from MSL source code
    NS::String *sourceString =
        NS::String::string(code.data(), NS::UTF8StringEncoding);
    NS::SharedPtr<MTL::CompileOptions> options =
        NS::TransferPtr(MTL::CompileOptions::alloc()->init());

    NS::Error *error = nullptr;
    library = NS::TransferPtr(
        mDevice->newLibrary(sourceString, options.get(), &error));
    if (error)
      RETURN_ERROR(
          Twine("Failed to create Metal library (compiling MSL to AIR): ") +
          error->localizedDescription()->cString(NS::UTF8StringEncoding));
  } else {
    // Load from binary metallib data
    dispatch_data_t binaryData = dispatch_data_create(
        code.data(), code.size(), nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);

    NS::Error *error = nullptr;
    library = NS::TransferPtr(mDevice->newLibrary(binaryData, &error));
    if (error)
      RETURN_ERROR(
          Twine("Failed to create Metal library (loading binary/metallib): ") +
          error->localizedDescription()->cString(NS::UTF8StringEncoding));

    // Release the dispatch data
    dispatch_release(binaryData);
  }

  // Get the function from the library
  NS::String *funcName =
      NS::String::string(name.str().c_str(), NS::UTF8StringEncoding);
  NS::SharedPtr<MTL::Function> mFunction =
      NS::TransferPtr(library->newFunction(funcName));
  if (!mFunction) {
    RETURN_ERROR("Failed to create Metal function: " + Twine(function_name));
  }

  // Create a compute pipeline state with reflection to get argument info
  NS::SharedPtr<MTL::ComputePipelineDescriptor> pipelineDesc =
      NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  pipelineDesc->setComputeFunction(mFunction.get());
  pipelineDesc->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);

  // Create pipeline with reflection to get argument count
  MTL::ComputePipelineReflection *reflection = nullptr;
  NS::Error *error = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> pipelineState =
      NS::TransferPtr(mDevice->newComputePipelineState(
          pipelineDesc.get(), MTL::PipelineOptionArgumentInfo, &reflection,
          &error));

  if (error) {
    RETURN_ERROR(
        Twine("Failed to create compute pipeline state (GPU machine code "
              "generation): ") +
        error->localizedDescription()->cString(NS::UTF8StringEncoding));
  }

  if (!pipelineState)
    RETURN_ERROR("Failed to create compute pipeline state (no error returned)");

  if (!reflection)
    RETURN_ERROR("Failed to extract pipeline argument information");

  // Extract argument count from reflection, defaulting to 0 if unavailable
  NS::Array *arguments = reflection->arguments();
  size_t argCount = arguments ? arguments->count() : 0;

  // Create and return a MetalDeviceFunction
  return new M_driver_function(context, /*moduleName,*/ name.str(),
                               pipelineState, argCount);

  return tl::unexpected(M_DRIVER_STATUS_UNIMPLEMENTED);
}

M_driver_status_t M_driver_function::destroy(M_driver_function_t function) {
  delete function;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_function_load(M_driver_driver_t driver, M_driver_context_t context,
                       const void *code, uint64_t code_size,
                       const char *function_name,
                       [[maybe_unused]] int32_t max_dynamic_shared_bytes,
                       [[maybe_unused]] const char *debug_level,
                       [[maybe_unused]] int32_t optimization_level,
                       [[maybe_unused]] M_driver_function_t *function) {
  auto result = M_driver_function::create(
      context, {static_cast<const char *>(code), code_size}, function_name);
  if (!result)
    return result.error();

  *function = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t
M_driver_function_unload(M_driver_driver_t driver, M_driver_context_t context,
                         M_driver_function_t function) {
  return M_driver_function::destroy(function);
}
