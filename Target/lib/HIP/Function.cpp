//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Function.h"
#include "Context.h"
#include "Device.h"
#include "Error.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

using M::Target::HIP::hipErrorStatus;

DriverResult<M_driver_runtime_bundle_t>
M_driver_runtime_bundle::load(M_driver_context &ctx,
                              M_driver_static_bundle_t bundle,
                              M_driver_bundle_compilation_options_t opts) {
  if (bundle == nullptr)
    return tl::unexpected(M_DRIVER_STATUS_BAD_BUNDLE);

  M_driver_device &device = ctx.getDevice();
  HIPApi &hip = device.getDriver().hip();

  if (auto result = ctx.getDevice().deviceMethodPreamble(); !result)
    return tl::unexpected(result.error());

  llvm::StringRef fileType(bundle->file_type, bundle->file_type_len);
  if (fileType != "hsaco")
    return tl::unexpected(M_DRIVER_STATUS_INCOMPATIBLE);

  hipModule_t module;
  if (hipError_t r = hip.hipModuleLoadData(&module, bundle->mapped_data.data);
      r != hipSuccess)
    return tl::unexpected(
        hipErrorStatus(&device.getDriver(), r, "hipModuleLoadData"));

  return new M_driver_runtime_bundle{.module = std::move(module)};
}

M_driver_status_t
M_driver_runtime_bundle::unload(M_driver_context &ctx,
                                M_driver_runtime_bundle_t bundle) {
  if (bundle == nullptr)
    return M_DRIVER_STATUS_BAD_BUNDLE;

  M_driver_device &device = ctx.getDevice();
  HIPApi &hip = device.getDriver().hip();

  if (auto result = device.deviceMethodPreamble(); !result)
    return result.error();

  if (hipError_t r = hip.hipModuleUnload(bundle->module); r != hipSuccess)
    return hipErrorStatus(&device.getDriver(), r, "hipModuleUnload");

  delete bundle;
  return M_DRIVER_STATUS_SUCCESS;
}

DriverResult<M_driver_function_t>
M_driver_function::load(M_driver_context &ctx, M_driver_runtime_bundle_t bundle,
                        const char *function_name, uint64_t function_name_len) {
  if (bundle == nullptr)
    return tl::unexpected(M_DRIVER_STATUS_BAD_BUNDLE);

  M_driver_device &device = ctx.getDevice();
  HIPApi &hip = device.getDriver().hip();

  if (auto result = device.deviceMethodPreamble(); !result)
    return tl::unexpected(result.error());

  hipFunction_t func;
  if (hipError_t r =
          hip.hipModuleGetFunction(&func, bundle->module, function_name);
      r != hipSuccess)
    return tl::unexpected(
        hipErrorStatus(&device.getDriver(), r,
                       "hipModuleGetFunction " + llvm::Twine(function_name)));

  return new M_driver_function{.function = std::move(func)};
}

M_driver_status_t M_driver_function::unload(M_driver_function_t func) {
  if (func == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;

  delete func;
  return M_DRIVER_STATUS_SUCCESS;
}
