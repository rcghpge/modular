//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Driver.h"
#include "HAL/M_driver_device.h"

#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

#include <dlpack/dlpack.h>

#include <cassert>
#include <cstring>
#include <filesystem>
#include <set>

using namespace llvm;

namespace {
/// Returns true if the installed SDK supports the indicated Major.Minor
/// revision of the Metal Shading Language.
/// Parses the SDK header once on first call and caches all discovered
/// MTLLanguageVersion values for the process lifetime.
static bool sdkSupportsMSLVersion(int maj, int min) {
  static const auto supportedVersions = [] {
    std::set<std::pair<int, int>> versions;

    // Resolve the active Xcode developer directory.
    std::filesystem::path devDir;
    if (const char *env = std::getenv("DEVELOPER_DIR")) {
      devDir = env;
    } else {
      // Same symlink that `xcode-select -p` resolves.
      std::error_code ec;
      devDir = std::filesystem::read_symlink("/var/db/xcode_select_link", ec);
      if (ec)
        return versions;
    }

    // Header that defines the MTLLanguageVersion enum.
    std::filesystem::path headerPath =
        devDir / "Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk" /
        "System/Library/Frameworks/Metal.framework" /
        "Versions/A/Headers/MTLLibrary.h";
    auto bufOrErr = llvm::MemoryBuffer::getFile(headerPath.c_str());
    if (!bufOrErr)
      return versions;

    // Scan for MTLLanguageVersionX_Y enum entries.
    StringRef prefix = "MTLLanguageVersion";
    for (llvm::line_iterator it(**bufOrErr, /*SkipBlanks=*/true, '\0');
         !it.is_at_end(); ++it) {
      StringRef line = *it;
      auto pos = line.find(prefix);
      if (pos == StringRef::npos)
        continue;
      StringRef suffix = line.substr(pos + prefix.size());
      auto [majStr, rest] = suffix.split('_');
      int major = 0, minor = 0;
      // majStr is clean (e.g. "1"); rest may have trailing text
      // (e.g. "1 API_AVAILABLE(...)"), so use consumeInteger.
      if (!majStr.getAsInteger(10, major) && !rest.consumeInteger(10, minor))
        versions.emplace(major, minor);
    }
    return versions;
  }();

  return supportedVersions.count({maj, min}) > 0;
}
} // namespace

DriverResult<int> M_driver_device::getComputeCapability(StringRef name) {
  if (name.empty())
    return 3;

  if (name.contains_insensitive("M5"))
    return 5;

  if (name.contains_insensitive("M4"))
    return 4;

  if (name.contains_insensitive("M3"))
    return 3;

  if (name.contains_insensitive("M2"))
    return 2;

  if (name.contains_insensitive("M1"))
    return 1;

  return tl::unexpected(driver->status.create("Unknown device name"));
}

M_driver_status_t M_driver_device::getProperty(llvm::StringRef name,
                                               void *value) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (name == M_DRIVER_DEVICE_PROPERTY_NAME) {
    std::strncpy(static_cast<char *>(value),
                 mDevice->name()->cString(NS::UTF8StringEncoding),
                 M_DRIVER_DEVICE_PROPERTY_MAX_LEN);
    return M_DRIVER_STATUS_SUCCESS;
  }

  if (name == M_DRIVER_DEVICE_PROPERTY_ARCH) {
    if (mDevice == nullptr)
      return driver->status.create("Failed to create Metal device.");
    std::string name = mDevice->name()->cString(NS::UTF8StringEncoding);

    auto cc = getComputeCapability(name);
    if (!cc)
      return cc.error();
    std::string archName = std::to_string(*cc);
    if (mDevice->supportsFamily(MTL::GPUFamilyMetal4) &&
        sdkSupportsMSLVersion(4, 0)) {
      // Prefer to use latest supported Metal version by default, but only if
      // the installed Xcode's Metal compiler can actually handle Metal 4.0 AIR.
      archName += "-metal4";
    }
    std::strncpy(static_cast<char *>(value), archName.c_str(),
                 M_DRIVER_DEVICE_PROPERTY_MAX_LEN);

    return M_DRIVER_STATUS_SUCCESS;
  }

  if (name == M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE ||
      name == M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE_PINNED) {
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLMetal;
    dl->device_id = 0;
    return M_DRIVER_STATUS_SUCCESS;
  }

  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

extern "C" M_driver_status_t M_driver_device_count(M_driver_driver_t driver,
                                                   int64_t *count) {
  assert(count && "count must not be null");
  if (!driver || driver->devices.size() == 0)
    return M_DRIVER_STATUS_NO_DEVICES;
  *count = driver->devices.size();
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t M_driver_device_get(M_driver_driver_t driver,
                                                 int64_t id,
                                                 M_driver_device_t *device) {
  assert(device && "device must not be null");
  if (!driver || driver->devices.size() == 0)
    return M_DRIVER_STATUS_NO_DEVICES;
  if (id < 0 || size_t(id) >= driver->devices.size())
    return M_DRIVER_STATUS_BAD_DEVICE;
  *device = &driver->devices[id];
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t M_driver_device_property(M_driver_device_t device,
                                                      const char *property_name,
                                                      void *value) {
  return device->getProperty(property_name, value);
}

extern "C" M_driver_status_t
M_driver_device_attribute([[maybe_unused]] M_driver_device_t device,
                          [[maybe_unused]] int32_t attribute,
                          [[maybe_unused]] int32_t *value) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
