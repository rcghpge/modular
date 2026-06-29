//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Process-wide virtual-device mode state.
//
// These accessors live in their own shared library (DeviceContextGlobals)
// rather than in DeviceContextImpl so that exactly one copy of the state
// exists per process. DeviceContextImpl is statically linked into several
// shared libraries (libmax, libAsyncRTMojoBindings, device plugins); if the
// state lived there, each image would carry its own copy. ELF flat-namespace
// interposition happens to collapse the duplicates on Linux, but Mach-O
// two-level namespace binds each reference to the library recorded at link
// time, so on macOS the setters and the device-creation path would read
// different copies and virtual-device mode would silently not apply. This is
// the same pattern as AsyncRTRuntimeGlobals and MSupportGlobals.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_VIRTUALDEVICEGLOBALS_H
#define MLRT_DRIVER_DEVICECONTEXT_VIRTUALDEVICEGLOBALS_H

#include "Support/LLVMForwardDecls.h"
#include "Support/SymbolExport.h"

#include "llvm/ADT/StringRef.h"

#include <string>

namespace M::Driver {

// Virtual device mode control functions.
// When virtual device mode is enabled (count > 0), Device::create() will return
// VirtualDevice instances instead of real hardware devices, and
// Device::numberOfDevices() will return the virtual count. This allows creating
// devices for GPU configurations that don't match the current hardware.
// These functions are thread-safe and the count applies globally to all
// threads.
MODULAR_CXX_EXPORT void setVirtualDeviceCount(int count);
MODULAR_CXX_EXPORT int getVirtualDeviceCount();
MODULAR_CXX_EXPORT bool isVirtualDeviceMode();

// Virtual device API control functions.
// These functions set and get the API (e.g., "cuda", "hip", "metal") for
// virtual devices created in compile-only mode. The API must be set before
// creating virtual devices. Thread-safe.
MODULAR_CXX_EXPORT void setVirtualDeviceAPI(StringRef api);
MODULAR_CXX_EXPORT std::string getVirtualDeviceAPI();

// Virtual device target architecture control functions.
// These functions set and get the target GPU architecture (e.g., "sm_80",
// "sm_90") for virtual devices created in compile-only mode. The architecture
// must be set before creating virtual devices. Thread-safe.
MODULAR_CXX_EXPORT void setVirtualDeviceTargetArch(StringRef arch);
MODULAR_CXX_EXPORT std::string getVirtualDeviceTargetArch();

} // namespace M::Driver

#endif // MLRT_DRIVER_DEVICECONTEXT_VIRTUALDEVICEGLOBALS_H
