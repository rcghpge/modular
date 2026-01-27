//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the NVML API interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_NVML_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_NVML_H

#include "Support/ErrorOr.h"
#include "Support/Preprocessor.h"
#include "Support/Profiling/Tracy.h"

#include "llvm/Support/raw_ostream.h"
#include <sstream>
#include <thread>

#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE 32
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE 16

namespace M::Driver {

using nvmlDevice_t = struct nvmlDevice_st *;

#define CUDA_NVML_CALL_LOG()                                                   \
  if (doCallLog) {                                                             \
    /* Make sure we emit a single entry per log call. */                       \
    std::ostringstream buffer;                                                 \
    buffer << "THR-" << std::this_thread::get_id()                             \
           << ": NVML::" << __FUNCTION__ << "\n";                              \
    llvm::errs() << buffer.str();                                              \
  }

struct nvmlPciInfo_t {
  char busIdLegacy[NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE];
  unsigned int domain;
  unsigned int bus;
  unsigned int device;
  unsigned int pciDeviceId;
  unsigned int pciSubSystemId;
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
};

struct nvmlMemory_t {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};

class NVML {
public:
  enum Result : unsigned int {
    SUCCESS = 0,
  };

  ~NVML();

  ErrorOr<std::string> GetDriverVersion() const;

// Define wrapper member function.
#define NVML_FUNC(name, ret, ...)                                              \
  ret name(EXPAND_PARAMS(__VA_ARGS__)) const {                                 \
    CUDA_NVML_CALL_LOG();                                                      \
    TRACY_ZONE_SCOPED_NC(#name, TRACY_COLOR_GREEN);                            \
    return name##_(EXPAND_ARGS(__VA_ARGS__));                                  \
  }

#define NVML_FUNC_NO_PARAMS(name, ret)                                         \
  ret name() const {                                                           \
    CUDA_NVML_CALL_LOG();                                                      \
    TRACY_ZONE_SCOPED_NC(#name, TRACY_COLOR_GREEN);                            \
    return name##_();                                                          \
  }
#include "NVML.inc"
#undef NVML_FUNC_NO_PARAMS
#undef NVML_FUNC

  static ErrorOr<const NVML &> get();

private:
  // Enable in-depth logging at build time.
  static constexpr bool doCallLog = false;

  static ErrorOr<const NVML &> allocAndInitNVML();
  static const char lib_name[];

#define NVML_FUNC(name, ret, ...)                                              \
  using t##name = ret(__VA_ARGS__);                                            \
  t##name *name##_;

#define NVML_FUNC_NO_PARAMS(name, ret)                                         \
  using t##name = ret();                                                       \
  t##name *name##_;
#include "NVML.inc"
#undef NVML_FUNC_NO_PARAMS
#undef NVML_FUNC
};

} // namespace M::Driver
#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_NVML_H
