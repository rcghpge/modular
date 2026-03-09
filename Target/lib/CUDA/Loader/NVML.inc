//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file lists NVML driver functions by name. The type of each function
// pointer is derived from the official nvml.h header via decltype(&::name).
//
// To add a new NVML API to the driver class, simply add:
//
//     NVML_FN(<API name>)
//
//===----------------------------------------------------------------------===//

// No macro guards since this is meant to be included multiple times, but make
// sure NVML_FN is defined.
#ifndef NVML_FN
#error "NVML.inc included without defining NVML_FN"
#endif

NVML_FN(nvmlInit_v2)
NVML_FN(nvmlShutdown)
NVML_FN(nvmlErrorString)
NVML_FN(nvmlDeviceGetCount_v2)
NVML_FN(nvmlDeviceGetHandleByIndex)
NVML_FN(nvmlDeviceGetCudaComputeCapability)
NVML_FN(nvmlSystemGetDriverVersion)
NVML_FN(nvmlDeviceGetPciInfo_v3)
NVML_FN(nvmlDeviceGetName)
NVML_FN(nvmlDeviceGetMemoryInfo)
