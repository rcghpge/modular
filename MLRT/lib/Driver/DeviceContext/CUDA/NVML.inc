//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file lists NVML driver functions along with their types. This is used
// by the NVML which defines NVML_FUNC macros and then include this to
// define wrappers, types, and so on, for all the NVML APIs easily.
//
// To add a new NVML APIs to the driver class, simply add an entry to this file
// in the format:
//
//     NVML_FUNC(<API name>, <return type>, <parameter type>...)
//
// Note: For parameters it should be the type only, no parameter names.
//
// Note: If a new API has more than 12 parameters, the macros in
// Support/Preprocessor.h will need to be updated.
//
//===----------------------------------------------------------------------===//

// No macro guards since this is meant to be included multiple times, but make
// sure NVML_FUNC and NVML_FUNC_NO_PARAMS are defined.
#ifndef NVML_FUNC
#error "NVML.inc included without defining NVML_FUNC"
#endif

#ifndef NVML_FUNC_NO_PARAMS
#error "NVML.inc included without defining NVML_FUNC_NO_PARAMS"
#endif

NVML_FUNC_NO_PARAMS(nvmlInit_v2, Result)
NVML_FUNC_NO_PARAMS(nvmlShutdown, Result)
NVML_FUNC(nvmlErrorString, const char*, Result)
NVML_FUNC(nvmlDeviceGetCount_v2, Result, unsigned int*)
NVML_FUNC(nvmlDeviceGetHandleByIndex, Result, int, nvmlDevice_t *)
NVML_FUNC(nvmlDeviceGetCudaComputeCapability, Result, nvmlDevice_t, int *, int *)
NVML_FUNC(nvmlSystemGetDriverVersion, Result, char *, int)
NVML_FUNC(nvmlDeviceGetPciInfo_v3, Result, nvmlDevice_t, nvmlPciInfo_t *)
