//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file lists HIP driver functions along with their types. This is used
// by the HIPDriver which defines HIP_FUNC macros and then include this to
// define wrappers, types, and so on, for all the HIP APIs easily.
//
// To add a new HIP APIs to the driver class, simply add an entry to this file
// in the format:
//
//     HIP_FUNC(<API name>, <return type>, <parameter type>...)
//
// Note: For parameters it should be the type only, no parameter names.

// Note: If a new API has more than 12 parameters, the macros in
// Support/Preprocessor.h will need to be updated.
//
//===----------------------------------------------------------------------===//

// No macro guards since this is meant to be included multiple times, but make
// sure HIP_FUNC is defined.
#ifndef HIP_FUNC
#error "HIPDriver.inc included without defining HIP_FUNC"
#endif

HIP_FUNC(hipInit, Result, unsigned int)

HIP_FUNC(hipDrvGetErrorName, Result, Result, const char **)
HIP_FUNC(hipDrvGetErrorString, Result, Result, const char **)

HIP_FUNC(hipDriverGetVersion, Result, int *)

HIP_FUNC(hipDeviceGetAttribute, Result, int *, hipDeviceAttribute_t,
         hipDevice_t)
HIP_FUNC(hipGetDeviceCount, Result, int *)
HIP_FUNC(hipSetDevice, Result, hipDevice_t)
HIP_FUNC(hipGetDevice, Result, int *)
HIP_FUNC(hipDeviceGetName, Result, char *, int, hipDevice_t)
HIP_FUNC(hipDeviceCanAccessPeer, Result, int *, int, int)
HIP_FUNC(hipDeviceEnablePeerAccess, Result, int, unsigned int)

HIP_FUNC(hipStreamCreate, Result, hipStream_t *, unsigned int)
HIP_FUNC(hipStreamCreateWithPriority, Result, hipStream_t *, unsigned int, int)
HIP_FUNC(hipDeviceGetStreamPriorityRange, Result, int *, int *)
HIP_FUNC(hipStreamDestroy, Result, hipStream_t)
HIP_FUNC(hipStreamSynchronize, Result, hipStream_t)
HIP_FUNC(hipStreamWaitEvent, Result, hipStream_t, hipEvent_t, unsigned int)
HIP_FUNC(hipStreamBeginCapture, Result, hipStream_t, hipStreamCaptureMode)
HIP_FUNC(hipStreamEndCapture, Result, hipStream_t, hipGraph_t *)
HIP_FUNC(hipStreamIsCapturing, Result, hipStream_t, hipStreamCaptureStatus *)

HIP_FUNC(hipGraphInstantiate, Result, hipGraphExec_t *, hipGraph_t, hipGraphNode_t *,
          char *, size_t)
HIP_FUNC(hipGraphInstantiateWithFlags, Result, hipGraphExec_t *, hipGraph_t,
          unsigned long long)
HIP_FUNC(hipGraphDestroy, Result, hipGraph_t)
HIP_FUNC(hipGraphExecDestroy, Result, hipGraphExec_t)
HIP_FUNC(hipGraphLaunch, Result, hipGraphExec_t, hipStream_t)
HIP_FUNC(hipGraphDebugDotPrint, Result, hipGraph_t, const char *, unsigned int)

HIP_FUNC(hipHostMalloc, Result, void **, size_t, unsigned int)
HIP_FUNC(hipHostFree, Result, void *)
HIP_FUNC(hipMallocAsync, Result, hipDeviceptr_t *, size_t, hipStream_t)
HIP_FUNC(hipFreeAsync, Result, hipDeviceptr_t, hipStream_t)
HIP_FUNC(hipMalloc, Result, hipDeviceptr_t *, size_t)
HIP_FUNC(hipFree, Result, hipDeviceptr_t)

HIP_FUNC(hipDeviceGetDefaultMemPool, Result, hipMemPool_t *, int)
HIP_FUNC(hipMemPoolSetAccess, Result, hipMemPool_t, const hipMemAccessDesc *,
         size_t)

HIP_FUNC(hipMemcpyHtoDAsync, Result, hipDeviceptr_t, const void *, size_t,
         hipStream_t)
HIP_FUNC(hipMemcpyDtoHAsync, Result, void *, hipDeviceptr_t, size_t,
         hipStream_t)
HIP_FUNC(hipMemcpyDtoDAsync, Result, hipDeviceptr_t, hipDeviceptr_t, size_t,
         hipStream_t)
HIP_FUNC(hipMemcpyHtoD, Result, hipDeviceptr_t, const void *, size_t)
HIP_FUNC(hipMemcpyDtoH, Result, void *, hipDeviceptr_t, size_t)
HIP_FUNC(hipMemcpyDtoD, Result, hipDeviceptr_t, hipDeviceptr_t, size_t)

HIP_FUNC(hipMemsetD8Async, Result, hipDeviceptr_t, unsigned char, size_t,
         hipStream_t)
HIP_FUNC(hipMemsetD16Async, Result, hipDeviceptr_t, unsigned short, size_t,
         hipStream_t)
HIP_FUNC(hipMemsetD32Async, Result, hipDeviceptr_t, unsigned int, size_t,
         hipStream_t)

HIP_FUNC(hipMemset2DAsync, Result, void *, size_t, int, size_t, size_t,
         hipStream_t)

HIP_FUNC(hipMemsetD8, Result, hipDeviceptr_t, unsigned char, size_t)
HIP_FUNC(hipMemsetD16, Result, hipDeviceptr_t, unsigned short, size_t)
HIP_FUNC(hipMemsetD32, Result, hipDeviceptr_t, unsigned int, size_t)

HIP_FUNC(hipMemset2D, Result, void *, size_t, int, size_t, size_t)

HIP_FUNC(hipModuleLoadData, Result, hipModule_t *, const void *)
HIP_FUNC(hipModuleLoadDataEx, Result, hipModule_t *, const void *, unsigned int,
         hipJitOption *, void **)
HIP_FUNC(hipModuleGetGlobal, Result, hipDeviceptr_t *, size_t *, hipModule_t,
         const char *)
HIP_FUNC(hipModuleUnload, Result, hipModule_t)
HIP_FUNC(hipModuleGetFunction, Result, hipFunction_t *, hipModule_t,
         const char *)
HIP_FUNC(hipFuncGetAttribute, Result, int *, hipFunction_attribute,
         hipFunction_t)
HIP_FUNC(hipModuleLaunchKernel, Result, hipFunction_t, unsigned int,
         unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
         unsigned int, hipStream_t, void **, void **)

HIP_FUNC(hipEventCreateWithFlags, Result, hipEvent_t *, unsigned int)
HIP_FUNC(hipEventDestroy, Result, hipEvent_t)
HIP_FUNC(hipEventRecord, Result, hipEvent_t, hipStream_t)
HIP_FUNC(hipEventQuery, Result, hipEvent_t)
HIP_FUNC(hipEventSynchronize, Result, hipEvent_t)
HIP_FUNC(hipEventElapsedTime, Result, float *, hipEvent_t, hipEvent_t)

HIP_FUNC(hipMemGetInfo, Result, size_t *, size_t *)

HIP_FUNC(hipPointerGetAttributes, Result, hipPointerAttribute_t *, const void *)

HIP_FUNC(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, Result, int *,
         hipFunction_t, int, size_t)
