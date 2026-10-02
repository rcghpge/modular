//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHCOMPILER_DRIVER_DEVICE_DEVICEMAP_H
#define GRAPHCOMPILER_DRIVER_DEVICE_DEVICEMAP_H

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/DeviceSpecs.h"
#include "Support/ErrorOr.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace M::GraphCompiler::Driver {

/// The device context backing each placement a compiled model refers to.
using DeviceMap = llvm::DenseMap<DeviceRef, M::Driver::DeviceContextRef>;

/// Returns a live, wrapped DeviceContextRef per entry in `deviceRefs`, after
/// synchronizing `deviceMap` so no driver alloc or copy is pending when the
/// first kernel runs.
///
/// `cpu` opens a fresh host context. Other placements use their `deviceMap`
/// entry, else open the accelerator at `DeviceRef::id` (erroring past the
/// physical count), which is only correct when the caller didn't supply the
/// device with the graph.
MODULAR_VISIBILITY_EXPORT ErrorOr<llvm::SmallVector<AsyncRT::AnyAsyncValueRef>>
mapDeviceRefsToDevices(const ArrayRef<DeviceRef> deviceRefs,
                       const DeviceMap &deviceMap, CPUDevice &cpuDevice);

} // namespace M::GraphCompiler::Driver

#endif // GRAPHCOMPILER_DRIVER_DEVICE_DEVICEMAP_H
