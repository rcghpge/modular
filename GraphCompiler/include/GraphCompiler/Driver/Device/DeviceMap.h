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

/// Returns one AnyAsyncValueRef-wrapped DeviceContextRef per entry in
/// `deviceRefs`, after synchronizing every context in `deviceMap` so that no
/// driver-side alloc or copy is still pending when the first kernel runs.
///
/// Every element is a live context; none is ever null. A `cpu` placement opens
/// a fresh host context and ignores `deviceMap` entirely. Any other placement
/// takes its `deviceMap` entry if there is one, and otherwise falls back to
/// opening an accelerator context at `DeviceRef::id`, erroring if that exceeds
/// the physical accelerator count. That fallback reads the placement id as a
/// physical device id, which only holds when the caller did not supply the
/// device along with the graph.
MODULAR_VISIBILITY_EXPORT ErrorOr<llvm::SmallVector<AsyncRT::AnyAsyncValueRef>>
mapDeviceRefsToDevices(const ArrayRef<DeviceRef> deviceRefs,
                       const DeviceMap &deviceMap, CPUDevice &cpuDevice);

} // namespace M::GraphCompiler::Driver

#endif // GRAPHCOMPILER_DRIVER_DEVICE_DEVICEMAP_H
