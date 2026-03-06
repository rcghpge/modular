//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides C++ bindings over the Hardware Abstraction Layer (HAL) C
// interface in SDK/Driver.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_HAL_BUFFER_H
#define MLRT_DRIVER_HAL_BUFFER_H

#include "Support/ReferenceCounted.h"

namespace M::Driver::HAL {

class Context;

struct BufferImpl;

class Buffer : public ReferenceCounted<Buffer> {
public:
  enum class AllocMode {
    HOST = 1,
    SYNC = 2,
    ASYNC = 3,
  };

  ~Buffer();

private:
  explicit Buffer(Context *context, void *memory)
      : context(context), memory(memory) {}

  Context *context;
  void *memory;

  friend Context;
};

using BufferRef = RCRef<Buffer>;

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_BUFFER_H
