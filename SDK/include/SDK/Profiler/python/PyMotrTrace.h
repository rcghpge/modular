//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef SDK_PROFILER_PYTHON_PYMOTR_TRACE_H
#define SDK_PROFILER_PYTHON_PYMOTR_TRACE_H

#include "Kernels/MotrTrace.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

namespace M::Profiler::Python {

/// PyObject wrapper around the MotrTrace C++ interface.
class PyMotrTrace {
public:
  PyMotrTrace(const char *message, uint64_t parentId)
      : motrTrace(message, "", parentId) {}

  PyMotrTrace &enter() {
    motrTrace.start();
    return *this;
  }

  void exit(nb::object exc_type, nb::object exc_value, nb::object traceback) {
    motrTrace.end();
  }

  void mark() { motrTrace.mark(); }

  // Add getter for the trace ID
  uint64_t getId() const { return motrTrace.getSpanId(); }

private:
  MotrTrace motrTrace;
};

}; // namespace M::Profiler::Python

#endif // SDK_PROFILER_PYTHON_PYMOTR_TRACE_H
