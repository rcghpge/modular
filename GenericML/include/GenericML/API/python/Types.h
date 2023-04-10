//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "modular/c/types.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace core {
M_Dtype mapPybind11Type(pybind11::dtype dtype) {
  if (dtype.is(pybind11::dtype::of<int8_t>()))
    return M_INT8;
  else if (dtype.is(pybind11::dtype::of<int16_t>()))
    return M_INT16;
  else if (dtype.is(pybind11::dtype::of<int32_t>()))
    return M_INT32;
  else if (dtype.is(pybind11::dtype::of<int64_t>()))
    return M_INT64;
  else if (dtype.is(pybind11::dtype::of<uint8_t>()))
    return M_UINT8;
  else if (dtype.is(pybind11::dtype::of<uint16_t>()))
    return M_UINT16;
  else if (dtype.is(pybind11::dtype::of<uint32_t>()))
    return M_UINT32;
  else if (dtype.is(pybind11::dtype::of<uint64_t>()))
    return M_UINT64;
  else if (dtype.is(pybind11::dtype::of<float>()))
    return M_FLOAT32;
  else if (dtype.is(pybind11::dtype::of<double>()))
    return M_DOUBLE;
  else if (dtype.is(pybind11::dtype::of<bool>()))
    return M_BOOL;
  else
    throw std::runtime_error("Unrecognized DType");
}

pybind11::dtype mapDType(M_Dtype dtype) {
  switch (dtype) {
  case M_INT8:
    return pybind11::dtype::of<int8_t>();
  case M_INT16:
    return pybind11::dtype::of<int16_t>();
  case M_INT32:
    return pybind11::dtype::of<int32_t>();
  case M_INT64:
    return pybind11::dtype::of<int64_t>();
  case M_UINT8:
    return pybind11::dtype::of<uint8_t>();
  case M_UINT16:
    return pybind11::dtype::of<uint16_t>();
  case M_UINT32:
    return pybind11::dtype::of<uint32_t>();
  case M_UINT64:
    return pybind11::dtype::of<uint64_t>();
  case M_FLOAT16:
    // TODO(10844): Figure out float16 support using pybind casting utilities
    return pybind11::dtype::of<float>();
  case M_FLOAT32:
    return pybind11::dtype::of<float>();
  case M_DOUBLE:
    return pybind11::dtype::of<double>();
  case M_BOOL:
    return pybind11::dtype::of<bool>();
  default:
    throw std::runtime_error("Unrecognized DType");
  }
}

std::string printDType(M_Dtype dtype) {
  switch (dtype) {
  case M_INT8:
    return "DType.si8";
  case M_INT16:
    return "DType.si16";
  case M_INT32:
    return "DType.si32";
  case M_INT64:
    return "DType.si64";
  case M_UINT8:
    return "DType.ui8";
  case M_UINT16:
    return "DType.ui16";
  case M_UINT32:
    return "DType.ui32";
  case M_UINT64:
    return "DType.ui64";
  case M_FLOAT16:
    return "DType.f16";
  case M_FLOAT32:
    return "DType.f32";
  case M_DOUBLE:
    return "DType.f64";
  case M_BOOL:
    return "DType.bool";
  default:
    throw std::runtime_error("Unrecognized DType");
  }
}
} // namespace core
