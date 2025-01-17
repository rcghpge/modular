//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/API/max/c/types.h"
#include "SDK/EngineAPI/python/Types.h"
#include "nanobind/nanobind.h"

using namespace M::core;

M_Dtype M::core::mapNanobindType(nb::dlpack::dtype dtype) {
  switch (dtype.code) {
  case /* nb::dlpack::dtype_code::Int*/ 0:
    switch (dtype.bits) {
    case 8:
      return M_INT8;
    case 16:
      return M_INT16;
    case 32:
      return M_INT32;
    case 64:
      return M_INT64;
    }
    break;
  case /* nb::dlpack::dtype_code::UInt */ 1:
    switch (dtype.bits) {
    case 8:
      return M_UINT8;
    case 16:
      return M_UINT16;
    case 32:
      return M_UINT32;
    case 64:
      return M_UINT64;
    }
    break;
  case /* nb::dlpack::dtype_code::Float */ 2:
    switch (dtype.bits) {
    case 16:
      return M_FLOAT16;
    case 32:
      return M_FLOAT32;
    case 64:
      return M_FLOAT64;
    }
    break;
  case /* nb::dlpack::dtype_code::Bfloat */ 4:
    switch (dtype.bits) {
    case 16:
      return M_BFLOAT16;
    }
    break;
  case /* nb::dlpack::dtype_code::Bool */ 6:
    return M_BOOL;
  }

  throw nb::type_error("Unrecognized DType");
}

nb::dlpack::dtype M::core::mapDType(M_Dtype dtype) {
  switch (dtype) {
  case M_INT8:
    return nb::dtype<int8_t>();
  case M_INT16:
    return nb::dtype<int16_t>();
  case M_INT32:
    return nb::dtype<int32_t>();
  case M_INT64:
    return nb::dtype<int64_t>();
  case M_UINT8:
    return nb::dtype<uint8_t>();
  case M_UINT16:
    return nb::dtype<uint16_t>();
  case M_UINT32:
    return nb::dtype<uint32_t>();
  case M_UINT64:
    return nb::dtype<uint64_t>();
  case M_FLOAT16:
    return nb::dlpack::dtype{
        static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};
  case M_BFLOAT16:
    return nb::dlpack::dtype{
        static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};
  case M_FLOAT32:
    return nb::dtype<float>();
  case M_FLOAT64:
    return nb::dtype<double>();
  case M_BOOL:
    return nb::dtype<bool>();
  default:
    throw nb::type_error("Unrecognized DType");
  }
}

std::string M::core::printDType(M_Dtype dtype) {
  switch (dtype) {
  case M_INT8:
    return "DType.int8";
  case M_INT16:
    return "DType.int16";
  case M_INT32:
    return "DType.int32";
  case M_INT64:
    return "DType.int64";
  case M_UINT8:
    return "DType.uint8";
  case M_UINT16:
    return "DType.uint16";
  case M_UINT32:
    return "DType.uint32";
  case M_UINT64:
    return "DType.uint64";
  case M_FLOAT16:
    return "DType.float16";
  case M_FLOAT32:
    return "DType.float32";
  case M_FLOAT64:
    return "DType.float64";
  case M_BFLOAT16:
    return "DType.bfloat16";
  case M_FLOAT8_E4M3:
    return "DType.f8e4m3";
  case M_FLOAT8_E4M3FNUZ:
    return "DType.f8e4m3fnuz";
  case M_FLOAT8_E5M2:
    return "DType.f8e5m2";
  case M_FLOAT8_E5M2FNUZ:
    return "DType.f8e5m2fnuz";
  case M_BOOL:
    return "DType.bool";
  default:
    throw nb::type_error("Unrecognized DType");
  }
}
